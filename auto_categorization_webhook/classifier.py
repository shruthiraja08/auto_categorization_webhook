"""
classifier.py
-------------
Core LLM orchestration layer for the Auto Categorization Webhook.

Responsibilities:
  - Accept a validated ``TicketRequest`` and return a ``ClassificationResponse``.
  - Build the few-shot prompt via ``PromptBuilder``.
  - Call the OpenAI API asynchronously using the ``parse()`` helper which
    enforces structured output against the ``ClassificationResult`` schema.
  - Apply the confidence threshold and set the ``low_confidence`` flag.
  - Measure and record end-to-end latency.
  - Write a JSONL audit record to ``logs/predictions.log`` after every call.
  - Retry on transient OpenAI errors using exponential backoff.

Design decisions:
  - The ``Classifier`` is instantiated once at app startup (singleton) and
    shared across all requests — the OpenAI ``AsyncClient`` is reused, which
    keeps the underlying HTTP connection pool alive.
  - ``tenacity`` handles retry logic declaratively, keeping the classify method
    clean and free of manual retry loops.
  - ``PredictionLogger`` encapsulates all logging concerns and writes JSONL
    asynchronously (via ``asyncio.to_thread``) so disk I/O never blocks the
    event loop.
  - ``reasoning`` from the LLM is written to the audit log but intentionally
    excluded from the API response to avoid leaking internal model chain-of-
    thought to callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types import CompletionUsage  # Correct import path for SDK v1.x
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from prompt_builder import PromptBuilder
from schemas import (
    ClassificationResponse,
    ClassificationResult,
    TicketRequest,
)

logger = logging.getLogger(__name__)


# ======================================================================= #
# Prediction Logger                                                         #
# ======================================================================= #


class PredictionLogger:
    """
    Writes structured JSONL audit records for every classification attempt.

    Each line in ``predictions.log`` is a self-contained JSON object enabling:
      - Real-time monitoring (``tail -f predictions.log | jq``)
      - Offline retraining dataset construction
      - SLA and latency reporting

    Parameters
    ----------
    log_path : str | Path
        Destination file for JSONL audit records.
    max_bytes : int
        Maximum log file size before rotation.
    backup_count : int
        Number of rotated backup files to retain.
    """

    def __init__(
        self,
        log_path: str | Path,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._handler = RotatingFileHandler(
            filename=str(self._log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))

        self._prediction_logger = logging.getLogger("predictions")
        self._prediction_logger.setLevel(logging.INFO)
        self._prediction_logger.addHandler(self._handler)
        # Prevent prediction records propagating to root logger
        self._prediction_logger.propagate = False

    def _build_record(
        self,
        *,
        request_id: str,
        ticket: TicketRequest,
        result: ClassificationResult,
        low_confidence: bool,
        latency_ms: int,
        model: str,
        usage: CompletionUsage | None,
    ) -> dict[str, Any]:
        """
        Assemble the JSONL audit record dict.

        Parameters
        ----------
        request_id : str
            Per-request UUID for log correlation.
        ticket : TicketRequest
            The original ticket payload.
        result : ClassificationResult
            The LLM classification output.
        low_confidence : bool
            Pre-computed flag (``True`` when confidence < threshold). Passed in
            from the caller so we avoid re-calling ``get_settings()`` inside this
            hot path on every prediction write.
        latency_ms : int
            End-to-end processing time.
        model : str
            The OpenAI model identifier returned in the API response.
        usage : CompletionUsage | None
            Token usage stats from the OpenAI response.

        Returns
        -------
        dict[str, Any]
            A flat dict suitable for JSON serialisation.
        """
        return {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "request_id": request_id,
            "ticket_id": ticket.ticket_id,
            "category": result.category.value,
            "subcategory": result.subcategory,
            "priority": result.priority.value,
            "confidence": result.confidence,
            "low_confidence": low_confidence,
            "reasoning": result.reasoning,
            "latency_ms": latency_ms,
            "model": model,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }

    async def log_prediction(
        self,
        *,
        request_id: str,
        ticket: TicketRequest,
        result: ClassificationResult,
        low_confidence: bool,
        latency_ms: int,
        model: str,
        usage: CompletionUsage | None,
    ) -> None:
        """
        Write an audit record asynchronously without blocking the event loop.

        Disk I/O is offloaded to a thread-pool via ``asyncio.to_thread``.

        Parameters
        ----------
        request_id : str
            Per-request UUID for log correlation.
        ticket : TicketRequest
            The original ticket payload.
        result : ClassificationResult
            The LLM classification output.
        low_confidence : bool
            Pre-computed confidence flag from the caller.
        latency_ms : int
            End-to-end latency in milliseconds.
        model : str
            The OpenAI model identifier returned in the API response.
        usage : CompletionUsage | None
            Token usage from the OpenAI response; may be None on error.
        """
        record = self._build_record(
            request_id=request_id,
            ticket=ticket,
            result=result,
            low_confidence=low_confidence,
            latency_ms=latency_ms,
            model=model,
            usage=usage,
        )
        line = json.dumps(record, ensure_ascii=False)
        await asyncio.to_thread(self._prediction_logger.info, line)


# ======================================================================= #
# Classifier                                                                #
# ======================================================================= #


class Classifier:
    """
    Orchestrates the full support ticket classification pipeline.

    Lifecycle:
      - Instantiated once at application startup via the FastAPI lifespan.
      - Shared as a dependency across all request handlers.
      - The underlying ``AsyncOpenAI`` client maintains a connection pool.

    Parameters
    ----------
    settings : Settings
        Application configuration (injected for testability).
    prompt_builder : PromptBuilder
        Pre-configured prompt builder instance.
    """

    def __init__(
        self,
        settings: Settings,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._settings = settings
        self._prompt_builder = prompt_builder
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,  # We handle retries ourselves via tenacity
        )
        self._prediction_logger = PredictionLogger(
            log_path=settings.predictions_log_path,
            max_bytes=settings.log_max_bytes,
            backup_count=settings.log_backup_count,
        )
        logger.info(
            "Classifier initialised. model=%s, confidence_threshold=%.2f",
            settings.openai_model,
            settings.confidence_threshold,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _call_openai(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[ClassificationResult, str, CompletionUsage | None]:
        """
        Call the OpenAI API with structured output and return the parsed result.

        Uses ``client.beta.chat.completions.parse()`` which:
          1. Sends the Pydantic model's JSON Schema as ``response_format``.
          2. Deserialises and validates the response directly into the model.

        Parameters
        ----------
        messages : list[dict[str, str]]
            The full message chain from ``PromptBuilder.build_messages()``.

        Returns
        -------
        tuple[ClassificationResult, str, CompletionUsage | None]
            A 3-tuple of: (parsed result, model name used, token usage).

        Raises
        ------
        openai.RateLimitError
            When the API rate limit is exceeded (retried by tenacity).
        openai.APITimeoutError
            When the request times out (retried by tenacity).
        openai.APIConnectionError
            When a network-level connection error occurs (retried by tenacity).
        openai.BadRequestError
            When the request is malformed — not retried, fail immediately.
        ValueError
            When the LLM returns a refusal instead of structured output.
        """
        response = await self._client.beta.chat.completions.parse(
            model=self._settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            response_format=ClassificationResult,
            temperature=0.0,  # Deterministic output for consistent classification
        )

        choice = response.choices[0]

        # Handle model refusal (content policy, etc.)
        if choice.message.refusal:
            raise ValueError(
                f"OpenAI refused to classify the ticket: {choice.message.refusal}"
            )

        result = choice.message.parsed
        if result is None:
            raise ValueError(
                "OpenAI returned an empty parsed result despite no refusal. "
                "This may indicate a schema mismatch."
            )

        return result, response.model, response.usage

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[ClassificationResult, str, CompletionUsage | None]:
        """
        Wrap ``_call_openai`` with tenacity exponential-backoff retry logic.

        Only transient, retriable errors trigger a retry:
          - ``RateLimitError``      (HTTP 429 — upstream throttling)
          - ``APITimeoutError``     (request timed out)
          - ``APIConnectionError``  (network-level failure)

        Non-retriable errors (``BadRequestError``, ``ValueError``, etc.)
        propagate immediately so the caller can handle them appropriately.

        Parameters
        ----------
        messages : list[dict[str, str]]
            The OpenAI message chain.

        Returns
        -------
        tuple[ClassificationResult, str, CompletionUsage | None]
            The successful classification result after 0–N retries.

        Raises
        ------
        RetryError
            Wraps the last exception if all retry attempts are exhausted.
        """
        retriable = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(retriable),
                stop=stop_after_attempt(self._settings.openai_max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                # reraise=False (default): on exhaustion tenacity raises RetryError,
                # which we catch below to log and re-raise the original cause.
            ):
                with attempt:
                    attempt_number = attempt.retry_state.attempt_number
                    if attempt_number > 1:
                        logger.warning(
                            "Retrying OpenAI call (attempt %d/%d).",
                            attempt_number,
                            self._settings.openai_max_retries,
                        )
                    return await self._call_openai(messages)
        except RetryError as exc:
            # RetryError wraps the last underlying exception.
            # Log it then re-raise the original so callers see the real error type.
            original = exc.last_attempt.exception()
            logger.error(
                "All %d OpenAI retry attempts exhausted. Last error: %s",
                self._settings.openai_max_retries,
                original,
            )
            raise original from exc  # type: ignore[misc]

        # Unreachable — satisfies the type checker only.
        raise RuntimeError("Unexpected exit from retry loop.")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def classify(
        self,
        ticket: TicketRequest,
        request_id: str | None = None,
    ) -> ClassificationResponse:
        """
        Classify a support ticket and return a fully populated response.

        Pipeline:
          1. Generate a request_id (if not provided).
          2. Build the few-shot message chain via ``PromptBuilder``.
          3. Call OpenAI with retry logic.
          4. Apply confidence threshold → set ``low_confidence`` flag.
          5. Write an async audit record to ``predictions.log``.
          6. Return the ``ClassificationResponse``.

        Parameters
        ----------
        ticket : TicketRequest
            The validated incoming support ticket payload.
        request_id : str | None
            Pre-generated request UUID. If ``None``, a new one is generated.
            Pass from the route handler to ensure the same ID appears in both
            the HTTP response and the audit log.

        Returns
        -------
        ClassificationResponse
            The complete, validated classification response ready for the caller.

        Raises
        ------
        openai.OpenAIError
            Propagated after retry exhaustion for transient errors.
        ValueError
            When the LLM returns a refusal or an unparseble response.
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.monotonic()

        logger.info(
            "classify() called. request_id=%s ticket_id=%s",
            request_id,
            ticket.ticket_id,
        )

        # ==========================================================
        # MOCK MODE
        # ==========================================================
        if self._settings.mock_mode:

            from schemas import Category, Priority

            text = (
                f"{ticket.title} {ticket.description}"
            ).lower()

            if any(word in text for word in ["password", "login", "signin"]):
                category = Category.AUTHENTICATION
                subcategory = "Password Reset"
                priority = Priority.MEDIUM

            elif any(word in text for word in ["payment", "billing", "invoice"]):
                category = Category.BILLING
                subcategory = "Payment Issue"
                priority = Priority.HIGH

            elif any(word in text for word in ["slow", "performance", "lag"]):
                category = Category.PERFORMANCE
                subcategory = "Slow Application"
                priority = Priority.MEDIUM

            elif any(word in text for word in ["bug", "error", "crash"]):
                category = Category.BUG_REPORT
                subcategory = "Application Error"
                priority = Priority.HIGH

            elif any(word in text for word in ["feature", "enhancement"]):
                category = Category.FEATURE_REQUEST
                subcategory = "Feature Request"
                priority = Priority.LOW

            else:
                category = Category.TECHNICAL_SUPPORT
                subcategory = "General Support"
                priority = Priority.MEDIUM

            latency_ms = 150

            logger.info(
                "MOCK MODE classification complete. "
                "ticket_id=%s category=%s",
                ticket.ticket_id,
                category.value,
            )

            return ClassificationResponse(
                request_id=request_id,
                ticket_id=ticket.ticket_id,
                category=category,
                subcategory=subcategory,
                priority=priority,
                confidence=0.95,
                low_confidence=False,
                latency_ms=latency_ms,
                model="mock-classifier-v1",
            )

        # Step 1: Build the prompt message chain
        messages = self._prompt_builder.build_messages(ticket)

        # Step 2: Call OpenAI with retry
        result, model_used, usage = await self._call_with_retry(messages)

        # Step 3: Compute latency
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Step 4: Apply confidence threshold
        low_confidence = result.confidence < self._settings.confidence_threshold
        if low_confidence:
            logger.warning(
                "Low confidence classification. request_id=%s ticket_id=%s "
                "confidence=%.3f threshold=%.2f category=%s priority=%s",
                request_id,
                ticket.ticket_id,
                result.confidence,
                self._settings.confidence_threshold,
                result.category.value,
                result.priority.value,
            )
        else:
            logger.info(
                "Classification complete. request_id=%s ticket_id=%s "
                "confidence=%.3f category=%s subcategory=%s priority=%s latency_ms=%d",
                request_id,
                ticket.ticket_id,
                result.confidence,
                result.category.value,
                result.subcategory,
                result.priority.value,
                latency_ms,
            )

        # Step 5: Await the audit log write.
        # log_prediction uses asyncio.to_thread internally, so disk I/O never
        # blocks the event loop. Awaiting directly (rather than create_task)
        # avoids the RuntimeError risk when no running loop exists (e.g. in tests).
        await self._prediction_logger.log_prediction(
            request_id=request_id,
            ticket=ticket,
            result=result,
            low_confidence=low_confidence,
            latency_ms=latency_ms,
            model=model_used,
            usage=usage,
        )

        # Step 6: Build and return the response
        return ClassificationResponse(
            request_id=request_id,
            ticket_id=ticket.ticket_id,
            category=result.category,
            subcategory=result.subcategory,
            priority=result.priority,
            confidence=result.confidence,
            low_confidence=low_confidence,
            latency_ms=latency_ms,
            model=model_used,
        )

    @property
    def examples_loaded(self) -> int:
        """
        Expose the number of loaded few-shot examples for the health endpoint.

        Returns
        -------
        int
            Count of valid examples loaded from ``samples.json``.
        """
        return self._prompt_builder.examples_loaded
