"""
app.py
------
FastAPI application entry point for the Auto Categorization Webhook.

Responsibilities:
  - Define the FastAPI app with a lifespan context that initialises shared
    resources (Classifier, PromptBuilder) once at startup and cleans up on
    shutdown.
  - Register all HTTP route handlers (POST /classify, GET /health).
  - Enforce X-API-Key authentication via a reusable FastAPI dependency.
  - Configure structured application logging (stdout + rotating file).
  - Register global exception handlers that return ``ErrorResponse`` JSON
    for all 4xx / 5xx cases, never leaking internal details in production.
  - Add CORS middleware for cross-origin webhook callers.

Design decisions:
  - The ``Classifier`` instance is stored in ``app.state`` during the lifespan
    so it can be injected into route handlers without module-level globals.
  - A per-request ``request_id`` (UUID4) is generated in the route handler and
    threaded through both the ``ClassificationResponse`` and the audit log,
    enabling end-to-end tracing across all log sinks.
  - Pydantic ``ValidationError`` from request bodies is caught by FastAPI's
    built-in handler (422). We add a custom handler only to homogenise the
    error response shape to ``ErrorResponse``.
"""

from __future__ import annotations

import logging
import logging.config
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from classifier import Classifier
from config import get_settings
from prompt_builder import PromptBuilder
from schemas import (
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    TicketRequest,
)

# ======================================================================= #
# Logging Setup                                                             #
# ======================================================================= #

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging() -> None:
    """
    Configure the root logger for the application.

    Sets up a ``StreamHandler`` (stdout) with the shared log format and
    applies the log level from ``settings.log_level``.  The prediction
    audit log is configured separately inside ``PredictionLogger``.

    This function is called once inside the lifespan, before any other
    application code runs.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler()],
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ======================================================================= #
# Application Lifespan                                                      #
# ======================================================================= #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown within a single async context.

    On startup:
      1. Configure logging.
      2. Validate settings (fail fast on bad config).
      3. Instantiate ``PromptBuilder`` (loads ``samples.json``).
      4. Instantiate ``Classifier`` (creates the ``AsyncOpenAI`` client).
      5. Store the classifier on ``app.state`` for dependency injection.

    On shutdown:
      - Logs a clean shutdown message. The ``AsyncOpenAI`` HTTP client
        closes its connection pool automatically on GC.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Yields
    ------
    None
        Control is yielded to FastAPI after startup is complete and reclaimed
        when the server is shutting down.
    """
    # --- Startup ---
    configure_logging()
    settings = get_settings()

    logger.info(
        "Starting Auto Categorization Webhook. model=%s debug=%s",
        settings.openai_model,
        settings.debug,
    )

    prompt_builder = PromptBuilder(
        samples_path=settings.samples_path,
        max_examples=8,
    )
    classifier = Classifier(
        settings=settings,
        prompt_builder=prompt_builder,
    )

    app.state.classifier = classifier
    app.state.settings = settings

    logger.info(
        "Application ready. %d few-shot examples loaded.",
        classifier.examples_loaded,
    )

    yield  # ← server is running here

    # --- Shutdown ---
    logger.info("Auto Categorization Webhook shutting down.")


# ======================================================================= #
# FastAPI App Factory                                                       #
# ======================================================================= #


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.

    Returns
    -------
    FastAPI
        The fully configured application ready for ASGI serving.
    """
    settings = get_settings()

    _app = FastAPI(
        title="Auto Categorization Webhook",
        description=(
            "A production-ready webhook that automatically classifies support "
            "tickets into Category, Subcategory, Priority, and Confidence Score "
            "using OpenAI's GPT-4o with few-shot learning."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # ------------------------------------------------------------------ #
    # Middleware                                                            #
    # ------------------------------------------------------------------ #

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten per-deployment via env var if needed
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # ------------------------------------------------------------------ #
    # Global Exception Handlers                                            #
    # ------------------------------------------------------------------ #

    @_app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """
        Convert Pydantic ``ValidationError`` (raised during response building,
        not request parsing) into a standardised ``ErrorResponse`` JSON body.

        Request-body validation errors are handled by FastAPI automatically
        and produce a 422 response; this handler covers edge cases where a
        Pydantic model fails during response construction.
        """
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "Pydantic ValidationError. request_id=%s path=%s errors=%s",
            request_id,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                request_id=request_id,
                detail="Request validation failed. Check your request body.",
                error_code="VALIDATION_ERROR",
            ).model_dump(mode="json"),
        )

    @_app.exception_handler(openai.RateLimitError)
    async def rate_limit_handler(
        request: Request, exc: openai.RateLimitError
    ) -> JSONResponse:
        """Return 503 when the OpenAI rate limit is exceeded after all retries."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "OpenAI rate limit exceeded. request_id=%s", request_id
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                request_id=request_id,
                detail="Classification service is temporarily unavailable. Please retry later.",
                error_code="RATE_LIMIT_EXCEEDED",
            ).model_dump(mode="json"),
        )

    @_app.exception_handler(openai.APITimeoutError)
    async def timeout_handler(
        request: Request, exc: openai.APITimeoutError
    ) -> JSONResponse:
        """Return 504 when the OpenAI API call times out after all retries."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "OpenAI API timeout. request_id=%s", request_id
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                request_id=request_id,
                detail="Classification timed out. Please retry later.",
                error_code="UPSTREAM_TIMEOUT",
            ).model_dump(mode="json"),
        )

    @_app.exception_handler(openai.APIConnectionError)
    async def connection_error_handler(
        request: Request, exc: openai.APIConnectionError
    ) -> JSONResponse:
        """Return 503 when a network-level connection to OpenAI fails."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "OpenAI connection error. request_id=%s error=%s",
            request_id,
            str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                request_id=request_id,
                detail="Unable to reach the classification service. Please retry later.",
                error_code="UPSTREAM_UNAVAILABLE",
            ).model_dump(mode="json"),
        )

    @_app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """
        Return 422 for ``ValueError`` raised by the classifier (e.g. LLM
        returned a refusal or an unparseable structured response).
        """
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "Classifier ValueError. request_id=%s error=%s",
            request_id,
            str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                request_id=request_id,
                detail="The ticket could not be classified. Please check the input and retry.",
                error_code="CLASSIFICATION_FAILED",
            ).model_dump(mode="json"),
        )

    @_app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for any unhandled exception.

        Logs the full traceback internally but returns only a generic message
        to the caller — no internal details are ever leaked in production.
        """
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception(
            "Unhandled exception. request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=request_id,
                detail="An unexpected error occurred. Please try again later.",
                error_code="INTERNAL_SERVER_ERROR",
            ).model_dump(mode="json"),
        )

    return _app


app = create_app()


# ======================================================================= #
# Dependencies                                                              #
# ======================================================================= #


def get_classifier(request: Request) -> Classifier:
    """
    FastAPI dependency that retrieves the shared ``Classifier`` from app state.

    Parameters
    ----------
    request : Request
        The current HTTP request (injected by FastAPI).

    Returns
    -------
    Classifier
        The singleton ``Classifier`` instance created during startup.
    """
    return request.app.state.classifier


async def verify_api_key(
    x_api_key: str | None = Header(
        None,
        alias="X-API-Key",
        description="Secret API key for authenticating classify requests.",
    ),
) -> str:
    """
    FastAPI dependency that validates the ``X-API-Key`` request header.

    Performs a constant-time string comparison using Python's ``==`` operator
    (acceptable here since we are not dealing with cryptographic secrets that
    require ``hmac.compare_digest`` — the key is a pre-shared static token).

    Parameters
    ----------
    x_api_key : str | None
        The value of the ``X-API-Key`` header, injected by FastAPI.

    Returns
    -------
    str
        The validated API key (passed through for potential downstream use).

    Raises
    ------
    HTTPException
        HTTP 401 if the key is missing or does not match the configured secret.
    """
    settings = get_settings()
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


# ======================================================================= #
# Routes                                                                    #
# ======================================================================= #


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Returns the service health status, the configured OpenAI model name, "
        "and the number of few-shot examples currently loaded in memory."
    ),
    tags=["Operations"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns
    -------
    HealthResponse
        Service status, model name, loaded example count, version, and timestamp.
    """
    classifier: Classifier = request.app.state.classifier
    settings = request.app.state.settings

    return HealthResponse(
        status="healthy",
        model=settings.openai_model,
        examples_loaded=classifier.examples_loaded,
        version="1.0.0",
    )


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Support Ticket",
    description=(
        "Accepts a support ticket (title + description) and returns a structured "
        "classification: Category, Subcategory, Priority, and Confidence Score. "
        "Requires a valid ``X-API-Key`` header."
    ),
    responses={
        200: {"description": "Ticket classified successfully.", "model": ClassificationResponse},
        401: {"description": "Missing or invalid API key.", "model": ErrorResponse},
        422: {"description": "Request validation failed or LLM could not classify.", "model": ErrorResponse},
        503: {"description": "Classification service temporarily unavailable.", "model": ErrorResponse},
        504: {"description": "Classification request timed out.", "model": ErrorResponse},
        500: {"description": "Unexpected server error.", "model": ErrorResponse},
    },
    tags=["Classification"],
    dependencies=[Depends(verify_api_key)],
)
async def classify_ticket(
    ticket: TicketRequest,
    request: Request,
    classifier: Classifier = Depends(get_classifier),
) -> ClassificationResponse:
    """
    Classify a support ticket using OpenAI GPT-4o with few-shot learning.

    Generates a unique ``request_id`` per call which appears in both the
    HTTP response body and the JSONL audit log for end-to-end traceability.

    Parameters
    ----------
    ticket : TicketRequest
        The validated incoming support ticket payload (parsed from request body).
    request : Request
        The current HTTP request (used to store ``request_id`` on state for
        exception handlers).
    classifier : Classifier
        The singleton classifier instance (injected via ``get_classifier``).

    Returns
    -------
    ClassificationResponse
        Structured classification result with category, subcategory, priority,
        confidence score, low-confidence flag, latency, and request metadata.
    """
    # Generate request_id early so exception handlers can include it too
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(
        "POST /classify request_id=%s ticket_id=%s",
        request_id,
        ticket.ticket_id,
    )

    response = await classifier.classify(ticket=ticket, request_id=request_id)

    logger.info(
        "POST /classify completed. request_id=%s ticket_id=%s "
        "category=%s priority=%s confidence=%.3f latency_ms=%d",
        request_id,
        ticket.ticket_id,
        response.category.value,
        response.priority.value,
        response.confidence,
        response.latency_ms,
    )

    return response


# ======================================================================= #
# Entry Point                                                               #
# ======================================================================= #

if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app:app",
        host=_settings.app_host,
        port=_settings.app_port,
        workers=_settings.app_workers,
        log_level=_settings.log_level.lower(),
        reload=_settings.debug,
    )
