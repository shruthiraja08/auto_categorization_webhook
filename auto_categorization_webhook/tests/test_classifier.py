"""
tests/test_classifier.py
------------------------
Unit tests for the Classifier class.

Strategy
~~~~~~~~
- ``AsyncOpenAI`` is patched at the ``classifier`` module namespace so the
  ``Classifier.__init__`` receives a mock HTTP client — no real network calls.
- ``PredictionLogger`` is patched to avoid creating log files on disk.
- Each test controls the mock client's ``beta.chat.completions.parse``
  return value or ``side_effect`` to simulate specific OpenAI responses.
"""

from __future__ import annotations

import pytest
import openai
from unittest.mock import AsyncMock, MagicMock
from openai.types import CompletionUsage


# ======================================================================= #
# Helpers                                                                    #
# ======================================================================= #


def _build_openai_response(
    classification_result,
    model: str = "gpt-4o",
    usage: CompletionUsage | None = None,
    refusal: str | None = None,
    parsed_none: bool = False,
) -> MagicMock:
    """
    Construct a mock object matching the shape of an OpenAI
    ``ParsedChatCompletion`` returned by ``beta.chat.completions.parse()``.

    Parameters
    ----------
    classification_result
        The ``ClassificationResult`` to embed in ``choice.message.parsed``.
    model : str
        Model name string to set on the response.
    usage : CompletionUsage | None
        Token usage object; defaults to a realistic fixture.
    refusal : str | None
        If set, ``choice.message.refusal`` is populated and ``parsed`` is None.
    parsed_none : bool
        If True, force ``parsed`` to None with no refusal (simulates a schema
        mismatch or empty response edge case).
    """
    if usage is None:
        usage = CompletionUsage(
            prompt_tokens=412, completion_tokens=38, total_tokens=450
        )

    mock_message = MagicMock()
    mock_message.parsed = None if (refusal or parsed_none) else classification_result
    mock_message.refusal = refusal

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = model
    mock_response.usage = usage

    return mock_response


# ======================================================================= #
# Fixtures                                                                   #
# ======================================================================= #


@pytest.fixture
def prompt_builder_mock():
    """Minimal mock PromptBuilder that returns a two-message chain."""
    mock = MagicMock()
    mock.examples_loaded = 20
    mock.build_messages.return_value = [
        {"role": "system", "content": "You are a classifier."},
        {"role": "user", "content": "Ticket Title: Test\n\nTicket Description:\nTest."},
    ]
    return mock


@pytest.fixture
def mock_ticket():
    """A mock TicketRequest."""
    ticket = MagicMock()
    ticket.ticket_id = "T-001"
    ticket.title = "Cannot reset my password"
    ticket.description = "Clicking Forgot Password does nothing."
    return ticket


@pytest.fixture
def classifier_and_client(prompt_builder_mock, mocker):
    """
    Return ``(Classifier instance, mock AsyncOpenAI client)``.

    ``classifier.AsyncOpenAI`` is patched so the constructor returns a mock
    client. ``PredictionLogger`` is also patched to suppress file I/O.
    """
    from classifier import Classifier
    from config import get_settings

    mock_openai_client = MagicMock()
    mocker.patch("classifier.AsyncOpenAI", return_value=mock_openai_client)
    mock_logger_class = mocker.patch("classifier.PredictionLogger")
    mock_logger_class.return_value.log_prediction = AsyncMock()

    instance = Classifier(
        settings=get_settings(),
        prompt_builder=prompt_builder_mock,
    )
    return instance, mock_openai_client


# ======================================================================= #
# Tests — Happy Path                                                         #
# ======================================================================= #


class TestClassifyHappyPath:
    """Tests for the successful end-to-end classification pipeline."""

    async def test_returns_classification_response_type(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        from schemas import ClassificationResponse

        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert isinstance(result, ClassificationResponse)

    async def test_echoes_ticket_id(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.ticket_id == mock_ticket.ticket_id

    async def test_category_from_llm_result(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.category == mock_classification_result.category

    async def test_subcategory_from_llm_result(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.subcategory == mock_classification_result.subcategory

    async def test_priority_from_llm_result(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.priority == mock_classification_result.priority

    async def test_model_name_from_openai_response(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        """Model name must come from the OpenAI response, not settings."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(
                mock_classification_result, model="gpt-4o-mini"
            )
        )

        result = await classifier.classify(mock_ticket)
        assert result.model == "gpt-4o-mini"

    async def test_latency_ms_is_non_negative(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.latency_ms >= 0

    async def test_prompt_builder_called_with_ticket(
        self, classifier_and_client, prompt_builder_mock, mock_classification_result, mock_ticket
    ):
        """PromptBuilder.build_messages() must receive the ticket object."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        await classifier.classify(mock_ticket)
        prompt_builder_mock.build_messages.assert_called_once_with(mock_ticket)


# ======================================================================= #
# Tests — Request ID                                                         #
# ======================================================================= #


class TestRequestId:
    """Tests for request_id generation and threading."""

    async def test_generates_uuid_when_none_provided(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket, request_id=None)
        assert result.request_id
        assert len(result.request_id) == 36  # UUID4

    async def test_uses_provided_request_id(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )
        fixed_id = "fixed-test-request-id-abcd"

        result = await classifier.classify(mock_ticket, request_id=fixed_id)
        assert result.request_id == fixed_id

    async def test_two_calls_produce_different_ids(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        r1 = await classifier.classify(mock_ticket)
        r2 = await classifier.classify(mock_ticket)
        assert r1.request_id != r2.request_id


# ======================================================================= #
# Tests — Confidence Threshold                                               #
# ======================================================================= #


class TestConfidenceThreshold:
    """Tests for the confidence threshold and low_confidence flag."""

    async def test_high_confidence_not_flagged(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        """confidence=0.94 > 0.75 → low_confidence must be False."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_classification_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.confidence == 0.94
        assert result.low_confidence is False

    async def test_low_confidence_flagged(
        self, classifier_and_client, mock_low_confidence_result, mock_ticket
    ):
        """confidence=0.55 < 0.75 → low_confidence must be True."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(mock_low_confidence_result)
        )

        result = await classifier.classify(mock_ticket)
        assert result.confidence == 0.55
        assert result.low_confidence is True

    async def test_confidence_at_threshold_boundary_not_flagged(
        self, classifier_and_client, mock_ticket
    ):
        """confidence == 0.75 (exactly at threshold) must NOT be flagged."""
        from schemas import Category, ClassificationResult, Priority

        at_threshold = ClassificationResult(
            category=Category.BILLING,
            subcategory="Refund Request",
            priority=Priority.MEDIUM,
            confidence=0.75,
            reasoning="Exactly at threshold.",
        )
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(at_threshold)
        )

        result = await classifier.classify(mock_ticket)
        assert result.low_confidence is False

    async def test_confidence_just_below_threshold_is_flagged(
        self, classifier_and_client, mock_ticket
    ):
        """confidence=0.749 < 0.75 → must be flagged."""
        from schemas import Category, ClassificationResult, Priority

        just_below = ClassificationResult(
            category=Category.OTHER,
            subcategory="General Query",
            priority=Priority.LOW,
            confidence=0.749,
            reasoning="Just below threshold.",
        )
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(just_below)
        )

        result = await classifier.classify(mock_ticket)
        assert result.low_confidence is True


# ======================================================================= #
# Tests — LLM Refusal & Unparseable Responses                               #
# ======================================================================= #


class TestRefusalHandling:
    """Tests for LLM refusal and structurally invalid responses."""

    async def test_refusal_raises_value_error(
        self, classifier_and_client, mock_ticket
    ):
        """When the LLM sets a refusal message, classify() must raise ValueError."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(
                None, refusal="I cannot classify this content."
            )
        )

        with pytest.raises(ValueError, match="refused"):
            await classifier.classify(mock_ticket)

    async def test_none_parsed_raises_value_error(
        self, classifier_and_client, mock_ticket
    ):
        """When parsed is None (no refusal), classify() must raise ValueError."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            return_value=_build_openai_response(None, parsed_none=True)
        )

        with pytest.raises(ValueError, match="empty parsed result"):
            await classifier.classify(mock_ticket)


# ======================================================================= #
# Tests — Retry Logic                                                        #
# ======================================================================= #


class TestRetryLogic:
    """Tests for tenacity retry behaviour on transient OpenAI errors."""

    @pytest.fixture
    def dummy_response(self):
        import httpx
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return httpx.Response(status_code=429, request=req)

    async def test_retries_on_rate_limit_then_succeeds(
        self, classifier_and_client, mock_classification_result, mock_ticket, dummy_response
    ):
        """Fail once with RateLimitError, then succeed — must return valid result."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=[
                openai.RateLimitError(
                    message="Rate limit exceeded",
                    response=dummy_response,
                    body=None,
                ),
                _build_openai_response(mock_classification_result),
            ]
        )

        result = await classifier.classify(mock_ticket)
        assert result.ticket_id == mock_ticket.ticket_id
        assert mock_client.beta.chat.completions.parse.call_count == 2

    async def test_retries_on_timeout_then_succeeds(
        self, classifier_and_client, mock_classification_result, mock_ticket
    ):
        """Fail once with APITimeoutError, then succeed."""
        import httpx
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=req),
                _build_openai_response(mock_classification_result),
            ]
        )

        result = await classifier.classify(mock_ticket)
        assert result is not None
        assert mock_client.beta.chat.completions.parse.call_count == 2

    async def test_raises_original_error_after_retry_exhaustion(
        self, classifier_and_client, mock_ticket, dummy_response
    ):
        """After max_retries failures, the original exception type must propagate."""
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=dummy_response,
                body=None,
            )
        )

        with pytest.raises(openai.RateLimitError):
            await classifier.classify(mock_ticket)

    async def test_does_not_retry_bad_request_error(
        self, classifier_and_client, mock_ticket
    ):
        """BadRequestError is not transient — must fail immediately (1 call only)."""
        import httpx
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(status_code=400, request=req)
        classifier, mock_client = classifier_and_client
        mock_client.beta.chat.completions.parse = AsyncMock(
            side_effect=openai.BadRequestError(
                message="Bad request",
                response=resp,
                body=None,
            )
        )

        with pytest.raises(openai.BadRequestError):
            await classifier.classify(mock_ticket)

        assert mock_client.beta.chat.completions.parse.call_count == 1


# ======================================================================= #
# Tests — Examples Property                                                  #
# ======================================================================= #


class TestExamplesLoadedProperty:
    """Tests for the examples_loaded property delegation."""

    def test_delegates_to_prompt_builder(
        self, classifier_and_client, prompt_builder_mock
    ):
        classifier, _ = classifier_and_client
        prompt_builder_mock.examples_loaded = 20
        assert classifier.examples_loaded == 20
