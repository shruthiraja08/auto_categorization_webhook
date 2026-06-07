"""
tests/test_api.py
-----------------
Integration tests for the FastAPI HTTP layer.

Coverage targets:
  - GET  /health    → 200 with correct schema
  - POST /classify  → happy path, low confidence, auth failures,
                      request validation, upstream error propagation

All OpenAI calls are short-circuited at the ``Classifier.classify`` mock
set up in ``conftest.async_client``. These tests exercise the HTTP contract,
authentication logic, request validation, and error-handler HTTP codes.
"""

from __future__ import annotations

import pytest
import httpx
import openai
from unittest.mock import AsyncMock

from tests.conftest import (
    CLASSIFY_URL,
    HEALTH_URL,
    TEST_API_KEY,
    VALID_TICKET_PAYLOAD,
)


# ======================================================================= #
# GET /health                                                                #
# ======================================================================= #


class TestHealthEndpoint:
    """Tests for the public GET /health endpoint."""

    async def test_returns_200(self, async_client):
        client, _ = async_client
        response = await client.get(HEALTH_URL)
        assert response.status_code == 200

    async def test_response_schema(self, async_client):
        """All required fields must be present in the health response."""
        client, _ = async_client
        data = (await client.get(HEALTH_URL)).json()

        assert data["status"] == "healthy"
        assert data["model"] == "gpt-4o"
        assert isinstance(data["examples_loaded"], int)
        assert data["examples_loaded"] >= 0
        assert "version" in data
        assert "timestamp" in data

    async def test_reports_examples_loaded_from_classifier(self, async_client):
        """examples_loaded must reflect the value on app.state.classifier."""
        client, mock_classifier = async_client
        mock_classifier.examples_loaded = 20
        data = (await client.get(HEALTH_URL)).json()
        assert data["examples_loaded"] == 20

    async def test_does_not_require_api_key(self, async_client):
        """Health endpoint must be publicly accessible without X-API-Key."""
        client, _ = async_client
        response = await client.get(HEALTH_URL)  # No auth header
        assert response.status_code == 200

    async def test_model_name_from_settings(self, async_client):
        """The model name in the response must match the configured model."""
        client, _ = async_client
        data = (await client.get(HEALTH_URL)).json()
        assert data["model"] == "gpt-4o"


# ======================================================================= #
# POST /classify — Happy Path                                                #
# ======================================================================= #


class TestClassifyHappyPath:
    """Tests for successful POST /classify responses."""

    async def test_returns_200(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200

    async def test_response_contains_all_required_fields(self, async_client):
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()

        required = {
            "request_id", "ticket_id", "category", "subcategory",
            "priority", "confidence", "low_confidence",
            "latency_ms", "classified_at", "model",
        }
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )

    async def test_echoes_ticket_id(self, async_client):
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert data["ticket_id"] == VALID_TICKET_PAYLOAD["ticket_id"]

    async def test_response_contains_uuid_request_id(self, async_client):
        """request_id must be a UUID4 string (36 chars with hyphens)."""
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert "request_id" in data
        assert len(data["request_id"]) == 36

    async def test_unique_request_ids_per_call(self, async_client):
        """Two consecutive calls must generate distinct request_ids passed to classifier."""
        client, mock_classifier = async_client
        headers = {"X-API-Key": TEST_API_KEY}
        await client.post(CLASSIFY_URL, json=VALID_TICKET_PAYLOAD, headers=headers)
        await client.post(CLASSIFY_URL, json=VALID_TICKET_PAYLOAD, headers=headers)
        
        calls = mock_classifier.classify.call_args_list
        req_id_1 = calls[0].kwargs["request_id"]
        req_id_2 = calls[1].kwargs["request_id"]
        assert req_id_1 != req_id_2

    async def test_valid_category_enum_value(self, async_client):
        """Category in response must be a valid Category enum string."""
        from schemas import Category
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        valid_categories = {c.value for c in Category}
        assert data["category"] in valid_categories

    async def test_valid_priority_enum_value(self, async_client):
        """Priority in response must be a valid Priority enum string."""
        from schemas import Priority
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        valid_priorities = {p.value for p in Priority}
        assert data["priority"] in valid_priorities

    async def test_confidence_in_range(self, async_client):
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    async def test_latency_ms_non_negative(self, async_client):
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert data["latency_ms"] >= 0

    async def test_classifier_called_exactly_once(self, async_client):
        """Classifier.classify() must be invoked exactly once per request."""
        client, mock_classifier = async_client
        await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        mock_classifier.classify.assert_called_once()

    async def test_passes_correct_ticket_to_classifier(self, async_client):
        """The TicketRequest passed to classify() must match the HTTP payload."""
        client, mock_classifier = async_client
        custom_payload = {**VALID_TICKET_PAYLOAD, "ticket_id": "ZD-99999"}
        await client.post(
            CLASSIFY_URL,
            json=custom_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        call_kwargs = mock_classifier.classify.call_args.kwargs
        assert call_kwargs["ticket"].ticket_id == "ZD-99999"

    async def test_high_confidence_sets_low_confidence_false(self, async_client):
        """When confidence >= threshold, low_confidence must be False."""
        client, _ = async_client
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert data["low_confidence"] is False
        assert data["confidence"] >= 0.75


# ======================================================================= #
# POST /classify — Low Confidence                                            #
# ======================================================================= #


class TestClassifyLowConfidence:
    """Tests for the low-confidence flag in the response."""

    async def test_low_confidence_flag_true(
        self, async_client, mock_low_confidence_response
    ):
        """When Classifier returns confidence < threshold, low_confidence must be True."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(return_value=mock_low_confidence_response)

        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert data["low_confidence"] is True
        assert data["confidence"] < 0.75

    async def test_low_confidence_still_returns_200(
        self, async_client, mock_low_confidence_response
    ):
        """Low confidence is NOT an error — must still return HTTP 200."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(return_value=mock_low_confidence_response)

        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200


# ======================================================================= #
# POST /classify — Authentication                                            #
# ======================================================================= #


class TestClassifyAuthentication:
    """Tests for X-API-Key header enforcement on POST /classify."""

    async def test_missing_api_key_returns_401(self, async_client):
        client, _ = async_client
        response = await client.post(CLASSIFY_URL, json=VALID_TICKET_PAYLOAD)
        assert response.status_code == 401

    async def test_wrong_api_key_returns_401(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": "completely-wrong-key"},
        )
        assert response.status_code == 401

    async def test_empty_api_key_returns_401(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": ""},
        )
        assert response.status_code == 401

    async def test_correct_api_key_is_accepted(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200

    async def test_auth_failure_does_not_call_classifier(self, async_client):
        """Classifier must NOT be invoked for unauthenticated requests."""
        client, mock_classifier = async_client
        await client.post(CLASSIFY_URL, json=VALID_TICKET_PAYLOAD)
        mock_classifier.classify.assert_not_called()


# ======================================================================= #
# POST /classify — Request Validation                                        #
# ======================================================================= #


class TestClassifyRequestValidation:
    """Tests for Pydantic request body validation (422 errors)."""

    async def test_missing_ticket_id_returns_422(self, async_client):
        client, _ = async_client
        payload = {k: v for k, v in VALID_TICKET_PAYLOAD.items() if k != "ticket_id"}
        response = await client.post(
            CLASSIFY_URL, json=payload, headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 422

    async def test_missing_title_returns_422(self, async_client):
        client, _ = async_client
        payload = {k: v for k, v in VALID_TICKET_PAYLOAD.items() if k != "title"}
        response = await client.post(
            CLASSIFY_URL, json=payload, headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 422

    async def test_missing_description_returns_422(self, async_client):
        client, _ = async_client
        payload = {k: v for k, v in VALID_TICKET_PAYLOAD.items() if k != "description"}
        response = await client.post(
            CLASSIFY_URL, json=payload, headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 422

    async def test_empty_title_returns_422(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json={**VALID_TICKET_PAYLOAD, "title": ""},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422

    async def test_whitespace_only_title_returns_422(self, async_client):
        """Title that is pure whitespace must fail after strip_whitespace validator."""
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json={**VALID_TICKET_PAYLOAD, "title": "   "},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422

    async def test_empty_description_returns_422(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL,
            json={**VALID_TICKET_PAYLOAD, "description": ""},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422

    async def test_empty_body_returns_422(self, async_client):
        client, _ = async_client
        response = await client.post(
            CLASSIFY_URL, json={}, headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 422


# ======================================================================= #
# POST /classify — Upstream Error Propagation                                #
# ======================================================================= #


class TestClassifyUpstreamErrors:
    """Tests that upstream OpenAI errors map to correct HTTP status codes."""

    @pytest.fixture
    def dummy_response(self):
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return httpx.Response(status_code=429, request=req)

    async def test_rate_limit_error_returns_503(self, async_client, dummy_response):
        """openai.RateLimitError must produce HTTP 503 Service Unavailable."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=dummy_response,
                body=None,
            )
        )
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 503

    async def test_timeout_error_returns_504(self, async_client):
        """openai.APITimeoutError must produce HTTP 504 Gateway Timeout."""
        client, mock_classifier = async_client
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_classifier.classify = AsyncMock(
            side_effect=openai.APITimeoutError(request=req)
        )
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 504

    async def test_connection_error_returns_503(self, async_client):
        """openai.APIConnectionError must produce HTTP 503 Service Unavailable."""
        client, mock_classifier = async_client
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_classifier.classify = AsyncMock(
            side_effect=openai.APIConnectionError(request=req)
        )
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 503

    async def test_value_error_returns_422(self, async_client):
        """ValueError (LLM refusal) must produce HTTP 422."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(
            side_effect=ValueError("OpenAI refused to classify the ticket.")
        )
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422

    async def test_unexpected_error_returns_500(self, async_client):
        """Unhandled exceptions must produce HTTP 500 Internal Server Error."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(
            side_effect=RuntimeError("Something unexpected exploded.")
        )
        response = await client.post(
            CLASSIFY_URL,
            json=VALID_TICKET_PAYLOAD,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 500

    async def test_error_response_contains_request_id(self, async_client):
        """Error responses must still include a non-empty request_id."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(
            side_effect=ValueError("Test error")
        )
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert "request_id" in data
        assert data["request_id"]

    async def test_error_response_contains_detail(self, async_client):
        """Error responses must include a human-readable detail field."""
        client, mock_classifier = async_client
        mock_classifier.classify = AsyncMock(
            side_effect=ValueError("Test error")
        )
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0

    async def test_error_response_does_not_leak_internal_details(self, async_client):
        """Internal exception messages must NOT appear in error response bodies."""
        client, mock_classifier = async_client
        secret_msg = "INTERNAL_DB_PASSWORD_123"
        mock_classifier.classify = AsyncMock(
            side_effect=RuntimeError(secret_msg)
        )
        data = (
            await client.post(
                CLASSIFY_URL,
                json=VALID_TICKET_PAYLOAD,
                headers={"X-API-Key": TEST_API_KEY},
            )
        ).json()
        assert secret_msg not in str(data)
