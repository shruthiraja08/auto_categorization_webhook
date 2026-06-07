"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Auto Categorization Webhook test suite.

Architecture
~~~~~~~~~~~~
- ``mock_env`` (autouse) : sets test env vars and clears the ``get_settings``
  ``lru_cache`` before and after every test, ensuring no stale config bleeds
  across tests running in the same process.

- ``async_client`` : spins up a full FastAPI test client **with lifespan**
  via ``httpx.AsyncClient + ASGITransport``. Patches ``app.Classifier`` and
  ``app.PromptBuilder`` at the module namespace level so the lifespan creates
  mock instances rather than real ones — no file I/O, no OpenAI connections.
  The mock Classifier's ``.classify`` is pre-configured to return a standard
  ``ClassificationResponse``; individual tests override it via ``mocker``.

- ``samples_file`` : writes a minimal ``samples.json`` to a ``tmp_path`` for
  fully isolated ``PromptBuilder`` unit tests.
"""

from __future__ import annotations

import json
import pathlib
from typing import Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock


# ======================================================================= #
# Shared constants                                                           #
# ======================================================================= #

TEST_API_KEY: str = "test-secret-api-key-1234"
TEST_OPENAI_KEY: str = "sk-test-key-for-testing-only"
TEST_MODEL: str = "gpt-4o"

CLASSIFY_URL: str = "/classify"
HEALTH_URL: str = "/health"

VALID_TICKET_PAYLOAD: dict = {
    "ticket_id": "T-001",
    "title": "Cannot reset my password",
    "description": (
        "I click Forgot Password but never receive the reset email. "
        "I have checked my spam folder. This has been happening for 2 days."
    ),
}


# ======================================================================= #
# Environment & settings fixtures                                            #
# ======================================================================= #


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Inject test environment variables and reset the ``get_settings`` cache.

    Marked ``autouse=True`` so every test starts with a clean settings state.
    The cache is cleared both before and after each test to prevent cross-test
    contamination when multiple tests run in the same process.
    """
    from config import get_settings

    get_settings.cache_clear()

    monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_KEY)
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    monkeypatch.setenv("OPENAI_MODEL", TEST_MODEL)
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SAMPLES_PATH", "samples.json")
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", "logs/predictions.log")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "3")
    monkeypatch.setenv("DEBUG", "False")

    yield  # type: ignore[misc]

    get_settings.cache_clear()


# ======================================================================= #
# Domain object fixtures                                                     #
# ======================================================================= #


@pytest.fixture
def sample_ticket():
    """Return a valid ``TicketRequest`` instance."""
    from schemas import TicketRequest

    return TicketRequest(**VALID_TICKET_PAYLOAD)


@pytest.fixture
def mock_classification_result():
    """Return a high-confidence ``ClassificationResult`` (above the 0.75 threshold)."""
    from schemas import Category, ClassificationResult, Priority

    return ClassificationResult(
        category=Category.AUTHENTICATION,
        subcategory="Password Reset",
        priority=Priority.MEDIUM,
        confidence=0.94,
        reasoning=(
            "User cannot reset password via forgot-password flow. "
            "Unambiguous Authentication issue."
        ),
    )


@pytest.fixture
def mock_low_confidence_result():
    """Return a ``ClassificationResult`` with confidence below the 0.75 threshold."""
    from schemas import Category, ClassificationResult, Priority

    return ClassificationResult(
        category=Category.OTHER,
        subcategory="General Inquiry",
        priority=Priority.LOW,
        confidence=0.55,
        reasoning="Ticket is ambiguous — could fit multiple categories.",
    )


@pytest.fixture
def mock_classification_response():
    """Return a valid ``ClassificationResponse`` as would be returned by Classifier."""
    from schemas import Category, ClassificationResponse, Priority

    return ClassificationResponse(
        request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        ticket_id="T-001",
        category=Category.AUTHENTICATION,
        subcategory="Password Reset",
        priority=Priority.MEDIUM,
        confidence=0.94,
        low_confidence=False,
        latency_ms=750,
        model="gpt-4o",
    )


@pytest.fixture
def mock_low_confidence_response():
    """Return a ``ClassificationResponse`` with ``low_confidence=True``."""
    from schemas import Category, ClassificationResponse, Priority

    return ClassificationResponse(
        request_id="aaaaaaaa-bbbb-cccc-dddd-ffffffffffff",
        ticket_id="T-002",
        category=Category.OTHER,
        subcategory="General Inquiry",
        priority=Priority.LOW,
        confidence=0.55,
        low_confidence=True,
        latency_ms=600,
        model="gpt-4o",
    )


# ======================================================================= #
# FastAPI integration test client                                            #
# ======================================================================= #


@pytest_asyncio.fixture
async def async_client(mock_env, mocker, mock_classification_response):
    """
    Full-stack async HTTP test client for FastAPI integration tests.

    Strategy
    --------
    Patches ``app.Classifier`` and ``app.PromptBuilder`` in the ``app`` module
    namespace before the lifespan runs. When the lifespan calls
    ``PromptBuilder(...)`` and ``Classifier(...)``, it receives the mock
    instances instead of real ones — avoiding all file I/O and network calls.

    The mock Classifier's ``.classify`` is pre-configured to return
    ``mock_classification_response``. Individual tests override it with::

        mock_classifier.classify = AsyncMock(return_value=other_response)

    Yields
    ------
    tuple[AsyncClient, MagicMock]
        ``(http_client, mock_classifier_instance)``
    """
    mock_classifier_instance = MagicMock()
    mock_classifier_instance.examples_loaded = 20
    mock_classifier_instance.classify = AsyncMock(
        return_value=mock_classification_response
    )

    mock_prompt_builder_instance = MagicMock()
    mock_prompt_builder_instance.examples_loaded = 20

    mocker.patch("app.Classifier", return_value=mock_classifier_instance)
    mocker.patch("app.PromptBuilder", return_value=mock_prompt_builder_instance)

    from app import app, lifespan

    async with lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield client, mock_classifier_instance


# ======================================================================= #
# PromptBuilder isolated fixtures                                            #
# ======================================================================= #


@pytest.fixture
def sample_examples() -> list[dict]:
    """Minimal list of 3 valid few-shot examples for PromptBuilder unit tests."""
    return [
        {
            "id": "ex-001",
            "title": "Cannot log in",
            "description": "Login is failing with wrong credentials error.",
            "category": "Authentication",
            "subcategory": "Login Failure",
            "priority": "High",
            "confidence": 0.97,
            "reasoning": "Clear authentication failure.",
        },
        {
            "id": "ex-002",
            "title": "Charged twice",
            "description": "I was billed twice for the same invoice.",
            "category": "Billing",
            "subcategory": "Duplicate Charge",
            "priority": "High",
            "confidence": 0.98,
            "reasoning": "Duplicate charge is a billing issue.",
        },
        {
            "id": "ex-003",
            "title": "App crashes on iPhone",
            "description": "App crashes immediately on startup after last update.",
            "category": "Bug Report",
            "subcategory": "App Crash on Launch",
            "priority": "Critical",
            "confidence": 0.97,
            "reasoning": "100% crash rate on iOS is a critical bug.",
        },
    ]


@pytest.fixture
def samples_file(tmp_path: pathlib.Path, sample_examples: list[dict]):
    """
    Write ``sample_examples`` to a temporary JSON file and return its ``Path``.

    Provides ``PromptBuilder`` tests with a fully isolated, deterministic
    set of examples without depending on the real ``samples.json`` content.
    """
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(sample_examples), encoding="utf-8")
    return path
