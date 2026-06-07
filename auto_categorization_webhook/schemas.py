"""
schemas.py
----------
All Pydantic v2 data models and Enum types for the Auto Categorization Webhook.

This module is the single source of truth for:
  - Domain enumerations (Category, Priority)
  - API request / response shapes
  - The LLM's structured output schema (ClassificationResult)

Design principles:
  - Enums constrain valid values at both API input and LLM output layers.
  - ``ClassificationResult`` is used directly as the OpenAI ``response_format``
    schema, ensuring the LLM always returns parseable, validated JSON.
  - All public models carry JSON Schema examples for auto-generated API docs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ======================================================================= #
# Domain Enumerations                                                       #
# ======================================================================= #


class Category(str, Enum):
    """
    Top-level support ticket categories.

    Inherits from ``str`` so enum members serialise as plain strings in JSON,
    making them transparent to API consumers and the LLM structured schema.
    """

    AUTHENTICATION = "Authentication"
    ACCOUNT_MANAGEMENT = "Account Management"
    BILLING = "Billing"
    PAYMENTS = "Payments"
    TECHNICAL_SUPPORT = "Technical Support"
    PERFORMANCE = "Performance"
    BUG_REPORT = "Bug Report"
    FEATURE_REQUEST = "Feature Request"
    SECURITY = "Security"
    OTHER = "Other"


class Priority(str, Enum):
    """
    Urgency/impact level assigned to a support ticket.

    The LLM is instructed to consider both the reported severity and the
    implied business impact when selecting a priority.
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ======================================================================= #
# API Request Models                                                        #
# ======================================================================= #


class TicketRequest(BaseModel):
    """
    Incoming webhook payload representing a support ticket to be classified.

    Attributes
    ----------
    ticket_id : str
        Unique identifier from the originating system (e.g. Zendesk ticket ID).
        Used for correlation in logs and responses; not validated for format.
    title : str
        Short subject line of the ticket (1–200 characters).
    description : str
        Full body / description of the ticket (1–5000 characters).
    """

    ticket_id: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=100,
            description="Unique identifier of the support ticket.",
            examples=["T-1042", "ZD-98231"],
        ),
    ]

    title: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=200,
            description="Short subject line of the ticket.",
            examples=["Cannot reset my password"],
        ),
    ]

    description: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=5000,
            description="Full description / body of the support ticket.",
            examples=["I click 'Forgot Password' but never receive the reset email."],
        ),
    ]

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Strip leading/trailing whitespace from text fields."""
        return value.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ticket_id": "T-1042",
                "title": "Cannot reset my password",
                "description": (
                    "I click the 'Forgot Password' link on the login page, "
                    "enter my email address, but I never receive a reset email. "
                    "I have checked my spam folder. This has been happening for 2 days."
                ),
            }
        }
    )


# ======================================================================= #
# LLM Structured Output Schema                                              #
# ======================================================================= #


class ClassificationResult(BaseModel):
    """
    Structured output schema returned directly by the OpenAI LLM.

    This model is passed as ``response_format`` to the OpenAI API, which
    guarantees the response JSON always conforms to this exact structure.
    The LLM is instructed to populate every field; no field is optional.

    Attributes
    ----------
    category : Category
        Top-level classification bucket.
    subcategory : str
        Finer-grained label within the category (free text, ≤ 60 chars).
    priority : Priority
        Estimated urgency/impact level.
    confidence : float
        The LLM's self-assessed certainty (0.0 = unsure, 1.0 = certain).
        Calibrated by few-shot examples showing appropriate uncertainty.
    reasoning : str
        A brief internal chain-of-thought explanation (not exposed to callers;
        used for debugging and audit logging).
    """

    category: Category = Field(
        ...,
        description="Top-level category for the support ticket.",
    )

    subcategory: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=60,
            description=(
                "Specific sub-topic within the category. "
                "Should be concise (2–5 words), title-cased."
            ),
        ),
    ]

    priority: Priority = Field(
        ...,
        description="Urgency and business-impact level of the ticket.",
    )

    confidence: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=1.0,
            description=(
                "Self-assessed classification confidence between 0.0 and 1.0. "
                "Express genuine uncertainty when the ticket is ambiguous."
            ),
        ),
    ]

    reasoning: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=500,
            description=(
                "Brief chain-of-thought justification for the classification. "
                "Not returned to API callers; logged internally for audit."
            ),
        ),
    ]


# ======================================================================= #
# API Response Models                                                       #
# ======================================================================= #


class ClassificationResponse(BaseModel):
    """
    Full HTTP response body returned from ``POST /classify``.

    Combines the caller's ``ticket_id`` with the LLM's classification,
    plus operational metadata (request_id, latency, confidence flag).

    Attributes
    ----------
    request_id : str
        UUID4 generated per-request for distributed tracing and log correlation.
    ticket_id : str
        Echo of the caller's ticket identifier.
    category : Category
        Top-level classification result.
    subcategory : str
        Finer-grained sub-topic label.
    priority : Priority
        Estimated urgency level.
    confidence : float
        LLM's self-assessed certainty score.
    low_confidence : bool
        ``True`` when ``confidence`` is below the configured threshold.
        Signals that the ticket may benefit from human review.
    latency_ms : int
        End-to-end processing time for the classify call, in milliseconds.
    classified_at : datetime
        UTC timestamp of when the classification was completed.
    model : str
        OpenAI model identifier used (e.g. ``"gpt-4o"``).
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request (UUID4).",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )

    ticket_id: str = Field(
        ...,
        description="Echo of the caller's ticket identifier.",
    )

    category: Category = Field(
        ...,
        description="Top-level classification result.",
    )

    subcategory: str = Field(
        ...,
        description="Finer-grained label within the category.",
    )

    priority: Priority = Field(
        ...,
        description="Estimated urgency and impact level.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence score for this classification (0.0–1.0).",
    )

    low_confidence: bool = Field(
        ...,
        description=(
            "True when confidence is below the configured threshold. "
            "Indicates the ticket may require manual review."
        ),
    )

    latency_ms: int = Field(
        ...,
        ge=0,
        description="Total processing time in milliseconds.",
    )

    classified_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp when the classification completed.",
    )

    model: str = Field(
        ...,
        description="OpenAI model used for classification.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "ticket_id": "T-1042",
                "category": "Authentication",
                "subcategory": "Password Reset",
                "priority": "Medium",
                "confidence": 0.94,
                "low_confidence": False,
                "latency_ms": 843,
                "classified_at": "2026-06-06T15:42:20Z",
                "model": "gpt-4o",
            }
        }
    )


# ======================================================================= #
# Health Check Response                                                     #
# ======================================================================= #


class HealthResponse(BaseModel):
    """
    Response body for the ``GET /health`` endpoint.

    Attributes
    ----------
    status : str
        Always ``"healthy"`` when the service is operating normally.
    model : str
        Name of the configured OpenAI model.
    examples_loaded : int
        Number of few-shot examples successfully loaded from ``samples.json``.
    version : str
        Application version string.
    timestamp : datetime
        UTC timestamp of when the health check was performed.
    """

    status: str = Field(default="healthy")
    model: str = Field(..., description="Configured OpenAI model name.")
    examples_loaded: int = Field(
        ...,
        ge=0,
        description="Number of few-shot examples loaded from samples.json.",
    )
    version: str = Field(default="1.0.0", description="Application version.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )


# ======================================================================= #
# Error Response                                                            #
# ======================================================================= #


class ErrorResponse(BaseModel):
    """
    Standardised error body returned on 4xx / 5xx responses.

    Attributes
    ----------
    request_id : str
        Per-request UUID for log correlation. Included even on errors so
        callers can trace failures in their own systems.
    detail : str
        Human-readable error description. Internal details (stack traces,
        raw exceptions) are never included in production responses.
    error_code : str | None
        Optional machine-readable error code for programmatic handling
        (e.g. ``"LOW_CONFIDENCE"``, ``"UPSTREAM_UNAVAILABLE"``).
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request-scoped UUID for log correlation.",
    )

    detail: str = Field(
        ...,
        description="Human-readable description of the error.",
    )

    error_code: str | None = Field(
        default=None,
        description="Optional machine-readable error code.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "detail": "Internal server error. Please try again later.",
                "error_code": "UPSTREAM_UNAVAILABLE",
            }
        }
    )
