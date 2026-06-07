"""
prompt_builder.py
-----------------
Constructs the few-shot prompt sent to the OpenAI API for ticket classification.

Responsibilities:
  - Load and validate few-shot examples from ``samples.json`` at startup.
  - Build a system prompt that defines the task, output constraints, and
    classification taxonomy.
  - Format selected few-shot examples into the message chain.
  - Build the final user message containing the ticket to classify.

Design decisions:
  - Examples are loaded once at module level and cached — no per-request I/O.
  - The system prompt encodes all valid Category and Priority enum values
    directly, so the LLM cannot hallucinate values outside the allowed set.
  - Few-shot examples are presented as ``assistant`` turns (not embedded in
    the system prompt) so the model learns the exact output JSON structure.
  - A configurable ``max_examples`` cap prevents the prompt from exceeding
    the model's context window when the samples file grows large.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from schemas import Category, Priority, TicketRequest

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Type alias for a single OpenAI message dict                          #
# ------------------------------------------------------------------ #
Message = dict[str, str]


# ------------------------------------------------------------------ #
# System Prompt Template                                               #
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """You are an expert support ticket classification AI for a SaaS platform.

Your task is to analyse a support ticket (title + description) and classify it into:
  1. category     — One of the allowed top-level categories (listed below).
  2. subcategory  — A concise 2–5 word label describing the specific issue within the category.
  3. priority     — One of the allowed priority levels (listed below).
  4. confidence   — Your certainty in this classification, from 0.0 (completely unsure) to 1.0 (certain).
                    Be honest: express lower confidence when the ticket is vague or ambiguous.
  5. reasoning    — A brief 1–3 sentence internal explanation of why you chose this classification.
                    This is for internal audit only and is not shown to the user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED CATEGORIES (use exact spelling):
{categories}

ALLOWED PRIORITIES (use exact spelling):
{priorities}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITY GUIDELINES:
  - Critical : System down, data breach, security vulnerability, production outage,
               app crash affecting all users. Requires immediate action.
  - High     : Core functionality broken for the user, cannot log in, payment failing,
               complete feature unusable. Requires same-day response.
  - Medium   : Important but has a workaround, billing inquiry, UI bug, slow but working.
               Requires response within 1-2 business days.
  - Low      : Feature request, cosmetic issue, documentation question, general inquiry.
               No operational impact.

RULES:
  - Always return all five fields. Never omit a field.
  - subcategory must be 2–5 words, title-cased (e.g. "Password Reset Failure").
  - If the ticket genuinely fits two categories equally, pick the one most directly
    blocking the user and note the ambiguity in your reasoning.
  - Do NOT invent categories or priorities outside the allowed lists.
  - Respond ONLY with the structured JSON output. No preamble, no explanation outside the JSON.
"""

# Human turn template for each few-shot example
_EXAMPLE_USER_TEMPLATE = "Ticket Title: {title}\n\nTicket Description:\n{description}"


def _build_system_prompt() -> str:
    """
    Render the system prompt with the current Category and Priority enum values.

    Reads values dynamically from the enums so the prompt is always in sync
    with ``schemas.py`` without manual duplication.

    Returns
    -------
    str
        The fully rendered system prompt string.
    """
    categories = "\n".join(f"  • {c.value}" for c in Category)
    priorities = "\n".join(f"  • {p.value}" for p in Priority)
    return _SYSTEM_PROMPT.format(categories=categories, priorities=priorities)


# ======================================================================= #
# PromptBuilder                                                             #
# ======================================================================= #


class PromptBuilder:
    """
    Loads few-shot examples and constructs the OpenAI message chain for
    each classification request.

    Parameters
    ----------
    samples_path : str | Path
        Path to the ``samples.json`` file containing labeled examples.
    max_examples : int
        Maximum number of few-shot examples to include per prompt.
        Defaults to 8 to balance context richness against token usage.
        The examples are taken from the head of the file, so ordering in
        ``samples.json`` matters — put the most representative ones first.

    Attributes
    ----------
    examples_loaded : int
        Number of examples successfully loaded. Exposed for the health endpoint.
    """

    def __init__(
        self,
        samples_path: str | Path = "samples.json",
        max_examples: int = 8,
    ) -> None:
        self._samples_path = Path(samples_path)
        self._max_examples = max_examples
        self._system_prompt: str = _build_system_prompt()
        self._examples: list[dict[str, Any]] = []
        self.examples_loaded: int = 0

        self._load_examples()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_examples(self) -> None:
        """
        Load and validate few-shot examples from ``samples.json``.

        Each example must contain: title, description, category, subcategory,
        priority, confidence, and reasoning.  Malformed entries are skipped
        with a warning rather than crashing the application.

        Raises
        ------
        FileNotFoundError
            If ``samples.json`` does not exist at the configured path.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        if not self._samples_path.exists():
            raise FileNotFoundError(
                f"samples.json not found at '{self._samples_path.resolve()}'. "
                "Ensure the file exists before starting the application."
            )

        raw = json.loads(self._samples_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array in '{self._samples_path.name}'.")

        required_keys = {"title", "description", "category", "subcategory", "priority", "confidence", "reasoning"}
        valid_examples: list[dict[str, Any]] = []

        for idx, example in enumerate(raw):
            missing = required_keys - example.keys()
            if missing:
                logger.warning(
                    "Skipping example index %d — missing required keys: %s",
                    idx,
                    missing,
                )
                continue

            # Validate category and priority values against enums
            try:
                Category(example["category"])
                Priority(example["priority"])
            except ValueError as exc:
                logger.warning(
                    "Skipping example index %d — invalid enum value: %s",
                    idx,
                    exc,
                )
                continue

            valid_examples.append(example)

        self._examples = valid_examples[: self._max_examples]
        self.examples_loaded = len(self._examples)

        logger.info(
            "PromptBuilder: loaded %d valid examples from '%s' (%d selected for prompts, max=%d).",
            self.examples_loaded,
            self._samples_path,
            len(self._examples),
            self._max_examples,
        )

    def _format_example_messages(self) -> list[Message]:
        """
        Convert loaded few-shot examples into alternating user/assistant messages.

        Each pair teaches the model the exact JSON structure it must output.

        .. important::
            We intentionally build the assistant-turn JSON **directly from the raw
            dict** rather than instantiating ``ClassificationResult``. Instantiating
            the Pydantic model here would trigger all field validators — including the
            ``max_length=500`` constraint on ``reasoning`` — which could silently fail
            or crash at startup if any sample has a longer reasoning string.

        Returns
        -------
        list[Message]
            A flat list of ``{"role": "user"|"assistant", "content": "..."}"
            dicts ready to be inserted into the OpenAI messages array.
        """
        messages: list[Message] = []

        for example in self._examples:
            # User turn: the ticket text
            user_content = _EXAMPLE_USER_TEMPLATE.format(
                title=example["title"],
                description=example["description"],
            )
            messages.append({"role": "user", "content": user_content})

            # Assistant turn: build JSON directly from raw dict — avoids Pydantic
            # validator side-effects (e.g. max_length on reasoning) at startup.
            assistant_json = json.dumps(
                {
                    "category": example["category"],
                    "subcategory": example["subcategory"],
                    "priority": example["priority"],
                    "confidence": float(example["confidence"]),
                    "reasoning": example["reasoning"],
                },
                indent=2,
                ensure_ascii=False,
            )
            messages.append({"role": "assistant", "content": assistant_json})

        return messages

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_messages(self, ticket: TicketRequest) -> list[Message]:
        """
        Build the full OpenAI ``messages`` array for a classification request.

        Structure:
          1. System message  — task definition, taxonomy, rules
          2. Few-shot pairs  — N user/assistant example exchanges
          3. User message    — the actual ticket to classify

        Parameters
        ----------
        ticket : TicketRequest
            The validated incoming ticket payload.

        Returns
        -------
        list[Message]
            The complete message chain ready for ``client.chat.completions.parse()``.
        """
        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
        ]

        # Insert few-shot examples (interleaved user/assistant turns)
        messages.extend(self._format_example_messages())

        # Final user turn: the ticket to classify
        user_content = _EXAMPLE_USER_TEMPLATE.format(
            title=ticket.title,
            description=ticket.description,
        )
        messages.append({"role": "user", "content": user_content})

        logger.debug(
            "Built prompt for ticket_id='%s' with %d messages (%d few-shot pairs).",
            ticket.ticket_id,
            len(messages),
            len(self._examples),
        )

        return messages
