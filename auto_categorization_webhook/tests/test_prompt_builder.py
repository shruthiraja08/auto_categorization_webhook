"""
tests/test_prompt_builder.py
----------------------------
Unit tests for the PromptBuilder class.

These tests use the ``samples_file`` fixture (from ``conftest.py``) to
provide an isolated, deterministic JSON file of few-shot examples, ensuring
tests do not depend on the live ``samples.json`` file which could change.
"""

import json
import pytest

from prompt_builder import PromptBuilder
from schemas import TicketRequest


# ======================================================================= #
# Tests — Initialisation & Validation                                        #
# ======================================================================= #


class TestPromptBuilderInit:
    """Tests for instantiation, file loading, and validation."""

    def test_loads_valid_examples_from_file(self, samples_file, sample_examples):
        builder = PromptBuilder(samples_path=str(samples_file), max_examples=5)
        assert builder.examples_loaded == len(sample_examples)

    def test_respects_max_examples_limit(self, samples_file, sample_examples):
        """If file has 3 examples but max_examples=2, only 2 should be loaded."""
        builder = PromptBuilder(samples_path=str(samples_file), max_examples=2)
        assert builder.examples_loaded == 2

    def test_raises_file_not_found_error_for_missing_file(self, tmp_path):
        bad_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="not found"):
            PromptBuilder(samples_path=str(bad_path))

    def test_raises_json_decode_error_for_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is not json }", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            PromptBuilder(samples_path=str(bad_file))

    def test_raises_value_error_if_file_is_not_a_list(self, tmp_path):
        bad_file = tmp_path / "not_a_list.json"
        bad_file.write_text('{"just": "a dict"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a JSON array"):
            PromptBuilder(samples_path=str(bad_file))

    def test_skips_invalid_examples_and_logs_warning(
        self, tmp_path, sample_examples, caplog
    ):
        """A sample missing a required field (e.g. category) should be skipped."""
        invalid_example = sample_examples[0].copy()
        del invalid_example["category"]  # Break the schema

        mixed_data = [invalid_example, sample_examples[1]]
        file_path = tmp_path / "mixed.json"
        file_path.write_text(json.dumps(mixed_data), encoding="utf-8")

        builder = PromptBuilder(samples_path=str(file_path))
        
        # Only 1 valid example loaded
        assert builder.examples_loaded == 1
        assert "Skipping example" in caplog.text


# ======================================================================= #
# Tests — Message Building                                                   #
# ======================================================================= #


class TestPromptBuilderMessages:
    """Tests for the build_messages() method and prompt construction."""

    @pytest.fixture
    def builder(self, samples_file):
        """Return a PromptBuilder instance with the 3 valid test examples."""
        return PromptBuilder(samples_path=str(samples_file), max_examples=2)

    @pytest.fixture
    def ticket(self):
        return TicketRequest(
            ticket_id="T-999",
            title="App crashes on startup",
            description="Since the update, the app immediately closes.",
        )

    def test_first_message_is_system_prompt(self, builder, ticket):
        messages = builder.build_messages(ticket)
        assert messages[0]["role"] == "system"
        assert "You are an expert support ticket classification AI" in messages[0]["content"]

    def test_last_message_is_user_ticket(self, builder, ticket):
        messages = builder.build_messages(ticket)
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        assert ticket.title in last_msg["content"]
        assert ticket.description in last_msg["content"]

    def test_correct_number_of_messages_generated(self, builder, ticket):
        """
        Structure:
          - 1 System message
          - (N * 2) Few-shot messages (user/assistant alternating pairs)
          - 1 Final user message (the ticket)
        
        With max_examples=2, total messages should be 1 + 4 + 1 = 6.
        """
        messages = builder.build_messages(ticket)
        assert len(messages) == 6

    def test_few_shot_pairs_alternate_roles(self, builder, ticket):
        """The few-shot section must alternate user -> assistant -> user -> assistant."""
        messages = builder.build_messages(ticket)
        # Slices from index 1 to the second-to-last message
        few_shot_msgs = messages[1:-1]
        
        roles = [msg["role"] for msg in few_shot_msgs]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_assistant_messages_contain_valid_json(self, builder, ticket):
        """The assistant turn in the few-shot pair must be valid JSON."""
        messages = builder.build_messages(ticket)
        
        # Check the first assistant response (index 2)
        assistant_content = messages[2]["content"]
        
        try:
            parsed = json.loads(assistant_content)
        except json.JSONDecodeError:
            pytest.fail("Assistant message content is not valid JSON")
            
        assert "category" in parsed
        assert "priority" in parsed
        assert "reasoning" in parsed
