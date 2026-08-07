# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Qwen3 reasoning parser.

Validates that ``Qwen3Parser`` correctly handles
``<think>``/``</think>`` reasoning with Qwen3 hardened invariants:
- Tool markup inside think stays reasoning text (never implicit end)
- Reasoning ends only on confirmed ``</think>`` (not mid-sentence mentions)
- Stripping ``<think>`` from generated output (old template compat)
- No confirmed ``</think>`` terminal text leaks into output
"""

import dataclasses

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    CHUNK_SIZES,
    DUMMY_TOOLS,
    collect_output,
    replay_streaming,
)
from tests.parser.engine.replay_harness import (
    MockTokenizer as ReplayMockTokenizer,
)
from tests.parser.engine.streaming_helpers import simulate_reasoning_streaming
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.engine.registered_adapters import (
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
)
from vllm.parser.qwen3 import Qwen3Parser, qwen3_config

_THINK_START_ID = 50
_THINK_END_ID = 51
_TOOL_CALL_ID = 60
_TOOL_CALL_END_ID = 61
_TEXT_ID = 100

_QWEN3_VOCAB = {
    "<think>": _THINK_START_ID,
    "</think>": _THINK_END_ID,
    "<tool_call>": _TOOL_CALL_ID,
    "</tool_call>": _TOOL_CALL_END_ID,
}


def _esc_tag(tag: str) -> str:
    """HTML-escaped form emitted for structural tags in reasoning prose."""
    return tag.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_ZWSP = "\u200b"


def _zwsp_tag(tag: str) -> str:
    """ZWSP-neutralized form emitted inside markdown code spans."""
    return tag.replace("<", "<" + _ZWSP, 1)


def _assert_tag_visible(text: str | None, tag: str) -> None:
    """Tag must appear as raw, escaped or ZWSP prose (never silently dropped)."""
    assert text is not None
    assert tag in text or _esc_tag(tag) in text or _zwsp_tag(tag) in text, (
        f"expected {tag!r} (raw/escaped/zwsp) in {text!r}"
    )


class _Qwen3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Qwen3ParserToolAdapter


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_QWEN3_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return Qwen3Parser(mock_tokenizer)


class TestNonStreaming:
    def test_reasoning_then_content(self, parser):
        text = "<think>Let me analyze.</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me analyze."
        assert content == "The answer is 42."

    def test_no_start_token_in_output(self, parser):
        """Qwen3.5+ style: <think> in prompt, only </think> in output."""
        text = "Let me think about this.</think>The answer is 42."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Let me think about this."
        assert content == "The answer is 42."

    def test_reasoning_only(self, parser):
        text = "<think>Still thinking...</think>"
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Still thinking..."
        assert content is None

    def test_no_end_tag_all_reasoning(self, parser):
        """No </think> means truncated output — everything is reasoning."""
        text = "Hello, no reasoning here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Hello, no reasoning here."
        assert content is None

    def test_multiline_reasoning(self, parser):
        text = (
            "<think>Step 1: parse.\nStep 2: compute.\nStep 3: output.</think>Result: 7."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert "Step 1" in reasoning
        assert "Step 3" in reasoning
        assert content == "Result: 7."

    def test_tool_call_inside_think_stays_reasoning(self, parser):
        """Tool markup inside <think> stays plain reasoning text."""
        text = (
            "<think>I need to read the file.\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=cmd>ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning is not None
        _assert_tag_visible(reasoning, "<tool_call>")
        _assert_tag_visible(reasoning, "<function=")
        assert "bash" in reasoning
        assert content is None
        assert not parser.is_reasoning_end([_THINK_START_ID, 1, _TOOL_CALL_ID])

    def test_tool_call_without_think_end_stays_reasoning(self, parser):
        """Without </think>, tool markup remains reasoning (truncated)."""
        text = (
            "I need to read the file.\n\n"
            "<tool_call>\n<function=bash>\n"
            "<parameter=cmd>ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning is not None
        _assert_tag_visible(reasoning, "<tool_call>")
        assert content is None

    def test_live_scenario_think_end_before_tool_call(self, parser):
        """Real model output: </think> immediately before <tool_call>.

        Regression test for the bug where </think> and <parameter=...>
        leaked into reasoning content.
        """
        text = (
            "The user wants to see what files are in the current directory"
            " and their contents. Let me start by listing the directory."
            "</think><tool_call><function=read>"
            "<parameter=filePath>/Users/test/demo</parameter>"
            "</function></tool_call>"
        )
        reasoning, content = parser.extract_reasoning(text, None)
        expected_reasoning = (
            "The user wants to see what files are in the current directory"
            " and their contents. Let me start by listing the directory."
        )
        assert reasoning == expected_reasoning
        assert "</think>" not in reasoning
        assert "<tool_call>" not in reasoning
        assert "<parameter=" not in reasoning

    def test_no_terminal_text_in_reasoning(self, parser):
        """Terminal text must never appear in reasoning output."""
        text = "Reasoning here.</think>Content here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert "</think>" not in (reasoning or "")
        assert "<think>" not in (reasoning or "")

    def test_no_terminal_text_in_content(self, parser):
        """Terminal text must never appear in content output."""
        text = "Reasoning here.</think>Content here."
        reasoning, content = parser.extract_reasoning(text, None)
        assert "</think>" not in (content or "")
        assert "<think>" not in (content or "")

    def test_duplicate_think_end_visible_in_content(self, parser):
        """Duplicate </think> in CONTENT stays visible (never silent-drop)."""
        text = "Reasoning here.</think>Content here.</think>More content."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Reasoning here."
        assert content == "Content here.</think>More content."


class TestIsReasoningEnd:
    def test_think_end_token(self, parser):
        assert parser.is_reasoning_end([_THINK_START_ID, 1, _THINK_END_ID])

    def test_no_end_token(self, parser):
        assert not parser.is_reasoning_end([_THINK_START_ID, 1, 2])

    def test_start_after_end_means_not_ended(self, parser):
        assert not parser.is_reasoning_end([_THINK_END_ID, _THINK_START_ID, 1])

    def test_tool_call_never_ends_reasoning(self, parser):
        """Unpaired <tool_call> must NOT end reasoning (hardened invariant)."""
        assert not parser.is_reasoning_end([_THINK_START_ID, 1, _TOOL_CALL_ID])

    def test_prompt_tool_example_before_generation_think_not_end(self, parser):
        """Tool examples before the generation <think> must not end reasoning."""
        assert not parser.is_reasoning_end([_TOOL_CALL_ID, _TEXT_ID, _THINK_START_ID])

    def test_paired_tool_call_not_end(self, parser):
        """Paired <tool_call>...</tool_call> inside think is NOT end."""
        assert not parser.is_reasoning_end(
            [_THINK_START_ID, 1, _TOOL_CALL_ID, 2, _TOOL_CALL_END_ID]
        )

    def test_tool_call_after_think_end(self, parser):
        """<tool_call> after </think> — already ended via think_end."""
        assert parser.is_reasoning_end(
            [_THINK_START_ID, 1, _THINK_END_ID, _TOOL_CALL_ID]
        )

    def test_empty_ids(self, parser):
        assert not parser.is_reasoning_end([])


class TestDelegatingPromptDetection:
    def test_prompt_tool_example_does_not_skip_streaming_reasoning(
        self, mock_tokenizer, mock_request
    ):
        parser = _Qwen3DelegatingParser(mock_tokenizer)
        prompt_ids = [_TOOL_CALL_ID, _TEXT_ID, _THINK_START_ID]

        delta = parser.parse_delta(
            "thinking",
            [_TEXT_ID],
            mock_request,
            prompt_token_ids=prompt_ids,
            finished=False,
        )

        assert delta is not None
        assert delta.reasoning == "thinking"
        assert delta.content is None


class TestStreaming:
    def test_basic_streaming(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>", "thinking", " hard", "</think>", "Done"],
            [
                (_THINK_START_ID,),
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking hard"
        assert content == "Done"

    def test_streaming_no_start_token(self, parser):
        """Qwen3.5 style: no <think> in output, just reasoning then </think>."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning ", "text", "</think>", "Content"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning text"
        assert content == "Content"

    def test_streaming_start_token_stripped(self, parser):
        """<think> in output (old template) should be stripped."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>reasoning", "</think>", "Content"],
            [
                (_THINK_START_ID, 1),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "Content"

    def test_streaming_tool_call_stays_in_reasoning(self, parser):
        """<tool_call> inside think streams as reasoning, not content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["I need to check.", "<tool_call>", "\n<function=test>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert "I need to check." in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")
        assert content == ""

    def test_mentioned_think_end_stays_reasoning(self, parser):
        """Discussing </think> mid-sentence must not open the answer phase.

        Regression: model prose like ``treating </think> or … as special
        tokens`` was split into reason/response at the mentioned tag.
        """
        text = (
            "Another potential issue is the tokenizer treating "
            "</think> or <|im_end|> as special tokens that trigger "
            "stops incorrectly."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "tokenizer treating" in reasoning
        assert "as special tokens that trigger" in reasoning
        _assert_tag_visible(reasoning, "</think>")
        assert "<|im_end|>" in reasoning

    def test_think_end_then_lowercase_closing_stays_reasoning(self, parser):
        """Screenshot: lowercase prose after </think> is still thinking."""
        text = (
            "Otherwise (thinking enabled, default), a missing\n"
            "</think>\n"
            "closing `</think>` is usually treated as truncated reasoning."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "a missing" in reasoning
        assert "closing" in reasoning
        assert "truncated reasoning" in reasoning
        _assert_tag_visible(reasoning, "</think>")

    def test_think_end_then_is_usually_stays_reasoning(self, parser):
        text = "text </think> is usually treated as truncated."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "is usually treated" in reasoning
        _assert_tag_visible(reasoning, "</think>")

    def test_streaming_mentioned_think_end_stays_reasoning(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "tokenizer treating ",
                "</think>",
                " or <|im_end|> as special tokens that trigger stops.",
            ],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert content == ""
        assert "tokenizer treating" in reasoning
        _assert_tag_visible(reasoning, "</think>")
        assert "as special tokens that trigger stops." in reasoning
        assert "<|im_end|>" in reasoning

    def test_streaming_content_after_think_end(self, parser):
        """Content deltas after </think> are routed as content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "Content1", " content2"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "Content1 content2"

    def test_streaming_tool_markup_after_think_end_is_content(self, parser):
        """After </think>, tool markup is handled by the tool FSM (content)."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["thinking", "</think>", "<tool_call>", "<function=f>"],
            [
                (1,),
                (_THINK_END_ID,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking"
        assert "<tool_call>" not in reasoning

    def test_streaming_end_grouped_with_content(self, parser):
        """</think> grouped with following content in one delta."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>The answer"],
            [
                (1,),
                (_THINK_END_ID, 2),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "The answer"

    def test_streaming_think_and_end_in_one_delta(self, parser):
        """<think> and </think> in the same delta."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>reasoning</think>"],
            [
                (_THINK_START_ID, 1, _THINK_END_ID),
            ],
        )
        assert reasoning == "reasoning"
        assert content == ""

    def test_streaming_pure_content_no_think(self, parser):
        """No think tokens at all — everything is reasoning (truncated)."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["hello ", "world"],
            [
                (1,),
                (2,),
            ],
        )
        assert reasoning == "hello world"
        assert content == ""

    def test_streaming_think_end_and_tool_call_same_delta(self, parser):
        """</think> and <tool_call> in the same delta — no leakage.

        Regression test: the old override split at <tool_call> without
        stripping </think>, causing </think> to leak into reasoning.
        """
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "Let me list the directory.",
                "</think><tool_call>",
                "<function=read>",
                "<parameter=filePath>/tmp</parameter>",
            ],
            [
                (1,),
                (_THINK_END_ID, _TOOL_CALL_ID),
                (2,),
                (3,),
            ],
        )
        assert reasoning == "Let me list the directory."
        assert "</think>" not in reasoning
        assert "<tool_call>" not in reasoning
        assert "<parameter=" not in reasoning
        assert content is not None

    def test_streaming_no_terminal_text_leaks(self, parser):
        """Terminal text must never appear in reasoning or content."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "Content"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert "</think>" not in reasoning
        assert "</think>" not in content
        assert "<think>" not in reasoning

    def test_streaming_duplicate_think_end_visible_in_content(self, parser):
        """Duplicate </think> in CONTENT emits as text (never silent-drop)."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "Content", "</think>", "More"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "Content</think>More"

    def test_streaming_think_end_in_content_keeps_stream(self, parser):
        """Mid-answer special </think> is visible text; stream continues."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "Foo ", "</think>", " bar"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "Foo </think> bar"
        assert "</think>" in content


class TestTrailingWhitespaceStripping:
    """When strip_trailing_reasoning_whitespace is True,
    trailing whitespace before </think> must be stripped.

    Models often generate trailing newlines before </think>, and these
    accumulate across multi-turn conversations via a feedback loop.
    """

    @pytest.fixture
    def parser_with_strip(self):
        cfg = dataclasses.replace(
            qwen3_config(),
            strip_trailing_reasoning_whitespace=True,
        )
        return Qwen3Parser(make_mock_tokenizer(_QWEN3_VOCAB), parser_engine_config=cfg)

    def test_non_streaming_trailing_newline(self, parser_with_strip):
        text = "Reasoning here.\n</think>Content."
        reasoning, content = parser_with_strip.extract_reasoning(text, None)
        assert reasoning == "Reasoning here."
        assert content == "Content."

    def test_non_streaming_multiple_trailing_newlines(self, parser_with_strip):
        text = "Reasoning here.\n\n\n</think>Content."
        reasoning, content = parser_with_strip.extract_reasoning(text, None)
        assert reasoning == "Reasoning here."
        assert content == "Content."

    def test_non_streaming_internal_newlines_preserved(self, parser_with_strip):
        text = "Step 1.\n\nStep 2.\n\nStep 3.</think>Answer."
        reasoning, content = parser_with_strip.extract_reasoning(text, None)
        assert reasoning == "Step 1.\n\nStep 2.\n\nStep 3."
        assert content == "Answer."

    def test_non_streaming_only_newlines_becomes_none(self, parser_with_strip):
        text = "\n\n\n</think>Content."
        reasoning, content = parser_with_strip.extract_reasoning(text, None)
        assert reasoning is None
        assert content == "Content."

    def test_streaming_trailing_newline_stripped(self, parser_with_strip):
        reasoning, content = simulate_reasoning_streaming(
            parser_with_strip,
            ["thinking.\n", "</think>", "Done"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking."
        assert content == "Done"

    def test_streaming_multiple_trailing_newlines_stripped(self, parser_with_strip):
        reasoning, content = simulate_reasoning_streaming(
            parser_with_strip,
            ["thinking.\n", "\n", "\n", "</think>", "Done"],
            [
                (1,),
                (2,),
                (3,),
                (_THINK_END_ID,),
                (4,),
            ],
        )
        assert reasoning == "thinking."
        assert content == "Done"

    def test_streaming_internal_newlines_preserved(self, parser_with_strip):
        reasoning, content = simulate_reasoning_streaming(
            parser_with_strip,
            ["Step 1.\n", "\nStep 2.\n", "</think>", "Answer"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "Step 1.\n\nStep 2."
        assert content == "Answer"

    def test_streaming_trailing_newlines_before_tool_call_in_think(
        self, parser_with_strip
    ):
        """Tool markup in think stays reasoning; strip only applies at </think>."""
        reasoning, content = simulate_reasoning_streaming(
            parser_with_strip,
            ["I'll check.\n\n", "<tool_call>", "<function=test>"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert "I'll check." in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")
        assert content == ""


class TestStructuralTagProseInvariants:
    """PR-title / citation tags must stay visible text; tools must not fire."""

    @pytest.fixture
    def parser(self, mock_tokenizer):
        return Qwen3Parser(mock_tokenizer)

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            )
        ]
        return Qwen3Parser(mock_tokenizer, tools=tools)

    def test_pr35687_title_tag_stays_in_reasoning(self, parser):
        text = "PR #35687: Treat <tool_call> as implicit reasoning end in Qwen3 parser."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "PR #35687" in reasoning
        assert "as implicit reasoning end" in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")

    def test_think_end_then_tool_call_prose_stays_reasoning(self, parser):
        text = (
            "Models may omit </think> or emit <tool_call> as a citation "
            "inside the think block."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        _assert_tag_visible(reasoning, "</think>")
        _assert_tag_visible(reasoning, "<tool_call>")
        assert "inside the think block" in reasoning

    def test_real_think_end_then_tool_still_works(
        self, parser_with_tools, mock_request
    ):
        text = (
            "Done thinking.</think>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        reasoning, content, tool_calls = parser_with_tools.parse(text, mock_request)
        assert reasoning == "Done thinking."
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"

    def test_think_end_and_tool_in_same_delta_still_emits(
        self, mock_tokenizer, mock_request
    ):
        """``</think>`` special-id delta that also carries plaintext tool XML.

        Detokenizer often flushes ``…</think>\\n\\n<tool_call>…`` in one
        chunk with only the think-end token id. The scanner must keep the
        suffix after ``</think>`` so the tool still parses.
        """
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "Brave_Search_brave_web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            )
        ]
        mock_request.tools = tools
        parser = Qwen3Parser(mock_tokenizer, tools=tools)
        # One detokenizer flush: special-id ``</think>`` plus plaintext tool XML.
        chunks: list[tuple[str, list[int]]] = [
            ("thinking done", [1]),
            (
                "</think>\n\n"
                "<tool_call>\n"
                "<function=Brave_Search_brave_web_search>\n"
                "<parameter=query>\nq\n</parameter>\n"
                "</function>\n"
                "</tool_call>",
                [_THINK_END_ID],
            ),
        ]
        names: list[str] = []
        for text, ids in chunks:
            delta = parser.parse_delta(text, ids, mock_request, finished=False)
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        flush = parser.parse_delta("", [], mock_request, finished=True)
        if flush and flush.tool_calls:
            for tc in flush.tool_calls:
                if tc.function and tc.function.name:
                    names.append(tc.function.name)
        assert "Brave_Search_brave_web_search" in names

    def test_think_end_special_then_plaintext_tool_call_still_emits(
        self, mock_tokenizer, mock_request
    ):
        """Regression: ``</think>`` as special token + ``<tool_call>`` as
        ordinary text must still emit tools (not dump XML into content).

        Qwen often tokenizes think-end as a dedicated id but emits tool
        tags as multi-token text. Strict token-id demotion used to treat
        that text ``<tool_call>`` as prose and leak the invoke.
        """
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "Brave_Search_brave_web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            )
        ]
        # (text, delta_token_ids) — think-end uses special ids; tool tags do not.
        chunks: list[tuple[str, list[int]]] = [
            ("Handle malformed (e.g., unclosed ", [1]),
            ("</think>", [_THINK_END_ID]),
            (", <|endoftext|> leaking). Let me search.\n", [2]),
            ("</think>\n\n", [_THINK_END_ID]),
            ("<tool_call>\n", []),
            ("<function=Brave_Search_brave_web_search>\n", []),
            ("<parameter=query>\nvLLM endoftext\n</parameter>\n", []),
            ("</function>\n", []),
            ("</tool_call>", []),
        ]
        mock_request.tools = tools
        parser = Qwen3Parser(mock_tokenizer, tools=tools)
        names: list[str] = []
        content_parts: list[str] = []
        for text, ids in chunks:
            delta = parser.parse_delta(text, ids, mock_request, finished=False)
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)
        flush = parser.parse_delta("", [], mock_request, finished=True)
        if flush and flush.tool_calls:
            for tc in flush.tool_calls:
                if tc.function and tc.function.name:
                    names.append(tc.function.name)
        if flush and flush.content:
            content_parts.append(flush.content)

        assert "Brave_Search_brave_web_search" in names
        content = "".join(content_parts)
        assert "<function=" not in content
        assert "</tool_call>" not in content

    def test_think_end_then_orphan_function_still_emits_tool(
        self, mock_tokenizer, mock_request
    ):
        """Qwen3.6 sometimes omits <tool_call> and emits <function=...> only."""
        from tests.parser.engine.streaming_helpers import (
            collect_function_name,
            simulate_tool_streaming,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "Brave_Search_brave_web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            )
        ]
        text = (
            "Keywords: vLLM qwen3_reasoning_parser PR endoftext.\n"
            "</think>\n"
            "\n"
            "<function=Brave_Search_brave_web_search>\n"
            "<parameter=query>\n"
            "vLLM site:github.com/vllm-project/vllm endoftext\n"
            "</parameter>\n"
            "</function>\n"
        )
        parser = Qwen3Parser(mock_tokenizer, tools=tools)
        reasoning, content, tool_calls = parser.parse(text, mock_request)
        assert reasoning is not None
        assert reasoning.startswith(
            "Keywords: vLLM qwen3_reasoning_parser PR endoftext."
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "Brave_Search_brave_web_search"
        assert "Brave_Search" not in reasoning
        assert "<function=" not in reasoning

        chunks = [
            "Keywords: endoftext.\n",
            "</think>\n\n",
            "<function=Brave_Search_brave_web_search>\n",
            "<parameter=query>\n",
            "vLLM endoftext\n",
            "</parameter>\n",
            "</function>\n",
        ]
        stream_parser = Qwen3Parser(mock_tokenizer, tools=tools)
        results = simulate_tool_streaming(stream_parser, mock_request, chunks)
        assert collect_function_name(results) == "Brave_Search_brave_web_search"

    def test_empty_special_decode_still_emits_tag_literal(self, mock_request):
        """tokenizer.decode(special_id) == '' must not drop the tag."""
        vocab = dict(_QWEN3_VOCAB)
        id_to_text = {v: k for k, v in vocab.items()}

        tokenizer = make_mock_tokenizer(vocab)

        def _decode(ids):
            parts = []
            for i in ids:
                if i in (_THINK_END_ID, _TOOL_CALL_ID):
                    parts.append("")  # empty special decode
                else:
                    parts.append(id_to_text.get(i, chr(i) if i < 128 else f"<{i}>"))
            return "".join(parts)

        tokenizer.decode.side_effect = _decode
        parser = Qwen3Parser(tokenizer)
        reasoning, content = simulate_reasoning_streaming(
            parser,
            [
                "Treat ",
                "<tool_call>",
                " as implicit.",
            ],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
            ],
        )
        assert content == ""
        assert "Treat " in reasoning
        assert "as implicit." in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")


class TestMarkdownInertStructuralTags:
    """Structural tags inside markdown code must not end think or emit tools."""

    @pytest.fixture
    def parser(self, mock_tokenizer):
        return Qwen3Parser(mock_tokenizer)

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            )
        ]
        return Qwen3Parser(mock_tokenizer, tools=tools)

    def test_backticked_think_end_stays_reasoning(self, parser):
        text = "docs say `</think>` closes think but this is a citation."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "docs say" in reasoning
        assert "closes think" in reasoning
        _assert_tag_visible(reasoning, "</think>")

    def test_backticked_tool_call_pr_title_stays_reasoning(self, parser):
        text = "PR #35687: Treat `<tool_call>` as implicit reasoning end."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert "PR #35687" in (reasoning or "")
        assert "as implicit reasoning end" in (reasoning or "")
        _assert_tag_visible(reasoning, "<tool_call>")

    def test_fenced_tool_xml_in_think_stays_reasoning(
        self, parser_with_tools, mock_request
    ):
        text = (
            "Example invoke:\n"
            "```xml\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "```\n"
            "Still thinking about it."
        )
        reasoning, content, tool_calls = parser_with_tools.parse(text, mock_request)
        assert content is None or content == ""
        assert tool_calls is None or tool_calls == []
        assert reasoning is not None
        assert "Still thinking about it." in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")

    def test_endoftext_slash_tool_call_transcript_stays_reasoning(self, parser):
        text = (
            "fixing the <endoftext> / <tool_call> token appearing "
            "in the reasoning block of Qwen3.6 in vLLM."
        )
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        assert "fixing the" in reasoning
        assert "appearing in the reasoning block" in reasoning
        _assert_tag_visible(reasoning, "<tool_call>")

    def test_think_end_then_endoftext_prose_stays_reasoning(self, parser):
        text = "</think><|endoftext|> as a stop token discussion continues."
        reasoning, content = parser.extract_reasoning(text, None)
        assert content is None or content == ""
        assert reasoning is not None
        _assert_tag_visible(reasoning, "</think>")
        assert "discussion continues" in reasoning

    def test_streaming_backticked_think_end(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["docs say `", "</think>", "` closes think."],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert content == ""
        assert "docs say" in reasoning
        assert "closes think." in reasoning
        _assert_tag_visible(reasoning, "</think>")

    def test_dangling_backtick_before_think_end_still_emits_tool(
        self, parser_with_tools, mock_request
    ):
        """Stray unclosed `` ` `` before newline must not swallow </think>/tools."""
        text = (
            "The user is asking about PRs that fix the Qwen3.6 `\n"
            "</think>\n"
            "\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        reasoning, content, tool_calls = parser_with_tools.parse(text, mock_request)
        assert reasoning is not None
        assert "Qwen3.6" in reasoning
        assert "<tool_call>" not in reasoning
        assert "<function=" not in reasoning
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert content is None or "<function=" not in content

    def test_streaming_dangling_backtick_then_tool(
        self, parser_with_tools, mock_request
    ):
        from tests.parser.engine.streaming_helpers import (
            collect_content,
            collect_function_name,
            simulate_tool_streaming,
        )

        chunks = [
            "Asking about Qwen3.6 `",
            "\n",
            "</think>\n\n",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        assert collect_function_name(results) == "get_weather"
        assert "<function=" not in collect_content(results)


class TestMidReasoningCitations:
    """Cited think tags mid-reasoning must never leave holes.

    Screenshot scenario: reasoning discusses the think tags themselves
    ("the issue where `<think>` appears inside the reasoning block").
    A cited ``<think>`` special id mid-think was silently absorbed by the
    leading-strip transition — the sentence renders with a hole / empty
    markdown code span. Inside backticks the tag must also stay raw:
    HTML entities are not interpreted within markdown code spans.
    """

    @pytest.fixture
    def parser(self, mock_tokenizer):
        return Qwen3Parser(mock_tokenizer)

    def test_bare_cited_think_start_visible(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["issue where ", "<think>", " appears. ", "</think>", "Answer."],
            [
                (1,),
                (_THINK_START_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        _assert_tag_visible(reasoning, "<think>")
        assert "issue where" in reasoning
        assert "appears." in reasoning
        assert content == "Answer."

    def test_bare_cited_think_start_visible_nonstreaming(self, parser):
        text = "<think>issue where <think> appears.</think>Answer."
        reasoning, content = parser.extract_reasoning(text, None)
        _assert_tag_visible(reasoning, "<think>")
        assert content == "Answer."

    def test_code_span_cited_think_start_raw(self, parser):
        # Inside backticks the tag is ZWSP-neutralized (never HTML-escaped):
        # client-side tool-markup scanners must not match the raw literal.
        reasoning, _ = simulate_reasoning_streaming(
            parser,
            ["docs say `", "<think>", "` opens think. ", "</think>", "Done"],
            [
                (1,),
                (_THINK_START_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert f"`{_zwsp_tag('<think>')}`" in reasoning
        assert "<think>" not in reasoning
        assert "&lt;" not in reasoning

    def test_code_span_cited_think_end_raw(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["docs say `", "</think>", "` closes think."],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert content == ""
        assert f"`{_zwsp_tag('</think>')}`" in reasoning
        assert "</think>" not in reasoning
        assert "&lt;" not in reasoning

    def test_fenced_block_cited_tag_zwsp(self, parser):
        reasoning, _ = simulate_reasoning_streaming(
            parser,
            ["example:\n```\n", "<tool_call>", "\n``` done. ", "</think>", "A"],
            [
                (1,),
                (_TOOL_CALL_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert _zwsp_tag("<tool_call>") in reasoning
        assert "<tool_call>" not in reasoning
        assert "&lt;" not in reasoning

    def test_leading_think_start_still_stripped(self, parser):
        # Control: the legit template <think> prefix never becomes text.
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>", "thinking. ", "</think>", "Done"],
            [
                (_THINK_START_ID,),
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking. "
        assert content == "Done"

    def test_real_think_end_still_ends_reasoning(self, parser):
        # Control: a genuine close after a cited tag still switches phase.
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["cite `", "<think>", "` here. ", "</think>", "Answer"],
            [
                (1,),
                (_THINK_START_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert content == "Answer"
        assert "</think>" not in reasoning

    def test_endoftext_text_cite_in_code_span_intact(self, mock_tokenizer):
        # Control: non-terminal specials cited as text pass through raw.
        vocab = dict(_QWEN3_VOCAB)
        vocab["<|endoftext|>"] = 70
        parser = Qwen3Parser(make_mock_tokenizer(vocab))
        reasoning, _ = simulate_reasoning_streaming(
            parser,
            ["issue where `", "<|endoftext|>", "` appears. ", "</think>", "A"],
            [
                (1,),
                (2,),
                (3,),
                (_THINK_END_ID,),
                (4,),
            ],
        )
        assert "`<|endoftext|>`" in reasoning


class TestMarkdownAnswerAfterThinkEnd:
    """Answers starting with markdown markup must commit ``</think>``.

    Screenshot scenario: the answer opens with a raw markdown heading
    («# Полный обзор: …») right after the real ``</think>`` id. The
    false-continuation heuristic treated any non-alnum start as think
    prose, so the whole answer rendered inside the reasoning block and
    no content ever arrived. Missing a real think end is catastrophic
    (answer swallowed); falsely closing on a citation only misroutes a
    tail — so continuation must require explicit evidence.
    """

    @pytest.fixture
    def parser(self, mock_tokenizer):
        return Qwen3Parser(mock_tokenizer)

    def _run(self, parser, answer_chunks):
        chunks = ["Разбираю вопрос. ", "</think>", *answer_chunks]
        ids = [(1,), (_THINK_END_ID,)] + [(100 + i,) for i in range(len(answer_chunks))]
        return simulate_reasoning_streaming(parser, chunks, ids)

    def test_heading_answer_goes_to_content(self, parser):
        reasoning, content = self._run(
            parser, ["\n\n# Полный обзор: сравнение подходов\n\nТекст."]
        )
        assert "# Полный обзор" in content
        assert "</think>" not in reasoning
        assert "# Полный обзор" not in reasoning

    def test_heading_hash_as_own_token(self, parser):
        # The heading marker arrives as its own tiny token.
        reasoning, content = self._run(
            parser, ["\n\n", "#", " Полный обзор", ": сравнение\n"]
        )
        assert "# Полный обзор" in content
        assert "</think>" not in reasoning

    def test_list_item_answer_goes_to_content(self, parser):
        reasoning, content = self._run(parser, ["\n\n- пункт первый\n- второй\n"])
        assert "- пункт первый" in content
        assert "</think>" not in reasoning

    def test_bold_answer_goes_to_content(self, parser):
        reasoning, content = self._run(parser, ["\n\n**Жирный** старт ответа."])
        assert "**Жирный**" in content
        assert "</think>" not in reasoning

    def test_blockquote_answer_goes_to_content(self, parser):
        reasoning, content = self._run(parser, ["\n\n> цитата в начале ответа\n"])
        assert "> цитата" in content
        assert "</think>" not in reasoning

    def test_control_code_span_citation_stays_reasoning(self, parser):
        # GREEN control: `` `</think>` `` inside backticks is a citation.
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["tag `", "</think>", "` cited. ", "</think>", "Answer."],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert content == "Answer."
        assert _zwsp_tag("</think>") in reasoning

    def test_control_lowercase_continuation_stays_reasoning(self, parser):
        # GREEN control: bare cited tag + lowercase tail continues think.
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["the ", "</think>", " tag ends it. ", "</think>", "Done."],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert content == "Done."
        assert "tag ends it." in reasoning

    def test_control_uppercase_answer_commits(self, parser):
        reasoning, content = self._run(parser, ["Ответ начинается с заглавной."])
        assert content == "Ответ начинается с заглавной."
        assert "</think>" not in reasoning


class TestWhitespaceStrippingDisabled:
    """When strip_trailing_reasoning_whitespace is False,
    trailing whitespace in reasoning must be preserved."""

    @pytest.fixture
    def parser_no_strip(self):
        cfg = dataclasses.replace(
            qwen3_config(),
            strip_trailing_reasoning_whitespace=False,
        )
        return Qwen3Parser(make_mock_tokenizer(_QWEN3_VOCAB), parser_engine_config=cfg)

    def test_non_streaming_preserves_trailing_newline(self, parser_no_strip):
        text = "Reasoning here.\n</think>Content."
        reasoning, content = parser_no_strip.extract_reasoning(text, None)
        assert reasoning == "Reasoning here.\n"
        assert content == "Content."

    def test_streaming_preserves_trailing_newlines(self, parser_no_strip):
        reasoning, content = simulate_reasoning_streaming(
            parser_no_strip,
            ["thinking.\n", "\n", "</think>", "Done"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking.\n\n"
        assert content == "Done"


class TestThinkingDisabled:
    """When ``enable_thinking=False``, the chat template pre-fills a closed
    ``<think>\\n\\n</think>\\n\\n`` block.  The model output starts in content
    state, so the parser's initial state must be CONTENT — not REASONING.
    """

    def test_thinking_disabled_initial_state_is_content(self, mock_tokenizer):
        p = Qwen3Parser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert p.parser_engine_config.initial_state == ParserState.CONTENT

    def test_thinking_enabled_initial_state_is_reasoning(self, mock_tokenizer):
        p = Qwen3Parser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert p.parser_engine_config.initial_state == ParserState.REASONING

    def test_default_initial_state_is_reasoning(self, mock_tokenizer):
        p = Qwen3Parser(mock_tokenizer)
        assert p.parser_engine_config.initial_state == ParserState.REASONING

    def test_thinking_disabled_streaming_content_only(self, mock_tokenizer):
        """Plain text with thinking disabled must stream as content, not
        reasoning.  Before the fix, the REASONING initial state caused all
        output to be emitted as reasoning chunks."""
        p = Qwen3Parser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )
        reasoning, content = simulate_reasoning_streaming(
            p,
            ["The answer", " is 42."],
            [
                (_TEXT_ID,),
                (_TEXT_ID,),
            ],
        )
        assert content == "The answer is 42."
        assert reasoning == ""

    def test_thinking_disabled_non_streaming(self, mock_tokenizer):
        p = Qwen3Parser(
            mock_tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )
        reasoning, content = p.extract_reasoning("The answer is 42.", None)
        assert reasoning is None
        assert content == "The answer is 42."


class TestDelegatingParserNoToolContentLeak:
    """Serving uses DelegatingParser adapters; tools must not also appear as
    visible content (the 'duplicate tool XML in the answer' bug)."""

    def test_orphan_tools_after_think_not_in_content(self, mock_tokenizer):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionToolsParam,
        )

        tools = [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "Brave_Search_brave_web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "Brave_Search_brave_llm_context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "maximum_number_of_tokens": {"type": "integer"},
                            "maximum_number_of_urls": {"type": "integer"},
                        },
                    },
                },
            ),
        ]
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            include_reasoning=True,
        )
        parser = _Qwen3DelegatingParser(mock_tokenizer, tools=tools)
        # Stream in coarse chunks (think / each tool) like the serving path.
        chunks = [
            "thinking about search.\n</think>\n\n",
            (
                "<function=Brave_Search_brave_web_search>\n"
                "<parameter=query>\n"
                "vllm qwen3 endoftext\n"
                "</parameter>\n"
                "</function>\n"
            ),
            (
                "<function=Brave_Search_brave_llm_context>\n"
                "<parameter=maximum_number_of_tokens>\n"
                "32000\n"
                "</parameter>\n"
                "<parameter=query>\n"
                "vllm PR #35687\n"
                "</parameter>\n"
                "</function>\n"
            ),
        ]
        content_parts: list[str] = []
        tool_names: list[str] = []
        for i, chunk in enumerate(chunks):
            delta = parser.parse_delta(
                chunk,
                [],
                request,
                prompt_token_ids=[1, 2, 3],
                finished=(i == len(chunks) - 1),
            )
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        tool_names.append(tc.function.name)

        content = "".join(content_parts)
        assert "Brave_Search_brave_web_search" in tool_names
        assert "Brave_Search_brave_llm_context" in tool_names
        assert "<function=" not in content
        assert "<parameter=" not in content
        assert "</function>" not in content


class TestToolTagCitationServing:
    """Serving-level contract: cited ``<tool_call>`` never swallows text.

    Screenshot scenario ('PR #35687 ("Treat `<tool_call>` ...' holes in
    both sections): the citation plus the whole tail vanished client-side.
    Pin the server behavior — through ``DelegatingParser.parse_delta``
    with tools enabled, citations (special-id and text form, reasoning
    and answer phase) stream through intact and emit no tool_calls.
    """

    _CITE_PRE = 'PR #35687 ("Treat `'
    _CITE_POST = '` as literal text"). '

    def _replay(self, tokens, chunk_size):
        tokenizer = ReplayMockTokenizer(dict(_QWEN3_VOCAB), tokens)
        parser = _Qwen3DelegatingParser(tokenizer, tools=DUMMY_TOOLS)
        deltas = replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
        )
        return collect_output(deltas)

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_special_id_citation_in_reasoning_code_span(self, chunk_size):
        out = self._replay(
            [
                (100, self._CITE_PRE),
                (_TOOL_CALL_ID, "<tool_call>"),
                (101, self._CITE_POST),
                (_THINK_END_ID, "</think>"),
                (102, "Answer."),
            ],
            chunk_size,
        )
        assert f"`{_zwsp_tag('<tool_call>')}`" in out.reasoning
        assert "<tool_call>" not in out.reasoning
        assert self._CITE_PRE in out.reasoning
        assert '` as literal text"). ' in out.reasoning
        assert out.content == "Answer."
        assert out.tool_calls == []

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_special_id_citation_in_answer_code_span(self, chunk_size):
        out = self._replay(
            [
                (100, "thinking. "),
                (_THINK_END_ID, "</think>"),
                (101, self._CITE_PRE),
                (_TOOL_CALL_ID, "<tool_call>"),
                (102, self._CITE_POST + "Tail continues."),
            ],
            chunk_size,
        )
        assert out.reasoning == "thinking. "
        assert f"`{_zwsp_tag('<tool_call>')}`" in out.content
        assert "<tool_call>" not in out.content
        assert self._CITE_PRE in out.content
        assert "Tail continues." in out.content
        assert out.tool_calls == []

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_special_id_citation_in_answer_prose_keeps_tail(self, chunk_size):
        out = self._replay(
            [
                (100, "thinking. "),
                (_THINK_END_ID, "</think>"),
                (101, "Treat "),
                (_TOOL_CALL_ID, "<tool_call>"),
                (102, " as text (tail preserved)."),
            ],
            chunk_size,
        )
        _assert_tag_visible(out.content, "<tool_call>")
        assert "Treat " in out.content
        assert "as text (tail preserved)." in out.content
        assert out.tool_calls == []

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_text_form_citation_in_reasoning_code_span(self, chunk_size):
        # Tag arrives as plain BPE text (non-special id), not the special id.
        out = self._replay(
            [
                (100, self._CITE_PRE),
                (103, "<tool_call>"),
                (101, self._CITE_POST),
                (_THINK_END_ID, "</think>"),
                (102, "Answer."),
            ],
            chunk_size,
        )
        assert f"`{_zwsp_tag('<tool_call>')}`" in out.reasoning
        assert "<tool_call>" not in out.reasoning
        assert '` as literal text"). ' in out.reasoning
        assert out.content == "Answer."
        assert out.tool_calls == []

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES, ids=lambda c: f"chunk={c}")
    def test_text_form_citation_in_answer_code_span(self, chunk_size):
        out = self._replay(
            [
                (100, "thinking. "),
                (_THINK_END_ID, "</think>"),
                (101, self._CITE_PRE),
                (103, "<tool_call>"),
                (102, self._CITE_POST + "Tail continues."),
            ],
            chunk_size,
        )
        assert f"`{_zwsp_tag('<tool_call>')}`" in out.content
        assert "<tool_call>" not in out.content
        assert "Tail continues." in out.content
        assert out.tool_calls == []
