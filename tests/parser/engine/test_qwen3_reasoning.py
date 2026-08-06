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


def _assert_tag_visible(text: str | None, tag: str) -> None:
    """Tag must appear as raw or HTML-escaped prose (never silently dropped)."""
    assert text is not None
    assert tag in text or _esc_tag(tag) in text, (
        f"expected {tag!r} (or escaped) in {text!r}"
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

    def test_duplicate_think_end_absorbed(self, parser):
        """Duplicate </think> in CONTENT state must not leak."""
        text = "Reasoning here.</think>Content here.</think>More content."
        reasoning, content = parser.extract_reasoning(text, None)
        assert reasoning == "Reasoning here."
        assert content == "Content here.More content."


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
            ["<think>", "thinking", " hard", "</think>", "done"],
            [
                (_THINK_START_ID,),
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking hard"
        assert content == "done"

    def test_streaming_no_start_token(self, parser):
        """Qwen3.5 style: no <think> in output, just reasoning then </think>."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning ", "text", "</think>", "content"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning text"
        assert content == "content"

    def test_streaming_start_token_stripped(self, parser):
        """<think> in output (old template) should be stripped."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["<think>reasoning", "</think>", "content"],
            [
                (_THINK_START_ID, 1),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "content"

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
            ["reasoning", "</think>", "content1", " content2"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "content1 content2"

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
            ["reasoning", "</think>the answer"],
            [
                (1,),
                (_THINK_END_ID, 2),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "the answer"

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
            ["reasoning", "</think>", "content"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert "</think>" not in reasoning
        assert "</think>" not in content
        assert "<think>" not in reasoning

    def test_streaming_duplicate_think_end_absorbed(self, parser):
        """Duplicate </think> token in CONTENT state must not leak."""
        reasoning, content = simulate_reasoning_streaming(
            parser,
            ["reasoning", "</think>", "content", "</think>", "more"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "reasoning"
        assert content == "contentmore"


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
            ["thinking.\n", "</think>", "done"],
            [
                (1,),
                (_THINK_END_ID,),
                (2,),
            ],
        )
        assert reasoning == "thinking."
        assert content == "done"

    def test_streaming_multiple_trailing_newlines_stripped(self, parser_with_strip):
        reasoning, content = simulate_reasoning_streaming(
            parser_with_strip,
            ["thinking.\n", "\n", "\n", "</think>", "done"],
            [
                (1,),
                (2,),
                (3,),
                (_THINK_END_ID,),
                (4,),
            ],
        )
        assert reasoning == "thinking."
        assert content == "done"

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
            ["thinking.\n", "\n", "</think>", "done"],
            [
                (1,),
                (2,),
                (_THINK_END_ID,),
                (3,),
            ],
        )
        assert reasoning == "thinking.\n\n"
        assert content == "done"


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
