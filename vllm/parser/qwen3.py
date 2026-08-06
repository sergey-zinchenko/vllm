# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 parser for tool calls and reasoning.

Qwen3 XML tool call format::

    <tool_call>
    <function=func_name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

The argument body consists of ``<parameter=NAME>VALUE</parameter>`` tags.
The ``_qwen3_arg_converter`` parses these into a JSON object.
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.qwen3_phase_stop import QWEN_END_OF_TEXT, QWEN_IM_END

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)


def _trim_wrapping_newlines(value: str) -> str:
    """Strip one leading and one trailing newline (the Qwen3 template markup)."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = _trim_wrapping_newlines(value)

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = _trim_wrapping_newlines(value)

    return json.dumps(params, ensure_ascii=False)


@functools.cache
def qwen3_config(
    thinking: bool = True,
    *,
    name: str = "qwen3",
    think_start: str = THINK_START,
    think_end: str = THINK_END,
    tool_start: str = TOOL_CALL_START,
    tool_end: str = TOOL_CALL_END,
    tool_call_ends_reasoning: bool = False,
    orphan_func_prefix: bool = True,
    validate_tool_names: bool = True,
) -> ParserEngineConfig:
    """Build the Qwen3-family parser config.

    Args:
        thinking: Start in REASONING when True, else CONTENT.
        tool_call_ends_reasoning: Legacy behavior — bare ``<tool_call>``
            ends think and starts a tool (Nemotron V3). Hardened Qwen3
            keeps this False so markup inside think is plain text.
        orphan_func_prefix: Bare ``<function=`` in CONTENT starts a tool.
            Prose citations without ``<parameter=`` are rejected back to
            text when ``validate_tool_names`` is on.
        validate_tool_names: Reject tool names absent from ``request.tools``.
    """
    transitions: dict[tuple[ParserState, str], Transition] = {
        # -- Reasoning transitions --
        (ParserState.REASONING, "THINK_START"): Transition(
            ParserState.REASONING,
            (),
        ),
        # Defer REASONING_END: models often *mention* ``</think>`` while
        # still thinking (e.g. discussing special tokens). Confirm via
        # streaming engine lookahead (THINK_END_PENDING).
        (ParserState.REASONING, "THINK_END"): Transition(
            ParserState.THINK_END_PENDING,
            (),
        ),
        # Absorb duplicate </think> — model may emit it after
        # already transitioning to CONTENT; drop it silently.
        (ParserState.CONTENT, "THINK_END"): Transition(
            ParserState.CONTENT,
            (),
        ),
        (ParserState.THINK_END_PENDING, "THINK_END"): Transition(
            ParserState.THINK_END_PENDING,
            (),
        ),
        # Bare <tool_call> after deferred </think> is not yet a real end —
        # confirm only when <function= follows (streaming engine). Prose
        # citations (e.g. PR titles) abort back into reasoning with both
        # tags kept as literal text.
        (ParserState.THINK_END_PENDING, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (),
        ),
        # Some Qwen3.6 turns omit <tool_call> and emit <function=...>
        # immediately after </think>. Treat that as a confirmed invoke.
        (ParserState.THINK_END_PENDING, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.REASONING_END, EventType.TOOL_CALL_START),
        ),
        # -- Tool call transitions --
        # Enter preamble without TOOL_CALL_START: a bare ``<tool_call>`` in
        # prose (e.g. citing a PR title) must not open a tool slot. Confirm
        # only when ``<function=`` follows; otherwise the streaming engine
        # aborts preamble back to content.
        (ParserState.CONTENT, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (),
        ),
        (ParserState.TOOL_PREAMBLE, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (),
        ),
        (ParserState.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
        (ParserState.TOOL_NAME, "CLOSE_ANGLE"): Transition(
            ParserState.TOOL_ARGS,
            (),
        ),
        # Malformed: </function> while still in TOOL_NAME (no closing >)
        (ParserState.TOOL_NAME, "FUNC_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        (ParserState.TOOL_ARGS, "FUNC_END"): Transition(
            ParserState.TOOL_BETWEEN,
            (EventType.TOOL_CALL_END,),
        ),
        # Close tool on </tool_call> even when </function> was omitted.
        # Streaming engine keeps nested </tool_call> inside <parameter=...>
        # as argument text via parameter-depth tracking.
        (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (EventType.TOOL_CALL_END,),
        ),
        # Next orphan <function=...> while still in args (previous tool
        # omitted </parameter></function></tool_call>).
        (ParserState.TOOL_ARGS, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START),
        ),
        (ParserState.TOOL_ARGS, "PARAM_START"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_ARGS, "PARAM_END"): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        ),
        (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
            ParserState.CONTENT,
            (),
        ),
        # Consecutive tool call without closing </tool_call>
        (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
            ParserState.TOOL_PREAMBLE,
            (),
        ),
        (ParserState.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        ),
    }
    if tool_call_ends_reasoning:
        # Legacy: <tool_call> from REASONING implicitly ends think.
        # TOOL_CALL_START still waits for <function= (same as content).
        transitions[(ParserState.REASONING, "TOOL_START")] = Transition(
            ParserState.TOOL_PREAMBLE,
            (EventType.REASONING_END,),
        )
    if orphan_func_prefix:
        transitions[(ParserState.CONTENT, "FUNC_PREFIX")] = Transition(
            ParserState.TOOL_NAME,
            (EventType.TOOL_CALL_START,),
        )

    return ParserEngineConfig(
        name=name,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            # Reasoning terminals
            "THINK_START": think_start,
            "THINK_END": think_end,
            # Tool call terminals
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={
            "THINK_START": think_start,
            "THINK_END": think_end,
            "TOOL_START": tool_start,
            "TOOL_END": tool_end,
        },
        transitions=transitions,
        arg_converter=_qwen3_arg_converter,
        stream_arg_deltas=True,
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
        validate_tool_names=validate_tool_names,
        defer_reasoning_end=True,
        # Real tools after short answer prose must still emit; markdown
        # fences/backticks keep doc examples inert instead.
        forbid_tools_after_content=False,
        escape_structural_tags_in_prose=True,
        # Keep discussed stop markers as visible text instead of silent DROP.
        preserve_tokens=frozenset({QWEN_IM_END, QWEN_END_OF_TEXT}),
    )


class Qwen3Parser(ParserEngine):
    """Qwen3 parser: ``<think>``/``</think>`` reasoning +
    ``<tool_call>`` XML tool calls in a single engine.

    Hardened invariants:
    - Tool markup inside reasoning is plain text (never ends think,
      never emits ``tool_calls``).
    - Reasoning ends only on confirmed ``</think>`` (never on unpaired
      ``<tool_call>``). Mid-sentence ``</think>`` / ``<tool_call>``
      mentions stay full reasoning text (no holes, no tool emit).
    - Bare ``<tool_call>`` in content is not a tool until
      ``<function=`` follows; otherwise it streams as text (citations).
    - Structural tags inside markdown `` `...` `` / fenced code are inert
      prose (no reasoning end, no tool emit).
    - Well-formed tools with a known name still emit after answer prose.
    - Orphan ``<function=`` may start a tool (models often omit
      ``<tool_call>``), including right after ``</think>``. Prose
      citations without ``<parameter=`` flush back to content.
    - Only complete invokes whose name is in ``request.tools`` emit
      ``tool_calls``; invalid names flush as content.
    - ``adjust_request`` bans ``<|endoftext|>`` for the whole turn.

    Subclasses that share the grammar but differ only in the four wrapper
    token strings (reasoning + tool-call) override the class attributes
    below; everything else is inherited unchanged.
    """

    CONFIG_NAME = "qwen3"
    THINK_START = THINK_START
    THINK_END = THINK_END
    TOOL_START = TOOL_CALL_START
    TOOL_END = TOOL_CALL_END

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config",
            qwen3_config(
                thinking=self.thinking_enabled,
                name=self.CONFIG_NAME,
                think_start=self.THINK_START,
                think_end=self.THINK_END,
                tool_start=self.TOOL_START,
                tool_end=self.TOOL_END,
            ),
        )
        super().__init__(
            tokenizer,
            tools,
            **kwargs,
        )
        self._tool_call_ends_reasoning = (
            ParserState.REASONING,
            "TOOL_START",
        ) in self.parser_engine_config.transitions
        vocab = self.vocab
        self._tool_call_token_id: int | None = vocab.get(self.TOOL_START)
        self._tool_call_end_token_id: int | None = vocab.get(self.TOOL_END)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Hardened: only ``</think>``. Legacy: unpaired ``<tool_call>`` too."""
        if super().is_reasoning_end(input_ids):
            return True
        if not self._tool_call_ends_reasoning:
            return False
        tool_call_id = self._tool_call_token_id
        tool_call_end_id = self._tool_call_end_token_id
        reasoning_start_id = self._reasoning_start_token_id
        if tool_call_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if (
                    reasoning_start_id is not None
                    and input_ids[i] == reasoning_start_id
                ):
                    return False
                if input_ids[i] == tool_call_id:
                    if tool_call_end_id is not None and any(
                        input_ids[j] == tool_call_end_id
                        for j in range(i + 1, len(input_ids))
                    ):
                        continue
                    return True
        return False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        # Phase-stop wiring is Qwen3-hardened only (not Nemotron legacy).
        if self._tool_call_ends_reasoning or self.CONFIG_NAME != "qwen3":
            return request
        from vllm.parser.qwen3_phase_stop import apply_endoftext_ban_to_request

        apply_endoftext_ban_to_request(
            request,
            self.vocab,
            think_start=self.THINK_START,
            think_end=self.THINK_END,
            tool_start=self.TOOL_START,
            tool_end=self.TOOL_END,
            initial_reasoning=self.thinking_enabled,
        )
        return request
