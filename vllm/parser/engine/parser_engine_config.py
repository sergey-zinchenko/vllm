# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative configuration for model tool-call and reasoning formats.

Each model format is described by a :class:`ParserEngineConfig` that specifies:

* **terminals** – literal strings or regex patterns that delimit the format
  (e.g. ``<tool_call>``, ``</think>``).
* **token_id_terminals** – terminals that should be matched by token ID
  rather than (or in addition to) text.
* **transitions** – a state machine mapping
  ``(state, terminal) → (new_state, events_to_emit)`` that drives semantic
  event generation during streaming.
* **content_events** – what :class:`EventType` to emit for plain content
  (non-terminal text) in each state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property

from vllm.parser.engine.events import EventType


class ParserState(Enum):
    CONTENT = auto()
    REASONING = auto()
    # After ``</think>`` but before we know if it was a real end or a
    # prose mention (e.g. discussing the tag). See ``defer_reasoning_end``.
    THINK_END_PENDING = auto()
    MESSAGE_HEADER = auto()
    TOOL_PREAMBLE = auto()
    TOOL_NAME = auto()
    TOOL_ARGS = auto()
    TOOL_BETWEEN = auto()


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: ParserState
    events: tuple[EventType, ...] = field(default_factory=tuple)
    skip_in_token_id_mode: bool = False


@dataclass(frozen=True)
class ParserEngineConfig:
    """Declarative description of a model's tool-call / reasoning format.

    The engine feeds terminals from the incremental lexer into the
    transition table and emits the corresponding semantic events.
    Content tokens (text between terminals) are classified by the
    current state via ``content_events``.
    """

    name: str

    terminals: dict[str, str] = field(default_factory=dict)

    token_id_terminals: dict[str, str] = field(default_factory=dict)

    transitions: dict[tuple[ParserState, str], Transition] = field(
        default_factory=dict,
    )

    content_events: dict[ParserState, EventType] = field(
        default_factory=lambda: {
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )

    initial_state: ParserState = ParserState.CONTENT

    arg_converter: Callable[[str, bool], str] | None = None

    stream_arg_deltas: bool = True

    tool_args_json: bool = True

    arg_structural_chars: frozenset[str] | None = None

    # Special tokens exempt from auto-drop but not state-machine terminals.
    preserve_tokens: frozenset[str] = field(default_factory=frozenset)

    # Prevents trailing-whitespace accumulation across multi-turn conversations.
    strip_trailing_reasoning_whitespace: bool = True

    # Drop content that is entirely whitespace when tool calls follow.
    drop_whitespace_only_content_before_tools: bool = True

    # .strip() content text when tool calls are present.
    strip_content_whitespace_with_tools: bool = True

    # Reject tool calls whose names are absent from the request tools.
    validate_tool_names: bool = False

    # When True, ``</think>`` enters THINK_END_PENDING instead of immediately
    # ending reasoning. Mid-sentence continuations treat the tag as prose.
    defer_reasoning_end: bool = False

    # When True, after non-whitespace answer content has been emitted, bare
    # ``<tool_call>`` is kept as text (markdown examples / citations) instead
    # of opening a tool. Tools must appear before the answer prose.
    forbid_tools_after_content: bool = False

    # HTML-escape known structural tag literals when emitting them as prose
    # (reasoning absorb / false think-end / post-content tool text) so UIs
    # do not render empty HTML widgets for raw ``<tool_call>`` etc.
    escape_structural_tags_in_prose: bool = False

    @cached_property
    def terminal_defs(self):
        from vllm.parser.engine.incremental_lexer import terminals_from_literals

        return terminals_from_literals(self.terminals)

    @cached_property
    def lexer_shape(self):
        from vllm.parser.engine.incremental_lexer import LexerShape

        return LexerShape(self.terminal_defs)
