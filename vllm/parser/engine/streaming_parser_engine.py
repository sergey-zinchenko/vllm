# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parser engine that orchestrates token ID scanning,
incremental lexing, and state-machine-driven semantic event emission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import (
    CONTENT_TERMINAL,
    IncrementalLexer,
    LexerShape,
    LexToken,
    TerminalDef,
)
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.token_id_scanner import (
    DROP_TERMINAL,
    LexerInput,
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)


@dataclass(slots=True)
class _DropInfo:
    lexer_shape: LexerShape
    extra_token_ids: dict[int, str]


def _build_drop_info(
    config: ParserEngineConfig,
    tokenizer,
) -> _DropInfo | None:
    try:
        special_tokens: list[str] = list(tokenizer.all_special_tokens)
        special_ids: list[int] = list(tokenizer.all_special_ids)
    except (AttributeError, NotImplementedError):
        return None

    if not special_tokens:
        return None

    configured_texts = (
        set(config.token_id_terminals.values())
        | set(config.terminals.values())
        | config.preserve_tokens
    )

    extra_token_ids: dict[int, str] = {}
    drop_texts: set[str] = set()
    for text, tid in zip(special_tokens, special_ids):
        if text not in configured_texts:
            extra_token_ids[tid] = DROP_TERMINAL
            drop_texts.add(text)

    if not drop_texts:
        return None

    import regex as re

    drop_terminal_defs = [
        TerminalDef(
            name=DROP_TERMINAL,
            pattern=re.compile(re.escape(text)),
            is_literal=True,
            literal=text,
        )
        for text in drop_texts
    ]

    all_terminal_defs = list(config.terminal_defs) + drop_terminal_defs
    lexer_shape = LexerShape(all_terminal_defs)

    return _DropInfo(
        lexer_shape=lexer_shape,
        extra_token_ids=extra_token_ids,
    )


class StreamingParserEngine:
    """Consumes ``(delta_text, delta_token_ids)`` pairs and produces a
    stream of :class:`SemanticEvent` instances.

    This is the main entry point for streaming parsing.
    Create one per request (it is stateful).

    The pipeline is::

        delta_text + delta_token_ids
            → TokenIDScanner  (special token pre-lexing)
            → IncrementalLexer  (text → terminal tokens with prefix buffering)
            → State Machine  (terminal → semantic events)
            → list[SemanticEvent]

    Usage::

        engine = StreamingParserEngine(config, tokenizer)
        for each streaming delta:
            events = engine.feed(delta_text, delta_token_ids)
            # convert events to DeltaMessage
    """

    def __init__(
        self,
        config: ParserEngineConfig,
        tokenizer,
        initial_state: ParserState | None = None,
        vocab: dict[str, int] | None = None,
    ) -> None:
        self.config = config

        resolved_token_ids: dict[int, str] = {}
        if tokenizer is not None:
            if vocab is None:
                vocab = tokenizer.get_vocab()
            if config.token_id_terminals:
                for terminal_name, token_text in config.token_id_terminals.items():
                    tid = vocab.get(token_text)
                    if tid is not None:
                        resolved_token_ids[tid] = terminal_name

        drop_info: _DropInfo | None = None
        if tokenizer is not None:
            drop_info = _build_drop_info(config, tokenizer)

        lexer_shape = config.lexer_shape
        if drop_info is not None:
            resolved_token_ids.update(drop_info.extra_token_ids)
            lexer_shape = drop_info.lexer_shape

        self._resolved_token_ids = resolved_token_ids
        self._has_drops = drop_info is not None

        self._scanner = TokenIDScanner(
            resolved_token_ids,
            tokenizer,
        )

        self._token_id_terminal_names: frozenset[str] = frozenset(
            resolved_token_ids.values()
        )

        self._lexer = IncrementalLexer(lexer_shape, content_terminal=CONTENT_TERMINAL)

        self._tool_terminals: frozenset[str] = frozenset(
            terminal
            for (state, terminal), tr in config.transitions.items()
            if tr.next_state in self._TOOL_STATES or state in self._TOOL_STATES
        )

        self.skip_tool_parsing = False
        self.reset(initial_state=initial_state)

    def _reset_args_state(self) -> None:
        self._args_buffer: str = ""
        self._args_safe_end: int = 0
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape_next: bool = False

    def reset(self, initial_state: ParserState | None = None) -> None:
        """Reset mutable state for reuse across requests.

        Preserves cached immutable structures (compiled terminals,
        resolved token IDs, lexer shape, token text cache) to avoid
        redundant initialization work.
        """
        self.state = (
            initial_state if initial_state is not None else self.config.initial_state
        )
        self.tool_index = -1
        self._ever_had_token_ids = False
        # DO NOT reset skip_tool_parsing here — callers set it before
        # calling methods that trigger reset() (e.g. extract_reasoning),
        # and clearing it silently breaks non-streaming tool-call-as-
        # implicit-reasoning-end (content returns None).
        self._scanner.reset()
        self._lexer.reset()
        self._message_header_buffer = ""
        self._tool_preamble_open = ""
        self._tool_preamble_buffer = ""
        self._think_end_marker = ""
        self._think_end_pending_buffer = ""
        self._answer_content_started = False
        # True when TOOL_PREAMBLE was entered from THINK_END_PENDING; real
        # REASONING_END waits for <function=, prose abort returns to reasoning.
        self._reasoning_end_before_tool = False
        # Markdown code tracking: structural tags inside `...` / ```...```
        # must stay inert (no REASONING_END / tool transitions).
        self._md_inline_odd = False
        self._md_in_fence = False
        # Open <parameter=...> depth inside TOOL_ARGS (nested </tool_call>).
        self._param_depth = 0
        self._reset_args_state()

    def feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> list[SemanticEvent]:
        if delta_token_ids:
            self._ever_had_token_ids = True

        # Fast path: skip scanner and lexer when the delta is plain
        # content with no special tokens and no terminal-starting chars.
        if (
            delta_text
            and not self._lexer.buffer
            and not self._scanner._deferred_terminals
            and self._lexer._literal_first_chars.isdisjoint(delta_text)
        ):
            has_special = False
            for tid in delta_token_ids:
                if tid in self._resolved_token_ids:
                    has_special = True
                    break
            if not has_special:
                return self._emit_for_state(delta_text)

        scanner_items = self._scanner.scan(delta_text, delta_token_ids)

        if len(scanner_items) == 1 and isinstance(scanner_items[0], TextChunk):
            lex_tokens = self._lexer.feed(scanner_items[0].text)
            if len(lex_tokens) == 1 and lex_tokens[0].terminal == CONTENT_TERMINAL:
                text = lex_tokens[0].value
                return self._emit_for_state(text)
            return self._process_lex_tokens(lex_tokens)

        return self._process_scanner_items(scanner_items)

    def _process_scanner_items(
        self, items: Sequence[LexerInput]
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        for item in items:
            if isinstance(item, PreLexedTerminal):
                events.extend(self._process_lex_tokens(self._lexer.flush()))
                events.extend(self._on_terminal(item.terminal, item.text))
            elif isinstance(item, TextChunk):
                events.extend(self._process_lex_tokens(self._lexer.feed(item.text)))
        return events

    def finish(self) -> list[SemanticEvent]:
        events = self._process_scanner_items(self._scanner.flush_pending())

        events.extend(self._process_lex_tokens(self._lexer.flush()))

        if self._args_buffer:
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""
            self._args_safe_end = 0

        if self.state == ParserState.TOOL_PREAMBLE:
            if self.tool_index >= 0:
                # Legacy configs emit TOOL_CALL_START on ``<tool_call>``;
                # finalize the open slot. Hardened Qwen3 confirms later.
                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
                self._tool_preamble_open = ""
                self._tool_preamble_buffer = ""
                self._reasoning_end_before_tool = False
                self.state = ParserState.CONTENT
            else:
                # Unconfirmed ``<tool_call>`` (no ``<function=``) → text.
                # Never leave the stream silent: dropped preamble text is what
                # makes clients appear "hung" while the GPU keeps decoding.
                events.extend(self._flush_tool_preamble_as_text())
        elif self.state in (
            ParserState.TOOL_ARGS,
            ParserState.TOOL_NAME,
            ParserState.TOOL_BETWEEN,
        ):
            if self.tool_index >= 0:
                events.append(
                    SemanticEvent(
                        EventType.TOOL_CALL_END,
                        tool_index=self.tool_index,
                    )
                )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.THINK_END_PENDING:
            # Stream ended after </think> — commit the deferred end.
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            if self._think_end_pending_buffer:
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=self._think_end_pending_buffer,
                        tool_index=self.tool_index,
                    )
                )
            self._think_end_marker = ""
            self._think_end_pending_buffer = ""
            self.state = ParserState.CONTENT
        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.MESSAGE_HEADER:
            if self._message_header_buffer:
                events.append(
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=self._message_header_buffer,
                        tool_index=self.tool_index,
                    )
                )
                self._message_header_buffer = ""
            self.state = ParserState.CONTENT

        return events

    def parse_complete(self, text: str) -> list[SemanticEvent]:
        token_ids: list[int] = []
        events = self.feed(text, token_ids)
        events.extend(self.finish())
        return events

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        strict = self._token_id_terminal_names if self._ever_had_token_ids else None
        for tok in tokens:
            # Once any special-token id has been seen, text-matched copies of
            # token_id_terminals are normally demoted to prose so lookalikes
            # (e.g. a user mentioning ``<tool_call>``) do not fire.  Exception:
            # when the current state has a real transition for that terminal,
            # keep it — Qwen often emits ``</think>`` as a special token but
            # ``<tool_call>`` as ordinary text, and demoting here dumps the
            # whole tool invoke into content (THINK_END_PENDING / CONTENT).
            if tok.terminal == CONTENT_TERMINAL or (
                strict
                and tok.terminal in strict
                and (self.state, tok.terminal) not in self.config.transitions
            ):
                events.extend(self._on_content(tok.value))
            else:
                events.extend(self._on_terminal(tok.terminal, tok.value))
        return events

    _TOOL_STATES = frozenset(
        {
            ParserState.TOOL_PREAMBLE,
            ParserState.TOOL_NAME,
            ParserState.TOOL_ARGS,
            ParserState.TOOL_BETWEEN,
        }
    )

    def _terminal_text(self, terminal: str, value: str) -> str:
        """Prefer *value*, else the configured literal (empty decode-safe)."""
        if value:
            return value
        return self.config.terminals.get(terminal, "") or ""

    def _in_markdown_code(self) -> bool:
        return self._md_in_fence or self._md_inline_odd

    def _is_structural_terminal(self, terminal: str) -> bool:
        lit = self.config.terminals.get(terminal, "")
        return bool(lit) and "<" in lit

    def _feed_markdown_state(self, text: str) -> None:
        """Update inline-backtick / fenced-code state from emitted text.

        Inline spans do not survive newlines: a dangling `` ` `` must not
        keep suppressing ``</think>`` / tool terminals on the next line.
        Fenced ``` blocks are unchanged.
        """
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            if text.startswith("```", i):
                self._md_in_fence = not self._md_in_fence
                if self._md_in_fence:
                    self._md_inline_odd = False
                i += 3
                continue
            if not self._md_in_fence and text[i] == "`":
                self._md_inline_odd = not self._md_inline_odd
                i += 1
                continue
            if not self._md_in_fence and self._md_inline_odd and text[i] == "\n":
                self._md_inline_odd = False
                i += 1
                continue
            i += 1

    def _escape_prose_text(self, text: str) -> str:
        """HTML-escape known structural tag literals for UI-safe prose."""
        if not text or not self.config.escape_structural_tags_in_prose:
            return text
        # Only tag-like literals (must contain ``<``). Skip bare ``>`` /
        # other punctuation terminals so prose like ``<|im_end|>`` is intact
        # except for known structural tags.
        literals = sorted(
            (lit for lit in self.config.terminals.values() if lit and "<" in lit),
            key=len,
            reverse=True,
        )
        for lit in literals:
            if lit in text:
                escaped = (
                    lit.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                text = text.replace(lit, escaped)
        return text

    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        key = (self.state, terminal)
        transition = self.config.transitions.get(key)
        text = self._terminal_text(terminal, value)

        if self._has_drops and terminal == DROP_TERMINAL and transition is None:
            return []

        # Inside markdown code: all structural tags are inert prose.
        if self._in_markdown_code() and self._is_structural_terminal(terminal):
            return self._emit_for_state(text)

        if transition is None:
            # Absorbed structural tag (e.g. <tool_call> inside REASONING).
            return self._emit_for_state(text)

        if self.skip_tool_parsing and terminal in self._tool_terminals:
            if self.state == ParserState.MESSAGE_HEADER:
                self.state = ParserState.CONTENT
                self._message_header_buffer = ""
                return [
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    )
                ]
            if EventType.REASONING_END in transition.events:
                self.state = ParserState.CONTENT
                return [
                    SemanticEvent(
                        EventType.REASONING_END,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                    SemanticEvent(
                        EventType.TEXT_CHUNK,
                        value=value,
                        tool_index=self.tool_index,
                    ),
                ]
            content_type = self.config.content_events.get(self.state)
            if content_type is not None:
                return [
                    SemanticEvent(content_type, value=value, tool_index=self.tool_index)
                ]
            return []

        if transition.skip_in_token_id_mode and self._ever_had_token_ids:
            return self._emit_for_state(text)

        # Nested </tool_call> inside an open <parameter=...> is argument text.
        if (
            self.state == ParserState.TOOL_ARGS
            and terminal == "TOOL_END"
            and self._param_depth > 0
        ):
            return self._emit_for_state(text)

        if self.state == ParserState.TOOL_ARGS and terminal == "PARAM_START":
            self._param_depth += 1
        elif self.state == ParserState.TOOL_ARGS and terminal == "PARAM_END":
            self._param_depth = max(0, self._param_depth - 1)

        return self._apply_transition(transition, text)

    def _note_answer_content(self, text: str) -> None:
        if text.strip():
            self._answer_content_started = True

    def _flush_tool_preamble_as_text(self, extra: str = "") -> list[SemanticEvent]:
        """Abort an unconfirmed tool preamble back to ordinary text.

        If preamble followed a deferred ``</think>`` (reasoning still open),
        reconstruct ``</think>`` + ``<tool_call>`` + prose as reasoning.
        Otherwise emit as content (answer-phase citations).
        """
        open_tag = self._terminal_text("TOOL_START", self._tool_preamble_open)
        preamble = f"{open_tag}{self._tool_preamble_buffer}{extra}"
        self._tool_preamble_open = ""
        self._tool_preamble_buffer = ""

        if self._reasoning_end_before_tool or self._think_end_marker:
            marker = self._terminal_text("THINK_END", self._think_end_marker)
            think_ws = self._think_end_pending_buffer
            self._think_end_marker = ""
            self._think_end_pending_buffer = ""
            self._reasoning_end_before_tool = False
            self.state = ParserState.REASONING
            raw = f"{marker}{think_ws}{preamble}"
            self._feed_markdown_state(raw)
            text = self._escape_prose_text(raw)
            if not text:
                return []
            return [
                SemanticEvent(
                    EventType.REASONING_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                )
            ]

        self.state = ParserState.CONTENT
        if not preamble:
            return []
        self._feed_markdown_state(preamble)
        self._note_answer_content(preamble)
        return [
            SemanticEvent(
                EventType.TEXT_CHUNK,
                value=preamble,
                tool_index=self.tool_index,
            )
        ]

    @classmethod
    def _is_false_think_end_continuation(cls, text: str) -> bool:
        """Whether text after ``</think>`` looks like mid-sentence prose.

        Real answers commit the end when they start with an Uppercase letter
        (or tool markup). Lowercase / punctuation / ``<|…|>`` prose means
        the tag was *mentioned* inside reasoning — stay in think.
        """
        if not text.strip():
            return True
        body = text.lstrip(" \t\r")
        check = body.lstrip("\n") if body.startswith("\n") else text.lstrip()
        if not check:
            return True
        # Uppercase starts a real answer. Leading '<' is usually still
        # think prose (e.g. <|endoftext|>). Real tool markup commits the
        # end; TOOL_START / FUNC_PREFIX terminals normally handle this,
        # but keep the same rule if those tags arrive as plain text.
        if check[0].isupper():
            return False
        if check.startswith("<function=") or check.startswith("<tool_call"):
            return False
        if check[0] == "<":
            return True
        if check[0] in ",;:)]}'\"`":
            return True
        # Lowercase (and any other non-answer start) continues reasoning.
        return True

    def _resolve_think_end_pending(self, text: str) -> list[SemanticEvent]:
        """Commit or abort a deferred ``</think>`` using following text."""
        if not text.strip():
            self._think_end_pending_buffer += text
            return []

        marker = self._terminal_text("THINK_END", self._think_end_marker)
        buffered = self._think_end_pending_buffer
        self._think_end_marker = ""
        self._think_end_pending_buffer = ""

        if self._is_false_think_end_continuation(text):
            self.state = ParserState.REASONING
            raw = f"{marker}{buffered}{text}"
            self._feed_markdown_state(raw)
            return [
                SemanticEvent(
                    EventType.REASONING_CHUNK,
                    value=self._escape_prose_text(raw),
                    tool_index=self.tool_index,
                )
            ]

        self.state = ParserState.CONTENT
        events = [
            SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index),
        ]
        content = f"{buffered}{text}"
        if not content:
            return events
        # Tool markup may arrive in the same content blob that confirms
        # ``</think>`` (no separate special-token scan). Re-lex it so
        # ``<tool_call>`` / ``<function=`` still open a tool instead of
        # leaking into answer text.
        stripped = content.lstrip(" \t\r\n")
        if stripped.startswith("<tool_call") or stripped.startswith("<function="):
            events.extend(self._process_lex_tokens(self._lexer.feed(content)))
            return events
        self._feed_markdown_state(content)
        self._note_answer_content(content)
        events.append(
            SemanticEvent(
                EventType.TEXT_CHUNK,
                value=content,
                tool_index=self.tool_index,
            )
        )
        return events

    def _emit_for_state(self, text: str) -> list[SemanticEvent]:
        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            return []
        if self.state == ParserState.THINK_END_PENDING:
            return self._resolve_think_end_pending(text)
        if self.state == ParserState.TOOL_PREAMBLE:
            # Real tools allow whitespace between ``<tool_call>`` and
            # ``<function=``. Anything else means this was prose.
            if not text.strip():
                self._tool_preamble_buffer += text
                return []
            return self._flush_tool_preamble_as_text(text)
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
            return [
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                )
            ]
        content_type = self.config.content_events.get(self.state)
        if content_type is not None:
            if content_type in (EventType.REASONING_CHUNK, EventType.TEXT_CHUNK):
                self._feed_markdown_state(text)
            if content_type == EventType.REASONING_CHUNK:
                text = self._escape_prose_text(text)
            if (
                content_type == EventType.TEXT_CHUNK
                and self.state == ParserState.CONTENT
            ):
                self._note_answer_content(text)
            return [SemanticEvent(content_type, value=text, tool_index=self.tool_index)]
        return []

    def _on_content(self, text: str) -> list[SemanticEvent]:
        if not text:
            return []
        return self._emit_for_state(text)

    def _apply_transition(
        self,
        transition: Transition,
        value: str,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        previous_state = self.state
        message_header = ""

        if (
            self.state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
            and self._args_buffer
        ):
            events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=self._args_buffer,
                    tool_index=self.tool_index,
                )
            )
            self._args_buffer = ""

        if previous_state == ParserState.MESSAGE_HEADER:
            message_header = self._message_header_buffer
            self._message_header_buffer = ""

        # Remember the opening tag when entering an unconfirmed preamble.
        if (
            transition.next_state == ParserState.TOOL_PREAMBLE
            and previous_state != ParserState.TOOL_PREAMBLE
        ):
            self._tool_preamble_open = self._terminal_text("TOOL_START", value)
            self._tool_preamble_buffer = ""
            if previous_state == ParserState.THINK_END_PENDING:
                # Keep </think> marker; confirm REASONING_END only on
                # <function= (real tool after think).
                self._reasoning_end_before_tool = True

        # Enter deferred </think> hold.
        if (
            transition.next_state == ParserState.THINK_END_PENDING
            and previous_state == ParserState.REASONING
        ):
            self._think_end_marker = self._terminal_text("THINK_END", value)
            self._think_end_pending_buffer = ""

        # Extra </think> while still deciding — keep as literal text.
        if (
            previous_state == ParserState.THINK_END_PENDING
            and transition.next_state == ParserState.THINK_END_PENDING
        ):
            self._think_end_pending_buffer += self._terminal_text("THINK_END", value)
            self.state = ParserState.THINK_END_PENDING
            return []

        # Empty ``<tool_call></tool_call>`` (no ``<function=``): surface as
        # text instead of a tool slot / silent drop.
        if (
            previous_state == ParserState.TOOL_PREAMBLE
            and transition.next_state == ParserState.CONTENT
            and EventType.TOOL_CALL_START not in transition.events
            and self.tool_index < 0
        ):
            return self._flush_tool_preamble_as_text(
                self._terminal_text("TOOL_END", value)
            )

        # Confirmed invoke from preamble: emit deferred REASONING_END first.
        if (
            previous_state == ParserState.TOOL_PREAMBLE
            and transition.next_state != ParserState.TOOL_PREAMBLE
            and EventType.TOOL_CALL_START in transition.events
        ):
            if self._reasoning_end_before_tool:
                events.append(
                    SemanticEvent(
                        EventType.REASONING_END,
                        tool_index=self.tool_index,
                    )
                )
                self._reasoning_end_before_tool = False
                self._think_end_marker = ""
                self._think_end_pending_buffer = ""
            self._tool_preamble_open = ""
            self._tool_preamble_buffer = ""
        elif (
            previous_state == ParserState.TOOL_PREAMBLE
            and transition.next_state != ParserState.TOOL_PREAMBLE
        ):
            self._tool_preamble_open = ""
            self._tool_preamble_buffer = ""

        # Legacy: REASONING_END on the same transition that leaves pending.
        if (
            previous_state == ParserState.THINK_END_PENDING
            and EventType.REASONING_END in transition.events
        ):
            self._think_end_marker = ""
            self._think_end_pending_buffer = ""
            self._reasoning_end_before_tool = False

        if (
            previous_state == ParserState.TOOL_ARGS
            and transition.next_state != ParserState.TOOL_ARGS
        ):
            self._param_depth = 0

        self.state = transition.next_state

        for event_type in transition.events:
            if event_type == EventType.TOOL_CALL_START:
                self.tool_index += 1
            event_value = (
                message_header
                if previous_state == ParserState.MESSAGE_HEADER
                and event_type == EventType.TEXT_CHUNK
                else value
            )
            if event_type == EventType.TEXT_CHUNK and not event_value:
                continue
            events.append(
                SemanticEvent(
                    event_type,
                    value=event_value,
                    tool_index=self.tool_index,
                )
            )

        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape_next = False
            self._args_safe_end = 0

        return events

    def _feed_args_text(self, text: str) -> list[SemanticEvent]:
        """Feed text into the JSON argument streaming buffer.

        Streams argument characters incrementally while holding back
        closing braces/brackets that might change as more input arrives.
        """
        events: list[SemanticEvent] = []
        for ch in text:
            result = self._feed_args_char(ch)
            events.extend(result)
        return events

    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:
        self._args_buffer += ch

        if self._args_escape_next:
            self._args_escape_next = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if self._args_in_string:
            if ch == "\\":
                self._args_escape_next = True
            elif ch == '"':
                self._args_in_string = False
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch == '"':
            self._args_in_string = True
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("{", "["):
            self._args_brace_depth += 1
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        if ch in ("}", "]"):
            if self._args_brace_depth > 0:
                self._args_brace_depth -= 1
            if self._args_brace_depth == 0:
                return []
            self._args_safe_end = len(self._args_buffer)
            return self._flush_safe_args()

        self._args_safe_end = len(self._args_buffer)
        return self._flush_safe_args()

    def _flush_safe_args(self) -> list[SemanticEvent]:
        """Emit buffered argument characters up to the safe-end watermark.

        Top-level closing braces are held back (safe_end not advanced)
        until confirmed safe by a subsequent character or finish().
        """
        if self._args_safe_end == 0:
            return []
        to_emit = self._args_buffer[: self._args_safe_end]
        self._args_buffer = self._args_buffer[self._args_safe_end :]
        self._args_safe_end = 0
        return [
            SemanticEvent(
                EventType.ARG_VALUE_CHUNK,
                value=to_emit,
                tool_index=self.tool_index,
            )
        ]
