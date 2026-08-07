# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 special-token stop / ban helpers (MTP-safe).

Product invariants:
1. ``<|endoftext|>`` is always banned for chat/tool serving (turn ends via
   ``im_end`` / max_tokens / tool end).
2. ``<|im_end|>`` is phase-aware: banned while the accepted output is still
   in REASONING; allowed once ``</think>`` has closed think.
3. After think closes, ``im_end`` is also banned for the single step where
   the previous token is a lone `` ` `` (stop right after an opening
   backtick). This is a non-sticky logits nudge only: it never affects
   ``check_stop``, and any non-backtick token re-allows ``im_end``.

Open ``<tool_call>`` regions intentionally do **not** ban ``im_end``.
Models often cite ``<tool_call>`` as prose in the answer phase without a
matching ``</tool_call>``; treating that as an open tool made stop
impossible (GPU keeps decoding, client stream looks hung). Incomplete
real tool calls are finalized by the parser ``finish()`` path instead.

These helpers intentionally avoid ``logit_bias`` (rejected by
``SamplingParams._validate_spec_decode`` under MTP). Endoftext is banned
via ``bad_words``; phase-aware ``im_end`` uses the builtin
``Qwen3PhaseStopLogitsProcessor`` (kept under speculative decoding) and
``check_stop`` as a safety net.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

QWEN_END_OF_TEXT = "<|endoftext|>"
QWEN_IM_END = "<|im_end|>"
QWEN_IM_START = "<|im_start|>"

# Default ChatML / Qwen special-token strings used for phase detection.
_THINK_START = "<think>"
_THINK_END = "</think>"
_TOOL_START = "<tool_call>"
_TOOL_END = "</tool_call>"
_INLINE_BACKTICK = "`"
_FENCE_BACKTICK = "```"


# Flattened SamplingParams.extra_args / vllm_xargs key.
# Values: [im_end_id, think_start_id, think_end_id, tool_start_id,
#          tool_end_id, initial_reasoning (0/1),
#          backtick_id?, fence_id?, *trailing_backtick_ids] — missing ids
# are -1. Trailing fields are optional for backward-compatible parse;
# indices 8+ are merged BPE tokens whose text ends with one backtick.
PHASE_BAN_XARG_KEY = "qwen3_phase_ban"

PhaseBanConfig = tuple[
    int,
    int | None,
    int | None,
    int | None,
    int | None,
    bool,
    int | None,
    int | None,
    frozenset[int],
]

# One vocab scan per tokenizer: cache keyed by caller-provided identity
# (vocab dicts are recreated per parser instance, tokenizers are not).
_TRAILING_BACKTICK_CACHE: dict[tuple[int, int], list[int]] = {}


def trailing_backtick_token_ids(
    vocab: Mapping[str, int],
    *,
    cache_key: int | None = None,
) -> list[int]:
    """Vocab ids whose token text ends with exactly one `` ` ``.

    Prose opening backticks arrive as merged BPE tokens (``" `"`` /
    ``"Ġ`"``, ``"(`"`` ...), not as the lone `` ` `` id. Tokens ending in
    ``` `` ``` are fence-ish closers and excluded (stop after them is
    legit). The lone `` ` `` itself is excluded — it travels as the
    dedicated config field.
    """
    if cache_key is not None:
        key = (cache_key, len(vocab))
        cached = _TRAILING_BACKTICK_CACHE.get(key)
        if cached is not None:
            return cached
    ids = sorted(
        tid
        for tok, tid in vocab.items()
        if tok != "`" and tok.endswith("`") and not tok.endswith("``")
    )
    if cache_key is not None:
        _TRAILING_BACKTICK_CACHE[(cache_key, len(vocab))] = ids
    return ids


def resolve_vocab_token_id(vocab: Mapping[str, int], token: str) -> int | None:
    """Return token id from *vocab*, or ``None`` if absent."""
    tid = vocab.get(token)
    return int(tid) if tid is not None else None


def resolve_endoftext_token_id(vocab: Mapping[str, int]) -> int | None:
    return resolve_vocab_token_id(vocab, QWEN_END_OF_TEXT)


def resolve_im_end_token_id(vocab: Mapping[str, int]) -> int | None:
    return resolve_vocab_token_id(vocab, QWEN_IM_END)


def apply_endoftext_ban_to_request(
    request: Any,
    vocab: Mapping[str, int],
    *,
    think_start: str = _THINK_START,
    think_end: str = _THINK_END,
    tool_start: str = _TOOL_START,
    tool_end: str = _TOOL_END,
    initial_reasoning: bool = True,
    vocab_cache_key: int | None = None,
) -> None:
    """Ban ``<|endoftext|>`` and enable phase-aware ``im_end`` bans.

    Do **not** set ``logit_bias``: ``SamplingParams._validate_spec_decode``
    rejects it when MTP / speculative decoding is enabled (the qwen36-27b
    production path). Use ``bad_words`` instead — that path works with
    drafts on both sampler implementations.

    ``vllm_xargs[qwen3_phase_ban]`` carries flattened ids for the builtin
    ``Qwen3PhaseStopLogitsProcessor`` / ``check_stop`` im_end masking.
    """
    eot_id = resolve_endoftext_token_id(vocab)
    if eot_id is not None:
        bad_words = getattr(request, "bad_words", None)
        if bad_words is None:
            # ResponsesRequest has no bad_words field — skip quietly.
            if hasattr(request, "bad_words"):
                request.bad_words = [QWEN_END_OF_TEXT]
        elif QWEN_END_OF_TEXT not in bad_words:
            bad_words.append(QWEN_END_OF_TEXT)

    im_end_id = resolve_im_end_token_id(vocab)
    if im_end_id is None:
        return

    def _id(token: str) -> int:
        tid = resolve_vocab_token_id(vocab, token)
        return tid if tid is not None else -1

    phase_ban = [
        im_end_id,
        _id(think_start),
        _id(think_end),
        _id(tool_start),
        _id(tool_end),
        1 if initial_reasoning else 0,
        _id(_INLINE_BACKTICK),
        _id(_FENCE_BACKTICK),
        *trailing_backtick_token_ids(vocab, cache_key=vocab_cache_key),
    ]
    xargs = getattr(request, "vllm_xargs", None)
    if xargs is None:
        if hasattr(request, "vllm_xargs"):
            request.vllm_xargs = {PHASE_BAN_XARG_KEY: phase_ban}
    else:
        xargs[PHASE_BAN_XARG_KEY] = phase_ban


def parse_phase_ban_config(
    extra_args: Mapping[str, Any] | None,
) -> PhaseBanConfig | None:
    """Parse flattened ``qwen3_phase_ban`` from SamplingParams.extra_args.

    Returns ``(im_end_id, think_start, think_end, tool_start, tool_end,
    initial_reasoning, backtick_id, fence_id, trailing_backtick_ids)`` or
    ``None`` if unset/invalid. Fields past index 5 are optional; legacy
    shorter configs parse with ``None`` / empty tail.
    """
    if not extra_args:
        return None
    raw = extra_args.get(PHASE_BAN_XARG_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        return None
    im_end_id = int(raw[0])
    if im_end_id < 0:
        return None

    def _opt(v: Any) -> int | None:
        i = int(v)
        return i if i >= 0 else None

    backtick_id = _opt(raw[6]) if len(raw) > 6 else None
    fence_id = _opt(raw[7]) if len(raw) > 7 else None
    trailing = frozenset(int(v) for v in raw[8:] if int(v) >= 0)
    return (
        im_end_id,
        _opt(raw[1]),
        _opt(raw[2]),
        _opt(raw[3]),
        _opt(raw[4]),
        bool(int(raw[5])),
        backtick_id,
        fence_id,
        trailing,
    )


def is_in_reasoning_or_tool_phase(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    tool_start_id: int | None,
    tool_end_id: int | None,
    initial_reasoning: bool = True,
) -> bool:
    """Heuristic phase detection from accepted output token ids.

    Returns ``True`` when ``im_end`` must be banned (still inside think).

    One-way latch: after the first ``think_end`` the phase never re-enters
    reasoning. A generation has at most one legitimate think block
    (post-tool re-think is a separate request), so a later ``think_start``
    id is a citation in the answer — re-arming the ban on it made
    ``im_end`` unbannable forever (endless generation / rewritten tails).

    ``tool_start_id`` / ``tool_end_id`` are accepted for config compatibility
    but do not affect the ban: unpaired ``<tool_call>`` in prose must not
    block stop tokens.
    """
    del tool_start_id, tool_end_id
    in_reasoning = initial_reasoning
    think_closed = False

    for tid in output_token_ids:
        if think_start_id is not None and tid == think_start_id:
            if not think_closed:
                in_reasoning = True
            continue
        if think_end_id is not None and tid == think_end_id:
            in_reasoning = False
            think_closed = True
            continue

    return in_reasoning


def ends_with_dangling_backtick(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    backtick_id: int | None,
    fence_id: int | None = None,
    initial_reasoning: bool = True,
    trailing_backtick_ids: frozenset[int] = frozenset(),
) -> bool:
    """True when the last token ends with an opening `` ` `` — any phase.

    Matches the lone `` ` `` id and merged BPE forms (``" `"``, ``"(`"``
    ...) whose text ends with exactly one backtick — prose openings almost
    never tokenize as the bare backtick, and the truncation happens right
    after those merged forms. Applies inside reasoning too: the model
    cites `` `</think>` `` mid-think, and only the structural-id ban keeps
    the real special token from closing the block mid-citation.

    Deliberately **not** span parity: counting one vocab id sees only one
    side of real inline spans, sticks odd, and bans ``im_end`` for the
    rest of the request (endless ``!!!`` tails, rewritten endings under
    MTP). Checking only the immediately preceding token blocks the "stop
    right after opening backtick" pattern while staying non-sticky: any
    other token re-allows ``im_end``.
    """
    del think_start_id, think_end_id, fence_id, initial_reasoning
    if not output_token_ids:
        return False
    last = output_token_ids[-1]
    return last == backtick_id or last in trailing_backtick_ids


def should_ban_im_end(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    tool_start_id: int | None,
    tool_end_id: int | None,
    initial_reasoning: bool = True,
    backtick_id: int | None = None,
    fence_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
) -> bool:
    """True when ``im_end`` must be banned for the next decode step."""
    if is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
    ):
        return True
    return ends_with_dangling_backtick(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        backtick_id=backtick_id,
        fence_id=fence_id,
        initial_reasoning=initial_reasoning,
        trailing_backtick_ids=trailing_backtick_ids,
    )


def step_banned_ids(
    output_token_ids: Sequence[int],
    *,
    im_end_id: int | None,
    think_start_id: int | None,
    think_end_id: int | None,
    tool_start_id: int | None,
    tool_end_id: int | None,
    initial_reasoning: bool = True,
    backtick_id: int | None = None,
    fence_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
) -> list[int]:
    """Token ids banned for the next decode step.

    Two independent rules compose:
    - reasoning phase: ``im_end`` banned while inside the think block;
    - dangling backtick: the whole structural family (``im_end`` + think
      and tool tags) banned for exactly one step after an opening
      `` ` ``, so a cited special token must be spelled as plain text
      instead of emitted as the real id (which closes reasoning or ends
      the message mid-citation).
    """
    banned: list[int] = []
    if im_end_id is not None and is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
    ):
        banned.append(im_end_id)
    if ends_with_dangling_backtick(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        backtick_id=backtick_id,
        fence_id=fence_id,
        initial_reasoning=initial_reasoning,
        trailing_backtick_ids=trailing_backtick_ids,
    ):
        for tid in (
            im_end_id,
            think_start_id,
            think_end_id,
            tool_start_id,
            tool_end_id,
        ):
            if tid is not None and tid not in banned:
                banned.append(tid)
    return banned


def phase_banned_token_ids(
    output_token_ids: Sequence[int],
    vocab: Mapping[str, int],
    *,
    think_start: str = _THINK_START,
    think_end: str = _THINK_END,
    tool_start: str = _TOOL_START,
    tool_end: str = _TOOL_END,
    initial_reasoning: bool = True,
    ban_endoftext: bool = True,
) -> list[int]:
    """Token ids that must receive logit ``-inf`` for the next step."""
    banned: list[int] = []

    if ban_endoftext:
        eot = resolve_endoftext_token_id(vocab)
        if eot is not None:
            banned.append(eot)

    banned.extend(
        step_banned_ids(
            output_token_ids,
            im_end_id=resolve_im_end_token_id(vocab),
            think_start_id=resolve_vocab_token_id(vocab, think_start),
            think_end_id=resolve_vocab_token_id(vocab, think_end),
            tool_start_id=resolve_vocab_token_id(vocab, tool_start),
            tool_end_id=resolve_vocab_token_id(vocab, tool_end),
            initial_reasoning=initial_reasoning,
            backtick_id=resolve_vocab_token_id(vocab, _INLINE_BACKTICK),
            fence_id=resolve_vocab_token_id(vocab, _FENCE_BACKTICK),
            trailing_backtick_ids=frozenset(trailing_backtick_token_ids(vocab)),
        )
    )
    return banned


def apply_phase_token_bans(
    logits,
    req_index: int,
    banned_ids: Sequence[int],
) -> None:
    """In-place ``logits[req_index, id] = -inf`` for each banned id."""
    if not banned_ids:
        return
    for tid in banned_ids:
        if 0 <= tid < logits.shape[-1]:
            logits[req_index, tid] = float("-inf")


def banned_ids_for_request(
    output_token_ids: Sequence[int],
    extra_args: Mapping[str, Any] | None,
) -> list[int]:
    """Return token ids to ban for one request given its phase-ban config."""
    cfg = parse_phase_ban_config(extra_args)
    if cfg is None:
        return []
    (
        im_end_id,
        think_start,
        think_end,
        tool_start,
        tool_end,
        initial,
        backtick_id,
        fence_id,
        trailing_backtick_ids,
    ) = cfg
    return step_banned_ids(
        output_token_ids,
        im_end_id=im_end_id,
        think_start_id=think_start,
        think_end_id=think_end,
        tool_start_id=tool_start,
        tool_end_id=tool_end,
        initial_reasoning=initial,
        backtick_id=backtick_id,
        fence_id=fence_id,
        trailing_backtick_ids=trailing_backtick_ids,
    )


def should_ignore_stop_token(
    token_id: int,
    output_token_ids_before: Sequence[int],
    extra_args: Mapping[str, Any] | None,
) -> bool:
    """Return True if *token_id* is a mid-think ``im_end`` stop.

    Only the reasoning phase ignores a sampled ``im_end``. The dangling
    backtick rule is deliberately excluded: once ``im_end`` is sampled we
    cannot rewrite history, and ignoring it makes generation run past the
    intended end (model writes a new tail, tries to stop again, gets
    ignored again — visible as rewritten endings in the client).
    """
    cfg = parse_phase_ban_config(extra_args)
    if cfg is None:
        return False
    (
        im_end_id,
        think_start,
        think_end,
        tool_start,
        tool_end,
        initial,
        _backtick_id,
        _fence_id,
        _trailing_backtick_ids,
    ) = cfg
    if token_id != im_end_id:
        return False
    return is_in_reasoning_or_tool_phase(
        output_token_ids_before,
        think_start_id=think_start,
        think_end_id=think_end,
        tool_start_id=tool_start,
        tool_end_id=tool_end,
        initial_reasoning=initial,
    )
