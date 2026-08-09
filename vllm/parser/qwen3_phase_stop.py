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
# Prose-spelled agent delimiters (BPE, not special ids) — ban + parser strip.
QWEN_MASK_START = "<|mask_start|>"
QWEN_MASK_END = "<|mask_end|>"
QWEN_MASK_PAD = "<|mask_pad|>"
QWEN_MASK_BAD_WORDS = (QWEN_MASK_START, QWEN_MASK_END, QWEN_MASK_PAD)

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
#          backtick_id?, fence_id?, close_paren_id?, open_paren_id?,
#          *trailing_backtick_ids] — missing ids are -1. Trailing fields
# are optional for backward-compatible parse; indices 10+ are merged BPE
# tokens whose text ends with one backtick. Legacy configs without paren
# slots still parse (paren None, trailing from index 8).
PHASE_BAN_XARG_KEY = "qwen3_phase_ban"
# Separate from phase_ban packing:
# [lt_id, lt_slash_id_or_-1, bang_id_or_-1, newline_id_or_-1?].
CITATION_NUDGE_XARG_KEY = "qwen3_citation_nudge"

_CLOSE_PAREN = ")"
_OPEN_PAREN = "("
_LT = "<"
_LT_SLASH = "</"
_BANG = "!"
_NEWLINE = "\n"
# HF Qwen BPE often stores newline as ``Ċ`` (U+010A), not ``"\n"``.
_NEWLINE_VOCAB_KEYS = ("\n", "Ċ")

# Soft logits nudge after a dangling opening `` ` `` (text-path citations).
CITATION_LT_BOOST = 4.0
# Soft boost for ``im_end`` once a bang/newline streak is detected after
# think has closed (break ``!\n!\n`` attractors without force-argmax).
BANG_STREAK_IM_END_BOOST = 8.0
BANG_STREAK_MIN_LEN = 4
BANG_STREAK_MIN_BANGS = 2
# Max filler tokens after `` ` `` that still count as a failed citation
# (prod chatcmpl-8822c896: `` `\n!`` then im_end).
FAILED_CITATION_MAX_TAIL = 16
# Think-tag citation loop (chatcmpl-8521cc): `` `</think>` / `</thinking>` / …``
# repeated inside reasoning. citation_nudge's ``<`` boost feeds the cycle;
# break by banning ``<`` / ``</`` and soft-boosting ``think_end``.
THINK_TAG_LOOP_WINDOW = 256
THINK_TAG_LOOP_MIN_CYCLES = 3
THINK_TAG_CITATION_MAX_INNER = 8
THINK_TAG_LOOP_TE_BOOST = 8.0

PhaseBanConfig = tuple[
    int,
    int | None,
    int | None,
    int | None,
    int | None,
    bool,
    int | None,
    int | None,
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


def resolve_newline_token_id(vocab: Mapping[str, int]) -> int | None:
    """Id for a single newline token (``\\n`` or HF ``Ċ`` spelling)."""
    for key in _NEWLINE_VOCAB_KEYS:
        tid = resolve_vocab_token_id(vocab, key)
        if tid is not None:
            return tid
    for tok, tid in vocab.items():
        if tok.replace("Ċ", "\n") == "\n":
            return int(tid)
    return None


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
    """Ban stop/mask prose markers and enable phase-aware ``im_end`` bans.

    Do **not** set ``logit_bias``: ``SamplingParams._validate_spec_decode``
    rejects it when MTP / speculative decoding is enabled (the qwen36-27b
    production path). Use ``bad_words`` instead — that path works with
    drafts on both sampler implementations.

    ``vllm_xargs[qwen3_phase_ban]`` carries flattened ids for the builtin
    ``Qwen3PhaseStopLogitsProcessor`` / ``check_stop`` im_end masking.
    Also sets ``qwen3_citation_nudge`` for text-path tag citations after
    a dangling backtick. ``bad_words`` includes ``<|endoftext|>`` and
    prose ``<|mask_*|>`` delimiters (model spells them in BPE).
    """
    ban_words = (QWEN_END_OF_TEXT, *QWEN_MASK_BAD_WORDS)
    bad_words = getattr(request, "bad_words", None)
    if bad_words is None:
        # ResponsesRequest has no bad_words field — skip quietly.
        if hasattr(request, "bad_words"):
            request.bad_words = list(ban_words)
    else:
        for word in ban_words:
            if word not in bad_words:
                bad_words.append(word)

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
        _id(_CLOSE_PAREN),
        _id(_OPEN_PAREN),
        *trailing_backtick_token_ids(vocab, cache_key=vocab_cache_key),
    ]
    newline_tid = resolve_newline_token_id(vocab)
    citation_nudge = [
        _id(_LT),
        _id(_LT_SLASH),
        _id(_BANG),
        newline_tid if newline_tid is not None else -1,
    ]
    xargs = getattr(request, "vllm_xargs", None)
    if xargs is None:
        if hasattr(request, "vllm_xargs"):
            request.vllm_xargs = {
                PHASE_BAN_XARG_KEY: phase_ban,
                CITATION_NUDGE_XARG_KEY: citation_nudge,
            }
    else:
        xargs[PHASE_BAN_XARG_KEY] = phase_ban
        xargs[CITATION_NUDGE_XARG_KEY] = citation_nudge


def parse_phase_ban_config(
    extra_args: Mapping[str, Any] | None,
) -> PhaseBanConfig | None:
    """Parse flattened ``qwen3_phase_ban`` from SamplingParams.extra_args.

    Returns ``(im_end_id, think_start, think_end, tool_start, tool_end,
    initial_reasoning, backtick_id, fence_id, close_paren_id, open_paren_id,
    trailing_backtick_ids)`` or ``None`` if unset/invalid. Fields past
    index 5 are optional; legacy shorter configs parse with ``None`` /
    empty tail. Layout with paren slots requires ``len >= 10``; otherwise
    trailing starts at index 8 (pre-paren configs).
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
    if len(raw) >= 10:
        close_paren_id = _opt(raw[8])
        open_paren_id = _opt(raw[9])
        trailing = frozenset(int(v) for v in raw[10:] if int(v) >= 0)
    else:
        close_paren_id = None
        open_paren_id = None
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
        close_paren_id,
        open_paren_id,
        trailing,
    )


def is_citation_think_end_at(
    output_token_ids: Sequence[int],
    index: int,
    *,
    think_end_id: int | None,
    backtick_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    newline_id: int | None = None,
) -> bool:
    """True when TE at *index* is a `` `</think>`` / `` `\\n</think>`` citation.

    Production chatcmpl-a0f7aa7: ``[backtick, nl, TE]`` was latched as a real
    think close → further TE banned while the parser stayed in REASONING.
    """
    if (
        think_end_id is None
        or index < 0
        or index >= len(output_token_ids)
        or output_token_ids[index] != think_end_id
    ):
        return False

    def _is_backtick(tid: int) -> bool:
        return tid == backtick_id or tid in trailing_backtick_ids

    if index >= 1 and _is_backtick(output_token_ids[index - 1]):
        return True
    return (
        index >= 2
        and newline_id is not None
        and output_token_ids[index - 1] == newline_id
        and _is_backtick(output_token_ids[index - 2])
    )


def is_citation_think_end_sequence_at(
    output_token_ids: Sequence[int],
    start: int,
    end_token_ids: Sequence[int],
    *,
    backtick_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    newline_id: int | None = None,
) -> bool:
    """Citation check for a multi-token think-end sequence starting at *start*."""
    if not end_token_ids or start < 0:
        return False
    end_index = start + len(end_token_ids) - 1
    if end_index >= len(output_token_ids):
        return False
    if list(output_token_ids[start : start + len(end_token_ids)]) != list(
        end_token_ids
    ):
        return False
    # Use the first TE token index for the backtick/nl lookbehind.
    return is_citation_think_end_at(
        output_token_ids,
        start,
        think_end_id=int(end_token_ids[0]),
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
    )


def has_real_think_end(
    output_token_ids: Sequence[int],
    *,
    think_end_id: int | None,
    backtick_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    newline_id: int | None = None,
) -> bool:
    """True when a non-citation ``think_end`` appears in *output_token_ids*."""
    if think_end_id is None:
        return False
    for i, tid in enumerate(output_token_ids):
        if tid != think_end_id:
            continue
        if not is_citation_think_end_at(
            output_token_ids,
            i,
            think_end_id=think_end_id,
            backtick_id=backtick_id,
            trailing_backtick_ids=trailing_backtick_ids,
            newline_id=newline_id,
        ):
            return True
    return False


def is_in_reasoning_or_tool_phase(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    tool_start_id: int | None,
    tool_end_id: int | None,
    initial_reasoning: bool = True,
    backtick_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    newline_id: int | None = None,
) -> bool:
    """Heuristic phase detection from accepted output token ids.

    Returns ``True`` when ``im_end`` must be banned (still inside think).

    One-way latch: after the first **real** ``think_end`` the phase never
    re-enters reasoning. Citation-shaped `` `</think>`` / `` `\\n</think>``
    are ignored so a sticky parser abort cannot latch the sampler closed
    (chatcmpl-a0f7aa7 runaway reasoning).

    ``tool_start_id`` / ``tool_end_id`` are accepted for config compatibility
    but do not affect the ban: unpaired ``<tool_call>`` in prose must not
    block stop tokens.
    """
    del tool_start_id, tool_end_id
    in_reasoning = initial_reasoning
    think_closed = False

    for i, tid in enumerate(output_token_ids):
        if think_start_id is not None and tid == think_start_id:
            if not think_closed:
                in_reasoning = True
            continue
        if think_end_id is not None and tid == think_end_id:
            if is_citation_think_end_at(
                output_token_ids,
                i,
                think_end_id=think_end_id,
                backtick_id=backtick_id,
                trailing_backtick_ids=trailing_backtick_ids,
                newline_id=newline_id,
            ):
                continue
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
    """True when an opening `` ` `` is in the last two tokens — any phase.

    Matches the lone `` ` `` id and merged BPE forms (``" `"``, ``"(`"``
    ...) whose text ends with exactly one backtick — prose openings almost
    never tokenize as the bare backtick, and the truncation happens right
    after those merged forms. Applies inside reasoning too: the ban covers
    ``im_end`` and all structural specials (think/tool); citations must use
    text/BPE, with a soft ``<`` / ``</`` boost and a hard ``!`` ban (see
    ``step_citation_logit_deltas`` / ``step_banned_ids``).

    Deliberately **not** span parity: counting one vocab id sees only one
    side of real inline spans, sticks odd, and bans ``im_end`` for the
    rest of the request (endless ``!!!`` tails, rewritten endings under
    MTP). A one-token window let `` `\n</think>`` clear the guard (newline
    or MTP first draft); the two-token window still blocks that path
    without sticking for the whole request.
    """
    del think_start_id, think_end_id, fence_id, initial_reasoning
    if not output_token_ids:
        return False

    def _is_backtick(tid: int) -> bool:
        return tid == backtick_id or tid in trailing_backtick_ids

    if _is_backtick(output_token_ids[-1]):
        return True
    return len(output_token_ids) >= 2 and _is_backtick(output_token_ids[-2])


def ends_with_bang_newline_streak(
    output_token_ids: Sequence[int],
    *,
    bang_id: int | None,
    newline_id: int | None,
    min_len: int = BANG_STREAK_MIN_LEN,
    min_bangs: int = BANG_STREAK_MIN_BANGS,
) -> bool:
    """True when the suffix is a ``!`` / ``\\n`` attractor (prod ``!\n!\n``).

    Requires at least *min_len* trailing tokens from ``{bang, newline}`` and
    at least *min_bangs* bangs so ordinary ``!`` / ``!\\n`` punctuation is
    untouched.
    """
    if bang_id is None or not output_token_ids:
        return False
    allowed = {bang_id}
    if newline_id is not None:
        allowed.add(newline_id)
    n = 0
    bangs = 0
    for tid in reversed(output_token_ids):
        if tid not in allowed:
            break
        n += 1
        if tid == bang_id:
            bangs += 1
    return n >= min_len and bangs >= min_bangs


def ends_with_think_tag_citation_loop(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    tool_start_id: int | None,
    tool_end_id: int | None,
    initial_reasoning: bool = True,
    backtick_id: int | None = None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    newline_id: int | None = None,
    lt_id: int | None = None,
    lt_slash_id: int | None = None,
    window: int = THINK_TAG_LOOP_WINDOW,
    min_cycles: int = THINK_TAG_LOOP_MIN_CYCLES,
    max_inner: int = THINK_TAG_CITATION_MAX_INNER,
) -> bool:
    """True when reasoning is stuck citing think-close tags in backticks.

    Counts closed `` `…` `` spans in the recent window whose interior
    contains ``<`` / ``</``. A single legitimate citation is fine; ≥
    *min_cycles* such spans (prod chatcmpl-8521cc) is the attractor that
    burns ``thinking_token_budget`` while citation_nudge keeps boosting
    ``<``.
    """
    if (lt_id is None and lt_slash_id is None) or not output_token_ids:
        return False
    if not is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
    ):
        return False

    def _is_backtick(tid: int) -> bool:
        return tid == backtick_id or tid in trailing_backtick_ids

    lt_ids = {tid for tid in (lt_id, lt_slash_id) if tid is not None}
    seq = (
        output_token_ids[-window:]
        if len(output_token_ids) > window
        else output_token_ids
    )
    bt_positions = [i for i, tid in enumerate(seq) if _is_backtick(tid)]
    cycles = 0
    for start, end in zip(bt_positions, bt_positions[1:]):
        inner = seq[start + 1 : end]
        if not inner or len(inner) > max_inner:
            continue
        if any(tid in lt_ids for tid in inner):
            cycles += 1
            if cycles >= min_cycles:
                return True
    return False


def ends_with_failed_citation_tail(
    output_token_ids: Sequence[int],
    *,
    backtick_id: int | None,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    bang_id: int | None = None,
    newline_id: int | None = None,
    max_tail: int = FAILED_CITATION_MAX_TAIL,
) -> bool:
    """True for `` `\\n!`` — open citation followed only by nl/bang.

    The two-token dangling window clears once ``!`` is accepted, which
    re-allowed ``im_end`` and cut the answer (chatcmpl-8822c896). Keep the
    citation guard until a non-filler token continues the sentence. Cap the
    filler suffix so an ancient backtick cannot stick the ban forever.
    """
    if not output_token_ids:
        return False
    filler: set[int] = set()
    if bang_id is not None:
        filler.add(bang_id)
    if newline_id is not None:
        filler.add(newline_id)
    if not filler:
        return False

    def _is_backtick(tid: int) -> bool:
        return tid == backtick_id or tid in trailing_backtick_ids

    last_bt = -1
    for i, tid in enumerate(output_token_ids):
        if _is_backtick(tid):
            last_bt = i
    if last_bt < 0:
        return False
    suffix = output_token_ids[last_bt + 1 :]
    # Empty suffix is plain dangling (handled separately).
    if not suffix or len(suffix) > max_tail:
        return False
    return all(tid in filler for tid in suffix)


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
    bang_id: int | None = None,
    newline_id: int | None = None,
) -> bool:
    """True when ``im_end`` must be banned for the next decode step."""
    if is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
    ):
        return True
    if ends_with_dangling_backtick(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        backtick_id=backtick_id,
        fence_id=fence_id,
        initial_reasoning=initial_reasoning,
        trailing_backtick_ids=trailing_backtick_ids,
    ):
        return True
    return ends_with_failed_citation_tail(
        output_token_ids,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        bang_id=bang_id,
        newline_id=newline_id,
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
    close_paren_id: int | None = None,
    open_paren_id: int | None = None,
    bang_id: int | None = None,
    newline_id: int | None = None,
    lt_id: int | None = None,
    lt_slash_id: int | None = None,
) -> list[int]:
    """Token ids banned for the next decode step.

    Rules compose:
    - reasoning phase: ``im_end`` banned while inside the think block;
    - after think latch (not in reasoning): ``think_end`` banned forever
      so mid-answer citations / MTP cannot sample another special close;
    - dangling backtick: ``im_end``, all structural specials
      (``think_start`` / ``think_end`` / ``tool_start`` / ``tool_end``),
      and ``!`` banned for a two-token window — citations must use
      text/BPE (covers `` `\n</think>`` / ``Treat `\n!``);
    - failed citation tail (`` `\n!``): same bans until a non-nl/bang
      token continues (otherwise ``im_end`` cuts the answer);
    - bang/newline streak: ``!`` hard-banned (does not touch ``think_end``)
      so ``!\n!\n`` attractors cannot run to ``max_tokens``;
    - think-tag citation loop: ``<`` / ``</`` hard-banned and ``think_end``
      kept samplable (even under dangling citation guard) so the cycle
      can exit early (chatcmpl-8521cc);
    - empty start (no output yet, initial reasoning): ``think_end`` and
      bare ``)`` / ``(`` banned so polluted history cannot open with a
      lone paren or instantly close think.
    """
    banned: list[int] = []
    in_reasoning = is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
    )
    dangling = ends_with_dangling_backtick(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        backtick_id=backtick_id,
        fence_id=fence_id,
        initial_reasoning=initial_reasoning,
        trailing_backtick_ids=trailing_backtick_ids,
    )
    citation_tail = ends_with_failed_citation_tail(
        output_token_ids,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        bang_id=bang_id,
        newline_id=newline_id,
    )
    citation_guard = dangling or citation_tail
    streak = ends_with_bang_newline_streak(
        output_token_ids,
        bang_id=bang_id,
        newline_id=newline_id,
    )
    tag_loop = ends_with_think_tag_citation_loop(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
        lt_id=lt_id,
        lt_slash_id=lt_slash_id,
    )
    if im_end_id is not None and (in_reasoning or citation_guard):
        banned.append(im_end_id)
    if citation_guard:
        for tid in (
            think_start_id,
            # Keep TE samplable under a think-tag citation loop so the
            # breaker can exit; otherwise dangling `` ` `` bans TE forever.
            None if tag_loop else think_end_id,
            tool_start_id,
            tool_end_id,
            bang_id,
        ):
            if tid is not None and tid not in banned:
                banned.append(tid)
    if tag_loop:
        for tid in (lt_id, lt_slash_id):
            if tid is not None and tid not in banned:
                banned.append(tid)
    if streak and bang_id is not None and bang_id not in banned:
        banned.append(bang_id)
    # One-way latch: after the first *real* think close, never sample
    # think_end again (chatcmpl-8b9c: ``Treat `\n</think>`` mid-answer).
    # Citation-shaped TE must not latch (chatcmpl-a0f7aa7 split-brain).
    if (
        not in_reasoning
        and think_end_id is not None
        and think_end_id not in banned
        and has_real_think_end(
            output_token_ids,
            think_end_id=think_end_id,
            backtick_id=backtick_id,
            trailing_backtick_ids=trailing_backtick_ids,
            newline_id=newline_id,
        )
    ):
        banned.append(think_end_id)
    if not output_token_ids and initial_reasoning:
        if think_end_id is not None and think_end_id not in banned:
            banned.append(think_end_id)
        for tid in (close_paren_id, open_paren_id):
            if tid is not None and tid not in banned:
                banned.append(tid)
    return banned


def step_citation_logit_deltas(
    output_token_ids: Sequence[int],
    *,
    think_start_id: int | None,
    think_end_id: int | None,
    backtick_id: int | None,
    fence_id: int | None = None,
    initial_reasoning: bool = True,
    trailing_backtick_ids: frozenset[int] = frozenset(),
    lt_id: int | None = None,
    lt_slash_id: int | None = None,
    bang_id: int | None = None,
    newline_id: int | None = None,
    im_end_id: int | None = None,
    tool_start_id: int | None = None,
    tool_end_id: int | None = None,
    lt_boost: float = CITATION_LT_BOOST,
    im_end_boost: float = BANG_STREAK_IM_END_BOOST,
    think_end_boost: float = THINK_TAG_LOOP_TE_BOOST,
) -> list[tuple[int, float]]:
    """Finite logit deltas for citation text-path and bang-loop breakout.

    - Dangling / failed-citation tail: boost ``<`` / ``</`` so tags spell
      in BPE. ``!`` is hard-banned separately.
    - Think-tag citation loop: do **not** boost ``<``; soft-boost
      ``think_end`` so reasoning can exit early (chatcmpl-8521cc).
    - Bang/newline streak after think close: soft-boost ``im_end``, unless
      still inside a failed citation tail (do not push stop mid-`` `\n!``).
    """
    deltas: list[tuple[int, float]] = []
    tag_loop = ends_with_think_tag_citation_loop(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        initial_reasoning=initial_reasoning,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
        lt_id=lt_id,
        lt_slash_id=lt_slash_id,
    )
    citation_guard = ends_with_dangling_backtick(
        output_token_ids,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
        backtick_id=backtick_id,
        fence_id=fence_id,
        initial_reasoning=initial_reasoning,
        trailing_backtick_ids=trailing_backtick_ids,
    ) or ends_with_failed_citation_tail(
        output_token_ids,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        bang_id=bang_id,
        newline_id=newline_id,
    )
    if tag_loop:
        if think_end_id is not None:
            deltas.append((think_end_id, think_end_boost))
    elif citation_guard:
        if lt_id is not None:
            deltas.append((lt_id, lt_boost))
        if lt_slash_id is not None and lt_slash_id != lt_id:
            deltas.append((lt_slash_id, lt_boost))
    if (
        im_end_id is not None
        and ends_with_bang_newline_streak(
            output_token_ids,
            bang_id=bang_id,
            newline_id=newline_id,
        )
        and not ends_with_failed_citation_tail(
            output_token_ids,
            backtick_id=backtick_id,
            trailing_backtick_ids=trailing_backtick_ids,
            bang_id=bang_id,
            newline_id=newline_id,
        )
        and not is_in_reasoning_or_tool_phase(
            output_token_ids,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
            tool_start_id=tool_start_id,
            tool_end_id=tool_end_id,
            initial_reasoning=initial_reasoning,
            backtick_id=backtick_id,
            trailing_backtick_ids=trailing_backtick_ids,
            newline_id=newline_id,
        )
    ):
        deltas.append((im_end_id, im_end_boost))
    return deltas


def parse_citation_nudge_config(
    extra_args: Mapping[str, Any] | None,
) -> tuple[int | None, int | None, int | None, int | None] | None:
    """Parse ``qwen3_citation_nudge``.

    Returns ``(lt_id, lt_slash_id, bang_id, newline_id)``. ``newline_id`` is
    optional (legacy 3-field configs → ``None``).
    """
    if not extra_args:
        return None
    raw = extra_args.get(CITATION_NUDGE_XARG_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None

    def _opt(v: Any) -> int | None:
        i = int(v)
        return i if i >= 0 else None

    newline_id = _opt(raw[3]) if len(raw) > 3 else None
    return _opt(raw[0]), _opt(raw[1]), _opt(raw[2]), newline_id


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
            close_paren_id=resolve_vocab_token_id(vocab, _CLOSE_PAREN),
            open_paren_id=resolve_vocab_token_id(vocab, _OPEN_PAREN),
            bang_id=resolve_vocab_token_id(vocab, _BANG),
            newline_id=resolve_newline_token_id(vocab),
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
        close_paren_id,
        open_paren_id,
        trailing_backtick_ids,
    ) = cfg
    nudge = parse_citation_nudge_config(extra_args)
    lt_id = lt_slash_id = bang_id = newline_id = None
    if nudge is not None:
        lt_id, lt_slash_id, bang_id, newline_id = nudge
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
        close_paren_id=close_paren_id,
        open_paren_id=open_paren_id,
        bang_id=bang_id,
        newline_id=newline_id,
        lt_id=lt_id,
        lt_slash_id=lt_slash_id,
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
        backtick_id,
        _fence_id,
        _close_paren_id,
        _open_paren_id,
        trailing_backtick_ids,
    ) = cfg
    if token_id != im_end_id:
        return False
    newline_id = None
    nudge = parse_citation_nudge_config(extra_args)
    if nudge is not None:
        newline_id = nudge[3]
    return is_in_reasoning_or_tool_phase(
        output_token_ids_before,
        think_start_id=think_start,
        think_end_id=think_end,
        tool_start_id=tool_start,
        tool_end_id=tool_end,
        initial_reasoning=initial,
        backtick_id=backtick_id,
        trailing_backtick_ids=trailing_backtick_ids,
        newline_id=newline_id,
    )
