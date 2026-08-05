# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 special-token stop / ban helpers (MTP-safe).

Product invariants:
1. ``<|endoftext|>`` is always banned for chat/tool serving (turn ends via
   ``im_end`` / max_tokens / tool end).
2. ``<|im_end|>`` is phase-aware: banned while the accepted output is still
   in REASONING or an open TOOL region; allowed in CONTENT outside tools.

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


def resolve_vocab_token_id(vocab: Mapping[str, int], token: str) -> int | None:
    """Return token id from *vocab*, or ``None`` if absent."""
    tid = vocab.get(token)
    return int(tid) if tid is not None else None


def resolve_endoftext_token_id(vocab: Mapping[str, int]) -> int | None:
    return resolve_vocab_token_id(vocab, QWEN_END_OF_TEXT)


def resolve_im_end_token_id(vocab: Mapping[str, int]) -> int | None:
    return resolve_vocab_token_id(vocab, QWEN_IM_END)


# Flattened SamplingParams.extra_args / vllm_xargs key.
# Values: [im_end_id, think_start_id, think_end_id, tool_start_id,
#          tool_end_id, initial_reasoning (0/1)] — missing ids are -1.
PHASE_BAN_XARG_KEY = "qwen3_phase_ban"


def apply_endoftext_ban_to_request(
    request: Any,
    vocab: Mapping[str, int],
    *,
    think_start: str = _THINK_START,
    think_end: str = _THINK_END,
    tool_start: str = _TOOL_START,
    tool_end: str = _TOOL_END,
    initial_reasoning: bool = True,
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
    ]
    xargs = getattr(request, "vllm_xargs", None)
    if xargs is None:
        if hasattr(request, "vllm_xargs"):
            request.vllm_xargs = {PHASE_BAN_XARG_KEY: phase_ban}
    else:
        xargs[PHASE_BAN_XARG_KEY] = phase_ban


def parse_phase_ban_config(
    extra_args: Mapping[str, Any] | None,
) -> tuple[int, int | None, int | None, int | None, int | None, bool] | None:
    """Parse flattened ``qwen3_phase_ban`` from SamplingParams.extra_args.

    Returns ``(im_end_id, think_start, think_end, tool_start, tool_end,
    initial_reasoning)`` or ``None`` if unset/invalid.
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

    return (
        im_end_id,
        _opt(raw[1]),
        _opt(raw[2]),
        _opt(raw[3]),
        _opt(raw[4]),
        bool(int(raw[5])),
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

    Returns ``True`` when ``im_end`` must be banned (still inside think, or
    inside an unclosed ``<tool_call>`` region).
    """
    in_reasoning = initial_reasoning
    tool_depth = 0

    for tid in output_token_ids:
        if think_start_id is not None and tid == think_start_id:
            in_reasoning = True
            continue
        if think_end_id is not None and tid == think_end_id:
            in_reasoning = False
            continue
        if tool_start_id is not None and tid == tool_start_id:
            # Tool calls only open in the answer phase; still track depth
            # so nested/accidental tags inside params don't end the ban
            # early when the outer tool_call is unclosed.
            if not in_reasoning:
                tool_depth += 1
            continue
        if tool_end_id is not None and tid == tool_end_id:
            if tool_depth > 0:
                tool_depth -= 1
            continue

    return in_reasoning or tool_depth > 0


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

    im_end = resolve_im_end_token_id(vocab)
    if im_end is not None and is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=resolve_vocab_token_id(vocab, think_start),
        think_end_id=resolve_vocab_token_id(vocab, think_end),
        tool_start_id=resolve_vocab_token_id(vocab, tool_start),
        tool_end_id=resolve_vocab_token_id(vocab, tool_end),
        initial_reasoning=initial_reasoning,
    ):
        banned.append(im_end)

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
    im_end_id, think_start, think_end, tool_start, tool_end, initial = cfg
    if is_in_reasoning_or_tool_phase(
        output_token_ids,
        think_start_id=think_start,
        think_end_id=think_end,
        tool_start_id=tool_start,
        tool_end_id=tool_end,
        initial_reasoning=initial,
    ):
        return [im_end_id]
    return []


def should_ignore_stop_token(
    token_id: int,
    output_token_ids_before: Sequence[int],
    extra_args: Mapping[str, Any] | None,
) -> bool:
    """Return True if *token_id* is a phase-banned stop (e.g. mid-think im_end)."""
    cfg = parse_phase_ban_config(extra_args)
    if cfg is None:
        return False
    im_end_id, think_start, think_end, tool_start, tool_end, initial = cfg
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
