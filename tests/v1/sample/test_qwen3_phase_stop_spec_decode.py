# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP draft-aware phase ban for citation bang loops (chatcmpl-8919ee)."""

from unittest.mock import MagicMock

import torch

from vllm.parser.qwen3_phase_stop import (
    CITATION_NUDGE_XARG_KEY,
    PHASE_BAN_XARG_KEY,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import Qwen3PhaseStopLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.rejection_sampler import RejectionSampler

_IM_END = 30
_THINK_START = 10
_THINK_END = 11
_TOOL_START = 20
_TOOL_END = 21
_BACKTICK = 40
_FENCE = 41
_SPACE_BACKTICK = 42
_BANG = 0
_NEWLINE = 198
_LT = 50
_LT_SLASH = 51
_VOCAB = 64


def _phase_ban_extra_args() -> dict:
    return {
        PHASE_BAN_XARG_KEY: [
            _IM_END,
            _THINK_START,
            _THINK_END,
            _TOOL_START,
            _TOOL_END,
            1,  # initial_reasoning
            _BACKTICK,
            _FENCE,
            -1,
            -1,
            _SPACE_BACKTICK,
        ],
        CITATION_NUDGE_XARG_KEY: [_LT, _LT_SLASH, _BANG, _NEWLINE],
    }


def _make_processor(
    output_tok_ids: list[int],
) -> Qwen3PhaseStopLogitsProcessor:
    device = torch.device("cpu")
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4
    proc = Qwen3PhaseStopLogitsProcessor(vllm_config, device, is_pin_memory=False)
    params = SamplingParams(extra_args=_phase_ban_extra_args())
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, None, output_tok_ids)],
            moved=[],
        )
    )
    assert 0 in proc.reqs
    return proc


def test_mtp_row_after_backtick_bans_bang():
    """Draft ``[backtick, bang]``: row1 must ban bang (chatcmpl-8919ee)."""
    out_ids: list[int] = [_THINK_START, 1]
    proc = _make_processor(out_ids)
    drafts = [[_SPACE_BACKTICK, _BANG]]
    logits = torch.zeros(2, _VOCAB)
    out = proc.apply_with_spec_decode(
        logits.clone(),
        num_draft_tokens=[2],
        draft_token_ids=drafts,
    )
    # Row 0: accepted has no backtick yet → bang free.
    assert not torch.isneginf(out[0, _BANG])
    # Row 1: prefix ends with backtick → bang banned.
    assert torch.isneginf(out[1, _BANG])
    assert torch.isneginf(out[1, _THINK_END])


def test_empty_spec_token_ids_uses_metadata_drafts():
    """Empty ``spec_token_ids`` placeholders must not disable draft prefixes."""
    out_ids: list[int] = [_THINK_START, 1]
    proc = _make_processor(out_ids)
    drafts = [[_SPACE_BACKTICK, _BANG]]
    logits = torch.zeros(2, _VOCAB)
    out = proc.apply_with_spec_decode(
        logits.clone(),
        num_draft_tokens=[2],
        spec_token_ids=[[]],  # placeholder hole
        draft_token_ids=drafts,
    )
    assert torch.isneginf(out[1, _BANG])


def test_empty_spec_without_metadata_drafts_is_unsafe_legacy():
    """Regression guard: empty spec alone cannot see in-step backtick."""
    out_ids: list[int] = [_THINK_START, 1]
    proc = _make_processor(out_ids)
    logits = torch.zeros(2, _VOCAB)
    out = proc.apply_with_spec_decode(
        logits.clone(),
        num_draft_tokens=[2],
        spec_token_ids=[[]],
        draft_token_ids=None,
    )
    # Both rows use accepted-only prefix → bang not banned (the hole).
    assert not torch.isneginf(out[0, _BANG])
    assert not torch.isneginf(out[1, _BANG])


def test_bonus_prefix_includes_full_drafts():
    """Bonus after draft ``[backtick, nl]`` must ban bang."""
    out_ids: list[int] = [_THINK_END, 1]
    proc = _make_processor(out_ids)
    drafts = [[_SPACE_BACKTICK, _NEWLINE]]
    logits = torch.zeros(1, _VOCAB)
    out = proc.apply_to_bonus(
        logits.clone(),
        draft_token_ids=drafts,
        spec_token_ids=[[]],
    )
    assert torch.isneginf(out[0, _BANG])
    assert torch.isneginf(out[0, _IM_END])


def test_bonus_apply_accepts_bfloat16_logits():
    """Prod crash: apply_to_bonus on raw bf16 bonus slice (dtype mismatch)."""
    out_ids: list[int] = [_THINK_END, 1]
    proc = _make_processor(out_ids)
    drafts = [[_SPACE_BACKTICK, _NEWLINE]]
    logits = torch.zeros(1, _VOCAB, dtype=torch.bfloat16)
    out = proc.apply_to_bonus(
        logits.clone(),
        draft_token_ids=drafts,
        spec_token_ids=[[]],
    )
    assert out.dtype == torch.bfloat16
    assert torch.isneginf(out[0, _BANG])


def test_split_draft_token_ids():
    flat = torch.tensor([40, 0, 198, 0], dtype=torch.int32)
    assert RejectionSampler._split_draft_token_ids(flat, [2, 2]) == [
        [40, 0],
        [198, 0],
    ]
