# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for min_p under speculative decoding (MTP)."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate


def test_validate_spec_decode_allows_min_p():
    params = SamplingParams(min_p=0.05, temperature=0.9)
    params._validate_spec_decode(speculative_config=object())


def test_validate_spec_decode_rejects_logit_bias():
    params = SamplingParams(logit_bias={1: -100.0})
    with pytest.raises(VLLMValidationError, match="logit_bias"):
        params._validate_spec_decode(speculative_config=object())


def test_min_p_apply_with_spec_decode_masks_low_prob_tokens():
    device = torch.device("cpu")
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4
    proc = MinPLogitsProcessor(vllm_config, device, is_pin_memory=False)

    params = SamplingParams(min_p=0.5, temperature=1.0)
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, None, [])],
            moved=[],
        )
    )

    vocab = 8
    # Two draft/verification rows for one request; token 0 dominates.
    logits = torch.full((2, vocab), -10.0)
    logits[:, 0] = 10.0

    out = proc.apply_with_spec_decode(logits.clone(), num_draft_tokens=[2])
    assert not torch.isneginf(out[:, 0]).any()
    assert torch.isneginf(out[:, 1:]).all()


def test_min_p_apply_with_spec_decode_noop_when_disabled():
    device = torch.device("cpu")
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4
    proc = MinPLogitsProcessor(vllm_config, device, is_pin_memory=False)

    params = SamplingParams(min_p=0.0, temperature=1.0)
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, None, [])],
            moved=[],
        )
    )

    logits = torch.randn(3, 16)
    out = proc.apply_with_spec_decode(logits.clone(), num_draft_tokens=[3])
    assert torch.equal(out, logits)
