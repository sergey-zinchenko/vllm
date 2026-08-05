# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3 phase-aware special-token bans."""

from unittest.mock import MagicMock

from vllm.parser.qwen3_phase_stop import (
    PHASE_BAN_XARG_KEY,
    QWEN_END_OF_TEXT,
    QWEN_IM_END,
    apply_endoftext_ban_to_request,
    banned_ids_for_request,
    is_in_reasoning_or_tool_phase,
    parse_phase_ban_config,
    should_ignore_stop_token,
)

_THINK_START = 10
_THINK_END = 11
_TOOL_START = 20
_TOOL_END = 21
_IM_END = 30
_EOT = 31


_VOCAB = {
    "<think>": _THINK_START,
    "</think>": _THINK_END,
    "<tool_call>": _TOOL_START,
    "</tool_call>": _TOOL_END,
    QWEN_IM_END: _IM_END,
    QWEN_END_OF_TEXT: _EOT,
}


class TestPhaseDetection:
    def test_initial_reasoning_bans_im_end(self):
        assert is_in_reasoning_or_tool_phase(
            [1, 2, 3],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
        )

    def test_after_think_end_not_banned(self):
        assert not is_in_reasoning_or_tool_phase(
            [_THINK_START, 1, _THINK_END, 2],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
        )

    def test_open_tool_region_bans_im_end(self):
        assert is_in_reasoning_or_tool_phase(
            [_THINK_END, _TOOL_START, 5],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=False,
        )

    def test_closed_tool_region_allows_im_end(self):
        assert not is_in_reasoning_or_tool_phase(
            [_THINK_END, _TOOL_START, 5, _TOOL_END],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=False,
        )

    def test_tool_markup_inside_think_does_not_open_tool_depth(self):
        # Tool tags inside reasoning must not toggle tool_depth.
        assert is_in_reasoning_or_tool_phase(
            [_THINK_START, _TOOL_START, 5, _TOOL_END],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
        )


class TestRequestWiring:
    def test_apply_endoftext_ban_sets_bad_words_not_logit_bias(self):
        """logit_bias is forbidden with MTP — ban via bad_words only."""
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=True)
        assert req.logit_bias is None
        assert QWEN_END_OF_TEXT in req.bad_words
        assert PHASE_BAN_XARG_KEY in req.vllm_xargs
        cfg = parse_phase_ban_config(req.vllm_xargs)
        assert cfg is not None
        assert cfg[0] == _IM_END

    def test_endoftext_ban_compatible_with_speculative_verify(self):
        """Regression: adjust_request must not trip MTP logit_bias reject."""
        from unittest.mock import MagicMock

        from vllm.sampling_params import SamplingParams

        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=False)

        params = SamplingParams(
            max_tokens=16,
            bad_words=list(req.bad_words),
            logit_bias=req.logit_bias,
            extra_args=dict(req.vllm_xargs) if req.vllm_xargs else None,
        )
        # Must not raise (stock image + MTP works; our old logit_bias broke it).
        params._validate_spec_decode(speculative_config=object())

    def test_banned_ids_for_request_in_reasoning(self):
        extra = {
            PHASE_BAN_XARG_KEY: [
                _IM_END,
                _THINK_START,
                _THINK_END,
                _TOOL_START,
                _TOOL_END,
                1,
            ]
        }
        assert banned_ids_for_request([1, 2], extra) == [_IM_END]
        assert banned_ids_for_request([_THINK_END, 3], extra) == []

    def test_should_ignore_stop_token(self):
        extra = {
            PHASE_BAN_XARG_KEY: [
                _IM_END,
                _THINK_START,
                _THINK_END,
                _TOOL_START,
                _TOOL_END,
                1,
            ]
        }
        assert should_ignore_stop_token(_IM_END, [1, 2], extra)
        assert not should_ignore_stop_token(_IM_END, [_THINK_END], extra)
        assert not should_ignore_stop_token(999, [1, 2], extra)
