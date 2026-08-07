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
    ends_with_dangling_backtick,
    is_in_reasoning_or_tool_phase,
    parse_phase_ban_config,
    should_ban_im_end,
    should_ignore_stop_token,
)

_THINK_START = 10
_THINK_END = 11
_TOOL_START = 20
_TOOL_END = 21
_IM_END = 30
_EOT = 31
_BACKTICK = 40
_FENCE = 41


_VOCAB = {
    "<think>": _THINK_START,
    "</think>": _THINK_END,
    "<tool_call>": _TOOL_START,
    "</tool_call>": _TOOL_END,
    QWEN_IM_END: _IM_END,
    QWEN_END_OF_TEXT: _EOT,
    "`": _BACKTICK,
    "```": _FENCE,
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

    def test_open_tool_markup_does_not_ban_im_end(self):
        # Prose citations of <tool_call> without </tool_call> must remain
        # stoppable — otherwise decoding never ends (GPU hang / silent stream).
        assert not is_in_reasoning_or_tool_phase(
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


class TestDanglingBacktick:
    """The im_end ban must be non-sticky: only the single step right after
    a lone `` ` `` token blocks stop. Parity counting deadlocks (BPE merges
    hide one side of real spans) and caused endless ``!!!`` tails."""

    def test_last_token_backtick_bans_im_end(self):
        ids = [_THINK_END, 1, _BACKTICK]
        assert ends_with_dangling_backtick(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
            initial_reasoning=False,
        )
        assert should_ban_im_end(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=False,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
        )

    def test_unpaired_backtick_earlier_does_not_stick(self):
        # Regression: odd count deep in the answer must NOT keep im_end
        # banned once any other token followed (no parity deadlock).
        ids = [_THINK_END, 1, _BACKTICK, 2]
        assert not ends_with_dangling_backtick(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
            initial_reasoning=False,
        )
        assert not should_ban_im_end(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=False,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
        )

    def test_still_in_reasoning_bans_regardless_of_backticks(self):
        ids = [_THINK_START, _BACKTICK, _BACKTICK]
        assert should_ban_im_end(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
        )
        # Backtick at the end of think content is not an answer-phase rule.
        assert not ends_with_dangling_backtick(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
            initial_reasoning=True,
        )

    def test_fence_token_does_not_ban(self):
        ids = [_THINK_END, 1, _FENCE]
        assert not ends_with_dangling_backtick(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
            initial_reasoning=False,
        )

    def test_empty_output_does_not_ban(self):
        assert not ends_with_dangling_backtick(
            [],
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            fence_id=_FENCE,
            initial_reasoning=False,
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
        assert cfg[6] == _BACKTICK
        assert cfg[7] == _FENCE

    def test_parse_phase_ban_config_backward_compatible_without_backtick(self):
        """Legacy 6-field configs still parse (backtick/fence None)."""
        cfg = parse_phase_ban_config(
            {
                PHASE_BAN_XARG_KEY: [
                    _IM_END,
                    _THINK_START,
                    _THINK_END,
                    _TOOL_START,
                    _TOOL_END,
                    1,
                ]
            }
        )
        assert cfg is not None
        assert cfg[6] is None
        assert cfg[7] is None

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
            min_p=0.05,
            bad_words=list(req.bad_words),
            logit_bias=req.logit_bias,
            extra_args=dict(req.vllm_xargs) if req.vllm_xargs else None,
        )
        # Must not raise: bad_words + min_p are MTP-safe; logit_bias is not.
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
                _BACKTICK,
                _FENCE,
            ]
        }
        assert banned_ids_for_request([1, 2], extra) == [_IM_END]
        assert banned_ids_for_request([_THINK_END, 3], extra) == []

    def test_banned_ids_for_request_dangling_backtick(self):
        extra = {
            PHASE_BAN_XARG_KEY: [
                _IM_END,
                _THINK_START,
                _THINK_END,
                _TOOL_START,
                _TOOL_END,
                0,
                _BACKTICK,
                _FENCE,
            ]
        }
        assert banned_ids_for_request([_THINK_END, _BACKTICK], extra) == [_IM_END]
        # Non-sticky: any token after the backtick re-allows im_end.
        assert banned_ids_for_request([_THINK_END, _BACKTICK, 1], extra) == []

    def test_should_ignore_stop_token_reasoning_only(self):
        """A sampled im_end is ignored only mid-think, never for backticks.

        Ignoring a real im_end in the answer phase makes generation run
        past the end (rewritten-tail artifact) — history cannot be
        rewritten once the token is sampled.
        """
        extra = {
            PHASE_BAN_XARG_KEY: [
                _IM_END,
                _THINK_START,
                _THINK_END,
                _TOOL_START,
                _TOOL_END,
                1,
                _BACKTICK,
                _FENCE,
            ]
        }
        assert should_ignore_stop_token(_IM_END, [1, 2], extra)
        assert not should_ignore_stop_token(_IM_END, [_THINK_END], extra)
        assert not should_ignore_stop_token(_IM_END, [_THINK_END, _BACKTICK], extra)
        assert not should_ignore_stop_token(999, [1, 2], extra)
