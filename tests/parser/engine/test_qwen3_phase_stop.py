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
_SPACE_BACKTICK = 42
_PAREN_BACKTICK = 43
_DOUBLE_BACKTICK = 44


_VOCAB = {
    "<think>": _THINK_START,
    "</think>": _THINK_END,
    "<tool_call>": _TOOL_START,
    "</tool_call>": _TOOL_END,
    QWEN_IM_END: _IM_END,
    QWEN_END_OF_TEXT: _EOT,
    "`": _BACKTICK,
    "```": _FENCE,
    # Merged BPE forms: prose opening backticks arrive as these, not as
    # the lone "`" token ("Ġ`" byte-level form ends with "`" the same way).
    " `": _SPACE_BACKTICK,
    "(`": _PAREN_BACKTICK,
    "``": _DOUBLE_BACKTICK,
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


class TestMergedBacktickStopGuard:
    """Merged trailing-backtick BPE tokens must arm the one-step im_end ban.

    Truncation scenario: answer reaches "... прямо внутри `" where the
    opening backtick is the merged " `" token (space+backtick), not the
    lone "`" id. The guard must fire for any token whose text ends with
    exactly one backtick, or the model can stop mid-citation.
    """

    def _cfg(self, vocab=_VOCAB):
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, vocab, initial_reasoning=True)
        return req.vllm_xargs

    def test_merged_space_backtick_bans_im_end(self):
        cfg = self._cfg()
        ids = [_THINK_END, 1, _SPACE_BACKTICK]
        assert banned_ids_for_request(ids, cfg) == [_IM_END]

    def test_merged_paren_backtick_bans_im_end(self):
        cfg = self._cfg()
        ids = [_THINK_END, 1, _PAREN_BACKTICK]
        assert banned_ids_for_request(ids, cfg) == [_IM_END]

    def test_config_carries_trailing_backtick_ids(self):
        raw = self._cfg()[PHASE_BAN_XARG_KEY]
        tail = set(raw[8:])
        assert _SPACE_BACKTICK in tail
        assert _PAREN_BACKTICK in tail

    def test_double_backtick_not_in_trailing_set(self):
        # "``" / "```" are fence-ish closers; stopping after them is legit.
        raw = self._cfg()[PHASE_BAN_XARG_KEY]
        tail = set(raw[8:])
        assert _DOUBLE_BACKTICK not in tail
        assert _FENCE not in tail

    def test_lone_backtick_still_bans(self):
        # Control (green on current code).
        cfg = self._cfg()
        assert banned_ids_for_request([_THINK_END, 1, _BACKTICK], cfg) == [_IM_END]

    def test_merged_backtick_ban_is_non_sticky(self):
        # Control: any token after the merged backtick re-allows im_end.
        cfg = self._cfg()
        assert banned_ids_for_request([_THINK_END, _SPACE_BACKTICK, 2], cfg) == []

    def test_stop_ignore_stays_reasoning_only(self):
        # Control: a sampled im_end after a merged backtick is never ignored.
        cfg = self._cfg()
        assert not should_ignore_stop_token(
            _IM_END, [_THINK_END, 1, _SPACE_BACKTICK], cfg
        )


class TestThinkCitationAfterClose:
    """A cited ``<think>`` special id in the answer must not re-arm bans.

    Screenshot scenario: chat about think tags, model emits a real
    ``<think>`` token id mid-answer after reasoning already closed. No
    second ``</think>`` ever comes, so re-entering the ban phase makes
    ``im_end`` banned forever (endless generation) and a sampled
    ``im_end`` ignored by check_stop (rewritten tails). One generation
    has at most one legitimate think block — post-tool re-think is a
    separate request.
    """

    _CFG = {
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
    # think closed, answer prose, cited <think> id, more prose
    _CITED_THINK = [_THINK_START, 1, _THINK_END, 2, _THINK_START, 3]

    def test_think_start_after_close_is_not_reasoning(self):
        assert not is_in_reasoning_or_tool_phase(
            self._CITED_THINK,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
        )

    def test_think_start_after_close_does_not_ban_im_end(self):
        assert banned_ids_for_request(self._CITED_THINK, self._CFG) == []

    def test_think_start_after_close_does_not_ignore_stop(self):
        assert not should_ignore_stop_token(_IM_END, self._CITED_THINK, self._CFG)

    def test_open_think_still_bans(self):
        # Control: genuine unclosed think keeps the ban (unchanged).
        assert banned_ids_for_request([_THINK_START, 1], self._CFG) == [_IM_END]

    def test_closed_think_still_allows(self):
        # Control: closed think allows im_end (unchanged).
        assert banned_ids_for_request([_THINK_START, 1, _THINK_END, 2], self._CFG) == []


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
