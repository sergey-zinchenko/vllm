# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3 phase-aware special-token bans."""

from unittest.mock import MagicMock

from vllm.parser.qwen3_phase_stop import (
    BANG_STREAK_IM_END_BOOST,
    CITATION_LT_BOOST,
    CITATION_NUDGE_XARG_KEY,
    PHASE_BAN_XARG_KEY,
    QWEN_END_OF_TEXT,
    QWEN_IM_END,
    QWEN_MASK_BAD_WORDS,
    QWEN_MASK_END,
    QWEN_MASK_START,
    apply_endoftext_ban_to_request,
    banned_ids_for_request,
    ends_with_bang_newline_streak,
    ends_with_dangling_backtick,
    ends_with_failed_citation_tail,
    has_real_think_end,
    is_citation_think_end_at,
    is_in_reasoning_or_tool_phase,
    parse_citation_nudge_config,
    parse_phase_ban_config,
    resolve_newline_token_id,
    should_ban_im_end,
    should_ignore_stop_token,
    step_banned_ids,
    step_citation_logit_deltas,
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
_CLOSE_PAREN = 45
_OPEN_PAREN = 46
_LT = 50
_LT_SLASH = 51
_BANG = 0
_NEWLINE = 198


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
    ")": _CLOSE_PAREN,
    "(": _OPEN_PAREN,
    "<": _LT,
    "</": _LT_SLASH,
    "!": _BANG,
    "\n": _NEWLINE,
}

_STRUCTURAL_SPECIALS = {
    _THINK_START,
    _THINK_END,
    _TOOL_START,
    _TOOL_END,
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


class TestCitationThinkEndLatch:
    """Citation-shaped TE must not latch phase closed (chatcmpl-a0f7aa7).

    Parser sticky-aborts `` `\n</think>`` and stays in REASONING, but the
    special id is already in ``output_token_ids``. Treating it as a real
    close bans further TE and disables thinking_budget force — runaway.
    """

    _TRAILING = frozenset({_SPACE_BACKTICK})

    def test_backtick_nl_te_is_citation(self):
        ids = [_THINK_START, 1, _BACKTICK, _NEWLINE, _THINK_END]
        assert is_citation_think_end_at(
            ids,
            len(ids) - 1,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            trailing_backtick_ids=self._TRAILING,
            newline_id=_NEWLINE,
        )
        assert not has_real_think_end(
            ids,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            trailing_backtick_ids=self._TRAILING,
            newline_id=_NEWLINE,
        )

    def test_backtick_te_is_citation(self):
        ids = [_THINK_START, 1, _BACKTICK, _THINK_END]
        assert is_citation_think_end_at(
            ids,
            len(ids) - 1,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            newline_id=_NEWLINE,
        )

    def test_citation_te_keeps_reasoning_phase(self):
        ids = [_THINK_START, 1, _BACKTICK, _NEWLINE, _THINK_END, 2]
        assert is_in_reasoning_or_tool_phase(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
            backtick_id=_BACKTICK,
            trailing_backtick_ids=self._TRAILING,
            newline_id=_NEWLINE,
        )

    def test_real_te_still_closes(self):
        ids = [_THINK_START, 1, _THINK_END, 2]
        assert not is_in_reasoning_or_tool_phase(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            tool_start_id=_TOOL_START,
            tool_end_id=_TOOL_END,
            initial_reasoning=True,
            backtick_id=_BACKTICK,
            newline_id=_NEWLINE,
        )

    def test_citation_te_does_not_post_latch_ban(self):
        """After citation TE only, further TE must remain samplable."""
        ids = [_THINK_START, 1, _BACKTICK, _NEWLINE, _THINK_END, 2]
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=self._TRAILING,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _IM_END in banned  # still reasoning
        assert _THINK_END not in banned  # not latched closed

    def test_real_te_after_citation_latches(self):
        ids = [
            _THINK_START,
            1,
            _BACKTICK,
            _NEWLINE,
            _THINK_END,
            2,
            _THINK_END,
            3,
        ]
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=self._TRAILING,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _IM_END not in banned
        assert _THINK_END in banned


class TestMergedBacktickStopGuard:
    """Merged trailing-backtick BPE tokens must arm the im_end ban window.

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
        assert _IM_END in banned_ids_for_request(ids, cfg)

    def test_merged_paren_backtick_bans_im_end(self):
        cfg = self._cfg()
        ids = [_THINK_END, 1, _PAREN_BACKTICK]
        assert _IM_END in banned_ids_for_request(ids, cfg)

    def test_config_carries_trailing_backtick_ids(self):
        raw = self._cfg()[PHASE_BAN_XARG_KEY]
        # [8]=close_paren, [9]=open_paren, [10+]=trailing backtick ids
        tail = set(raw[10:])
        assert _SPACE_BACKTICK in tail
        assert _PAREN_BACKTICK in tail

    def test_double_backtick_not_in_trailing_set(self):
        # "``" / "```" are fence-ish closers; stopping after them is legit.
        raw = self._cfg()[PHASE_BAN_XARG_KEY]
        tail = set(raw[10:])
        assert _DOUBLE_BACKTICK not in tail
        assert _FENCE not in tail

    def test_lone_backtick_still_bans(self):
        cfg = self._cfg()
        assert _IM_END in banned_ids_for_request([_THINK_END, 1, _BACKTICK], cfg)

    def test_merged_backtick_plus_one_still_bans_im_end(self):
        # Two-token window: one token after merged backtick still bans.
        cfg = self._cfg()
        assert _IM_END in banned_ids_for_request(
            [_THINK_END, _SPACE_BACKTICK, 2], cfg
        )

    def test_merged_backtick_ban_is_non_sticky_after_two(self):
        # Control: two tokens after the merged backtick re-allows im_end;
        # post-latch still bans think_end.
        cfg = self._cfg()
        assert banned_ids_for_request(
            [_THINK_END, _SPACE_BACKTICK, 2, 3], cfg
        ) == [_THINK_END]

    def test_stop_ignore_stays_reasoning_only(self):
        # Control: a sampled im_end after a merged backtick is never ignored.
        cfg = self._cfg()
        assert not should_ignore_stop_token(
            _IM_END, [_THINK_END, 1, _SPACE_BACKTICK], cfg
        )


class TestBacktickCitationIdsAllowed:
    """Dangling backtick bans all structural specials + ``im_end`` + ``!``.

    Citations must use text/BPE (soft ``<`` / ``</`` boost); specials after
    `` ` `` caused early think-close / ``Treat `\n!`` when TE was banned
    alone. Two-token window still covers `` `\n</think>`` (MTP).
    """

    _DANGLING = _STRUCTURAL_SPECIALS | {_IM_END, _BANG}

    def _cfg(self):
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=True)
        return req.vllm_xargs

    def test_reasoning_backtick_bans_all_structural_specials(self):
        banned = set(banned_ids_for_request([1, _SPACE_BACKTICK], self._cfg()))
        assert banned >= self._DANGLING

    def test_answer_backtick_bans_all_structural_specials(self):
        ids = [_THINK_END, 1, _SPACE_BACKTICK]
        banned = set(banned_ids_for_request(ids, self._cfg()))
        assert banned >= self._DANGLING

    def test_lone_backtick_bans_tool_ids(self):
        banned = set(banned_ids_for_request([1, _BACKTICK], self._cfg()))
        assert _TOOL_START in banned
        assert _TOOL_END in banned
        assert _THINK_END in banned
        assert _IM_END in banned
        assert _BANG in banned

    def test_backtick_newline_still_bans_specials(self):
        """Regression chatcmpl-a2c610: `` `\n</think>`` mid-citation."""
        banned = set(
            banned_ids_for_request([1, _BACKTICK, _NEWLINE], self._cfg())
        )
        assert banned >= self._DANGLING

    def test_merged_backtick_newline_still_bans_specials(self):
        banned = set(
            banned_ids_for_request([1, _SPACE_BACKTICK, _NEWLINE], self._cfg())
        )
        assert banned >= self._DANGLING

    def test_ban_sticky_one_token_then_all_specials(self):
        # One token after backtick: still dangling.
        cfg = self._cfg()
        banned = set(banned_ids_for_request([1, _SPACE_BACKTICK, 2], cfg))
        assert banned == self._DANGLING

    def test_ban_is_non_sticky_after_two_tokens(self):
        # Two tokens later only the reasoning-phase im_end ban remains.
        cfg = self._cfg()
        assert banned_ids_for_request([1, _SPACE_BACKTICK, 2, 3], cfg) == [_IM_END]

    def test_tools_allowed_again_after_window(self):
        cfg = self._cfg()
        banned = set(
            banned_ids_for_request(
                [_THINK_END, 1, _SPACE_BACKTICK, 2, 3], cfg
            )
        )
        assert banned.isdisjoint({_TOOL_START, _TOOL_END, _THINK_START, _BANG})
        assert _THINK_END in banned  # post-close latch
        assert _IM_END not in banned

    def test_stop_ignore_stays_reasoning_only(self):
        # Control: the backtick rule never rescues an already-sampled stop.
        cfg = self._cfg()
        assert not should_ignore_stop_token(
            _IM_END, [_THINK_END, 1, _SPACE_BACKTICK], cfg
        )


class TestCitationTextPathNudge:
    """Soft ``<`` / ``</`` boost while dangling; ``!`` is hard-banned."""

    def _nudge_ids(self):
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=True)
        assert CITATION_NUDGE_XARG_KEY in req.vllm_xargs
        return parse_citation_nudge_config(req.vllm_xargs)

    def test_apply_sets_citation_nudge_xarg(self):
        lt, lt_slash, bang, newline = self._nudge_ids()
        assert lt == _LT
        assert lt_slash == _LT_SLASH
        assert bang == _BANG
        assert newline == _NEWLINE

    def test_parse_legacy_three_field_nudge(self):
        cfg = parse_citation_nudge_config(
            {CITATION_NUDGE_XARG_KEY: [_LT, _LT_SLASH, _BANG]}
        )
        assert cfg == (_LT, _LT_SLASH, _BANG, None)

    def test_deltas_active_when_dangling(self):
        deltas = dict(
            step_citation_logit_deltas(
                [1, _SPACE_BACKTICK],
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                lt_id=_LT,
                lt_slash_id=_LT_SLASH,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert deltas[_LT] == CITATION_LT_BOOST
        assert deltas[_LT_SLASH] == CITATION_LT_BOOST
        assert _BANG not in deltas

    def test_dangling_hard_bans_bang(self):
        banned = set(
            step_banned_ids(
                [1, _SPACE_BACKTICK],
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _BANG in banned

    def test_deltas_empty_when_not_dangling(self):
        assert (
            step_citation_logit_deltas(
                [1, _SPACE_BACKTICK, 2, 3],
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                lt_id=_LT,
                lt_slash_id=_LT_SLASH,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
            == []
        )

    def test_spec_row_after_backtick_newline_bans_tool_and_think_end(self):
        """MTP: draft ``[` , \\n, TE]`` bans TE + tool_start + bang."""
        accepted = [_THINK_END, 18307]
        draft = [_SPACE_BACKTICK, _NEWLINE, _THINK_END]
        banned = set(
            step_banned_ids(
                accepted + draft[:2],
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert {_THINK_END, _TOOL_START, _IM_END, _BANG} <= banned
        deltas = dict(
            step_citation_logit_deltas(
                accepted + draft[:1],
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                lt_id=_LT,
                lt_slash_id=_LT_SLASH,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _LT in deltas
        assert _BANG not in deltas


class TestBangNewlineStreak:
    """Anti-loop for production ``!\n!\n`` after dangling backtick."""

    def test_streak_detector_requires_min_len_and_bangs(self):
        assert not ends_with_bang_newline_streak(
            [_BANG], bang_id=_BANG, newline_id=_NEWLINE
        )
        assert not ends_with_bang_newline_streak(
            [_BANG, _NEWLINE], bang_id=_BANG, newline_id=_NEWLINE
        )
        assert not ends_with_bang_newline_streak(
            [_BANG, _NEWLINE, _BANG], bang_id=_BANG, newline_id=_NEWLINE
        )
        assert ends_with_bang_newline_streak(
            [_BANG, _NEWLINE, _BANG, _NEWLINE],
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )

    def test_streak_bans_bang_not_think_end(self):
        ids = [_THINK_END, 1, _BANG, _NEWLINE, _BANG, _NEWLINE]
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _BANG in banned
        # Post-latch still bans TE; streak must not add extra structural bans.
        assert banned == {_BANG, _THINK_END}

    def test_streak_after_think_close_boosts_im_end(self):
        ids = [_THINK_END, 1, _BANG, _NEWLINE, _BANG, _NEWLINE]
        deltas = dict(
            step_citation_logit_deltas(
                ids,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
                im_end_id=_IM_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
            )
        )
        assert deltas[_IM_END] == BANG_STREAK_IM_END_BOOST

    def test_streak_in_reasoning_does_not_boost_im_end(self):
        ids = [1, _BANG, _NEWLINE, _BANG, _NEWLINE]
        deltas = step_citation_logit_deltas(
            ids,
            think_start_id=_THINK_START,
            think_end_id=_THINK_END,
            backtick_id=_BACKTICK,
            bang_id=_BANG,
            newline_id=_NEWLINE,
            im_end_id=_IM_END,
            initial_reasoning=True,
        )
        assert deltas == []
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _BANG in banned
        assert _THINK_END not in banned
        assert _IM_END in banned

    def test_short_bang_punctuation_not_banned(self):
        for ids in ([_THINK_END, 1, _BANG], [_THINK_END, 1, _BANG, _NEWLINE]):
            banned = set(
                step_banned_ids(
                    ids,
                    im_end_id=_IM_END,
                    think_start_id=_THINK_START,
                    think_end_id=_THINK_END,
                    tool_start_id=_TOOL_START,
                    tool_end_id=_TOOL_END,
                    initial_reasoning=True,
                    bang_id=_BANG,
                    newline_id=_NEWLINE,
                )
            )
            assert _BANG not in banned


class TestNewlineResolve:
    """Qwen HF vocab stores newline as ``Ċ``, not ``\"\\n\"``."""

    def test_resolve_prefers_literal_newline(self):
        assert resolve_newline_token_id({"\n": 198, "Ċ": 199}) == 198

    def test_resolve_falls_back_to_c_dot(self):
        assert resolve_newline_token_id({"Ċ": 198, "!": 0}) == 198

    def test_apply_sets_newline_from_c_dot_vocab(self):
        vocab = {k: v for k, v in _VOCAB.items() if k != "\n"}
        vocab["Ċ"] = _NEWLINE
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, vocab, initial_reasoning=True)
        lt, lt_slash, bang, newline = parse_citation_nudge_config(req.vllm_xargs)
        assert newline == _NEWLINE
        assert bang == _BANG
        assert lt == _LT
        assert lt_slash == _LT_SLASH


class TestFailedCitationTail:
    """Regression chatcmpl-8822c896: `` `\\n!`` must not release ``im_end``."""

    _DANGLING = _STRUCTURAL_SPECIALS | {_IM_END, _BANG}

    def test_detector_backtick_newline_bang(self):
        assert ends_with_failed_citation_tail(
            [_THINK_END, 1, _BACKTICK, _NEWLINE, _BANG],
            backtick_id=_BACKTICK,
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )
        assert ends_with_failed_citation_tail(
            [_THINK_END, _SPACE_BACKTICK, _NEWLINE, _BANG],
            backtick_id=_BACKTICK,
            trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )

    def test_detector_clears_on_continuation(self):
        assert not ends_with_failed_citation_tail(
            [_THINK_END, _BACKTICK, _NEWLINE, _BANG, 99],
            backtick_id=_BACKTICK,
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )

    def test_detector_false_without_backtick(self):
        assert not ends_with_failed_citation_tail(
            [_THINK_END, 1, _BANG, _NEWLINE],
            backtick_id=_BACKTICK,
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )

    def test_backtick_newline_bang_bans_im_end_and_bang(self):
        ids = [_THINK_END, 1, _BACKTICK, _NEWLINE, _BANG]
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert banned >= self._DANGLING

    def test_continuation_after_tail_frees_im_end(self):
        ids = [_THINK_END, 1, _BACKTICK, _NEWLINE, _BANG, 99]
        banned = set(
            step_banned_ids(
                ids,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert _IM_END not in banned
        assert _BANG not in banned
        assert _THINK_END in banned  # post-latch

    def test_short_bang_without_backtick_not_guarded(self):
        for ids in ([_THINK_END, 1, _BANG], [_THINK_END, 1, _BANG, _NEWLINE]):
            assert not ends_with_failed_citation_tail(
                ids,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
            banned = set(
                step_banned_ids(
                    ids,
                    im_end_id=_IM_END,
                    think_start_id=_THINK_START,
                    think_end_id=_THINK_END,
                    tool_start_id=_TOOL_START,
                    tool_end_id=_TOOL_END,
                    initial_reasoning=True,
                    backtick_id=_BACKTICK,
                    bang_id=_BANG,
                    newline_id=_NEWLINE,
                )
            )
            assert _IM_END not in banned
            assert _BANG not in banned

    def test_streak_on_citation_tail_does_not_boost_im_end(self):
        ids = [
            _THINK_END,
            _BACKTICK,
            _NEWLINE,
            _BANG,
            _NEWLINE,
            _BANG,
            _NEWLINE,
        ]
        assert ends_with_failed_citation_tail(
            ids,
            backtick_id=_BACKTICK,
            bang_id=_BANG,
            newline_id=_NEWLINE,
        )
        deltas = dict(
            step_citation_logit_deltas(
                ids,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                backtick_id=_BACKTICK,
                bang_id=_BANG,
                newline_id=_NEWLINE,
                im_end_id=_IM_END,
                lt_id=_LT,
                lt_slash_id=_LT_SLASH,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
            )
        )
        assert _IM_END not in deltas
        assert deltas[_LT] == CITATION_LT_BOOST

    def test_mtp_row_after_backtick_newline_still_bans_bang(self):
        """MTP: accepted+draft[:2] == `` `\\n``; next draft bang stays banned."""
        accepted = [_THINK_END, 18307]
        draft = [_SPACE_BACKTICK, _NEWLINE, _BANG]
        banned = set(
            step_banned_ids(
                accepted + draft[:2],
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert {_IM_END, _BANG, _THINK_END} <= banned
        # After draft bang accepted into prefix: citation tail still guards.
        banned_tail = set(
            step_banned_ids(
                accepted + draft[:3],
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
                bang_id=_BANG,
                newline_id=_NEWLINE,
            )
        )
        assert {_IM_END, _BANG} <= banned_tail


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
        # im_end free; think_end stays banned after the one-way latch.
        assert banned_ids_for_request(self._CITED_THINK, self._CFG) == [
            _THINK_END
        ]

    def test_think_start_after_close_does_not_ignore_stop(self):
        assert not should_ignore_stop_token(_IM_END, self._CITED_THINK, self._CFG)

    def test_open_think_still_bans(self):
        # Control: genuine unclosed think keeps the ban (unchanged).
        assert banned_ids_for_request([_THINK_START, 1], self._CFG) == [_IM_END]

    def test_closed_think_still_allows_im_end(self):
        # Closed think allows im_end; latch still bans further think_end.
        assert banned_ids_for_request(
            [_THINK_START, 1, _THINK_END, 2], self._CFG
        ) == [_THINK_END]

    def test_closed_think_bans_further_think_end(self):
        """Regression chatcmpl-8b9c: no second special </think> mid-answer."""
        banned = set(
            banned_ids_for_request([_THINK_END, 1, 2], self._CFG)
        )
        assert banned == {_THINK_END}
        assert _IM_END not in banned


class TestDanglingBacktick:
    """The im_end ban uses a two-token window after `` ` ``, not span parity.

    Parity counting deadlocks (BPE merges hide one side of real spans) and
    caused endless ``!!!`` tails. A one-token window let `` `\n</think>``
    clear the guard; two tokens covers newline/MTP without sticking.
    """

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

    def test_backtick_plus_one_still_dangling(self):
        ids = [_THINK_END, 1, _BACKTICK, 2]
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
        # banned once two other tokens followed (no parity deadlock).
        ids = [_THINK_END, 1, _BACKTICK, 2, 3]
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
        # The backtick rule itself still fires mid-think (im_end +
        # think_end guard); tool / think_start ids stay legal.
        assert ends_with_dangling_backtick(
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
        assert QWEN_MASK_START in req.bad_words
        assert QWEN_MASK_END in req.bad_words
        for word in QWEN_MASK_BAD_WORDS:
            assert word in req.bad_words
        assert PHASE_BAN_XARG_KEY in req.vllm_xargs
        cfg = parse_phase_ban_config(req.vllm_xargs)
        assert cfg is not None
        assert cfg[0] == _IM_END
        assert cfg[6] == _BACKTICK
        assert cfg[7] == _FENCE
        assert cfg[8] == _CLOSE_PAREN
        assert cfg[9] == _OPEN_PAREN

    def test_parse_phase_ban_config_backward_compatible_without_backtick(self):
        """Legacy 6-field configs still parse (backtick/fence/paren None)."""
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
        assert cfg[8] is None
        assert cfg[9] is None

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
        assert banned_ids_for_request([_THINK_END, 3], extra) == [_THINK_END]

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
        banned = set(banned_ids_for_request([_THINK_END, _BACKTICK], extra))
        assert _STRUCTURAL_SPECIALS | {_IM_END} <= banned
        # Two-token window: one token after backtick still bans specials.
        assert _STRUCTURAL_SPECIALS | {_IM_END} <= set(
            banned_ids_for_request([_THINK_END, _BACKTICK, 1], extra)
        )
        # Non-sticky for im_end; latch keeps think_end banned.
        assert banned_ids_for_request(
            [_THINK_END, _BACKTICK, 1, 2], extra
        ) == [_THINK_END]

    def test_spec_draft_prefix_bans_think_end_after_backtick_newline(self):
        """MTP hole: draft ``[space_backtick, newline, think_end]`` mid-answer.

        After the first think_end latch, row k=2 must see accepted+draft[:2]
        (dangling) and ban TE — same shape as chatcmpl-8b9c Treat/`/TE.
        """
        newline = 198
        treat = 18307
        # Prior accepted output already closed think once.
        accepted = [_THINK_END, treat]
        draft = [_SPACE_BACKTICK, newline, _THINK_END]
        # Row for think_end (index 2): prefix ends with backtick, newline.
        banned = set(
            step_banned_ids(
                accepted + draft[:2],
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
            )
        )
        assert _STRUCTURAL_SPECIALS | {_IM_END} <= banned
        # Row 0 (before draft backtick): latch alone bans TE; im_end free.
        banned0 = set(
            step_banned_ids(
                accepted,
                im_end_id=_IM_END,
                think_start_id=_THINK_START,
                think_end_id=_THINK_END,
                tool_start_id=_TOOL_START,
                tool_end_id=_TOOL_END,
                initial_reasoning=True,
                backtick_id=_BACKTICK,
                fence_id=_FENCE,
                trailing_backtick_ids=frozenset({_SPACE_BACKTICK}),
            )
        )
        assert banned0 == {_THINK_END}

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


class TestEmptyStartBan:
    """First decode step must not open with ``)``/``(`` or instant ``</think>``."""

    def _cfg(self):
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=True)
        return req.vllm_xargs

    def test_empty_output_bans_think_end_and_parens(self):
        banned = set(banned_ids_for_request([], self._cfg()))
        assert {_IM_END, _THINK_END, _CLOSE_PAREN, _OPEN_PAREN} <= banned

    def test_after_first_token_parens_and_instant_close_allowed(self):
        # One accepted token clears empty-start bans; phase im_end remains.
        banned = set(banned_ids_for_request([1], self._cfg()))
        assert banned == {_IM_END}
        assert _THINK_END not in banned
        assert _CLOSE_PAREN not in banned
        assert _OPEN_PAREN not in banned

    def test_empty_without_initial_reasoning_skips_empty_start(self):
        req = MagicMock()
        req.logit_bias = None
        req.bad_words = []
        req.vllm_xargs = None
        apply_endoftext_ban_to_request(req, _VOCAB, initial_reasoning=False)
        banned = set(banned_ids_for_request([], req.vllm_xargs))
        assert _THINK_END not in banned
        assert _CLOSE_PAREN not in banned
        assert _OPEN_PAREN not in banned
