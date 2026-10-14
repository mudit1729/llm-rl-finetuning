"""Unit tests for wordle_turns.shaping (pure functions). Run: uv run pytest tests -q"""

import random

import textarena as ta

from wordle_turns.shaping import (
    ShapingWeights,
    answer_text,
    candidates,
    consistent,
    feedback,
    info_gain,
    length_ramp,
    outcome_reward,
    turn_reward,
)


def _engine_feedback(guess: str, secret: str) -> str:
    """Ground truth: TextArena's own _evaluate_guess."""
    env = ta.make(env_id="Wordle-v0")
    env.reset(num_players=1)
    core = env
    while hasattr(core, "env"):
        core = core.env
    core.state.game_state["secret_word"] = secret
    return "".join(core._evaluate_guess(guess))


def test_feedback_matches_textarena_including_duplicates():
    pairs = [("crane", "spade"), ("slate", "spade"), ("apple", "paper"), ("eerie", "there"),
             ("sheep", "steep"), ("lever", "level"), ("aaaaa", "apple"), ("spade", "spade")]
    for guess, secret in pairs:
        assert feedback(guess, secret) == _engine_feedback(guess, secret), (guess, secret)


def test_candidates_and_consistency():
    pool = ["spade", "shade", "space", "slate", "crane", "grain"]
    hist = [("crane", feedback("crane", "spade"))]
    cands = candidates(pool, hist)
    assert "spade" in cands and "crane" not in cands and "grain" not in cands
    assert consistent("spade", hist) and not consistent("crane", hist)


def test_info_gain_bounds():
    assert info_gain(156, 156, 156) == 0.0
    assert abs(info_gain(156, 1, 156) - 1.0) < 1e-9
    assert 0 < info_gain(156, 20, 156) < 1


def test_turn_reward_signs():
    w = ShapingWeights()
    good = turn_reward(accepted=True, cutoff=False, tokens=300, is_consistent=True, ig=0.6, w=w)
    incons = turn_reward(accepted=True, cutoff=False, tokens=300, is_consistent=False, ig=0.6, w=w)
    rejected = turn_reward(accepted=False, cutoff=False, tokens=300, is_consistent=None, ig=0.0, w=w)
    cut = turn_reward(accepted=False, cutoff=True, tokens=1024, is_consistent=None, ig=0.0, w=w)
    assert good > incons > rejected > cut
    assert abs(good - 0.3) < 1e-9 and abs(cut - (-0.5 - 0.5 - 0.2)) < 1e-9


def test_length_ramp():
    w = ShapingWeights(soft_start=768, budget=1024)
    assert length_ramp(700, w) == 0.0 and length_ramp(896, w) == 0.5 and length_ramp(2000, w) == 1.0


def test_outcome_reward():
    w = ShapingWeights(speed_bonus=0.1)
    assert outcome_reward(False, 6, w) == 0.0
    assert outcome_reward(True, 3, w) == 1.3 and outcome_reward(True, 6, w) == 1.0


def test_answer_text():
    assert answer_text("<think>maybe [crane]</think>\n\n[slate]") == "[slate]"
    assert answer_text("<think>cut off while quoting [crane]") == ""
    assert answer_text("[slate]") == "[slate]"


def test_textarena_pool_is_the_156_words():
    random.seed(0)
    env = ta.make(env_id="Wordle-v0")
    core = env
    while hasattr(core, "env"):
        core = core.env
    assert len(core.word_list) == 156
