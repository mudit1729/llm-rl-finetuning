"""wordle-turns: TextArena Wordle-v0 with per-turn shaped rewards.

Same tasks, prompts and game engine as the `wordle` taskset (seeded Wordle-v0), so
seed selection and the held-out split carry over unchanged. Differences:

- The game reads only the answer after `</think>` (`shaping.answer_text`), as in
  training, whether or not the inference server runs a reasoning parser.
- Every model turn is recorded in `trace.info["turns"]` with its shaped reward
  (`shaping.turn_reward`); prime-rl's `turn_grpo` algorithm turns them into
  per-turn advantages.
- The episode reward is the outcome: win, scaled up for unused guesses. The game's
  own partial-credit reward is kept as the `game_reward` metric.
"""

from __future__ import annotations

import random
import re

import textarena as ta
import verifiers.v1 as vf
from verifiers.v1.tasksets.textarena.taskset import TextArenaEnv, TextArenaEnvConfig, _latest_feedback

from wordle_turns.shaping import (
    ShapingWeights,
    TurnRecord,
    answer_text,
    candidates,
    consistent,
    info_gain,
    outcome_reward,
    turn_reward,
)

_MOVE = re.compile(r"\[(\w+)\]")


class WordleTurnsEnvConfig(TextArenaEnvConfig):
    info_weight: float = 0.5
    """Weight of a valid guess's normalized information gain."""
    invalid_penalty: float = 0.5
    """Penalty for a rejected move (no [word], wrong length, repeat, not a word)."""
    cutoff_penalty: float = 0.5
    """Penalty for a turn that hit the per-turn token limit."""
    length_penalty: float = 0.2
    """Penalty at the token limit for the soft length ramp (0 below `soft_start`)."""
    inconsistent_penalty: float = 0.2
    """Penalty for an accepted guess that contradicts earlier feedback."""
    soft_start: int = 768
    """Completion tokens where the soft length ramp starts."""
    turn_budget: int = 1024
    """Per-turn completion budget (match the sampling max tokens)."""
    speed_bonus: float = 0.1
    """Outcome bonus per unused guess on a win."""

    def weights(self) -> ShapingWeights:
        return ShapingWeights(
            info=self.info_weight, invalid=self.invalid_penalty, cutoff=self.cutoff_penalty,
            length=self.length_penalty, inconsistent=self.inconsistent_penalty,
            soft_start=self.soft_start, budget=self.turn_budget, speed_bonus=self.speed_bonus,
        )


def _inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _game_move(text: str) -> str | None:
    """The word the game will read (mirrors ActionFormattingWrapper + the Wordle regex)."""
    action = text if ("[" in text or "]" in text) else f"[{text}]"
    m = _MOVE.search(action)
    return m.group(1).lower() if m else None


def _rejected_reason(move: str | None, history: list[tuple[str, str]], word_len: int) -> str:
    if move is None:
        return "format"
    if len(move) != word_len:
        return "length"
    if move in {g for g, _ in history}:
        return "repeat"
    return "not_a_word"


class WordleTurnsEnv(TextArenaEnv, vf.Env[WordleTurnsEnvConfig]):
    async def run(self, task, agents):
        w = self.config.weights()
        # Same seeding as TextArenaEnv.run: seed + make + reset with no await between.
        random.seed(task.data.info["seed"])
        game = ta.make(env_id=task.data.info["game"])
        game.reset(num_players=1)
        core = _inner(game)
        pool = list(core.word_list)
        word_len = core.state.game_state["word_length"]

        history: list[tuple[str, str]] = []  # accepted (guess, feedback)
        turns: list[TurnRecord] = []
        outcome: dict = {}
        async with agents.player.interaction(task) as interaction:
            calls_seen = len(interaction.trace.calls)
            segment = await interaction.turn()
            while not segment.terminated:
                calls = interaction.trace.calls
                call = calls[-1] if len(calls) > calls_seen else None
                calls_seen = len(calls)
                tokens = int(call.usage.completion_tokens) if call and call.usage else 0
                cutoff = bool(call and call.finish_reason == "length")

                text = answer_text(segment.last_reply)
                move = _game_move(text)
                before = candidates(pool, history)
                n_hist = len(core.state.game_state["guess_history"])
                game.step(text)
                guess_history = core.state.game_state["guess_history"]
                accepted = len(guess_history) > n_hist
                if accepted:
                    guess, fb = guess_history[-1][0], "".join(guess_history[-1][1])
                    is_consistent = consistent(guess, history)
                    history.append((guess, fb))
                    after = candidates(pool, history)
                    ig = info_gain(len(before), len(after), len(pool))
                    reason = None
                else:
                    is_consistent, after, ig = None, None, 0.0
                    reason = _rejected_reason(move, history, word_len)
                turns.append(TurnRecord(
                    index=len(turns), tokens=tokens, cutoff=cutoff, move=move, accepted=accepted,
                    rejected_reason=reason, consistent=is_consistent, cand_before=len(before),
                    cand_after=len(after) if after is not None else None, info_gain=round(ig, 4),
                    reward=round(turn_reward(accepted=accepted, cutoff=cutoff, tokens=tokens,
                                             is_consistent=is_consistent, ig=ig, w=w), 4),
                ))
                if game.state.done:
                    outcome["reward"] = float((game.state.rewards or {}).get(0, 0.0))
                    outcome["reason"] = str(game.state.game_info[0]["reason"])
                    break
                _, observation = game.get_observation()
                segment = await interaction.turn(_latest_feedback(str(observation)))

        trace = interaction.trace
        won = outcome.get("reason", "").startswith("Congratulations")
        trace.record_reward("outcome", outcome_reward(won, len(history), w), 1.0)
        n = max(1, len(turns))
        trace.record_metrics({
            "won": float(won),
            "game_reward": outcome.get("reward", 0.0),
            "valid_guesses": float(len(history)),
            "cutoff_turn_rate": sum(t.cutoff for t in turns) / n,
            "rejected_move_rate": sum(not t.accepted for t in turns) / n,
            "consistent_guess_rate": (sum(bool(t.consistent) for t in turns if t.accepted)
                                      / max(1, sum(t.accepted for t in turns))),
            "turn_reward_mean": sum(t.reward for t in turns) / n,
        })
        trace.info["turns"] = [t.to_dict() for t in turns]
        if "reason" in outcome:
            trace.info["game_outcome"] = outcome["reason"]
