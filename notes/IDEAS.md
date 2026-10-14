# Ideas to raise the held-out Wordle win rate

Starting point: 40.5% (full RL) and 42.5% (LoRA) on 18 held-out words, with reasoning stripped.

## What our traces say is losing games

These come from the reasoning-stripped evals of the RL models, 200 games each.

| Signal | Full RL | LoRA |
|---|---|---|
| Games lost to two invalid moves | 88 | 87 |
| Games lost after using all 6 guesses | 31 | 28 |
| Turns cut off at 1,024 tokens (no move) | 15% | 17% |
| Guesses consistent with earlier feedback | 54% | 55% |
| Guesses that could still be the answer | 14% | 15% |

When the model guesses, a median of 3 of the 156 secret words are still possible. A trivial solver that filters the word list wins 18/18 held-out words in 2.8 guesses.

**The model loses on validity and on using its constraints, not on deep strategy.** Losing tokens to over-thinking compounds both.

## Ranked ideas

### 0. Free diagnostics: evals only, no training

1. **Standard eval with the reasoning parser on** (`--vllm.reasoning-parser qwen3`). Keep the example eval only as a secondary number.
2. **Same eval at 2,048 tokens per turn, and at 1,024 with a forced `</think>` at about 900 tokens** (Qwen3-style budget forcing). This measures how much the 15–17% of cut-off turns cost us. Source: s1 budget forcing, https://arxiv.org/abs/2501.19393
3. **Eval at Qwen3's recommended thinking settings** (T=0.6, top_p 0.95, top_k 20) as well as T=1.0. AWS saw solve rates change a lot with decoding. Source: https://github.com/aws-samples/sample-multi-turn-rl-wordle

*Caveat:* our full-model weights were not kept; only the LoRA adapter came back, and its base (our SFT export) is gone. The next run should keep the SFT export, or re-run SFT first (about 8 min).

### 1. Per-turn reward and turn-level credit (highest expected gain)

**The problem:** GRPO currently gives every token of an episode the same advantage. A cut-off turn, a repeated guess or a feedback-violating guess is never blamed on its own.

**Per-turn reward.** For each turn, the reward is:
- the information gain of the guess, `log2(|C_before| / |C_after|) / log2(156)`, where C is the set of answers still possible;
- minus a penalty for clue violations, repeats and non-words;
- minus a penalty for a cut-off turn.

**Win reward.** The win reward still dominates, and it is scaled by speed (for example 1.0 + 0.1 × guesses left) so that guesses-to-solve can finally move.

**Advantage.** Per-turn advantages are normalized across the 16 tries at the same word and discounted over later turns (MT-GRPO / IGPO). On turn 1 every try starts from the same state, so the GiGPO step term ranks openers there.

**Implementation:**
- a small custom prime-rl `Algorithm`, which accepts per-token advantages (`score_group` → `assign_advantages` with a token-aligned list);
- an env wrapper that records per-turn candidate counts;
- both applied as a patch to the pinned prime-rl.

**Evidence:**
- GiGPO: Qwen2.5-1.5B ALFWorld 86.7 vs GRPO 72.8. https://arxiv.org/abs/2505.10978
- MT-PPO: Sokoban 0.656 vs GRPO 0.078. https://arxiv.org/abs/2505.11821
- IGPO. https://arxiv.org/abs/2510.14967
- AWS (gpt-oss-20b, 15→69–84%) and Predibase used Wordle-specific shaping: consistency, information gain, repeat and non-word penalties. https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo
- GEM reports that GRPO falls short in multi-turn games because every turn gets the same advantage. https://arxiv.org/abs/2510.01051

**Pitfalls:**
- Keep the shaping bounded and smaller than the win term; summing per-guess rewards gave AWS's lost games positive reward.
- Information gain alone can favour a non-candidate guess late in the game.

### 2. Stop the cut-off turns

- **If 1,024 tokens per turn is fixed:** train Elastic-style. Cap thinking at about 900 tokens, force `</think>`, then allow about 100 answer tokens, and leave the forced tokens out of the loss. Source: https://arxiv.org/abs/2505.05315
- **Add a per-turn truncation penalty** (this folds into idea 1).
- **If the budget isn't fixed,** raising it to 2,048 tokens is simplest. It needs `seq_len` ≈ 16K and roughly doubles rollout time.
- **Length penalty on wins only, switched on after the reward plateaus.** Penalising all samples from the start collapsed outputs in several papers. prime-rl's built-in length penalty hits every rollout and uses whole-episode token counts, so it needs a patch to apply to wins only. Sources: https://arxiv.org/abs/2502.04463, https://arxiv.org/abs/2505.12284

### 3. A structured state summary in each observation

Append a line to every feedback message: known greens and yellows, excluded letters, words already guessed, guesses left. The current env sends only the latest feedback.

**Evidence:**
- charbull (Gemma-3-4B): win rate 16→31% with game history and clue summaries. https://github.com/charbull/wordle-lora-rl
- PABU: removing the observation history dropped an 8B model from 40% to 4% on LMRL-Gym Wordle. https://arxiv.org/abs/2602.09138
- AWS adds a plain-English state summary to every observation.

This targets our 46% of inconsistent guesses directly and should shorten the thinking. It changes the task interface, so report it as a separate track next to the unchanged eval.

### 4. Better starting data

- **Mid-game starts:** build training episodes that begin with 0–4 prior guesses already played (by a solver, or as random valid guesses). This was charbull's biggest single gain, and Predibase trains on mid-game states. It gives late-game turns, where constraint use matters, far more training signal.
- **Better SFT traces:** replace or augment V3-wordle with solver-generated or rejection-sampled games that show short, explicit constraint tracking ("greens: _ _ A _ E; not R, N, C…"). Keep the held-out words excluded.

### 5. Recipe knobs (cheap A/Bs in config)

- **Train longer.** Reward was still rising at step 200, and each word has been seen only about 12 times. Try 600–1,000 steps, with a held-out eval every 100 steps. Source: ProRL, https://arxiv.org/abs/2505.24864
- **More words per step.** Go from 8 to 16 words × 16 tries (prime-rl's reference run used 64 × 16), keeping group size at 16. Source: ScaleRL, https://arxiv.org/abs/2510.13786
- **Difficulty curriculum** (`curriculum.sampler = "difficulty_pool"`): down-weight solved words (mean reward ≥ 0.9) and add an `advantage_range` gate. Expect about +1–4 points, mostly from speed. Source: https://arxiv.org/abs/2504.03380
- **Training temperature 1.1–1.2, and log entropy.** RAGEN's "echo trap" starts with falling reward std and entropy. Sources: https://arxiv.org/abs/2504.20073, https://arxiv.org/abs/2505.22312
- **`loss_aggregation = "prompt"`:** worth one comparison run (ScaleRL).
- **LoRA:** α=32 at rank 8–16, lr 1e-5–1.5e-5, gives the same quality as full fine-tuning for less memory. Source: https://thinkingmachines.ai/blog/lora/
- **Staleness of up to 8 versions is fine** with importance-weighted losses (AReaL, ScaleRL), so no change is needed.

## Suggested order

1. **Run A (baseline++).** Re-run SFT and keep its export. Then run the diagnostics from idea 0, and LoRA RL for 600 steps with 16 words × 16 tries, the difficulty curriculum and a held-out eval every 100 steps. This shows how far scale alone goes.
2. **Run B (credit).** Same as run A plus the per-turn reward and advantages (ideas 1 and 2). This is the main bet.
3. **Run C (interface).** Run B plus the state summary and mid-game starts (ideas 3 and 4), reported as a separate track.

Compare all three on the same 200 held-out games: win rate, guesses to solve, consistency rate and cut-off turns.
