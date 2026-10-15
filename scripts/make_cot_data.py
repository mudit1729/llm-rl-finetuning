"""Structured-thinking SFT data for run 3: solver-played Wordle games in the env's exact format.

Each game is a training seed (its secret is one of the 138 training words), played by a
hard-mode solver: every guess fits all feedback so far. Guesses come from common English
words (single Qwen3 tokens that TextArena accepts) with the 18 held-out words removed, so
the model learns to propose likely words from its own vocabulary rather than the answer
list. Each turn's thinking writes the clue state down, spells the candidate out letter by
letter and checks it: length, not a repeat, then every clue.

    Clues:
    crane → X X G X G
    plate → X X G Y G
    Known: _ _ a _ e
    Misplaced: t (not 4th)
    Ruled out: c l n p r
    Try state = s t a t e: t can't be 4th ✗
    Try taste = t a s t e: 5 letters ✓, not guessed yet ✓, a 3rd ✓, e 5th ✓, has t (not 4th) ✓, no ruled-out letters ✓
    taste fits every clue.

About half the turns first reject a near miss: a word breaking a clue, a word of the wrong
length (common 4- and 6-letter words) or an earlier guess. The first version of this data had
no length or repeat checks, and that SFT model lost 48% of eval games to invalid moves, mostly
4-letter words (notes/REPORT.md, run 3).

Messages mirror the wordle env: its SYSTEM_PROMPT, the seeded board as the first user turn,
then `_latest_feedback(observation)` after each guess. Assistant turns are
`<think>\\n…\\n</think>\\n[word]` in `content`, exactly what the model samples during RL
(prime-qwen3 strips whitespace from `reasoning_content`, so that field can't reproduce it).

Writes data/cot-wordle/train.jsonl and results/cot_data_summary.json.

    uv run python scripts/make_cot_data.py [--games 2560]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import textarena as ta
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from textarena.envs.utils.word_lists import EnglishDictionary
from verifiers.v1.tasksets.textarena.taskset import SYSTEM_PROMPT, _latest_feedback
from wordle_turns.shaping import consistent

ROOT = Path(__file__).resolve().parent.parent
ORD = ["1st", "2nd", "3rd", "4th", "5th"]
SEQ_LEN = 1024  # sft_cot.toml data.seq_len: whole games must fit
OPENERS = ["crane", "slate", "trace", "stare", "arise", "least", "crate", "later", "raise", "alert"]


def common_words(dictionary: EnglishDictionary, length: int, exclude: set[str] = frozenset(),
                 always: set[str] = frozenset()) -> list[str]:
    """Words of `length` letters that are one Qwen3 token (a proxy for common) and valid in the game."""
    vocab = json.loads(Path(hf_hub_download("PrimeIntellect/Qwen3-1.7B", "tokenizer.json")).read_text())
    tokens = vocab["model"]["vocab"]
    words = {t[1:] for t in tokens if t.startswith("Ġ") and len(t) == length + 1 and t[1:].isascii()
             and t[1:].isalpha() and t[1:].islower()}
    words = {w for w in words if dictionary.is_english_word(w)} | set(always)
    return sorted(words - set(exclude))


def spell(word: str) -> str:
    return " ".join(word)


def clue_state(history: list[tuple[str, str]]):
    """What the feedback so far says, as a player would write it down (never wrong, may omit)."""
    known = ["_"] * 5
    misplaced: dict[str, set[int]] = {}
    present: set[str] = set()
    for guess, fb in history:
        for i, (ch, f) in enumerate(zip(guess, fb)):
            if f == "G":
                known[i] = ch
                present.add(ch)
            elif f == "Y":
                misplaced.setdefault(ch, set()).add(i)
                present.add(ch)
    ruled_out = sorted({ch for g, fb in history for ch, f in zip(g, fb) if f == "X"} - present)
    misplaced = {ch: sorted(p) for ch, p in sorted(misplaced.items()) if ch not in known}
    return known, misplaced, ruled_out


def checks(word: str, known, misplaced, ruled_out) -> list[tuple[str, str, bool]]:
    """(pass text, fail text, passed) for each written-down clue."""
    out = []
    for i, ch in enumerate(known):
        if ch != "_":
            out.append((f"{ch} {ORD[i]}", f"{ORD[i]} letter must be {ch}", word[i] == ch))
    for ch, pos in misplaced.items():
        if ch not in word:
            out.append(("", f"no {ch}", False))
        else:
            bad = [i for i in pos if word[i] == ch]
            text = f"has {ch} (not {'/'.join(ORD[i] for i in pos)})"
            out.append((text, f"{ch} can't be {ORD[bad[0]]}" if bad else "", not bad))
    if ruled_out:
        hit = [ch for ch in ruled_out if ch in word]
        out.append(("no ruled-out letters", f"{hit[0]} is ruled out" if hit else "", not hit))
    return out


def near_miss(vocab: list[str], guessed: set[str], known, misplaced, ruled_out, rng: random.Random):
    """A word that keeps the known letters but breaks another clue, with the first broken one."""
    for word in rng.sample(vocab, min(400, len(vocab))):
        if word in guessed:
            continue
        cs = checks(word, known, misplaced, ruled_out)
        greens_ok = all(ok for _, _, ok in cs[:sum(c != "_" for c in known)])
        failed = [fail for _, fail, ok in cs if not ok]
        if greens_ok and failed:
            return word, failed[0]
    return None


def thinking(history, word, vocab, off_length, rng: random.Random) -> str:
    if not history:
        return f"No clues yet. Open with {word} = {spell(word)}: 5 different common letters."
    guessed = [g for g, _ in history]
    known, misplaced, ruled_out = clue_state(history)
    lines = ["Clues:"] + [f"{g} → {' '.join(fb)}" for g, fb in history]
    lines.append("Known: " + " ".join(known))
    lines.append("Misplaced: " + (", ".join(f"{ch} (not {'/'.join(ORD[i] for i in p)})"
                                            for ch, p in misplaced.items()) or "none"))
    lines.append("Ruled out: " + (" ".join(ruled_out) or "none"))
    r = rng.random()
    if r < 0.25 and (miss := near_miss(vocab, set(guessed), known, misplaced, ruled_out, rng)):
        lines.append(f"Try {miss[0]} = {spell(miss[0])}: {miss[1]} ✗")
    elif 0.25 <= r < 0.45:
        letters = {c for c in known if c != "_"} | set(misplaced)
        near = [w for w in off_length if letters & set(w)] or off_length
        miss = rng.choice(near)
        lines.append(f"Try {miss} = {spell(miss)}: {len(miss)} letters ✗")
    elif 0.45 <= r < 0.55:
        miss = rng.choice(guessed)
        lines.append(f"Try {miss} = {spell(miss)}: already guessed ✗")
    passed = ["5 letters", "not guessed yet"] + [ok for ok, _, good in checks(word, known, misplaced, ruled_out)]
    lines.append(f"Try {word} = {spell(word)}: " + ", ".join(f"{t} ✓" for t in passed))
    lines.append(f"{word} fits every clue.")
    return "\n".join(lines)


def inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def play(game, seed: int, vocab: list[str], off_length: list[str], openers: list[str]):
    """One game, seeded exactly like the env (seed → reset); the solver has its own RNG."""
    random.seed(seed)
    game.reset(num_players=1)
    core = inner(game)
    rng = random.Random(f"cot-{seed}")
    history: list[tuple[str, str]] = []
    completion = []
    while True:
        guessed = {g for g, _ in history}
        if not history:
            word = rng.choice(openers)
        else:
            cands = [w for w in vocab if w not in guessed and consistent(w, history)]
            distinct = [w for w in cands if len(set(w)) == 5]
            word = rng.choice(distinct if len(cands) > 10 and distinct else cands)
        text = thinking(history, word, vocab, off_length, rng)
        completion.append({"role": "assistant", "content": f"<think>\n{text}\n</think>\n[{word}]"})
        n = len(core.state.game_state["guess_history"])
        game.step(f"[{word}]")
        gh = core.state.game_state["guess_history"]
        assert len(gh) == n + 1, f"seed {seed}: game rejected [{word}]"
        history.append((gh[-1][0], "".join(gh[-1][1])))
        if game.state.done:
            break
        _, observation = game.get_observation()
        completion.append({"role": "user", "content": _latest_feedback(str(observation))})
    won = str(game.state.game_info[0]["reason"]).startswith("Congratulations")
    return core.state.game_state["secret_word"], completion, history, won


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2560)
    ap.add_argument("--out", default=str(ROOT / "data" / "cot-wordle"))
    args = ap.parse_args()

    split = json.loads((ROOT / "configs" / "heldout_split.json").read_text())
    held_out, train_words = set(split["held_out_words"]), set(split["train_words"])
    excluded = set(split["excluded_train_seeds"])
    seeds = [s for s in range(*split["train_range"]) if s not in excluded]
    random.Random(0).shuffle(seeds)

    # The task prompt is the seeded board, identical for every Wordle seed (as in the taskset).
    def board(seed: int) -> str:
        random.seed(seed)
        env = ta.make(env_id="Wordle-v0")
        env.reset(num_players=1)
        return str(env.get_observation()[1])
    prompt = board(0)
    assert board(1) == prompt, "Wordle's first observation became seed-specific"

    game = ta.make(env_id="Wordle-v0")
    dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)
    vocab = common_words(dictionary, 5, held_out, train_words)
    off_length = common_words(dictionary, 4) + common_words(dictionary, 6)
    openers = [w for w in OPENERS if w in vocab]

    # Rendered length ≈ content tokens + 5 per message (role tags); tests/test_cot_data.py
    # checks the exact prime-qwen3 length.
    tok = Tokenizer.from_file(hf_hub_download("PrimeIntellect/Qwen3-1.7B", "tokenizer.json"))
    n_tokens = lambda msgs: sum(len(tok.encode(m["content"]).ids) + 5 for m in msgs)
    rows, lost, too_long = [], 0, 0
    for seed in seeds:
        if len(rows) == args.games:
            break
        secret, completion, history, won = play(game, seed, vocab, off_length, openers)
        assert secret in train_words, f"seed {seed} plays held-out word {secret}"
        if not won:
            lost += 1
            continue
        row = {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "completion": completion, "answer": secret, "seed": seed, "guesses": len(history),
        }
        if n_tokens(row["prompt"] + completion) > SEQ_LEN - 16:
            too_long += 1
            continue
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    guesses = [r["guesses"] for r in rows]
    summary = {
        "games": len(rows), "lost_and_dropped": lost, "too_long_and_dropped": too_long, "vocab_size": len(vocab), "off_length_words": len(off_length),
        "answers": len({r["answer"] for r in rows}),
        "guesses_mean": round(statistics.mean(guesses), 2),
        "guesses_hist": {k: guesses.count(k) for k in range(1, 7)},
        "held_out_words": sorted(held_out),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cot_data_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "held_out_words"}))


if __name__ == "__main__":
    main()
