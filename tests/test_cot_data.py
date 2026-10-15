"""Checks for run 3's structured-thinking SFT data (scripts/make_cot_data.py).

Needs data/cot-wordle/train.jsonl (`make cot-data`). The rendering checks also need the
PrimeIntellect/Qwen3-1.7B tokenizer (downloaded on first use) and, for the comparison with
real RL turns, pulled training traces under artifacts/.
"""

import glob
import json
import re
from pathlib import Path

import pytest
from verifiers.v1.tasksets.textarena.taskset import SYSTEM_PROMPT
from wordle_turns.shaping import consistent, feedback

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cot-wordle" / "train.jsonl"
SPLIT = json.loads((ROOT / "configs" / "heldout_split.json").read_text())
ORD = ["1st", "2nd", "3rd", "4th", "5th"]
FEEDBACK = re.compile(r"^[A-Z]( [A-Z]){4}\n[GYX]( [GYX]){4}\nYou have \d guesses left\.$")
TURN = re.compile(r"^<think>\n(.+)\n</think>\n\[([a-z]{5})\]$", re.S)


@pytest.fixture(scope="module")
def rows():
    if not DATA.exists():
        pytest.skip("no data/cot-wordle/train.jsonl (make cot-data)")
    return [json.loads(line) for line in DATA.open()]


def turns(row):
    """(thinking, guess) per assistant turn, plus the feedback messages between them."""
    out, fbs = [], []
    for m in row["completion"]:
        if m["role"] == "assistant":
            match = TURN.match(m["content"])
            assert match, f"seed {row['seed']}: bad assistant turn {m['content'][:80]!r}"
            out.append((match.group(1), match.group(2)))
        else:
            fbs.append(m["content"])
    return out, fbs


def test_held_out_words_never_appear(rows):
    held_out = set(SPLIT["held_out_words"])
    for row in rows:
        assert row["answer"] in set(SPLIT["train_words"])
        assert row["seed"] not in set(SPLIT["excluded_train_seeds"])
        text = " ".join(m["content"] for m in row["completion"] if m["role"] == "assistant")
        assert not set(re.findall(r"[a-z]+", text)) & held_out, f"seed {row['seed']} mentions an eval word"


def test_messages_match_the_env(rows):
    for row in rows:
        assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert row["prompt"][1]["content"].startswith("\n[GAME] You are Playing Wordle.")
        roles = [m["role"] for m in row["completion"]]
        assert roles == ["assistant", "user"] * (len(roles) // 2) + ["assistant"], "alternating, ends on the win"
        for m in row["completion"]:
            if m["role"] == "user":
                assert FEEDBACK.match(m["content"]), m["content"]


def test_hard_mode_and_every_written_clue_is_true(rows):
    for row in rows:
        answer, history = row["answer"], []
        (steps, fbs) = turns(row)
        assert steps[-1][1] == answer and len(steps) == row["guesses"] <= 6
        for k, (think, guess) in enumerate(steps):
            assert consistent(guess, history), f"seed {row['seed']}: {guess} breaks a clue"
            lines = dict(line.split(": ", 1) for line in think.splitlines() if ": " in line and not line.startswith("Try"))
            if history:
                known = lines["Known"].split()
                assert all(c == "_" or answer[i] == c for i, c in enumerate(known))
                for item in filter(None, lines["Misplaced"].split(", ")):
                    if item == "none":
                        continue
                    ch, pos = re.match(r"(\w) \(not ([\w/]+)\)", item).groups()
                    assert ch in answer and all(answer[ORD.index(p)] != ch for p in pos.split("/"))
                if lines["Ruled out"] != "none":
                    assert not set(lines["Ruled out"].split()) & set(answer)
                assert think.splitlines()[-1] == f"{guess} fits every clue."
                assert all(f"{g} → {' '.join(fb)}" in think for g, fb in history)
                tries = [line for line in think.splitlines() if line.startswith("Try ")]
                for line in tries:
                    word, spelled = re.match(r"Try (\w+) = ([a-z](?: [a-z])*): ", line).groups()
                    assert spelled == " ".join(word)
                assert tries[-1].startswith(f"Try {guess} = {' '.join(guess)}: 5 letters ✓, not guessed yet ✓")
                for line in tries[:-1]:  # a rejected near miss, and it really is wrong
                    word = line.split()[1]
                    assert line.endswith("✗")
                    if line.endswith("letters ✗"):
                        assert len(word) != 5 and line.endswith(f": {len(word)} letters ✗")
                    elif line.endswith("already guessed ✗"):
                        assert word in {g for g, _ in history}
                    else:
                        assert len(word) == 5 and not consistent(word, history)
            fb = feedback(guess, answer)
            if k < len(fbs):
                assert fbs[k].splitlines()[1] == " ".join(fb) and fbs[k].splitlines()[0] == " ".join(guess.upper())
            history.append((guess, fb))


@pytest.fixture(scope="module")
def renderer():
    transformers = pytest.importorskip("transformers")
    from renderers.base import create_renderer
    from renderers.configs import PrimeQwen3RendererConfig
    tok = transformers.AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-1.7B")
    return tok, create_renderer(tok, PrimeQwen3RendererConfig())


def render(renderer, messages):
    from renderers.base import build_training_sample
    return build_training_sample(renderer, messages, role_to_mask=None, tools=[], content_sft_roles=None,
                                 ensure_final_stop=True, message_loss_mask=None)


def test_renders_within_seq_len_with_every_turn_trained(rows, renderer):
    tok, r = renderer
    think = tok.convert_tokens_to_ids("<think>")
    for row in rows[:200]:
        s = render(r, row["prompt"] + row["completion"])
        ids = list(s.token_ids)
        assert len(ids) <= 1024, "sft_cot.toml uses seq_len 1024"
        trained = [i for i, m in zip(ids, s.loss_mask) if m]
        assert trained.count(think) == row["guesses"], "<think> is one special token per trained turn"


def test_assistant_turns_render_like_sampled_rl_turns(renderer):
    """Rebuilt as `<think>\\n…\\n</think>\\n[word]`, real RL turns render to the sampled text."""
    tok, r = renderer
    files = sorted(glob.glob(str(ROOT / "artifacts/*/results/train_samples/wordle-rl.jsonl")))
    if not files:
        pytest.skip("no pulled training traces")
    norm = lambda s: re.sub(r"\s+", "", s)  # sampled whitespace varies (e.g. before </think>); stored reasoning is stripped
    checked = 0
    for line in list(open(files[-1]))[:30]:
        for trace in json.loads(line)["traces"]:
            msgs, ids = [], []
            for node in trace["nodes"]:
                m = {k: v for k, v in node["message"].items() if v is not None}
                if m["role"] == "assistant":
                    rc = m.pop("reasoning_content", None)
                    m["content"] = ("<think>\n" + rc + "\n</think>\n" if rc is not None else "") + m.get("content", "")
                msgs.append(m)
                ids.extend(node["token_ids"])
            if any(c["finish_reason"] != "stop" for c in trace["calls"]):
                continue  # cut-off turns have no closing tag in the sampled text
            assert norm(tok.decode(list(render(r, msgs).token_ids))) == norm(tok.decode(ids))
            checked += 1
    assert checked >= 10
