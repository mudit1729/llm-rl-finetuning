"""Summarize Wordle eval traces: win rate, avg reward, turns, format errors.

Reads verifiers episode records (one JSON episode per line, each with `traces`):
vf-eval's traces.jsonl, or prime-rl's zstd trace stream
(<run>/monitors/file/traces/stream/*.jsonl.zst). For a parent dir, the newest run is used.

    uv run python scripts/summarize_eval.py results/local            # latest run under it
    uv run python scripts/summarize_eval.py path/to/traces.jsonl --json out.json

Definitions (one game = one trace):
  win              game_outcome starts with "Congratulations" (reward 1.0)
  avg_reward       mean game_reward score (1.0 win, else partial credit of the last valid guess)
  avg_turns        mean assistant turns per game, invalid attempts included
  turns_to_solve   mean assistant turns over won games only (includes the one allowed invalid retry)
  guesses_to_solve mean VALID guesses over won games (the usual Wordle score: 1-6)
  format_error_rate  share of model TURNS that did not yield a well-formed move: cut off at the
                   token limit (finish_reason "length"), or rejected for format (no [word], not 5
                   letters). A truncated <think> has no </think>, so vLLM's qwen3 parser returns the
                   whole thought as the answer and the game reads the first bracketed word in it --
                   usually a guess quoted from earlier -- as a "repeat"; those count here.
  format_error     game had >=1 such turn
  invalid_move     game had >=1 rejected move of any kind (format, repeat, not a word)
  invalid_end      game ended on the second invalid move
  truncated_turns  share of model calls that hit the token limit (finish_reason == "length")
"""

import argparse
import io
import json
import re
import statistics
from pathlib import Path

INVALID_PREFIX = "You attempted an invalid move. Reason: "
FEEDBACK = re.compile(r"^[A-Z] [A-Z] [A-Z] [A-Z] [A-Z]\n[GYX] [GYX] [GYX] [GYX] [GYX]")
FORMAT_REASONS = ("wrong format", "must be exactly")


def find_trace_files(paths: list[str], all_runs: bool) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
            continue
        streams = sorted(p.rglob("traces/stream/*.jsonl.zst"))
        if streams:  # prime-rl run dir: every chunk of its stream
            files.extend(streams)
            continue
        found = sorted(p.rglob("traces*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not found:
            raise SystemExit(f"no traces under {p}")
        files.extend(found if all_runs else found[:1])
    return files


def read_lines(f: Path) -> list[str]:
    if f.suffix == ".zst":
        import zstandard
        with zstandard.ZstdDecompressor().stream_reader(f.open("rb")) as r:
            return io.TextIOWrapper(r, encoding="utf-8").read().splitlines()
    return f.read_text().splitlines()


def game_stats(trace: dict) -> dict:
    messages = [n["message"] for n in trace.get("nodes", [])]
    assistant_turns = sum(m["role"] == "assistant" for m in messages)
    outcome = str(trace.get("info", {}).get("game_outcome", ""))
    reasons = [
        m["content"][len(INVALID_PREFIX):]
        for m in messages
        if m["role"] == "user" and isinstance(m.get("content"), str) and m["content"].startswith(INVALID_PREFIX)
    ]
    if outcome.startswith("Invalid Move: "):
        reasons.append(outcome[len("Invalid Move: "):])
    calls = trace.get("calls", [])
    reward = trace.get("rewards", {}).get("game_reward", {}).get("score")
    # Per assistant turn: truncated, or rejected for format (warning in the next user message,
    # or the game-ending outcome after the last turn).
    turn_bad = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        k = len(turn_bad)
        truncated = k < len(calls) and calls[k].get("finish_reason") == "length"
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is not None and nxt["role"] == "user" and isinstance(nxt.get("content"), str):
            reason = nxt["content"][len(INVALID_PREFIX):] if nxt["content"].startswith(INVALID_PREFIX) else ""
        else:
            reason = outcome[len("Invalid Move: "):] if outcome.startswith("Invalid Move: ") else ""
        turn_bad.append(truncated or any(f in reason for f in FORMAT_REASONS))
    return {
        "seed": trace.get("task", {}).get("data", {}).get("info", {}).get("seed"),
        "reward": float(reward) if reward is not None else 0.0,
        "win": outcome.startswith("Congratulations"),
        "turns": assistant_turns,
        "valid_guesses": sum(1 for m in messages if m["role"] == "user" and isinstance(m.get("content"), str)
                             and FEEDBACK.match(m["content"])) + (1 if outcome.startswith("Congratulations") else 0),
        "format_error_turns": sum(turn_bad),
        "format_error": any(turn_bad),
        "invalid_move": bool(reasons),
        "invalid_end": outcome.startswith("Invalid Move"),
        "calls": len(calls),
        "truncated_calls": sum(c.get("finish_reason") == "length" for c in calls),
        "errored": bool(trace.get("errors")) or not trace.get("ok", True),
        "outcome": outcome,
    }


def summarize(games: list[dict]) -> dict:
    n = len(games)
    wins = [g for g in games if g["win"]]
    calls = sum(g["calls"] for g in games)
    mean = lambda xs: statistics.fmean(xs) if xs else float("nan")  # noqa: E731
    return {
        "games": n,
        "win_rate": len(wins) / n,
        "avg_reward": mean([g["reward"] for g in games]),
        "avg_turns": mean([g["turns"] for g in games]),
        "turns_to_solve": mean([g["turns"] for g in wins]),
        "guesses_to_solve": mean([g["valid_guesses"] for g in wins]),
        "format_error_rate": sum(g["format_error_turns"] for g in games) / max(1, sum(g["turns"] for g in games)),
        "format_error_game_rate": mean([g["format_error"] for g in games]),
        "invalid_move_rate": mean([g["invalid_move"] for g in games]),
        "invalid_end_rate": mean([g["invalid_end"] for g in games]),
        "truncated_call_rate": sum(g["truncated_calls"] for g in games) / calls if calls else float("nan"),
        "errored_games": sum(g["errored"] for g in games),
        "distinct_seeds": len({g["seed"] for g in games}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--all-runs", action="store_true", help="read every run under a dir, not just the newest")
    ap.add_argument("--json", help="also write the summary (and per-game rows) to this file")
    ap.add_argument("--games", action="store_true", help="print one line per game")
    args = ap.parse_args()

    files = find_trace_files(args.paths, args.all_runs)
    games = []
    for f in files:
        for line in read_lines(f):
            if line.strip():
                games.extend(game_stats(t) for t in json.loads(line).get("traces", []))
    if not games:
        raise SystemExit("no games found")

    summary = summarize(games)
    print(f"source: {', '.join(map(str, files))}")
    for k, v in summary.items():
        print(f"  {k:<20} {v:.3f}" if isinstance(v, float) else f"  {k:<20} {v}")
    if args.games:
        for g in games:
            print(f"  seed={g['seed']} win={g['win']} reward={g['reward']:.2f} turns={g['turns']} "
                  f"trunc={g['truncated_calls']}/{g['calls']} :: {g['outcome'][:90]!r}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "games": games, "sources": list(map(str, files))}, indent=2))


if __name__ == "__main__":
    main()
