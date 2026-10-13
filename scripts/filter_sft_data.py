"""Drop SFT games whose answer is a held-out eval word (runs on the instance).

willcb/V3-wordle has 1000 games; 98 have one of our 18 eval words as the answer.
Writes ~/wordle-rl/data/v3-wordle-heldout/train.parquet, which configs/wordle/sft.toml
loads by path (datasets.load_dataset on a local dir).

    uv run python ~/wordle-rl/scripts/filter_sft_data.py      # from ~/prime-rl
"""

import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path.home() / "wordle-rl"
held_out = set(json.loads((ROOT / "configs" / "heldout_split.json").read_text())["held_out_words"])
ds = load_dataset("willcb/V3-wordle", split="train")
kept = ds.filter(lambda row: str(row["answer"]).strip().lower() not in held_out)
out = ROOT / "data" / "v3-wordle-heldout"
out.mkdir(parents=True, exist_ok=True)
kept.to_parquet(str(out / "train.parquet"))
summary = {"source": "willcb/V3-wordle", "rows": len(ds), "kept": len(kept), "dropped": len(ds) - len(kept),
           "held_out_words": sorted(held_out)}
(ROOT / "results" / "sft_data_filter.json").write_text(json.dumps(summary, indent=1))
print(summary)
assert not {str(a).strip().lower() for a in kept["answer"]} & held_out
