"""Keep Aq fixed and compare column-wise and uniform final soft thresholds."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np

from ranking_fixed_threshold import SOFT_MODES, TASKS, save_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((args.reference / "cases.json").read_text(encoding="utf-8"))
    scores_path = args.reference / "verified_results.jsonl"
    prior = {(row["task"], row["fold"], row["method"]): row
             for row in map(json.loads, scores_path.read_text(encoding="utf-8").splitlines())}
    modes = [mode for mode in SOFT_MODES if mode.startswith("column_norm_")]
    results = []
    comparison = []
    for task in TASKS:
        for fold in range(6):
            name = f"{task}-fold{fold}"
            checkpoint = args.cases / f"{name}-column_norm.pt"
            case_path = args.cases / f"{name}.npz"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["checkpoint_verification"][f"{name}-gate_only"]["checkpoint_sha256"]
            assert hashlib.sha256(case_path.read_bytes()).hexdigest() == identity[name]["sha256"]
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            assert state["mode"] == "column_norm" and state["config"] == manifest["config"]
            with np.load(case_path) as case:
                for mode in modes:
                    raw = save_readout(state, mode, case, apply_soft=False)
                    divided = save_readout(state, mode, case, apply_soft=True)
                    uniform = save_readout(state, mode, case, apply_soft=True, divide_by_a=False)
                    for suffix, recomputed in (("raw", raw), ("soft", divided)):
                        reference = prior[(task, fold, f"readout_{mode}_{suffix}")]
                        assert all(reference[key] == value for key, value in recomputed.items()), (name, mode, suffix)
                    results.append(dict(task=task, fold=fold, method=mode, status="ok", **uniform))
                    comparison.append(dict(task=task, fold=fold, mode=mode, final_rho=state["rho"],
                        divided_positive_recall=divided["positive_recall"], uniform_positive_recall=uniform["positive_recall"],
                        raw_positive_recall=raw["positive_recall"], uniform_nonzero=uniform["nonzero"],
                        divided_nonzero=divided["nonzero"],
                        top_k_changed=set(uniform["metrics"]["top_k"]) != set(divided["metrics"]["top_k"])))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in results), encoding="utf-8")
    (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2)+'\n', encoding="utf-8")
    print(f"Verified {len(results)} paired Aq readouts")


if __name__ == "__main__":
    main()
