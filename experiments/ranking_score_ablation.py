"""Compare gate and Taylor threshold scores on the saved ranking folds."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_suite import metrics
from src.feature_selection import RANKING_MODES, RankingConfig, fit_ranking


TASKS = ("xor", "ring", "ring+xor", "ring+xor+sum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=RANKING_MODES, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=24)
    args = parser.parse_args()
    jobs = [(task, fold) for task in TASKS for fold in range(6)]
    if not 0 <= args.start < args.stop <= len(jobs):
        parser.error("Require 0 <= start < stop <= 24")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = RankingConfig(epochs=500, warmup_epochs=120, protocol="legacy")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    case_hashes = {}
    for task, fold in jobs:
        case_path = args.cases / f"{task}-fold{fold}.npz"
        case_hashes[case_path.stem] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    manifest = {
        "config": asdict(config), "seed": 0, "device": "cpu", "torch": torch.__version__,
        "numpy": np.__version__, "data_sha256": case_hashes, "modes": args.modes,
        "range": [args.start, args.stop], "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in ("src/feature_selection.py", "src/feature_scores.py",
                         "experiments/ranking_score_ablation.py", "experiments/baseline_suite.py")
        },
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume requires identical code, data and settings")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = args.output / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["task"], row["fold"], row["method"]))
    for task, fold in jobs[args.start:args.stop]:
        name = f"{task}-fold{fold}"
        with np.load(args.cases / f"{name}.npz") as case:
            train = case["train"]
            x = case["x"][train][:, case["permutation"]]
            y = case["y"][train]
            support = case["support"]
        for mode in args.modes:
            if (task, fold, mode) in done:
                continue
            scores, state = fit_ranking(torch.from_numpy(x), torch.from_numpy(y),
                                        mode=mode, seed=0, config=config)
            ranking = scores["ranking"].numpy()
            checkpoint = f"{name}-{mode}.pt"
            torch.save(state, args.output / checkpoint)
            row = dict(task=task, fold=fold, method=mode, status="ok", scores=ranking.tolist(),
                       metrics=metrics(ranking, support, expected_dim=x.shape[1]),
                       gate_metrics=metrics(scores["gate"].numpy(), support, expected_dim=x.shape[1]),
                       dimensions=x.shape[1], data_sha256=case_hashes[name], checkpoint=checkpoint)
            with results.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(task, fold, mode, row["metrics"]["recall_at_k"], flush=True)


if __name__ == "__main__":
    main()
