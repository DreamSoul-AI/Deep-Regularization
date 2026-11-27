"""Paired ranking-only pilot on the original XOR/Ring generators."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

# Initialize the PyTorch native runtime first on the existing Windows environment.
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_selection import RankingConfig, fit_ranking
from src.feature_selection_data import generate_dataset


def ranking_metrics(score, support):
    score = np.asarray(score)
    if not np.isfinite(score).all():
        raise FloatingPointError("Non-finite ranking score")
    truth = np.zeros(len(score), dtype=int)
    truth[support] = 1
    order = np.argsort(-score, kind="stable")
    k = len(support)
    return dict(recall_at_k=float(truth[order[:k]].sum() / k),
                recall_at_2k=float(truth[order[:2*k]].sum() / k),
                average_precision=float(average_precision_score(truth, score)),
                auroc=float(roc_auc_score(truth, score)) if k < len(score) else None,
                unique_scores=int(len(np.unique(score))), top_k=order[:k].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["xor", "ring", "ring+xor", "ring+xor+sum"], default=["xor", "ring"])
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; prior results are not overwritten")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = RankingConfig(epochs=args.epochs, warmup_epochs=args.warmup)
    modes = ["column_norm", "taylor_first", "taylor_fisher"]
    manifest = dict(arguments={**vars(args), "output": str(args.output)},
                    config=asdict(config), device="cpu", torch=torch.__version__,
                    python=sys.version, numpy=np.__version__,
                    primary_readout="ranking", downstream_refits=False,
                    source_baseline="555ba11e8872bd6618cd721bd3b08f7128db9e6b",
                    scope="128-dimension paired ranking pilot; no search or test-label selection",
                    files={})
    root = Path(__file__).resolve().parents[1]
    for name in ["src/feature_scores.py", "src/feature_selection.py", "src/feature_selection_data.py", "experiments/ranking_taylor_fisher.py"]:
        manifest["files"][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    result_file = args.output / "results.jsonl"
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for task_index, task in enumerate(args.tasks):
        np.random.seed(args.seed + task_index)
        k = {"xor": 2, "ring": 2, "ring+xor": 4, "ring+xor+sum": 6}[task]
        if args.dimensions <= k:
            raise ValueError("Pilot requires more input dimensions than relevant features")
        x, _, y = generate_dataset(task, args.samples, args.dimensions)
        permutation = np.random.default_rng(args.seed + 100 + task_index).permutation(args.dimensions)
        x = x[:, permutation].astype(np.float32)
        support = np.flatnonzero(permutation < k)
        splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(x, y))
        np.savez(args.output/f"{task}-data.npz", x=x, y=y, support=support, permutation=permutation,
                 **{f"train_{i}": train for i, (train, _) in enumerate(splits)},
                 **{f"test_{i}": test for i, (_, test) in enumerate(splits)})
        for fold, (train, _) in enumerate(splits):
            for mode in modes:
                start = time.perf_counter()
                scores, state = fit_ranking(torch.from_numpy(x[train]), torch.from_numpy(y[train]),
                                            mode=mode, seed=args.seed + fold, config=config)
                elapsed = time.perf_counter() - start
                checkpoint = f"{task}-fold{fold}-{mode}.pt"
                torch.save(state, args.output/checkpoint)
                record = dict(task=task, fold=fold, mode=mode, elapsed_seconds=elapsed,
                              checkpoint=checkpoint, final_task_loss=state["trace"][-1]["task_loss"],
                              scores={name: score.numpy().tolist() for name, score in scores.items()},
                              metrics={name: ranking_metrics(score.numpy(), support) for name, score in scores.items()})
                with result_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(task, fold, mode, record["metrics"]["ranking"], f"{elapsed:.1f}s", flush=True)
    print("Completed ranking-only pilot.", flush=True)


if __name__ == "__main__":
    main()
