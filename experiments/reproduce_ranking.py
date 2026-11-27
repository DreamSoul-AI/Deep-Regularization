"""Reproduce the fixed historical ranking protocol and compare new scores."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.legacy_ranking_reference import COMMIT, load_reference, train_reference
from experiments.ranking_taylor_fisher import ranking_metrics
from src.feature_selection import RankingConfig, fit_ranking
from src.feature_selection_data import generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--tasks", nargs="+", choices=["xor","ring","ring+xor","ring+xor+sum"],
                        default=["xor","ring","ring+xor","ring+xor+sum"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reference, source_sha = load_reference()
    config = RankingConfig(epochs=500, warmup_epochs=120, protocol="legacy")
    root = Path(__file__).resolve().parents[1]
    files = ["src/feature_selection.py", "src/feature_scores.py", "src/feature_selection_data.py",
             "experiments/reproduce_ranking.py", "experiments/legacy_ranking_reference.py",
             "experiments/ranking_taylor_fisher.py"]
    manifest = dict(source_commit=COMMIT, source_sha256=source_sha, config=asdict(config),
                    seed=0, dimensions=args.dimensions, tasks=args.tasks,samples=1000, folds=6,
                    split="KFold(6), shuffle=False", permutation="per-fold NumPy shuffle, continuous seed-0 state",
                    device="cpu", python=sys.version, torch=torch.__version__, numpy=np.__version__,
                    primary_readout="gate", scoring_readouts="also stored for frozen-model diagnostics",
                    downstream_refits=False, hyperparameter_search=False,
                    files={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
    manifest_path=args.output/"manifest.json"
    if args.resume:
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous[key]!=manifest[key] for key in manifest):
            raise ValueError("Resume source, configuration or data protocol differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result_path=args.output/"results.jsonl"
    done=set()
    if result_path.exists():
        done={(r["task"],r["fold"],r["mode"]) for r in map(json.loads,result_path.read_text().splitlines())}
    for task, k in [("xor", 2), ("ring", 2), ("ring+xor", 4), ("ring+xor+sum", 6)]:
        if task not in args.tasks:
            continue
        np.random.seed(0)
        torch.manual_seed(0)
        x, _, y = generate_dataset(task, 1000, args.dimensions)
        x = 2.0 * x - 1.0
        for fold, (train, test) in enumerate(KFold(6).split(x)):
            permutation = np.arange(args.dimensions)
            np.random.shuffle(permutation)
            support = np.flatnonzero(permutation < k)
            train_x, train_y = x[train][:, permutation], y[train]
            case = f"{task}-fold{fold}"
            case_path=args.output/f"{case}.npz"
            values=dict(x=x,y=y,train=train,test=test,permutation=permutation,support=support)
            if case_path.exists():
                saved=np.load(case_path)
                if any(not np.array_equal(saved[key],value) for key,value in values.items()):
                    raise ValueError("Saved paired data differs: "+case)
            else:
                np.savez(case_path,**values)
            if args.prepare_only or all((task,fold,mode) in done for mode in ["column_norm","taylor_first","taylor_fisher"]):
                continue
            start = time.perf_counter()
            original_path=args.output/f"{case}-original.pt"
            if original_path.exists():
                original=reference["GatedFeatureSelectionMLP"](args.dimensions,2)
                original.load_state_dict(torch.load(original_path,weights_only=True))
                reference_seconds=None
            else:
                original = train_reference(reference, train_x, train_y)
                reference_seconds = time.perf_counter() - start
                torch.save(original.state_dict(),original_path)
            for mode in ["column_norm", "taylor_first", "taylor_fisher"]:
                if (task,fold,mode) in done:
                    continue
                start = time.perf_counter()
                scores, state = fit_ranking(torch.from_numpy(train_x), torch.from_numpy(train_y),
                                            mode=mode, seed=0, config=config)
                elapsed = time.perf_counter() - start
                parity = None
                if mode == "column_norm":
                    parity = max((original.state_dict()[key] - value).abs().max().item()
                                 for key, value in state["model"].items())
                    if parity != 0:
                        raise AssertionError(f"Reference training mismatch in {case}: {parity}")
                record = dict(task=task, fold=fold, mode=mode, seconds=elapsed,
                              original_seconds=reference_seconds if mode == "column_norm" else None,
                              reference_max_parameter_difference=parity,
                              metrics={key: ranking_metrics(value.numpy(), support) for key, value in scores.items()},
                              scores={key: value.numpy().tolist() for key, value in scores.items()},
                              final_task_loss=state["trace"][-1]["task_loss"])
                name = f"{case}-{mode}.pt"
                torch.save(state, args.output/name)
                record["checkpoint"] = name
                with (args.output/"results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(case, mode, record["metrics"]["gate"], f"parity={parity} {elapsed:.1f}s", flush=True)
    print("Completed original-source reproduction and paired ranking.", flush=True)


if __name__ == "__main__":
    main()
