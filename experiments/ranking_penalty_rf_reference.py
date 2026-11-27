"""Run the dashboard's Random Forest ranking on the same 512-D folds."""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ranking_seed_sweep import SEEDS, TASKS, digest, generate_cases
from train_ranking_penalty_sweep import DIMENSION, write_json

RF_ESTIMATORS = 500
RF_SEED = 0xCAFE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for fold, case in enumerate(generate_cases(args.seed, task, DIMENSION)):
            path = args.output / f"{task}-fold{fold}.json"
            identity = dict(dimension=DIMENSION, data_seed=args.seed, task=task, fold=fold,
                            random_state=RF_SEED, n_estimators=RF_ESTIMATORS, data_sha256=digest(case))
            if path.exists():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if any(stored[key] != value for key, value in identity.items()):
                    raise ValueError(f"Saved baseline differs from requested fold: {path}")
                continue
            train = case["train"]
            x_train = case["x"][train][:, case["permutation"]]
            x_train = (x_train - x_train.mean(0)) / (x_train.std(0) + 1e-8)
            forest = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RF_SEED, n_jobs=2)
            forest.fit(x_train, case["y"][train])
            score = forest.feature_importances_
            k = len(case["support"])
            selected = np.argsort(-score, kind="stable")[:k].tolist()
            recall = np.isin(selected, case["support"]).sum().item() / k
            write_json(path, dict(**identity, selected=selected, recall=recall))
        print(f"Random Forest seed={args.seed} {task} finished", flush=True)


if __name__ == "__main__":
    main()
