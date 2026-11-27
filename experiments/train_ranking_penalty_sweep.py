"""Fit the original 512-feature model along a small C path."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ranking_seed_sweep import CONFIG, MODEL_SEED, SEEDS, TASKS, digest, fit_ranking, generate_cases

PENALTIES = (0.0125, 0.025, 0.05, 0.1)
DIMENSION = 512


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--penalty", type=float, required=True, choices=PENALTIES)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = replace(CONFIG, penalty=args.penalty)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = ("experiments/train_ranking_penalty_sweep.py", "experiments/ranking_seed_sweep.py",
               "src/feature_selection.py", "src/feature_selection_data.py", "src/feature_scores.py")
    identity = dict(dimension=DIMENSION, data_seed=args.seed, model_seed=MODEL_SEED, config=asdict(config),
                    source_sha256={name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                   for name in sources})
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Resume requires identical sources and training configuration: {output}")
    write_json(manifest, identity)
    for task in TASKS:
        for fold, case in enumerate(generate_cases(args.seed, task, DIMENSION)):
            path = output / f"{task}-fold{fold}.json"
            if path.exists():
                continue
            train = case["train"]
            x = case["x"][train][:, case["permutation"]]
            _, state = fit_ranking(torch.from_numpy(x), torch.from_numpy(case["y"][train]), mode="column_norm",
                                   seed=MODEL_SEED, config=config)
            checkpoint = path.with_suffix(".pt")
            torch.save(state, checkpoint)
            write_json(path, dict(data_sha256=digest(case), checkpoint_sha256=sha256(checkpoint)))
        print(f"C={args.penalty:g} seed={args.seed} {task} finished", flush=True)


if __name__ == "__main__":
    main()
