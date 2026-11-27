"""Train one fixed legacy model per fold and compare four paired score readouts."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranking_fixed_threshold import TASKS, save_readout
from src.feature_selection import RankingConfig, fit_ranking
from src.feature_selection_data import generate_dataset

DIMENSIONS = (128, 256, 512, 1024)
SEEDS = (1, 2, 3, 4, 5)
MODES = ("column_norm_gate", "column_norm_taylor_first", "column_norm_taylor_second", "column_norm_taylor_fisher")
SUPPORT_SIZES = dict(zip(TASKS, (2, 2, 4, 6)))
CONFIG = RankingConfig(epochs=500, warmup_epochs=120, protocol="legacy")
MODEL_SEED = 0


def generate_cases(data_seed, task, dimension):
    # Anchor each seed at the historical dimension and keep labels across dimensions.
    np.random.seed(data_seed)
    x, _, y = generate_dataset(task, 1000, 512)
    if dimension > 512:
        noise = np.random.default_rng(np.random.SeedSequence([data_seed, 512, TASKS.index(task)]))
        x = np.column_stack((x, noise.random((len(x), dimension - 512))))
    x = 2.0 * x[:, :dimension] - 1.0
    for train, test in KFold(6).split(x):
        permutation = np.random.permutation(dimension)
        yield dict(x=x, y=y, train=train, test=test, permutation=permutation,
                   support=np.flatnonzero(permutation < SUPPORT_SIZES[task]))


def digest(data):
    sha = hashlib.sha256()
    for name in sorted(data):
        values = np.ascontiguousarray(data[name])
        sha.update(repr((name, values.shape, values.dtype.str)).encode("ascii"))
        sha.update(values.tobytes())
    return sha.hexdigest()


def summarize_readouts(state, case):
    readouts = {}
    for mode in MODES:
        result = save_readout(state, mode, case, apply_soft=True)
        top = result["metrics"]["top_k"]
        positive = [j for j in top if result["scores"][j] > 0]
        readouts[mode] = dict(selected=positive, nonzero=result["nonzero"],
                              recall=result["positive_recall"], average_precision=result["metrics"]["average_precision"])
    return readouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = ("experiments/ranking_seed_sweep.py", "experiments/ranking_fixed_threshold.py",
               "src/feature_selection.py", "src/feature_selection_data.py", "src/feature_scores.py")
    identity = dict(dimension=args.dimension, seed=args.seed, model_seed=MODEL_SEED, config=asdict(CONFIG),
                    torch=torch.__version__, numpy=np.__version__,
                    source_sha256={n:hashlib.sha256((root/n).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for n in sources})
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError("Resume requires the same experiment configuration and sources")
    manifest.write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    for task in TASKS:
        for fold, case in enumerate(generate_cases(args.seed, task, args.dimension)):
            name = f"{task}-fold{fold}"
            path = args.output / f"{name}.json"
            if path.exists():
                continue
            train = case["train"]
            x = case["x"][train][:, case["permutation"]]
            _, state = fit_ranking(torch.from_numpy(x), torch.from_numpy(case["y"][train]), mode="column_norm",
                                   seed=MODEL_SEED, config=CONFIG)
            checkpoint = path.with_suffix(".pt")
            torch.save(state, checkpoint)
            result = dict(task=task, fold=fold, dimension=args.dimension, seed=args.seed, data_sha256=digest(case),
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          support=case["support"].tolist(), final_rho=state["rho"], scores=summarize_readouts(state,case))
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result)+"\n", encoding="utf-8")
            temp.replace(path)
        print(f"{args.dimension}, seed {args.seed}: {task} completed", flush=True)


if __name__ == "__main__":
    main()
