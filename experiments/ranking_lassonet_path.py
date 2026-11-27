"""Record the original LassoNet path on the paired 512-D ranking folds."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import torch
import numpy as np
from lassonet import LassoNetClassifier

from ranking_seed_sweep import TASKS, digest, generate_cases
from train_ranking_penalty_sweep import DIMENSION, write_json

NATIVE = dict(hidden_dims=(32, 32), n_iters=(30, 30), batch_size=64, dropout=0, patience=10,
              lambda_start=3.2768, path_multiplier=1.02, tol=0.9999, verbose=False)
SEED = 0xCAFE


def record_path(case, device, lambda_start, path_multiplier):
    train = case["train"]
    x = case["x"][train][:, case["permutation"]]
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    config = {**NATIVE, "lambda_start": lambda_start, "path_multiplier": path_multiplier}
    model = LassoNetClassifier(**config, device=device, random_state=SEED, torch_seed=SEED)
    history = model.path(x, case["y"][train])
    score = model.feature_importances_.numpy()
    final_support = history[-1].selected.numpy()
    if final_support.any() or not np.isfinite(score).all():
        raise ValueError("The path has not resolved every feature's first exit")
    support = case["support"]
    top = np.argsort(-score, kind="stable")[:len(support)]
    rows = [dict(penalty=h.lambda_, active=int(h.selected.sum()), val_loss=h.val_loss)
            for h in history]
    if not all(np.isfinite(r["val_loss"]) for r in rows):
        raise ValueError("Non-finite loss on the LassoNet path")
    return dict(path=rows, first_exit=score.tolist(), top_k=top.tolist(),
                recall=np.isin(top, support).sum().item() / len(support))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=range(6))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--lambda-start", type=float, default=NATIVE["lambda_start"])
    parser.add_argument("--path-multiplier", type=float, default=NATIVE["path_multiplier"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.lambda_start <= 0 or args.path_multiplier <= 1:
        parser.error("The lambda path requires a positive start and a multiplier greater than one")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for task in ((args.task,) if args.task else TASKS):
        for fold, case in enumerate(generate_cases(args.seed, task, DIMENSION)):
            if args.fold is not None and fold != args.fold:
                continue
            path = args.output / f"{task}-fold{fold}.json"
            identity = dict(seed=args.seed, task=task, fold=fold, device=args.device, backend_seed=SEED,
                            lambda_start=args.lambda_start, path_multiplier=args.path_multiplier,
                            lassonet_version=version("lassonet"), data_sha256=digest(case))
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if any(previous[key] != value for key, value in identity.items()):
                    raise ValueError(f"Resume requires the same path specification: {path}")
                continue
            print(f"LassoNet seed={args.seed} {task} fold={fold} started", flush=True)
            scores = record_path(case, args.device, args.lambda_start, args.path_multiplier)
            write_json(path, dict(**identity, support=case["support"].tolist(), **scores))
            print(f"LassoNet seed={args.seed} {task} fold={fold}, steps={len(scores['path'])}, recall={scores['recall']}",
                  flush=True)


if __name__ == "__main__":
    main()
