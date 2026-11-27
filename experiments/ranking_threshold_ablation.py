"""Pair soft-thresholded and unthresholded training at identical ranking readouts."""

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
from ranking_score_ablation import TASKS
from src.feature_scores import score_from_moments
from src.feature_selection import RankingConfig, TAYLOR_MODES, fit_ranking

SOFT_MODES = ("gate_adaptive", *TAYLOR_MODES, "column_norm_gate",
              *(f"column_norm_{mode}" for mode in TAYLOR_MODES))


def readout(state, mode):
    score_mode = mode.removeprefix("column_norm_")
    if score_mode in TAYLOR_MODES:
        score = score_from_moments(state["first_moment"], state["second_moment"], score_mode).abs()
    else:
        score = state["model"]["gate"].abs()
    if mode.startswith("column_norm_"):
        score = score * state["model"]["layers.1.weight"].norm(dim=0)
    return score.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soft-checkpoints", type=Path, required=True,
                        help="Prior score comparison shards holding the seven reusable soft-threshold runs")
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
    source_hashes = {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                     for name in ("experiments/ranking_threshold_ablation.py", "src/feature_selection.py",
                                  "src/feature_scores.py", "experiments/ranking_score_ablation.py")}
    for task, fold in jobs[args.start:args.stop]:
        name = f"{task}-fold{fold}"
        data = args.cases / f"{name}.npz"
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        output = args.output / f"{name}.json"
        if output.exists():
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved["data_sha256"] != data_hash or saved["source_sha256"] != source_hashes:
                raise ValueError(f"Resume requires identical data and code for {name}")
            continue
        with np.load(data) as case:
            train = case["train"]
            x = torch.from_numpy(case["x"][train][:, case["permutation"]])
            y = torch.from_numpy(case["y"][train])
            support = case["support"]
        _, no_soft = fit_ranking(x, y, mode="gate_only", seed=0, config=config, soft_threshold=False)
        _, soft_gate = fit_ranking(x, y, mode="gate_adaptive", seed=0, config=config)
        torch.save(no_soft, args.output / f"{name}-no-soft.pt")
        torch.save(soft_gate, args.output / f"{name}-gate-adaptive.pt")
        results = []
        for mode in SOFT_MODES:
            if mode == "gate_adaptive":
                soft = soft_gate
            else:
                paths = list(args.soft_checkpoints.glob(f"scores-*/{name}-{mode}.pt"))
                # The first-order and combined no-A controls retain the original rho50 checkpoints.
                if not paths and mode in ("taylor_first", "taylor_fisher"):
                    paths = [args.cases / f"{name}-{mode}.pt"]
                if len(paths) != 1:
                    raise ValueError(f"Expected one prior soft-threshold model for {name} {mode}")
                soft = torch.load(paths[0], weights_only=True, map_location="cpu")
                if soft["config"] != asdict(config) or soft["mode"] != mode or soft["seed"] != 0:
                    raise ValueError(f"Prior model configuration differs for {name} {mode}")
            for state, threshold in ((no_soft, "none"), (soft, "soft")):
                score = readout(state, mode)
                gate_score = state["model"]["gate"].abs().numpy()
                results.append(dict(task=task, fold=fold, method=f"{threshold}_{mode}", status="ok",
                                    data_sha256=data_hash, scores=score.tolist(),
                                    metrics=metrics(score, support, expected_dim=x.shape[1]),
                                    gate_metrics=metrics(gate_score, support, expected_dim=x.shape[1])))
        receipt = dict(data_sha256=data_hash, source_sha256=source_hashes, results=results)
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        temp.replace(output)
        print(f"{name}: 16 paired ranking records", flush=True)


if __name__ == "__main__":
    main()
