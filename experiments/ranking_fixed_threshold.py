"""512-D left-input ablations with the original column-norm threshold."""

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
from ranking_threshold_ablation import SOFT_MODES
from src.feature_scores import ranking_score
from src.feature_selection import RankingConfig, fit_ranking


def save_readout(state, mode, case, apply_soft, *, divide_by_a=True):
    score = ranking_score(state["model"], mode, state["first_moment"], state["second_moment"])
    if apply_soft:
        penalty = state["config"]["penalty"]
        if divide_by_a:
            a = state["model"]["layers.1.weight"].norm(dim=0)
            penalty = penalty / (a + 1e-8)
        else:
            penalty = score.new_tensor(penalty)
        threshold = penalty.clamp(1e-8, 0.5) / state["rho"]
        score = (score - threshold).clamp_min(0)
    score = score.numpy()
    k = len(case["support"])
    nonzero = int(np.count_nonzero(score))
    stable = np.argsort(-score, kind="stable")[:k]
    return dict(scores=score.tolist(), metrics=metrics(score, case["support"], expected_dim=512),
                nonzero=nonzero, positive_recall=np.isin(stable[score[stable] > 0], case["support"]).sum().item()/k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("readout", "score", "weighted_consensus", "uniform_training"), required=True)
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
    hashes = {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for name in ("src/feature_selection.py", "src/feature_scores.py", "experiments/ranking_fixed_threshold.py")}
    for task, fold in jobs[args.start:args.stop]:
        name = f"{task}-fold{fold}"
        data = args.cases / f"{name}.npz"
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        with np.load(data) as case:
            if case["x"].shape[1] != 512:
                raise ValueError("This run expects 512 features")
            x = torch.from_numpy(case["x"][case["train"]][:, case["permutation"]])
            y = torch.from_numpy(case["y"][case["train"]])
            if args.arm == "uniform_training":
                models = tuple(mode for mode in SOFT_MODES if mode.startswith("column_norm_"))
            else:
                models = SOFT_MODES if args.arm == "score" else ("gate_only",)
            for mode in models:
                output = args.output / f"{name}-{mode}.json"
                identity = dict(data_sha256=data_hash, source_sha256=hashes, config=asdict(config),
                                arm=args.arm, task=task, fold=fold, mode=mode)
                if output.exists():
                    saved = json.loads(output.read_text(encoding="utf-8"))
                    if any(saved[key] != value for key, value in identity.items()):
                        raise ValueError(f"Resume requires identical condition: {output}")
                    continue
                try:
                    if args.arm == "readout":
                        checkpoint = args.cases / f"{name}-column_norm.pt"
                        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                        if state["mode"] != "column_norm" or state["config"] != asdict(config):
                            raise ValueError(f"Unexpected original checkpoint: {checkpoint}")
                    else:
                        uniform_training = args.arm == "uniform_training"
                        _, state = fit_ranking(x, y, mode=mode, seed=0, config=config,
                                               prox_input="score" if uniform_training else args.arm,
                                               threshold_rule="uniform" if uniform_training else "adaptive")
                        checkpoint = output.with_suffix(".pt")
                        torch.save(state, checkpoint)
                    modes = (mode,) if args.arm in ("score", "uniform_training") else SOFT_MODES
                    rows = []
                    for score_mode in modes:
                        for apply_soft in (False, True):
                            method = f"{args.arm}_{score_mode}_{'soft' if apply_soft else 'raw'}"
                            rows.append(dict(task=task, fold=fold, method=method, status="ok",
                                             **save_readout(state, score_mode, case, apply_soft)))
                    result = dict(identity, status="ok", checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  final_rho=state["rho"], results=rows)
                except FloatingPointError as exc:
                    result = dict(identity, status="failed", error=str(exc))
                temp = output.with_suffix(".tmp")
                temp.write_text(json.dumps(result) + "\n", encoding="utf-8")
                temp.replace(output)
                print(name, mode, result["status"], result.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
