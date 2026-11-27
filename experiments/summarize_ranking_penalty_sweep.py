"""Verify the 512-D C sweep and aggregate same-fold feature rankings."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

import torch
import numpy as np

from ranking_penalty_rf_reference import RF_ESTIMATORS, RF_SEED
from ranking_seed_sweep import CONFIG, MODEL_SEED, MODES, SEEDS, TASKS, digest, generate_cases, summarize_readouts
from src.feature_scores import ranking_score
from summarize_ranking_seed_sweep import load, paired_summary
from train_ranking_penalty_sweep import DIMENSION, PENALTIES, sha256, write_json


def readout(score, support):
    top = np.argsort(-score, kind="stable")[:len(support)]
    selected = top[score[top] > 0]
    return dict(selected=selected.tolist(), retained=int(np.count_nonzero(score)),
                recall=np.isin(selected, support).sum().item() / len(support))


def fold_rankings(folder, name, penalty, case, data_sha256, reference=False):
    path = folder / f"{name}.pt"
    receipt = load(path.with_suffix(".json"))
    assert receipt["checkpoint_sha256"] == sha256(path) and receipt["data_sha256"] == data_sha256, path
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert state["mode"] == "column_norm" and state["seed"] == MODEL_SEED, path
    assert state["config"] == asdict(replace(CONFIG, penalty=penalty)), path
    assert state["prox_input"] == "default" and state["threshold_rule"] == "adaptive", path
    assert len(state["trace"]) == CONFIG.epochs and np.isfinite([r["task_loss"] for r in state["trace"]]).all()
    x = case["x"][case["train"]][:, case["permutation"]]
    np.testing.assert_array_equal(x.mean(0), state["mean"].numpy())
    np.testing.assert_array_equal(x.std(0) + 1e-8, state["std"].numpy())
    support = case["support"]
    if reference:
        assert receipt["support"] == support.tolist() and receipt["scores"] == summarize_readouts(state, case), path
    a = state["model"]["layers.1.weight"].norm(dim=0)
    cutoff = (penalty / (a + 1e-8)).clamp(1e-8, 0.5) / state["rho"]
    rankings = {"raw": {}, "soft": {}}
    for mode in MODES:
        score = ranking_score(state["model"], mode, state["first_moment"], state["second_moment"])
        assert score.shape == (DIMENSION,) and torch.isfinite(score).all(), path
        rankings["raw"][mode] = readout(score.numpy(), support)
        rankings["soft"][mode] = readout((score - cutoff).clamp_min(0).numpy(), support)
        if reference:
            assert rankings["soft"][mode]["recall"] == receipt["scores"][mode]["recall"], path
    assert torch.isfinite(state["z"]).all(), path
    return dict(checkpoint_sha256=receipt["checkpoint_sha256"], rho=state["rho"],
                active_z=int(state["z"].count_nonzero()), **rankings)


def summarize(folds):
    gate, first, second, combined = MODES
    by_seed = {}
    for seed in SEEDS:
        tasks = {}
        for task in TASKS:
            rows = [r for r in folds if (r["seed"], r["task"]) == (seed, task)]
            assert len(rows) == 6
            tasks[task] = {kind: {mode: mean(r[kind][mode]["recall"] for r in rows) for mode in MODES}
                           for kind in ("raw", "soft")}
            tasks[task]["rf"] = mean(r["rf"]["recall"] for r in rows)
        by_seed[seed] = {kind: {mode: mean(r[kind][mode] for r in tasks.values()) for mode in MODES}
                         for kind in ("raw", "soft")}
        by_seed[seed].update(tasks=tasks, rf=mean(r["rf"] for r in tasks.values()))
    result = {kind: {mode: mean(r[kind][mode] for r in by_seed.values()) for mode in MODES}
              for kind in ("raw", "soft")}
    result["rf"] = mean(r["rf"] for r in by_seed.values())
    result["mean_active_z"] = mean(row["active_z"] for row in folds)
    result["by_seed"] = by_seed
    result["contrasts"] = {}
    for kind in ("raw", "soft"):
        result["contrasts"][kind] = {
            "second_minus_gate": paired_summary([r[kind][second] - r[kind][gate] for r in by_seed.values()]),
            "combined_minus_gate": paired_summary([r[kind][combined] - r[kind][gate] for r in by_seed.values()]),
            "combined_minus_second": paired_summary([r[kind][combined] - r[kind][second] for r in by_seed.values()]),
            "combined_minus_first": paired_summary([r[kind][combined] - r[kind][first] for r in by_seed.values()]),
            "gate_minus_rf": paired_summary([r[kind][gate] - r["rf"] for r in by_seed.values()]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    repository = Path(__file__).resolve().parents[1]
    for c in PENALTIES:
        if c == CONFIG.penalty:
            continue
        for seed in SEEDS:
            manifest_path = args.runs / f"{c:g}-{seed}" / "manifest.json"
            manifest = load(manifest_path)
            assert (manifest["dimension"], manifest["data_seed"], manifest["model_seed"]) == (DIMENSION, seed, MODEL_SEED)
            assert manifest["config"] == asdict(replace(CONFIG, penalty=c)), manifest_path
            for source, expected in manifest["source_sha256"].items():
                actual = hashlib.sha256((repository / source).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                assert actual == expected, source
    by_penalty = {c: [] for c in PENALTIES}
    for seed in SEEDS:
        for task in TASKS:
            for fold, case in enumerate(generate_cases(seed, task, DIMENSION)):
                name = f"{task}-fold{fold}"
                data_sha256 = digest(case)
                rf_path = args.runs / f"rf-{seed}" / f"{name}.json"
                rf = load(rf_path)
                assert (rf["task"], rf["fold"], rf["data_seed"], rf["dimension"]) == (task, fold, seed, DIMENSION)
                assert rf["random_state"] == RF_SEED and rf["n_estimators"] == RF_ESTIMATORS
                assert rf["data_sha256"] == data_sha256 and len(set(rf["selected"])) == len(case["support"]), rf_path
                assert np.isin(rf["selected"], case["support"]).sum().item()/len(case["support"]) == rf["recall"], rf_path
                for c in PENALTIES:
                    reference = c == CONFIG.penalty
                    folder = args.reference / f"{DIMENSION}-{seed}" if reference else args.runs / f"{c:g}-{seed}"
                    rankings = fold_rankings(folder, name, c, case, data_sha256, reference)
                    by_penalty[c].append(dict(seed=seed, fold=fold, task=task, penalty=c, support=case["support"].tolist(),
                                              data_sha256=data_sha256, rf=rf, **rankings))
    args.output.mkdir(parents=True, exist_ok=True)
    records = [row for folds in by_penalty.values() for row in folds]
    write_json(args.output / "summary.json", dict(dimension=DIMENSION, data_seeds=SEEDS, model_seed=MODEL_SEED,
               base_config=asdict(CONFIG), n_models=len(records), rf_folds=len(records)//len(PENALTIES),
               by_penalty={c: summarize(folds) for c, folds in by_penalty.items()}))
    with (args.output / "verified_folds.jsonl").open("w", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row) + "\n")
    print(f"Verified {len(records)} original-training models, with {len(records)//len(PENALTIES)} paired RF folds.")


if __name__ == "__main__":
    main()
