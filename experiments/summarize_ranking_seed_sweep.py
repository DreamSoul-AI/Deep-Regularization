"""Verify frozen-model readouts and aggregate paired differences by data seed."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch
import numpy as np

from ranking_seed_sweep import CONFIG, DIMENSIONS, MODES, MODEL_SEED, SEEDS, TASKS, digest, generate_cases, summarize_readouts


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_summary(differences):
    values = [round(value, 12) for value in differences]
    return dict(mean=mean(values), stdev=stdev(values), minimum=min(values), maximum=max(values),
                wins=sum(value > 0 for value in values), ties=values.count(0), losses=sum(value < 0 for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    gate, first, second, fisher = MODES
    contrasts = {"first_minus_gate": (first, gate), "fisher_minus_gate": (fisher, gate),
                 "second_minus_first": (second, first), "fisher_minus_first": (fisher, first)}
    paired = {}
    receipts = []
    common = None
    for dimension in DIMENSIONS:
        paired[dimension] = {}
        for seed in SEEDS:
            directory = args.runs / f"{dimension}-{seed}"
            identity = load(directory / "manifest.json")
            assert identity["dimension"] == dimension and identity["seed"] == seed
            assert identity["config"] == asdict(CONFIG) and identity["model_seed"] == MODEL_SEED
            protocol = {k: v for k, v in identity.items() if k not in ("dimension", "seed")}
            if common is None:
                common = protocol
                for file, expected in common["source_sha256"].items():
                    actual = hashlib.sha256((root/file).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    assert actual == expected, file
            assert protocol == common
            task_summary = {}
            for task in TASKS:
                folds = []
                for fold, case in enumerate(generate_cases(seed, task, dimension)):
                    name = f"{task}-fold{fold}"
                    row = load(directory / f"{name}.json")
                    checkpoint = directory / f"{name}.pt"
                    assert (row["dimension"],row["seed"],row["task"],row["fold"]) == (dimension,seed,task,fold)
                    assert row["data_sha256"] == digest(case) and row["support"] == case["support"].tolist()
                    assert row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
                    assert state["mode"] == "column_norm" and state["config"] == asdict(CONFIG) and state["seed"] == MODEL_SEED
                    assert state["prox_input"] == "default" and state["threshold_rule"] == "adaptive"
                    assert len(state["trace"]) == CONFIG.epochs and state["rho"] == row["final_rho"]
                    assert np.isfinite([step["task_loss"] for step in state["trace"]]).all()
                    assert torch.isfinite(state["z"]).all()
                    x_train = case["x"][case["train"]][:,case["permutation"]]
                    np.testing.assert_array_equal(state["mean"].numpy(), x_train.mean(0))
                    np.testing.assert_array_equal(state["std"].numpy(), x_train.std(0) + 1e-8)
                    assert summarize_readouts(state,case) == row["scores"]
                    support = set(row["support"])
                    for score in row["scores"].values():
                        selected = set(score["selected"])
                        assert len(selected) == min(score["nonzero"],len(support))
                        assert len(selected & support)/len(support) == score["recall"]
                    receipts.append(row)
                    folds.append(row)
                assert len(folds) == 6
                task_summary[task] = {mode:{"recall":mean(r["scores"][mode]["recall"] for r in folds),
                    "average_precision":mean(r["scores"][mode]["average_precision"] for r in folds),
                    "all_zero_folds":sum(r["scores"][mode]["nonzero"]==0 for r in folds)} for mode in MODES}
            macro = {mode:mean(task_summary[task][mode]["recall"] for task in TASKS) for mode in MODES}
            macro_ap = {mode:mean(task_summary[task][mode]["average_precision"] for task in TASKS) for mode in MODES}
            paired[dimension][seed] = dict(tasks=task_summary,macro=macro,macro_ap=macro_ap,
                deltas={name:macro[a]-macro[b] for name,(a,b) in contrasts.items()})
    by_dimension = {}
    for dimension, seeds in paired.items():
        by_dimension[dimension] = dict(by_seed=seeds,
            macro={mode:mean(seed["macro"][mode] for seed in seeds.values()) for mode in MODES},
            macro_ap={mode:mean(seed["macro_ap"][mode] for seed in seeds.values()) for mode in MODES},
            tasks={task:{mode:mean(seed["tasks"][task][mode]["recall"] for seed in seeds.values()) for mode in MODES}
                   for task in TASKS},
            contrasts={name:paired_summary([seed["deltas"][name] for seed in seeds.values()]) for name in contrasts})
    args.output.mkdir(parents=True,exist_ok=True)
    summary = dict(dimensions=list(DIMENSIONS), independent_data_seeds=list(SEEDS), model_seed=MODEL_SEED,
        samples_per_task=1000,folds=6,models_verified=len(receipts),rankings_verified=len(receipts)*len(MODES),
        legacy_seed_zero_included=False,protocol=common,by_dimension=by_dimension)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (args.output/"verified_folds.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in receipts),encoding="utf-8")
    print(f"Verified {len(receipts)} original-training checkpoints and {summary['rankings_verified']} paired ranking readouts.")


if __name__ == "__main__":
    main()
