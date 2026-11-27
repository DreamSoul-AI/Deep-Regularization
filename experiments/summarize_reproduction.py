"""Validate and summarize completed paired ranking reproduction artifacts."""

import argparse
import json
from pathlib import Path
import sys

import torch
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_scores import gate_score_moments, score_from_moments
from src.feature_selection import GatedMLP, RankingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    p = args.directory
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (p/"results.jsonl").read_text().splitlines()]
    manifest=json.loads((p/"manifest.json").read_text(encoding="utf-8"))
    tasks = manifest.get("tasks",["xor", "ring", "ring+xor", "ring+xor+sum"])
    dimensions=manifest["dimensions"]
    modes = ["column_norm", "taylor_first", "taylor_fisher"]
    keys = {(r["task"], r["fold"], r["mode"]) for r in rows}
    assert len(rows) == len(keys) == len(tasks)*6*len(modes)
    assert keys == {(task, fold, mode) for task in tasks for fold in range(6) for mode in modes}
    diagnostics = []
    for row in rows:
        state = torch.load(p/row["checkpoint"], weights_only=True, map_location="cpu")
        case = f"{row['task']}-fold{row['fold']}"
        data = np.load(p/f"{case}.npz")
        assert set(data["train"]).isdisjoint(data["test"])
        assert len(data["train"]) + len(data["test"]) == 1000
        k = 2 if row["task"] in ["xor", "ring"] else (4 if row["task"] == "ring+xor" else 6)
        np.testing.assert_array_equal(data["support"], np.flatnonzero(data["permutation"] < k))
        model = GatedMLP(dimensions, config=RankingConfig(**state["config"]))
        model.load_state_dict(state["model"])
        x = data["x"][data["train"]][:, data["permutation"]]
        x = torch.as_tensor((x-state["mean"].numpy()) / state["std"].numpy(), dtype=torch.float32)
        ids = state["calibration_ids"]
        a, b = gate_score_moments(model, x[ids], torch.from_numpy(data["y"][data["train"]])[ids])
        torch.testing.assert_close(a, state["first_moment"], rtol=0, atol=0)
        torch.testing.assert_close(b, state["second_moment"], rtol=0, atol=0)
        np.testing.assert_array_equal(model.gate.detach().abs().numpy(), row["scores"]["gate"])
        for mode in modes[1:]:
            np.testing.assert_array_equal(score_from_moments(a,b,mode).abs().numpy(), row["scores"][mode])
        if row["mode"] == "column_norm":
            original = torch.load(p/f"{case}-original.pt", weights_only=True, map_location="cpu")
            for key, value in original.items():
                torch.testing.assert_close(value, state["model"][key], rtol=0, atol=0)
            diagnostics.append(dict(task=row["task"], fold=row["fold"],
                                    fixed_model_score_rank_correlation=float(spearmanr(row["scores"]["taylor_first"], row["scores"]["taylor_fisher"]).statistic),
                                    second_to_absolute_first_sum=float((0.5*b.sum()) / a.abs().sum().clamp_min(1e-12))))
    summary = []
    for task in tasks:
        for mode in modes:
            selected = [r for r in rows if r["task"] == task and r["mode"] == mode]
            metrics = {}
            for readout in ["gate", "taylor_first", "taylor_fisher"]:
                metrics[readout] = {metric:dict(mean=float(np.mean([r["metrics"][readout][metric] for r in selected])),
                                               std=float(np.std([r["metrics"][readout][metric] for r in selected], ddof=1)))
                                    for metric in ["recall_at_k", "recall_at_2k", "average_precision", "auroc"]}
            summary.append(dict(task=task, mode=mode, metrics=metrics))
    pairs = []
    for reference in ["column_norm", "taylor_first"]:
        for task in tasks:
            a = {r["fold"]:r["metrics"]["gate"]["recall_at_k"] for r in rows if r["task"] == task and r["mode"] == reference}
            b = {r["fold"]:r["metrics"]["gate"]["recall_at_k"] for r in rows if r["task"] == task and r["mode"] == "taylor_fisher"}
            pairs.append(dict(task=task, reference=reference, wins=sum(b[i]>a[i] for i in a),
                              ties=sum(b[i]==a[i] for i in a), losses=sum(b[i]<a[i] for i in a)))
    result = dict(summary=summary, paired_recall=pairs, fixed_model_diagnostics=diagnostics,
                  verified=dict(original_parameter_parity_cases=len(tasks)*6,
                                reloaded_checkpoints=len(rows)+len(tasks)*6,
                                recomputed_score_moments=len(rows), train_test_disjoint=True))
    (p/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("task | column norm | first | first+Fisher")
    for task in tasks:
        print(task, [next(s["metrics"]["gate"]["recall_at_k"]["mean"] for s in summary if s["task"]==task and s["mode"]==mode) for mode in modes])
    print("average", [float(np.mean([s["metrics"]["gate"]["recall_at_k"]["mean"] for s in summary if s["mode"]==mode])) for mode in modes])
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
