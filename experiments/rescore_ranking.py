"""Read final threshold scores from the existing 128-D and 512-D checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_suite import metrics
from ranking_score_ablation import TASKS
from src.feature_scores import ranking_score
from src.feature_selection import TAYLOR_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(128, 512), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dim = args.dimension
    dashboard = Path("docs/ranking-dashboard")
    runs = Path(".tmp/all-baselines-512-20261128")
    if dim == 128:
        cases = Path(".tmp/ranking-taylor-fisher/reproduction-128-v2")
        previous = Path("docs/研究记录 Research Record/20261128-baselines-128")
    else:
        cases = runs / "cases"
        previous = dashboard / "512"
    identities = json.loads((previous / "cases.json").read_text(encoding="utf-8"))
    paths = [previous / "verified_results.jsonl"]
    if dim == 512:
        paths.append(dashboard / "score-ablation-512/verified_results.jsonl")
    prior = {(row["task"], row["fold"], row["method"]): row["scores"]
             for file in paths for row in map(json.loads, file.read_text(encoding="utf-8").splitlines())}
    ablation = Path(".tmp/ranking-score-ablation-512")
    threshold = Path(".tmp/ranking-threshold-ablation-512")
    args.results.mkdir(parents=True, exist_ok=True)
    records = []
    checkpoint_hashes = {}
    for task in TASKS:
        for fold in range(6):
            case = f"{task}-fold{fold}"
            data = cases / f"{case}.npz"
            case_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            assert case_sha == identities[case]["sha256"], case
            with np.load(data) as dataset:
                raw = dataset["x"][dataset["train"]][:, dataset["permutation"]]
                public_raw = dataset["x"][:, dataset["permutation"]][dataset["train"]]
                support = dataset["support"]
            assert raw.shape[1] == dim

            def load(path, training_rows=raw):
                relative = path.as_posix()
                checkpoint_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                state = torch.load(path, map_location="cpu", weights_only=True)
                np.testing.assert_array_equal(training_rows.mean(0), state["mean"].numpy())
                np.testing.assert_array_equal(training_rows.std(0) + 1e-8, state["std"].numpy())
                return state

            def save(state, method, score_mode, previous_method=None):
                if previous_method:
                    actual_gate = state["model"]["gate"].abs().tolist()
                    assert actual_gate == prior[(task, fold, previous_method)], (case, previous_method)
                score = ranking_score(state["model"], score_mode, state.get("first_moment"),
                                      state.get("second_moment")).numpy()
                records.append(dict(task=task, fold=fold, method=method, status="ok", dimensions=dim,
                                    scores=score.tolist(), metrics=metrics(score, support, expected_dim=dim),
                                    data_sha256=case_sha, score_mode=score_mode))

            prefix = "ours_" if dim == 128 else "raw_snapshot_"
            training = {}
            for mode in ("column_norm", "taylor_first", "taylor_fisher"):
                training[mode] = load(cases / f"{case}-{mode}.pt")
                save(training[mode], prefix + mode, mode, prefix + mode)
            if dim == 512:
                public = load(runs / "public-defaults" / f"{case}.pt", public_raw)
                save(public, "public_512_defaults_rho200", "column_norm", "public_512_defaults_rho200")
                weighted = ("column_norm_gate", *(f"column_norm_{mode}" for mode in TAYLOR_MODES))
                for mode in ("gate_only", "taylor_second", *weighted):
                    matches = list(ablation.glob(f"*/{case}-{mode}.pt"))
                    if len(matches) != 1:
                        raise ValueError(f"Expected one checkpoint for {case} and {mode}")
                    training[mode] = load(matches[0])
                    save(training[mode], mode, mode, mode)
                (gate_adaptive,) = threshold.glob(f"part*/{case}-gate-adaptive.pt")
                training["gate_adaptive"] = load(gate_adaptive)
                # Evaluate both scores on each soft-training model, independently of its threshold score.
                for mode in ("gate_adaptive", *TAYLOR_MODES, *weighted):
                    base = mode.removeprefix("column_norm_")
                    a_mode = "column_norm_gate" if base in ("gate", "gate_adaptive") else f"column_norm_{base}"
                    for suffix, score_mode in (("q", base), ("aq", a_mode)):
                        save(training[mode], f"soft_{mode}__rank_{suffix}", score_mode)
    (args.results / "results.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in records), encoding="utf-8")
    manifest = {"dimension": dim, "ranking_rule": "Final model threshold score, descending",
                "checkpoint_sha256": checkpoint_hashes, "evaluations": len(records),
                "source_sha256": {name: hashlib.sha256(Path(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for name in ("src/feature_scores.py", "experiments/rescore_ranking.py")}}
    (args.results / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"{dim} dimensions: {len(records)} verified readouts from {len(checkpoint_hashes)} checkpoints")


if __name__ == "__main__":
    main()
