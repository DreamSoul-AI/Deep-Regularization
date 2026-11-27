"""Run legacy ranking baselines against saved, paired reproduction cases.

Use the project-local baseline environment and a staged fixed legacy src package.
No dataset regeneration, support-based tuning, or downstream refit is performed.
"""

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Keep this comparison on CPU; do not contend with unrelated active GPU jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ATTRIBUTIONS = ["Saliency", "InputXGradient", "IG_noMul", "SmoothGrad", "GuidedBackprop",
                "DeepLift", "Deconvolution", "FeatureAblation", "FeaturePermutation", "ShapleyValueSampling"]


def metrics(scores, support):
    # Adapters supply their native importance convention. Relief is signed:
    # negative relevance must not become positive importance through abs().
    values = np.asarray(scores).reshape(-1)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid feature score shape/values: {values.shape}")
    truth = np.zeros(128, dtype=int)
    truth[support] = 1
    order = np.argsort(-values, kind="stable")
    k = len(support)
    return dict(recall_at_k=float(truth[order[:k]].sum()/k),
                average_precision=float(average_precision_score(truth, values)),
                auroc=float(roc_auc_score(truth, values)), top_k=order[:k].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.method == "stg":
        # STG 0.1.2 uses aliases removed from collections in Python 3.10.
        import collections
        import collections.abc
        for alias in ("Sequence", "Mapping", "Set"):
            if not hasattr(collections, alias):
                setattr(collections, alias, getattr(collections.abc, alias))
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.stage.resolve()))
    core = importlib.import_module("src.core")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing an undocumented CPU fallback")
    torch.set_num_threads(1)
    seed = 0xCAFE
    result = args.output/(args.method+".jsonl")
    done = set()
    if result.exists():
        done = {(r["task"], r["fold"], r["method"]) for r in
                (json.loads(l) for l in result.read_text().splitlines()) if r["status"]=="ok"}
    def save(record):
        with result.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        print(record["task"], record["fold"], record["method"], record["status"],
              record.get("metrics",record.get("error")),flush=True)
    for task in ["xor", "ring", "ring+xor", "ring+xor+sum"]:
        for fold in range(6):
            case_index = ["xor", "ring", "ring+xor", "ring+xor+sum"].index(task)*6+fold
            if not args.start <= case_index < args.start+args.limit:
                continue
            names = ATTRIBUTIONS if args.method=="attributions" else [args.method]
            if all((task,fold,name) in done for name in names):
                continue
            data = np.load(args.cases/f"{task}-fold{fold}.npz")
            train, test = data["train"], data["test"]
            raw = data["x"][:,data["permutation"]]
            mean, std = raw[train].mean(0), raw[train].std(0)+1e-8
            x, xt = (raw[train]-mean)/std, (raw[test]-mean)/std
            y = data["y"][train]
            support = data["support"]
            np.random.seed(seed)
            torch.manual_seed(seed)
            calibration = x[np.linspace(0,len(x)-1,128).astype(int)]
            # Reconstruct original synthetic knockoffs with the original generator.
            rng = np.random.get_state()
            np.random.seed(0)
            _, knockoffs, _ = importlib.import_module("src.data").generate_dataset(task,1000,128)
            np.random.set_state(rng)
            knockoffs = 2.0*knockoffs[:,data["permutation"]]-1.0
            knockoff_train = (knockoffs[train]-mean)/std
            knockoff_calibration = knockoff_train[np.linspace(0,len(x)-1,128).astype(int)]
            start = time.perf_counter()
            with (args.output/f"{task}-{fold}-{args.method}.log").open("a",encoding="utf-8") as log:
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        wrapper = None
                        if args.method=="attributions":
                            wrapper = core.NNwrapper.create(task,128,2)
                            wrapper.fit(x,y)
                            torch.save(wrapper.model.state_dict(),args.output/f"{task}-{fold}-nn.pt")
                    for name in names:
                        if (task,fold,name) in done:continue
                        try:
                            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                                if wrapper is not None:
                                    score = core.run_fa_method(wrapper,calibration.astype(np.float32),name)
                                elif name=="treeshap":
                                    import shap
                                    clf=core.RandomForestClassifier(n_estimators=500,n_jobs=2,random_state=seed)
                                    clf.fit(x,y)
                                    attribution=np.asarray(shap.TreeExplainer(clf).shap_values(calibration))
                                    # New SHAP returns [sample, feature, class], old versions a list per class.
                                    if attribution.shape==(128,128,2): attribution=attribution[:,:,1]
                                    elif attribution.shape==(2,128,128): attribution=attribution[1]
                                    else: raise ValueError(f"Unexpected SHAP shape: {attribution.shape}")
                                    score=np.abs(attribution).mean(0)
                                elif name=="xgboost":
                                    from xgboost import XGBClassifier
                                    clf=XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.3,n_jobs=2,random_state=seed)
                                    clf.fit(x,y);score=clf.feature_importances_
                                elif name=="stg":
                                    _,_,score,_=core.run_stg_fs(x,y,calibration,2,device="cpu")
                                elif name=="fsnet":
                                    # The legacy wrapper's second 2k fit only serves separate 2k metrics/predictions.
                                    selector=core.FSNet(core.Model(len(support),2),128,30,len(support),2)
                                    selector.fit(x,y)
                                    score=selector.get_feature_importances()
                                else:
                                    # RF parallelism bounded; statistical model unchanged.
                                    if name=="rf":
                                        clf=core.RandomForestClassifier(n_estimators=500,n_jobs=2,random_state=seed)
                                        clf.fit(x,y);score=clf.feature_importances_
                                    else:
                                        _,_,score,_=core.run_fs_method(task,name,x,knockoff_train,y,calibration,
                                                                      knockoff_calibration,len(support))
                            save(dict(task=task,fold=fold,method=name,status="ok",seed=seed,device=args.device,
                                      seconds=time.perf_counter()-start,metrics=metrics(score,support),
                                      scores=np.asarray(score).reshape(-1).tolist()))
                        except Exception as error:
                            traceback.print_exc(file=log)
                            save(dict(task=task,fold=fold,method=name,status="failed",error=str(error)))
                except Exception as error:
                    traceback.print_exc(file=log)
                    for name in names:
                        save(dict(task=task,fold=fold,method=name,status="failed",error=str(error)))


if __name__=="__main__":
    main()
