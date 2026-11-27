"""Load only the audited baseline definitions from a fixed local Git object.

No source substitution, third-party dependency installation, or branch checkout.
The original training function runs unchanged, on CPU, with its legacy loader.
"""

import ast
import hashlib
from pathlib import Path
import subprocess
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

COMMIT = "555ba11e8872bd6618cd721bd3b08f7128db9e6b"


def load_reference():
    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                   "show", f"{COMMIT}:src/admm_input_group_wrapper.py"], cwd=root)
    source = raw.decode("utf-8")
    wanted = {"ColumnNormalizedLinear", "GatedFeatureSelectionMLP", "_get_feature_penalty_scores",
              "soft_thresholding", "safe_norm", "safe_cbrt", "solve_cubic_ratio_norm",
              "solve_cubic_paper", "_train_input_group"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted

    def training_set(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    namespace = dict(torch=torch, nn=nn, np=np, Optional=Optional, DataLoader=DataLoader,
                     TrainingSet=training_set, DEFAULT_SADMM_LATENT_SIZE=32,
                     DEFAULT_SADMM_HIDDEN_LAYERS=2, DEFAULT_SADMM_DROPOUT=0.04308691548552568,
                     DEFAULT_SADMM_ACTIVATION="mish")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{COMMIT}:baseline", "exec"), namespace)
    return namespace, hashlib.sha256(raw).hexdigest()


def train_reference(namespace, x, y, *, seed=0, epochs=500, warmup=120):
    mean, std = x.mean(0), x.std(0) + 1e-8
    scaled = (x - mean) / std
    torch.manual_seed(seed)
    model = namespace["GatedFeatureSelectionMLP"](x.shape[1], 2)
    namespace["_train_input_group"](
        model, scaled, y, 2, lr=0.005, C=0.05, epochs=epochs,
        warmup_epochs=warmup, batch_size=64, rho_init=50., device="cpu",
        use_ratio_norm=True, use_admm=True, n_features=x.shape[1],
    )
    return model
