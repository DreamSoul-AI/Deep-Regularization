"""Paired ranking adapters for additional published feature selectors.

Author sources live in --assets; use the frozen source manifests alongside results.
This transfers selectors to the synthetic ranking benchmark, not the papers' full
hyperparameter-search or downstream evaluation protocols.
"""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from baseline_suite import metrics


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_model(method, x, y, k, assets, output, case):
    if method == 'univariate':
        return np.nan_to_num(f_classif(x, y)[0])
    if method == 'lasso':
        model = LogisticRegression(penalty='l1', solver='saga', C=1,
                                   random_state=1, max_iter=10000).fit(x, y)
        return np.linalg.norm(model.coef_, axis=0)
    if method == 'localized_lasso':
        selector = load_module(assets/'localized-py3.py', 'localized').LocalizedLasso()
        graph = kneighbors_graph(x, 5, mode='connectivity', include_self=False)
        graph = ((graph+graph.T)*0.5).toarray()
        selector.fit_regression(x.T, y.astype(float), graph)
        np.save(output/(case+'-localized-weights.npy'), selector.W)
        return np.linalg.norm(selector.W, axis=1)
    if method == 'e2efs':
        wrapper = load_module(assets/'legacy/src/e2efs_wrapper.py', 'legacy_e2e')
        a, b, c, d = train_test_split(x, y, test_size=.2, random_state=0xCAFE, stratify=y)
        with contextlib.chdir(output.resolve()):
            selector = wrapper._fit_selector(k, a, c, b, d, batch_size=64, max_epochs=50)
        return np.asarray(selector.get_mask()).reshape(-1)
    if method == 'cae_keras':
        import importlib
        import tensorflow as tf
        for name in ['keras','keras.backend','keras.layers','keras.callbacks',
                     'keras.initializers','keras.optimizers']:
            sys.modules[name] = importlib.import_module(name.replace('keras','tf_keras',1))
        import tf_keras as keras
        from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
        keras.backend.clear_session()
        tf.random.set_seed(0xCAFE)
        def decoder(z):
            z = keras.layers.GaussianNoise(.1)(z)
            for _ in range(2):
                z = keras.layers.Dense(32)(z)
                z = keras.layers.Dropout(.2)(z)
                z = keras.layers.LeakyReLU(alpha=.2)(z)
            return keras.layers.Dense(1, activation='sigmoid')(z)
        selector = ConcreteAutoencoderFeatureSelector(k, decoder, num_epochs=300,
                    start_temp=10, min_temp=.01, learning_rate=.0001, tryout_limit=1)
        selector.fit(x, y)
        return selector.get_mask()
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--tasks',nargs='+',choices=['xor','ring','ring+xor','ring+xor+sum'],
                   default=['xor','ring','ring+xor','ring+xor+sum'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.output/(args.method+'.jsonl')
    done = set()
    if result.exists():
        done = {(r['task'],r['fold']) for r in map(json.loads,result.read_text().splitlines()) if r['status']=='ok'}
    torch.set_num_threads(1)
    cases = [(t,f) for t in args.tasks for f in range(6)]
    for task, fold in cases[args.start:args.start+args.limit]:
        if (task,fold) in done:
            continue
        np.random.seed(0xCAFE)
        torch.manual_seed(0xCAFE)
        data = np.load(args.cases/f'{task}-fold{fold}.npz')
        raw = data['x'][:,data['permutation']][data['train']]
        x = (raw-raw.mean(0))/(raw.std(0)+1e-8)
        y = data['y'][data['train']]
        record = dict(task=task, fold=fold, method=args.method)
        start = time.perf_counter()
        with (args.output/f'{task}-{fold}-{args.method}.log').open('a',encoding='utf-8') as log:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    scores = score_model(args.method,x,y,len(data['support']),args.assets,args.output,f'{task}-{fold}')
                record.update(status='ok', seconds=time.perf_counter()-start,
                              metrics=metrics(scores,data['support'],expected_dim=x.shape[1]),scores=np.asarray(scores).tolist())
            except Exception as error:
                traceback.print_exc(file=log)
                record.update(status='failed',error=str(error))
        with result.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(task,fold,args.method,record['status'],record.get('metrics',record.get('error')),flush=True)


if __name__ == '__main__':
    main()
