"""Verify this portable document/evidence bundle using only Python stdlib."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    index = {}
    for row in manifest['sources']:
        path = ROOT / row['snapshot']
        assert path.resolve().is_relative_to(ROOT)
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], row['original_path']
        index[row['original_path']] = path
    for relative, digest in manifest['artifacts_sha256'].items():
        assert sha(ROOT / relative) == digest, relative

    def source(suffix):
        matches = [path for original, path in index.items() if original.endswith(suffix)]
        assert len(matches) == 1, suffix
        return json.loads(matches[0].read_text(encoding='utf-8'))

    benchmark = source('/docs/reports/meeting_20260918/benchmark_evidence.json')
    assert len(benchmark['table_128']) == 25
    assert len(benchmark['plot_rows']) == 476
    assert len({(v['method'], v['task'], v['p']) for v in benchmark['plot_rows']}) == 476
    means = [row['macro_recall'] for row in benchmark['table_128']]
    assert all(a + 1e-12 >= b for a, b in zip(means, means[1:]))
    table = (ROOT / 'baseline_128_table.md').read_text(encoding='utf-8')
    for row in benchmark['table_128']:
        assert row['markdown'] in table
        assert abs(sum(row['recalls']) / 4 - row['macro_recall']) < 1e-12
    wn = source('/results/main/synthetic_m_sweep_weightnorm_current_1000_legacywn_highm.json')
    poly = source('/results/ablation/run8_diagnostics_20260510_full_20260510_142920.json')
    grid = source('/results/ablation/five_version_full_grid_r62_20260915/full_grid.json')
    old_rows = {(v['method'], v['task'], v['p']): v for v in grid['rows']}
    poly_counts = {}
    for task, k in [('xor', 2), ('ring', 2), ('ring+xor', 4), ('ring+xor+sum', 6)]:
        folds = poly['results']['warmup0'][task]['folds']
        assert len(folds) == 6
        hits = sum(len(set(f['final_eval']['selected']) & set(range(k))) for f in folds)
        poly_counts[task] = (hits, 6 * k)
        assert abs(hits / (6 * k) - poly['results']['warmup0'][task]['summary']['best_k_mean']) < 1e-12
    for row in benchmark['plot_rows']:
        task = row['task'].replace('_', '+')
        if row['method'] == 'WN-legacy':
            folds = wn['results'][task][str(row['p'])]['fold_best_k']
            assert abs(sum(folds) / len(folds) - row['recall']) < 1e-12
        elif row['method'] == 'Legacy-poly2':
            hits, total = poly_counts[task]
            assert row['p'] == 128 and row['recall'] == hits / total
        else:
            assert row == old_rows[row['method'], row['task'], row['p']]

    summary = (ROOT / 'README.md').read_text(encoding='utf-8')
    for task, label in [('xor', 'XOR'), ('ring', 'Ring'), ('ring+xor', 'Ring+XOR'), ('ring+xor+sum', 'Ring+XOR+Sum')]:
        a = source('/results/ablation/legacy_wd_family_r58_20260912/selection/Base/summary.json')['results'][task]
        b = source('/results/ablation/legacy_wd_family_r58_20260912/selection/M31/summary.json')['results'][task]
        hits, total = poly_counts[task]
        assert f"| {label} | {hits}/{total} | {a['coordinates']}/{a['opportunities']} | {b['coordinates']}/{b['opportunities']} |" in summary
    for p in (64, 128, 256, 512, 1024, 2048):
        counts = [next(v for v in grid['table'] if v['arm'] == 'Five-v1' and v['dataset'] == t and v['n_features'] == p)['metrics']['support_top2']['coordinates'] for t in ('xor', 'ring')]
        assert f'| {p} | {counts[0]}/24 | {counts[1]}/24 |' in summary

    link_count = 0
    for name in ('README.md', 'MEETING_REPORT.md', 'TECHNICAL_APPENDIX.md', 'PROVENANCE.md'):
        path = ROOT / name
        doc = path.read_text(encoding='utf-8')
        definitions = dict(re.findall(r'^\[([^\]]+)\]:\s+(.+)$', doc, re.M))
        used = re.findall(r'\[[^\]\n]+\]\[([^\]\n]+)\]', doc)
        assert all(label in definitions for label in used)
        targets = list(definitions.values()) + re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', doc)
        for target in targets:
            if target.startswith(('http:', 'https:', '#')):
                continue
            resolved = (path.parent / target.split('#')[0]).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.is_file(), (name, target)
            link_count += 1
    assert manifest['new_training'] == manifest['new_inference'] == 0
    assert manifest['selector_ledger'] == 4080 and manifest['experiment_status'] == 'paused'
    result = dict(status='pass', sources=len(index), artifacts=len(manifest['artifacts_sha256']),
                  main_document_links=link_count, methods=25, table_cells=100, unique_plot_points=476,
                  new_training=0, new_inference=0, selector_ledger=4080,
                  manifest_sha256=sha(ROOT / 'manifest.json'))
    (ROOT / 'bundle_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
