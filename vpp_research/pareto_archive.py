"""保留偏好和失败样本的可审计解集；不将跨种子合并集当作单策略前沿。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from vpp_mappo.pareto import non_dominated
from .suite import feasible
from .environment import OBJECTIVE_VERSION


def solutions(entry, expected_seeds):
    groups = {}
    for row in entry.get('test', []):
        groups.setdefault(tuple(row['preference']), []).append(row)
    records = []
    for preference, rows in groups.items():
        seeds = [r['seed'] for r in rows]
        complete = sorted(seeds) == sorted(expected_seeds)
        eligible = complete and all(feasible(r) for r in rows)
        records.append(dict(training_seed=entry['seed'], family=entry['family'],
            preference=list(preference), scenario_seeds=seeds, complete=complete,
            eligible=eligible, failed_or_infeasible=sum(not feasible(r) for r in rows),
            mean_vector=np.mean([r['vector'] for r in rows], axis=0).tolist() if eligible else None,
            unseen_preference=all(r.get('unseen_preference', False) for r in rows),
            scenarios=[{k: r[k] for k in ('seed', 'vector', 'failed', 'violations', 'ac_violations', 'ac_failed', 'reserve_invalid', 'ev_unmet_kwh', 'dr_backlog_kwh', 'error') if k in r} for r in rows], non_dominated_within_policy_set=False))
    indices = [i for i, r in enumerate(records) if r['eligible']]
    front = non_dominated([records[i]['mean_vector'] for i in indices])
    for j in front:
        records[indices[j]]['non_dominated_within_policy_set'] = True
    return records


def run(runs, output):
    root, out = Path(runs), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    entries, sources = [], []
    for group in ('campaign', 'ablations'):
        path = root / group / 'results.json'
        data = path.read_bytes()
        sources.append(dict(path=f'{group}/results.json', sha256=hashlib.sha256(data).hexdigest()))
        payload = json.loads(data)
        if payload['manifest'].get('reserve_mode','linear') != 'linear' or payload['manifest']['protocol'] != 'c3-budget-campaign-v1' or payload['manifest']['test_seeds'] != [9100, 9101, 9102]:
            raise ValueError('此报告仅适用于 stage9 固定协议')
        entries.extend(payload['entries'])
    records = [r for e in entries for r in solutions(e, [9100, 9101, 9102])]
    report = dict(objective_version=OBJECTIVE_VERSION, scales=[100, 100, 100],
        convention='maximize (-cost, -carbon_kg, linear_reserve_kwh) / 100',
        scope='stage9 protocol only; descriptive test archive; not used for model selection',
        sources=sources, failed_training_entries=[dict(seed=e['seed'], family=e['family']) for e in entries if e['failed']],
        expected_test_scenarios=[9100, 9101, 9102], solutions=records,
        counts=dict(solutions=len(records), eligible=sum(r['eligible'] for r in records),
                    unseen=sum(r['unseen_preference'] for r in records)))
    (out / 'pareto_archive.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 固定最小训练种子仅作描述；其余种子保留在解集。禁止挑选最佳种子。
    seed = min(e['seed'] for e in entries)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = ['Cost (scaled)', 'Carbon (scaled)', 'Linear reserve proxy (scaled)']
    for family in sorted({r['family'] for r in records}):
        rows = [r for r in records if r['family'] == family and r['training_seed'] == seed and r['eligible']]
        if not rows: continue
        x = np.array([r['mean_vector'] for r in rows]) * [-1, -1, 1]
        front = np.array([r['non_dominated_within_policy_set'] for r in rows])
        for ax, (i, j) in zip(axes, pairs):
            artist = ax.scatter(x[:, i], x[:, j], s=26, alpha=.65, label=family)
            ax.scatter(x[front, i], x[front, j], s=65, facecolors='none', edgecolors=artist.get_facecolor()[:1])
            ax.set(xlabel=labels[i], ylabel=labels[j]); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'Stage 9 test solutions: fixed training seed {seed}\nRings: 3D nondominated within each method; linear reserve, not AC-certified')
    fig.tight_layout()
    fig.savefig(out / 'pareto_projections.png', dpi=150)
    fig.savefig(out / 'pareto_projections.svg')
    plt.close(fig)
    return report['counts']


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); print(run(args.runs, args.output))
