"""CSV 数据接口：每个 scenario 对应一个完整有限时域回合。"""
import csv
from pathlib import Path
import hashlib
import json
import numpy as np

FIELDS = ('load_kw', 'pv_kw', 'wind_kw', 'price')


class CSVProfiles:
    def __init__(self, path, horizon):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.provenance = None
        manifest_path = self.path.parent / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if 'splits' in manifest and 'dt_hours' in manifest:
                if self.sha256 not in [v['sha256'] for v in manifest['splits'].values()]:
                    raise ValueError('CSV 与预处理 manifest 的哈希不匹配')
                if manifest['horizon'] != horizon:
                    raise ValueError('CSV 预处理时域与配置不匹配')
                self.provenance = manifest
        groups = {}
        with self.path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'scenario', 'step', *FIELDS}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('CSV 必须包含：' + ', '.join(sorted(required)))
            optional = [k for k in ('carbon_g_per_kwh','carbon_observed_g_per_kwh') if k in reader.fieldnames]
            for row in reader:
                groups.setdefault(row['scenario'], []).append(row)
        if not groups:
            raise ValueError('CSV 不得为空')
        self.profiles = []
        self.scenario_names = []
        self.fingerprints = []
        for name in sorted(groups):
            rows = sorted(groups[name], key=lambda x: int(x['step']))
            if [int(r['step']) for r in rows] != list(range(horizon)):
                raise ValueError(f'{name} 的 step 必须恰好为 0 到 {horizon-1}')
            profile = {k: np.asarray([float(r[k]) for r in rows], dtype=float) for k in FIELDS}
            for k in optional:
                profile[k] = np.asarray([float(r[k]) for r in rows], dtype=float)
                if (profile[k] < 0).any():
                    raise ValueError(k + ' 不得为负数')
            if any(not np.isfinite(v).all() for v in profile.values()):
                raise ValueError('CSV 存在缺失值或非有限数值')
            if any((profile[k] < 0).any() for k in FIELDS[:3]):
                raise ValueError('负荷与可再生出力不得为负数')
            self.profiles.append(profile)
            self.scenario_names.append(name)
            self.fingerprints.append(hashlib.sha256(b''.join(profile[k].astype('<f8').tobytes() for k in FIELDS)).hexdigest())

    def get(self, index):
        return {k: v.copy() for k, v in self.profiles[index % len(self.profiles)].items()}
