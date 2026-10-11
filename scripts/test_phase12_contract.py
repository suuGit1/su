"""真实封存完整性与已执行机制记录的只读验收，不读取确认集策略表现。"""
import hashlib
import json
from pathlib import Path
import unittest


class Phase12ContractTests(unittest.TestCase):
    def test_sealed_data_unchanged_and_disjoint(self):
        root=Path('data/real/confirmation_2019Q2');m=json.loads((root/'seal.json').read_text())
        self.assertEqual(hashlib.sha256((root/'sealed.csv').read_bytes()).hexdigest(),m['sha256'])
        self.assertEqual(len(m['days']),len(set(m['days'])))
        self.assertTrue(all('2019-04-01'<=day<='2019-06-30' for day in m['days']))
        original=json.loads(Path('data/real/gb/profiles/manifest.json').read_text())
        self.assertEqual(m['scales'],original['scales'])
        self.assertTrue(set(m['days']).isdisjoint(m['dropped']))

    def test_design_and_no_confirmation_use(self):
        f=json.loads(Path('runs/phase12_v1/frozen.json').read_text());d=f['designs']
        self.assertEqual(len(d),532);self.assertEqual(len({r['id'] for r in d}),532)
        self.assertTrue(all(r['date'].startswith('2019-03-') for r in d))
        self.assertEqual(sum(r['group']=='dt_factorial' for r in d),116)
        self.assertEqual(sum(r['group']=='c3' for r in d),336)
        self.assertEqual(sum(r['group']=='tasks' for r in d),32)
        self.assertEqual(sum(r['group']=='faults' for r in d),48)

    def test_frozen_implementation(self):
        f=json.loads(Path('runs/phase12_v1/frozen.json').read_text())
        self.assertEqual(hashlib.sha256(Path('scripts/run_phase12.py').read_bytes()).hexdigest(),f['runner_sha256'])
        for path,sha in f['source_hashes'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)
        for path,sha in json.loads(Path('experiments/phase12/resource_hashes.json').read_text()).items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)

if __name__=='__main__':unittest.main()
