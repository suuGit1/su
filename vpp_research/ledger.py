"""事务式环境交互账本：区分调用、成功、异常、未知和已提交PPO更新。"""
import json
import sqlite3
from pathlib import Path


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, attempt TEXT, episode INTEGER, step INTEGER, status TEXT, detail TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS commits (episode INTEGER PRIMARY KEY, attempt TEXT, steps INTEGER)')
        self.db.commit()

    def begin(self, attempt, episode, step):
        with self.db:
            cur = self.db.execute('INSERT INTO calls(attempt,episode,step,status) VALUES(?,?,?,?)', (attempt,episode,step,'pending'))
        return cur.lastrowid

    def finish(self, ident, status, detail=None):
        if status not in ('success','error'):raise ValueError('交互结束状态非法')
        with self.db:
            cur=self.db.execute('UPDATE calls SET status=?,detail=? WHERE id=? AND status=?',
                                (status,json.dumps(detail,ensure_ascii=False),ident,'pending'))
            if cur.rowcount!=1:raise ValueError('交互不存在或已经结束')

    def commit_episode(self, episode, attempt, horizon):
        steps=[r[0] for r in self.db.execute('SELECT step FROM calls WHERE attempt=? AND episode=? AND status=? ORDER BY id', (attempt,episode,'success'))]
        if steps!=list(range(horizon)):raise ValueError('PPO回合交互不完整，禁止提交')
        with self.db:self.db.execute('INSERT OR REPLACE INTO commits VALUES(?,?,?)',(episode,attempt,horizon))

    def reconcile(self, episode_attempts, horizon):
        if self.summary()['unknown_calls']:raise ValueError('账本存在结果不明的环境调用；保留目录供审计，禁止自动续训')
        with self.db:self.db.execute('DELETE FROM commits WHERE episode>=?',(len(episode_attempts),))
        for ep,attempt in enumerate(episode_attempts):self.commit_episode(ep,attempt,horizon)

    def summary(self):
        counts=dict(self.db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status'))
        return dict(schema='interaction-ledger-v3',attempted_calls=sum(counts.values()),successful_calls=counts.get('success',0),
                    error_calls=counts.get('error',0),unknown_calls=counts.get('pending',0),
                    committed_steps=self.db.execute('SELECT COALESCE(SUM(steps),0) FROM commits').fetchone()[0])

    def close(self):self.db.close()


def audit_training(folder):
    """只读审计；绝不依据缺失日志补造实际交互。"""
    import csv
    root=Path(folder);items=[]
    logs=set(root.rglob('attempts.jsonl'))|{p.parent/'attempts.jsonl' for p in root.rglob('interactions.sqlite')}
    for log in sorted(logs):
        records=[json.loads(s) for s in log.read_text().splitlines() if s.strip()] if log.exists() else []
        d=log.parent;history=list(csv.DictReader((d/'training.csv').open())) if (d/'training.csv').exists() else []
        nominal=int(history[-1]['env_steps']) if history else 0
        logged=sum(r.get('event')=='step' for r in records)
        item=dict(path=str(d),nominal_steps=nominal,logged_steps=logged,ledger_available=(d/'interactions.sqlite').exists())
        if item['ledger_available']:
            ledger=Ledger(d/'interactions.sqlite');item.update(ledger.summary())
            expected=[tuple(r) for r in ledger.db.execute('SELECT attempt,episode,step FROM calls WHERE status=? ORDER BY id',('success',))]
            observed=[(r.get('attempt_id'),r.get('episode'),r.get('step')) for r in records if r.get('event')=='step']
            item['step_identities_match']=expected==observed
            ledger.close()
            trace=d/'trajectory.jsonl'
            item['training_trace_matches']=None
            if trace.exists():
                data=[json.loads(s) for s in trace.read_text().splitlines() if s.strip()]
                item['training_trace_matches']=expected==[(r.get('attempt_id'),r.get('episode'),r.get('step')) for r in data]
            item['consistent']=item['committed_steps']==nominal and item['step_identities_match'] and item['unknown_calls']==0 and item['training_trace_matches'] is not False
        else:item['consistent']=False
        items.append(item)
    return dict(schema='training-audit-v3',runs=items,consistent=bool(items) and all(x['consistent'] for x in items),
                note='旧日志不能证明完整实际交互；未知调用不自动视为成功；重做回合计入successful_calls')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(audit_training(a.folder),ensure_ascii=False,indent=2))
