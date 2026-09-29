"""事务式环境交互账本：区分调用、成功、异常、未知和已提交PPO更新。"""
import json
import sqlite3
import os
import hashlib
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
        from .trace import plain
        if status not in ('success','error'):raise ValueError('交互结束状态非法')
        with self.db:
            cur=self.db.execute('UPDATE calls SET status=?,detail=? WHERE id=? AND status=?',
                                (status,json.dumps(detail,ensure_ascii=False,default=plain),ident,'pending'))
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

    def evidence(self):
        """完成时一次读取全体记录，避免跨文件读取到不同提交版本。"""
        calls=[dict(id=r[0],attempt=r[1],episode=r[2],step=r[3],status=r[4],detail=json.loads(r[5]) if r[5] else None)
               for r in self.db.execute('SELECT id,attempt,episode,step,status,detail FROM calls ORDER BY id')]
        commits=[dict(episode=r[0],attempt=r[1],steps=r[2]) for r in self.db.execute('SELECT * FROM commits ORDER BY episode')]
        return dict(schema='completed-evidence-v3',calls=calls,commits=commits,summary=self.summary())

    def close(self):self.db.close()

    def export(self):
        """只导出事务内真实保存的记录；不能为历史空载荷补造轨迹。"""
        records=[json.loads(r[0]) for r in self.db.execute('SELECT detail FROM calls WHERE status=? ORDER BY id',('success',))]
        if any(not isinstance(r,dict) or not {'attempt_id','episode','step'}<=r.keys() for r in records):
            raise ValueError('历史账本未保存完整载荷，不能重建轨迹')
        for name,rows in [('successful_calls.jsonl',[dict(event='step',**{k:r[k] for k in ('attempt_id','episode','step')}) for r in records]),
                          ('trajectory_complete.jsonl',[r for r in records if 'feedback' in r])]:
            path=self.path.parent/name;temp=path.with_suffix(path.suffix+'.tmp')
            with temp.open('w') as f:
                f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));f.flush();os.fsync(f.fileno())
            temp.replace(path)
        evidence=self.evidence()
        encoded=json.dumps(evidence,sort_keys=True,ensure_ascii=False)
        receipt=dict(evidence=evidence,sha256=hashlib.sha256(encoded.encode()).hexdigest())
        path=self.path.parent/'completed_evidence.json';temp=path.with_suffix('.json.tmp')
        with temp.open('w') as f:
            f.write(json.dumps(receipt,ensure_ascii=False,indent=2));f.flush();os.fsync(f.fileno())
        temp.replace(path)


def completed_evidence(folder):
    """校验完整快照的摘要及内部计数；不存在时返回None，不补造历史快照。"""
    path=Path(folder)/'completed_evidence.json'
    if not path.exists():return None
    receipt=json.loads(path.read_text(encoding='utf-8'));e=receipt['evidence']
    if hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=receipt['sha256']:raise ValueError('完成快照摘要错误')
    calls=e['calls'];s=e['summary']
    if s['attempted_calls']!=len(calls) or any(s[key]!=sum(c['status']==status for c in calls) for key,status in [('successful_calls','success'),('unknown_calls','pending'),('error_calls','error')]):raise ValueError('完成快照计数不一致')
    if s['committed_steps']!=sum(c['steps'] for c in e['commits']):raise ValueError('完成快照提交计数不一致')
    if len({c['id'] for c in calls})!=len(calls):raise ValueError('完成快照交互ID重复')
    for c in e['commits']:
        steps=[x['step'] for x in calls if x['attempt']==c['attempt'] and x['episode']==c['episode'] and x['status']=='success']
        if steps!=list(range(c['steps'])):raise ValueError('完成快照提交回合缺步')
    return e


def audit_training(folder):
    """只读审计；绝不依据缺失日志补造实际交互。"""
    import csv
    root=Path(folder);items=[]
    logs=set(root.rglob('attempts.jsonl'))|{p.parent/'attempts.jsonl' for p in root.rglob('interactions.sqlite')}
    for log in sorted(logs):
        records=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines() if s.strip()] if log.exists() else []
        d=log.parent;history=[]
        if (d/'training.csv').exists():
            with (d/'training.csv').open(encoding='utf-8') as handle:history=list(csv.DictReader(handle))
        nominal=int(history[-1]['env_steps']) if history else 0
        logged=sum(r.get('event')=='step' for r in records)
        item=dict(path=str(d),nominal_steps=nominal,logged_steps=logged,ledger_available=(d/'interactions.sqlite').exists())
        if item['ledger_available']:
            ledger=Ledger(d/'interactions.sqlite');item.update(ledger.summary())
            expected=[tuple(r) for r in ledger.db.execute('SELECT attempt,episode,step FROM calls WHERE status=? ORDER BY id',('success',))]
            observed=[(r.get('attempt_id'),r.get('episode'),r.get('step')) for r in records if r.get('event')=='step']
            item['step_identities_match']=expected==observed
            payloads=[json.loads(r[0]) for r in ledger.db.execute('SELECT detail FROM calls WHERE status=? ORDER BY id',('success',))]
            item['transaction_payload_complete']=bool(payloads) and all(isinstance(r,dict) and {'attempt_id','episode','step'}<=r.keys() for r in payloads)
            if item['transaction_payload_complete']:
                item['transaction_payload_complete']=expected==[(r['attempt_id'],r['episode'],r['step']) for r in payloads]
            ledger.close()
            trace=d/'trajectory.jsonl'
            item['training_trace_matches']=None
            if trace.exists():
                data=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines() if s.strip()]
                item['training_trace_matches']=expected==[(r.get('attempt_id'),r.get('episode'),r.get('step')) for r in data]
            item['consistent']=item['committed_steps']==nominal and item['step_identities_match'] and item['unknown_calls']==0 and item['training_trace_matches'] is not False
            if item['transaction_payload_complete']:
                full_trace=all('feedback' in r for r in payloads) if trace.exists() else True
                item['consistent']=item['committed_steps']==nominal and item['unknown_calls']==0 and full_trace
                item['accounting_source']='事务载荷；追加式JSON日志仅作辅助副本'
            snapshot=completed_evidence(d)
            if snapshot is not None:
                item['live_ledger_summary']={k:item[k] for k in snapshot['summary']}
                item['snapshot_matches_live_ledger']=all(item[k]==v for k,v in snapshot['summary'].items())
                item.update(snapshot['summary'])
                good=[c for c in snapshot['calls'] if c['status']=='success']
                item['transaction_payload_complete']=all(isinstance(c['detail'],dict) and
                    (c['detail'].get('attempt_id'),c['detail'].get('episode'),c['detail'].get('step'))==(c['attempt'],c['episode'],c['step']) and
                    ('feedback' in c['detail'] if trace.exists() else True) for c in good)
                item['consistent']=item['committed_steps']==nominal and item['unknown_calls']==0 and item['transaction_payload_complete']
                item['accounting_source']='完整事务载荷的完成快照；内部回合与计数已校验，其他副本差异单列'
        else:item['consistent']=False
        items.append(item)
    return dict(schema='training-audit-v3',runs=items,consistent=bool(items) and all(x['consistent'] for x in items),
                note='旧日志不能证明完整实际交互；未知调用不自动视为成功；重做回合计入successful_calls')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(audit_training(a.folder),ensure_ascii=False,indent=2))
