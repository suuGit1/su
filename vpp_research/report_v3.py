"""生成v3审计报告及统一图表，不把缺证据的运行当成完整实验。"""
import json
from pathlib import Path
import numpy as np
from .real_report import report
from .evidence_v3 import audit_campaign
from .ledger import audit_training


def report_v3(root):
    root=Path(root);base=report(root);traces=audit_campaign(root);training=audit_training(root)
    (root/'training_audit.json').write_text(json.dumps(training,ensure_ascii=False,indent=2))
    from .plot_v2 import plot
    plot(root)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    rows=base['rows'];methods=sorted({r['method'] for r in rows})
    for j,(name,sign) in enumerate([('Cost',-1),('Carbon (kg)',-1),('Reserve (kWh)',1)]):
        for k,m in enumerate(methods):
            points=[sign*100*p[j] for r in rows if r['method']==m for p in r['front']]
            axes[j].scatter([k]*len(points),points,s=18)
        axes[j].set_xticks(range(len(methods)),methods,rotation=45);axes[j].set_ylabel(name)
    fig.suptitle('Feasible front points (not independent samples)')
    fig.savefig(root/'figures/objectives.png',dpi=160);fig.savefig(root/'figures/objectives.svg');plt.close(fig)
    scales=json.loads((root/'manifest.json').read_text()).get('experiment_contract',{}).get('scales',[100,100,100])
    if list(scales)!=[100,100,100]:raise ValueError('绘图尺度与协议不一致')
    learned=any(r['method'] not in ('mpc','milp_oracle') for r in rows)
    audit_ok=traces['consistent'] and (training['consistent'] if learned else True)
    from .suite import stats
    source=json.loads((root/'results.json').read_text())
    strict={}
    for method in methods:
        if method in ('mpc','milp_oracle'):continue
        selected=[]
        for entry in source['entries']:
            target=source['manifest']['train_steps_per_method_seed'][method]
            if entry['method']==method and entry.get('audit_consistent',False) and entry.get('budget_complete',False) and entry.get('attempted_training_calls')==entry.get('actual_training_steps')==target and not entry.get('failed',False):
                selected.extend(r['hv'] for r in rows if r['method']==method and r['seed']==entry['seed'])
        strict[method]=stats(selected) if selected else None
    result=dict(version='3.0.0',evidence_complete=audit_ok,training=training,traces=traces,strict_training_budget_hv=strict,
                nominal_budget_complete=all(r['budget_complete'] for r in rows),
                note='审计完整不等于方法优越；预算早停、重试与选择交互需单独比较；旧v2不补造证据')
    (root/'v3_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['# v3实验审计','',f'- 逐步证据完整：{audit_ok}',f'- 名义回合预算完整：{result["nominal_budget_complete"]}',
           '- 训练实际交互来自事务账本完整载荷及提交回合；辅助追加日志差异单列。',
           '- 测试和验证三目标、安全计数从轨迹复算；详情见trace_audit.json。',
           '- 不能把固定权重、OLS子策略的选择交互忽略后声称开发总预算相同。',
           '- HV/IGD见report.md；三目标图只展示各方法自身的可行非支配点，不代表真实前沿。']
    (root/'v3_report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();report_v3(a.folder)
