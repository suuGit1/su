"""从已完成日期结果生成报告；不将未完成实验、点态AC检查冒充鲁棒证据。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def report(folder):
    folder=Path(folder);r=json.loads((folder/'results.json').read_text());p=json.loads((folder/'protocol.json').read_text())
    rows=[]
    for method,e in r['estimation'].items():
        cases=[c for c in r['control'] if c['method']==method]
        steps=[s for c in cases for s in c['steps']]
        complete=[c for c in cases if not c['failed'] and c.get('completed')]
        rows.append(dict(method=method,estimation_rmse=e['rmse'],estimation_day_coverage=e['simultaneous_scenario_coverage'],
            mean_normalized_width=e['mean_interval_width'],planned_control_days=p['control_days'],recorded_control_days=len(cases),
            complete_control_days=len(complete),failed_control_days=sum(c['failed'] for c in cases),
            mean_daily_cost=float(np.mean([c['cost'] for c in complete])) if complete else None,
            control_rmse=float(np.sqrt(np.mean([s['mse'] for s in steps]))) if steps else None,
            control_step_coverage=float(np.mean([s['covered'] for s in steps])) if steps else None,
            control_day_coverage=sum(all(s['covered'] for s in c['steps']) for c in complete)/len(complete) if complete else None,
            certified_steps=sum(s['certificate'] for s in steps),executed_steps=len(steps),
            ac_violations=sum(s['ac_violations'] for s in steps),reserve_invalid=sum(not s['reserve_valid'] for s in steps)))
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    finished=all(x['recorded_control_days']==x['planned_control_days'] for x in rows)
    lines=['# v4.2 实测结果', '',f'完整运行状态：{finished}。{len(p["source"]["profiles"]["train"]["days"])} 天训练，{p["calibration_days"]} 天校准，{p["eval_days"]} 天回顾性测试。','',
           '| 方法 | 估计 RMSE（归一化） | 全天联合覆盖 | 平均区间宽度 | 完成/计划控制日 | 平均日成本 | 线性盒证书步数 | AC 违规 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        cost=f'{x["mean_daily_cost"]:.2f}' if x['mean_daily_cost'] is not None else 'NA'
        lines.append(f'| {x["method"]} | {x["estimation_rmse"]:.5f} | {x["estimation_day_coverage"]:.1%} | {x["mean_normalized_width"]:.4f} | {x["complete_control_days"]}/{x["planned_control_days"]} | {cost} | {x["certified_steps"]}/{x["executed_steps"]} | {x["ac_violations"]} |')
    lines+=['','## 配对日成本差（负数代表前者较低）','']
    for key,v in r.get('paired_cost',{}).items():
        lines.append(f'- {key}: {json.dumps(v,ensure_ascii=False)}')
    lines+=['','## 解释边界','',
        '- 各方法完成率、区间覆盖、证书与成本必须联合判断；不以窄区间或少数成功日期单独宣称优势。',
        '- AC 检查是执行点数值验证，线性盒证书也不是连续 AC 鲁棒性或全时域安全证明。',
        '- 日成本包含协议定义的控制成本与终端惩罚；非实站账单。',
        '- 未进行新版本大预算多种子 MAPPO 性能比较；本报告不能证明 Pareto-MAPPO 优于普通 MAPPO。',
        '- physics 与 residual 同时改变点预测与区间校准，当前收益是整套 DT 的联合效应，尚未分离两者贡献。',
        '- 日期具有时间相关性；现有配对 t 区间只作描述，尚需新月份/地域的确认性测试。',
        '- 真实能源与碳曲线驱动仿真。真实 EV/DR、漂移覆盖、硬件能耗与大配电网验证仍未由本轮补足。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();print(json.dumps(report(a.folder),ensure_ascii=False,indent=2))
