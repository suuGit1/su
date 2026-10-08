"""从项目根目录重建 v4.1 费用、预算、HV及安全诊断报告。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.report_v3 import build
from vpp_research.report_v4 import report_v4
from vpp_research.report_v41 import report_v41
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    build(a.folder);report_v4(a.folder);report_v41(a.folder)
