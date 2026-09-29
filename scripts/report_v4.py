"""重新生成v4完整审计、分项报告及图表。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.report_v3 import build
from vpp_research.report_v4 import report_v4
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();build(a.folder);report_v4(a.folder)
