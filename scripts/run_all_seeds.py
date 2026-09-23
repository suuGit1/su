"""从项目根目录运行真实数据多种子实验。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.real_campaign import main

if __name__=='__main__':main()
