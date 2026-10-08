"""v4.1 安全闭环与延迟感知 MPC 入口，自动启用 UTF-8。"""
import os
import sys
if __name__=='__main__':
    if not sys.flags.utf8_mode:os.execv(sys.executable,[sys.executable,'-X','utf8',*sys.argv])
    from vpp_research.release_v2 import main
    main(version=4,default_config='configs/v41_ieee33.json')
