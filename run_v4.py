"""v4统一入口，Windows自动启用UTF-8。"""
import os
import sys
if __name__=='__main__':
    if not sys.flags.utf8_mode:os.execv(sys.executable,[sys.executable,'-X','utf8',*sys.argv])
    from vpp_research.release_v2 import main
    main(version=4)
