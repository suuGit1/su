"""VPP 的官方 MAPPO 适配框架。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'vendor/mappo'):
    sys.path.insert(0, str(path))
