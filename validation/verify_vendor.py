"""校验固定版本官方文件，除包入口外要求 Git blob 与上游一致。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'vendor' / 'mappo'
manifest = json.loads((root / 'upstream_manifest.json').read_text())
for entry in manifest['files']:
    content = (root / entry['path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['sha256']:
        raise RuntimeError('本地内容已改变：' + entry['path'])
    blob = b'blob ' + str(len(content)).encode() + b'\0' + content
    if entry['path'] != 'onpolicy/__init__.py' and hashlib.sha1(blob).hexdigest() != entry['sha']:
        raise RuntimeError('与上游 Git blob 不一致：' + entry['path'])
print(f"官方文件校验通过：{len(manifest['files'])} 个文件，提交 {manifest['commit']}")
