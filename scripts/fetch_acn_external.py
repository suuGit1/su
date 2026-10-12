"""获取公开ACN派生缓存并审核字段；缺少申报时间时拒绝将其当在线需求。"""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

URL='https://raw.githubusercontent.com/chrisyeh96/sustaingym/main/sustaingym/data/evcharging/acn_data/caltech/2019-05-01%202019-08-31.csv.gz'
SHA='d23a07e3b5175dca0da107610557f604fa177f71bfc696687c70a0470964047b'

def main():
    path=Path('data/real/external/acn_sustaingym/caltech_201905_201908.csv.gz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():data=path.read_bytes()
    else:
        with urlopen(URL,timeout=30) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=SHA:raise ValueError('公开源内容发生变化，必须重新审核，不静默替换')
    rows=csv.DictReader(io.StringIO(gzip.decompress(data).decode()));items=list(rows)
    record=dict(source=URL,sha256=SHA,bytes=len(data),rows=len(items),columns=rows.fieldnames,
                online_demand_eligible=False,reason='派生CSV没有userInputs.modifiedAt，无法证明所选申报在控制前可见',
                geographic_scope='美国加州Caltech，禁止冒充英国2019年VPP实测会话',
                next_source='https://ev.caltech.edu/dataset.html',required_fields=['connectionTime','disconnectTime','sessionID','userInputs.modifiedAt','userInputs.kWhRequested','userInputs.requestedDeparture'])
    path.write_bytes(data);path.with_suffix('.audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps({k:record[k] for k in ('rows','online_demand_eligible','reason')},ensure_ascii=False))
if __name__=='__main__':main()
