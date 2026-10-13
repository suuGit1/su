"""使用用户自行取得的ACN API令牌下载完整字段和所有分页；令牌不写入日志或文件。"""
import argparse,base64,hashlib,json,os
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode,urljoin,urlparse
from datetime import datetime,timezone
from email.utils import format_datetime
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',choices=['caltech','jpl','office001'],default='caltech');p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--token-env',default='ACN_API_TOKEN');p.add_argument('--output',required=True);a=p.parse_args()
start=datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc);end=datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
if start>=end:raise ValueError('开始日期必须早于结束日期')
token=os.environ.get(a.token_env)
if not token:raise ValueError('请先在本地环境变量中设置官方API令牌；请勿把令牌提交到仓库')
out=Path(a.output)
if out.exists():raise ValueError('拒绝覆盖原始下载')
base='https://ev.caltech.edu/api/v1/'
where=f'connectionTime>="{format_datetime(start,usegmt=True)}" and connectionTime<"{format_datetime(end,usegmt=True)}"'
url=base+f'sessions/{a.site}?'+urlencode({'where':where});items=[];seen=set();pages=0
while url:
 parsed=urlparse(url)
 if parsed.scheme!='https' or parsed.netloc!='ev.caltech.edu' or not parsed.path.startswith('/api/v1/sessions/'):raise ValueError('分页链接超出官方API范围')
 if url in seen:raise ValueError('检测到重复分页')
 seen.add(url)
 request=Request(url,headers={'Authorization':'Basic '+base64.b64encode((token+':').encode()).decode()})
 with urlopen(request,timeout=60) as response:batch=json.load(response)
 items.extend(batch['_items']);pages+=1
 link=batch.get('_links',{}).get('next',{}).get('href');url=urljoin(base,link) if link else None
ids=[r['sessionID'] for r in items]
if len(ids)!=len(set(ids)):raise ValueError('分页结果包含重复会话，需人工核对')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'_items':items},ensure_ascii=False,indent=2))
audit=dict(site=a.site,start=a.start,end=a.end,pages=pages,sessions=len(items),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source='https://ev.caltech.edu/dataset.html',note='完整API字段不等于每条都有可因果使用申报；须继续运行ev_import审核')
out.with_suffix('.download.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(audit,ensure_ascii=False))
