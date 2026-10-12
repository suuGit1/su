"""真实数据验收官方MAPPO与集中PPO的动作掩码及检查点重载。"""
import sys,json,zipfile,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_mappo.config import Config
from vpp_research.real_inputs import connect
from vpp_research.train_v4 import train_v4
from vpp_research.train import load
c,*_=connect('data/real',Config.load('configs/v44_ieee33.json'));c.episodes=1;c.seed=43
with zipfile.ZipFile('experiments/v42/dt_study_evidence.zip') as z:model=json.loads(z.read('residual.json'))
for method in ('ordinary','central'):
 folder=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/v44_baseline_smoke')/method
 train_v4(copy.deepcopy(c),model,folder,method=method,reserve_mode='pcc_checked',resume=(folder/'resume.pt').exists())
 state,cfg,agent=load(folder/'latest.pt')
 print(method,'检查点重载通过',flush=True)
