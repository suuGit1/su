"""验证同步替换、进程丢失和损坏时账本仍如实记录。"""
import json,os,sqlite3,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vpp_research.ledger import Ledger

class JournalTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'interactions.sqlite'
  self.env=patch.dict(os.environ,{'VPP_LEDGER_JOURNAL':'1'});self.env.start()
 def tearDown(self):self.env.stop();self.tmp.cleanup()
 def test_lost_process_keeps_unknown(self):
  l=Ledger(self.path);l.begin('a',0,0);l.db.close()
  l=Ledger(self.path)
  self.assertEqual(l.summary()['unknown_calls'],1)
  with self.assertRaisesRegex(ValueError,'结果不明'):l.reconcile([],1)
  l.close()
 def test_replacement_and_replay(self):
  l=Ledger(self.path);i=l.begin('a',0,0)
  other=self.path.with_suffix('.replacement');sqlite3.connect(other).close();other.replace(self.path)
  l.finish(i,'success',dict(attempt_id='a',episode=0,step=0));l.commit_episode(0,'a',1)
  expected=l.evidence();l.export();l.db.close()
  with patch.dict(os.environ,{'VPP_LEDGER_JOURNAL':'0'}):
   l=Ledger(self.path);self.assertEqual(l.evidence(),expected);l.close()
 def test_corruption_rejected(self):
  l=Ledger(self.path);l.begin('a',0,0);l.close()
  f=next(l.events.glob('[0-9]*.json'));e=json.loads(f.read_text());e['params'][0]='tampered';f.write_text(json.dumps(e))
  with self.assertRaisesRegex(ValueError,'损坏'):Ledger(self.path)
 def test_duplicate_finish_and_partial_commit(self):
  l=Ledger(self.path);i=l.begin('a',0,0);l.finish(i,'error',{})
  with self.assertRaises(ValueError):l.finish(i,'success',{})
  with self.assertRaises(ValueError):l.commit_episode(0,'a',1)
  l.close();l=Ledger(self.path);self.assertEqual(l.summary()['error_calls'],1);l.close()
 def test_legacy_migration(self):
  with patch.dict(os.environ,{'VPP_LEDGER_JOURNAL':'0'}):
   l=Ledger(self.path);i=l.begin('old',0,0);l.finish(i,'success',{});l.commit_episode(0,'old',1);l.close()
  l=Ledger(self.path);self.assertEqual(l.summary()['committed_steps'],1);l.close()
if __name__=='__main__':unittest.main()
