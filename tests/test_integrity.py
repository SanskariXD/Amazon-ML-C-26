import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from src.audit import build,connect
from src.blocking import build_index,Retriever
from src.smoke import make_data
from src.submission import validate

class Integrity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        make_data(self.root/'dataset');self.db=connect(build(self.root/'dataset',self.root/'work',2026))
    def tearDown(self):self.db.close();self.temp.cleanup()
    def test_retrieval_invariant_to_shards(self):
        cfg=json.loads(Path('configs/baseline.json').read_text());cfg.update(target_shard=100,vocabulary_sample=500,max_features=1000,top_k_per_view=4)
        rows=list(self.db.execute("SELECT * FROM records WHERE split='test' AND source=1 LIMIT 4"))
        a=Retriever(build_index(self.db,'test',self.root/'a',cfg),cfg).query(rows)
        cfg['target_shard']=31
        b=Retriever(build_index(self.db,'test',self.root/'b',cfg),cfg).query(rows)
        self.assertEqual(a,b)
    def write_outputs(self,bad=None):
        m=self.root/'m.tsv';c=self.root/'c.tsv'
        with m.open('w',newline='') as mf,c.open('w',newline='') as cf:
            mw=csv.writer(mf,delimiter='\t');cw=csv.writer(cf,delimiter='\t')
            mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
            for i,r in enumerate(self.db.execute("SELECT id FROM records WHERE split='test' AND source=1 ORDER BY id")):
                mw.writerow([r[0], 'S2-D000000' if bad=='subset' and i==0 else ''])
                cw.writerow([r[0], 'S2-D000000,S2-D000000' if bad=='duplicate' and i==0 else ''])
        return m,c
    def test_empty_predictions_keep_every_entity(self):
        self.assertEqual(validate(*self.write_outputs(),self.db)['entities'],36)
    def test_reject_match_outside_candidates(self):
        with self.assertRaises(ValueError):validate(*self.write_outputs('subset'),self.db)
    def test_reject_duplicate_candidates(self):
        with self.assertRaises(ValueError):validate(*self.write_outputs('duplicate'),self.db)

if __name__=='__main__':unittest.main()
