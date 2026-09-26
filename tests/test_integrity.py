import csv
import json
from pathlib import Path
import sqlite3
import os
import itertools
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
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
    def test_macro_batch_and_micro_batch_same_candidates(self):
        cfg=json.loads(Path('configs/baseline.json').read_text());cfg.update(target_shard=60,vocabulary_sample=500,max_features=1000,top_k_per_view=4,query_batch=2)
        rows=list(self.db.execute("SELECT * FROM records WHERE split='test' AND source=1 LIMIT 6"))
        retriever=Retriever(build_index(self.db,'test',self.root/'index',cfg),cfg)
        together=retriever.query(rows)
        separately=[retriever.query([r])[0] for r in rows]
        self.assertEqual(together,separately)
    def test_checkpoint_resume_and_identity(self):
        cfg=json.loads(Path('configs/baseline.json').read_text());cfg.update(target_shard=200,vocabulary_sample=500,max_features=1000,top_k_per_view=4)
        rows=list(self.db.execute("SELECT * FROM records WHERE split='test' AND source=1 LIMIT 3"))
        retriever=Retriever(build_index(self.db,'test',self.root/'index',cfg),cfg)
        checkpoint=self.root/'search.json'
        expected=retriever.query(rows,checkpoint)
        # Completed checkpoint must produce results with zero matrix reloads.
        with patch('src.blocking.sparse.load_npz',side_effect=AssertionError('Reloaded completed work')):
            self.assertEqual(expected,retriever.query(rows,checkpoint))
        with self.assertRaises(ValueError):retriever.query(rows[:1],checkpoint)
    def test_local_database_with_persistent_checkpoints(self):
        with patch.dict(os.environ,{'ER_LOCAL_CACHE':str(self.root/'local')}):
            with patch('src.audit.time.monotonic',side_effect=itertools.count(0,200)):
                path=build(self.root/'dataset',self.root/'durable',2026)
        with connect(path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM records WHERE source=1").fetchone()[0],276)
        self.assertTrue((self.root/'durable/dataset_profile.json').exists())
    def test_resume_after_first_view(self):
        cfg=json.loads(Path('configs/baseline.json').read_text());cfg.update(target_shard=200,vocabulary_sample=500,max_features=1000,top_k_per_view=4)
        rows=list(self.db.execute("SELECT * FROM records WHERE split='test' AND source=1 LIMIT 3"))
        retriever=Retriever(build_index(self.db,'test',self.root/'index',cfg),cfg)
        checkpoint=self.root/'interrupted.json'
        from scipy import sparse
        real=sparse.load_npz;counter=[0]
        def interrupt(path):
            counter[0]+=1
            if counter[0]>len(retriever.shards):raise RuntimeError('Simulated disconnect')
            return real(path)
        with patch('src.blocking.sparse.load_npz',side_effect=interrupt):
            with self.assertRaises(RuntimeError):retriever.query(rows,checkpoint)
        self.assertTrue(checkpoint.exists())
        self.assertEqual(retriever.query(rows),retriever.query(rows,checkpoint))
    def test_cutoff_keeps_correct_id_ties(self):
        hits=Retriever.local_top(np.arange(5),np.array([.8,.9,.8,.8,.1]),['z','y','a','b','x'],3)
        self.assertEqual(hits,[(.9,'y'),(.8,'a'),(.8,'b')])
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
