import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.audit import build, connect
from src.smoke import make_data
from src import lexical
from src.retrieval import build_index, Retriever


class LexicalIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        make_data(self.root/'data')
        self.db=connect(build(self.root/'data',self.root/'prepared',2026))
        self.cfg=json.loads(Path('configs/colab_screen.json').read_text())
        self.cfg.update(posting_buckets=4,target_shard=31,top_k_per_view=10)
        self.rows=list(self.db.execute("SELECT * FROM records WHERE split='test' AND source=1"))
    def tearDown(self):
        self.db.close();self.tmp.cleanup()
    def test_full_pool_country_and_shard_invariance(self):
        path=build_index(self.db,'test',self.root/'a',self.cfg)
        meta=json.loads((path/'complete.json').read_text())
        self.assertEqual(meta['targets'],self.db.execute("SELECT COUNT(*) FROM records WHERE split='test' AND source IN (2,3)").fetchone()[0])
        a=Retriever(self.db,'test',path,self.cfg).query(self.rows)
        cfg=dict(self.cfg,target_shard=17,posting_buckets=7)
        other=build_index(self.db,'test',self.root/'b',cfg)
        self.assertEqual(a,Retriever(self.db,'test',other,cfg).query(self.rows))
        for row,candidates in zip(self.rows,a):
            for tid in candidates:
                self.assertEqual(row['country'],self.db.execute("SELECT country FROM records WHERE split='test' AND id=?",(tid,)).fetchone()[0])
            i=int(row['id'].split('-')[1])
            if i%5:self.assertIn(f'S2-{i:06d}',candidates)
        self.assertTrue(any(r['country']=='France' for r in self.rows))
    def test_resume_uncommitted_posting_tail(self):
        real=lexical.atomic_json; calls=[0]
        def interrupt(path,value):
            if path.name=='ingestion.json':
                calls[0]+=1
                if calls[0]==2:raise RuntimeError('disconnect before checkpoint')
            return real(path,value)
        with patch('src.lexical.atomic_json',side_effect=interrupt):
            with self.assertRaises(RuntimeError):build_index(self.db,'test',self.root/'resume',self.cfg)
        p=build_index(self.db,'test',self.root/'resume',self.cfg)
        q=build_index(self.db,'test',self.root/'fresh',self.cfg)
        self.assertEqual(Retriever(self.db,'test',p,self.cfg).query(self.rows),Retriever(self.db,'test',q,self.cfg).query(self.rows))
        with patch('src.lexical.key_views',side_effect=AssertionError('reindexed')):
            self.assertEqual(p,build_index(self.db,'test',self.root/'resume',self.cfg))
    def test_resume_partial_compaction(self):
        real=lexical.atomic_json
        def interrupt(path,value):
            if path.name=='bucket_001.json':raise RuntimeError('disconnect before bucket commit')
            return real(path,value)
        with patch('src.lexical.atomic_json',side_effect=interrupt):
            with self.assertRaises(RuntimeError):build_index(self.db,'test',self.root/'resume',self.cfg)
        p=build_index(self.db,'test',self.root/'resume',self.cfg)
        q=build_index(self.db,'test',self.root/'fresh',self.cfg)
        self.assertEqual(Retriever(self.db,'test',p,self.cfg).query(self.rows),Retriever(self.db,'test',q,self.cfg).query(self.rows))
    def test_missing_address_and_common_key_accounting(self):
        cfg=dict(self.cfg,max_key_postings=2)
        p=build_index(self.db,'test',self.root/'rare',cfg)
        self.assertGreater(json.loads((p/'complete.json').read_text())['omitted_common_keys'],0)
        p=build_index(self.db,'test',self.root/'normal',self.cfg)
        row=dict(self.rows[1]);row['address']='';row['latin_address']=''
        candidates=Retriever(self.db,'test',p,self.cfg).query([row])[0]
        self.assertIn('S2-000001',candidates)
    def test_compact_text_preserves_normalized_views(self):
        for row in self.rows:
            self.assertEqual(row['name'],row['latin_name'])
            self.assertEqual(row['address'],row['latin_address'])
        stored=self.db.execute("SELECT COUNT(*) FROM records WHERE latin_name IS NULL AND name!=''").fetchone()[0]
        self.assertGreater(stored,0)

if __name__=='__main__':unittest.main()
