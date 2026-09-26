"""Disk-sharded exact top-k sparse retrieval over the FULL target universe.

Vocabulary/IDF uses a deterministic label-free target sample. Retrieval searches
all targets, not just that sample. Each view retains global top-k across shards.
"""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .runtime import atomic_json

VIEWS=('name','address','combined')


def texts(rows, view):
    if view=='name':return [r['name']+' '+r['latin_name'] for r in rows]
    if view=='address':return [r['address']+' '+r['latin_address'] for r in rows]
    return [r['latin_name']+' '+r['latin_address'] for r in rows]


def vectorizer(view, max_features, vocabulary=None):
    return TfidfVectorizer(analyzer='word' if view=='combined' else 'char',
      ngram_range=(1,2) if view=='combined' else (3,4), lowercase=False,
      dtype=np.float32,max_features=max_features,sublinear_tf=True,vocabulary=vocabulary)


def build_index(db, split, root, cfg):
    root=Path(root)/split;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return root
    sample=list(db.execute('SELECT * FROM records WHERE split=? AND source IN (2,3) ORDER BY h LIMIT ?', (split,cfg['vocabulary_sample'])))
    if not sample:raise ValueError(f'No targets in {split}')
    vecs={}
    for view in VIEWS:
        v=vectorizer(view,cfg['max_features'])
        docs=texts(sample,view)
        try:v.fit(docs)
        except ValueError:v.fit(['empty placeholder'])
        atomic_json(root/f'{view}.json',{'vocab':{k:int(x) for k,x in v.vocabulary_.items()},'idf':v.idf_.tolist()})
        vecs[view]=v
    del sample
    cursor=db.execute('SELECT * FROM records WHERE split=? AND source IN (2,3) ORDER BY country,id',(split,))
    shards=[];i=0
    while True:
        rows=cursor.fetchmany(cfg['target_shard'])
        if not rows:break
        stem=f'{i:06d}'
        marker=root/f'{stem}.json'
        if not marker.exists():
            for view in VIEWS:
                path=root/f'{stem}_{view}.npz';tmp=root/f'{stem}_{view}.tmp.npz'
                sparse.save_npz(tmp,vecs[view].transform(texts(rows,view)));tmp.replace(path)
            atomic_json(marker,{'ids':[r['id'] for r in rows],'countries':[r['country'] for r in rows]})
        shards.append(stem);i+=1
        if i%20==0:print(f'Indexed {split}: {i} target shards',flush=True)
    atomic_json(root/'complete.json',{'shards':shards,'full_target_pool':True})
    return root


class Retriever:
    def __init__(self, root, cfg):
        self.root=Path(root);self.cfg=cfg;self.vecs={}
        self.shards=json.loads((self.root/'complete.json').read_text())['shards']
        for view in VIEWS:
            obj=json.loads((self.root/f'{view}.json').read_text())
            v=vectorizer(view,cfg['max_features'],obj['vocab']);v.fit(['placeholder'])
            v.idf_=np.array(obj['idf'],dtype=np.float32);self.vecs[view]=v

    def query(self, rows):
        k=self.cfg['top_k_per_view'];best=[[[] for _ in VIEWS] for r in rows]
        for vi,view in enumerate(VIEWS):
            Q=self.vecs[view].transform(texts(rows,view))
            for stem in self.shards:
                meta=json.loads((self.root/f'{stem}.json').read_text())
                T=sparse.load_npz(self.root/f'{stem}_{view}.npz')
                # At most query_batch * target_shard nonzeros; no full-universe matrix.
                product=(Q@T.T).tocsr()
                for qi,row in enumerate(rows):
                    start,end=product.indptr[qi:qi+2];ix=product.indices[start:end];values=product.data[start:end]
                    if self.cfg['country_block']:
                        keep=np.array([meta['countries'][j]==row['country'] for j in ix],dtype=bool);ix=ix[keep];values=values[keep]
                    if len(ix)>k:
                        # Lexicographic tie-breaking makes results invariant to target sharding.
                        order=np.lexsort((np.array([meta['ids'][j] for j in ix]),-values))[:k];ix=ix[order];values=values[order]
                    merged=best[qi][vi]+[(float(s),meta['ids'][j]) for j,s in zip(ix,values) if s>0]
                    best[qi][vi]=sorted(merged,key=lambda x:(-x[0],x[1]))[:k]
                del T,product
        result=[]
        for views in best:
            union={}
            for vi,hits in enumerate(views):
                for rank,(score,tid) in enumerate(hits,1):
                    a=union.setdefault(tid,[0.,0.,0.,0.,0.,0.]);a[vi]=score;a[vi+3]=1./rank
            result.append(union)
        return result
