"""Disk-sharded exact top-k sparse retrieval over the FULL target universe.

Vocabulary/IDF uses a deterministic label-free target sample. Retrieval searches
all targets, not just that sample. Each view retains global top-k across shards.
"""
import json
import hashlib
import time
import os
import shutil
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

    def local_file(self,path):
        """Read-through local cache for immutable indexes; Drive remains the durable copy."""
        cache=os.environ.get('ER_LOCAL_CACHE')
        if not cache:return path
        key=hashlib.sha256(str(path.parent.resolve()).encode()).hexdigest()[:16]
        destination=Path(cache)/'indexes'/key/path.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and destination.stat().st_size==path.stat().st_size:return destination
        if shutil.disk_usage(destination.parent).free<path.stat().st_size+2*2**30:return path
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        shutil.copyfile(path,temporary);temporary.replace(destination)
        return destination

    @staticmethod
    def local_top(ix, values, ids, k):
        """Linear-time cutoff, then stable ID tie-breaking (including boundary ties)."""
        if len(ix)>k:
            cutoff=np.partition(values,len(values)-k)[len(values)-k]
            keep=values>=cutoff;ix=ix[keep];values=values[keep]
        order=np.lexsort((np.array([ids[j] for j in ix]),-values))[:k]
        return [(float(values[j]),ids[ix[j]]) for j in order if values[j]>0]

    def query(self, rows, checkpoint=None):
        """Load each target shard once for many queries; multiply in small RAM-bounded blocks.

        Checkpoint stores global best lists and the next (view, target-shard) step.
        No target-pair matrix for the entire query batch is materialized.
        """
        k=self.cfg['top_k_per_view'];best=[[[] for _ in VIEWS] for r in rows]
        identity=hashlib.sha256(json.dumps({'queries':[dict(r) for r in rows],
            'index':str(self.root.resolve()),'k':k,'country_block':self.cfg['country_block']},sort_keys=True).encode()).hexdigest()
        checkpoint=Path(checkpoint) if checkpoint else None
        next_step=0
        if checkpoint and checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if saved['identity']!=identity:raise ValueError('Retrieval checkpoint belongs to different queries/config/index')
            best=saved['best'];next_step=saved['next_step']
        last_save=time.monotonic()
        step=0;micro=max(1,self.cfg['query_batch'])
        for vi,view in enumerate(VIEWS):
            if next_step>=(vi+1)*len(self.shards):
                step+=(len(self.shards));continue
            Q=self.vecs[view].transform(texts(rows,view))
            for stem in self.shards:
                if step<next_step:step+=1;continue
                meta=json.loads((self.root/f'{stem}.json').read_text())
                T=sparse.load_npz(self.local_file(self.root/f'{stem}_{view}.npz'))
                countries=np.array(meta['countries'])
                for lo in range(0,len(rows),micro):
                    block=rows[lo:lo+micro]
                    product=(Q[lo:lo+micro]@T.T).tocsr()
                    for local_q,row in enumerate(block):
                        qi=lo+local_q
                        start,end=product.indptr[local_q:local_q+2]
                        ix=product.indices[start:end];values=product.data[start:end]
                        if self.cfg['country_block']:
                            keep=countries[ix]==row['country'];ix=ix[keep];values=values[keep]
                        hits=self.local_top(ix,values,meta['ids'],k)
                        merged=list(best[qi][vi])+hits
                        best[qi][vi]=sorted(merged,key=lambda x:(-x[0],x[1]))[:k]
                    del product
                del T
                step+=1
                if checkpoint and (time.monotonic()-last_save>=30 or step%len(self.shards)==0):
                    atomic_json(checkpoint,{'identity':identity,'next_step':step,'best':best})
                    last_save=time.monotonic()
                    print(f'Retrieval checkpoint: {step}/{len(VIEWS)*len(self.shards)} index blocks, {len(rows)} queries',flush=True)
        result=[]
        for views in best:
            union={}
            for vi,hits in enumerate(views):
                for rank,(score,tid) in enumerate(hits,1):
                    a=union.setdefault(tid,[0.,0.,0.,0.,0.,0.]);a[vi]=score;a[vi+3]=1./rank
            result.append(union)
        return result
