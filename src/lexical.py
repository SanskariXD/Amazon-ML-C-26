"""Experimental full-universe lexical blocking with compact disk-backed postings.

Every target is indexed. Overfull individual keys are omitted, never sampled.
Recall loss from omitted/common keys is explicitly measured by the benchmark.
No labels or holdout queries participate in constructing keys or postings.
"""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .runtime import atomic_json
from .normalize import numbers

DTYPE=np.dtype([('key','<u8'),('rid','<u4')])


def key_views(row,country_block):
    n=row['latin_name'];a=row['latin_address']
    nw=sorted(set(w for w in n.split() if len(w)>1),key=lambda x:(-len(x),x))[:4]
    aw=sorted(set(w for w in a.split() if len(w)>2 and not w.isdigit()),key=lambda x:(-len(x),x))[:4]
    nums=sorted(numbers(a))[:2]
    keys=[]
    if n:keys.append(('N:'+n,0))
    keys += [('W:'+w,0) for w in nw]
    keys += [('P:'+':'.join(sorted(p)),0) for p in list(itertools.combinations(nw,2))[:3]]
    compact=''.join(n.split())
    grams=set(compact[i:i+3] for i in range(max(0,len(compact)-2)))
    # Stable min-hash subset distributes character evidence instead of truncating a name prefix.
    grams=sorted(grams,key=lambda g:hashlib.blake2b(g.encode(),digest_size=8).digest())[:4]
    keys += [('G:'+g,0) for g in grams]
    keys += [('A:'+w,1) for w in aw]
    keys += [('J:'+num+':'+w,2) for num in nums for w in nw[:2]]
    keys += [('H:'+num+':'+w,2) for num in nums for w in aw[:2]]
    prefix=row['country']+'\0' if country_block else ''
    return {int.from_bytes(hashlib.blake2b((prefix+k).encode(),digest_size=8).digest(),'little'):v for k,v in keys}


def build_index(db,split,root,cfg):
    root=Path(root)/split;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return root
    buckets=int(cfg.get('posting_buckets',32));cap=int(cfg.get('max_key_postings',2048))
    if buckets<1 or cap<1:raise ValueError('Positive posting buckets/cap required')
    marker=root/'ingestion.json'
    state=json.loads(marker.read_text()) if marker.exists() else {'last_rid':0,'rows':0,'sizes':[0]*buckets,'ingested':False}
    total=db.execute('SELECT COUNT(*) FROM records WHERE split=? AND source IN (2,3)',(split,)).fetchone()[0]
    if not state['ingested']:
        # Truncate only uncommitted tails from this index's append-only posting files.
        for b in range(buckets):
            path=root/f'raw_{b:03d}.bin'
            if not path.exists() and state['sizes'][b]:raise RuntimeError('A committed posting file is missing')
            with path.open('ab') as f:pass
            with path.open('r+b') as f:
                if path.stat().st_size<state['sizes'][b]:raise RuntimeError('A committed posting file is truncated')
                f.truncate(state['sizes'][b])
        cursor=db.execute('SELECT rowid AS rid,* FROM records WHERE split=? AND source IN (2,3) AND rowid>? ORDER BY rowid',(split,state['last_rid']))
        while True:
            rows=cursor.fetchmany(cfg['target_shard'])
            if not rows:break
            pairs=[]
            for row in rows:
                if row['rid']>=2**32:raise ValueError('Dataset exceeds uint32 record address space')
                pairs.extend((key,row['rid']) for key in key_views(row,cfg['country_block']))
            data=np.array(pairs,dtype=DTYPE);del pairs
            if shutil.disk_usage(root).free < data.nbytes*3+256*2**20:raise OSError('Insufficient disk for next posting shard')
            bucket_ids=data['key']%buckets
            for b in range(buckets):
                part=data[bucket_ids==b]
                with (root/f'raw_{b:03d}.bin').open('ab') as f:f.write(part.tobytes())
                state['sizes'][b]+=part.nbytes
            state['last_rid']=rows[-1]['rid'];state['rows']+=len(rows)
            atomic_json(marker,state)
            if state['rows']% (cfg['target_shard']*10)==0:
                print(f'{split} lexical index: {state["rows"]:,}/{total:,} targets',flush=True)
        state['ingested']=True;atomic_json(marker,state)
    kept_keys=omitted_keys=kept_postings=0
    for b in range(buckets):
        ready=root/f'bucket_{b:03d}.json';raw=root/f'raw_{b:03d}.bin'
        if ready.exists():
            stats=json.loads(ready.read_text())
        else:
            # Each bucket is sorted independently. No full-universe sort/table in RAM.
            import psutil
            if raw.stat().st_size*8>psutil.virtual_memory().available*.60:
                raise MemoryError('Posting bucket exceeds sort budget; increase posting_buckets in a new config')
            data=np.fromfile(raw,dtype=DTYPE);data.sort(order=['key','rid'])
            keys,starts,counts=np.unique(data['key'],return_index=True,return_counts=True)
            allowed=counts<=cap
            take=np.repeat(allowed,counts)
            ids=data['rid'][take].copy();keys=keys[allowed];lengths=counts[allowed]
            offsets=np.concatenate([np.zeros(1,dtype=np.uint64),np.cumsum(lengths,dtype=np.uint64)])
            for suffix,array in [('keys',keys),('offsets',offsets),('ids',ids)]:
                path=root/f'{b:03d}_{suffix}.npy';tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
                tmp.replace(path)
            stats={'keys':len(keys),'omitted_common_keys':int((~allowed).sum()),'postings':len(ids)}
            atomic_json(ready,stats)
            del data,keys,starts,counts,take,ids,offsets,lengths
        # Sorted bucket is durable before removing its temporary raw input.
        raw.unlink(missing_ok=True)
        kept_keys+=stats['keys'];omitted_keys+=stats['omitted_common_keys'];kept_postings+=stats['postings']
        print(f'{split} lexical index: sorted bucket {b+1}/{buckets}',flush=True)
    atomic_json(root/'complete.json',{'engine':'lexical','full_target_pool':True,'targets':total,'buckets':buckets,
      'kept_keys':kept_keys,'omitted_common_keys':omitted_keys,'postings':kept_postings,'max_key_postings':cap,
      'country_block':cfg['country_block'],'array_bytes':sum(p.stat().st_size for p in root.glob('*.npy'))})
    return root


class Retriever:
    def __init__(self,root,cfg):
        self.root=Path(root);self.cfg=cfg;self.meta=json.loads((self.root/'complete.json').read_text());self.arrays={}
        # Reuse the existing immutable read-through cache, then mmap the arrays locally.
        from .blocking import Retriever as CacheHelper
        for b in range(self.meta['buckets']):
            arrays=[]
            for suffix in ('keys','offsets','ids'):
                path=CacheHelper.local_file(self,self.root/f'{b:03d}_{suffix}.npy')
                arrays.append(np.load(path,mmap_mode='r',allow_pickle=False))
            self.arrays[b]=arrays

    def query(self,rows,checkpoint=None):
        # Bounded query batches finish independently; no scan across all targets per query.
        k=self.cfg['top_k_per_view'];result=[]
        for row in rows:
            evidence=[];weights=[];views=[];den=np.zeros(3)
            for key,view in key_views(row,self.cfg['country_block']).items():
                keys,offsets,ids=self.arrays[key%self.meta['buckets']]
                i=int(np.searchsorted(keys,np.uint64(key)))
                if i>=len(keys) or int(keys[i])!=key:continue
                lo,hi=int(offsets[i]),int(offsets[i+1]);weight=np.log1p(self.meta['targets']/(hi-lo))
                evidence.append(ids[lo:hi]);weights.append(np.full(hi-lo,weight));views.append(np.full(hi-lo,view,dtype=np.int8));den[view]+=weight
            if not evidence:result.append({});continue
            unique,inverse=np.unique(np.concatenate(evidence),return_inverse=True)
            w=np.concatenate(weights);v=np.concatenate(views);scores=[]
            for view in range(3):
                mask=v==view
                scores.append(np.bincount(inverse[mask],weights=w[mask],minlength=len(unique))/max(den[view],1e-9))
            # Joint keys contribute to the combined view; independent views preserve missing-address routes.
            scores[2]=.4*scores[0]+.3*scores[1]+.3*scores[2]
            selected={}
            for view,score in enumerate(scores):
                ix=np.flatnonzero(score>0)
                if len(ix)>k:
                    cutoff=np.partition(score[ix],len(ix)-k)[len(ix)-k];ix=ix[score[ix]>=cutoff]
                order=np.lexsort((unique[ix],-score[ix]))[:k]
                for rank,j in enumerate(ix[order],1):
                    f=selected.setdefault(int(unique[j]),[0.]*6);f[view]=float(score[j]);f[view+3]=1./rank
            result.append(selected)
        return result


def resolve_ids(db,split,result):
    rids=sorted({rid for candidates in result for rid in candidates});mapping={}
    for lo in range(0,len(rids),500):
        chunk=rids[lo:lo+500]
        for row in db.execute('SELECT rowid AS rid,id FROM records WHERE split=? AND rowid IN ('+','.join('?'*len(chunk))+')',[split]+chunk):mapping[row['rid']]=row['id']
    if len(mapping)!=len(rids):raise ValueError('Posting index does not match the normalized records database')
    return [{mapping[rid]:feat for rid,feat in candidates.items()} for candidates in result]
