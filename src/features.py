"""Deterministic similarity features; missing evidence is explicitly marked."""
import numpy as np
from rapidfuzz import fuzz
from .normalize import numbers

NAMES=['name_ratio','name_token_sort','name_token_set','addr_ratio','addr_token_sort','addr_token_set',
 'unicode_name_ratio','unicode_addr_ratio','name_exact','addr_exact','name_missing','addr_missing',
 'number_jaccard','number_conflict','number_missing','country_equal','target_source3',
 'name_length_ratio','addr_length_ratio','name_addr_min','name_addr_product',
 'retrieval_name','retrieval_address','retrieval_combined','rank_name','rank_address','rank_combined']


def pair(q,t,retrieval):
    n,m=q['latin_name'],t['latin_name'];a,b=q['latin_address'],t['latin_address']
    def ratio(fn,x,y):return fn(x,y)/100 if x and y else 0.
    ns=[ratio(fn,n,m) for fn in (fuzz.ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio)]
    ads=[ratio(fn,a,b) for fn in (fuzz.ratio,fuzz.token_sort_ratio,fuzz.token_set_ratio)]
    na,nb=numbers(a),numbers(b)
    return ns+ads+[ratio(fuzz.ratio,q['name'],t['name']),ratio(fuzz.ratio,q['address'],t['address']),
      float(bool(n) and n==m),float(bool(a) and a==b),float(not n or not m),float(not a or not b),
      len(na&nb)/len(na|nb) if na|nb else 0.,float(bool(na and nb) and not na&nb),float(not na or not nb),
      float(bool(q['country']) and q['country']==t['country']),float(t['source']==3),
      min(len(n),len(m))/max(len(n),len(m),1),min(len(a),len(b))/max(len(a),len(b),1),min(ns[0],ads[0]),ns[0]*ads[0]]+retrieval


def target_records(db,split,ids):
    ids=list(ids);result={}
    for i in range(0,len(ids),500):
        chunk=ids[i:i+500]
        for r in db.execute('SELECT * FROM records WHERE split=? AND id IN ('+','.join('?'*len(chunk))+')',[split]+chunk):result[r['id']]=r
    return result


def context(X, probabilities, queries):
    """Query-group context only. No ground-truth or partial-graph target margins."""
    out=np.zeros((len(X),5),dtype=np.float32)
    groups={}
    for i,q in enumerate(queries):groups.setdefault(q,[]).append(i)
    for indices in groups.values():
        ix=np.array(indices);p=probabilities[ix];order=np.argsort(-p,kind='stable')
        ranks=np.empty(len(ix));ranks[order]=np.arange(1,len(ix)+1)
        out[ix]=np.column_stack([p,np.full(len(ix),p.max()),p.max()-p,1/ranks,np.full(len(ix),p.sum())])
    return np.concatenate([X,out],axis=1)
