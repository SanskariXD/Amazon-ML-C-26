"""Tune decisions on a dedicated group. Independent Bernoulli expected-F is optional."""
import numpy as np
from .metric import evaluate


def expected_f(probabilities):
    """Exact expectation conditional on calibrated independent candidate probabilities.
    Missing true targets outside retrieval are NOT modeled; compare on held-out data.
    """
    p=np.asarray(probabilities,dtype=float)
    if np.any((p<0)|(p>1)):raise ValueError('Invalid probability')
    order=np.argsort(-p,kind='stable');p=p[order];n=len(p)
    prefix=[np.array([1.])]
    for x in p:prefix.append(np.convolve(prefix[-1],[1-x,x]))
    suffix=[None]*(n+1);suffix[n]=np.array([1.])
    for i in range(n-1,-1,-1):suffix[i]=np.convolve(suffix[i+1],[1-p[i],p[i]])
    best=float(np.prod(1-p));best_k=0
    for k in range(1,n+1):
        a=np.arange(k+1)[:,None];b=np.arange(n-k+1)[None,:]
        score=float(np.sum(prefix[k][:,None]*suffix[k][None,:]*(1.25*a/(k+.25*(a+b)))))
        if score>best:best,best_k=score,k
    return order[:best_k]


def predict(frame, method='threshold', threshold=.8):
    result={}
    for q,g in frame.groupby('q',sort=False):
        if method=='expected_f':selected=g.iloc[expected_f(g.p.to_numpy())]
        else:selected=g[g.p>=threshold]
        result[q]=set(selected.t)
    return result


def tune(frame, truth, allow_expected=False):
    options=[]
    for threshold in np.linspace(.05,.99,95):
        pred=predict(frame,threshold=float(threshold))
        options.append((evaluate(truth,pred)['macro_f0.5'],float(threshold)))
    score,threshold=max(options)
    rule={'method':'threshold','threshold':threshold,'tune_macro_f0.5':score}
    if allow_expected:
        score_ef=evaluate(truth,predict(frame,method='expected_f'))['macro_f0.5']
        rule['expected_f_tune_score']=score_ef
        if score_ef>score:rule.update({'method':'expected_f','tune_macro_f0.5':score_ef})
    return rule
