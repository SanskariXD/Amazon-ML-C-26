"""Inner entity OOF stage-1 context; external tune/dev/holdout never fit either stage."""
import numpy as np
import lightgbm as lgb
from .features import context
from .split import hash_id
from .train_stage1 import train


def train_stack(X,y,queries,cfg,model_dir):
    folds=np.array([hash_id(q,cfg['seed']+1)%cfg['folds'] for q in queries])
    oof=np.zeros(len(y),dtype=np.float32);models=[]
    for f in range(cfg['folds']):
        tr=folds!=f;va=~tr
        if not va.any() or not tr.any():raise ValueError('Insufficient entities for inner folds')
        path=model_dir/f'stage1_fold{f}.txt'
        if path.exists():model=lgb.Booster(model_file=str(path))
        else:
            model=train(X[tr],y[tr],cfg)
            model.save_model(str(path)+'.tmp');__import__('pathlib').Path(str(path)+'.tmp').replace(path)
        oof[va]=model.predict(X[va],num_threads=4);models.append(model)
    stage2=train(context(X,oof,queries),y,cfg)
    path=model_dir/'stage2.txt'
    stage2.save_model(str(path)+'.tmp');__import__('pathlib').Path(str(path)+'.tmp').replace(path)
    return models,stage2


def score(models,stage2,X,queries):
    p=np.mean([m.predict(X,num_threads=4) for m in models],axis=0)
    return stage2.predict(context(X,p,queries),num_threads=4) if stage2 else p
