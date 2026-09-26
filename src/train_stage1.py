"""CPU LightGBM, native text serialization and bounded training samples."""
import os
import lightgbm as lgb


def train(X,y,cfg):
    if len(set(y))<2:raise ValueError('Training requires retrieved positives AND negatives; inspect blocking')
    model=lgb.LGBMClassifier(n_estimators=cfg['rounds'],num_leaves=cfg['num_leaves'],
      min_child_samples=cfg['min_child_samples'],learning_rate=.06,colsample_bytree=.9,
      reg_lambda=1.,verbosity=-1,n_jobs=min(os.cpu_count() or 2,4),random_state=cfg['seed'],
      deterministic=True,force_col_wise=True)
    model.fit(X,y)
    return model.booster_
