"""Choose an explicitly configured retrieval engine."""
from . import blocking,lexical


def build_index(db,split,root,cfg):
    engine=cfg.get('retrieval_engine','tfidf')
    if engine not in ('tfidf','lexical'):raise ValueError('Unknown retrieval engine')
    return (lexical if engine=='lexical' else blocking).build_index(db,split,root,cfg)


class Retriever:
    def __init__(self,db,split,root,cfg):
        self.db=db;self.split=split;self.lexical=cfg.get('retrieval_engine','tfidf')=='lexical'
        self.engine=(lexical if self.lexical else blocking).Retriever(root,cfg)
    def query(self,rows,checkpoint=None):
        result=self.engine.query(rows,checkpoint)
        return lexical.resolve_ids(self.db,self.split,result) if self.lexical else result
