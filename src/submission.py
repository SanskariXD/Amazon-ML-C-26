"""Stream TSV output; verify all IDs, exact coverage and candidate subset."""
import csv
from pathlib import Path
from .audit import read_tsv


def validate(matching,candidates,db):
    qs=db.execute("SELECT id FROM records WHERE split='test' AND source=1 ORDER BY id")
    mr=read_tsv(matching,['source1_entity_id','matched_entity_ids'])
    cr=read_tsv(candidates,['source1_entity_id','candidate_entity_ids'])
    from itertools import zip_longest
    count=0
    for q,m,c in zip_longest(qs,mr,cr):
        if q is None or m is None or c is None:raise ValueError('Output coverage mismatch')
        if q['id']!=m['source1_entity_id'] or q['id']!=c['source1_entity_id']:raise ValueError('IDs missing/duplicate/out of order')
        def ids(value):
            values=value.split(',') if value else []
            if len(values)!=len(set(values)):raise ValueError('Duplicate target')
            return set(values)
        a,b=ids(m['matched_entity_ids']),ids(c['candidate_entity_ids'])
        if not a<=b:raise ValueError('Final matches outside scored candidate set')
        for t in b:
            if not db.execute("SELECT 1 FROM records WHERE split='test' AND source IN (2,3) AND id=?",(t,)).fetchone():raise ValueError(f'Invalid target {t}')
        count+=1
    return {'pass':True,'entities':count,'final_subset_of_candidates':True}
