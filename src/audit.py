"""Stream official TSVs into an indexed working database; originals stay unchanged."""
import csv
import json
from pathlib import Path
import sqlite3
from .normalize import record
from .split import hash_id, partition
from .runtime import atomic_json, sha256

FIELDS=['entity_id','business_name','business_address','country']


def source_files(dataset):
    d=Path(dataset)
    return [d/s/f'{s}_source{k}.tsv' for s in ('train','test') for k in (1,2,3)]+[d/'train/train_ground_truth.tsv']


def manifest(dataset):
    result={}
    for p in source_files(dataset):
        if not p.is_file(): raise FileNotFoundError(p)
        result[str(p.relative_to(dataset))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    return result


def connect(path):
    db=sqlite3.connect(path); db.row_factory=sqlite3.Row
    db.execute('PRAGMA cache_size=-65536');db.execute('PRAGMA temp_store=FILE')
    return db


def read_tsv(path, expected):
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames != expected: raise ValueError(f'Unexpected header in {path}: {reader.fieldnames}')
        for row in reader:
            if None in row or any(v is None for v in row.values()): raise ValueError(f'Malformed TSV row in {path}')
            yield row


def build(dataset, work, seed):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    dest=work/'records.sqlite'
    if (work/'dataset_profile.json').exists() and dest.exists(): return dest
    tmp=work/'records.tmp.sqlite'
    db=connect(tmp)
    db.execute('CREATE TABLE IF NOT EXISTS records (split TEXT,id TEXT,source INTEGER,country TEXT,name TEXT,address TEXT,latin_name TEXT,latin_address TEXT,part TEXT,h INTEGER,PRIMARY KEY(split,id))')
    db.execute('CREATE TABLE IF NOT EXISTS truth(q TEXT,t TEXT,PRIMARY KEY(q,t))')
    db.execute('CREATE TABLE IF NOT EXISTS truth_entities(q TEXT PRIMARY KEY)')
    db.execute('CREATE TABLE IF NOT EXISTS ingest(file TEXT PRIMARY KEY,rows INTEGER)')
    for s in ('train','test'):
        for k in (1,2,3):
            key=f'{s}_source{k}.tsv'
            progress=db.execute('SELECT rows FROM ingest WHERE file=?',(key,)).fetchone()
            done=progress[0] if progress else 0
            batch=[];position=done
            for position,row in enumerate(read_tsv(Path(dataset)/s/key,FIELDS),1):
                if position<=done:continue
                if not row['entity_id'].startswith(f'S{k}-'): raise ValueError('Incorrect source prefix')
                r=record(row)
                batch.append((s,r['entity_id'],k,r['country'],r['name'],r['address'],r['latin_name'],r['latin_address'],partition(r['entity_id'],seed),hash_id(r['entity_id'],seed)))
                if len(batch)>=10000:
                    db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',batch)
                    db.execute('INSERT OR REPLACE INTO ingest VALUES (?,?)',(key,position));db.commit();batch=[]
            db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',batch)
            db.execute('INSERT OR REPLACE INTO ingest VALUES (?,?)',(key,position));db.commit()
            print(f'Prepared {s} source {k}',flush=True)
    key='train_ground_truth.tsv'
    progress=db.execute('SELECT rows FROM ingest WHERE file=?',(key,)).fetchone()
    done=progress[0] if progress else 0;position=done
    for position,row in enumerate(read_tsv(Path(dataset)/'train'/key,['source1_entity_id','matched_entity_ids']),1):
        if position<=done:continue
        q=row['source1_entity_id'];db.execute('INSERT INTO truth_entities VALUES (?)',(q,))
        targets=[x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
        if len(targets)!=len(set(targets)):raise ValueError('Duplicate ground-truth target')
        db.executemany('INSERT INTO truth VALUES (?,?)',[(q,t) for t in targets])
        if position%10000==0:
            db.execute('INSERT OR REPLACE INTO ingest VALUES (?,?)',(key,position));db.commit()
    db.execute('INSERT OR REPLACE INTO ingest VALUES (?,?)',(key,position));db.commit()
    db.execute('CREATE INDEX IF NOT EXISTS records_query ON records(split,source,part,h)')
    db.execute('CREATE INDEX IF NOT EXISTS records_country ON records(split,country,source,id)')
    db.execute('CREATE INDEX IF NOT EXISTS truth_target ON truth(t)');db.commit()
    checks={
      'missing_truth_rows':db.execute("SELECT COUNT(*) FROM records r LEFT JOIN truth_entities g ON r.id=g.q WHERE r.split='train' AND r.source=1 AND g.q IS NULL").fetchone()[0],
      'invalid_truth_queries':db.execute("SELECT COUNT(*) FROM truth_entities g LEFT JOIN records r ON r.split='train' AND r.id=g.q AND r.source=1 WHERE r.id IS NULL").fetchone()[0],
      'invalid_truth_targets':db.execute("SELECT COUNT(*) FROM truth g LEFT JOIN records r ON r.split='train' AND r.id=g.t AND r.source IN (2,3) WHERE r.id IS NULL").fetchone()[0],
      'cross_country_pairs':db.execute("SELECT COUNT(*) FROM truth g JOIN records q ON q.split='train' AND q.id=g.q JOIN records t ON t.split='train' AND t.id=g.t WHERE q.country != t.country OR q.country='' OR t.country=''").fetchone()[0],
      'shared_targets':db.execute('SELECT COUNT(*) FROM (SELECT t FROM truth GROUP BY t HAVING COUNT(*)>1)').fetchone()[0]}
    if any(checks[k] for k in ('missing_truth_rows','invalid_truth_queries','invalid_truth_targets')):raise ValueError(checks)
    # Entity splitting is not safe when the same labelled target crosses entity groups.
    if checks['shared_targets']:raise ValueError('Shared labelled targets found; implement connected-component splitting before training')
    counts=[dict(r) for r in db.execute("SELECT split,source,country,COUNT(*) n,SUM(name='') empty_name,SUM(address='') empty_address FROM records GROUP BY split,source,country")]
    singleton=db.execute('SELECT COUNT(*) FROM truth_entities WHERE q NOT IN (SELECT q FROM truth)').fetchone()[0]
    db.close();tmp.replace(dest)
    atomic_json(work/'dataset_profile.json',{'counts':counts,'checks':checks,'singletons':singleton,'raw_files_unchanged':True})
    return dest


def truth_for(db, ids):
    result={q:set() for q in ids}
    for i in range(0,len(ids),500):
        chunk=ids[i:i+500]
        for r in db.execute('SELECT q,t FROM truth WHERE q IN ('+','.join('?'*len(chunk))+')',chunk):result[r['q']].add(r['t'])
    return result
