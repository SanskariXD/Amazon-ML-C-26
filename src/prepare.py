"""Safely extract the user's ZIP to persistent storage with per-file resume."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from .runtime import atomic_json,sha256


def extract(archive,destination):
    archive=Path(archive);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    digest=sha256(archive);root=destination/digest[:16];root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        total=sum(i.file_size for i in files if not (root/i.filename).exists())
        if total>shutil.disk_usage(root).free*.9:raise RuntimeError('Not enough disk for extraction')
        for info in files:
            path=(root/info.filename).resolve()
            if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe ZIP member path')
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink ZIP member')
            if path.exists() and path.stat().st_size==info.file_size:continue
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.extracting')
            with z.open(info) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            tmp.replace(path)
    matches=list(root.rglob('train_source1.tsv'))
    if len(matches)!=1:raise ValueError('Expected one official dataset/train/train_source1.tsv')
    dataset=matches[0].parent.parent
    atomic_json(destination/'dataset_location.json',{'archive_sha256':digest,'dataset':str(dataset.resolve())})
    return dataset

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--destination',required=True);a=p.parse_args()
    print(extract(a.zip,a.destination))
