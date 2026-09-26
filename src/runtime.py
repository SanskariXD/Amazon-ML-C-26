"""Resource detection and atomic, fingerprinted checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import psutil


def resources(path='.'):
    gpu = None
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    m = psutil.virtual_memory()
    return {'ram_total_gb':round(m.total/2**30,2), 'ram_available_gb':round(m.available/2**30,2),
            'disk_free_gb':round(shutil.disk_usage(path).free/2**30,2), 'cpu_cores':os.cpu_count(), 'gpu':gpu}


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8'); tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def signature(config, dataset_manifest):
    h=hashlib.sha256(json.dumps([config,dataset_manifest],sort_keys=True).encode())
    for p in sorted(Path(__file__).parent.glob('*.py')): h.update(p.read_bytes())
    return h.hexdigest()[:16]
