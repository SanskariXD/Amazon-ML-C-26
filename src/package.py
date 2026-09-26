"""Package only validated outputs, runnable code and a completed methodology."""
from pathlib import Path
import zipfile


def package(work):
    root=Path(__file__).resolve().parent.parent
    doc=work/'Documentation_template.md'
    if not doc.exists() or 'TODO' in doc.read_text():
        raise ValueError('Complete work/Documentation_template.md using the organizer template and measured results before packaging')
    dest=work/'AmazonML2026_submission.zip'
    with zipfile.ZipFile(str(dest)+'.tmp','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('matching_results.tsv','candidate_pairs.tsv'):z.write(work/'outputs'/name,'output/'+name)
        z.write(doc,'Documentation_template.md')
        for d in ('src','configs','notebooks'):
            for p in (root/d).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,'code/business_entity_resolution/'+str(p.relative_to(root)))
        for name in ('README.md','requirements.txt'):z.write(root/name,'code/business_entity_resolution/'+name)
        z.write(work/'config.json','code/business_entity_resolution/configs/submitted.json')
    Path(str(dest)+'.tmp').replace(dest)
    return dest
