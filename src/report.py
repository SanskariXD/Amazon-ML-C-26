"""Export compact run reports for review; excludes raw records, features and models."""
import json
from pathlib import Path
import zipfile
from .runtime import atomic_json


def export_report(work):
    work=Path(work);path=work/'review_report.zip'
    names=['config.json','dataset_profile_preliminary.json','dataset_profile.json','runtime.json','dataset_manifest.json',
      'experiments/blocking_benchmark.json','experiments/dev.json','experiments/baseline_run.json',
      'experiments/results.csv','experiments/loco_summary.json','models/decision.json','outputs/official_validation.json']
    present=[name for name in names if (work/name).is_file()]
    with zipfile.ZipFile(str(path)+'.tmp','w',zipfile.ZIP_DEFLATED) as z:
        for name in present:z.write(work/name,name)
        z.writestr('status.json',json.dumps({'available_reports':present,'development_score_available':'experiments/dev.json' in present,
          'holdout_excluded':True,'raw_records_excluded':True,'errors_examples_excluded':True},indent=2))
    Path(str(path)+'.tmp').replace(path)
    return path
