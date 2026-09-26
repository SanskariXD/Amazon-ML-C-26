# Amazon ML Challenge 2026 — independent Colab baseline

An independent, CPU-compatible business entity-resolution pipeline for **SanskariXD/Amazon-ML-C-26**. This is working baseline infrastructure, not a claimed leaderboard-winning model. No reference implementation or supplied model weights were copied into this repository.

**Status:** synthetic correctness/resume tests; real dataset audit, training and competition scores pending dataset access and execution. Read [the reference audit](research/reference_repo_audit.md) before interpreting anyone's reported scores.

## Start in Colab

Open `notebooks/Amazon_ML_2026_Master.ipynb` using Colab's GitHub picker with this repository. Choose a standard CPU runtime; GPU is optional and unused by the baseline. Run the cells in order. The notebook mounts Google Drive, reads `MyDrive/6ab10eb3b23ba_student_resource.zip`, installs pinned dependencies and runs repository scripts. Update `ZIP_PATH` only if needed.

GitHub stores code. Colab computes. Google Drive stores data and completed work. The original ZIP and TSVs remain unchanged. Do not commit datasets, credentials, candidate shards or trained artifacts.

## Local / command-line execution

Python 3.11 or 3.12 recommended, Linux/Colab. From this repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.smoke
python -m src.prepare --zip /path/6ab10eb3b23ba_student_resource.zip --destination /persistent/AmazonML2026/data/raw
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage audit
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage baseline
```

`baseline` builds full-target indexes, retrieves candidates for deterministic entity samples, computes features, fits LightGBM, tunes a threshold on a separate group and evaluates development entities. It **does not** open the final holdout or infer test outputs automatically.

```bash
# Only after architecture selection; this group must remain untouched during experiments.
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage holdout
# Freeze your selected config; the same config is required for every stage of a run.
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage infer
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage validate
```

Pass `--config configs/stage2.json` to test inner-OOF two-stage LightGBM. A model trained on one config is never silently substituted for another. Default inference uses the sampled fitted model, not a claimed full-data retrain. Increase the entity sample only after measuring the learning curve and RAM budget; LightGBM fitting is guarded, not out-of-core.

The CLI prints its run directory, and `runs/latest_run.json` records it. Code+config+raw-file hashes identify a run. Each run holds `records.sqlite`, `dataset_profile.json`, `candidates/index/`, `features/`, `models/`, `checkpoints/`, `experiments/` and `outputs/`. Scored candidate rows are exactly the rows exported to `candidate_pairs.tsv`, including rejected matches. Empty-candidate and singleton queries still get output rows.

`infer` runs internal validation and the organizer's `utils/validate_submission.py` from the original resource bundle, adding `--check-ids` when supported. Missing or failing organizer validation is a visible error. It does not submit anything to the portal. Only upload the resulting `matching_results.tsv` manually after checking the validation result.

## Design and validation

- Exact per-entity macro F0.5 includes all S1, with correct empty-set behavior.
- Stable hash partitions: 70% fitting, 10% decision tuning, 10% development, 10% final holdout. Samples are taken by entity, keeping full candidate groups. Shared true targets are detected and require connected-component splitting before training.
- All train S2/S3 remain in the retrieval universe. Vocabulary sampling is label-free and does not remove target records. Country filtering defaults off. Test countries are open strings, including France.
- Three views: character name, character address, word name+address. Top-k is retained globally across target shards and unioned; default upper bound is 60 candidates/S1. No forced ground-truth injection.
- Label-free Unicode and transliterated views, explicit missing/numeric signals, native LightGBM model text. No downloaded business enrichment and no untrusted pickle loads.
- Optional two-stage model uses stage-1 entity OOF predictions inside the fit group for query context. External tuning/development/holdout groups never fit either stage. Graph-wide competition, reverse retrieval, exclusivity, ensembles, LOCO and neural reranking remain future experiments.
- Precision/recall diagnostics are micro pair-level; the optimized score is macro entity F0.5. Candidate entity recall means at least one true candidate for non-singletons; complete-entity recall is reported separately.
- Expected-F is optional (`"expected_f": true`): exact under independent calibrated candidate probabilities, not a guarantee under missing retrieval candidates. Default is a tune-selected global threshold.

## Resume and resource limits

Completed query shards, index shards and model folds are reused after reconnecting. Partial files use temporary names; completion markers are written last. Raw hashes are recomputed when a stage starts, so changes invalidate stale work. Different code/config creates a new run; old artifacts are preserved. Reconnect with the same config and code to resume.

Default target shards are 25k rows and query batches 128; first-run sizing reduces these to fit available memory. Effective sizing is saved, so a later runtime's memory does not change shard boundaries. A smaller runtime can require a new smaller config. Pair tables are on disk. Training uses 15k fitting S1 by default and rejects a sample estimated to exceed 60% of available RAM.

**Full-scale runtime and disk usage are unmeasured.** Searching every target shard per query batch is deliberately bounded but may be slow. Benchmark before launching millions of queries. At 60 candidates/S1, 1M S1 has up to 6.48 GB of raw numeric features alone, plus IDs/indexes/scores/TSVs. Check both Drive quota and Colab local disk; filesystem free-space reports may not reflect Drive account quota. Free Drive storage may not be enough for all artifacts. See stage estimates in the audit.

## Experiments and packaging

Each run writes `experiments/results.csv`, metric JSON and up to 100 error examples. The repository's `experiments/results.csv` is a schema-only template, not fabricated results. Record IDs E001, E002, etc. Compare development scores and candidate recall/cost before promoting a change; never select repeatedly against the final holdout or public leaderboard.

Fill the organizer's `Documentation_template.md` using measured results and save it at the printed run directory. Then:

```bash
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage package
```

The package includes validated output, runnable code, pinned requirements, the effective configuration and your completed document. Reproduce using `--config configs/submitted.json`. Final-data retraining and optional research ideas must be validated before claiming them in that document.

## Layout

- `src/`: audit, normalization, splitting, blocking, features, metrics, training, decisions, submission, packaging and CLI
- `configs/`: single-stage and optional two-stage experiment settings
- `notebooks/Amazon_ML_2026_Master.ipynb`: Drive/Colab orchestration
- `tests/`: exact metric and decision correctness checks
- `research/`: reference audit and immutable snapshot hashes
- `experiments/`: tracking schema and synthetic verification evidence

Model library: LightGBM (MIT). Repository implementation is independently written for this project; reference source-code licence uncertainties and provenance are documented in the audit. Organizer-provided utilities remain with the user's resource bundle.
