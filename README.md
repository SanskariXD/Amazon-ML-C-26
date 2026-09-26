# Amazon ML Challenge 2026 — independent Colab baseline

An independent, CPU-compatible business entity-resolution pipeline for **SanskariXD/Amazon-ML-C-26**. This is working baseline infrastructure, not a claimed leaderboard-winning model. No reference implementation or supplied model weights were copied into this repository.

**Status:** 13 correctness tests and synthetic end-to-end/resume checks pass. The provided private Drive ZIP is confirmed (1,094,823,222 bytes), but exceeds the connector’s 256 MiB download limit. Real-data execution must currently run in your mounted Colab Drive; competition scores remain unmeasured. Read [the reference audit](research/reference_repo_audit.md) before interpreting anyone's reported scores.

## Start in Colab

Open `notebooks/Amazon_ML_2026_Master.ipynb` using Colab's GitHub picker with this repository. Choose a standard CPU runtime; GPU is optional and unused by the baseline. Run the cells in order. The notebook mounts Google Drive, reads `MyDrive/6ab10eb3b23ba_student_resource.zip`, installs pinned dependencies and runs repository scripts. Update `ZIP_PATH` only if needed. **For the first real-data run, run through section 7 and upload the downloaded `review_report.zip` to ChatGPT before launching full training/inference.** The first report contains streamed counts, missing-field statistics and match-count distribution, not raw business records. Section 10 adds full-pool blocking measurements after the initial size review.

GitHub stores code. Colab computes. Google Drive stores data and completed work. The original ZIP and TSVs remain unchanged. Do not commit datasets, credentials, candidate shards or trained artifacts.

## Local / command-line execution

Python 3.11 or 3.12 recommended, Linux/Colab. From this repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.smoke
python -m src.prepare --zip /path/6ab10eb3b23ba_student_resource.zip --destination /persistent/AmazonML2026/data/raw
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage profile
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage report
# After reviewing data size, run the full relational audit.
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage audit
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage benchmark
python -m src.pipeline --dataset /path/student_resource/dataset --work /persistent/AmazonML2026/runs --stage report
# Review candidate recall, throughput and projected disk/time first.
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

The CLI prints its run directory, and `runs/latest_run.json` records it. Code+config+raw-file hashes identify a run. Each run holds `dataset_profile.json`, `artifact_paths.json`, `models/`, `checkpoints/`, `experiments/` and `outputs/`. Shared normalized SQLite data, indexes and features live under `runs/_shared/`; `artifact_paths.json` contains their exact paths. Single-stage and two-stage experiments with identical retrieval/sample settings reuse those artifacts. Scored candidate rows are exactly the rows exported to `candidate_pairs.tsv`, including rejected matches. Empty-candidate and singleton queries still get output rows.

`infer` runs internal validation and the organizer's `utils/validate_submission.py` from the original resource bundle, adding `--check-ids` when supported. Missing or failing organizer validation is a visible error. It does not submit anything to the portal. Only upload the resulting `matching_results.tsv` manually after checking the validation result.

## Design and validation

- Exact per-entity macro F0.5 includes all S1, with correct empty-set behavior.
- Stable hash partitions: 70% fitting, 10% decision tuning, 10% development, 10% final holdout. Samples are taken by entity, keeping full candidate groups. Shared true targets are detected and require connected-component splitting before training.
- All train S2/S3 remain in the retrieval universe. Vocabulary sampling is label-free and does not remove target records. Country filtering defaults off. Test countries are open strings, including France.
- Three views: character name, character address, word name+address. Top-k is retained globally across target shards and unioned; default upper bound is 60 candidates/S1. No forced ground-truth injection.
- Label-free Unicode and transliterated views, explicit missing/numeric signals, native LightGBM model text. No downloaded business enrichment and no untrusted pickle loads.
- Optional two-stage model uses stage-1 entity OOF predictions inside the fit group for query context. External tuning/development/holdout groups never fit either stage. Strict leave-country-out testing is available with `--stage loco`: only the source country supplies fit and threshold-tuning labels; target-country development labels are used for evaluation. Graph-wide competition, reverse retrieval, exclusivity, ensembles and neural reranking remain future experiments.
- Precision/recall diagnostics are micro pair-level; the optimized score is macro entity F0.5. Candidate entity recall means at least one true candidate for non-singletons; complete-entity recall is reported separately.
- Expected-F is optional (`"expected_f": true`): exact under independent calibrated candidate probabilities, not a guarantee under missing retrieval candidates. Default is a tune-selected global threshold.

## Resume and resource limits

Completed query shards, index shards and model folds are reused after reconnecting. Long retrieval passes checkpoint their best candidates every ~30 seconds or at view completion; normalization saves durable SQLite snapshots about every two minutes when using local cache. Partial files use temporary names; completion markers are written last. Raw hashes are recomputed when a stage starts, so changes invalidate stale work. Different code/config creates a new model run; old artifacts are preserved. Stage-specific fingerprints let compatible normalization, indexes and features be reused across model experiments. Installed numerical-library versions are included in fingerprints. Reconnect with the same config and code to resume.

Default target shards are 25k rows, query multiplication batches 128 and retrieval batches 4,096; first-run sizing reduces these to fit available memory. Effective sizing is saved, so a later runtime's memory does not change shard boundaries. A smaller runtime can require a new smaller config. Pair tables are on disk. Training uses 15k fitting S1 by default and rejects a sample estimated to exceed 60% of available RAM.

**Full-scale runtime and disk usage are unmeasured.** Each target shard is loaded once per retrieval batch, with smaller matrix multiplications inside that batch. This avoids reloading it for each 128-query block. Set `ER_LOCAL_CACHE=/content/amazon-ml-local-cache` (the notebook does this) for local SQLite processing and index reads, while durable snapshots remain on Drive. Index caching falls back to Drive when local disk is low. The implementation is still exhaustive sparse retrieval and must be benchmarked at full target density. Benchmark before launching millions of queries. At 60 candidates/S1, 1M S1 has up to 6.48 GB of raw numeric features alone, plus IDs/indexes/scores/TSVs. Check both Drive quota and Colab local disk; filesystem free-space reports may not reflect Drive account quota. Free Drive storage may not be enough for all artifacts. See stage estimates in the audit.

## Experiments and packaging

Each run writes `experiments/results.csv`, metric JSON and up to 100 error examples. `--stage benchmark` compares top 10/20/30/50 per view using only fitting queries against all train targets; it reports recall, candidate distributions, RAM and approximate time/storage projections. `--stage report` exports `review_report.zip` with compact metrics/configs, excluding raw records, models, error examples and holdout results. The repository's `experiments/results.csv` is a schema-only template, not fabricated results. Record IDs E001, E002, etc. Compare development scores and candidate recall/cost before promoting a change; never select repeatedly against the final holdout or public leaderboard.

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


## September 26 scalability update

- Verified candidate equivalence to the prior implementation; synthetic 36-query benchmark reduced matrix loads from 27 to 3. This is not a real-data speed claim.
- Replaced sorting all sparse matches with a cutoff followed by deterministic tie-breaking.
- Added interruption-safe retrieval progress, local database/index caching and shared stage artifacts.
- Added full-pool blocking sweeps, strict country-transfer evaluation and compact report export.
- Saved the chosen code commit in Drive so reconnecting does not silently upgrade the source.

No model is promoted based on these engineering tests. Use measured real-data development F0.5 and resource cost before selecting an architecture. The untouched holdout remains closed during these experiments.

### Continue after the uploaded dataset profile

[Open the continuation notebook in Colab](https://colab.research.google.com/github/SanskariXD/Amazon-ML-C-26/blob/main/notebooks/Amazon_ML_2026_Continue.ipynb) and run all cells. It reuses extracted data, preserves the master notebook, and downloads an audit/retrieval report. The experimental lexical engine avoids exhaustive target scans per query; real recall and runtime must be measured before promotion. See [profile review](research/data_profile_review.md). The original TF-IDF baseline remains available. Reconnect with the continuation notebook to resume its pinned revision.
