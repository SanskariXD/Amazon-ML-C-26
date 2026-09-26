# Reference repository audit — 26 September 2026

Scope: static inspection of the three user-supplied ZIP snapshots, their executable Python/notebook code, configuration, reports and documentation. No supplied pickle or pretrained model was loaded. Scores below are author-reported, not reproduced. Snapshot hashes are in `reference_manifest.json`. These snapshots contain no Git history, so commit counts and exact upstream commits cannot be verified from them.

## Comparison

| Aspect | Sank19: ML_hackathon-main | Aamod007: Business-Entity-Resolution | AyanAhmedKhan: amazon-ml-challenge-main |
|---|---|---|---|
| Actual architecture | Label-derived transliteration, lexical/char sparse top-k, pair features, cross-fitted LightGBM, group/sibling features, second LightGBM, decision rules | Uploaded implementation is a notebook: inverted lexical/postal index, 72-position feature vector, XGBoost/LightGBM/CatBoost, optional logistic meta-learner, fast rules and target conflict resolution | Five-view TF-IDF including reverse retrieval; optional static multilingual embeddings; LightGBM cross-fit; collective features; second LightGBM |
| Normalization | Extensive aliases, street forms, numeric/OCR transforms, learned cross-script map | Unicode/transliteration and frequency-aware name/address features | Learned native-script/abbreviation maps, deterministic Unicode/script handling |
| Candidate generation | Per-country combined/name/address top-k union; compact hashed sparse features | Name-token/postal inverted indices; capped candidate lists | Multiple name/address/combined/reverse sparse retrieval views |
| Features | README reports 74 base features; code adds group/competition/sibling features | README describes 55; current notebook pads to 72, illustrating documentation drift | Documentation reports 87 initial features plus about 20 collective features |
| Training | Two entity fold ensembles; defaults sample 12% of S1 for each stage | 15k S1 training sample, grouped five-fold model OOF, all truth positives plus ten graded retrieval negatives per query | Three entity folds, separate evaluation fold, density variants |
| Validation | Cross-fitted model scores plus separate decision-tuning halves | 10k cold S1 from a different row slice; ground-truth targets plus fixed background slices | Full target universe; numeric-ID fold assignment; dense/LOCO variants |
| Reported local score | 0.9828865 in shipped train_summary.json | README held-out 0.98006; CV mean 0.97342 | Notes report 0.98703 stacked, with explicit caveats; dense stage-1 0.98479 |
| Reported public score | Documentation says 0.9737 for an earlier 0.9816 local model; not independently verified | No independently verified public score in inspected materials | No independently verified public score in inspected materials |
| Memory/runtime evidence | README reports 16 GB, 16 CPU threads, about 2.9 hours; not a free-Colab benchmark | No trustworthy full-scale Colab measurement; Python dictionaries and accumulated feature lists can be large | README requires at least 32 vCPU / 128 GB for its full implementation; notes describe ~88M test feature rows |
| Source-code licence | No explicit licence grant located in supplied snapshot | README explicitly claims Apache-2.0 and MIT licensing, but linked LICENSE file is absent and the dual-licence terms are not supplied | README says “Code: the project's own code”; library licence list does not grant rights to repository code |
| Useful principles | Bounded retrieval, ambiguity/context, two-stage experiment, decision-rule comparison | Entity grouping, retrieval hard negatives, structured features, incremental ensemble experiments | Multi-view and reverse retrieval, full-pool validation, density diagnostics, cross-country testing |
| Reject as initial implementation | Global label-derived maps; loading supplied pickle weights | Wholesale notebook copy, assumed fixed thresholds, reduced-pool score as competition estimate | Full in-memory feature graph and 128 GB execution assumptions |

## Findings grounded in code

### Sank19

- `src/pipeline.py:stage_translit` joins **all** train ground truth to names before learning its dictionary. Fold-level validation therefore benefits from held-out labels in preprocessing. Relearning within each outer training split is necessary for a strict estimate.
- Stage 1 and stage 2 are cross-fitted, but stage-2 training features come from stage-1 models trained on the opposite folds. An outer holdout is still needed to rule out indirect label influence through stacking and graph context.
- `src/train_test_split.py` is a separate utility that keeps only targets belonging to validation entities. That creates a reduced validation target universe. The main cross-fit pipeline should not be conflated with this utility.
- Decision-rule tuning halves do not create a completely untouched model-development holdout. The rule family is selected using the reported held-out comparisons.
- `src/model.py` loads pickle files. We do not execute/load those uploaded artefacts.
- Country equality and one-target ownership are assumptions to audit in our actual data.

### Aamod007

- README lists `code/business_entity_resolution/src/` and experiment files that are **absent** from this ZIP. The notebook is the auditable implementation; its 72-dimensional representation differs from the README's 55-feature description.
- Notebook cell 8 selects training S1 from rows 0–500k and cold S1 from row 1M onward. Disjoint rows prevent direct S1 overlap, but ordering can induce distribution bias. Each pool includes labelled positives plus 200k-row background slices per country/source, not the complete universe.
- Cell 14 injects all known positives, including ones not retrieved, into model fitting. OOF evaluation on that constructed pair table is not end-to-end blocking evaluation. The ten-negative sampling scheme also changes probability prevalence relative to inference.
- The logistic meta-learner predicts the same OOF-feature rows it was fit on; those meta-predictions are not themselves OOF. Current threshold tuning uses the fixed weighted blend instead, so distinguish unused leakage-prone code from the active path.
- A global threshold is selected on OOF predictions, but cold inference applies configured segment/global thresholds. This is not a faithfully connected tune→infer path.
- Exact-match/postal fast rules bypass the model, and maximum-match caps need independent validation. Candidate reporting must cover every route that can yield a final match.
- The metric's zero-denominator guard uses `1.25*p+r` rather than `0.25*p+r`; with nonnegative precision/recall both vanish in the same cases, so this is a readability defect rather than an observed numerical scoring error.

### Ayan

- `VALIDATION_STRATEGY.md` explicitly admits learned normalization maps use validation labels. Strict cross-country evaluation must also exclude the target country's labels from map learning.
- `PROJECT_STATE.md` explicitly warns that 0.98703 used `p1_tmargin` on a fold-restricted graph. Test competitors differ, so that number should not choose our architecture.
- Current `configs/final.json` disables `p1_competition`, reflecting a fix, but old reports are not evidence that this corrected configuration reproduces the headline.
- Current source distinguishes OOF-selected decisions from `best_rule_on_val_optimistic`; this is an improvement over selecting and reporting on the same evaluation split.
- Documentation is more candid than the headline. Its full-pool validation principle is valuable even though the execution design is unsuitable for free Colab without substantial reimplementation.

## Our initial architecture and gates

Raw official TSV → immutable-input hashes → SQLite normalized working copy → three sparse TF-IDF views over all targets in bounded shards → 27 deterministic pair/retrieval features → LightGBM → tune-only threshold → development evaluation → untouched holdout at architecture freeze → test scoring → both TSVs → organizer validator.

Optional E002 adds query-group context from inner entity OOF stage-1 scores and a second LightGBM. It does not add partial-graph target-competition features. Thresholding remains default. Exact independent-Bernoulli expected-F is implemented as an optional experiment, with its calibration/missing-candidate limitations stated. Country blocking defaults off; if enabled, training GT is checked for cross-country or missing-country pairs. Exclusivity is **not** enforced by default. Shared labelled targets cause a stop until component-level splitting is implemented.

Labels never train normalization. Training positives must be retrieved; unretrieved positives count as misses in validation. All targets remain in retrieval even when S1 is sampled. Unknown countries are processed normally, with no country one-hot encoding. Sampled S1 evaluation is still a local estimate; full test density and France can differ.

## Resource budget — engineering estimates, not measured on competition data

| Stage | Planned working RAM | Disk | Limit/caveat |
|---|---:|---|---|
| Streaming audit / normalization | roughly 0.3–1.5 GB | normalized SQLite often 1–3× raw text size | Raw ZIP and extracted TSVs also consume space |
| Sample vocabulary + target indexing | roughly 0.5–3 GB | sparse indexes depend on string lengths/vocabulary | 40k fit sample; 25k target shards; JSON vocab and NPZ matrices |
| Candidate retrieval + features | roughly 0.5–3 GB | 27 float32 features = 108 bytes/pair before IDs/Parquet | query×target block bounded; at most 60 candidates/query in default union |
| 15k-S1 LightGBM sample | target below 6 GB | text model files, feature shards | conservative preflight estimate refuses unsafe fitting |
| Optional two-stage fit | target below 8 GB | three stage-1 fold models + stage-2 | OOF arrays and contextual matrices add copies |
| Inference/export | roughly 0.5–3 GB | features + score shards + two TSVs | streamed, checkpointed per query shard |

At 60 candidates/S1, one million S1 implies up to 60 million rows: **6.48 GB for numeric feature values alone**, before IDs, scores and compression. Free Drive quota may be the binding constraint. Runtime cannot be responsibly estimated before a real full-pool benchmark. The bounded implementation rescans target index shards for each query batch, trading speed/I/O for predictable memory. It is not yet a proven deadline-ready full-scale pipeline.

## Experiment sequence

1. SMOKE_ONLY: synthetic 240 training S1 / 36 test S1, including singletons, distractors, one-to-many matches and France; both stages and resume tested. No competition-score claim.
2. E001: single-stage default, real full target pool; profile, candidate recall/distribution, memory, time, local macro F0.5 and country/error breakdowns.
3. E002: identical retrieval, inner-OOF two-stage query context. Promote only with development improvement; preserve untouched holdout.
4. E003–E005: k=10/20/30 per view, country blocking only after data evidence, reverse retrieval implementation only after throughput and recall gaps are known.
5. Later: calibration, hard-negative second round, XGBoost/CatBoost ablations, strict leave-country-out validation and graph-wide exclusivity. These are pending experiments, not claimed implemented results.
6. Neural reranking only after plateau and licence/resource checks.

## Rules reviewed and open items

User-supplied statement: every test S1 exactly once; only existing test S2/S3 IDs; no duplicates; empty singleton list; final matches subset of exact scored candidates; MIT/Apache-2.0 final model, at most 8B parameters. LightGBM is the only model used here. Uploaded guidelines also prohibit plagiarism and describe submission limits; no external business lookups are used. The statement asks for the filled organizer documentation template; bundled guidelines mention a shorter approach document. Check the current portal for any superseding announcement before final submission. No competition portal access or live leaderboard verification has occurred.

The user's dataset ZIP was not returned by connected Drive exact-name, broader-name or root discovery. A direct file URL is needed for an authenticated fetch here; Colab can also read the named ZIP directly from the user's mounted MyDrive. Real dataset profiling/training, full-scale timing and organizer validation on real test outputs remain pending.
