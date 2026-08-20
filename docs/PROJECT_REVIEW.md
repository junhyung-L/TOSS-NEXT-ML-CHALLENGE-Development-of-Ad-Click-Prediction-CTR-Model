# Project Review

This is an evidence-based assessment of the repository, not a measured
production-readiness score.

| Area | Assessment | Evidence and caveat |
|---|---:|---|
| Problem framing | 8/10 | Clear CTR objective and binary-label training interface in `src/train.py`. |
| Data handling | 6/10 | Sequence parsing, padding, and hashing are implemented; dataset schema, scale, and provenance are absent. |
| Feature engineering | 7/10 | Train-only category mappings and a bounded sequence representation are explicit in `src/data.py` and `src/train.py`. |
| Modelling | 8/10 | Cross/deep tabular layers and three sequence backbones are implemented in `src/models.py`. |
| Experimental rigor | 6/10 | Stratification, seed 42, early stopping, and retained run metadata exist; no repeated-seed analysis or external test evidence is stored. |
| Evaluation | 7/10 | ROC-AUC and PR-AUC are calculated in `src/train.py`; threshold/business metrics are not recorded. |
| Reproducibility | 5/10 | Shared defaults, an import-derived requirements file, and a run manifest now exist; raw data, pinned versions, and checkpoints remain absent. |
| Overall | 6.7/10 | A technically substantive CTR prototype with verifiable implementation and results, but incomplete experiment provenance. |

## Strengths

- The training path handles class imbalance explicitly through `pos_weight`.
- Sequence hashing and a maximum length provide concrete memory bounds.
- Run metadata captures feature lists, major hyperparameters, and validation
  metrics for several configurations.
- The repository retains both ROC-AUC and PR-AUC rather than relying on a
  single ranking metric.

## Limitations

- `main.py` and `src/train.py` are separate modelling paths, which can confuse
  the executable entry point.
- The metadata omits dataset row counts, class prevalence, package versions,
  runtime/hardware, and source-data version.
- Notebook-only experimental variants cannot be treated as equivalent to the
  maintained CLI without consolidation.
- No held-out test, calibration, ranking-at-K, or business-cost evaluation is
  tracked.

## Highest-value next steps

1. Add a locked dependency file and a schema/fixture that exercises the CLI.
2. Write a single experiment manifest containing data version, split counts,
   prevalence, seed, source commit, device, and output paths.
3. Consolidate the preferred model path or clearly label notebooks as archival.
4. Add ranking-at-K and calibration metrics suited to CTR decisions, then keep
   external evaluation evidence separate from validation tuning.
