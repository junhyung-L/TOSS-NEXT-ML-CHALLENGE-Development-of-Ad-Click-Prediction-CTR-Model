# CTR Prediction with DCN-V2 and Behaviour Sequences

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

CTR prediction needs both tabular context—user, inventory, and time features—and the sequence of what a user has recently interacted with. This project combines a DCN-V2-style cross/deep tower with interchangeable DIN, DIEN, and BST sequence encoders to test both forms of signal together.

In the retained validation metadata, DCN-V2 + DIEN reaches ROC-AUC 0.7413 and PR-AUC 0.0792. A DUSIN-labelled extension reaches ROC-AUC 0.7417 and PR-AUC 0.0780. Because the best ROC-AUC and PR-AUC belong to different configurations, the project does not claim one universal winner.

## The modelling choices

The project addresses two separate problems: learning nonlinear interactions among tabular features and representing variable-length behaviour histories within a fixed memory budget. CrossNetMix supplies explicit feature crosses through low-rank, gated experts; a parallel MLP learns broader nonlinear patterns. Their outputs are combined with the sequence-interest vector only in the final prediction head.

Histories are parsed from several string formats, truncated to the most recent 50 items, and hashed into 262,144 buckets. The hash bounds the embedding vocabulary; left padding makes batch length fixed. The sequence encoders are deliberately comparable within the same tabular backbone:

- **DIN** attends from the current candidate to historical item embeddings.
- **DIEN** first passes the history through a GRU, then attends over evolving interest states.
- **BST** adds positional encoding and a Transformer encoder before attention pooling.

## Data boundary and training protocol

The source Parquet files and data card are not versioned, so the project does not invent a row count, provenance, or click rate. The maintained code establishes the observable interface: `clicked` is the default label, `seq` the behaviour sequence, and `ID` the submission identifier. Continuous fields are converted to `float32` and zero-filled. Category mappings are fit on the training split only, and unseen validation/test values use an OOV index.

The supported CLI uses seed 42 and a stratified 85/15 train/validation split. It applies `BCEWithLogitsLoss` with `pos_weight = negative / positive`, AdamW, cosine annealing, and ROC-AUC-based early stopping. The principal saved runs record full-data training, batch size 512, learning rate 0.001, dropout 0.2, and three epochs.

## Retained validation comparison

| Saved run | ROC-AUC | PR-AUC | Reading of the result |
|---|---:|---:|---|
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention-based sequence baseline |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | Transformer-based sequence encoder |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | highest retained PR-AUC |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | highest retained ROC-AUC |

These are saved validation metrics, not external-test or leaderboard results. There are no repeated-seed confidence intervals, calibration results, or ranking-at-K business metrics. DIEN is therefore a useful reference configuration for PR-AUC, while the model choice should still follow the operating objective.

## What is implemented—and what is archival

`src/train.py` is the maintained path: it loads files, splits data, builds training-only category maps, trains a selected sequence backbone, and writes predictions plus run metadata. It makes DIN/DIEN/BST comparisons reproducible from the same CLI contract.

`main.py` is a separate LightGBM feature-engineering stub. Some DUSIN experiments are notebook-only extensions and are not equivalent to the maintained CLI path. The documentation keeps this distinction explicit rather than presenting all notebook experiments as one deployed system.

## Next iteration

The immediate improvement is a complete experiment manifest: data version, row counts, class prevalence, split counts, package versions, hardware, and artefact paths. Time-ordered evaluation, repeated seeds, calibration, and Precision/Recall@K would turn the current modelling prototype into stronger evidence for an advertising decision workflow.

## Evidence

- [Training pipeline](src/train.py)
- [Data and sequence handling](src/data.py)
- [DCN-V2, DIN, DIEN, and BST implementation](src/models.py)
- [DIEN metadata](results/dcnv2_dien_meta.json) and [DUSIN metadata](results/dcnv2_dien_dusin_full_meta.json)
- [Run manifest](research/RUN_MANIFEST.md)
