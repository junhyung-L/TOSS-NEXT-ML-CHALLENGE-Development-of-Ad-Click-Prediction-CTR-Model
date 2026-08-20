# Run Manifest

## Supported entry point

The maintained experiment path is `src/train.py`. Run it from the repository
root with either form:

```powershell
python src\train.py --train_path .\train.parquet --test_path .\test.parquet
# or
python -m src.train --train_path .\train.parquet --test_path .\test.parquet
```

The first form requires explicit input paths because the legacy defaults were
written for execution from `src/`. The defaults themselves have not changed.

## Result-preservation boundary

This refactor centralizes command-line defaults and import paths only. It keeps
the prior default seed (42), split fraction (0.15), architecture defaults,
optimizer, scheduler, loss, early-stopping rule, and JSON output schema.

Exact numerical reproduction still requires the original Parquet inputs and
matching library/runtime versions. Neither is stored in this repository, and no
training run was performed as part of this refactor.
