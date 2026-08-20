import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def parse_seq_string(s: str) -> list[int]:
    """Parse supported sequence-string formats into integer item identifiers."""
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
        tokens = [token.strip().strip("'\"") for token in s.split(",")]
    else:
        s = re.sub(r"[^\d]+", ",", s)
        tokens = [token for token in s.split(",") if token]

    parsed = []
    for token in tokens:
        try:
            parsed.append(int(token))
        except Exception:
            try:
                parsed.append(int(float(token)))
            except Exception:
                pass
    return parsed


def seq_to_ids_hash(
    values: Sequence[int],
    max_len: int = 50,
    hash_buckets: int = 262144,
    pad_id: int = 0,
    seq_base: int = 2,
) -> np.ndarray:
    """Hash, truncate, and left-pad sequence IDs using the legacy scheme."""
    ids = [seq_base + (int(value) % hash_buckets) for value in values][-max_len:]
    if len(ids) < max_len:
        ids = [pad_id] * (max_len - len(ids)) + ids
    return np.array(ids, dtype=np.int32)


class CTRDataset(Dataset):
    """Materialize tabular features and lazily encode one behaviour sequence."""

    def __init__(
        self,
        df: pd.DataFrame,
        cont_cols: Sequence[str],
        cat_cols: Sequence[str],
        cat_maps: Mapping[str, Mapping[Any, int]],
        seq_col: str,
        max_seq_len: int,
        hash_buckets: int,
        cat_cards: Mapping[str, int],
        label_col: str | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cont_cols, self.cat_cols = cont_cols, cat_cols
        self.cat_maps, self.seq_col = cat_maps, seq_col
        self.max_seq_len = max_seq_len
        self.hash_buckets = hash_buckets
        self.cat_cards = cat_cards
        self.has_label = label_col is not None
        self.label_col = label_col
        
        self.Xc = self.df[self.cont_cols].astype(np.float32).fillna(0.0).values if self.cont_cols else None
        self.Xcats = {c: self.df[c].map(self.cat_maps[c]).fillna(self.cat_cards[c]).astype(np.int64).values for c in self.cat_cols}
        if self.has_label:
            self.y = self.df[self.label_col].astype(np.float32).values
            
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        xc = torch.from_numpy(self.Xc[idx]) if self.Xc is not None else torch.empty(0)
        cats = {c: torch.tensor(self.Xcats[c][idx], dtype=torch.long) for c in self.cat_cols}
        lst = parse_seq_string(self.df.at[idx, self.seq_col])
        seq = torch.from_numpy(seq_to_ids_hash(lst, self.max_seq_len, self.hash_buckets)).long()
        
        if self.has_label:
            return xc, cats, seq, torch.tensor(self.y[idx], dtype=torch.float32)
        return xc, cats, seq
