"""Optional LightGBM baseline helpers.

The maintained deep-learning experiment entry point is ``src/train.py``.
This module remains a small, independent baseline utility and is not invoked by
the sequence-model training pipeline.
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the legacy recency-decay feature when its source column exists."""
    logging.info("Starting feature engineering...")
    
    # Example: Time-decay features
    if 'days_since_last_click' in df.columns:
        df['recency_decay'] = np.exp(-df['days_since_last_click'] / 30)
    
    logging.info("Feature engineering complete.")
    return df

def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """Fit the legacy class-weighted LightGBM baseline."""
    logging.info("Initializing LightGBM model...")
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        class_weight='balanced',
        importance_type='gain'
    )
    model.fit(X_train, y_train)
    logging.info("Training complete.")
    return model

if __name__ == "__main__":
    logging.info("LightGBM baseline helpers loaded. Use src/train.py for the CTR pipeline.")
