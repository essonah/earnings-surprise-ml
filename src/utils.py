from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TECHNICAL_CSV = PROCESSED_DIR / "features_technical.csv"
SENTIMENT_CSV = PROCESSED_DIR / "features_news_sentiment.csv"
MASTER_CSV = PROCESSED_DIR / "master_features.csv"

# Train/test cutoff used everywhere a script needs a chronological split —
# matches notebooks/algorithmic_model.ipynb, which is what the reported
# model metrics are evaluated against.
SPLIT_DATE = pd.Timestamp("2025-07-01")

METADATA_COLS = [
    "ticker",
    "earnings_date",
    "actual_eps",
    "consensus_eps",
    "fiscal_date_ending",
    "sector_group",
    "surprise_amount",
    "target_label",
]


def normalize_earnings_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date
