import pandas as pd

from utils import (
    MASTER_CSV,
    METADATA_COLS,
    PROCESSED_DIR,
    SENTIMENT_CSV,
    TECHNICAL_CSV,
    normalize_earnings_date,
)

NUMERICAL_FILL_DEFAULTS = {
    "ret_14d": 0.0,
    "ret_std_14d": 0.015,
    "rsi_14_at_T1": 50.0,
    "macd_hist_at_T1": 0.0,
    "px_vs_sma20_T1": 0.0,
    "px_vs_sma10_T1": 0.0,
    "vol_ratio_mean": 1.0,
}


def merge_multimodal_features() -> pd.DataFrame | None:
    """
    Merge technical features (02_feature_engineering.ipynb) with FinBERT
    sentiment features (Colab), impute missing values, and save the
    final modeling table.
    """
    if not TECHNICAL_CSV.exists():
        print(f"Error: missing technical features at {TECHNICAL_CSV}")
        return None

    if not SENTIMENT_CSV.exists():
        print(f"Error: missing sentiment features at {SENTIMENT_CSV}")
        return None

    df_num = pd.read_csv(TECHNICAL_CSV)
    df_sent = pd.read_csv(SENTIMENT_CSV)

    df_num["earnings_date"] = normalize_earnings_date(df_num["earnings_date"])
    df_sent["earnings_date"] = normalize_earnings_date(df_sent["earnings_date"])

    print(f"Loaded {len(df_num)} technical rows and {len(df_sent)} sentiment rows.")

    df_master = df_num.merge(
        df_sent[["ticker", "earnings_date", "finbert_avg_sentiment"]],
        on=["ticker", "earnings_date"],
        how="left",
    )

    missing_sentiment = df_master["finbert_avg_sentiment"].isna().sum()
    if missing_sentiment:
        print(
            f"Imputing {missing_sentiment} rows with no news sentiment as 0.0 (neutral)."
        )
        df_master["finbert_avg_sentiment"] = df_master["finbert_avg_sentiment"].fillna(
            0.0
        )

    fill_defaults = NUMERICAL_FILL_DEFAULTS.copy()
    if "ret_std_14d" in df_master.columns:
        fill_defaults["ret_std_14d"] = df_master["ret_std_14d"].median()
    if "vol_mean" in df_master.columns:
        fill_defaults["vol_mean"] = df_master["vol_mean"].median()

    for col, default_val in fill_defaults.items():
        if col in df_master.columns:
            df_master[col] = df_master[col].fillna(default_val)

    feature_cols = [col for col in df_master.columns if col not in METADATA_COLS]
    column_order = (
        [col for col in METADATA_COLS if col != "target_label" and col in df_master.columns]
        + feature_cols
        + ["target_label"]
    )
    df_master = df_master[column_order]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(MASTER_CSV, index=False)

    print(f"Saved master dataset to {MASTER_CSV}")
    print(f"Shape: {df_master.shape[0]} rows x {df_master.shape[1]} columns")

    distribution = df_master["target_label"].value_counts()
    print(f"Target distribution — beats (1): {distribution.get(1, 0)}, misses (0): {distribution.get(0, 0)}")

    return df_master


if __name__ == "__main__":
    merge_multimodal_features()
