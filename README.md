# Earnings Surprise ML

Predict whether companies beat quarterly EPS consensus using pre-earnings market and news signals.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**TA-Lib:** The Python package needs the native TA-Lib library installed first.

- macOS: `brew install ta-lib`
- Linux: install `ta-lib` from your package manager, then `pip install TA-Lib`

Create a `.env` file in the project root:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
```

## Pipeline

Run the notebooks in order from the `notebooks/` directory:

| Notebook | Purpose |
|----------|---------|
| `01_data_collection.ipynb` | Earnings labels (Alpha Vantage) + daily prices (Yahoo Finance) |
| `news_collection.ipynb` | Pre-earnings news headlines (Finnhub) |
| `02_feature_engineering.ipynb` | Technical indicators for the 14 trading days before each event |
| `src/features.py` | Merge technical + FinBERT sentiment into `master_features.csv` |

After Colab produces `data/processed/features_news_sentiment.csv`, run:

```bash
python src/features.py
```

## Outputs

Generated data is gitignored and written under `data/`:

- `data/raw/earnings_base.csv` — labels and metadata
- `data/raw/ticker_prices/` — per-ticker OHLCV history
- `data/raw_news_for_colab.csv` — news text for NLP features
- `data/processed/features_technical.csv` — modeling table with technical features
- `data/processed/features_news_sentiment.csv` — FinBERT sentiment from Colab
- `data/processed/master_features.csv` — merged training dataset

## Project layout

```
notebooks/     Jupyter pipeline
src/           Shared Python modules (WIP)
data/          Raw and processed datasets (local only)
```
