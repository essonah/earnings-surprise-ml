# Earnings Surprise ML

Predict whether companies beat quarterly EPS consensus using pre-earnings market and news signals.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy requests python-dotenv yfinance TA-Lib scikit-learn shap matplotlib seaborn
```

**TA-Lib:** The Python package needs the native TA-Lib library installed first.

- macOS: `brew install ta-lib`
- Linux: install `ta-lib` from your package manager, then `pip install TA-Lib`

Create a `.env` file in the project root:

```env
AV_API_KEY=your_key_here
```

Everything — earnings labels and news — is pulled from **Alpha Vantage** (premium tier: 75 requests/min, no daily cap). Price history comes from `yfinance`, which needs no key. Finnhub is not used: an earlier version of the news collection notebook used Finnhub's `company_news` endpoint, but its free tier only archives ~1 year of news, so earnings windows before mid-2025 silently returned zero articles and got imputed to neutral sentiment — fabricating sentiment for most of the dataset. It was replaced with Alpha Vantage's `NEWS_SENTIMENT` endpoint, which archives back to 2021 for these tickers.

## Pipeline

Run the notebooks in order from the `notebooks/` directory:

| Step | Purpose |
|------|---------|
| `01_data_collection.ipynb` | Earnings labels (Alpha Vantage `EARNINGS`) + daily prices (`yfinance`) |
| `news_collection.ipynb` | Pre-earnings news text per earnings window (Alpha Vantage `NEWS_SENTIMENT`), caches raw articles to disk, writes `raw_news_for_colab.csv` |
| **Google Colab: FinBERT sentiment** | Upload `raw_news_for_colab.csv` to Colab and run the FinBERT pipeline (below) to score sentiment with GPU acceleration; download the result back into `data/processed/features_news_sentiment.csv` |
| `02_feature_engineering.ipynb` | Technical indicators (TA-Lib) for the 14 trading days before each event |
| `src/features.py` | Merge technical + FinBERT sentiment into `master_features.csv` |
| `notebooks/algorithmic_model.ipynb` | Train and compare classifiers (logistic regression, random forest) on `master_features.csv`; ROC curves, confusion matrix, SHAP feature importance |

After Colab produces `data/processed/features_news_sentiment.csv`, run:

```bash
python src/features.py
```

### Sentiment scoring (Google Colab)

News sentiment isn't scored locally — it runs as a separate notebook on Google Colab to use GPU acceleration for FinBERT inference. The pipeline:

1. Loads `raw_news_for_colab.csv` (uploaded manually to the Colab workspace).
2. Runs [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) via a Hugging Face `text-classification` pipeline, batched (batch size 32) on a T4 GPU.
3. Maps each headline's label/score to a signed value: positive → `+score`, negative → `-score`, neutral → `0.0`.
4. Averages sentiment per `(ticker, earnings_date)` window into a single `finbert_avg_sentiment` feature.
5. Saves `features_news_sentiment.csv`, downloaded and placed into `data/processed/` for `src/features.py` to merge in.

## Outputs

Generated data is gitignored and written under `data/`:

- `data/raw/earnings_base.csv` — labels and metadata
- `data/raw/ticker_prices/` — per-ticker OHLCV history
- `data/raw/av_news_cache/` — cached raw Alpha Vantage news articles per ticker (resumable backfill)
- `data/raw/news_coverage_report.csv` — article coverage per earnings window
- `data/raw_news_for_colab.csv` — news text for FinBERT scoring in Colab
- `data/processed/features_technical.csv` — modeling table with technical features
- `data/processed/features_news_sentiment.csv` — FinBERT sentiment from Colab
- `data/processed/master_features.csv` — merged training dataset

## Project layout

```
notebooks/     Jupyter pipeline (data collection, feature engineering, modeling)
src/           Shared Python modules + figure generation scripts
figures/       Generated charts for the README/write-up
poster_assets/ Charts and scripts for the project poster
data/          Raw and processed datasets (local only)
```
