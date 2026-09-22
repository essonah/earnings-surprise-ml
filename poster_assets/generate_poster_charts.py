"""
Generates poster-ready (300 DPI) PNG charts for the earnings-surprise-ml project:
  1-6: dataset description charts
  7-10: model performance charts (reproduces the Model A/B/C pipeline from
        notebooks/algorithmic_model.ipynb to get real evaluation numbers)

Run from the project root:
    python poster_assets/generate_poster_charts.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# ── Palette (validated defaults — dataviz skill, references/palette.md) ──────
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

STATUS_GOOD = "#0ca30c"     # Beat
STATUS_CRITICAL = "#d03b3b"  # Miss

SERIES_BLUE = "#2a78d6"     # slot 1 — Model A: Price-only
SERIES_GREEN = "#008300"    # slot 2 — Model B: Text-only
SERIES_MAGENTA = "#e87ba4"  # slot 3 — Model C: Multimodal

SEQ_BLUE = "#256abf"        # sequential single-hue magnitude
DIV_BLUE = "#2a78d6"
DIV_RED = "#e34948"
DIV_MID = "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": BASELINE,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_PRIMARY,
    "ytick.color": INK_PRIMARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "processed", "master_features.csv")
OUT = HERE


def clean_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(grid_axis != "y")
    ax.spines["bottom"].set_color(BASELINE)
    if grid_axis == "y":
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA, parse_dates=["earnings_date"])

# ── 1. Class balance ─────────────────────────────────────────────────────
counts = df["target_label"].value_counts().sort_index()
labels = ["Miss", "Beat"]
values = [counts.get(0, 0), counts.get(1, 0)]
pct = [v / sum(values) * 100 for v in values]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, width=0.5, color=[STATUS_CRITICAL, STATUS_GOOD], zorder=3)
for bar, v, p in zip(bars, values, pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.015,
            f"{v:,} ({p:.1f}%)", ha="center", va="bottom", fontsize=12, fontweight="medium")
ax.set_ylim(0, max(values) * 1.15)
clean_axes(ax, "y")
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Class Balance: Earnings Beat vs. Miss  (n={sum(values):,})", fontsize=14, pad=14)
plt.tight_layout()
savefig(fig, "01_class_balance.png")

# ── 2. Sector distribution ──────────────────────────────────────────────
sector_counts = df["sector_group"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(sector_counts.index.str.replace("_", " ").str.title(), sector_counts.values,
               color=SEQ_BLUE, zorder=3, height=0.65)
for bar, v in zip(bars, sector_counts.values):
    ax.text(bar.get_width() + max(sector_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v}", va="center", fontsize=10, color=INK_PRIMARY)
clean_axes(ax, "x")
ax.set_xlabel("Number of earnings events", fontsize=11)
ax.set_title("Earnings Events by Sector", fontsize=14, pad=14)
plt.tight_layout()
savefig(fig, "02_sector_distribution.png")

# ── 3. Events over time ─────────────────────────────────────────────────
by_q = df.set_index("earnings_date").resample("QE").size()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(by_q.index, by_q.values, width=70, color=SEQ_BLUE, zorder=3)
clean_axes(ax, "y")
ax.set_ylabel("Earnings events", fontsize=11)
ax.set_title("Earnings Events Over Time (by Quarter)", fontsize=14, pad=14)
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
plt.tight_layout()
savefig(fig, "03_events_over_time.png")

# ── 4. Correlation heatmap (diverging) ──────────────────────────────────
feature_cols = [
    "rsi_14_at_T1", "macd_hist_at_T1", "px_vs_sma20_T1", "px_vs_sma10_T1",
    "ret_14d", "rsi_mean", "rsi_slope", "vol_mean", "vol_ratio_mean",
    "ret_std_14d", "finbert_avg_sentiment", "target_label",
]
corr = df[feature_cols].corr()
div_cmap = sns.blend_palette([DIV_RED, DIV_MID, DIV_BLUE], as_cmap=True)

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=div_cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, linecolor=SURFACE, cbar_kws={"shrink": 0.8},
            ax=ax, annot_kws={"fontsize": 8.5})
ax.set_title("Feature Correlation Matrix", fontsize=14, pad=14)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
plt.tight_layout()
savefig(fig, "04_correlation_heatmap.png")

# ── 5. Sentiment split by target label ──────────────────────────────────
plot_df = df[["finbert_avg_sentiment", "target_label"]].copy()
plot_df["Outcome"] = plot_df["target_label"].map({0: "Miss", 1: "Beat"})

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.violinplot(data=plot_df, x="Outcome", y="finbert_avg_sentiment", order=["Miss", "Beat"],
               palette=[STATUS_CRITICAL, STATUS_GOOD], ax=ax, inner="quartile", cut=0)
for patch in ax.collections:
    patch.set_alpha(0.85)
clean_axes(ax, "y")
ax.set_ylabel("Pre-earnings FinBERT sentiment", fontsize=11)
ax.set_xlabel("")
ax.set_title("News Sentiment by Earnings Outcome", fontsize=14, pad=14)
plt.tight_layout()
savefig(fig, "05_sentiment_by_outcome.png")

# ── 6. Surprise amount distribution ─────────────────────────────────────
trimmed = df["surprise_amount"].clip(-1, 1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(trimmed, bins=40, color=SEQ_BLUE, zorder=3)
ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=4)
clean_axes(ax, "y")
ax.set_xlabel("EPS surprise amount (clipped to [-1, 1])", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of EPS Surprise Amount", fontsize=14, pad=14)
plt.tight_layout()
savefig(fig, "06_surprise_amount_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# Model pipeline — reproduces notebooks/algorithmic_model.ipynb
# ══════════════════════════════════════════════════════════════════════════
SPLIT_DATE = pd.to_datetime("2025-07-01")
train_df = df[df["earnings_date"] < SPLIT_DATE].copy()
test_df = df[df["earnings_date"] >= SPLIT_DATE].copy()

price_features = [
    "rsi_14_at_T1", "macd_hist_at_T1", "px_vs_sma20_T1", "px_vs_sma10_T1", "ret_14d",
    "rsi_mean", "rsi_slope", "vol_mean", "vol_ratio_mean", "ret_std_14d",
]
text_features = ["finbert_avg_sentiment"]
multimodal_features = price_features + text_features

y_train = train_df["target_label"]
y_test = test_df["target_label"]

scaler = StandardScaler()
X_train_price = scaler.fit_transform(train_df[price_features])
X_train_text = train_df[text_features].values
X_train_multi = np.hstack((X_train_price, X_train_text))

X_test_price = scaler.transform(test_df[price_features])
X_test_text = test_df[text_features].values
X_test_multi = np.hstack((X_test_price, X_test_text))

X_train_multi_df = pd.DataFrame(X_train_multi, columns=multimodal_features)
X_test_multi_df = pd.DataFrame(X_test_multi, columns=multimodal_features)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

print("Training Model A (price-only)...")
gs_price = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                         param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_price.fit(X_train_price, y_train)

print("Training Model B (text-only)...")
gs_text = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                        param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_text.fit(X_train_text, y_train)

print("Training Model C (multimodal)...")
gs_multi = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                         param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
gs_multi.fit(X_train_multi_df, y_train)

models = {
    "Model A: Price-Only": (gs_price.best_estimator_, X_test_price, SERIES_BLUE),
    "Model B: Text-Only": (gs_text.best_estimator_, X_test_text, SERIES_GREEN),
    "Model C: Multimodal": (gs_multi.best_estimator_, X_test_multi_df, SERIES_MAGENTA),
}

results = []
for name, (model, X_te, _color) in models.items():
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": (preds == y_test).mean(),
        "Balanced Acc": balanced_accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "PR-AUC": average_precision_score(y_test, probs),
    })

test_base_rate = y_test.mean()
naive_preds = np.ones(len(y_test))
results.append({
    "Model": "Naive Baseline",
    "Accuracy": test_base_rate,
    "Balanced Acc": balanced_accuracy_score(y_test, naive_preds),
    "Precision": test_base_rate,
    "Recall": 1.0,
    "F1": f1_score(y_test, naive_preds),
    "ROC-AUC": 0.5,
    "PR-AUC": average_precision_score(y_test, naive_preds),
})
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT, "model_comparison_scorecard.csv"), index=False)

# ── 7. ROC curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.5,
        label="Naive baseline (AUC = 0.50)", zorder=2)
for name, (model, X_te, color) in models.items():
    probs = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", linewidth=2.5, color=color, zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
clean_axes(ax, "y")
ax.spines["bottom"].set_visible(True)
ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.set_title("ROC Curves: Price vs. Text vs. Multimodal", fontsize=14, pad=14)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
plt.tight_layout()
savefig(fig, "07_roc_curves.png")

# ── 8. Confusion matrix (multimodal model) ──────────────────────────────
multi_preds = gs_multi.best_estimator_.predict(X_test_multi_df)
cm = confusion_matrix(y_test, multi_preds)
seq_cmap = sns.light_palette(SEQ_BLUE, as_cmap=True)

fig, ax = plt.subplots(figsize=(6, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap=seq_cmap, cbar=False, ax=ax,
            xticklabels=["Predicted\nMiss", "Predicted\nBeat"],
            yticklabels=["Actual\nMiss", "Actual\nBeat"],
            annot_kws={"fontsize": 16, "fontweight": "medium"}, linewidths=2, linecolor=SURFACE)
ax.set_title("Confusion Matrix: Model C (Multimodal)", fontsize=14, pad=14)
ax.tick_params(labelsize=10)
plt.tight_layout()
savefig(fig, "08_confusion_matrix.png")

# ── 9. SHAP feature importance ──────────────────────────────────────────
explainer = shap.TreeExplainer(gs_multi.best_estimator_)
shap_values = explainer.shap_values(X_test_multi_df)
class1_shap = shap_values[:, :, 1] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 else (
    shap_values[1] if isinstance(shap_values, list) else shap_values
)
mean_abs = pd.Series(np.abs(class1_shap).mean(axis=0), index=multimodal_features).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(mean_abs.index, mean_abs.values, color=SEQ_BLUE, zorder=3, height=0.65)
for bar, v in zip(bars, mean_abs.values):
    ax.text(bar.get_width() + mean_abs.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9.5)
clean_axes(ax, "x")
ax.set_xlabel("Mean |SHAP value| (impact on beat prediction)", fontsize=11)
ax.set_title("SHAP Feature Importance: Model C (Multimodal)", fontsize=14, pad=14)
plt.tight_layout()
savefig(fig, "09_shap_feature_importance.png")

# ── 10. Model comparison — ROC-AUC & PR-AUC ─────────────────────────────
plot_names = [r["Model"] for r in results]
roc_vals = [r["ROC-AUC"] for r in results]
pr_vals = [r["PR-AUC"] for r in results]
x = np.arange(len(plot_names))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w / 2, roc_vals, w, label="ROC-AUC", color=SERIES_BLUE, zorder=3)
b2 = ax.bar(x + w / 2, pr_vals, w, label="PR-AUC", color=SERIES_GREEN, zorder=3)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(": ", ":\n") for n in plot_names], fontsize=9.5)
ax.set_ylim(0, 1.05)
clean_axes(ax, "y")
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison: ROC-AUC vs. PR-AUC", fontsize=14, pad=14)
ax.legend(loc="upper right", frameon=False, fontsize=10)
plt.tight_layout()
savefig(fig, "10_model_comparison_scorecard.png")

print("\nAll poster charts written to:", OUT)
