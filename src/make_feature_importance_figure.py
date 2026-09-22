"""Poster figure: random forest feature importances on the training split."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import MASTER_CSV, METADATA_COLS

BLUE = "#2a78d6"  # categorical slot 1 — single series, no identity contrast needed
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RANDOM_STATE = 42
TEST_FRACTION = 0.2

LABELS = {
    "macd_hist_at_T1": "MACD histogram",
    "rsi_mean": "RSI (14d mean)",
    "vol_ratio_mean": "Volume ratio (mean)",
    "vol_mean": "Volume (mean)",
    "ret_std_14d": "Return volatility (14d)",
    "rsi_slope": "RSI slope",
    "finbert_avg_sentiment": "News sentiment (FinBERT)",
    "ret_14d": "14-day price return",
    "rsi_14_at_T1": "RSI (at T-1)",
    "px_vs_sma10_T1": "Price vs. 10d SMA",
    "px_vs_sma20_T1": "Price vs. 20d SMA",
}


def compute_importances():
    df = pd.read_csv(MASTER_CSV, parse_dates=["earnings_date"]).sort_values("earnings_date")
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X, y = df[feature_cols], df["target_label"]

    cutoff = df["earnings_date"].quantile(1 - TEST_FRACTION)
    train_mask = df["earnings_date"] <= cutoff
    X_train, y_train = X[train_mask], y[train_mask]

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    return imp


def main():
    imp = compute_importances()
    n = len(imp)
    names = [LABELS.get(c, c) for c in imp.index]
    values = imp.values

    TOP_MARGIN, TITLE_GAP, SUBTITLE_GAP = 0.18, 0.42, 0.40
    ROW_H_IN = 0.42
    BOTTOM_MARGIN = 0.55
    fig_w = 8.5
    fig_h = TOP_MARGIN + TITLE_GAP + SUBTITLE_GAP + n * ROW_H_IN + BOTTOM_MARGIN

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    plot_top = 1 - (TOP_MARGIN + TITLE_GAP + SUBTITLE_GAP) / fig_h
    plot_bottom = BOTTOM_MARGIN / fig_h
    fig.subplots_adjust(top=plot_top, bottom=plot_bottom, left=0.30, right=0.90)

    y_pos = range(n)
    bar_h = 0.55
    x_max = values.max() * 1.28

    for y, v in zip(y_pos, values):
        ax.add_patch(
            FancyBboxPatch(
                (0, y - bar_h / 2), v, bar_h,
                boxstyle="round,pad=0,rounding_size=0.012",
                linewidth=0, facecolor=BLUE, mutation_aspect=1 / (x_max / (n * 1.4)),
            )
        )
        ax.text(v + x_max * 0.015, y, f"{v:.1%}", va="center", ha="left",
                 fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10, color=INK_PRIMARY)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, n - 0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of random forest importance", fontsize=9.5, color=INK_SECONDARY)

    def y_at(inches_from_top):
        return 1 - inches_from_top / fig_h

    cursor = TOP_MARGIN
    fig.text(0.03, y_at(cursor), "What drives the beat/miss prediction",
              fontsize=14, color=INK_PRIMARY, fontweight="bold", va="top")
    cursor += TITLE_GAP
    fig.text(0.03, y_at(cursor),
              "Random forest feature importances, trained on the 929-event training split · "
              "no single feature dominates — signal is spread across technical + sentiment features",
              fontsize=9, color=INK_MUTED, va="top", wrap=True)

    fig.savefig("figures/feature_importance.png", dpi=300, facecolor=SURFACE)
    fig.savefig("figures/feature_importance.svg", facecolor=SURFACE)
    print("Saved figures/feature_importance.png and figures/feature_importance.svg")


if __name__ == "__main__":
    main()
