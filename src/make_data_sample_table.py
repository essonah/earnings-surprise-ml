"""Poster figure: a readable sample of rows from master_features.csv."""
import matplotlib.pyplot as plt
import pandas as pd

from utils import MASTER_CSV

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# A hand-picked, mixed set of beats/misses across recognizable tickers —
# chosen for readability and to include one counterintuitive case
# (positive sentiment that still missed), not a random head(8).
PICKS = [
    ("GOOGL", "2022-04-26"),
    ("QCOM", "2023-05-03"),
    ("MAR", "2024-05-01"),
    ("AMD", "2024-07-30"),
    ("QCOM", "2024-11-06"),
    ("PEP", "2025-04-24"),
    ("GOOGL", "2025-04-24"),
    ("MAR", "2026-02-10"),
]

COLUMNS = ["Ticker", "Date", "RSI (14d)", "14d return", "Sentiment", "Outcome"]
# (anchor_x, align) — anchors leave a real gap between columns so header text
# (wider than the data cells) can't run into its neighbor.
COL_ANCHORS = [
    (0.03, "left"),
    (0.17, "left"),
    (0.50, "right"),
    (0.67, "right"),
    (0.85, "right"),
    (0.97, "right"),
]
COL_X = [a for a, _ in COL_ANCHORS]
COL_ALIGN = [a for _, a in COL_ANCHORS]


def load_rows():
    df = pd.read_csv(MASTER_CSV, parse_dates=["earnings_date"])
    rows = []
    for ticker, date in PICKS:
        r = df[(df.ticker == ticker) & (df.earnings_date == pd.Timestamp(date))].iloc[0]
        rows.append(r)
    return rows


def main():
    rows = load_rows()
    n = len(rows)

    # Inch-based vertical rhythm, laid out with a top-down cursor — figure
    # height falls out of the content instead of being guessed independently,
    # so there's no leftover blank band anywhere.
    TOP_MARGIN = 0.18
    TITLE_GAP = 0.42          # title baseline -> subtitle baseline
    SUBTITLE_GAP = 0.45       # subtitle baseline -> header row
    ROW_H_IN = 0.46           # header row and each data row
    FOOTER_GAP = 0.40         # last row -> footer baseline
    BOTTOM_MARGIN = 0.18
    fig_w = 8.5

    fig_h = TOP_MARGIN + TITLE_GAP + SUBTITLE_GAP + (n + 1) * ROW_H_IN + FOOTER_GAP + BOTTOM_MARGIN

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def y_at(inches_from_top):
        return 1 - inches_from_top / fig_h

    row_h = ROW_H_IN / fig_h
    cursor = TOP_MARGIN

    fig.text(0.03, y_at(cursor), "A sample of the modeling table", fontsize=14,
              color=INK_PRIMARY, fontweight="bold", va="top")
    cursor += TITLE_GAP
    fig.text(0.03, y_at(cursor), "master_features.csv — pre-earnings technical + sentiment features, and the beat/miss label",
              fontsize=9.5, color=INK_MUTED, va="top")
    cursor += SUBTITLE_GAP

    top = y_at(cursor)  # header row baseline

    def row_y(i):
        return top - i * row_h

    # header row
    y = row_y(0)
    for x, label, align in zip(COL_X, COLUMNS, COL_ALIGN):
        ax.text(x, y, label, fontsize=9.5, color=INK_SECONDARY, fontweight="bold",
                 ha=align, va="center")
    ax.plot([0.03, 0.97], [y - row_h * 0.42, y - row_h * 0.42], color=BASELINE, linewidth=1.2)

    # data rows
    for i, r in enumerate(rows, start=1):
        y = row_y(i)
        beat = int(r["target_label"]) == 1
        vals = [
            r["ticker"],
            r["earnings_date"].strftime("%Y-%m-%d"),
            f"{r['rsi_14_at_T1']:.1f}",
            f"{r['ret_14d']:+.1%}",
            f"{r['finbert_avg_sentiment']:+.2f}",
        ]
        for x, v, align in zip(COL_X[:5], vals, COL_ALIGN[:5]):
            weight = "bold" if align == "left" and v == r["ticker"] else "normal"
            ax.text(x, y, v, fontsize=10, color=INK_PRIMARY, ha=align, va="center",
                     fontweight=weight, fontfamily="monospace" if align == "right" else None)

        color = GOOD if beat else CRITICAL
        label = "Beat" if beat else "Miss"
        ax.text(COL_X[5], y, label, fontsize=10, color=color, ha="right", va="center",
                 fontweight="bold")

        if i < n:
            ax.plot([0.03, 0.97], [y - row_h * 0.46, y - row_h * 0.46],
                     color=GRIDLINE, linewidth=1)

    fig.text(0.03, y_at(cursor + n * ROW_H_IN + FOOTER_GAP),
              f"{n} of 1,161 earnings events shown · full table has 12 features per event",
              fontsize=8.5, color=INK_MUTED, va="top")

    fig.savefig("figures/data_sample_table.png", dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig("figures/data_sample_table.svg", facecolor=SURFACE, bbox_inches="tight")
    print("Saved figures/data_sample_table.png and figures/data_sample_table.svg")


if __name__ == "__main__":
    main()
