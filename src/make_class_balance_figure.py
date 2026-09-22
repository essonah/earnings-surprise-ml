"""Poster figure: class balance (beat/miss) across All / Train / Test splits."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from utils import MASTER_CSV

TEST_FRACTION = 0.2

# Reference palette (dataviz skill) — light mode, status pair
GOOD = "#0ca30c"    # beat
CRITICAL = "#d03b3b"  # miss
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

GAP = 6  # counts, i.e. the "surface gap" between stacked segments


def load_splits():
    df = pd.read_csv(MASTER_CSV, parse_dates=["earnings_date"]).sort_values("earnings_date")
    cutoff = df["earnings_date"].quantile(1 - TEST_FRACTION)
    train_mask = df["earnings_date"] <= cutoff
    train, test = df[train_mask], df[~train_mask]

    rows = []
    for name, d in [("All", df), ("Train", train), ("Test", test)]:
        vc = d["target_label"].value_counts()
        beat, miss = int(vc.get(1, 0)), int(vc.get(0, 0))
        rows.append({"split": name, "beat": beat, "miss": miss, "total": beat + miss})
    return rows


SMALL_SEGMENT_FRACTION = 0.10  # below this share of the y-range, label goes outside


def main():
    rows = load_splits()
    y_max = max(r["total"] for r in rows) * 1.24
    small_threshold = y_max * SMALL_SEGMENT_FRACTION

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_width = 0.5
    x = range(len(rows))

    for i, row in enumerate(rows):
        miss_h = max(row["miss"] - GAP / 2, 0)
        beat_bottom = row["miss"] + GAP / 2
        beat_h = row["beat"] - GAP / 2

        ax.add_patch(
            FancyBboxPatch(
                (i - bar_width / 2, 0), bar_width, miss_h,
                boxstyle="round,pad=0,rounding_size=4", linewidth=0, facecolor=CRITICAL,
                mutation_aspect=0.02,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (i - bar_width / 2, beat_bottom), bar_width, beat_h,
                boxstyle="round,pad=0,rounding_size=4", linewidth=0, facecolor=GOOD,
                mutation_aspect=0.02,
            )
        )

        # beat segment is always the larger share here — label inline
        ax.text(
            i, beat_bottom + beat_h / 2, f"{row['beat']}\n({row['beat']/row['total']:.0%})",
            ha="center", va="center", color="white", fontsize=10, fontweight="medium",
            linespacing=1.3,
        )

        # miss segment: inline if tall enough, otherwise a leader label to the right
        if row["miss"] >= small_threshold:
            ax.text(
                i, row["miss"] / 2, f"{row['miss']}\n({row['miss']/row['total']:.0%})",
                ha="center", va="center", color="white", fontsize=10, fontweight="medium",
                linespacing=1.3,
            )
        else:
            label_y = small_threshold * 0.9
            ax.plot(
                [i, i + bar_width / 2 + 0.18], [row["miss"] / 2, label_y],
                color=INK_MUTED, linewidth=1, zorder=5,
            )
            ax.text(
                i + bar_width / 2 + 0.22, label_y,
                f"{row['miss']} miss ({row['miss']/row['total']:.0%})",
                ha="left", va="center", color=INK_SECONDARY, fontsize=9.5,
            )

        ax.text(
            i, row["total"] + y_max * 0.025, f"N = {row['total']}",
            ha="center", va="bottom", color=INK_SECONDARY, fontsize=10,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["split"] for r in rows], fontsize=11.5, color=INK_PRIMARY)
    ax.set_xlim(-0.6, len(rows) - 1 + 0.9)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Earnings events", fontsize=10, color=INK_SECONDARY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=GOOD, markersize=10, label="Beat"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=CRITICAL, markersize=10, label="Miss"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
        frameon=False, fontsize=10, labelcolor=INK_PRIMARY, handletextpad=0.6,
    )

    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.10, right=0.96)

    fig.suptitle(
        "Class balance across dataset splits",
        fontsize=15, color=INK_PRIMARY, fontweight="bold", x=0.10, y=0.965, ha="left",
    )
    fig.text(
        0.10, 0.905,
        "1,161 earnings events, 60 tickers, Aug 2021–Jul 2026 · time-based train/test split",
        fontsize=10, color=INK_MUTED, ha="left",
    )

    fig.savefig("figures/class_balance.png", dpi=300, facecolor=SURFACE)
    fig.savefig("figures/class_balance.svg", facecolor=SURFACE)
    print("Saved figures/class_balance.png and figures/class_balance.svg")


if __name__ == "__main__":
    main()
