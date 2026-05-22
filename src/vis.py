import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

# Court drawing (Helper functions)

def draw_court(ax, color="#aaaaaa", lw=1.5):

    # Hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Backboard
    backboard = plt.Line2D([-30, 30], [-7.5, -7.5], linewidth=lw, color=color)

    # Paint (key)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Free throw top arc
    top_free_throw = Arc(
        (0, 142.5), 120, 120, theta1=0, theta2=180,
        linewidth=lw, color=color, fill=False
    )
    # Free throw bottom arc (dashed)
    bottom_free_throw = Arc(
        (0, 142.5), 120, 120, theta1=180, theta2=0,
        linewidth=lw, color=color, linestyle="dashed", fill=False
    )

    # Restricted area
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)

    # Three-point line
    corner_three_a = plt.Line2D([-220, -220], [-47.5, 92.5], linewidth=lw, color=color)
    corner_three_b = plt.Line2D([220, 220], [-47.5, 92.5], linewidth=lw, color=color)
    three_arc = Arc(
        (0, 0), 475, 475, theta1=22, theta2=158,
        linewidth=lw, color=color, fill=False
    )

    # Center court
    center_outer = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, fill=False)
    center_inner = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color, fill=False)

    elements = [
        hoop, backboard, outer_box, inner_box,
        top_free_throw, bottom_free_throw, restricted,
        corner_three_a, corner_three_b, three_arc,
        center_outer, center_inner,
    ]
    for el in elements:
        if isinstance(el, plt.Line2D):
            ax.add_line(el)
        else:
            ax.add_patch(el)

    return ax

# Main shot chart

def plot_shot_chart(
    df: pd.DataFrame,
    player_name: str = "",
    season_label: str = "",
    figsize: tuple = (10, 9),
    made_color: str = "#1d9e75",
    missed_color: str = "#d85a30",
    alpha: float = 0.55,
    marker_size: float = 18,
    save_path: str = None,
):

    made = df[df["SHOT_MADE_FLAG"] == 1]
    missed = df[df["SHOT_MADE_FLAG"] == 0]

    has_xfg = "xFG" in df.columns

    fig, ax = plt.subplots(figsize=figsize, facecolor="#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    if has_xfg:
        # Size encodes xFG — higher probability = larger dot
        size_scale = marker_size * 6
        ax.scatter(
            missed["LOC_X"], missed["LOC_Y"],
            c=missed_color, s=missed["xFG"] * size_scale + 4,
            alpha=alpha, linewidths=0.3, edgecolors="white", marker="x", zorder=3,
        )
        ax.scatter(
            made["LOC_X"], made["LOC_Y"],
            c=made_color, s=made["xFG"] * size_scale + 4,
            alpha=alpha, linewidths=0.3, edgecolors="white", marker="o", zorder=3,
        )
        subtitle_extra = "  •  dot size = xFG probability"
    else:
        ax.scatter(
            missed["LOC_X"], missed["LOC_Y"],
            c=missed_color, s=marker_size, alpha=alpha,
            linewidths=0.3, edgecolors="white", marker="x", zorder=3,
        )
        ax.scatter(
            made["LOC_X"], made["LOC_Y"],
            c=made_color, s=marker_size, alpha=alpha,
            linewidths=0.3, edgecolors="white", marker="o", zorder=3,
        )
        subtitle_extra = ""

    draw_court(ax, color="#999999", lw=1.2)

    ax.set_xlim(-260, 260)
    ax.set_ylim(-50, 440)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    title = player_name if player_name else "Shot Chart"
    subtitle_parts = []
    if season_label:
        subtitle_parts.append(season_label)
    fg_pct = df["SHOT_MADE_FLAG"].mean() * 100
    n = len(df)
    subtitle_parts.append(f"{n} shots  •  {fg_pct:.1f}% FG{subtitle_extra}")

    ax.set_title(
        title,
        fontsize=16, fontweight="bold", pad=12, color="#222222",
    )
    ax.text(
        0.5, 0.98, "  •  ".join(subtitle_parts),
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color="#666666",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=made_color, label="Made"),
        mpatches.Patch(color=missed_color, label="Missed"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        framealpha=0.7, fontsize=10, edgecolor="#cccccc",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Shot chart saved to {save_path}")

    return fig, ax

# xFG heatmap variant

def plot_xfg_heatmap(
    df: pd.DataFrame,
    player_name: str = "",
    season_label: str = "",
    bins: int = 25,
    figsize: tuple = (10, 9),
    cmap: str = "RdYlGn",
    save_path: str = None,
):

    if "xFG" not in df.columns:
        raise ValueError("df must contain an 'xFG' column. Run predict() first.")

    fig, ax = plt.subplots(figsize=figsize, facecolor="#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    # Hexbin — C encodes mean xFG per hex cell
    hb = ax.hexbin(
        df["LOC_X"], df["LOC_Y"],
        C=df["xFG"],
        reduce_C_function=np.mean,
        gridsize=bins,
        cmap=cmap,
        alpha=0.75,
        mincnt=3,           # only show cells with ≥3 shots
        vmin=0.2, vmax=0.7,
        zorder=2,
    )

    draw_court(ax, color="#555555", lw=1.5)

    ax.set_xlim(-260, 260)
    ax.set_ylim(-50, 440)
    ax.set_aspect("equal")
    ax.axis("off")

    title = f"{player_name} — xFG heatmap" if player_name else "xFG heatmap"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12, color="#222222")
    if season_label:
        ax.text(
            0.5, 0.98, season_label,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#666666",
        )

    cb = fig.colorbar(hb, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("avg xFG", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"xFG heatmap saved to {save_path}")

    return fig, ax