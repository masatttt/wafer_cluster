"""
Wafer map visualization for DBSCAN result.

Reads the output CSV of wafer_dbscan.py and draws grid-style wafer maps
colour-coded by cluster membership.

Usage:
    python visualize_wafermap.py --input result.csv
    python visualize_wafermap.py --input result.csv --wafer_id WAFER_ID --wafer W01
    python visualize_wafermap.py --input result.csv --wafer_id WAFER_ID --show
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


def build_wafer_map(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    title: str,
    chip_size: float = 0.9,
) -> None:
    """Draw a single wafer map on the given Axes."""

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    # --- assign a numeric category to each chip ---
    #   0 = good,  1 = noise (cluster_id == -1),  2+ = cluster 0, 1, 2, ...
    cluster_ids = (
        df.loc[df["is_defect"] & df["cluster_id"].notna() & (df["cluster_id"] >= 0), "cluster_id"]
        .unique()
    )
    cluster_ids = sorted(int(c) for c in cluster_ids)

    # colour palette
    good_color = "#D9D9D9"
    noise_color = "#222222"
    cluster_cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(cluster_ids), 1))
    cluster_colors = {cid: tuple(cluster_cmap(i / max(len(cluster_ids), 1))) for i, cid in enumerate(cluster_ids)}

    legend_handles = []

    for _, row in df.iterrows():
        x, y = row[x_col], row[y_col]

        if not row["is_defect"]:
            color = good_color
        elif pd.isna(row["cluster_id"]) or int(row["cluster_id"]) == -1:
            color = noise_color
        else:
            color = cluster_colors[int(row["cluster_id"])]

        rect = plt.Rectangle(
            (x - chip_size / 2, y - chip_size / 2),
            chip_size,
            chip_size,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.add_patch(rect)

    # --- legend ---
    legend_handles.append(mpatches.Patch(color=good_color, label="Good"))
    legend_handles.append(mpatches.Patch(color=noise_color, label="Noise (isolated)"))
    for cid in cluster_ids:
        legend_handles.append(
            mpatches.Patch(color=cluster_colors[cid], label=f"Cluster {cid}")
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_max + 1, y_min - 1)  # invert Y
    ax.set_aspect("equal")

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(False)


def add_cluster_labels(
    df: pd.DataFrame, x_col: str, y_col: str, ax: plt.Axes
) -> None:
    """Overlay cluster ID text on each clustered chip."""
    clustered = df[df["is_defect"] & df["cluster_id"].notna() & (df["cluster_id"] >= 0)]
    for _, row in clustered.iterrows():
        ax.text(
            row[x_col], row[y_col], str(int(row["cluster_id"])),
            ha="center", va="center", fontsize=6, fontweight="bold", color="white",
        )


def render_single(
    df: pd.DataFrame, x_col: str, y_col: str, title: str,
    out_path: str | None, show: bool, show_labels: bool,
) -> None:
    x_range = df[x_col].max() - df[x_col].min() + 2
    y_range = df[y_col].max() - df[y_col].min() + 2
    base_size = 10
    if x_range >= y_range:
        fig_w, fig_h = base_size, base_size * y_range / x_range
    else:
        fig_w, fig_h = base_size * x_range / y_range, base_size
    fig_h = max(fig_h, 3)
    fig_w = max(fig_w, 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    build_wafer_map(df, x_col, y_col, ax, title)
    if show_labels:
        add_cluster_labels(df, x_col, y_col, ax)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize DBSCAN wafer map results")
    parser.add_argument("--input", required=True, help="Result CSV from wafer_dbscan.py")
    parser.add_argument("--x_col", default="X")
    parser.add_argument("--y_col", default="Y")
    parser.add_argument("--wafer_id", default=None,
                        help="Wafer ID column name (omit if single wafer)")
    parser.add_argument("--wafer", nargs="*", default=None,
                        help="Specific wafer IDs to plot (omit for all)")
    parser.add_argument("--out_dir", default=None,
                        help="Directory for output images (default: same dir as input file)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Image format (default: png)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save image files (use with --show)")
    parser.add_argument("--no_labels", action="store_true",
                        help="Do not overlay cluster ID numbers on chips")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    for col in [args.x_col, args.y_col, "is_defect", "cluster_id"]:
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Is this a wafer_dbscan.py result file?")

    df[args.x_col] = pd.to_numeric(df[args.x_col], errors="coerce")
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df["is_defect"] = df["is_defect"].astype(str).str.strip().str.lower() == "true"
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df["cluster_id"] = df["cluster_id"].astype("Int64")

    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.input))
    os.makedirs(args.out_dir, exist_ok=True)

    show_labels = not args.no_labels

    if args.wafer_id is None:
        out_path = None if args.no_save else os.path.join(args.out_dir, f"wafermap.{args.format}")
        render_single(df, args.x_col, args.y_col, "Wafer Map", out_path, args.show, show_labels)
        return

    if args.wafer_id not in df.columns:
        sys.exit(f"Column '{args.wafer_id}' not found. Available: {list(df.columns)}")

    wafer_ids = sorted(df[args.wafer_id].unique())
    if args.wafer:
        wafer_ids = [w for w in wafer_ids if str(w) in args.wafer]
        if not wafer_ids:
            sys.exit(f"No matching wafers. Available: {sorted(df[args.wafer_id].unique())}")

    os.makedirs(args.out_dir, exist_ok=True)

    for wid in wafer_ids:
        subset = df[df[args.wafer_id] == wid]
        safe_name = str(wid).replace("/", "_").replace("\\", "_")
        out_path = (
            None if args.no_save
            else os.path.join(args.out_dir, f"wafermap_{safe_name}.{args.format}")
        )
        render_single(
            subset, args.x_col, args.y_col,
            f"Wafer {wid}", out_path, args.show, show_labels,
        )

    print(f"Done - {len(wafer_ids)} wafer(s) processed.")


if __name__ == "__main__":
    main()
