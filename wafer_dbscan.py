"""
Wafer defect cluster detection using DBSCAN.

Usage:
    python wafer_dbscan.py --input data.csv --label BIN --defect_value 0
                           --eps 1.5 --min_samples 3 --output result.csv
                           --wafer_id WAFER_ID
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def load_data(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("xls", "xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def run_dbscan_single(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    defect_value,
    eps: float,
    min_samples: int,
) -> pd.DataFrame:
    """Run DBSCAN on a single wafer's data."""
    result = df.copy()

    col_dtype = df[label_col].dtype
    try:
        typed_defect = col_dtype.type(defect_value)
    except (ValueError, TypeError):
        typed_defect = defect_value

    result["is_defect"] = result[label_col] == typed_defect

    defect_mask = result["is_defect"]
    result["cluster_id"] = pd.NA

    if defect_mask.sum() == 0:
        return result

    coords = result.loc[defect_mask, [x_col, y_col]].to_numpy(dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    result.loc[defect_mask, "cluster_id"] = labels
    result["cluster_id"] = result["cluster_id"].astype("Int64")

    return result


def run_dbscan(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    defect_value,
    eps: float,
    min_samples: int,
    wafer_col: str | None = None,
) -> pd.DataFrame:
    """Run DBSCAN per wafer. If wafer_col is None, treat all data as one wafer."""

    if wafer_col is None:
        result = run_dbscan_single(
            df, x_col, y_col, label_col, defect_value, eps, min_samples
        )
        _print_stats(result, "ALL")
        return result

    parts = []
    for wafer_id, group in df.groupby(wafer_col, sort=True):
        part = run_dbscan_single(
            group, x_col, y_col, label_col, defect_value, eps, min_samples
        )
        _print_stats(part, wafer_id)
        parts.append(part)

    return pd.concat(parts).reset_index(drop=True)


def _print_stats(result: pd.DataFrame, wafer_id) -> None:
    defect_mask = result["is_defect"]
    n_defect = defect_mask.sum()
    if n_defect == 0:
        print(f"[{wafer_id}] No defective chips found.")
        return
    cluster_ids = result.loc[defect_mask, "cluster_id"].dropna()
    n_clusters = len(set(cid for cid in cluster_ids if cid >= 0))
    n_noise = int((cluster_ids == -1).sum())
    print(f"[{wafer_id}] Defects: {n_defect}, Clusters: {n_clusters}, Noise: {n_noise}")


def summarize_clusters(
    result: pd.DataFrame, x_col: str, y_col: str, wafer_col: str | None = None
) -> pd.DataFrame:
    clustered = result[result["cluster_id"].notna() & (result["cluster_id"] >= 0)]
    if clustered.empty:
        return pd.DataFrame()

    group_keys = [wafer_col, "cluster_id"] if wafer_col else ["cluster_id"]

    summary = (
        clustered.groupby(group_keys)
        .agg(
            chip_count=(x_col, "count"),
            x_min=(x_col, "min"),
            x_max=(x_col, "max"),
            y_min=(y_col, "min"),
            y_max=(y_col, "max"),
            x_center=(x_col, "mean"),
            y_center=(y_col, "mean"),
        )
        .reset_index()
    )
    return summary


def visualize(
    result: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: str,
    wafer_col: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    if wafer_col is None:
        _plot_single(result, x_col, y_col, out_path, title="Wafer Defect Clusters")
        return

    base, ext = os.path.splitext(out_path)
    for wafer_id, group in result.groupby(wafer_col, sort=True):
        safe_name = str(wafer_id).replace("/", "_").replace("\\", "_")
        path = f"{base}_{safe_name}{ext}"
        _plot_single(group, x_col, y_col, path, title=f"Wafer {wafer_id}")


def _plot_single(
    result: pd.DataFrame, x_col: str, y_col: str, out_path: str, title: str
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, 8))

    good = result[~result["is_defect"]]
    ax.scatter(good[x_col], good[y_col], c="lightgray", s=20, label="Good", zorder=1)

    defects = result[result["is_defect"]]
    cluster_ids = defects["cluster_id"].dropna().unique()

    cmap = plt.cm.get_cmap("tab10", max(len(cluster_ids), 1))
    handles = []

    for cid in sorted(cluster_ids):
        mask = defects["cluster_id"] == cid
        color = "black" if cid == -1 else cmap(int(cid) % 10)
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        ax.scatter(
            defects.loc[mask, x_col],
            defects.loc[mask, y_col],
            c=[color],
            s=60,
            zorder=2,
        )
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title} (DBSCAN)")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Wafer defect DBSCAN clustering")
    parser.add_argument("--input", required=True, help="Input CSV/Excel file")
    parser.add_argument("--x_col", default="X", help="Column name for X coordinate")
    parser.add_argument("--y_col", default="Y", help="Column name for Y coordinate")
    parser.add_argument("--label", required=True, help="Column name used as defect flag")
    parser.add_argument("--defect_value", required=True, help="Value that indicates a defect")
    parser.add_argument("--wafer_id", default=None,
                        help="Column name for wafer ID (omit to treat all data as one wafer)")
    parser.add_argument("--eps", type=float, default=1.5,
                        help="DBSCAN neighbourhood radius in chip-pitch units (default 1.5)")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="DBSCAN minimum chips to form a cluster (default 3)")
    parser.add_argument("--output", default="result.csv", help="Output CSV file")
    parser.add_argument("--plot", default="wafer_clusters.png", help="Output plot image")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    df = load_data(args.input)

    required_cols = [args.x_col, args.y_col, args.label]
    if args.wafer_id:
        required_cols.append(args.wafer_id)
    for col in required_cols:
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Available: {list(df.columns)}")

    result = run_dbscan(
        df,
        x_col=args.x_col,
        y_col=args.y_col,
        label_col=args.label,
        defect_value=args.defect_value,
        eps=args.eps,
        min_samples=args.min_samples,
        wafer_col=args.wafer_id,
    )

    summary = summarize_clusters(result, args.x_col, args.y_col, args.wafer_id)
    if not summary.empty:
        print("\nCluster summary:")
        print(summary.to_string(index=False))

    result.to_csv(args.output, index=False)
    print(f"\nResult saved to {args.output}")

    if not args.no_plot:
        visualize(result, args.x_col, args.y_col, args.plot, args.wafer_id)


if __name__ == "__main__":
    main()
