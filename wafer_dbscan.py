"""
Wafer defect cluster detection using DBSCAN.

Usage:
    python wafer_dbscan.py --input data.csv --label BIN --defect_value 0
                           --eps 1.5 --min_samples 3 --output result.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def load_data(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("xls", "xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def run_dbscan(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    defect_value,
    eps: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    Flag defective chips and assign DBSCAN cluster IDs.

    Adds two columns to df:
      is_defect  : bool  — True when label_col == defect_value
      cluster_id : int   — >=0 cluster index, -1 = noise (isolated defect),
                           NaN = not a defect chip
    """
    result = df.copy()

    # --- cast defect_value to the same dtype as the label column -----------
    col_dtype = df[label_col].dtype
    try:
        typed_defect = col_dtype.type(defect_value)
    except (ValueError, TypeError):
        typed_defect = defect_value

    result["is_defect"] = result[label_col] == typed_defect

    defect_mask = result["is_defect"]
    result["cluster_id"] = pd.NA  # non-defect chips: no cluster

    if defect_mask.sum() == 0:
        print("No defective chips found with the given label/value.")
        return result

    coords = result.loc[defect_mask, [x_col, y_col]].to_numpy(dtype=float)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

    result.loc[defect_mask, "cluster_id"] = labels
    result["cluster_id"] = result["cluster_id"].astype("Int64")  # nullable int

    n_clusters = int((labels >= 0).sum() > 0 and labels.max() + 1 or 0)
    n_clusters = len(set(labels[labels >= 0]))
    n_noise = int((labels == -1).sum())

    print(f"Defective chips  : {defect_mask.sum()}")
    print(f"Clusters found   : {n_clusters}")
    print(f"Noise (isolated) : {n_noise}")

    return result


def summarize_clusters(result: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return per-cluster summary (chip count, bounding box, centroid)."""
    clustered = result[result["cluster_id"].notna() & (result["cluster_id"] >= 0)]
    if clustered.empty:
        return pd.DataFrame()

    summary = (
        clustered.groupby("cluster_id")
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


def visualize(result: pd.DataFrame, x_col: str, y_col: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Good chips
    good = result[~result["is_defect"]]
    ax.scatter(good[x_col], good[y_col], c="lightgray", s=20, label="Good", zorder=1)

    # Defective chips — colour by cluster
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
    ax.set_title("Wafer Defect Clusters (DBSCAN)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # wafer convention: row 0 at top

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
    parser.add_argument("--eps", type=float, default=1.5,
                        help="DBSCAN neighbourhood radius in chip-pitch units (default 1.5)")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="DBSCAN minimum chips to form a cluster (default 3)")
    parser.add_argument("--output", default="result.csv", help="Output CSV file")
    parser.add_argument("--plot", default="wafer_clusters.png", help="Output plot image")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    df = load_data(args.input)

    for col in (args.x_col, args.y_col, args.label):
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
    )

    summary = summarize_clusters(result, args.x_col, args.y_col)
    if not summary.empty:
        print("\nCluster summary:")
        print(summary.to_string(index=False))

    result.to_csv(args.output, index=False)
    print(f"\nResult saved to {args.output}")

    if not args.no_plot:
        visualize(result, args.x_col, args.y_col, args.plot)


if __name__ == "__main__":
    main()
