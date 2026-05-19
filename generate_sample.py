"""
Generate a sample wafer CSV for testing wafer_dbscan.py.

Produces 3 wafers (25x25 grid each) with different defect patterns.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

GRID = 25

wafers = []

# --- Wafer definitions: (wafer_id, cluster_list, n_noise) ------------------
definitions = [
    (
        "W01",
        [
            [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3)],
            [(12, 10), (13, 10), (12, 11), (13, 11), (14, 11), (12, 12)],
            [(20, 18), (21, 18), (20, 19), (21, 19)],
        ],
        8,
    ),
    (
        "W02",
        [
            [(7, 7), (7, 8), (8, 7), (8, 8), (9, 8), (9, 9)],
        ],
        5,
    ),
    (
        "W03",
        [
            [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2)],
            [(15, 15), (16, 15), (15, 16), (16, 16), (17, 16), (16, 17)],
            [(22, 5), (23, 5), (22, 6), (23, 6)],
            [(10, 20), (11, 20), (10, 21), (11, 21), (12, 21)],
        ],
        3,
    ),
]

for wafer_id, clusters, n_noise in definitions:
    xs, ys = np.meshgrid(range(GRID), range(GRID))
    df = pd.DataFrame({"WAFER_ID": wafer_id, "X": xs.ravel(), "Y": ys.ravel()})
    df["BIN"] = 1

    for chips in clusters:
        for x, y in chips:
            df.loc[(df["X"] == x) & (df["Y"] == y), "BIN"] = 0

    good_idx = df.index[df["BIN"] == 1]
    noise_idx = RNG.choice(good_idx, size=n_noise, replace=False)
    df.loc[noise_idx, "BIN"] = 0

    wafers.append(df)

result = pd.concat(wafers, ignore_index=True)
result.to_csv("sample_wafer.csv", index=False)

for wid, g in result.groupby("WAFER_ID"):
    n_def = (g["BIN"] == 0).sum()
    print(f"  {wid}: {len(g)} chips, {n_def} defects")
print(f"Saved sample_wafer.csv ({len(result)} chips total)")
