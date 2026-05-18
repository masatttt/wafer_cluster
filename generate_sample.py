"""
Generate a sample wafer CSV for testing wafer_dbscan.py.

Produces a 25x25 chip grid with:
  - 3 seeded defect clusters
  - some random isolated defects (noise)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

GRID = 25  # chips per side

xs, ys = np.meshgrid(range(GRID), range(GRID))
df = pd.DataFrame({"X": xs.ravel(), "Y": ys.ravel()})
df["BIN"] = 1  # 1 = good

# --- seeded clusters -------------------------------------------------------
clusters = [
    [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3)],           # cluster 0
    [(12, 10), (13, 10), (12, 11), (13, 11), (14, 11), (12, 12)],  # cluster 1
    [(20, 18), (21, 18), (20, 19), (21, 19)],            # cluster 2
]
for chips in clusters:
    for x, y in chips:
        df.loc[(df["X"] == x) & (df["Y"] == y), "BIN"] = 0

# --- random isolated defects -----------------------------------------------
noise_idx = RNG.choice(df.index[(df["BIN"] == 1)], size=8, replace=False)
df.loc[noise_idx, "BIN"] = 0

df.to_csv("sample_wafer.csv", index=False)
print(f"Saved sample_wafer.csv  ({len(df)} chips, {(df['BIN']==0).sum()} defects)")
