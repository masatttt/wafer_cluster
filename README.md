# Wafer Defect Cluster Detection (DBSCAN)

Wafer上に格子状に配置されたChipの不良クラスタを [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) で検出するPythonツールです。

## Features

- CSV / Excel 入力に対応
- **複数Wafer一括処理** — Wafer IDカラム指定でWaferごとに独立してDBSCAN実行
- 指定ラベル・指定値で不良チップを判定
- DBSCANによるクラスタリングで不良の塊を自動検出
- クラスタごとのサマリ出力（チップ数、座標範囲、重心）
- Waferマップの可視化画像をWaferごとに出力

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python wafer_dbscan.py \
    --input data.csv \
    --wafer_id WAFER_ID \
    --label BIN \
    --defect_value 0 \
    --eps 1.5 \
    --min_samples 3 \
    --output result.csv \
    --plot wafer_clusters.png
```

### Options

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | 入力ファイル (CSV / Excel) |
| `--x_col` | `X` | X座標のカラム名 |
| `--y_col` | `Y` | Y座標のカラム名 |
| `--label` | (required) | 不良判定に使うカラム名 |
| `--defect_value` | (required) | 不良とみなす値 |
| `--wafer_id` | (なし) | Wafer IDカラム名（省略時は全データを1枚として処理） |
| `--eps` | `1.5` | DBSCAN近傍半径（チップピッチ単位） |
| `--min_samples` | `3` | クラスタ形成に必要な最小チップ数 |
| `--output` | `result.csv` | 出力CSVファイル |
| `--plot` | `wafer_clusters.png` | 出力画像ファイル |
| `--no_plot` | - | 画像出力をスキップ |

### Output Columns

| Column | Description |
|---|---|
| `is_defect` | 指定条件に合致した不良チップ (`True` / `False`) |
| `cluster_id` | `≥0`: クラスタ番号（Waferごとに独立採番）, `-1`: ノイズ（孤立不良）, `NaN`: 良品 |

### Plot Output

- `--wafer_id` 指定時: Waferごとに `wafer_clusters_W01.png`, `wafer_clusters_W02.png`, ... と個別画像を出力
- `--wafer_id` 省略時: 1枚の `wafer_clusters.png` を出力

## eps の選び方

チップが1ピッチ間隔の格子配置の場合:

| eps | 近傍の範囲 |
|---|---|
| `1.0` | 上下左右の4方向のみ |
| `1.5` | 上下左右＋斜め（8方向） |
| `2.5` | 2チップ先まで許容 |

## Quick Start (サンプルデータ)

```bash
python generate_sample.py
python wafer_dbscan.py --input sample_wafer.csv --label BIN --defect_value 0 --wafer_id WAFER_ID
```

## License

MIT
