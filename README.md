# Wafer Defect Cluster Detection (DBSCAN)

Wafer上に格子状に配置されたChipの不良クラスタを [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) で検出するPythonツールです。

## Features

- CSV / Excel 入力に対応
- **複数Wafer一括処理** — Wafer IDカラム指定でWaferごとに独立してDBSCAN実行
- 指定ラベル・指定値で不良チップを判定（`--negate` で条件反転も可能）
- DBSCANによるクラスタリングで不良の塊を自動検出
- クラスタごとのサマリをCSV出力（チップ数、座標範囲、重心）
- Waferマップの可視化画像をWaferごとに出力（縦横比は自動調整）
- 出力フォルダ名にパラメータ条件を自動付与（例: `eps1p5_min3/`）

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
    --min_samples 3
```

### Options

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | 入力ファイル (CSV / Excel) |
| `--x_col` | `X` | X座標のカラム名 |
| `--y_col` | `Y` | Y座標のカラム名 |
| `--label` | (required) | 不良判定に使うカラム名 |
| `--defect_value` | (required) | 不良とみなす値 |
| `--negate` | - | 条件反転: `defect_value` に一致**しない**チップを不良とみなす |
| `--wafer_id` | (なし) | Wafer IDカラム名（省略時は全データを1枚として処理） |
| `--eps` | `1.5` | DBSCAN近傍半径（チップピッチ単位） |
| `--min_samples` | `3` | クラスタ形成に必要な最小チップ数 |
| `--out_dir` | (auto) | 出力先ベースディレクトリ（省略時はカレント） |
| `--no_plot` | - | 画像出力をスキップ |

### 不良判定の例

```bash
# BIN が 0 のチップを不良とする
--label BIN --defect_value 0

# BIN が 1 以外のチップを不良とする
--label BIN --defect_value 1 --negate
```

### Output

出力は `eps{値}_min{値}/` フォルダに自動整理されます。`--out_dir` を指定すると、その配下に条件フォルダが作られます。

```
eps1p5_min3/
  result.csv            # 全チップの判定結果
  cluster_summary.csv   # クラスタごとのサマリ
  wafer_clusters_W01.png
  wafer_clusters_W02.png
  ...
```

| File | Description |
|---|---|
| `result.csv` | Wafer ID, X, Y, ラベル, `is_defect`, `cluster_id` |
| `cluster_summary.csv` | クラスタごとのチップ数・座標範囲・重心 |

`cluster_id` の意味:
- `>=0`: クラスタ番号（Waferごとに独立採番）
- `-1`: ノイズ（孤立した不良、クラスタ未所属）
- `NaN`: 良品チップ

## eps の選び方

チップが1ピッチ間隔の格子配置の場合:

| eps | 近傍の範囲 |
|---|---|
| `1.0` | 上下左右の4方向のみ |
| `1.5` | 上下左右＋斜め（8方向） |
| `2.5` | 2チップ先まで許容 |

## Wafer Map 可視化 (visualize_wafermap.py)

DBSCAN結果をグリッド形式のWafer Mapとして描画します。各チップにクラスタIDが表示されます。画像の縦横比はWaferのX/Y範囲から自動調整されます。

```bash
# 全Waferの画像を出力（result.csv と同じフォルダに保存）
python visualize_wafermap.py --input eps1p5_min3/result.csv --wafer_id WAFER_ID

# 特定Waferだけ
python visualize_wafermap.py --input eps1p5_min3/result.csv --wafer_id WAFER_ID --wafer W01

# 画面表示のみ（ファイル保存なし）
python visualize_wafermap.py --input eps1p5_min3/result.csv --wafer_id WAFER_ID --show --no_save
```

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | wafer_dbscan.py の出力CSV |
| `--wafer_id` | (なし) | Wafer IDカラム名 |
| `--wafer` | (全Wafer) | 描画対象のWafer IDを指定 (複数可) |
| `--out_dir` | (inputと同じフォルダ) | 画像出力先ディレクトリ |
| `--format` | `png` | 出力形式 (`png` / `pdf` / `svg`) |
| `--show` | - | インタラクティブ表示 |
| `--no_save` | - | ファイル保存をスキップ |
| `--no_labels` | - | チップ上のクラスタID表示を消す |

## Quick Start (サンプルデータ)

```bash
python generate_sample.py
python wafer_dbscan.py --input sample_wafer.csv --label BIN --defect_value 0 --wafer_id WAFER_ID
python visualize_wafermap.py --input eps1p5_min3/result.csv --wafer_id WAFER_ID
```

## License

MIT
