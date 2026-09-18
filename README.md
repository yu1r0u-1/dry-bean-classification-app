# 🫘 豆の品種分類AI

豆の形状を表す数値データから、機械学習モデルが豆の品種を予測するWebアプリです。

データの探索・モデルの比較と評価・誤分類分析・特徴量重要度の確認から、Streamlitを使ったWebアプリ化までを一貫して行いました。

## アプリの概要

豆の面積、周囲長、軸の長さ、形状指標などを入力すると、学習済みモデルが7種類の豆から該当する品種を予測します。

予測された品種だけでなく、各品種に分類される確率も確認できます。

## 予測対象の品種

* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SEKER
* SIRA

## 主な機能

* 豆の特徴量をフォームから入力
* 学習済みモデルによる品種予測
* 各品種の予測確率を表示
* 入力した数値を表形式で確認
* Streamlitによるブラウザ上での操作

## 使用技術

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* joblib
* Streamlit

## ディレクトリ構成

```text
seed-sort-ai/
├── app.py                         # Streamlitアプリ
├── train_model.py                 # モデルの学習・評価・保存
├── models/
│   └── bean_model.joblib          # 学習済みモデル
├── notebooks/
│   ├── 01_EDA.py                  # データ分析
│   └── Dry_Bean_Dataset.csv       # 使用データ
├── requirements.txt               # 必要なライブラリ
└── README.md
```

## セットアップ

### 1. リポジトリを取得

```bash
git clone <このリポジトリのURL>
cd seed-sort-ai
```

### 2. 仮想環境を作成

Windowsの場合：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS・Linuxの場合：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

## アプリの起動方法

プロジェクトのフォルダで、次のコマンドを実行します。

```bash
streamlit run app.py
```

コマンドを実行するとブラウザ上でアプリが開き、豆の特徴量を入力して品種を予測できます。

## モデルの再学習

学習済みモデルを作り直す場合は、次のコマンドを実行します。

```bash
python train_model.py
```

モデルの学習では、主に以下の処理を行います。

1. データの読み込み
2. 説明変数と目的変数への分割
3. 学習データとテストデータへの分割
4. 前処理を含むPipelineの構築
5. 複数モデルの比較
6. 選択したモデルの学習と評価
7. 誤分類の組み合わせの分析
8. 特徴量重要度の確認
9. 学習済みモデルのjoblib形式での保存

## データセット

[UCI Machine Learning RepositoryのDry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)を使用しています。

豆の画像から抽出された形状情報をもとに、7種類の豆を分類するデータセットです。13,611個の豆について、16種類の特徴量と品種ラベルが収録されています。

### 主な特徴量

| 特徴量               | 内容              |
| ----------------- | --------------- |
| `Area`            | 豆の面積            |
| `Perimeter`       | 豆の周囲長           |
| `MajorAxisLength` | 長軸の長さ           |
| `MinorAxisLength` | 短軸の長さ           |
| `AspectRatio`     | 長軸と短軸の比率        |
| `Eccentricity`    | 楕円の細長さ          |
| `ConvexArea`      | 凸包の面積           |
| `EquivDiameter`   | 同じ面積を持つ円の直径     |
| `Extent`          | 外接矩形に対する豆の面積の割合 |
| `Solidity`        | 凸包に対する豆の面積の割合   |
| `Roundness`       | 形状の円形度          |
| `Compactness`     | 形状のコンパクトさ       |

### 出典とライセンス

* **データセット名:** Dry Bean Dataset
* **著者:** Murat Koklu、Ilker Ali Ozkan
* **提供元:** UCI Machine Learning Repository
* **DOI:** [10.24432/C50S4B](https://doi.org/10.24432/C50S4B)
* **ライセンス:** [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/)

引用情報：

> Dry Bean [Dataset]. (2020). UCI Machine Learning Repository.
> https://doi.org/10.24432/C50S4B

データセットにはCC BY 4.0ライセンスが適用されます。このライセンスは、本リポジトリのソースコードには適用されません。

## 開発で取り組んだこと

* 基本統計量やクラス分布を確認するEDA
* 前処理とモデルをまとめたPipelineの作成
* DummyClassifierを基準としたモデル比較
* 交差検証による性能確認
* 混同行列と誤分類数によるエラー分析
* Permutation Importanceなどによる特徴量重要度の確認
* 学習処理と予測アプリの分離
* Streamlitによる機械学習モデルのWebアプリ化

## 今後の改善案

* テストデータに対する評価指標を画面上に掲載する
* 入力例や特徴量の説明を追加して操作性を改善する
* 誤分類されやすい品種同士の違いを可視化する
* モデルやハイパーパラメータをさらに比較する
* Streamlit Community Cloudなどへデプロイする
* 自動テストを追加する

## 注意事項

このアプリは、機械学習の学習およびポートフォリオ作成を目的としています。

予測結果は、実際の種子選別や品質保証への利用を想定したものではありません。

