日本株 予測Webアプリ v2（スキーマ自動検出・可変予測窓）
=====================================================

本プロジェクトは、日本株の株価を「可視化 + 予測」するフルスタックWebアプリです。既存のSQLiteデータベース（テーブル名: `prices`）のスキーマを自動検出し、その結果に適応して予測を行います。

構成
----
- バックエンド: FastAPI (Python 3.11+), ポート 8082
- フロントエンド: React + Vite + TypeScript + Tailwind, ポート 4002
- データベース: SQLite（`.env` の `DB_PATH` で指定、既定は `/Users/takashiui/Documents/Python/stock_analysis/stock_data.db`）

クイックスタート
----------------
1) 環境変数用意

```
cp .env.example .env
cp backend/.env.example backend/.env
```

2) セットアップ

```
make setup
```

3) 同時起動（バックエンド + フロント）

```
make dev
```

4) アクセス

- フロント: `http://localhost:4002/?stock=0001&lookback=126&horizon=63&method=ensemble&feature_mode=auto`
- バックエンド: `http://localhost:8082`

環境変数（.env）
-----------------
- `DB_PATH` : SQLite DB の絶対パス（テーブルは `prices` 固定）
- `API_PORT`: バックエンドポート（既定 8082）
- `WEB_PORT`: フロントポート（既定 4002）
- `VITE_API_BASE_URL`: フロントからのAPIベースURL（既定 `http://localhost:8082`）

スキーマ自動検出（超重要）
----------------------------
- バックエンド起動時に SQLite へ接続し、`PRAGMA table_info('prices')` と `SELECT * FROM prices LIMIT 1` を実行。
- 論理列マッピングを推定（優先候補に基づく）
  - `code`: [`stock_code`, `code`, `ticker`, `symbol`]
  - `date`: [`date`, `trade_date`, `datetime`]
  - `open`: [`open`, `Open`, `o`]
  - `high`: [`high`, `High`, `h`]
  - `low` : [`low`, `Low`, `l`]
  - `close`: [`adj_close`, `adjclose`, `close`, `price`]
  - `volume`: [`volume`, `vol`, `shares`, `turnover`]
- OHLCV/日付/コード以外の数値列（REAL/INTEGER）は「追加特徴量候補」として抽出
- 検出結果は `backend/db/schema_cache.json` にキャッシュされ、`GET /api/meta` から取得可能
- 列名に数値始まり（例: `150MA`）などSQL的に特別な名前が含まれていても、内部で安全にクオートして参照します

API 一覧
--------
- `GET /api/health` → `{ "status": "ok" }`
- `GET /api/meta` → スキーマ検出結果（列・論理マップ・追加数値列）
- `GET /api/stocks?q=&limit=20` → 銘柄コードサジェスト（英数字可、パラメトリックLIKE）
- `GET /api/prices?stock=XXXX&start=YYYY-MM-DD&end=YYYY-MM-DD` → ヒストリカル（`close` は `adj_close > close > price` 優先で採用）
- `GET /api/forecast?stock=XXXX&lookback_days=126&horizon_days=63&method=ensemble&feature_mode=auto&seed=42`
  - 出力: `dates`, `observed`, `predicted_path(p50)`, `quantiles(p10/p50/p90)`, `diagnostics`

予測手法（v2 実装）
-------------------
- baseline: ランダムウォーク + ドリフト。学習窓のボラで正規近似 → p10/p50/p90。
- direct: ElasticNetCV(TimeSeriesSplit)で「累積ログリターン」を直接回帰。残差の経験分布で帯推定。
- multistep: 1日先GBR（GradientBoosting）。lag_ret_* を逐次更新し、残差ブートストラップで帯形成。
- ensemble: 1スライスWF-CVのMAEから温度付きソフトマックスで重み算出（安定化のため軽いクリッピング）。

パラメータと特徴量
------------------
- `lookback_days`: 学習窓の営業日数（例: 126）
- `horizon_days`: 予測地平の営業日数（例: 63）
- `feature_mode`:
  - `ohlcv_only`: OHLCV由来の標準特徴のみ
  - `auto`: OHLCV + 有望な追加数値列を自動選択（相関・欠損率などで簡易フィルタ）
  - `all_numeric`: 使える数値列を全投入（正則化前提）
- リーケージ防止: ラグ/ローリング/expandingを用い、学習時点までの情報のみを使用。

フロントエンド
---------------
- ルート `/` はURLクエリを解釈し、`/api/meta`→`/api/prices`→`/api/forecast` の順で取得。
- ParamsPanel で `stock / lookback / horizon / method / feature_mode` を切替。
- Chart.js で観測線、p50 破線、p10–p90 の帯を描画。

テスト
------
- 軽量APIテスト: `make test`（`/api/health`, `/api/meta`）

推奨インデックス
-----------------
大量データ時の性能向上:

```sql
CREATE INDEX IF NOT EXISTS idx_prices_code_date ON prices(stock_code, date);
```

トラブルシューティング
----------------------
- 予測が出ない/帯が崩れる: ブラウザのネットワークタブで `/api/forecast` のHTTPステータスと `diagnostics` の `components.error` を確認。
- 予測差が小さい: `feature_mode=auto` に切替、`lookback_days` を調整。
- 重い: `horizon_days` を短縮、`feature_mode=ohlcv_only` を試行。

既知の制約・今後の改善
----------------------
- 分位推定の強化（Quantile回帰・より厳密な校正）
- 外部説明変数（FX/金利/VIX など）の導入
- マルチスライスWF-CVによる重み推定の堅牢化

開発コマンド
------------
- セットアップ: `make setup`
- 開発起動: `make dev`
- テスト: `make test`

Git へのプッシュ
-----------------
既に `.git` は初期化済みです。初回コミット/プッシュは以下を参考にしてください。

```
git add -A
git commit -m "docs: 日本語README追加と予測手法の精緻化"
# ブランチ名を main に統一したい場合
git branch -M main
# まだリモート未設定の場合
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

