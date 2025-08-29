# Codex CLI 指示書（日本株 予測Webアプリ v2：スキーマ自動検出・可変予測窓）

> **目的**：このワークスペースに、日本株の**株価予測 & 可視化**フルスタックWebアプリを構築してください。  
> **前提**：ローカル **Codex CLI**（承認モード: Auto / 作業ディレクトリ内のみ書込）。  
> **重要**：まず**SQLiteスキーマを自動検出**してから、検出結果に適応する「理想実装」を生成してください。

---

## 0) ゴール（Definition of Done）

- **Frontend**：React + Vite + TypeScript + Tailwind（devポート **4002**）。
- **Backend**：FastAPI（Python 3.11+、ポート **8082**）。
- **Database**：SQLite。DBパスは `.env` の `DB_PATH` を用い、既定は **`/Users/takashiui/Documents/Python/stock_analysis/stock_data.db`**。  
  価格テーブルは **`prices`**。
- **stock_code**：**英数字混在を許容**（例：`218A`）。**特例**として `'0001'` = **日経平均**、`'0002'` = **TOPIX**（これらも通常の銘柄コードとして扱う）。
- **URLクエリ**：`?stock={code}&lookback={n}&horizon={m}&method={name}&feature_mode={mode}` を受け取り、対象銘柄について：
  1. ヒストリカル価格（OHLC/終値）を描画
  2. **学習窓（lookback: 任意営業日）** → **予測地平（horizon: 任意営業日）** を実行
  3. **点推定＋区間（p10/p50/p90）** を帯で可視化
  4. モデル切替（`baseline` / `direct` / `multistep` / `ensemble`）
  5. **feature_mode**（`auto` | `ohlcv_only` | `all_numeric`）で特徴量の範囲を選択
- **主要API**：
  - `GET /api/health` → `{"status":"ok"}`
  - `GET /api/meta` → スキーマ検出結果（列一覧・推定マッピング・利用可能なオプション）
  - `GET /api/stocks?q=...&limit=20` → `stock_code` サジェスト（英数字対応）
  - `GET /api/prices?stock=XXXX&start=YYYY-MM-DD&end=YYYY-MM-DD`
  - `GET /api/forecast?stock=XXXX&lookback_days=126&horizon_days=63&method=ensemble&feature_mode=auto&seed=42`
    - **出力JSON**：
      ```json
      {
        "dates": ["2025-01-01", "..."],
        "observed": [/* 予測直前までの終値（価格） */],
        "predicted_path": [/* p50 価格推移 */],
        "quantiles": { "p10": [...], "p50": [...], "p90": [...] },
        "diagnostics": {
          "method": "ensemble",
          "components": { "baseline": {...}, "direct": {...}, "multistep": {...} },
          "cv": { "mae": ..., "rmse": ... },
          "feature_mode": "auto",
          "schema_map": { "code": "stock_code", "date": "date", "close": "price", "open": "open", "high": "high", "low": "low", "volume": "volume" }
        }
      }
      ```
- **CORS**：`http://localhost:4002` を許可。
- **Makefile**：`make setup` / `make dev` / `make test` を備える。
- **README**：起動手順、環境変数、DB差替、スキーマ検出の挙動、既知の制約、拡張案。
- **軽量テスト**：APIヘルス・簡易予測のユニットテスト 1-2 本。

---

## 1) データベース仕様とスキーマ自動検出

### 1-1. 既知の前提
- DB：`/Users/takashiui/Documents/Python/stock_analysis/stock_data.db`
- テーブル：`prices`（固定）

### 1-2. まず最初に行う処理（**必須**）
1. SQLite に接続し、`PRAGMA table_info('prices')` と `SELECT * FROM prices LIMIT 1` を実行。  
2. 列名一覧と型の候補を得て**スキーママップ**を構築：
   - **必須の論理列**（見つかったものだけで可）
     - `code`: 候補 `["stock_code","code","ticker","symbol"]`
     - `date`: 候補 `["date","trade_date","datetime"]`
     - `open`: 候補 `["open","Open","o"]`
     - `high`: 候補 `["high","High","h"]`
     - `low` : 候補 `["low","Low","l"]`
     - `close`: 候補 `["adj_close","AdjClose","close","Close","price","Price"]`
     - `volume`: 候補 `["volume","Volume","vol","shares","turnover"]`
   - **任意の追加数値列**：上記以外の数値（`REAL/INTEGER`）列は **追加特徴**として利用可能（`feature_mode=all_numeric` または `auto` のとき）。
3. 推定結果を `backend/db/schema_cache.json` に保存し、`/api/meta` から返せるようにする。  
4. `code` 列は **TEXT** として扱い、**英数字を許容**（例：`218A`）。

### 1-3. インデックス推奨
- ユーザ承認時に次を提案（自動実行は要承認）：
  ```sql
  CREATE INDEX IF NOT EXISTS idx_prices_code_date ON prices(stock_code, date);
  ```

---

## 2) プロジェクト構成（生成物の要求）

```
.
├── backend/
│   ├── app.py
│   ├── core/config.py
│   ├── db/session.py
│   ├── db/inspect.py          # PRAGMAとサンプル行からスキーマ自動検出 → schema_cache.json
│   ├── models/price.py        # ORM（必要最小限。直クエリ中心でもOK）
│   ├── services/prices.py     # 価格取得/整形（スキーママップ対応）
│   ├── services/features.py   # 特徴量生成（スキーマに応じてOHLCV/追加数値列を扱う）
│   ├── services/forecast.py   # 予測ロジック（baseline/direct/multistep/ensemble）
│   ├── schemas/               # pydantic models
│   ├── tests/test_basic.py
│   ├── requirements.txt  (or pyproject.toml)
│   └── .env.example
├── frontend/
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── pages/Home.tsx
│       ├── components/StockSearch.tsx
│       ├── components/PriceChart.tsx
│       ├── components/ParamsPanel.tsx
│       └── lib/api.ts
├── data/               # 空でOK（DBは絶対パス参照）。
├── .env.example
├── Makefile
├── README.md
└── .gitignore
```

**Git**：初期化・初回コミット（`node_modules`, `.venv`, `.env` は除外）。

---

## 3) Backend 実装詳細（FastAPI 8082）

### 3-1. 依存
- `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`
- `SQLAlchemy`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`

### 3-2. 設定とCORS
- `.env` で `DB_PATH`, `API_PORT`, `WEB_PORT` を読み込み。未設定時は既定を使用。  
- CORSは `http://localhost:4002` を許可。

### 3-3. スキーマ検出とメタ情報
- `backend/db/inspect.py`：
  - DB接続 → 列一覧と型の収集 → **スキーママッピング**の決定（上記候補優先）。
  - 追加数値列の抽出（`code/date/OHLCV` を除く数値列）。
  - `schema_cache.json` に保存（`/api/meta` が返却）。

### 3-4. API
- `GET /api/health` → `{"status":"ok"}`
- `GET /api/meta` → `{ columns, schema_map, extra_numeric_columns }`
- `GET /api/stocks?q=&limit=20`
  - `SELECT DISTINCT <code_col> FROM prices WHERE <code_col> LIKE :q ORDER BY 1 LIMIT :limit`
  - **英数字**対応、**パラメトリック**クエリ（SQLインジェクション防止）。
- `GET /api/prices?stock=XXXX&start=YYYY-MM-DD&end=YYYY-MM-DD`
  - 返却：`{ dates, open?, high?, low?, close, volume? }`
  - `close` は `adj_close` > `close` > `price` の順で採用（存在するもの）。
- `GET /api/forecast?stock=XXXX&lookback_days=126&horizon_days=63&method=ensemble&feature_mode=auto&seed=42`
  - **可変**：`lookback_days` と `horizon_days` は**任意**に設定可能（営業日換算、DBの連続日列を使用）。
  - **feature_mode**：
    - `ohlcv_only`：OHLCVから構築した**標準特徴**のみ
    - `auto`：OHLCVに加え、**情報量の高そうな追加数値列**を自動選択（相関・欠損率・定常性の簡易チェックでフィルタ）
    - `all_numeric`：使用可能な**全数値列**（過学習に注意しL1/L2で正則化）
  - **特徴量（例）**：
    - リターン系：log return、累積、ラグ（1,5,10,21,63）
    - トレンド/モメンタム：SMA/EMA、RSI(14)
    - ボラ：rolling std(21,63)、**パーキンソン** or **ガーマン–クラース**（`high/low/open/close` が揃う場合）
    - 出来高：`volume` の変化率、z-score、**OBV**（closeと併用）
    - 追加数値列：z-score化し、リーケージ防止のため**学習データでfit→全期間にtransform**
  - **モデル**：
    1. **baseline**：Random Walk + drift。過去ボラで正規近似 → p10/p50/p90。
    2. **direct**：
       - 63日直**回帰**（ElasticNet or Lasso）で「累積リターン」を直接予測。  
       - `statsmodels` **ARIMA** もバックアップ。  
       - 分位点は**残差の経験分布**または**QuantileRegressor/GBR(loss='quantile')**で推定。
    3. **multistep**：1日先回帰器（`GradientBoostingRegressor`）を**逐次63回**。残差ブートストラップで帯形成。
    4. **ensemble**：上記の**逆誤差重み**で加重（簡易WF-CVでMAE/RMSE算出）。
  - **出力**：観測最終日の**翌営業日から** `horizon_days` の価格系列（p50）。p10/p90 は価格レベル。  
  - **乱数**：`seed` で固定可能。

### 3-5. 検証
- 最新期の直前で1スライスの**ウォークフォワードCV**を実行し、`diagnostics.cv` に格納。
- データ不足（`lookback_days` 未満）は 400 を返し理由を明記。

---

## 4) Frontend 実装（React + Vite + TS + Tailwind, 4002）

### 4-1. 依存
- `react`, `react-dom`, `react-router-dom`
- `axios`
- `chart.js`, `react-chartjs-2`
- `zustand`（状態管理）

### 4-2. 動作
- Vite の dev ポートは **4002**。
- `.env` に `VITE_API_BASE_URL`（既定 `http://localhost:8082`）。
- ルート `/`：URLクエリを読み取り、初回ロードで `/api/meta` → `/api/prices` → `/api/forecast` を順に取得。
- **ParamsPanel**：
  - `stock` 入力（サジェスト連動）
  - `lookback`（例：126）と `horizon`（例：63）を**自由に設定**
  - `method` 選択（baseline/direct/multistep/ensemble）
  - `feature_mode` 選択（auto/ohlcv_only/all_numeric）
- **PriceChart**：
  - 観測（ヒストリカル）ライン
  - 予測 p50 ライン
  - `p10`〜`p90` の**帯（背景）**
- **エラー表示**：銘柄未指定・データ不足・サーバエラー時の丁寧なメッセージ。

---

## 5) Makefile / スクリプト / 起動

- `Makefile`：
  ```make
  setup:
  	python3 -m venv .venv && . .venv/bin/activate && pip install -r backend/requirements.txt
  	cd frontend && npm install

  dev:
  	. .venv/bin/activate && uvicorn backend.app:app --host 0.0.0.0 --port $${API_PORT:-8082} --reload & \
  	cd frontend && npm run dev -- --port $${WEB_PORT:-4002}

  test:
  	. .venv/bin/activate && pytest -q
  ```

- `.env.example`（ルート & `backend/` に用意しても良い）
  ```env
  DB_PATH=/Users/takashiui/Documents/Python/stock_analysis/stock_data.db
  API_PORT=8082
  WEB_PORT=4002
  VITE_API_BASE_URL=http://localhost:8082
  ```

- 起動:
  ```bash
  cp .env.example .env && cp backend/.env.example backend/.env
  make setup
  make dev
  # → http://localhost:4002/?stock=0001&lookback=126&horizon=63&method=ensemble&feature_mode=auto
  ```

---

## 6) 実装上の注意（重要）

- **リーケージ禁止**：特徴量は当日までで算出（未来情報は使わない）。
- **営業日**：DBに存在する日付を**正**とする（欠損日を無理に補完しない）。`horizon_days` は**営業日数**として扱い、日付配列は観測最終日からの**連続営業行**を生成。
- **英数字コード**：SQLは**パラメータ化**して注入を防ぐ。`stock_code` は TEXT。
- **数値列の扱い**：外れ値はウィンズライジング（両側1〜2%）を検討。標準化は**学習データでfit**。
- **パフォーマンス**：`(stock_code, date)` 複合インデックス推奨。
- **既知の特例**： `'0001'`（日経平均）, `'0002'`（TOPIX）も通常コードとして機能することを**例示**（特別扱いは不要）。

---

## 7) 完了時に出力すべき情報

- バックエンド起動URLと `/api/health` 結果
- フロントエンドURL（例：`http://localhost:4002/?stock=0001&lookback=126&horizon=63&method=ensemble&feature_mode=auto`）
- `/api/meta` のレスポンス（検出列・スキーママップ・追加数値列）
- 予測ダイアグ（CV誤差・各モデル重み）
- 既知の制約と次の改善案（例：為替/金利/VIXなど外生導入、分位点の厳密化、より高度な時系列モデル）

---

## 8) 具体的な実装指示（要点）

1. 依存インストール（Python仮想環境・npm）。
2. `db/inspect.py` を最初に実行して**スキーマキャッシュ**を作る → `/api/meta` 実装。
3. `services/prices.py` と `services/features.py` で、スキーママップに基づく**安全な抽出と特徴生成**。
4. `services/forecast.py` に各手法（baseline/direct/multistep/ensemble）と**分位点推定**（残差経験分布/ブートストラップ/QuantileRegressor）。
5. FastAPI ルーティングとCORS、`.env` 読み込み。
6. Vite + React + Tailwind 初期化、ParamsPanelとPriceChart実装。URLクエリとUI状態を同期。
7. `make dev` で同時起動、軽量テスト作成、初回コミット。

---

**注意**：作業は**現ワークスペース内**に限定してください。外部ネットワークアクセス（依存取得等）が必要な場合は都度承認を求めてください。