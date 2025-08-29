from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .db.inspect import ensure_cache
from .schemas.meta import MetaResponse
from .services import prices as prices_service
from .services.forecast import forecast


app = FastAPI(title="JP Stock Forecast API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    # Ensure schema cache exists
    ensure_cache()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta", response_model=MetaResponse)
def get_meta():
    meta = ensure_cache()
    return meta


@app.get("/api/stocks")
def get_stocks(q: str = Query(default="", min_length=0), limit: int = Query(default=20, ge=1, le=100)):
    items = prices_service.suggest_stocks(q, limit)
    return {"items": items}


@app.get("/api/prices")
def get_prices(stock: str, start: str | None = None, end: str | None = None):
    data = prices_service.get_prices(stock=stock, start=start, end=end)
    return data


@app.get("/api/forecast")
def get_forecast(
    stock: str,
    lookback_days: int = 126,
    horizon_days: int = 63,
    method: str = "ensemble",
    feature_mode: str = "auto",
    seed: int = 42,
):
    try:
        res = forecast(
            stock=stock,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            method=method,
            feature_mode=feature_mode,
            seed=seed,
        )
        return {
            "dates": res.dates,
            "observed": res.observed,
            "predicted_path": res.predicted_path,
            "quantiles": res.quantiles,
            "diagnostics": res.diagnostics,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

