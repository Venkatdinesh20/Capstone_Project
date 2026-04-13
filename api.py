"""
CreditAI — FastAPI Backend
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

# ── App ───────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up RAG agent in background on startup (non-blocking)
    import threading
    from rag_agent import get_agent
    threading.Thread(target=get_agent, daemon=True).start()
    yield

app = FastAPI(
    title="CreditAI — Credit Risk Prediction API",
    description="Home Credit Default Prediction powered by LightGBM. AUC-ROC 0.803.",
    version="2.0.0",
    lifespan=lifespan,
)

def _load():
    model      = joblib.load(os.path.join(BASE, "models", "lightgbm.pkl"))
    scaler     = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
    num_cols   = joblib.load(os.path.join(BASE, "models", "numerical_cols.pkl"))
    feat_names = model.feature_name()
    scaler_cols = (
        list(scaler.feature_names_in_)
        if hasattr(scaler, "feature_names_in_") else list(num_cols)
    )
    return model, scaler, list(num_cols), feat_names, scaler_cols

_MODEL, _SCALER, _NUM_COLS, _FEAT_NAMES, _SCALER_COLS = _load()

# ── Request / Response schemas ────────────────────────────────────────────────
class ApplicantInput(BaseModel):
    week_num:       int   = Field(26,        ge=1,   le=52,        description="Week of year (1–52)")
    month:          int   = Field(6,         ge=1,   le=12,        description="Application month (1–12)")
    is_q3:          bool  = Field(False,                           description="Application in Q3 (Jul–Sep)")
    mobile_phones:  int   = Field(1,         ge=0,   le=10,        description="Mobile phones on file")
    home_phones:    int   = Field(1,         ge=0,   le=10,        description="Home phones on file")
    annuity:        float = Field(8500.0,    ge=0,   le=500_000,   description="Monthly annuity / instalment ($)")
    num_payments:   int   = Field(36,        ge=0,   le=360,       description="Number of scheduled payments")
    payments_count: int   = Field(120,       ge=0,   le=5000,      description="Historical payment count")
    payments_sum:   float = Field(250_000.0, ge=0,   le=10_000_000,description="Total historical payments ($)")
    late_pct:       float = Field(5.0,       ge=0.0, le=100.0,     description="% of instalments paid late (>1 day)")
    days90:         int   = Field(0,         ge=0,   le=1000,      description="Days past due (90-day bucket)")
    total_settled:  float = Field(200_000.0, ge=0,   le=10_000_000,description="Total amount settled ($)")
    eir:            float = Field(12.0,      ge=0.0, le=100.0,     description="Effective interest rate (%)")

class PredictionResponse(BaseModel):
    probability:    float
    risk_tier:      str
    risk_color:     str
    advice:         str
    decisions: dict[str, str]
    risk_factors:   list[dict]

class ExplainResponse(BaseModel):
    probability_pct: float
    risk_level:      str
    key_drivers:     list[dict]
    rag_docs_used:   int
    rag_snippets:    list[str]
    narrative:       str

# ── Feature vector builder ────────────────────────────────────────────────────
def _build_features(inp: ApplicantInput) -> pd.DataFrame:
    row = dict.fromkeys(_FEAT_NAMES, 0.0)
    row["WEEK_NUM"]                        = float(inp.week_num)
    row["MONTH"]                           = float(inp.month)
    row["mobilephncnt_593L"]               = float(inp.mobile_phones)
    row["homephncnt_628L"]                 = float(inp.home_phones)
    row["annuity_780A"]                    = float(inp.annuity)
    row["pmtnum_254L"]                     = float(inp.num_payments)
    row["pmtscount_423L"]                  = float(inp.payments_count)
    row["pmtssum_45A"]                     = float(inp.payments_sum)
    row["pctinstlsallpaidlate1d_3546856L"] = inp.late_pct / 100.0
    row["days90_310L"]                     = float(inp.days90)
    row["totalsettled_863A"]               = float(inp.total_settled)
    row["eir_270L"]                        = inp.eir / 100.0
    row["thirdquarter_1082L"]              = 1.0 if inp.is_q3 else 0.0

    df = pd.DataFrame([row])[_FEAT_NAMES]
    cols_to_scale = [c for c in _NUM_COLS if c in df.columns]
    if cols_to_scale:
        scale_df = pd.DataFrame(np.zeros((1, len(_SCALER_COLS))), columns=_SCALER_COLS)
        for c in cols_to_scale:
            if c in scale_df.columns:
                scale_df[c] = df[c].values
        scaled_df = pd.DataFrame(_SCALER.transform(scale_df), columns=_SCALER_COLS)
        for c in cols_to_scale:
            if c in scaled_df.columns:
                df[c] = scaled_df[c].values
    return df

# ── Risk factor breakdown ─────────────────────────────────────────────────────
def _risk_factors(inp: ApplicantInput, prob: float) -> list[dict]:
    weights = {
        "Late Payment Rate":     (inp.late_pct / 100,                    0.28),
        "90-Day Past Due":       (min(inp.days90 / 200, 1.0),            0.20),
        "Interest Rate":         (min(inp.eir / 80, 1.0),                0.16),
        "Monthly Annuity":       (min(inp.annuity / 50_000, 1.0),        0.10),
        "Settlement History":    (1 - min(inp.total_settled / 1e6, 1.0), 0.08),
        "Payment History":       (1 - min(inp.payments_count / 500, 1.0),0.07),
        "Scheduled Payments":    (1 - min(inp.num_payments / 240, 1.0),  0.05),
        "Phone Contacts":        (1 - min((inp.mobile_phones + inp.home_phones) / 8, 1.0), 0.04),
    }
    raw   = {k: n * w for k, (n, w) in weights.items()}
    total = sum(raw.values()) or 1
    return [
        {
            "label":  k,
            "value":  round(v / total * prob * 100, 2),
            "color":  "#EF4444" if v / total * prob * 100 > 5
                      else "#F59E0B" if v / total * prob * 100 > 2
                      else "#22C55E",
        }
        for k, v in sorted(raw.items(), key=lambda x: -x[1])
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model":  "LightGBM",
        "auc_roc": 0.803,
        "features": len(_FEAT_NAMES),
    }

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(inp: ApplicantInput):
    try:
        X    = _build_features(inp)
        prob = float(_MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    if prob < 0.30:
        tier, color = "LOW RISK",    "#22C55E"
        advice = "Applicant profile is within acceptable risk parameters. Safe to proceed."
    elif prob < 0.60:
        tier, color = "MEDIUM RISK", "#F59E0B"
        advice = "Elevated risk detected. Manual review and additional documentation recommended."
    else:
        tier, color = "HIGH RISK",   "#EF4444"
        advice = "High default probability. Enhanced due diligence and senior review required."

    thresholds = {
        "Conservative (0.30)": "DECLINE" if prob >= 0.30 else "APPROVE",
        "Balanced (0.45)":     "DECLINE" if prob >= 0.45 else "APPROVE",
        "Cost-Optimal (0.60)": "DECLINE" if prob >= 0.60 else "APPROVE",
        "High Precision (0.75)":"DECLINE" if prob >= 0.75 else "APPROVE",
    }

    return PredictionResponse(
        probability   = round(prob * 100, 2),
        risk_tier     = tier,
        risk_color    = color,
        advice        = advice,
        decisions     = thresholds,
        risk_factors  = _risk_factors(inp, prob),
    )

@app.post("/api/explain", response_model=ExplainResponse, tags=["AI Analysis"])
def explain(inp: ApplicantInput):
    """
    RAG-powered explanation: retrieves feature definitions from the vector
    knowledge base and generates a natural-language credit risk narrative.
    """
    try:
        X    = _build_features(inp)
        prob = float(_MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    try:
        from rag_agent import get_agent
        agent  = get_agent()
        result = agent.explain(inp.model_dump(), prob)
        return ExplainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {e}")

# ── Serve static frontend ─────────────────────────────────────────────────────
_STATIC = os.path.join(BASE, "static")
os.makedirs(_STATIC, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(_STATIC, "index.html"))
