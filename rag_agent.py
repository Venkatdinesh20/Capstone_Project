"""
CreditAI — RAG Explanation Agent
Uses ChromaDB for vector retrieval + sentence-transformers for local embeddings.
No API keys required. Fully offline.
"""
import os
import csv
import re
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

BASE       = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE, "data_processed", "chroma_kb")

# ── SHAP importance weights for the 13 user-input features (from step 7) ─────
SHAP_WEIGHTS = {
    "mobilephncnt_593L":               0.122,
    "pctinstlsallpaidlate1d_3546856L": 0.108,
    "homephncnt_628L":                 0.103,
    "pmtnum_254L":                     0.086,
    "pmtssum_45A":                     0.075,
    "days90_310L":                     0.073,
    "pmtscount_423L":                  0.072,
    "annuity_780A":                    0.071,
    "WEEK_NUM":                        0.148,
    "MONTH":                           0.067,
    "thirdquarter_1082L":              0.066,
    "eir_270L":                        0.065,
    "totalsettled_863A":               0.059,
}

# ── Feature metadata: display names, units, risk thresholds ──────────────────
FEATURE_META = {
    "pctinstlsallpaidlate1d_3546856L": {
        "display":        "Late Payment Rate",
        "unit":           "pct",
        "risk_fn":        lambda v: "CRITICAL" if v > 0.50 else ("HIGH" if v > 0.15 else "LOW"),
        "threshold_desc": "Industry benchmark <15%. Values above 50% are a strong default predictor.",
        "search_query":   "percentage installments paid late credit risk",
    },
    "days90_310L": {
        "display":        "90-Day Delinquency Count",
        "unit":           "count",
        "risk_fn":        lambda v: "CRITICAL" if v > 10 else ("HIGH" if v > 0 else "LOW"),
        "threshold_desc": "Any value >0 is a significant negative signal.",
        "search_query":   "days past due 90 day delinquency default",
    },
    "eir_270L": {
        "display":        "Effective Interest Rate",
        "unit":           "pct_raw",
        "risk_fn":        lambda v: "HIGH" if v > 0.25 else ("MEDIUM" if v > 0.15 else "LOW"),
        "threshold_desc": "Normal range 8–20%. Rates above 25% correlate with higher default.",
        "search_query":   "effective interest rate annual credit contract",
    },
    "annuity_780A": {
        "display":        "Monthly Annuity",
        "unit":           "dollar",
        "risk_fn":        lambda v: "HIGH" if v > 30_000 else ("MEDIUM" if v > 15_000 else "LOW"),
        "threshold_desc": "High monthly obligation relative to income increases repayment burden.",
        "search_query":   "monthly annuity installment payment amount",
    },
    "pmtnum_254L": {
        "display":        "Scheduled Payment Count",
        "unit":           "count",
        "risk_fn":        lambda v: "MEDIUM" if v < 12 else "LOW",
        "threshold_desc": "More scheduled payments indicates structured, long-term credit engagement.",
        "search_query":   "number of payments loan schedule",
    },
    "pmtscount_423L": {
        "display":        "Historical Payment Transactions",
        "unit":           "count",
        "risk_fn":        lambda v: "HIGH" if v < 20 else ("MEDIUM" if v < 60 else "LOW"),
        "threshold_desc": "Higher count signals longer credit history and more data for scoring.",
        "search_query":   "payment count history transactions credit",
    },
    "pmtssum_45A": {
        "display":        "Total Payment Volume",
        "unit":           "dollar",
        "risk_fn":        lambda v: "HIGH" if v < 10_000 else ("MEDIUM" if v < 50_000 else "LOW"),
        "threshold_desc": "Larger cumulative payment volume indicates established credit use.",
        "search_query":   "total payments sum credit history amount",
    },
    "totalsettled_863A": {
        "display":        "Total Settled Amount",
        "unit":           "dollar",
        "risk_fn":        lambda v: "HIGH" if v < 10_000 else ("MEDIUM" if v < 80_000 else "LOW"),
        "threshold_desc": "Higher settlement history reduces default probability.",
        "search_query":   "total settled amount previous loans closed",
    },
    "mobilephncnt_593L": {
        "display":        "Mobile Phone Contacts",
        "unit":           "count",
        "risk_fn":        lambda v: "HIGH" if v == 0 else "LOW",
        "threshold_desc": "Phone verification is a key fraud and identity signal.",
        "search_query":   "mobile phone count contact verification",
    },
    "homephncnt_628L": {
        "display":        "Home Phone Contacts",
        "unit":           "count",
        "risk_fn":        lambda v: "MEDIUM" if v == 0 else "LOW",
        "threshold_desc": "Home contact availability supports identity verification.",
        "search_query":   "home phone count contact addresses",
    },
    "WEEK_NUM": {
        "display":        "Application Week of Year",
        "unit":           "week",
        "risk_fn":        lambda v: "LOW",
        "threshold_desc": "Temporal feature capturing weekly application patterns.",
        "search_query":   "week number application time stability",
    },
    "MONTH": {
        "display":        "Application Month",
        "unit":           "month",
        "risk_fn":        lambda v: "LOW",
        "threshold_desc": "Monthly seasonality affects credit demand and approval patterns.",
        "search_query":   "month application seasonal credit",
    },
    "thirdquarter_1082L": {
        "display":        "Q3 Application Flag",
        "unit":           "flag",
        "risk_fn":        lambda v: "LOW",
        "threshold_desc": "Q3 (Jul–Sep) shows distinct default rate patterns historically.",
        "search_query":   "third quarter application flag seasonal",
    },
}

# Maps API input field names → model feature names
INPUT_TO_FEATURE = {
    "late_pct":       "pctinstlsallpaidlate1d_3546856L",
    "days90":         "days90_310L",
    "eir":            "eir_270L",
    "annuity":        "annuity_780A",
    "num_payments":   "pmtnum_254L",
    "payments_count": "pmtscount_423L",
    "payments_sum":   "pmtssum_45A",
    "total_settled":  "totalsettled_863A",
    "mobile_phones":  "mobilephncnt_593L",
    "home_phones":    "homephncnt_628L",
    "week_num":       "WEEK_NUM",
    "month":          "MONTH",
    "is_q3":          "thirdquarter_1082L",
}

RISK_SCORE = {"LOW": 0.0, "MEDIUM": 0.4, "HIGH": 0.8, "CRITICAL": 1.0}


# ── Custom embedding function (works with any sentence-transformers version) ──
class _LocalEmbedder:
    """Wraps sentence-transformers into ChromaDB's EmbeddingFunction interface."""
    name = "local-minilm"   # Required by chromadb 1.x

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:  # ChromaDB interface
        vecs = self._model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        return vecs.tolist()


def _get_embedder():
    """Return the best available local embedding function for chromadb 1.x."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    return ONNXMiniLM_L6_V2()


# ── RAG Agent ────────────────────────────────────────────────────────────────
class RAGAgent:
    def __init__(self):
        self._collection = None
        self._ready      = False

    # ── Build knowledge base ─────────────────────────────────────────────────
    def build(self):
        try:
            import chromadb

            os.makedirs(CHROMA_DIR, exist_ok=True)
            ef     = _get_embedder()
            client = chromadb.PersistentClient(path=CHROMA_DIR)

            existing = [c.name for c in client.list_collections()]
            if "credit_kb" in existing:
                self._collection = client.get_collection(
                    "credit_kb", embedding_function=ef
                )
                self._ready = True
                print(f"[RAG] Loaded KB: {self._collection.count()} docs")
                return

            self._collection = client.create_collection(
                "credit_kb", embedding_function=ef
            )

            docs, ids, metas = [], [], []

            # 1. Feature definitions CSV
            feat_csv = os.path.join(BASE, "feature_definitions.csv")
            if os.path.exists(feat_csv):
                with open(feat_csv, newline="", encoding="utf-8") as f:
                    for i, row in enumerate(csv.DictReader(f)):
                        var  = row.get("Variable", "").strip()
                        desc = row.get("Description", "").strip()
                        if var and desc:
                            docs.append(f"Feature '{var}': {desc}")
                            ids.append(f"f{i}")
                            metas.append({"type": "feature", "variable": var})

            # 2. Key doc sections (technical plan, epic definition, data dict)
            for excerpt, meta in _doc_excerpts():
                i = len(docs)
                docs.append(excerpt)
                ids.append(f"d{i}")
                metas.append(meta)

            # Batch-insert to avoid OOM
            BATCH = 64
            for start in range(0, len(docs), BATCH):
                self._collection.add(
                    documents=docs[start:start + BATCH],
                    ids=ids[start:start + BATCH],
                    metadatas=metas[start:start + BATCH],
                )

            self._ready = True
            print(f"[RAG] Built KB: {self._collection.count()} docs indexed")

        except Exception as exc:
            print(f"[RAG] WARNING — KB build failed: {exc}")
            self._ready = False

    # ── Retrieve context ─────────────────────────────────────────────────────
    def retrieve(self, query: str, n: int = 3) -> list[str]:
        if not self._ready or not self._collection:
            return []
        try:
            cap = min(n, self._collection.count())
            if cap == 0:
                return []
            res = self._collection.query(query_texts=[query], n_results=cap)
            return res["documents"][0]
        except Exception:
            return []

    # ── Explain a prediction ─────────────────────────────────────────────────
    def explain(self, inp_dict: dict, probability: float) -> dict:
        prob_pct = round(probability * 100, 1)

        # Map inputs to model features and compute risk signals
        feature_values = _map_inputs(inp_dict)
        ranked = _rank_features(feature_values)

        # Top drivers = features with HIGH/CRITICAL risk, else top 3 by score
        key_drivers = [r for r in ranked if r["risk"] in ("HIGH", "CRITICAL")][:4]
        if not key_drivers:
            key_drivers = ranked[:3]

        # RAG: retrieve definitions for top 3 drivers
        retrieved_docs = []
        for d in key_drivers[:3]:
            hits = self.retrieve(d["meta"]["search_query"], n=2)
            retrieved_docs.extend(hits)

        # Determine tier
        if prob_pct >= 60:
            risk_level = "HIGH RISK"
        elif prob_pct >= 30:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK"

        narrative = _narrative(prob_pct, risk_level, key_drivers, retrieved_docs)

        return {
            "probability_pct":  prob_pct,
            "risk_level":       risk_level,
            "key_drivers": [
                {
                    "feature":   d["meta"]["display"],
                    "raw_value": d["raw"],
                    "value":     _fmt(d["raw"], d["meta"]["unit"]),
                    "risk":      d["risk"],
                    "insight":   d["meta"]["threshold_desc"],
                }
                for d in key_drivers
            ],
            "rag_docs_used":    len(retrieved_docs),
            "rag_snippets":     retrieved_docs[:2],   # first 2 for transparency
            "narrative":        narrative,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _map_inputs(inp_dict: dict) -> dict:
    """Convert API input dict → {feat_key: normalized_value}."""
    out = {}
    for api_key, feat_key in INPUT_TO_FEATURE.items():
        val = inp_dict.get(api_key, 0) or 0
        if api_key == "late_pct":
            val = float(val) / 100.0
        elif api_key == "eir":
            val = float(val) / 100.0
        elif api_key == "is_q3":
            val = 1.0 if val else 0.0
        else:
            val = float(val)
        out[feat_key] = val
    return out


def _rank_features(fv: dict) -> list[dict]:
    rows = []
    for feat_key, meta in FEATURE_META.items():
        val  = fv.get(feat_key, 0.0)
        risk = meta["risk_fn"](val)
        w    = SHAP_WEIGHTS.get(feat_key, 0.05)
        rows.append({
            "feat_key": feat_key,
            "raw":      val,
            "risk":     risk,
            "score":    RISK_SCORE[risk] * w,
            "meta":     meta,
        })
    rows.sort(key=lambda x: -x["score"])
    return rows


def _fmt(val: float, unit: str) -> str:
    if unit == "pct":
        return f"{round(val * 100, 1)}%"
    if unit == "pct_raw":
        return f"{round(val * 100, 1)}%"
    if unit == "dollar":
        return f"${val:,.0f}"
    if unit == "flag":
        return "Yes" if val else "No"
    if unit in ("count", "week", "month"):
        return str(int(val))
    return str(round(val, 2))


def _narrative(prob_pct: float, risk_level: str, drivers: list, rag_docs: list) -> str:
    lines = [f"Assessment: {risk_level} — {prob_pct}% default probability.\n"]

    if drivers:
        lines.append("Key Risk Drivers Identified:")
        for d in drivers:
            prefix = {
                "CRITICAL": "[CRITICAL]",
                "HIGH":     "[HIGH]    ",
                "MEDIUM":   "[MODERATE]",
                "LOW":      "[LOW]     ",
            }.get(d["risk"], "")
            lines.append(
                f"  {prefix} {d['meta']['display']}: {_fmt(d['raw'], d['meta']['unit'])}  —  {d['meta']['threshold_desc']}"
            )

    lines.append("")
    if prob_pct >= 60:
        lines.append(
            "Recommendation: Application presents significant default risk. "
            "Recommend DECLINE or require substantial collateral and co-signer. "
            "Consider a smaller secured loan as an alternative pathway."
        )
    elif prob_pct >= 30:
        lines.append(
            "Recommendation: Application carries moderate risk. "
            "Consider conditional approval: higher interest rate, reduced credit limit, "
            "or mandatory payment monitoring for the first 6 months."
        )
    else:
        lines.append(
            "Recommendation: Application shows a strong credit profile. "
            "Approve under standard terms. Routine 3-month payment monitoring is advisable."
        )

    if rag_docs:
        lines.append(
            f"\nAnalysis cross-referenced with {len(rag_docs)} definitions "
            f"retrieved from the credit feature knowledge base."
        )

    return "\n".join(lines)


def _doc_excerpts() -> list[tuple[str, dict]]:
    """Split key markdown docs into paragraphs for RAG indexing."""
    excerpts = []
    targets  = ["technical_plan.md", "epic_definition.md", "data_dictionary.md"]
    doc_dir  = os.path.join(BASE, "docs")
    for fname in targets:
        fpath = os.path.join(doc_dir, fname)
        if not os.path.exists(fpath):
            continue
        text = open(fpath, encoding="utf-8").read()
        # Split on markdown headers, keep non-trivial sections
        for i, section in enumerate(re.split(r"\n#{1,3} ", text)[:25]):
            section = section.strip()
            if len(section) > 120:
                excerpts.append((
                    section[:500],
                    {"type": "doc", "source": fname, "section": i},
                ))
    return excerpts


# ── Singleton ─────────────────────────────────────────────────────────────────
_AGENT: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = RAGAgent()
        _AGENT.build()
    return _AGENT
