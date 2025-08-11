# backend/routes/report.py
from __future__ import annotations

import os, tempfile, time
from datetime import datetime
from typing import Any, Dict, List, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.pdfgen import canvas  # pip install reportlab

import modules.data_store as data_store  # your in-memory store

router = APIRouter(prefix="/report", tags=["Report"])

# ---------- Pydantic models ----------
StatusVal = Literal["ready", "processing", "error"]

class Status(BaseModel):
    forecast: StatusVal
    goals: StatusVal
    loans: StatusVal

class Insights(BaseModel):
    ef_months: float
    apr_save_12m: float
    invest_uplift_pct: float
    goal_buffer_months: int

class OverviewResponse(BaseModel):
    updated_at: datetime
    status: Status
    score: int
    insights: Insights
    improvements: List[str]
    next_move: str

class GenerateResponse(BaseModel):
    report_id: str
    download: str

# ---------- Core helpers ----------
def _get_clean_df() -> pd.DataFrame:
    # Try multiple in-memory keys (KV + legacy attrs)
    candidates = (
        "stored_df", "uploaded_df", "transactions_df",  # KV preferred
        "report_df", "forecast_df", "memory_df",        # possible others
        "stored_df", "stored_df_loans",                 # legacy attrs
    )
    df = None
    for key in candidates:
        val = None
        try:
            if hasattr(data_store, "get"):
                val = data_store.get(key)
        except Exception:
            val = None
        if val is None:
            val = getattr(data_store, key, None)
        if isinstance(val, pd.DataFrame) and not val.empty:
            df = val.copy()
            break

    if df is None:
        raise HTTPException(400, "No uploaded data found. Please upload a statement first.")

    # Normalize to Month / Income / Expense
    colmap = {
        "Month":"Month", "date":"Month", "posted date":"Month", "transaction date":"Month",
        "Income":"Income", "credit":"Income", "credit ($)":"Income", "deposit":"Income", "inflow":"Income",
        "Expense":"Expense", "debit":"Expense", "debit ($)":"Expense", "withdrawal":"Expense", "outflow":"Expense",
    }
    have = {c.lower(): c for c in df.columns}
    df = df.rename(columns={have[k]: v for k, v in colmap.items() if k in have})

    if "Month" not in df.columns:
        raise HTTPException(500, "Unable to find date column (Month/date).")
    for k in ("Income", "Expense"):
        if k not in df.columns:
            df[k] = 0.0

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.dropna(subset=["Month"])
    if df.empty:
        raise HTTPException(500, "No valid dated rows after cleaning.")
    return df

def _avg_monthly_expense(df: pd.DataFrame) -> float:
    m = df.resample("ME", on="Month").sum(numeric_only=True)
    last6 = m.tail(6)
    avg = float(last6["Expense"].mean()) if "Expense" in last6 else 0.0
    return max(avg, 1.0)  # avoid divide-by-zero

def _compute_score(ef_months: float, utilization: float) -> int:
    # 50 base + 30 (EF up to 6m) + 20 (low utilization)
    score = 50 + 30 * min(ef_months / 6.0, 1.0) + 20 * max(0.0, 1.0 - utilization)
    return int(round(max(0, min(100, score))))

def _metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # last 3 months view
    m = df.resample("ME", on="Month").sum(numeric_only=True)
    last3 = m.tail(3)
    income_avg  = float(last3.get("Income", pd.Series(dtype=float)).mean() or 0.0)
    expense_avg = float(last3.get("Expense", pd.Series(dtype=float)).mean() or 0.0)
    net_avg     = income_avg - expense_avg

    # EF + credit knobs (fallback to sensible defaults if not set)
    ef_balance  = float(data_store.get("emergency_fund_balance", 0.0))
    cc_balance  = float(data_store.get("cc_balance", 0.0))
    cc_apr      = float(data_store.get("cc_apr", 0.22))
    cc_limit    = float(data_store.get("cc_limit", 0.0))
    utilization = min(1.0, cc_balance / cc_limit) if cc_limit > 0 else float(data_store.get("cc_utilization", 0.0))

    # avg monthly expense (6 months window) for EF months calc
    avg_exp_6m  = float(m.tail(6).get("Expense", pd.Series(dtype=float)).mean() or 1.0)
    ef_months   = round(ef_balance / max(avg_exp_6m, 1.0), 1)
    apr_save_12m = round(cc_balance * cc_apr, 2)  # rough avoided interest if paid off in 12m
    invest_uplift = 2.3  # placeholder; keep as-is or wire to your optimizer

    # top spending categories (last ~90 days)
    cat_col = next((c for c in df.columns if c.lower() == "category"), None)
    top_cats: List[Dict[str, Any]] = []
    if cat_col:
        since = (df["Month"].max() - pd.DateOffset(months=3)) if not pd.isna(df["Month"].max()) else None
        dfx = df if since is None else df[df["Month"] >= since]
        g = dfx.groupby(cat_col, dropna=False)["Expense"].sum().sort_values(ascending=False).head(3)
        top_cats = [{"category": str(k if k==k else "Other"), "amount": float(v)} for k, v in g.items()]

    return dict(
        income_avg=income_avg, expense_avg=expense_avg, net_avg=net_avg,
        ef_balance=ef_balance, ef_months=ef_months,
        apr_save_12m=apr_save_12m, invest_uplift=invest_uplift,
        utilization=utilization, top_cats=top_cats
    )

def _build_improvements(mx: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    # Themes (broad areas to improve) — no amounts here
    if mx["ef_months"] < 3:
        out.append("Build emergency fund to 3–6 months")
    elif mx["ef_months"] < 6:
        out.append("Top up emergency fund to 6 months")
    if mx["net_avg"] < 0:
        out.append("Reduce monthly shortfall with targeted spending cuts")
    if mx["utilization"] > 0.3:
        out.append("Lower credit card utilization below 30%")
    if mx["top_cats"]:
        top = mx["top_cats"][0]
        out.append(f"Rein in {top['category']} spending (largest category)")
    # Always cap it to 3–4 items
    return out[:4] or ["Tighten variable expenses and automate savings"]

def _build_recommendations(mx: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    # Concrete actions — include amounts / cadence
    if mx["ef_months"] < 6:
        # target: reach 6 months in ~6 months if possible
        # need = (target months - current) * avg expense estimate; fall back to expense_avg
        target_buf = max(6.0 - mx["ef_months"], 0) * max(mx["expense_avg"], 1.0)
        monthly = round(target_buf / 6.0, 0) if target_buf > 0 else 0
        if monthly > 0:
            out.append(f"Auto-transfer ~${monthly:,.0f}/mo to emergency fund until you hit 6 months")
    if mx["utilization"] > 0.3 and mx["apr_save_12m"] > 0:
        extra = max(round(mx["apr_save_12m"] / 12.0 / 5.0, 0), 50)  # rough extra payment
        out.append(f"Add ~${extra:,.0f}/mo to high-APR card (est. interest avoided ${mx['apr_save_12m']:,.0f}/yr)")
    if mx["net_avg"] > 50:
        invest_amt = round(mx["net_avg"] * 0.7, 0)
        out.append(f"Invest ~${invest_amt:,.0f}/mo into your target allocation (after EF/CC priorities)")
    if mx["top_cats"]:
        top = mx["top_cats"][0]
        cap = max(round(top["amount"] / 3.0, 0), 50)  # cap at ~1/3 of recent 3-mo total
        out.append(f"Cap {top['category']} to ~${cap:,.0f}/mo; review merchants and cancel unused subs")

    # keep 3–4 crisp items
    return out[:4] or ["Set a small weekly spending cap and review after 30 days"]

def _pick_next_move(mx: Dict[str, Any]) -> str:
    if mx["net_avg"] < 0 and mx["top_cats"]:
        need = round(abs(mx["net_avg"]), 0)
        return f"Trim {mx['top_cats'][0]['category']} by ~${need:,.0f}/mo to break even"
    if mx["ef_months"] < 3:
        return "Set an automatic transfer to build your emergency fund"
    if mx["utilization"] > 0.3:
        return "Schedule an extra payment toward your high-APR card this month"
    if mx["net_avg"] > 50:
        return "Automate monthly investing toward your target allocation"
    return "Review last 90 days for one-time cuts and set weekly limits"


# ---------- Section helpers ----------
def _monthly_trend(df: pd.DataFrame) -> List[Dict[str, Any]]:
    m = df.resample("ME", on="Month").sum(numeric_only=True).reset_index()
    m["Month"] = m["Month"].dt.strftime("%Y-%m")
    if "Income" not in m: m["Income"] = 0.0
    if "Expense" not in m: m["Expense"] = 0.0
    return m[["Month", "Income", "Expense"]].to_dict(orient="records")

def _category_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cat_col = next((c for c in df.columns if c.lower() == "category"), None)
    if not cat_col:
        total = float(df["Expense"].sum()) if "Expense" in df.columns else 0.0
        return [{"category": "Uncategorized", "amount": round(total, 2)}]
    g = df.groupby(cat_col, dropna=False)["Expense"].sum().reset_index()
    g = g.sort_values("Expense", ascending=False).head(8)
    return [{"category": str(r[cat_col] or "Other"), "amount": float(r["Expense"])} for _, r in g.iterrows()]

def _exec_summary(ef_months: float, apr_save_12m: float, invest_uplift: float, score: int) -> str:
    tone = "strong" if ef_months >= 6 else "improving"
    return (
        f"Overall financial health is {tone} with a score of {score}. "
        f"Emergency fund covers ~{ef_months:.1f} months. "
        f"Paying high-APR debt first could avoid about ${apr_save_12m:,.0f} interest over 12 months. "
        f"A modest portfolio optimization may improve expected returns by ~{invest_uplift:.1f}%."
    )

# ---------- Routes ----------
@router.get("/overview", response_model=OverviewResponse)
def overview() -> OverviewResponse:
    df = _get_clean_df()
    mx = _metrics(df)

    score = _compute_score(mx["ef_months"], mx["utilization"])

    payload = OverviewResponse(
        updated_at=datetime.utcnow(),
        status=Status(forecast="ready", goals="ready", loans="ready"),
        score=score,
        insights=Insights(
            ef_months=mx["ef_months"],
            apr_save_12m=mx["apr_save_12m"],
            invest_uplift_pct=mx["invest_uplift"],
            goal_buffer_months=int(data_store.get("goal_buffer_months", 2)),
        ),
        improvements=_build_improvements(mx),
        next_move=_pick_next_move(mx),
    )
    data_store.set("latest_report", payload.model_dump())
    return payload


@router.get("/sections")
def sections():
    df = _get_clean_df()
    mx = _metrics(df)
    score = _compute_score(mx["ef_months"], mx["utilization"])

    payload = {
        "executive_summary": _exec_summary(mx["ef_months"], mx["apr_save_12m"], mx["invest_uplift"], score),
        "data_methods": {
            "sources": ["Uploaded statements (PDF/CSV/Excel)", "OCR (PDF)"],
            "processing": ["Date normalization", "Income/Expense split", "Monthly resampling"],
            "modeling": ["Prophet trend (forecast)", "Credit payoff simulation"],
        },
        "insights": [
            {"title": "Emergency Fund", "value": f"{mx['ef_months']:.1f} months"},
            {"title": "High-APR Interest Avoided (12m)", "value": f"${mx['apr_save_12m']:,.0f}"},
            {"title": "Investment Opportunity", "value": f"+{mx['invest_uplift']:.1f}% expected return"},
            {"title": "Financial Health Score", "value": f"{score}/100"},
        ],
        "goals": data_store.get("goals") or [],
        "recommendations": _build_recommendations(mx),
        "charts": {"trend": _monthly_trend(df), "categories": _category_breakdown(df)},
        "updated_at": datetime.utcnow().isoformat(),
    }
    data_store.set("latest_sections", payload)
    return payload

@router.post("/generate", response_model=GenerateResponse)
def generate() -> GenerateResponse:
    rpt: Dict = data_store.get("latest_report") or overview().model_dump()
    sec: Dict = data_store.get("latest_sections") or sections()

    rid = f"{int(time.time())}"
    path = os.path.join(tempfile.gettempdir(), f"finbuddy_{rid}.pdf")

    c = canvas.Canvas(path)
    c.setTitle("FinBuddy Report")

    def page_header(title: str, y_start: int = 800):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_start, title)
        c.setFont("Helvetica", 11)
        return y_start - 24

    # Cover
    y = page_header("FinBuddy – Personalized Report")
    c.drawString(50, y, f"Updated: {rpt['updated_at']}")
    y -= 24
    c.drawString(50, y, f"Financial Health Score: {rpt['score']}")
    c.showPage()

    # Executive Summary
    y = page_header("Executive Summary")
    for line in _wrap(sec["executive_summary"], 90):
        c.drawString(50, y, line); y -= 16
    c.showPage()

    # Data & Methods
    y = page_header("Data & Methods")
    for key in ("sources", "processing", "modeling"):
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, key.capitalize()); y -= 18
        c.setFont("Helvetica", 11)
        for item in sec["data_methods"][key]:
            c.drawString(65, y, f"• {item}"); y -= 16
    c.showPage()

    # Key Insights
    y = page_header("Key Insights")
    for ins in sec["insights"]:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, ins["title"]); y -= 16
        c.setFont("Helvetica", 11); c.drawString(65, y, f"{ins['value']}"); y -= 18
        if y < 120: c.showPage(); y = page_header("Key Insights")
    c.showPage()

    # Goals
    y = page_header("Goals & Feasibility")
    goals = sec.get("goals") or []
    if not goals:
        c.drawString(50, y, "No goals captured yet."); y -= 18
    else:
        for g in goals:
            c.drawString(50, y, f"• {g.get('name')} — target {g.get('target')} by {g.get('year')} (monthly {g.get('monthly')})"); y -= 16
            if y < 120: c.showPage(); y = page_header("Goals & Feasibility")
    c.showPage()

    # Recommendations
    y = page_header("Recommendations – Next 30 Days")
    for r in sec["recommendations"]:
        for line in _wrap(r, 90):
            c.drawString(50, y, f"• {line}"); y -= 16
            if y < 120: c.showPage(); y = page_header("Recommendations – Next 30 Days")
    c.showPage(); c.save()

    reports = data_store.get("reports_index") or {}
    reports[rid] = path
    data_store.set("reports_index", reports)

    return GenerateResponse(report_id=rid, download=f"/report/download/{rid}.pdf")

def _wrap(text: str, width: int) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width)

@router.get("/download/{rid}.pdf")
def download(rid: str):
    reports = data_store.get("reports_index") or {}
    path = reports.get(rid)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="application/pdf", filename="FinBuddy_Report.pdf")
