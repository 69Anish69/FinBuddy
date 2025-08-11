# backend/routes/goals.py
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import locale, os
from typing import Tuple, Optional, List
import pandas as pd

try:
    locale.setlocale(locale.LC_ALL, "en_CA.UTF-8")
except Exception:
    pass

try:
    from ..state import get_stored_df
except ImportError:
    from state import get_stored_df

router = APIRouter()

class GoalItem(BaseModel):
    name: str
    price: float
    year: int

class GoalPlanRequest(BaseModel):
    goals: List[GoalItem]

def _currency(x: float) -> str:
    try:
        return locale.currency(float(x), grouping=True)
    except Exception:
        return f"${float(x):,.2f}"

def _load_df_with_fallback() -> Optional[pd.DataFrame]:
    df, _ = get_stored_df()
    if df is not None and len(df) > 0:
        return df
    p = os.path.join("uploaded_files", "latest_upload.csv")
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def _to_num_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _pick(work: pd.DataFrame, *names) -> pd.Series:
    cols = {str(c).strip().lower(): c for c in work.columns}
    for n in names:
        key = n.lower()
        if key in cols:
            return work[cols[key]]
    return pd.Series([0.0] * len(work))

def _compute_finance_metrics(df: Optional[pd.DataFrame]) -> Tuple[float, float, float]:
    """
    Returns (current_balance, avg_monthly_surplus, avg_monthly_expense).
    """
    if df is None or len(df) == 0:
        return 0.0, 0.0, 0.0
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    month_raw = _pick(work, "Month", "month", "date", "Date")
    work["Month"] = pd.to_datetime(month_raw, errors="coerce")

    income = _to_num_series(_pick(work, "Income", "income", "credit", "credits", "deposits", "deposit"))
    expense = _to_num_series(_pick(work, "Expense", "expense", "debit", "withdrawals", "withdrawal", "debits"))

    bal_series = _pick(work, "Balance", "balance", "bal")
    if bal_series.any():
        try:
            current_balance = float(str(bal_series.iloc[-1]).replace("$", "").replace(",", ""))
        except Exception:
            current_balance = 0.0
    else:
        current_balance = float((income - expense).cumsum().iloc[-1])

    if work["Month"].notna().any():
        work["__period"] = work["Month"].dt.to_period("M")
        monthly = pd.DataFrame({"__period": work["__period"], "income": income, "expense": expense}) \
            .groupby("__period").sum(numeric_only=True)
        monthly["net"] = monthly["income"] - monthly["expense"]
        monthly_surplus = float(monthly["net"].mean()) if len(monthly) else 0.0
        avg_exp = float(monthly["expense"].mean()) if len(monthly) else 0.0
    else:
        monthly_surplus = 0.0
        avg_exp = 0.0

    return current_balance, monthly_surplus, avg_exp

def _ef_recommendation(avg_exp: float, monthly: pd.Series | None = None) -> Tuple[int, float]:
    """
    Recommend 3..6 months. Simple heuristic on variability if monthly provided.
    """
    months = 3
    if monthly is not None and len(monthly) > 0:
        net = monthly.get("net")
        if net is not None and len(net) > 1:
            vol = float(net.std(ddof=0))
            if vol >= 0.5 * max(avg_exp, 1.0):
                months = 6
            elif vol >= 0.25 * max(avg_exp, 1.0):
                months = 4
            else:
                months = 3
    return months, round(avg_exp * months, 2)

@router.post("/goals/")
async def calculate_goals(request: GoalPlanRequest):
    current_year = datetime.now().year
    df = _load_df_with_fallback()

    # compute core numbers (+ monthly table again for EF heuristic)
    cur_balance, monthly_surplus, avg_exp = _compute_finance_metrics(df)
    monthly_tbl = None
    if df is not None and len(df) > 0:
        w = df.copy()
        w.columns = [str(c).strip() for c in w.columns]
        w["Month"] = pd.to_datetime(_pick(w, "Month", "month", "date", "Date"), errors="coerce")
        inc = _to_num_series(_pick(w, "Income", "income", "credit", "credits", "deposits", "deposit"))
        exp = _to_num_series(_pick(w, "Expense", "expense", "debit", "withdrawals", "withdrawal", "debits"))
        if w["Month"].notna().any():
            w["__p"] = w["Month"].dt.to_period("M")
            monthly_tbl = pd.DataFrame({"__p": w["__p"], "income": inc, "expense": exp}) \
                .groupby("__p").sum(numeric_only=True)
            monthly_tbl["net"] = monthly_tbl["income"] - monthly_tbl["expense"]

    months_ef, ef_target = _ef_recommendation(avg_exp, monthly_tbl)

    # budget accounting: allocate monthly need across goals in request order
    used_monthly = 0.0
    plans: List[dict] = []

    for g in request.goals:
        years_left = max(1, g.year - current_year)
        months_left = years_left * 12
        monthly_required_raw = round(g.price / months_left, 2)

        affordable = (used_monthly + monthly_required_raw) <= monthly_surplus + 1e-9
        if affordable:
            used_monthly += monthly_required_raw

        progress = round((cur_balance / g.price) * 100, 1) if g.price > 0 else 0.0
        feasible = affordable and (monthly_surplus >= monthly_required_raw)
        status = "On Track" if feasible else "Behind"

        plans.append({
            "name": g.name,
            "target": f"{_currency(g.price)} CAD",
            "year": g.year,
            "monthly": _currency(monthly_required_raw),
            "monthly_raw": monthly_required_raw,  # numeric for frontend checks
            "current": _currency(cur_balance),
            "progress": progress,
            "feasible": feasible,
            "status": status,
            "surplus": _currency(monthly_surplus),
            "affordable": affordable,
            "over_by": 0.0 if affordable else round(used_monthly + monthly_required_raw - monthly_surplus, 2),
        })

    budget = {
        "surplus_raw": round(monthly_surplus, 2),
        "used_raw": round(used_monthly, 2),
        "remaining_raw": round(max(0.0, monthly_surplus - used_monthly), 2),
        "surplus": _currency(monthly_surplus),
        "used": _currency(used_monthly),
        "remaining": _currency(max(0.0, monthly_surplus - used_monthly)),
    }

    emergency = {
        "avg_monthly_expense": _currency(avg_exp),
        "months": months_ef,
        "recommended": _currency(ef_target),
        "raw": {"avg_exp": avg_exp, "ef_target": ef_target}
    }

    return {"goals": plans, "budget": budget, "emergency": emergency}
