# backend/routes/loans.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os, json, re
import numpy as np
import pandas as pd

router = APIRouter(prefix="/loans", tags=["loans"])

# ---------- simple JSON persistence ----------
DB_PATH = "db/loans.json"
os.makedirs("db", exist_ok=True)

try:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        LOAN_DB: List[Dict[str, Any]] = json.load(f)
except Exception:
    LOAN_DB = []

def _save_db():
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(LOAN_DB, f, indent=2)

# ---------- helpers for detection ----------
LENDER_REGEXES = [
    r"AMEX|AMERICAN EXPRESS|CHASE|CITI|CAPITAL\s?ONE|MASTERCARD|VISA",
    r"RBC|TD|BMO|SCOTIABANK|CIBC",
    r"NAVIENT|NELNET|SALLIE MAE|MOHELA",
    r"ALLY|TOYOTA\s?FINANC(?:E|IAL)|HONDA\s?FINANCE|FORD\s?CREDIT",
    r"SOFI|LIGHTSTREAM|UPSTART|LENDINGCLUB",
    r"MORTGAGE|HELOC",
    r"PAYPAL\s*CREDIT|AFFIRM|KLARNA|AFTERPAY",
    r"LOAN\s*PYMT|EMI|E\.M\.I\."
]
RGX = re.compile("|".join(LENDER_REGEXES), re.I)

def _clean(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", (s or "")).upper()
    return re.sub(r"\s+", " ", s).strip()

def _looks_monthly(dates: pd.Series) -> bool:
    d = sorted(pd.to_datetime(dates).unique())
    if len(d) < 3: return False
    gaps = np.diff([x.toordinal() for x in d])
    med = float(np.median(gaps))
    return 27 <= med <= 33

def _get_stored_df():
    """DataFrame from the Upload step for loans detection."""
    try:
        from modules import data_store
        return getattr(data_store, "stored_df_loans", None)
    except Exception:
        return None

# ---------- endpoints ----------
@router.get("")
def get_loans():
    return LOAN_DB

@router.post("")
def upsert_loan(loan: Dict[str, Any]):
    name = loan.get("name")
    if not name:
        raise HTTPException(400, "Missing loan.name")

    # normalize types
    loan.setdefault("type", "loan")
    loan["balance"] = float(loan.get("balance", 0) or 0)
    if loan.get("apr") not in (None, ""):
        loan["apr"] = float(loan["apr"])
    else:
        loan["apr"] = None
    if loan.get("payment") not in (None, ""):
        loan["payment"] = float(loan["payment"])
    else:
        loan["payment"] = None
    if loan.get("termMonths") in ("", None):
        loan["termMonths"] = None
    else:
        loan["termMonths"] = int(loan["termMonths"])

    global LOAN_DB
    LOAN_DB = [l for l in LOAN_DB if l.get("name") != name]
    LOAN_DB.append(loan)
    _save_db()
    return {"ok": True}

# accept trailing slash too (prevents 307/308 redirect on POST)
@router.post("/", include_in_schema=False)
def upsert_loan_slash(loan: Dict[str, Any]):
    return upsert_loan(loan)

@router.post("/detect")
def detect_loans():
    df = _get_stored_df()
    if df is None or getattr(df, "empty", True):
        raise HTTPException(400, "No bank data loaded. Upload a statement first.")

    tx = df.copy()
    desc_col = "Description" if "Description" in tx.columns else ("description" if "description" in tx.columns else None)
    if not {"date", "credit", "debit"}.issubset(tx.columns) or not desc_col:
        raise HTTPException(400, "Need columns: date, credit, debit, Description.")

    tx["amount"] = tx["debit"].fillna(0) - tx["credit"].fillna(0)
    tx = tx[tx["amount"] > 0]
    tx["payee"] = tx[desc_col].astype(str).map(_clean)
    tx = tx[tx["payee"].str.contains(RGX, na=False)]

    proposals: List[Dict[str, Any]] = []
    for payee, g in tx.groupby("payee"):
        if not _looks_monthly(g["date"]): 
            continue
        amt = float(g["amount"].median())
        first = pd.to_datetime(g["date"]).min().date().isoformat()
        last  = pd.to_datetime(g["date"]).max().date().isoformat()

        p = payee.upper()
        ltype = "loan"
        if re.search(r"MORTGAGE|HELOC", p): ltype = "mortgage"
        elif re.search(r"AUTO|TOYOTA|HONDA|FORD|ALLY", p): ltype = "auto"
        elif re.search(r"NAVIENT|NELNET|SALLIE|MOHELA", p): ltype = "student"
        elif re.search(r"AMEX|CHASE|CITI|CAPITAL ONE|VISA|MASTERCARD|CARD|PAYPAL CREDIT", p): ltype = "credit_card"

        proposals.append({
            "name": payee.title(),
            "type": ltype,
            "minPayment": round(amt, 2),
            "apr": None,
            "balance": None,
            "firstSeen": first,
            "lastSeen": last,
            "source": "bank_ledger",
        })
    return proposals
