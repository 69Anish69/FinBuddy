# backend/routes/loan_docs.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import pdfplumber, io, re
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/loan-docs", tags=["loans"])

AMOUNT = r"\$?\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[0-9]+(?:\.[0-9]{2}))"
PCT    = r"([0-9]{1,2}(?:\.[0-9]{1,3})?)\s?%"

def _clean_text(text: str) -> str:
    # Collapse spaces; normalize
    return re.sub(r"[ \t]+", " ", text.replace("\x00"," "))

def _amount(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace(",", "").replace("$", "").strip()
    try: return float(s)
    except: return None

def _percent(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace("%", "").strip()
    try: return float(s)
    except: return None

def extract_credit_card(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": "credit_card"}
    # New/Statement balance
    m = re.search(r"(?i)(new|statement)\s+balance[: ]+\$?\s*" + AMOUNT, text)
    if m: out["balance"] = _amount(m.group(2))
    # Minimum payment due
    m = re.search(r"(?i)(minimum\s+payment\s+due|min\.?\s+payment)[: ]+\$?\s*" + AMOUNT, text)
    if m: out["minPayment"] = _amount(m.group(2))
    # Due date
    m = re.search(r"(?i)(payment\s+due\s+date|due\s+date)[: ]+([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})", text)
    if m: out["dueDate"] = m.group(2)
    # Credit limit (RBC/Scotiabank/CIBC wording varies)
    m = re.search(r"(?i)(credit|authorized)\s+limit[: ]+\$?\s*" + AMOUNT, text)
    if m: out["creditLimit"] = _amount(m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2))

    # APR table (RBC/Scotia style: Purchases / Cash advances / Balance transfers)
    apr_lines: List[Dict[str, Any]] = []
    for label in ["purchase", "purchases", "cash advance", "cash advances", "balance transfer", "balance transfers"]:
        rgx = re.compile(rf"(?i){label}.*?{PCT}", re.S)
        m = rgx.search(text)
        if m:
            apr_lines.append({"bucket": label, "apr": _percent(m.group(1))})
    # Some statements show “Annual Interest Rate … 20.99%”
    if not apr_lines:
        for m in re.finditer(r"(?i)(annual\s+(percentage\s+)?rate|interest\s+rate)[^%]{0,40}" + PCT, text):
            apr_lines.append({"bucket": "rate", "apr": _percent(m.group(m.lastindex or 1))})

    # choose a representative APR: highest bucket (conservative)
    if apr_lines:
        out["aprDetails"] = apr_lines
        out["apr"] = max([x["apr"] for x in apr_lines if x.get("apr") is not None], default=None)

    return out

def extract_installment(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": "installment"}
    # Principal balance / outstanding balance
    m = re.search(r"(?i)(principal|outstanding|current)\s+balance[: ]+\$?\s*" + AMOUNT, text)
    if m: out["balance"] = _amount(m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2))
    # Interest rate / APR
    m = re.search(r"(?i)(interest\s+rate|annual\s+interest\s+rate|apr)[: ]+" + PCT, text)
    if m: out["apr"] = _percent(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
    # Regular/Monthly payment
    m = re.search(r"(?i)(regular|monthly)\s+payment[: ]+\$?\s*" + AMOUNT, text)
    if m: out["minPayment"] = _amount(m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2))
    # Term remaining (months)
    m = re.search(r"(?i)(remaining|amortization|term)[^0-9]{0,10}(\d{1,3})\s*(months|mos?)", text)
    if m: out["termMonths"] = int(m.group(2))
    return out

@router.post("/parse")
async def parse_statement(file: UploadFile = File(...), name: Optional[str] = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF statement.")
    content = await file.read()

    # Extract text
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except Exception as e:
        raise HTTPException(400, f"Cannot read PDF: {e}")
    text = _clean_text("\n".join(pages))

    card = extract_credit_card(text)
    inst = extract_installment(text)

    # choose the one with more filled fields
    score_card = len([k for k in ["balance","minPayment","apr"] if card.get(k) is not None])
    score_inst = len([k for k in ["balance","minPayment","apr"] if inst.get(k) is not None])
    best = card if score_card >= score_inst else inst
    best["extractedFrom"] = file.filename
    if name: best["name"] = name

    # small sanity: coerce floats
    for k in ["balance","minPayment","apr","creditLimit"]:
        if k in best and isinstance(best[k], str):
            best[k] = _amount(best[k]) if k != "apr" else _percent(best[k])
    return best
