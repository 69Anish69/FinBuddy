from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/assistant", tags=["Assistant"])

@router.get("/suggestions")
def suggestions():
    return {"tips": ["Cut unused subscriptions", "Set 20% auto-savings", "Refinance high-APR debt"]}

class AskIn(BaseModel):
    question: str

@router.post("/ask")
def ask(req: AskIn):
    return {"answer": "Placeholder response."}

class LoanPlanIn(BaseModel):
    extra_payment: float | None = 0.0

@router.post("/loans/plan")
def loans_plan(req: LoanPlanIn):
    return {"status": "ok", "plan": []}
