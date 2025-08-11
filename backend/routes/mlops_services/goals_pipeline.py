from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import locale
import logging

logger = logging.getLogger("goals_pipeline")
logging.basicConfig(level=logging.INFO)
router = APIRouter()
locale.setlocale(locale.LC_ALL, 'en_CA.UTF-8')

router = APIRouter()

class GoalItem(BaseModel):
    name: str
    price: float
    year: int
    current: float = 0

class GoalPlanRequest(BaseModel):
    goals: list[GoalItem]

@router.post("/mlops_services/goals_pipelines/")
async def calculate_goals(request: GoalPlanRequest):
    current_year = datetime.now().year
    plans = []

    for goal in request.goals:
        years_left = max(1, goal.year - current_year)
        total_months = years_left * 12
        monthly = round(goal.price / total_months, 2)
        progress = round((goal.current / goal.price) * 100, 1) if goal.price > 0 else 0

        plans.append({
            "name": goal.name,
            "target": f"{locale.currency(goal.price, grouping=True)} CAD by {goal.year}",
            "monthly": f"{locale.currency(monthly, grouping=True)}",
            "current": f"{locale.currency(goal.current, grouping=True)}",
            "progress": progress
        })

    return {"goals": plans}
