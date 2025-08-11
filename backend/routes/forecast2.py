from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()
stored_df = None

class ForecastRequest(BaseModel):
    target: str
    months: int

@router.post("/forecast/")
async def forecast_api(req: ForecastRequest):
    global stored_df
    if stored_df is None:
        return {"error": "Please upload a financial CSV/XLS file first."}

    df = stored_df.copy()
    target_col = req.target.lower()
    if target_col == "expenses":
        target_col = "expense"
    elif target_col == "income":
        target_col = "income"
    else:
        return {"error": "Target must be Income or Expense."}

    required_cols = {"month", "income", "expense"}
    if not required_cols.issubset(set(df.columns.str.lower())):
        return {"error": "Missing required columns: Month, Income, Expense."}

    try:
        df.columns = df.columns.str.lower()
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df.dropna(subset=["month"], inplace=True)
        df.sort_values("month", inplace=True)
        prophet_df = df[["month", target_col]].rename(columns={"month": "ds", target_col: "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=req.months, freq="M")
        forecast = model.predict(future)
        recent_value = df[target_col].iloc[-1]
        avg_proj = forecast.tail(req.months)["yhat"].mean()
        growth = avg_proj - recent_value

        if target_col == "income":
            strategy = (
                f"✅ Income may increase by CAD {int(growth)}\n\nStrategies:\n• Contribute to TFSA/RRSP.\n• Allocate surplus to long-term investments.\n• Avoid lifestyle inflation."
                if growth > 0 else
                f"❌ Income may decrease by CAD {abs(int(growth))}\n\nStrategies:\n• Build an emergency fund.\n• Consider part-time work.\n• Delay major expenses."
            )
        else:
            strategy = (
                f"⚠️ Expenses may increase by CAD {int(growth)}\n\nStrategies:\n• Audit categories like groceries/utilities.\n• Use budgeting tools.\n• Switch to cost-effective services."
                if growth > 0 else
                f"✅ Expenses may reduce by CAD {abs(int(growth))}\n\nStrategies:\n• Redirect savings to investments.\n• Reinforce budgeting.\n• Invest in short-term goals."
            )

        fig = model.plot(forecast)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"forecast_text": strategy, "chart_url": image_base64}

    except Exception as e:
        print("❌ Forecast error:", e)
        return {"error": f"Forecasting failed: {str(e)}"}
