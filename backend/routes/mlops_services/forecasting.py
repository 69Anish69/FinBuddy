
from fastapi import APIRouter, Request
import modules.data_store as data_store
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback
import time
import os
import logging

logger = logging.getLogger("Forecasting")
logging.basicConfig(level=logging.INFO)
router = APIRouter()
UPLOAD_PATH = "uploaded_files/latest_upload.csv"

if not hasattr(data_store, "last_upload_time"):
    data_store.last_upload_time = 0

def load_dataframe():
    df = getattr(data_store, 'stored_df', None)

    if df is None or df.empty:
        if os.path.exists(UPLOAD_PATH):
            try:
                df = pd.read_csv(UPLOAD_PATH)
                # ✅ Clean & rename columns
                df.columns = [col.strip().capitalize() for col in df.columns]

                rename_map = {
                    "Date": "Month",
                    "Credit": "Income",
                    "Debit": "Expense",
                }
                df.rename(columns=rename_map, inplace=True)

                if "Month" in df:
                    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

                return df, "disk"
            except Exception as e:
                print("❌ Failed to load from disk:", e)
                return None, None
    else:
        return df, "memory"

def detect_anomalies(df, target):
    anomalies = []
    mean = df[target].mean()
    std = df[target].std()
    threshold = mean + 1.2 * std
    high_rows = df[df[target] > threshold]

    for _, row in high_rows.iterrows():
        if row[target] > 0:
            anomalies.append(f"{row['Month'].strftime('%B %Y')} {target.lower()} was unusually high: ${row[target]:.2f}")
    return anomalies

def generate_monthly_outlook(df, target):
    recent = df.tail(3).reset_index(drop=True)
    outlook = []
    avg_income = df["Income"].mean()
    avg_expense = df["Expense"].mean()

    for i, row in recent.iterrows():
        month = row["Month"].strftime("%B")
        income = row.get("Income", 0) or 0
        expense = row.get("Expense", 0) or 0

        prev_income = recent.at[i-1, "Income"] if i > 0 else avg_income
        prev_expense = recent.at[i-1, "Expense"] if i > 0 else avg_expense

        # Income messages only if target is Income
        if target == "Income":
            if income > prev_income * 1.05:
                comment = "📈 Income increased compared to previous month. Good job!"
            elif income < prev_income * 0.95:
                comment = "⚠️ Income dropped compared to previous month. Consider additional income streams."
            else:
                comment = "Income stable compared to previous month."

        # Expense messages only if target is Expense
        elif target == "Expense":
            if expense > avg_expense * 1.2:
                comment = "💸 Expenses are higher than your average. Review your spending categories."
            elif expense < avg_expense * 0.8:
                comment = "👍 Great expense control this month!"
            else:
                comment = "Expenses are within normal range."
        else:
            comment = ""

        outlook.append(f"• {month}:\n{comment}")

    return outlook

def calculate_goal_progress(df):
    total_saved = (df['Income'] - df['Expense']).sum()
    return [
        {"label": "Emergency Fund", "percent": min(100, round((total_saved / 10000) * 100))},
        {"label": "Travel Fund", "percent": min(100, round((total_saved / 3000) * 100))}
    ]

def run_forecast(df, target, months):
    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors='coerce')
    if df["Month"].isnull().all():
        raise ValueError("❌ Failed to parse 'Month' column into datetime.")

    df = df.dropna(subset=["Month", target])
    df = df.sort_values("Month")

    prophet_df = df[["Month", target]].rename(columns={"Month": "ds", target: "y"})
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=months, freq='M')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_title(f"{target} Forecast with Trend Line & Confidence Interval", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"{target} ($)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend manually
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Forecast Line (yhat)", "Upper Bound", "Lower Bound"], loc="upper left")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")

    trend_comment = "📊 Your forecast shows an upward trend." if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else "📉 A downward trend is predicted. Adjust your planning."

    strategy_text = (
        f"📈 Your average monthly {target.lower()} is ${df[target].mean():,.2f}.\n"
        "You're on track to meet your financial goals.\n\n"
        + trend_comment + "\n\n"
    )

    monthly_outlook = generate_monthly_outlook(df, target)

    return {
        "chart": chart_base64,
        "strategy": strategy_text,
        "goal_progress": calculate_goal_progress(df),
        "anomalies": detect_anomalies(df, target),
        "suggestions": monthly_outlook,
        "status": "success"
    }

@router.post("/forecast/")
@router.post("/mlops_services/forecast-goal")
async def forecast_post(req: Request):
    try:
        logger.info("🚀 [Forecast Pipeline] Started")

        body = await req.json()
        target = body.get("target", "Income")
        months = int(body.get("months", 6))

        df, source = load_dataframe()
        if df is None or df.empty:
            logger.warning("❌ No uploaded data found.")
            return {"error": "❌ No uploaded data found. Please upload a file first."}

        logger.info(f"📊 Using {source} data for forecasting")

        result = run_forecast(df, target, months)

        logger.info("✅ [Forecast Pipeline] Completed successfully")

        return result
    except Exception as e:
        logger.error(f"❌ Forecasting failed: {str(e)}")
        traceback.print_exc()
        return {"error": f"❌ Forecasting failed: {str(e)}"}

