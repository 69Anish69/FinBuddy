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

router = APIRouter()

UPLOAD_PATH = "uploaded_files/latest_upload.csv"

# Ensure memory timestamp exists
if not hasattr(data_store, "last_upload_time"):
    data_store.last_upload_time = 0

def load_dataframe():
    # Step 1: Use memory if available
    df = getattr(data_store, 'stored_df', None)
    if df is not None and not df.empty:
        return df, "memory"

    # Step 2: Try loading from disk
    if os.path.exists(UPLOAD_PATH):
        try:
            df = pd.read_csv(UPLOAD_PATH)
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
            return df, "disk"
        except Exception as e:
            print("❌ Failed to load from disk:", e)

    return None, None

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

def generate_suggestions(df, target):
    recent = df.tail(3)
    suggestions = []
    avg = df[target].mean()

    for _, row in recent.iterrows():
        month = row['Month'].strftime('%B')
        value = row[target]

        if target == "Income":
            if value < avg * 0.9:
                msg = f"{month}: Income dropped below average. Avoid large purchases and boost income."
            elif value > avg * 1.2:
                msg = f"{month}: High income. Consider allocating surplus to investments."
            else:
                msg = f"{month}: Income stable. Maintain discipline."
        else:
            if value > avg * 1.2:
                msg = f"{month}: Expenses unusually high. Review spending categories."
            elif value < avg * 0.8:
                msg = f"{month}: Low expenses. Great job managing costs!"
            else:
                msg = f"{month}: Expenses in control. Stay consistent."

        suggestions.append({"month": month, "suggestion": msg})

    return suggestions

def calculate_goal_progress(df):
    total_saved = (df['Income'] - df['Expense']).sum()
    return [
        {"label": "Emergency Fund", "percent": min(100, round((total_saved / 10000) * 100))},
        {"label": "Travel Fund", "percent": min(100, round((total_saved / 3000) * 100))}
    ]

def run_forecast(df, target, months):
    required_cols = ["Month", "Income", "Expense"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
    buf = BytesIO()
    plt.title(f"{target} Forecast")
    fig.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")

    strategy_text = f"""📈 Based on past data, your average monthly {target.lower()} is ${df[target].mean():.2f}.
If current trends continue, you're on track to hit your goals within {months} months."""

    return {
        "chart": chart_base64,
        "strategy": strategy_text,
        "goal_progress": calculate_goal_progress(df),
        "anomalies": detect_anomalies(df, target),
        "suggestions": generate_suggestions(df, target),
        "status": "success"
    }

@router.post("/forecast/")
async def forecast_post(req: Request):
    try:
        body = await req.json()
        target = body.get("target", "Income")
        months = int(body.get("months", 6))

        df, source = load_dataframe()

        if df is None or df.empty:
            print("❌ No valid data found.")
            return {"error": "❌ No uploaded data found. Please upload a valid file first."}

        print(f"✅ Forecasting from {source} | Target: {target} | Months: {months}")
        result = run_forecast(df, target, months)
        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": f"❌ Forecasting failed: {str(e)}"}
