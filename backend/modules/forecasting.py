import pandas as pd
from prophet import Prophet

def run_forecast(df):
    if 'Month' in df.columns:
        df['Date'] = pd.to_datetime(df['Month'], errors='coerce')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("No date column found.")

    df = df.dropna(subset=['Date'])

    if 'Income' in df.columns and 'Expense' in df.columns:
        df['Net'] = df['Income'] - df['Expense']
    elif 'Net' in df.columns:
        pass
    else:
        raise ValueError("Required columns not found.")

    df = df[['Date', 'Net']]
    df = df.groupby('Date').sum().reset_index()

    df.rename(columns={"Date": "ds", "Net": "y"}, inplace=True)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    return {
        "forecast": forecast[['ds', 'yhat']].tail(6).to_dict(orient="records")
    }