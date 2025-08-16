from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import io
from reportlab.pdfgen import canvas

app = FastAPI(title="FinBuddy Mock API", version="1.0.0")

_LAST_UPLOAD = None

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global _LAST_UPLOAD
    _LAST_UPLOAD = datetime.utcnow().isoformat() + "Z"
    # We discard content; this is a mock
    await file.read()
    return {"ok": True, "filename": file.filename, "size": "mock", "stored": True}

@app.get("/upload/last")
def last_upload():
    return {"last_upload": _LAST_UPLOAD}

@app.post("/forecast/")
async def forecast(payload: dict):
    # Return the exact shape your UI expects
    # (strategy text lines, small suggestion list, no heavy ML)
    scenario = payload.get("scenario", "current")
    months = int(payload.get("months", 6))
    target = payload.get("target", "Income")

    strategy = (
        f"Scenario: {scenario}\n"
        f"Target: {target} for {months} months\n"
        "1) Track top 3 expense categories weekly.\n"
        "2) Automate a fixed transfer on payday.\n"
        "3) Cap discretionary spend to a weekly budget."
    )

    # 1x1 transparent PNG (base64) would be fine too, but we can just omit chart
    # UI handles when chart is empty — or provide a tiny placeholder chart later if needed.
    return {
        "strategy": strategy,
        "chart": "",  # leave empty or inject a base64 later
        "goal_progress": [
            {"label": "Emergency Fund", "percent": 42},
            {"label": "Debt Paydown", "percent": 25},
        ],
        "anomalies": [
            "March spending unusually high in Dining Out",
            "Subscription spike detected in June",
        ],
        "suggestions": [
            "Cancel unused subscriptions (~$20–$30/mo).",
            "Set weekly dining cap and review on Sunday.",
            "Sweep leftover balance to savings each payday.",
        ],
        "risk_level": "MEDIUM",
        "impact_estimate_6m": 450,  # positive = savings
    }

@app.post("/forecast/pdf")
def forecast_pdf(payload: dict):
    # Generate a tiny 1-page PDF so “Export PDF” works
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.setTitle("FinBuddy – Mock Forecast")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "FinBuddy – Forecast (Mock)")
    c.setFont("Helvetica", 11)
    y = 720
    for line in [
        f"Scenario: {payload.get('scenario', 'current')}",
        f"Target: {payload.get('target', 'Income')}",
        f"Months: {payload.get('months', 6)}",
        "Tips:",
        " • Cancel unused subscriptions",
        " • Weekly cap on dining",
        " • Automate savings on payday",
    ]:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": 'attachment; filename="FinBuddy_Mock_Forecast.pdf"'} )
