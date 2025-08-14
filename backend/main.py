from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.forecast import router as forecast_router
from routes.goals import router as goals_router
from routes.user import router as user_router
from routes.loans import router as loans_router
from routes.loan_docs import router as loan_docs_router
from routes.report import router as report_router

# ✅ If assistant.py lives at backend/routes/assistant.py
from routes.assistant import router as assistant_router
# ❌ If you keep assistant.py next to main.py instead, use:
# import assistant

app = FastAPI()

origins = [
    "http://localhost:3000","http://127.0.0.1:3000",
    "http://localhost:8080","http://127.0.0.1:8080",
]
app.add_middleware(CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# ✅ Include each router ONCE only
app.include_router(upload_router)
app.include_router(forecast_router)
# ❌ DELETE this line; it's the cause of the NameError loop
# app.include_router(forecast.router)

app.include_router(goals_router)
app.include_router(user_router)
app.include_router(loans_router)
app.include_router(loan_docs_router)
app.include_router(report_router)

# If you imported assistant_router:
app.include_router(assistant_router)
# If you used `import assistant` instead:
# app.include_router(assistant.router)

@app.get("/")
def root():
    return {"message": "🚀 FinBuddy API running"}
