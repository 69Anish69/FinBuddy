from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import user
from routes.mlops_services import forecasting, upload_pipeline, goals_pipeline

app = FastAPI()

# ✅ Allow cross-origin requests (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register API routers
app.include_router(forecasting.router)
app.include_router(upload_pipeline.router)
app.include_router(goals_pipeline.router)
app.include_router(user.router)

# ✅ Root route
@app.get("/")
def root():
    return {"message": "🚀 FinBuddy API running"}
