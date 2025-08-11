from fastapi import APIRouter, Depends
from auth.firebase_auth import verify_token

router = APIRouter()

@router.post("/auth/firebase-login")
def firebase_login(user=Depends(verify_token)):
    return {"message": f"Welcome, {user['email']}!"}
