# /auth/firebase_auth.py
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Request, HTTPException, Depends

cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # contains email, uid, etc.
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
