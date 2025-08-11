from fastapi import APIRouter
from pydantic import BaseModel
from modules.chatbot import answer_question

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

@router.post("/chatbot/")
async def chatbot_api(request: ChatRequest):
    response = answer_question(request.prompt)
    return {"reply": response}
