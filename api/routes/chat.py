from fastapi import APIRouter, Depends

from api.dependencies import get_chatbot, get_session_store
from api.security import verify_api_key
from src.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    store = get_session_store()
    bot = get_chatbot()

    session = store.get_or_create(request.session_id)
    response_text, prediction = await bot.handle_message(session, request.message)

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        prediction=prediction,
    )
