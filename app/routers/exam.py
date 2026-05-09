from fastapi import APIRouter, Request

from app.schemas.exam import ExamPayload
from app.services.exam_generator import generate_exam

router = APIRouter(prefix="/exam", tags=["exam"])


@router.post("/generate")
def generate(payload: ExamPayload, request: Request):
    settings = request.app.state.settings
    result = generate_exam(
        payload=payload,
        openai_client=request.app.state.openai,
        pinecone_client=request.app.state.pinecone,
        index_name=settings.pinecone_index,
        embed_model=settings.openai_embed_model,
        chat_model=settings.openai_chat_model,
    )
    return result.model_dump(by_alias=True, exclude_none=True)
