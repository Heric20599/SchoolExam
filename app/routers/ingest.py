from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.ingest import ChunkDocument
from app.services.book_scope import scoped_book_id
from app.services.embeddings import embed_texts
from app.services.pdf_loader import pdf_to_chunk_documents
from app.services.pinecone_store import upsert_chunks

router = APIRouter(prefix="/books", tags=["ingest"])


@router.post("/upload")
async def upload_book(
    request: Request,
    file: UploadFile = File(...),
    subject: int = Form(...),
    class_id: int = Form(..., alias="class"),
    chapter: int = Form(...),
    publication: int = Form(...),
):
    settings = request.app.state.settings
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.max_pdf_mb:
        raise HTTPException(status_code=413, detail=f"PDF too large. Max allowed is {settings.max_pdf_mb} MB")

    book_id = scoped_book_id(publication=publication, class_id=class_id, subject=subject, chapter=chapter)

    temp_dir = Path("tmp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}-{file.filename}"
    with temp_path.open("wb") as fh:
        fh.write(raw)

    try:
        docs: list[ChunkDocument] = pdf_to_chunk_documents(
            temp_path,
            book_id=book_id,
            class_str=str(class_id),
            subject=str(subject),
            publication=str(publication),
            default_chapter=chapter,
            default_chapter_name=f"Chapter {chapter}",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vectors = embed_texts(request.app.state.openai, settings.openai_embed_model, [d.text for d in docs])
        return {
            "status": "completed",
            "message": "Book uploaded and indexed successfully.",
            "book_id": book_id,
            "publication": publication,
            "class": class_id,
            "subject": subject,
            "chapter": chapter,
            "chunks_upserted": upsert_chunks(
                request.app.state.pinecone, settings.pinecone_index, docs, vectors
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(str(exc).strip() or "Upload processing failed due to an internal error."),
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
