from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.services.book_scope import scoped_book_id
from app.services.embeddings import embed_texts
from app.services.pinecone_store import delete_book, delete_chapter, update_book_metadata

router = APIRouter(tags=["admin"])


class BookMetadataUpdateRequest(BaseModel):
    """Partial metadata update; values are numeric ids like upload (stored as digit strings in Pinecone)."""

    model_config = ConfigDict(populate_by_name=True)

    class_id: int | None = Field(default=None, alias="class")
    subject: int | None = None
    publication: int | None = None


def _resolve_book_id(
    *,
    book_id: str | None,
    class_id: int | None,
    subject: int | None,
    publication: int | None,
    chapter: int | None,
) -> str:
    if book_id and book_id.strip():
        return book_id.strip()
    if (
        class_id is not None
        and subject is not None
        and publication is not None
        and chapter is not None
    ):
        return scoped_book_id(
            publication=publication,
            class_id=class_id,
            subject=subject,
            chapter=chapter,
        )
    raise HTTPException(
        status_code=400,
        detail="Provide either query book_id, or all of: class, subject, publication, chapter (integers).",
    )


@router.delete("/books", summary="Delete book or one indexed chapter")
def delete_books(
    request: Request,
    book_id: str | None = Query(
        default=None,
        description="Indexed book id. If set, class/subject/publication/chapter are ignored for targeting.",
    ),
    class_id: int | None = Query(
        default=None,
        alias="class",
        description="Class id (integer). With subject, publication, and chapter, derives book_id (same as upload).",
    ),
    subject: int | None = Query(default=None, description="Subject id (integer); part of scope when book_id is omitted."),
    publication: int | None = Query(
        default=None,
        description="Publication id (integer); part of scope when book_id is omitted.",
    ),
    upload_scope_chapter: int | None = Query(
        default=None,
        alias="chapter",
        description=(
            "Upload-scope chapter (integer): the same `chapter` you used in POST /books/upload; "
            "it is encoded into book_id as …-h{chapter}. It does **not** select which chapter’s vectors to delete—use `indexed_chapter` for that."
        ),
    ),
    indexed_chapter: int | None = Query(
        default=None,
        description=(
            "Pinecone metadata `chapter` to delete for the resolved book only. "
            "Omit to delete the entire book. Independent of query `chapter` when the latter is only used to build book_id."
        ),
    ),
):
    """Delete a whole book, or one indexed chapter inside it. Target via `book_id` OR upload scope (four ints)."""
    settings = request.app.state.settings
    resolved = _resolve_book_id(
        book_id=book_id,
        class_id=class_id,
        subject=subject,
        publication=publication,
        chapter=upload_scope_chapter,
    )
    if indexed_chapter is not None:
        delete_chapter(
            request.app.state.pinecone,
            settings.pinecone_index,
            book_id=resolved,
            chapter=indexed_chapter,
        )
        return {
            "message": "Chapter deleted",
            "book_id": resolved,
            "indexed_chapter": indexed_chapter,
        }
    delete_book(request.app.state.pinecone, settings.pinecone_index, book_id=resolved)
    return {"message": "Book deleted", "book_id": resolved}


@router.patch("/books", summary="Patch book metadata")
def patch_books(
    request: Request,
    payload: BookMetadataUpdateRequest,
    book_id: str | None = Query(
        default=None,
        description="Indexed book id. If set, class/subject/publication/chapter are ignored for targeting.",
    ),
    class_id: int | None = Query(
        default=None,
        alias="class",
        description="Class id (integer). With subject, publication, and chapter, derives book_id (same as upload).",
    ),
    subject: int | None = Query(default=None, description="Subject id (integer); part of scope when book_id is omitted."),
    publication: int | None = Query(
        default=None,
        description="Publication id (integer); part of scope when book_id is omitted.",
    ),
    upload_scope_chapter: int | None = Query(
        default=None,
        alias="chapter",
        description="Upload-scope chapter (integer); same `chapter` as POST /books/upload; encoded into book_id.",
    ),
):
    """Update metadata for vectors of one book. Target via `book_id` OR upload scope (four ints)."""
    settings = request.app.state.settings
    resolved = _resolve_book_id(
        book_id=book_id,
        class_id=class_id,
        subject=subject,
        publication=publication,
        chapter=upload_scope_chapter,
    )
    updates: dict = {}
    if payload.class_id is not None:
        updates["class"] = str(payload.class_id)
    if payload.subject is not None:
        updates["subject"] = str(payload.subject)
    if payload.publication is not None:
        updates["publication"] = str(payload.publication)

    if not updates:
        raise HTTPException(status_code=400, detail="Provide at least one field to update in the body.")

    probe = embed_texts(request.app.state.openai, settings.openai_embed_model, [resolved])[0]
    updated_count = update_book_metadata(
        request.app.state.pinecone,
        settings.pinecone_index,
        probe_vector=probe,
        book_id=resolved,
        metadata_updates=updates,
    )
    if updated_count == 0:
        raise HTTPException(status_code=404, detail="Book not found")

    return {
        "message": "Book metadata updated",
        "book_id": resolved,
        "vectors_updated": updated_count,
        "applied_updates": updates,
    }
