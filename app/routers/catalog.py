from fastapi import APIRouter, HTTPException, Query, Request

from app.services.book_scope import scoped_book_id
from app.services.embeddings import embed_texts
from app.services.pinecone_store import get_book_overview, list_books

router = APIRouter(tags=["catalog"])


def _str_id(value: int) -> str:
    return str(value)


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


def _slim_book_row(book: dict) -> dict:
    """Stable shape for list responses; `class` is the class id (new metadata key `class`, legacy `grade`)."""
    pub = book.get("publication") or book.get("book_type")
    cls = book.get("class") if book.get("class") is not None else book.get("grade")
    return {
        "book_id": book.get("book_id"),
        "class": cls,
        "subject": book.get("subject"),
        "publication": pub,
        "chapter_count": book.get("chapter_count"),
    }


@router.get("/books", summary="List books with filter facets")
def get_books(
    request: Request,
    class_id: int | None = Query(default=None, alias="class", description="Class id (integer); same as upload."),
    subject: int | None = Query(default=None, description="Subject id (integer); same as upload."),
    publication: int | None = Query(
        default=None,
        description="Publication id (integer); same as upload.",
    ),
):
    """List indexed books. Optional `class`, `subject`, `publication` narrow the list. Same response always includes facet arrays for UI dropdowns (values taken from the filtered set)."""
    settings = request.app.state.settings
    probe = embed_texts(request.app.state.openai, settings.openai_embed_model, ["school books"])[0]
    books = list_books(request.app.state.pinecone, settings.pinecone_index, probe_vector=probe)

    filtered = books
    if class_id is not None:
        want = _str_id(class_id)
        filtered = [
            b
            for b in filtered
            if str((b.get("class") if b.get("class") is not None else b.get("grade")) or "").strip() == want
        ]
    if subject is not None:
        want = _str_id(subject)
        filtered = [b for b in filtered if str(b.get("subject") or "").strip() == want]
    if publication is not None:
        want = _str_id(publication)
        filtered = [
            b
            for b in filtered
            if str(b.get("publication") or b.get("book_type") or "").strip() == want
        ]

    classes = sorted(
        {
            str(b.get("class") if b.get("class") is not None else b.get("grade"))
            for b in filtered
            if (b.get("class") if b.get("class") is not None else b.get("grade")) is not None
        }
    )
    subjects = sorted({str(b.get("subject")) for b in filtered if b.get("subject") is not None})
    pub_vals: set[str] = set()
    for b in filtered:
        p = str(b.get("publication") or b.get("book_type") or "").strip()
        if p:
            pub_vals.add(p)
    publications = sorted(pub_vals)

    return {
        "count": len(filtered),
        "books": [_slim_book_row(b) for b in filtered],
        "classes": classes,
        "subjects": subjects,
        "publications": publications,
    }


@router.get("/books/one", summary="Get one book (detail)")
def get_book_one(
    request: Request,
    book_id: str | None = Query(default=None, description="Indexed book id (e.g. from upload response)."),
    class_id: int | None = Query(default=None, alias="class"),
    subject: int | None = None,
    publication: int | None = None,
    chapter: int | None = None,
):
    """Single book detail. Pass `book_id` OR all of `class`, `subject`, `publication`, `chapter` (ints)."""
    settings = request.app.state.settings
    resolved_id = _resolve_book_id(
        book_id=book_id,
        class_id=class_id,
        subject=subject,
        publication=publication,
        chapter=chapter,
    )
    probe = embed_texts(request.app.state.openai, settings.openai_embed_model, [resolved_id])[0]
    overview = get_book_overview(
        request.app.state.pinecone,
        settings.pinecone_index,
        probe_vector=probe,
        book_id=resolved_id,
    )
    if overview is None:
        raise HTTPException(status_code=404, detail="Book not found")

    raw_chapters = overview.pop("chapters", [])
    chapter_numbers: list[int] = []
    for row in raw_chapters:
        if isinstance(row, dict) and row.get("chapter") is not None:
            chapter_numbers.append(int(row["chapter"]))
        elif isinstance(row, int):
            chapter_numbers.append(row)
    chapter_numbers = sorted(set(chapter_numbers))
    bid = overview.pop("book_id", resolved_id)
    overview.pop("source_id", None)  # legacy metadata only
    stats = {
        "chunks": overview.pop("chunks_found"),
        "pages": overview.pop("page_count"),
        "chapter_count": len(chapter_numbers),
    }
    return {
        "book_id": bid,
        "class": overview.get("class"),
        "subject": overview.get("subject"),
        "publication": overview.get("publication"),
        "stats": stats,
        "chapters": chapter_numbers,
    }
