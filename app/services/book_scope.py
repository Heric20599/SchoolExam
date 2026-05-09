"""Book identity aligned with multipart upload (publication, class, subject, chapter)."""


def scoped_book_id(*, publication: int, class_id: int, subject: int, chapter: int) -> str:
    """Stable Pinecone `book_id`: p{pub}-c{class}-s{subject}-h{chapter}."""
    return f"p{publication}-c{class_id}-s{subject}-h{chapter}"
