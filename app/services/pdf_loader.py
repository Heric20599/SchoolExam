from __future__ import annotations

import re
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from app.schemas.ingest import ChunkDocument

CHAPTER_PATTERN = re.compile(r"^\s*chapter\s+(\d+)\s*[:\-\s]*(.*)$", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + chunk_size]
        if not piece:
            continue
        decoded = enc.decode(piece).strip()
        if decoded:
            chunks.append(decoded)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def pdf_to_chunk_documents(
    pdf_path: str | Path,
    *,
    book_id: str,
    class_str: str,
    subject: str,
    publication: str,
    default_chapter: int | None,
    default_chapter_name: str | None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[ChunkDocument]:
    reader = PdfReader(str(pdf_path))
    documents: list[ChunkDocument] = []

    chapter_num = default_chapter or 1
    chapter_name = _normalize_text(default_chapter_name or "General")
    lock_chapter = default_chapter is not None

    for page_i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        if not lock_chapter:
            first_line = page_text.splitlines()[0] if page_text.splitlines() else ""
            m = CHAPTER_PATTERN.match(first_line)
            if m:
                chapter_num = int(m.group(1))
                chapter_name = _normalize_text(m.group(2) or f"Chapter {chapter_num}")

        chunks = _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{book_id}::ch{chapter_num}::p{page_i}::c{chunk_idx}"
            documents.append(
                ChunkDocument(
                    id=doc_id,
                    text=chunk,
                    class_str=class_str,
                    subject=subject,
                    book_id=book_id,
                    publication=publication,
                    chapter=chapter_num,
                    chapter_name=chapter_name,
                    page=page_i,
                    chunk_index=chunk_idx,
                )
            )
    return documents
