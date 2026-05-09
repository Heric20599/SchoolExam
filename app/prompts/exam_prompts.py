from app.services.pinecone_store import metadata_publication_string


def build_exam_prompt(payload: dict, context_chunks: list[dict]) -> str:
    teacher_notes = (payload.get("description") or "").strip()
    teacher_notes_section = (
        f"Teacher instructions:\n{teacher_notes}\n\nUse these instructions to enhance the exam while staying faithful to the provided context."
        if teacher_notes
        else "Teacher instructions:\nNone provided."
    )
    context_lines = []
    for i, c in enumerate(context_chunks, start=1):
        md = c.get("metadata", {})
        context_lines.append(
            f"[{i}] book={md.get('book_id')} publication={metadata_publication_string(md)} "
            f"chapter={md.get('chapter')} page={md.get('page')} "
            f"text={md.get('text', '')[:1200]}"
        )
    context = "\n".join(context_lines)
    return f"""
You are an expert school exam setter.
Generate questions strictly from the context.
Do not invent facts outside context.

Payload:
{payload}

{teacher_notes_section}

Context chunks:
{context}

Rules:
- Respect exact numberOfQuestions per type and difficulty (each questionTypes row: 1–50 questions).
- Ensure final JSON matches schema exactly.
- Add per-question sources with book_id, chapter, page.
- Treat `description` as teacher guidance (focus/topics/style/constraints), not as chapter content.
- If the payload lists multiple `chapters`, draw questions fairly across all of them using the provided context chunks.
- For each question include: questionCode (Q1..), type, difficulty, text, displayOrder.
- MCQ options must be objects with: optionLabel, text, displayOrder. You MUST also set `correctOption` to the optionLabel of the one correct choice (e.g. `"B"`).
- TOF: set boolean `answer` (true if the statement is correct, false if incorrect). Put the statement text in `text` (or legacy `statement` which is copied to `text`).
- FIB: include `text` (stem with _____ or clear blanks) and `answers` only; do not include a `blanks` field in output.
- MTF: include **exactly one** object in `matchPairs` (a single leftText / pairKey / displayOrder triplet). Do not output multiple rows for one MTF item. Always set a non-empty `text` (short instruction like “Match X to Y”); never leave `text` as an empty string.
- DES: put ONLY the learner-facing prompt in `text`. Put marking content in `modelAnswer` only (if the model emits rubric bullets in `keyPoints` or `rubric`, the server merges them into `modelAnswer`; clients do not see `keyPoints`).
- At the root, include strings `summary` and `analysis` for the model only; the API merges them into `description` in this order: (1) exam summary, (2) exam analytics, (3) the request `description` / teacher instructions last. Clients only see the single `description` field.
"""
