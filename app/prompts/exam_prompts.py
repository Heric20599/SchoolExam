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
- Respect exact numberOfQuestions per type and difficulty (each row: 1–30; **total across all rows must not exceed 30**).
- Output **at most** the requested question count in total (never more than 30 questions); never add extra questions beyond `questionTypes`. If context is thin, return fewer rather than inventing more.
- Ensure final JSON matches schema exactly.
- Treat `description` as teacher guidance (focus/topics/style/constraints), not as chapter content.
- If the payload lists multiple `chapters`, draw questions fairly across all of them using the provided context chunks.
- Paper layout — `questions` must be sections that club `type` + `difficulty` (MCQ, TOF, FIB, MTF, DES):
  * One section per (`type`, `difficulty`) from `questionTypes`, ordered as in the payload.
  * Shape: `{{"type": "MCQ", "difficulty": "VERY_EASY", "questions": [ ... ]}}` — `type` and `difficulty` only on the section.
  * Nested items: questionCode, text, displayOrder (1..N across the paper), plus type fields (options/correctOption, answer, answers, matchPairs, modelAnswer).
- MCQ options must be objects with: optionLabel, text, displayOrder. You MUST also set `correctOption` to the optionLabel of the one correct choice (e.g. `"B"`).
- TOF: set boolean `answer` (true if the statement is correct, false if incorrect). Put the statement text in `text` (or legacy `statement` which is copied to `text`).
- FIB: include `text` (stem with _____ or clear blanks) and `answers` only; do not include a `blanks` field in output.
- MTF: each MTF question must include **at least two** objects in `matchPairs` (leftText, pairKey, displayOrder each). Never fewer than two pairs per MTF item.
  * Put `instruction` (e.g. “Match each item in Column A with the correct option in Column B.”) **only on the first MTF section** in the whole exam; omit `instruction` on all later MTF sections. Every nested MTF question’s `text` is **only** a short `leftText` list (e.g. “Throwing, Catching”).
- DES: put ONLY the learner-facing prompt in `text`. **Every DES item MUST have a non-empty `modelAnswer`** (marking reference for teachers): 3–6 bullet points or a short paragraph with expected points, aligned to the chapter. Never leave `modelAnswer` as `""`. Do not put the model answer in `text`. If you use `keyPoints` or `rubric`, the server merges them into `modelAnswer`; clients only see `modelAnswer`.
- At the root, include strings `summary` and `analysis` for the model only; the API merges them into `description` in this order: (1) exam summary, (2) exam analytics, (3) the request `description` / teacher instructions last. Clients only see the single `description` field.
"""
