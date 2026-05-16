from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
from pinecone import Pinecone
from pydantic import ValidationError

from app.errors import ConflictError, UpstreamError
from app.prompts.exam_prompts import build_exam_prompt
from app.schemas.exam import (
    EXAM_MAX_TOTAL_QUESTIONS,
    ExamPayload,
    ExamResponse,
    MTF_MAX_MATCH_PAIRS,
    MTF_MIN_MATCH_PAIRS,
)
from app.services.embeddings import embed_texts
from app.services.pinecone_store import (
    metadata_class_string,
    metadata_publication_string,
    pinecone_class_or_legacy_filter,
    pinecone_publication_or_legacy_filter,
    query_chunks,
)

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.strip().lower())
    return " ".join(cleaned.split())


def _normalize_class_id(value: str) -> str:
    compact = "".join(ch for ch in value if ch.isdigit())
    return compact or _normalize_text(value)


def _metadata_text(md: dict, key: str) -> str:
    return str(md.get(key) or "").strip()


def _collect_unique_sources(matches: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], set[int]] = {}
    for m in matches:
        md = m.get("metadata") or {}
        book_id = md.get("book_id")
        chapter = md.get("chapter")
        page = md.get("page")
        if book_id is None or chapter is None or page is None:
            continue
        key = (str(book_id), int(chapter))
        grouped.setdefault(key, set()).add(int(page))
    return [
        {"book_id": book_id, "chapter": chapter, "pages": sorted(pages)}
        for (book_id, chapter), pages in grouped.items()
    ]


def _make_schema_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("type", "object")
            node["additionalProperties"] = False
            props = node.get("properties")
            if isinstance(props, dict):
                # OpenAI strict json_schema requires required to include every property key.
                node["required"] = list(props.keys())
        for value in node.values():
            _make_schema_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_schema_strict(item)


def _count_non_strict_objects(node: Any) -> int:
    count = 0
    if isinstance(node, dict):
        is_object_like = node.get("type") == "object" or "properties" in node
        if is_object_like and node.get("additionalProperties") is not False:
            count += 1
        for value in node.values():
            count += _count_non_strict_objects(value)
    elif isinstance(node, list):
        for item in node:
            count += _count_non_strict_objects(item)
    return count


def _infer_question_type(question: dict) -> str | None:
    if question.get("type"):
        return str(question["type"])
    if isinstance(question.get("options"), list):
        return "MCQ"
    if "statement" in question and isinstance(question.get("answer"), bool):
        return "TOF"
    if isinstance(question.get("blanks"), list):
        return "FIB"
    if isinstance(question.get("leftColumn"), list) and isinstance(question.get("rightColumn"), list):
        return "MTF"
    if "modelAnswer" in question or "keyPoints" in question:
        return "DES"
    return None


def _normalize_source_list(value: Any) -> list[dict]:
    raw_items: list[dict] = []
    if isinstance(value, list):
        raw_items = [v for v in value if isinstance(v, dict)]
    elif isinstance(value, dict):
        raw_items = [value]
    grouped: dict[tuple[str, int], set[int]] = {}
    for item in raw_items:
        book_id = item.get("book_id")
        chapter = item.get("chapter")
        if book_id is None or chapter is None:
            continue
        key = (str(book_id), int(chapter))
        pages = item.get("pages")
        if isinstance(pages, list):
            for p in pages:
                if isinstance(p, int):
                    grouped.setdefault(key, set()).add(p)
                elif isinstance(p, str) and p.isdigit():
                    grouped.setdefault(key, set()).add(int(p))
        elif isinstance(item.get("page"), int):
            grouped.setdefault(key, set()).add(int(item["page"]))
    return [
        {"book_id": book_id, "chapter": chapter, "pages": sorted(pages)}
        for (book_id, chapter), pages in grouped.items()
        if pages
    ]


def _normalize_mcq_options(question: dict) -> None:
    raw_options = question.get("options")
    normalized_options: list[dict] = []
    if isinstance(raw_options, list):
        for idx, opt in enumerate(raw_options, start=1):
            label = chr(64 + idx) if idx <= 26 else f"O{idx}"
            if isinstance(opt, dict):
                normalized_options.append(
                    {
                        "optionLabel": str(opt.get("optionLabel") or label),
                        "text": str(opt.get("text") or ""),
                        "displayOrder": int(opt.get("displayOrder") or idx),
                    }
                )
            else:
                normalized_options.append(
                    {
                        "optionLabel": label,
                        "text": str(opt),
                        "displayOrder": idx,
                    }
                )
    if not normalized_options:
        normalized_options = [
            {"optionLabel": "A", "text": "", "displayOrder": 1},
            {"optionLabel": "B", "text": "", "displayOrder": 2},
        ]
    question["options"] = normalized_options


def _normalize_mcq_correct_option(question: dict) -> None:
    co = question.get("correctOption")
    if isinstance(co, str) and co.strip():
        question["correctOption"] = co.strip().upper()[:8]
        return
    for key in ("correctOptionLabel", "correctAnswer", "correct"):
        v = question.get(key)
        if isinstance(v, str) and v.strip():
            question["correctOption"] = v.strip().upper()[:8]
            return
    ca = question.get("correctAnswer")
    if isinstance(ca, int) and isinstance(question.get("options"), list):
        opts = question["options"]
        if 1 <= ca <= len(opts):
            question["correctOption"] = str(opts[ca - 1].get("optionLabel") or "").strip().upper()[:8]
            return
    opts = question.get("options")
    if isinstance(opts, list):
        for o in opts:
            if not isinstance(o, dict):
                continue
            if o.get("isCorrect") is True or o.get("correct") is True:
                question["correctOption"] = str(o.get("optionLabel") or "").strip().upper()[:8]
                return
    question.setdefault("correctOption", "")


def _normalize_tof_answer(question: dict) -> None:
    a = question.get("answer")
    if isinstance(a, bool):
        return
    if isinstance(a, str):
        s = a.strip().lower()
        if s in ("true", "t", "yes", "1"):
            question["answer"] = True
        elif s in ("false", "f", "no", "0"):
            question["answer"] = False
        return
    ca = question.get("correctAnswer")
    if isinstance(ca, bool):
        question["answer"] = ca
    elif isinstance(ca, str):
        s = ca.strip().lower()
        if s in ("true", "t", "yes", "1"):
            question["answer"] = True
        elif s in ("false", "f", "no", "0"):
            question["answer"] = False


def _normalize_match_pairs(question: dict) -> None:
    if isinstance(question.get("matchPairs"), list) and question["matchPairs"]:
        out = []
        for idx, p in enumerate(question["matchPairs"], start=1):
            if isinstance(p, dict):
                out.append(
                    {
                        "leftText": str(p.get("leftText") or ""),
                        "pairKey": str(p.get("pairKey") or ""),
                        "displayOrder": int(p.get("displayOrder") or idx),
                    }
                )
        if out:
            question["matchPairs"] = out
            return
    left = question.get("leftColumn")
    right = question.get("rightColumn")
    if isinstance(left, list) and isinstance(right, list):
        size = min(len(left), len(right))
        question["matchPairs"] = [
            {
                "leftText": str(left[i]),
                "pairKey": str(right[i]),
                "displayOrder": i + 1,
            }
            for i in range(size)
        ]
        return
    question["matchPairs"] = [
        {"leftText": "", "pairKey": "", "displayOrder": 1},
        {"leftText": "", "pairKey": "", "displayOrder": 2},
    ]


def _ensure_mtf_match_pairs(
    question: dict,
    min_pairs: int = MTF_MIN_MATCH_PAIRS,
    max_pairs: int = MTF_MAX_MATCH_PAIRS,
) -> None:
    mp = question.get("matchPairs")
    if not isinstance(mp, list):
        mp = []
    if len(mp) > max_pairs:
        mp = mp[:max_pairs]
    while len(mp) < min_pairs:
        mp.append({"leftText": "", "pairKey": "", "displayOrder": len(mp) + 1})
    for i, p in enumerate(mp, start=1):
        if isinstance(p, dict):
            p["displayOrder"] = i
    question["matchPairs"] = mp


def _map_openai_error(exc: Exception) -> UpstreamError:
    message_lower = str(exc).lower()
    if isinstance(exc, RateLimitError):
        return UpstreamError(
            "AI request limit reached. Please wait a moment and try again.",
            {"reason": "rate_limit"},
            status_code=429,
            code="rate_limit_exceeded",
        )
    if isinstance(exc, APIConnectionError):
        return UpstreamError(
            "Could not reach the AI service. Please try again shortly.",
            {"reason": "connection_error"},
            status_code=503,
            code="service_unavailable",
        )
    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, APIStatusError) and status_code == 429:
        return UpstreamError(
            "AI request limit reached. Please wait a moment and try again.",
            {"reason": "rate_limit"},
            status_code=429,
            code="rate_limit_exceeded",
        )
    if isinstance(exc, APIError):
        code = str(getattr(exc, "code", "") or "")
        if code in {"insufficient_quota", "billing_hard_limit_reached"} or "quota" in message_lower:
            return UpstreamError(
                "AI service quota exceeded. Please try again later or contact support.",
                {"reason": "quota_exceeded", "provider_code": code or None},
                status_code=429,
                code="quota_exceeded",
            )
        if status_code == 429 or "rate limit" in message_lower:
            return UpstreamError(
                "AI request limit reached. Please wait a moment and try again.",
                {"reason": "rate_limit"},
                status_code=429,
                code="rate_limit_exceeded",
            )
    return UpstreamError("Exam generation failed", {"reason": str(exc)})


def _requested_question_total(payload: ExamPayload) -> int:
    total = sum(spec.numberOfQuestions for spec in payload.questionTypes)
    return min(total, EXAM_MAX_TOTAL_QUESTIONS)


def _cap_questions_to_payload(questions: list[dict], payload: ExamPayload) -> list[dict]:
    """Keep at most the requested count per questionTypes row; drop extras, never pad."""
    remaining = list(questions)
    capped: list[dict] = []

    for spec in payload.questionTypes:
        q_type = spec.type.value
        difficulty = spec.difficultyLevel.value
        limit = spec.numberOfQuestions
        if limit <= 0:
            continue

        picked: list[dict] = []
        still: list[dict] = []
        for q in remaining:
            if len(picked) >= limit:
                still.append(q)
                continue
            if q.get("type") == q_type and q.get("difficulty") == difficulty:
                picked.append(q)
            else:
                still.append(q)

        if len(picked) < limit:
            retry: list[dict] = []
            for q in still:
                if len(picked) >= limit:
                    retry.append(q)
                elif q.get("type") == q_type:
                    picked.append(q)
                else:
                    retry.append(q)
            still = retry

        capped.extend(picked[:limit])
        remaining = still

    max_total = _requested_question_total(payload)
    if len(capped) > max_total:
        capped = capped[:max_total]
    return capped


def _normalize_fib_answers(question: dict) -> None:
    br = question.get("blanks")
    if not str(question.get("text") or "").strip() and br:
        if isinstance(br, list) and br and isinstance(br[0], dict):
            question["text"] = " | ".join(
                str(b.get("prompt") or b.get("label") or b.get("text") or "") for b in br
            )
        elif isinstance(br, list):
            question["text"] = " | ".join(str(b) for b in br)
    br = question.get("blanks")
    if isinstance(br, list) and br and isinstance(br[0], dict):
        stems: list[str] = []
        answers: list[str] = []
        for b in br:
            if not isinstance(b, dict):
                continue
            stems.append(str(b.get("prompt") or b.get("clue") or b.get("label") or b.get("text") or "").strip())
            answers.append(str(b.get("answer") or b.get("solution") or b.get("correct") or "").strip())
        if stems:
            question["answers"] = answers
    raw_ans = question.get("answers")
    if isinstance(raw_ans, list) and all(not isinstance(x, dict) for x in raw_ans):
        question["answers"] = [str(x) for x in raw_ans]
    elif not isinstance(raw_ans, list):
        for key in ("correctAnswers", "correct_answers", "fibAnswers", "solutions"):
            v = question.get(key)
            if isinstance(v, list) and v:
                question["answers"] = [str(x) for x in v]
                break
        else:
            sol = question.get("solution")
            if isinstance(sol, str) and sol.strip():
                question["answers"] = [sol.strip()]
            else:
                a = question.get("answer")
                if isinstance(a, list):
                    question["answers"] = [str(x) for x in a]
                elif isinstance(a, str) and a.strip():
                    question["answers"] = [a.strip()]
                else:
                    question.setdefault("answers", [])
    question.setdefault("answers", [])
    question.pop("blanks", None)


def _mtf_left_items(match_pairs: Any) -> list[str]:
    if not isinstance(match_pairs, list):
        return []
    items: list[str] = []
    for p in match_pairs:
        if not isinstance(p, dict):
            continue
        lt = str(p.get("leftText") or "").strip()
        if lt and lt not in items:
            items.append(lt)
    return items


def _mtf_pair_keys(match_pairs: Any) -> list[str]:
    if not isinstance(match_pairs, list):
        return []
    keys: list[str] = []
    for p in match_pairs:
        if not isinstance(p, dict):
            continue
        pk = str(p.get("pairKey") or "").strip()
        if pk and pk not in keys:
            keys.append(pk)
    return keys


_MTF_SECTION_INSTRUCTION = "Match each item in Column A with the correct option in Column B."


def _is_generic_mtf_stem(text: str) -> bool:
    lower = text.strip().lower()
    if not lower:
        return True
    if lower in {"match the following.", "match the following", "match the following:"}:
        return True
    if lower.startswith("match each of these with the correct option"):
        return True
    if lower.startswith("match each item in column a"):
        return True
    if not lower.startswith("match the following") and not lower.startswith("match each"):
        return False
    generic_tails = (
        " to their importance",
        " to their descriptions",
        " to their categories",
        " to the correct",
        " with the correct",
        " to the right",
        " with their",
        " to their",
    )
    return any(tail in lower for tail in generic_tails) or len(lower) < 120


def _build_mtf_compact_text(match_pairs: Any) -> str:
    """Item-only label for follow-up MTF questions in the same section (no repeated match instruction)."""
    left_items = _mtf_left_items(match_pairs)
    if not left_items:
        return "See Column A."
    return ", ".join(left_items[:8])


def _normalize_mtf_text(question: dict) -> None:
    """Per-question text is item labels only; section `instruction` is set when grouping."""
    mp = question.get("matchPairs")
    text = str(question.get("text") or "").strip()
    compact = _build_mtf_compact_text(mp) or "See Column A."
    if not text or _is_generic_mtf_stem(text):
        question["text"] = compact
        return
    if text.lower().startswith("match "):
        question["text"] = compact


def _coerce_des_model_answer_value(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        if not parts:
            return ""
        if all(p.startswith("•") for p in parts):
            return "\n".join(parts)
        return "\n".join(f"• {p}" for p in parts)
    if value is None:
        return ""
    return str(value).strip()


def _normalize_des_for_response(question: dict) -> None:
    ma = question.get("modelAnswer")
    if isinstance(ma, list):
        question["modelAnswer"] = _coerce_des_model_answer_value(ma)
        ma = question["modelAnswer"]
    elif ma is not None and not isinstance(ma, str):
        question["modelAnswer"] = _coerce_des_model_answer_value(ma)
        ma = question["modelAnswer"]
    if ma is None or not str(ma).strip():
        for key in (
            "answer",
            "suggestedAnswer",
            "model_answer",
            "exemplarAnswer",
            "markingNotes",
            "markingScheme",
            "expectedAnswer",
            "sampleAnswer",
            "solution",
            "markingCriteria",
            "marking_notes",
            "expectedResponse",
        ):
            v = question.get(key)
            if isinstance(v, str) and v.strip():
                question["modelAnswer"] = v.strip()
                break
            if isinstance(v, list) and v:
                question["modelAnswer"] = "\n".join(str(x).strip() for x in v if str(x).strip()).strip()
                break
    extras: list[str] = []
    for key in ("keyPoints", "rubric", "markingPoints", "criteria"):
        raw = question.get(key)
        if isinstance(raw, list):
            extras.extend(str(x).strip() for x in raw if str(x).strip())
        elif isinstance(raw, str) and raw.strip():
            extras.append(raw.strip())
    if extras:
        bullets = "\n".join(f"• {x}" for x in extras)
        ma = str(question.get("modelAnswer") or "").strip()
        question["modelAnswer"] = f"{ma}\n\n{bullets}".strip() if ma else bullets
    for k in (
        "keyPoints",
        "rubric",
        "markingPoints",
        "criteria",
        "answer",
        "suggestedAnswer",
        "model_answer",
        "exemplarAnswer",
        "markingNotes",
        "markingScheme",
        "expectedAnswer",
        "sampleAnswer",
        "solution",
        "markingCriteria",
        "marking_notes",
        "expectedResponse",
    ):
        question.pop(k, None)
    if not str(question.get("modelAnswer") or "").strip():
        question["modelAnswer"] = ""


def _des_missing_model_answers(sections: list[Any]) -> list[str]:
    """Return questionCodes of DES items with empty modelAnswer after repair."""
    missing: list[str] = []
    for section in sections:
        if not isinstance(section, dict) or section.get("type") != "DES":
            continue
        nested = section.get("questions")
        if not isinstance(nested, list):
            continue
        for q in nested:
            if not isinstance(q, dict):
                continue
            if not str(q.get("modelAnswer") or "").strip():
                missing.append(str(q.get("questionCode") or "?"))
    return missing


def _normalize_difficulty(value: Any, fallback: str) -> str:
    txt = str(value or fallback).strip().upper()
    return txt if txt in {"VERY_EASY", "EASY", "MEDIUM", "HARD", "VERY_HARD"} else fallback


def _flatten_raw_questions(raw_questions: list[Any]) -> list[dict]:
    """Grouped sections or legacy flat list from the LLM."""
    flat: list[dict] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        nested = item.get("questions")
        if isinstance(nested, list) and item.get("type"):
            section_type = str(item.get("type") or "")
            section_diff = item.get("difficulty") or item.get("difficultyLevel")
            for q in nested:
                if not isinstance(q, dict):
                    continue
                qq = dict(q)
                qq.setdefault("type", section_type)
                if section_diff is not None:
                    qq.setdefault("difficulty", section_diff)
                flat.append(qq)
        else:
            flat.append(dict(item))
    return flat


def _strip_question_section_fields(question: dict) -> dict:
    return {k: v for k, v in question.items() if k not in ("type", "difficulty", "difficultyLevel")}


def _group_questions_for_paper_layout(questions: list[dict], payload: ExamPayload) -> list[dict]:
    """Club type + difficulty into sections; renumber displayOrder / questionCode."""
    section_order: list[tuple[str, str]] = []
    for spec in payload.questionTypes:
        key = (spec.type.value, spec.difficultyLevel.value)
        if key not in section_order:
            section_order.append(key)

    buckets: dict[tuple[str, str], list[dict]] = {k: [] for k in section_order}
    trailing: list[dict] = []
    for q in questions:
        key = (
            str(q.get("type") or ""),
            str(q.get("difficulty") or q.get("difficultyLevel") or ""),
        )
        if key in buckets:
            buckets[key].append(q)
        else:
            trailing.append(q)

    sections: list[dict] = []
    global_idx = 0
    mtf_instruction_shown = False

    def _append_section(q_type: str, difficulty: str, block: list[dict]) -> None:
        nonlocal global_idx, mtf_instruction_shown
        if not block:
            return
        inner: list[dict] = []
        for q in sorted(block, key=lambda x: int(x.get("displayOrder") or 0)):
            global_idx += 1
            item = _strip_question_section_fields(q)
            item["displayOrder"] = global_idx
            item["questionCode"] = f"Q{global_idx}"
            inner.append(item)
        if q_type == "MTF":
            for item in inner:
                item["text"] = _build_mtf_compact_text(item.get("matchPairs")) or "See Column A."
            section_dict: dict = {
                "type": q_type,
                "difficulty": difficulty,
                "questions": inner,
            }
            if not mtf_instruction_shown:
                section_dict["instruction"] = _MTF_SECTION_INSTRUCTION
                mtf_instruction_shown = True
            sections.append(section_dict)
        else:
            sections.append({"type": q_type, "difficulty": difficulty, "questions": inner})

    for q_type, difficulty in section_order:
        _append_section(q_type, difficulty, buckets[(q_type, difficulty)])

    trailing_keys: list[tuple[str, str]] = []
    for q in trailing:
        key = (
            str(q.get("type") or ""),
            str(q.get("difficulty") or q.get("difficultyLevel") or ""),
        )
        if key not in trailing_keys:
            trailing_keys.append(key)
    for q_type, difficulty in trailing_keys:
        block = [
            q
            for q in trailing
            if str(q.get("type") or "") == q_type
            and str(q.get("difficulty") or q.get("difficultyLevel") or "") == difficulty
        ]
        _append_section(q_type, difficulty, block)

    return sections


def _count_questions_in_sections(sections: list[Any]) -> int:
    total = 0
    for section in sections:
        if not isinstance(section, dict):
            continue
        nested = section.get("questions")
        if isinstance(nested, list):
            total += len(nested)
    return total


def _repair_generated_exam_data(data: dict, payload: ExamPayload, context_matches: list[dict]) -> dict:
    repaired = dict(data)
    raw_questions = repaired.get("questions")
    if not isinstance(raw_questions, list):
        raw_questions = []
    flat_questions = _flatten_raw_questions(raw_questions)
    default_difficulty = payload.questionTypes[0].difficultyLevel.value if payload.questionTypes else "EASY"
    normalized_questions: list[dict] = []
    for idx, item in enumerate(flat_questions, start=1):
        if not isinstance(item, dict):
            continue
        q = dict(item)
        q_type = _infer_question_type(q)
        if q_type:
            q["type"] = q_type
        q["questionCode"] = str(q.get("questionCode") or f"Q{idx}")
        q["text"] = str(q.get("text") or q.get("question") or q.get("statement") or q.get("prompt") or "")
        q["displayOrder"] = int(q.get("displayOrder") or idx)
        q["difficulty"] = _normalize_difficulty(q.get("difficulty") or q.get("difficultyLevel"), default_difficulty)
        q.pop("sources", None)
        if q.get("type") == "MCQ":
            _normalize_mcq_options(q)
            _normalize_mcq_correct_option(q)
        if q.get("type") == "TOF":
            _normalize_tof_answer(q)
        if q.get("type") == "MTF":
            _normalize_match_pairs(q)
            _ensure_mtf_match_pairs(q)
            _normalize_mtf_text(q)
        if q.get("type") == "FIB":
            _normalize_fib_answers(q)
        if q.get("type") == "DES":
            _normalize_des_for_response(q)
        normalized_questions.append(q)

    requested_total = _requested_question_total(payload)
    capped_questions = _cap_questions_to_payload(normalized_questions, payload)
    if len(normalized_questions) > len(capped_questions):
        logger.info(
            "Trimmed exam questions from model: had=%d capped=%d requested=%d",
            len(normalized_questions),
            len(capped_questions),
            requested_total,
        )
    if len(capped_questions) < requested_total:
        logger.warning(
            "Exam returned fewer questions than requested: got=%d requested=%d",
            len(capped_questions),
            requested_total,
        )

    repaired["questions"] = _group_questions_for_paper_layout(capped_questions, payload)
    repaired.pop("sources", None)
    base = str(payload.description or "").strip()
    summ = str(repaired.get("summary") or repaired.get("examSummary") or "").strip()
    ana = str(repaired.get("analysis") or repaired.get("paperAnalysis") or "").strip()
    for k in ("summary", "analysis", "examSummary", "paperAnalysis"):
        repaired.pop(k, None)
    parts: list[str] = []
    if summ:
        parts.append(f"Exam summary\n{summ}")
    if ana:
        parts.append(f"Exam analytics\n{ana}")
    if base:
        parts.append(f"Teacher instructions\n{base}")
    repaired["description"] = "\n\n".join(parts).strip()
    return repaired


def _chapter_exists(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    chapter_name: str,
    publication: str | None,
) -> bool:
    parts: list[dict] = [
        pinecone_class_or_legacy_filter(class_str),
        {"subject": {"$eq": subject}},
        {"chapter_name": {"$eq": chapter_name}},
    ]
    if publication:
        parts.append(pinecone_publication_or_legacy_filter(publication))
    metadata_filter: dict = {"$and": parts} if len(parts) > 1 else parts[0]
    matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=1,
        metadata_filter=metadata_filter,
    )
    return len(matches) > 0


def _resolve_chapter_by_number(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    publication: str | None,
    chapter_number: int,
) -> dict | None:
    parts: list[dict] = [
        pinecone_class_or_legacy_filter(class_str),
        {"subject": {"$eq": subject}},
        {"chapter": {"$eq": chapter_number}},
    ]
    if publication:
        parts.append(pinecone_publication_or_legacy_filter(publication))
    metadata_filter: dict = {"$and": parts}
    matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=1,
        metadata_filter=metadata_filter,
    )
    if not matches:
        # Fallback for metadata drift in class/subject/publication formatting.
        broad_matches = query_chunks(
            pc=pc,
            index_name=index_name,
            vector=probe_vector,
            top_k=900,
            metadata_filter={"chapter": {"$eq": chapter_number}},
        )
        requested_class = _normalize_class_id(class_str)
        requested_subject = _normalize_text(subject)
        requested_publication = _normalize_text(publication or "")
        for m in broad_matches:
            md = m.get("metadata") or {}
            candidate_class = _normalize_class_id(metadata_class_string(md))
            candidate_subject = _normalize_text(_metadata_text(md, "subject"))
            candidate_publication = _normalize_text(metadata_publication_string(md))
            if candidate_class != requested_class:
                continue
            if candidate_subject != requested_subject:
                continue
            if requested_publication and candidate_publication != requested_publication:
                continue
            matches = [m]
            break
        if not matches:
            return None
    md = matches[0].get("metadata") or {}
    return {
        "class": metadata_class_string(md) or class_str,
        "subject": _metadata_text(md, "subject") or subject,
        "publication": metadata_publication_string(md) or (publication or ""),
        "chapter_name": _metadata_text(md, "chapter_name") or f"Chapter {chapter_number}",
    }


def _resolve_chapter_match(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    publication: str | None,
    requested_chapter_name: str | int,
) -> dict | None:
    if isinstance(requested_chapter_name, int):
        return _resolve_chapter_by_number(
            pc=pc,
            index_name=index_name,
            probe_vector=probe_vector,
            class_str=class_str,
            subject=subject,
            publication=publication,
            chapter_number=requested_chapter_name,
        )
    if _chapter_exists(
        pc=pc,
        index_name=index_name,
        probe_vector=probe_vector,
        class_str=class_str,
        subject=subject,
        chapter_name=requested_chapter_name,
        publication=publication,
    ):
        return {
            "class": class_str,
            "subject": subject,
            "publication": publication or "",
            "chapter_name": requested_chapter_name,
        }

    # Fallback for metadata drift (case/spacing/punctuation or "Class 9" vs "9", etc.).
    nearby_matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=600,
        metadata_filter={},
    )

    requested_normalized = _normalize_text(requested_chapter_name)
    requested_subject = _normalize_text(subject)
    requested_class = _normalize_class_id(class_str)
    requested_publication = _normalize_text(publication or "")

    strict_candidates: list[dict] = []
    loose_candidates: list[dict] = []
    for match in nearby_matches:
        md = match.get("metadata") or {}
        chapter_name = _metadata_text(md, "chapter_name")
        if not chapter_name or _normalize_text(chapter_name) != requested_normalized:
            continue

        candidate = {
            "class": metadata_class_string(md),
            "subject": _metadata_text(md, "subject"),
            "publication": metadata_publication_string(md),
            "chapter_name": chapter_name,
        }
        subject_ok = _normalize_text(candidate["subject"]) == requested_subject
        class_ok = _normalize_class_id(candidate["class"]) == requested_class
        publication_ok = requested_publication == "" or _normalize_text(candidate["publication"]) == requested_publication
        if subject_ok and class_ok and publication_ok:
            strict_candidates.append(candidate)
        else:
            loose_candidates.append(candidate)

    if not strict_candidates and not loose_candidates:
        # Try a metadata-scoped fallback to avoid semantic miss in broad nearest-neighbor retrieval.
        scoped_parts: list[dict] = [
            pinecone_class_or_legacy_filter(class_str),
            {"subject": {"$eq": subject}},
        ]
        if publication:
            scoped_parts.append(pinecone_publication_or_legacy_filter(publication))
        scoped_filter: dict = {"$and": scoped_parts}
        scoped_matches = query_chunks(
            pc=pc,
            index_name=index_name,
            vector=probe_vector,
            top_k=900,
            metadata_filter=scoped_filter,
        )
        requested_tokens = set(requested_normalized.split())
        for match in scoped_matches:
            md = match.get("metadata") or {}
            chapter_name = _metadata_text(md, "chapter_name")
            normalized_chapter = _normalize_text(chapter_name)
            if not chapter_name or not normalized_chapter:
                continue
            chapter_tokens = set(normalized_chapter.split())
            overlap = len(requested_tokens & chapter_tokens)
            if normalized_chapter == requested_normalized or overlap >= max(2, len(requested_tokens) - 1):
                strict_candidates.append(
                    {
                        "class": metadata_class_string(md),
                        "subject": _metadata_text(md, "subject"),
                        "publication": metadata_publication_string(md),
                        "chapter_name": chapter_name,
                    }
                )

    if strict_candidates:
        return strict_candidates[0]
    if loose_candidates:
        return loose_candidates[0]
    return None


def generate_exam(payload: ExamPayload, openai_client: OpenAI, pinecone_client: Pinecone, index_name: str, embed_model: str, chat_model: str) -> ExamResponse:
    class_str = str(payload.class_id)
    subject_str = str(payload.subject)
    publication_str = str(payload.publication)
    chapter_numbers = list(payload.chapters)

    logger.info(
        "Exam request received: class=%s subject=%s publication=%s chapters=%s question_types=%d",
        payload.class_id,
        payload.subject,
        payload.publication,
        chapter_numbers,
        len(payload.questionTypes),
    )
    ch_label = " ".join(str(c) for c in chapter_numbers)
    try:
        probe = embed_texts(
            openai_client,
            embed_model,
            [f"{subject_str} {publication_str} chapters {ch_label}"],
        )[0]
    except Exception as exc:
        raise _map_openai_error(exc) from exc

    resolved_chapters: list[dict] = []
    missing: list[int] = []
    for ch_num in chapter_numbers:
        resolved = _resolve_chapter_by_number(
            pc=pinecone_client,
            index_name=index_name,
            probe_vector=probe,
            class_str=class_str,
            subject=subject_str,
            publication=publication_str,
            chapter_number=ch_num,
        )
        if resolved is None:
            missing.append(ch_num)
        else:
            resolved_chapters.append(resolved)
    if missing:
        logger.warning(
            "Exam request missing chapters after resolution: class=%s subject=%s publication=%s missing=%s",
            payload.class_id,
            payload.subject,
            payload.publication,
            missing,
        )
        raise ConflictError(
            "Some chapters are not uploaded yet. Please upload first.",
            details={
                "missing_chapters": sorted(set(missing)),
                "hint": "POST /books/upload with the same class, subject, publication, and each chapter you reference in `chapters` (or legacy `chapter`).",
            },
        )

    context_matches: list[dict] = []
    seen_ids: set[str] = set()
    for chapter_match in resolved_chapters:
        try:
            chapter_query_vec = embed_texts(
                openai_client,
                embed_model,
                [f"{chapter_match['subject'] or subject_str} {chapter_match['chapter_name']} exam questions"],
            )[0]
        except Exception as exc:
            raise _map_openai_error(exc) from exc
        parts_pf: list[dict] = [
            pinecone_class_or_legacy_filter(chapter_match["class"]),
            {"subject": {"$eq": chapter_match["subject"]}},
            {"chapter_name": {"$eq": chapter_match["chapter_name"]}},
        ]
        pub = (chapter_match.get("publication") or "").strip()
        if pub:
            parts_pf.append(pinecone_publication_or_legacy_filter(pub))
        primary_filter = {"$and": parts_pf}
        chapter_matches = query_chunks(
            pc=pinecone_client,
            index_name=index_name,
            vector=chapter_query_vec,
            top_k=8,
            metadata_filter=primary_filter,
        )
        for m in chapter_matches:
            mid = m.get("id")
            key = str(mid) if mid is not None else None
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)
            context_matches.append(m)
    logger.info("Resolved chapters=%d context_matches=%d", len(resolved_chapters), len(context_matches))

    prompt = build_exam_prompt(payload.model_dump(by_alias=True), context_matches)

    schema = ExamResponse.model_json_schema()
    _make_schema_strict(schema)
    non_strict_after_patch = _count_non_strict_objects(schema)
    logger.info("Schema strictness check: non_strict_objects=%d", non_strict_after_patch)
    for attempt in range(2):
        try:
            logger.info("Calling OpenAI for exam generation attempt=%d", attempt + 1)
            completion = openai_client.chat.completions.create(
                model=chat_model,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
                # OpenAI response_format json_schema currently rejects oneOf in nested fields.
                # We request JSON object output and enforce the full schema via Pydantic below.
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            data = json.loads(content)
            data = _repair_generated_exam_data(data, payload, context_matches)
            data["generated_at"] = data.get("generated_at") or datetime.now(timezone.utc).isoformat()
            data["class"] = class_str
            data["subject"] = subject_str
            questions_out = data.get("questions") or []
            data["totalQuestions"] = _count_questions_in_sections(questions_out)
            data["publication"] = str(payload.publication)
            data["chapters"] = list(payload.chapters)
            missing_des = _des_missing_model_answers(questions_out)
            if missing_des:
                logger.warning(
                    "DES missing modelAnswer attempt=%d codes=%s",
                    attempt + 1,
                    missing_des,
                )
                if attempt == 1:
                    raise UpstreamError(
                        "Exam has descriptive questions without model answers. Please try again.",
                        {"questionCodes": missing_des},
                        code="missing_des_model_answer",
                    )
                prompt += (
                    "\n\nPrevious output omitted modelAnswer on DES question(s): "
                    + ", ".join(missing_des)
                    + ". Every DES MUST include a non-empty modelAnswer (3–6 marking points from the chapter). "
                    "Return strictly valid JSON."
                )
                continue
            logger.info(
                "Exam generation success attempt=%d questions=%d requested_max=%d",
                attempt + 1,
                data["totalQuestions"],
                _requested_question_total(payload),
            )
            return ExamResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Exam JSON/validation error attempt=%d reason=%s", attempt + 1, str(exc))
            if attempt == 1:
                raise UpstreamError(
                    "LLM returned invalid exam format",
                    {"reason": str(exc)},
                    code="invalid_exam_format",
                ) from exc
            prompt += (
                "\n\nPrevious output failed schema validation. Return strictly valid JSON. "
                "Every DES question MUST have a non-empty modelAnswer field."
            )
        except Exception as exc:  # pragma: no cover - network path
            mapped = _map_openai_error(exc)
            if mapped.status_code in (429, 503):
                logger.warning("Exam generation blocked attempt=%d: %s", attempt + 1, mapped.message)
            else:
                logger.exception("Exam generation upstream error attempt=%d", attempt + 1)
            raise mapped from exc

    raise UpstreamError("Exam generation failed unexpectedly")
