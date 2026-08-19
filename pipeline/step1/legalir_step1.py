"""Step 1 data preparation for UIT DSC 2026 LegalIR.

This module implements:
- train/public/context schema validation
- deterministic train/dev split
- legal-text cleaning
- structure-aware chunking
- official metric and submission validation

It uses only the Python standard library so it can run on Kaggle without
installing extra packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote


DEFAULT_KAGGLE_INPUT = Path(
    "/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test"
)
DEFAULT_OUTPUT = (
    Path("/kaggle/working/step1")
    if Path("/kaggle").exists()
    else Path("task1/pipeline/step1/outputs")
)
DEFAULT_LOCAL_CONTEXT_DIR = Path("selected-contexts/selected-contexts")
MAX_SUBMISSION_DOCS = 5


@dataclass(frozen=True)
class ChunkConfig:
    window_words: int = 320
    overlap_words: int = 60
    long_section_words: int = 900
    max_chunk_warning_per_context: int = 420


@dataclass
class ValidationIssue:
    severity: str
    kind: str
    item_id: str
    message: str


@dataclass
class ContextRecord:
    doc_id: str
    source_path: str
    link: str
    name: str
    name_source: str
    passage: str
    is_empty_passage_fallback: bool = False


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_match(text: str) -> str:
    return strip_accents(text).lower()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def find_input_paths(
    input_root: Path | None,
    train_file: Path | None,
    public_file: Path | None,
    context_dir: Path | None,
) -> tuple[Path, Path, Path]:
    root_candidates: list[Path] = []
    if input_root is not None:
        root_candidates.append(input_root)
    root_candidates.extend(
        [
            DEFAULT_KAGGLE_INPUT,
            Path("task1"),
            Path("."),
        ]
    )

    resolved_train = train_file
    resolved_public = public_file
    resolved_context = context_dir

    for root in root_candidates:
        if resolved_train is None and (root / "train.json").exists():
            resolved_train = root / "train.json"
        if resolved_public is None and (root / "public-official.json").exists():
            resolved_public = root / "public-official.json"
        if resolved_context is None:
            context_candidates = [
                root / "selected-contexts",
                root / "selected-contexts" / "selected-contexts",
                Path("selected-contexts") / "selected-contexts",
                Path("selected-contexts"),
            ]
            for candidate in context_candidates:
                if candidate.exists() and any(candidate.glob("context_*.json")):
                    resolved_context = candidate
                    break

    missing = []
    if resolved_train is None or not resolved_train.exists():
        missing.append("train.json")
    if resolved_public is None or not resolved_public.exists():
        missing.append("public-official.json")
    if resolved_context is None or not resolved_context.exists():
        missing.append("selected-contexts directory")
    if missing:
        raise FileNotFoundError(
            "Cannot locate required input(s): "
            + ", ".join(missing)
            + ". Pass --input-root or explicit --train-file/--public-file/--context-dir."
        )

    return resolved_train, resolved_public, resolved_context


def validate_query_file(
    payload: Any,
    *,
    file_kind: str,
    require_answers: bool,
    context_ids: set[str] | None = None,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    stats: dict[str, Any] = {
        "num_queries": 0,
        "num_duplicate_questions": 0,
        "answer_count_distribution": {},
        "num_missing_gold_ids": 0,
    }
    if not isinstance(payload, dict):
        issues.append(
            ValidationIssue("error", "schema", file_kind, "Root must be a JSON object.")
        )
        return issues, stats

    question_to_ids: dict[str, list[str]] = defaultdict(list)
    answer_counts: Counter[int] = Counter()
    missing_gold = 0

    for qid, row in payload.items():
        str_qid = str(qid)
        if not isinstance(row, dict):
            issues.append(
                ValidationIssue("error", "schema", str_qid, "Query row must be an object.")
            )
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            issues.append(
                ValidationIssue(
                    "error", "schema", str_qid, "`question` must be a non-empty string."
                )
            )
        else:
            question_to_ids[re.sub(r"\s+", " ", question.strip())].append(str_qid)

        if require_answers:
            if not isinstance(answer, list) or not answer:
                issues.append(
                    ValidationIssue(
                        "error", "schema", str_qid, "`answer` must be a non-empty list."
                    )
                )
                continue
            normalized_answer: list[str] = []
            for doc_id in answer:
                if not isinstance(doc_id, (str, int)):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "schema",
                            str_qid,
                            "Every answer ID must be a string or integer.",
                        )
                    )
                    continue
                normalized_answer.append(str(doc_id))
            if len(normalized_answer) != len(set(normalized_answer)):
                issues.append(
                    ValidationIssue(
                        "warning", "duplicate_gold", str_qid, "Duplicate gold IDs in answer."
                    )
                )
            if len(normalized_answer) > MAX_SUBMISSION_DOCS:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "answer_count",
                        str_qid,
                        f"Gold answer has more than {MAX_SUBMISSION_DOCS} IDs.",
                    )
                )
            answer_counts[len(normalized_answer)] += 1
            if context_ids is not None:
                for doc_id in normalized_answer:
                    if doc_id not in context_ids:
                        missing_gold += 1
                        issues.append(
                            ValidationIssue(
                                "error",
                                "missing_gold",
                                str_qid,
                                f"Gold context ID {doc_id} is missing from corpus.",
                            )
                        )
        else:
            if answer is not None:
                issues.append(
                    ValidationIssue(
                        "warning", "schema", str_qid, "Public answer should be null."
                    )
                )
            answer_counts[0] += 1

    duplicate_questions = {q: ids for q, ids in question_to_ids.items() if len(ids) > 1}
    for question, ids in duplicate_questions.items():
        issues.append(
            ValidationIssue(
                "warning",
                "duplicate_question",
                ",".join(ids[:10]),
                f"Duplicate question text appears {len(ids)} times: {question[:120]}",
            )
        )

    stats.update(
        {
            "num_queries": len(payload),
            "num_duplicate_questions": len(duplicate_questions),
            "answer_count_distribution": dict(sorted(answer_counts.items())),
            "num_missing_gold_ids": missing_gold,
        }
    )
    return issues, stats


def fallback_name_from_link(link: str) -> str:
    if not link:
        return ""
    decoded = unquote(link)
    stem = decoded.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.aspx?$", "", stem, flags=re.IGNORECASE)
    return stem.strip()


def fallback_name_from_passage(passage: str) -> str:
    for line in normalize_newlines(passage).splitlines():
        candidate = re.sub(r"\s+", " ", line).strip()
        if 12 <= len(candidate) <= 220 and not re.fullmatch(r"[\W\d_]+", candidate):
            return candidate
    return ""


def load_contexts(context_dir: Path) -> tuple[list[ContextRecord], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    records: list[ContextRecord] = []
    seen_doc_ids: set[str] = set()

    for path in sorted(context_dir.glob("context_*.json")):
        try:
            row = read_json(path)
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ValidationIssue("error", "context_json", str(path), f"Cannot read JSON: {exc}")
            )
            continue
        if not isinstance(row, dict):
            issues.append(
                ValidationIssue("error", "context_schema", str(path), "Context must be object.")
            )
            continue

        raw_id = row.get("id")
        doc_id = str(raw_id) if isinstance(raw_id, (str, int)) else ""
        if not doc_id:
            match = re.search(r"context_(\d+)\.json$", path.name)
            doc_id = match.group(1) if match else path.stem
            issues.append(
                ValidationIssue(
                    "warning", "context_schema", str(path), "Missing `id`; using filename."
                )
            )
        if doc_id in seen_doc_ids:
            issues.append(
                ValidationIssue("error", "duplicate_context_id", doc_id, str(path))
            )
        seen_doc_ids.add(doc_id)

        passage = row.get("passage")
        has_empty_passage = not isinstance(passage, str) or not passage.strip()
        if has_empty_passage:
            issues.append(
                ValidationIssue(
                    "warning",
                    "empty_passage_fallback",
                    doc_id,
                    "`passage` is empty; using inferred name/link as fallback text.",
                )
            )
            passage = ""

        link = row.get("link")
        if not isinstance(link, str):
            issues.append(
                ValidationIssue("warning", "context_schema", doc_id, "`link` is missing.")
            )
            link = ""

        name = row.get("name")
        name_source = "name"
        if not isinstance(name, str) or not name.strip():
            name = fallback_name_from_link(link)
            name_source = "link"
        if not name:
            name = fallback_name_from_passage(passage)
            name_source = "passage"
        if not name:
            name = f"context_{doc_id}"
            name_source = "id"
            issues.append(
                ValidationIssue(
                    "warning", "context_name", doc_id, "Cannot infer name; using context ID."
                )
            )

        if has_empty_passage:
            passage = "\n".join(part for part in [name, link] if part).strip()

        records.append(
            ContextRecord(
                doc_id=doc_id,
                source_path=str(path),
                link=link,
                name=name.strip(),
                name_source=name_source,
                passage=passage,
                is_empty_passage_fallback=has_empty_passage,
            )
        )

    if not records:
        issues.append(
            ValidationIssue("error", "context_dir", str(context_dir), "No context_*.json files.")
        )
    return records, issues


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


HEADER_PATTERNS = [
    r"^\s*CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\s*$",
    r"^\s*Độc\s+lập\s*[-–]\s*Tự\s+do\s*[-–]\s*Hạnh\s+phúc\s*$",
    r"^\s*Số\s*:\s*[\w./-]+.*$",
    r"^\s*(Hà\s+Nội|TP\.?\s*Hồ\s+Chí\s+Minh|Thành\s+phố\s+Hồ\s+Chí\s+Minh),?\s+ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}.*$",
]
HEADER_RE = re.compile("|".join(f"(?:{p})" for p in HEADER_PATTERNS), re.IGNORECASE)
DIVIDER_RE = re.compile(r"^\s*[-=_.*•\s]{5,}\s*$")
FOOTER_RE = re.compile(
    r"\n\s*(Nơi\s+nhận\s*:|KT\.\s*[^:\n]{0,80}\n|TM\.\s*[^:\n]{0,80}\n)",
    re.IGNORECASE,
)


def clean_legal_text(text: str) -> tuple[str, dict[str, int]]:
    original_len = len(text)
    text = normalize_newlines(text)
    text = text.replace("\ufeff", "").replace("\u200b", "")

    lines = []
    removed_header_lines = 0
    removed_divider_lines = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if HEADER_RE.match(line):
            removed_header_lines += 1
            continue
        if DIVIDER_RE.match(line):
            removed_divider_lines += 1
            continue
        lines.append(raw_line.rstrip())
    text = "\n".join(lines)

    footer_removed_chars = 0
    footer_match = FOOTER_RE.search(text)
    if footer_match and footer_match.start() > len(text) * 0.45:
        footer_removed_chars = len(text) - footer_match.start()
        text = text[: footer_match.start()]

    text = re.sub(r"([A-Za-zÀ-ỹ])- *\n+ *([A-Za-zÀ-ỹ])", r"\1\2", text)
    text = re.sub(r"(?<![.!?:;])\n[ \t]*(?=[a-zà-ỹ])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = text.strip()

    stats = {
        "original_chars": original_len,
        "clean_chars": len(text),
        "removed_chars": max(original_len - len(text), 0),
        "removed_header_lines": removed_header_lines,
        "removed_divider_lines": removed_divider_lines,
        "footer_removed_chars": footer_removed_chars,
    }
    return text, stats


DOC_TYPE_PATTERNS = [
    ("luat", r"\b(luật|bo luat|bộ luật)\b"),
    ("nghi_dinh", r"\bnghị\s+định\b"),
    ("thong_tu", r"\bthông\s+tư\b"),
    ("quyet_dinh", r"\bquyết\s+định\b"),
    ("nghi_quyet", r"\bnghị\s+quyết\b"),
    ("qcvn", r"\bqcvn\b|quy\s+chuẩn"),
    ("tcvn", r"\btcvn\b|tiêu\s+chuẩn"),
    ("cong_van", r"\bcông\s+văn\b"),
]


def extract_metadata(name: str, link: str, text: str) -> dict[str, Any]:
    haystack = f"{name}\n{link}\n{text[:4000]}"
    haystack_norm = normalize_for_match(haystack)

    doc_type = "unknown"
    for label, pattern in DOC_TYPE_PATTERNS:
        if re.search(pattern, haystack_norm, flags=re.IGNORECASE):
            doc_type = label
            break

    years = sorted(set(re.findall(r"\b(19\d{2}|20\d{2})\b", haystack)))
    number_match = re.search(
        r"\b\d{1,4}/\d{4}/[A-ZĐA-Z0-9.-]+|\b\d{1,5}/[A-ZĐA-Z0-9.-]+|\b\d{1,5}\s*[-/]\s*\d{4}\b",
        haystack,
        flags=re.IGNORECASE,
    )
    normalized = normalize_for_match(haystack)
    flags = {
        "has_amendment_signal": any(
            kw in normalized
            for kw in [
                "sua doi",
                "bo sung",
                "thay the",
                "bai bo",
                "het hieu luc",
                "dinh chinh",
            ]
        ),
        "has_repeal_signal": any(
            kw in normalized for kw in ["bai bo", "het hieu luc", "ngung hieu luc"]
        ),
    }
    return {
        "doc_type": doc_type,
        "issue_number": number_match.group(0).strip() if number_match else "",
        "years": years,
        **flags,
    }


BOUNDARY_RE = re.compile(
    r"(?im)^(?P<title>\s*(?:"
    r"Chương\s+[IVXLCDM\d]+[^\n]{0,180}|"
    r"Mục\s+\d+[^\n]{0,180}|"
    r"Điều\s+\d+[a-zA-Z]?\s*[.:]?[^\n]{0,220}|"
    r"Khoản\s+\d+[^\n]{0,180}|"
    r"Điểm\s+[a-zđ]\s*[).][^\n]{0,180}|"
    r"Phụ\s*lục\s+[A-Z\dIVXLCDM]*[^\n]{0,180}"
    r"))\s*$"
)


def update_heading_stack(title: str, stack: dict[str, str]) -> None:
    normalized = normalize_for_match(title)
    if normalized.startswith("chuong"):
        stack["chapter"] = title
        stack["section"] = ""
        stack["article"] = ""
        stack["clause"] = ""
        stack["point"] = ""
    elif normalized.startswith("muc"):
        stack["section"] = title
        stack["article"] = ""
        stack["clause"] = ""
        stack["point"] = ""
    elif normalized.startswith("dieu"):
        stack["article"] = title
        stack["clause"] = ""
        stack["point"] = ""
    elif normalized.startswith("khoan"):
        stack["clause"] = title
        stack["point"] = ""
    elif normalized.startswith("diem"):
        stack["point"] = title
    elif normalized.startswith("phu luc"):
        stack["appendix"] = title
        stack["chapter"] = ""
        stack["section"] = ""
        stack["article"] = ""
        stack["clause"] = ""
        stack["point"] = ""


def split_structural_sections(text: str) -> list[dict[str, Any]]:
    matches = list(BOUNDARY_RE.finditer(text))
    if not matches:
        return [
            {
                "text": text,
                "heading": "",
                "structure": {
                    "chapter": "",
                    "section": "",
                    "article": "",
                    "clause": "",
                    "point": "",
                    "appendix": "",
                },
            }
        ]

    sections: list[dict[str, Any]] = []
    stack = {
        "chapter": "",
        "section": "",
        "article": "",
        "clause": "",
        "point": "",
        "appendix": "",
    }

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                {"text": preamble, "heading": "", "structure": dict(stack)}
            )

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = re.sub(r"\s+", " ", match.group("title")).strip()
        update_heading_stack(title, stack)
        section_text = text[start:end].strip()
        if section_text:
            sections.append(
                {
                    "text": section_text,
                    "heading": title,
                    "structure": dict(stack),
                }
            )
    return sections


def make_windows(words: list[str], window: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    if not words:
        return
    if len(words) <= window:
        yield 0, len(words), " ".join(words)
        return
    step = max(window - overlap, 1)
    start = 0
    while start < len(words):
        end = min(start + window, len(words))
        yield start, end, " ".join(words[start:end])
        if end == len(words):
            break
        start += step


def chunk_context(record: ContextRecord, config: ChunkConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_text, clean_stats = clean_legal_text(record.passage)
    metadata = extract_metadata(record.name, record.link, clean_text)
    metadata["is_empty_passage_fallback"] = record.is_empty_passage_fallback
    clean_checksum = sha1_text(clean_text)
    raw_checksum = sha1_text(record.passage)
    sections = split_structural_sections(clean_text)
    chunks: list[dict[str, Any]] = []

    for section_idx, section in enumerate(sections):
        section_text = section["text"].strip()
        words = re.findall(r"\S+", section_text)
        use_sliding = len(words) > config.long_section_words or not section["heading"]
        windows = (
            make_windows(words, config.window_words, config.overlap_words)
            if use_sliding
            else [(0, len(words), section_text)]
        )
        for window_idx, (start_word, end_word, chunk_body) in enumerate(windows):
            heading_parts = [
                record.name,
                section["structure"].get("appendix", ""),
                section["structure"].get("chapter", ""),
                section["structure"].get("section", ""),
                section["structure"].get("article", ""),
                section["structure"].get("clause", ""),
                section["structure"].get("point", ""),
            ]
            heading = " | ".join(part for part in heading_parts if part)
            chunk_text = f"{heading}\n{chunk_body}".strip() if heading else chunk_body
            chunk_key = (
                f"{record.doc_id}:{section_idx}:{window_idx}:{start_word}:{end_word}:"
                f"{clean_checksum}"
            )
            chunks.append(
                {
                    "chunk_id": sha1_text(chunk_key)[:20],
                    "doc_id": record.doc_id,
                    "section_idx": section_idx,
                    "window_idx": window_idx,
                    "start_word": start_word,
                    "end_word": end_word,
                    "word_count": word_count(chunk_body),
                    "heading": heading,
                    "text": chunk_text,
                    "checksum": sha1_text(chunk_text),
                    "structure": section["structure"],
                    "metadata": metadata,
                }
            )

    cleaned_record = {
        "doc_id": record.doc_id,
        "source_path": record.source_path,
        "link": record.link,
        "name": record.name,
        "name_source": record.name_source,
        "raw_checksum": raw_checksum,
        "clean_checksum": clean_checksum,
        "raw_word_count": word_count(record.passage),
        "clean_word_count": word_count(clean_text),
        "cleaning": clean_stats,
        "metadata": metadata,
        "clean_text": clean_text,
    }
    return chunks, cleaned_record


def deterministic_stratified_split(
    train_payload: dict[str, Any],
    *,
    dev_size: int = 1000,
    seed: int = 42,
) -> tuple[list[str], list[str], dict[str, Any]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for qid, row in train_payload.items():
        answer = row.get("answer") if isinstance(row, dict) else []
        groups[len(answer) if isinstance(answer, list) else 0].append(str(qid))

    rng = random.Random(seed)
    for ids in groups.values():
        ids.sort()
        rng.shuffle(ids)

    total = sum(len(ids) for ids in groups.values())
    if dev_size >= total:
        raise ValueError("dev_size must be smaller than train size.")

    dev_ids: list[str] = []
    fractions: list[tuple[float, int, int]] = []
    for answer_count, ids in groups.items():
        exact = len(ids) * dev_size / total
        take = int(exact)
        dev_ids.extend(ids[:take])
        fractions.append((exact - take, answer_count, len(ids)))

    remaining = dev_size - len(dev_ids)
    for _, answer_count, _ in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        current_taken = sum(1 for qid in dev_ids if qid in set(groups[answer_count]))
        if current_taken < len(groups[answer_count]):
            dev_ids.append(groups[answer_count][current_taken])
            remaining -= 1

    dev_set = set(dev_ids)
    train_ids = sorted([str(qid) for qid in train_payload if str(qid) not in dev_set])
    dev_ids = sorted(dev_ids)
    stats = {
        "seed": seed,
        "train_size": len(train_ids),
        "dev_size": len(dev_ids),
        "strata": {
            str(k): {
                "total": len(v),
                "train": sum(1 for qid in v if qid not in dev_set),
                "dev": sum(1 for qid in v if qid in dev_set),
            }
            for k, v in sorted(groups.items())
        },
    }
    return train_ids, dev_ids, stats


def subset_queries(payload: dict[str, Any], ids: Iterable[str]) -> dict[str, Any]:
    return {qid: payload[qid] for qid in ids}


def precision_recall_for_query(
    prediction: list[str],
    gold: list[str],
    *,
    max_docs: int = MAX_SUBMISSION_DOCS,
) -> dict[str, float]:
    if len(prediction) > max_docs or not prediction:
        return {"precision": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    pred = [str(x) for x in prediction]
    gold_set = {str(x) for x in gold}
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    hits = [doc_id for doc_id in pred if doc_id in gold_set]
    first_rank = next((idx + 1 for idx, doc_id in enumerate(pred) if doc_id in gold_set), None)
    return {
        "precision": len(hits) / len(pred),
        "recall": len(set(hits)) / len(gold_set),
        "hit": 1.0 if hits else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def evaluate_predictions(
    predictions: dict[str, list[str]],
    gold_payload: dict[str, Any],
    *,
    max_docs: int = MAX_SUBMISSION_DOCS,
) -> dict[str, Any]:
    per_query = {}
    sums = Counter()
    n = 0
    for qid, row in gold_payload.items():
        gold = row.get("answer", []) if isinstance(row, dict) else []
        pred = predictions.get(str(qid), [])
        metrics = precision_recall_for_query(pred, gold, max_docs=max_docs)
        per_query[str(qid)] = metrics
        sums.update(metrics)
        n += 1
    macro = {key: (value / n if n else 0.0) for key, value in sums.items()}
    return {"macro": macro, "per_query": per_query, "num_queries": n}


def validate_submission(
    predictions: Any,
    public_payload: dict[str, Any],
    context_ids: set[str],
    *,
    max_docs: int = MAX_SUBMISSION_DOCS,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    if not isinstance(predictions, dict):
        return [
            ValidationIssue("error", "submission_schema", "root", "Submission must be object.")
        ], {}

    public_ids = {str(qid) for qid in public_payload}
    pred_ids = {str(qid) for qid in predictions}
    missing = sorted(public_ids - pred_ids)
    extra = sorted(pred_ids - public_ids)
    for qid in missing[:20]:
        issues.append(
            ValidationIssue("error", "submission_missing_query", qid, "Missing public query.")
        )
    for qid in extra[:20]:
        issues.append(
            ValidationIssue("warning", "submission_extra_query", qid, "Query not in public set.")
        )

    answer_lengths: Counter[int] = Counter()
    for qid, answer in predictions.items():
        str_qid = str(qid)
        if not isinstance(answer, list):
            issues.append(
                ValidationIssue("error", "submission_schema", str_qid, "Answer must be list.")
            )
            continue
        answer_lengths[len(answer)] += 1
        if not (1 <= len(answer) <= max_docs):
            issues.append(
                ValidationIssue(
                    "error",
                    "submission_answer_count",
                    str_qid,
                    f"Answer must contain 1-{max_docs} IDs.",
                )
            )
        normalized = [str(x) for x in answer]
        if any(not isinstance(x, str) for x in answer):
            issues.append(
                ValidationIssue(
                    "warning",
                    "submission_id_type",
                    str_qid,
                    "Answer IDs should be strings.",
                )
            )
        if len(normalized) != len(set(normalized)):
            issues.append(
                ValidationIssue("error", "submission_duplicate_id", str_qid, "Duplicate IDs.")
            )
        for doc_id in normalized:
            if doc_id not in context_ids:
                issues.append(
                    ValidationIssue(
                        "error",
                        "submission_unknown_context",
                        str_qid,
                        f"Unknown context ID {doc_id}.",
                    )
                )

    stats = {
        "num_public_queries": len(public_ids),
        "num_prediction_queries": len(pred_ids),
        "num_missing_queries": len(missing),
        "num_extra_queries": len(extra),
        "answer_length_distribution": dict(sorted(answer_lengths.items())),
        "num_errors": sum(issue.severity == "error" for issue in issues),
        "num_warnings": sum(issue.severity == "warning" for issue in issues),
    }
    return issues, stats


def summarize_chunks(chunks: list[dict[str, Any]], config: ChunkConfig) -> dict[str, Any]:
    by_doc = Counter(chunk["doc_id"] for chunk in chunks)
    lengths = [int(chunk["word_count"]) for chunk in chunks]
    warned_docs = {
        doc_id: count
        for doc_id, count in by_doc.items()
        if count > config.max_chunk_warning_per_context
    }
    return {
        "num_chunks": len(chunks),
        "num_docs_with_chunks": len(by_doc),
        "chunks_per_doc": {
            "min": min(by_doc.values()) if by_doc else 0,
            "max": max(by_doc.values()) if by_doc else 0,
            "mean": statistics.fmean(by_doc.values()) if by_doc else 0.0,
        },
        "chunk_word_count": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "median": statistics.median(lengths) if lengths else 0,
        },
        "num_docs_over_chunk_warning": len(warned_docs),
        "docs_over_chunk_warning": dict(sorted(warned_docs.items(), key=lambda x: -x[1])[:50]),
    }


def run_step1(args: argparse.Namespace) -> dict[str, Any]:
    train_path, public_path, context_dir = find_input_paths(
        args.input_root, args.train_file, args.public_file, args.context_dir
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ChunkConfig(
        window_words=args.chunk_window,
        overlap_words=args.chunk_overlap,
        long_section_words=args.long_section_words,
        max_chunk_warning_per_context=args.max_chunk_warning_per_context,
    )

    train_payload = read_json(train_path)
    public_payload = read_json(public_path)
    contexts, context_issues = load_contexts(context_dir)
    context_ids = {record.doc_id for record in contexts}

    train_issues, train_stats = validate_query_file(
        train_payload, file_kind="train", require_answers=True, context_ids=context_ids
    )
    public_issues, public_stats = validate_query_file(
        public_payload, file_kind="public", require_answers=False
    )

    if not isinstance(train_payload, dict):
        raise ValueError("train.json root must be object.")
    train_ids, dev_ids, split_stats = deterministic_stratified_split(
        train_payload, dev_size=args.dev_size, seed=args.seed
    )
    write_json(output_dir / "splits" / "train_ids.json", train_ids)
    write_json(output_dir / "splits" / "dev_ids.json", dev_ids)
    write_json(output_dir / "splits" / "train_split.json", subset_queries(train_payload, train_ids))
    write_json(output_dir / "splits" / "dev_split.json", subset_queries(train_payload, dev_ids))

    cleaned_rows: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    for idx, record in enumerate(contexts, start=1):
        chunks, cleaned = chunk_context(record, config)
        cleaned_rows.append(cleaned)
        all_chunks.extend(chunks)
        if args.limit_contexts and idx >= args.limit_contexts:
            break

    append_jsonl(output_dir / "corpus" / "cleaned_contexts.jsonl", cleaned_rows)
    append_jsonl(output_dir / "corpus" / "chunks.jsonl", all_chunks)

    report = {
        "inputs": {
            "train_path": str(train_path),
            "public_path": str(public_path),
            "context_dir": str(context_dir),
            "output_dir": str(output_dir),
        },
        "config": asdict(config),
        "train": train_stats,
        "public": public_stats,
        "contexts": {
            "num_context_files": len(contexts),
            "num_context_ids": len(context_ids),
            "num_missing_names": sum(1 for record in contexts if record.name_source != "name"),
        },
        "split": split_stats,
        "chunks": summarize_chunks(all_chunks, config),
        "issues": [asdict(issue) for issue in context_issues + train_issues + public_issues],
    }
    report["summary"] = {
        "num_errors": sum(issue["severity"] == "error" for issue in report["issues"]),
        "num_warnings": sum(issue["severity"] == "warning" for issue in report["issues"]),
    }
    write_json(output_dir / "reports" / "validation_report.json", report)
    write_json(
        output_dir / "corpus" / "chunk_manifest.json",
        {
            "config": asdict(config),
            "num_chunks": len(all_chunks),
            "cleaned_contexts_jsonl": "corpus/cleaned_contexts.jsonl",
            "chunks_jsonl": "corpus/chunks.jsonl",
            "validation_report": "reports/validation_report.json",
        },
    )

    if args.prediction_file:
        predictions = read_json(args.prediction_file)
        sub_issues, sub_stats = validate_submission(predictions, public_payload, context_ids)
        write_json(
            output_dir / "reports" / "submission_validation.json",
            {"stats": sub_stats, "issues": [asdict(issue) for issue in sub_issues]},
        )

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--public-file", type=Path, default=None)
    parser.add_argument("--context-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--chunk-window", type=int, default=320)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument("--long-section-words", type=int, default=900)
    parser.add_argument("--max-chunk-warning-per-context", type=int, default=420)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=0,
        help="Debug only: process the first N contexts after validation/split.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_step1(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
