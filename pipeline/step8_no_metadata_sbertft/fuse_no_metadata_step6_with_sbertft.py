from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path("task1/pipeline/step8_no_metadata_sbertft")
OUTPUT_DIR = ROOT / "outputs"
PUBLIC_FILE = Path("task1/public-official.json")
NO_METADATA_RANKINGS = Path(
    "task1/pipeline/step8_no_metadata/step8_no_metadata_step6/"
    "rankings/aiteamvn_vietnamese_reranker_finetuned_no_metadata/"
    "public_rankings_step6_fused.jsonl"
)
NO_METADATA_SUBMISSION = Path(
    "task1/pipeline/step8_no_metadata/step8_no_metadata_step6/"
    "submission/aiteamvn_vietnamese_reranker_finetuned_no_metadata/"
    "submission.json"
)
SBERT_RANKED_CONTEXTS = Path(
    "task1/pipeline/step6+sbertft/step6+sbertft/"
    "sbertft/public_submission/public_ranked_contexts.csv"
)
SBERT_SUBMISSION = Path(
    "task1/pipeline/step6+sbertft/step6+sbertft/"
    "sbertft/public_submission/submission.json"
)
BASELINECUR_SUBMISSION = Path("task1/pipeline/baselinecur/submission.json")

MAX_SUBMISSION_DOCS = 5
RRF_K = 60
CANDIDATES = {
    "nm_sbert0p25": {"step_weight": 1.0, "sbert_weight": 0.25},
    "nm_sbert0p40": {"step_weight": 1.0, "sbert_weight": 0.40},
    "nm_sbert0p50": {"step_weight": 1.0, "sbert_weight": 0.50},
    "nm_sbert0p55": {"step_weight": 1.0, "sbert_weight": 0.55},
    "nm_sbert0p60": {"step_weight": 1.0, "sbert_weight": 0.60},
    "nm_sbert0p65": {"step_weight": 1.0, "sbert_weight": 0.65},
    "nm_sbert0p70": {"step_weight": 1.0, "sbert_weight": 0.70},
    "nm_sbert0p75": {"step_weight": 1.0, "sbert_weight": 0.75},
    "nm_sbert0p80": {"step_weight": 1.0, "sbert_weight": 0.80},
    "nm_sbert0p85": {"step_weight": 1.0, "sbert_weight": 0.85},
    "nm_sbert0p90": {"step_weight": 1.0, "sbert_weight": 0.90},
    "nm_sbert0p95": {"step_weight": 1.0, "sbert_weight": 0.95},
    "nm_sbert1p00": {"step_weight": 1.0, "sbert_weight": 1.00},
    "nm_sbert1p20": {"step_weight": 1.0, "sbert_weight": 1.20},
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_public_qids(path: Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, dict):
        rows = data.get("questions") if isinstance(data.get("questions"), list) else data
        if isinstance(rows, list):
            return [str(row.get("question_id") or row.get("id") or row.get("qid")) for row in rows]
        return [str(qid) for qid in data]
    if isinstance(data, list):
        return [str(row.get("question_id") or row.get("id") or row.get("qid")) for row in data]
    raise TypeError(f"Unsupported public file schema: {path}")


def load_json_submission(path: Path) -> dict[str, list[str]]:
    data = read_json(path)
    return {str(qid): [str(doc_id) for doc_id in row["answer"]] for qid, row in data.items()}


def load_step_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            docs = row.get("fused_doc_ids") or row.get("reranked_doc_ids") or row.get("base_doc_ids") or []
            rankings[qid] = [str(doc_id) for doc_id in docs]
    return rankings


def load_sbert_rankings(csv_path: Path, submission_path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"])
            raw_docs = row.get("top20_rerank") or row.get("answer") or "[]"
            docs = json.loads(raw_docs)
            rankings[qid] = [str(doc_id) for doc_id in docs]

    submission = load_json_submission(submission_path)
    for qid, docs in submission.items():
        if qid not in rankings:
            rankings[qid] = docs
            continue
        seen = set(rankings[qid])
        rankings[qid].extend(doc_id for doc_id in docs if doc_id not in seen)
    return rankings


def weighted_rrf(left_docs: list[str], right_docs: list[str], *, left_weight: float, right_weight: float) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for branch_offset, (docs, weight) in enumerate(((left_docs, left_weight), (right_docs, right_weight))):
        for rank, doc_id in enumerate(docs, start=1):
            first_seen.setdefault(doc_id, branch_offset * 1_000_000 + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))]


def build_submission(
    qids: list[str],
    no_metadata_rankings: dict[str, list[str]],
    sbert_rankings: dict[str, list[str]],
    *,
    step_weight: float,
    sbert_weight: float,
) -> dict[str, dict[str, list[str]]]:
    submission: dict[str, dict[str, list[str]]] = {}
    for qid in qids:
        fused = weighted_rrf(
            no_metadata_rankings.get(qid, []),
            sbert_rankings.get(qid, []),
            left_weight=step_weight,
            right_weight=sbert_weight,
        )
        submission[qid] = {"answer": fused[:MAX_SUBMISSION_DOCS]}
    return submission


def validate_submission(submission: dict[str, dict[str, list[str]]], qids: list[str]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected = set(qids)
    actual = set(submission)
    for qid in sorted(expected - actual)[:20]:
        issues.append({"kind": "missing_query", "query_id": qid})
    for qid in sorted(actual - expected)[:20]:
        issues.append({"kind": "extra_query", "query_id": qid})

    answer_lengths: dict[int, int] = {}
    for qid in qids:
        row = submission.get(qid)
        answer = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(answer, list):
            issues.append({"kind": "answer_not_array", "query_id": qid})
            continue
        answer_lengths[len(answer)] = answer_lengths.get(len(answer), 0) + 1
        if not (1 <= len(answer) <= MAX_SUBMISSION_DOCS):
            issues.append({"kind": "bad_answer_length", "query_id": qid, "length": len(answer)})
        if len(answer) != len(set(answer)):
            issues.append({"kind": "duplicate_doc_id", "query_id": qid})
        for doc_id in answer:
            if not isinstance(doc_id, str):
                issues.append({"kind": "doc_id_not_string", "query_id": qid, "doc_id": repr(doc_id)})
    return {
        "num_public_queries": len(qids),
        "num_submission_queries": len(submission),
        "answer_length_distribution": dict(sorted(answer_lengths.items())),
        "num_errors": len(issues),
        "issues": issues,
    }


def zip_submission(submission_json: Path, submission_zip: Path) -> None:
    submission_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="submission.json")
    with zipfile.ZipFile(submission_zip, "r") as zf:
        assert zf.namelist() == ["submission.json"], zf.namelist()


def top5_overlap(a: dict[str, list[str]], b: dict[str, list[str]], qids: list[str]) -> float:
    return sum(len(set(a.get(qid, [])[:5]) & set(b.get(qid, [])[:5])) for qid in qids) / len(qids)


def changed_count(a: dict[str, list[str]], b: dict[str, list[str]], qids: list[str]) -> int:
    return sum(a.get(qid, [])[:5] != b.get(qid, [])[:5] for qid in qids)


def main() -> None:
    for path in [
        PUBLIC_FILE,
        NO_METADATA_RANKINGS,
        NO_METADATA_SUBMISSION,
        SBERT_RANKED_CONTEXTS,
        SBERT_SUBMISSION,
        BASELINECUR_SUBMISSION,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    qids = load_public_qids(PUBLIC_FILE)
    no_metadata_rankings = load_step_rankings(NO_METADATA_RANKINGS)
    no_metadata_submission = load_json_submission(NO_METADATA_SUBMISSION)
    sbert_rankings = load_sbert_rankings(SBERT_RANKED_CONTEXTS, SBERT_SUBMISSION)
    sbert_submission = load_json_submission(SBERT_SUBMISSION)
    baselinecur_submission = load_json_submission(BASELINECUR_SUBMISSION)

    audit_rows: list[dict[str, Any]] = []
    for slug, cfg in CANDIDATES.items():
        submission = build_submission(qids, no_metadata_rankings, sbert_rankings, **cfg)
        validation = validate_submission(submission, qids)
        candidate_dir = OUTPUT_DIR / "candidates" / slug
        submission_json = candidate_dir / "submission.json"
        submission_zip = candidate_dir / "submission.zip"
        already_exists = (
            submission_json.exists()
            and submission_zip.exists()
            and (candidate_dir / "submission_validation.json").exists()
        )
        if not already_exists:
            write_json(submission_json, submission)
            write_json(candidate_dir / "submission_validation.json", validation)
            zip_submission(submission_json, submission_zip)

        submission_rankings = {qid: row["answer"] for qid, row in submission.items()}
        audit_rows.append(
            {
                "candidate": slug,
                "method": "weighted_rrf_no_metadata_step6_plus_sbertft_public_ranking",
                "rrf_k": RRF_K,
                "step_weight": cfg["step_weight"],
                "sbert_weight": cfg["sbert_weight"],
                "validation": validation,
                "already_existed_before_this_run": already_exists,
                "changed_queries_vs_no_metadata_step6": changed_count(
                    submission_rankings, no_metadata_submission, qids
                ),
                "changed_queries_vs_sbert": changed_count(submission_rankings, sbert_submission, qids),
                "changed_queries_vs_baselinecur": changed_count(submission_rankings, baselinecur_submission, qids),
                "avg_top5_overlap_no_metadata_step6": top5_overlap(
                    submission_rankings, no_metadata_submission, qids
                ),
                "avg_top5_overlap_sbert": top5_overlap(submission_rankings, sbert_submission, qids),
                "avg_top5_overlap_baselinecur": top5_overlap(submission_rankings, baselinecur_submission, qids),
                "submission_zip": str(submission_zip),
            }
        )

    report = {
        "status": "ok",
        "note": (
            "CPU-only fusion trial: replace baselinecur Step6 branch with Step8 no-metadata "
            "Step6 branch, keep the same SBERT-FT public ranking."
        ),
        "inputs": {
            "public_file": str(PUBLIC_FILE),
            "no_metadata_rankings": str(NO_METADATA_RANKINGS),
            "no_metadata_submission": str(NO_METADATA_SUBMISSION),
            "sbert_ranked_contexts": str(SBERT_RANKED_CONTEXTS),
            "sbert_submission": str(SBERT_SUBMISSION),
            "baselinecur_submission": str(BASELINECUR_SUBMISSION),
        },
        "baseline_public_scores": {
            "baselinecur_rrf_sbert0p60": {"precision": 0.19740000000000005, "recall": 0.9173333333333332},
            "step8_no_metadata_step6": {"precision": 0.18940000000000004, "recall": 0.8825},
            "nm_sbert0p60": {"precision": 0.19660000000000002, "recall": 0.9146666666666666},
            "sbertft_standalone": {"precision": 0.19100000000000003, "recall": 0.8905833333333334},
        },
        "candidate_audit": audit_rows,
    }
    write_json(OUTPUT_DIR / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
