from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        item = str(item)
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def public_qids(path: Path) -> list[str]:
    obj = read_json(path)
    if isinstance(obj, dict):
        return [str(x) for x in obj.keys()]
    if isinstance(obj, list):
        return [str(item.get("id", item.get("qid", idx))) for idx, item in enumerate(obj)]
    raise TypeError(type(obj))


def load_step5_rankings(path: Path) -> dict[str, dict[str, Any]]:
    rankings: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = str(row["query_id"])
        rankings[qid] = {
            "question": str(row.get("question", "")),
            "docs": unique_keep_order([str(x) for x in row["fused_doc_ids"]]),
        }
    return rankings


def load_sbert_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"])
            if row.get("top20_rerank"):
                docs = json.loads(row["top20_rerank"])
            elif row.get("answer"):
                docs = json.loads(row["answer"])
            else:
                docs = []
            rankings[qid] = unique_keep_order([str(x) for x in docs])
    return rankings


def rrf_fuse(
    step5_docs: list[str],
    sbert_docs: list[str],
    *,
    step5_weight: float,
    sbert_weight: float,
    rrf_k: int,
    depth: int,
) -> list[str]:
    scores: dict[str, float] = {}
    tie_rank: dict[str, int] = {}
    for rank, doc_id in enumerate(step5_docs[:depth], start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + step5_weight / (rrf_k + rank)
        tie_rank.setdefault(doc_id, rank)
    for rank, doc_id in enumerate(sbert_docs[:depth], start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + sbert_weight / (rrf_k + rank)
        tie_rank.setdefault(doc_id, depth + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], tie_rank[doc_id], doc_id))


def make_candidate(
    qids: list[str],
    step5: dict[str, dict[str, Any]],
    sbert: dict[str, list[str]],
    *,
    step5_weight: float,
    sbert_weight: float,
    rrf_k: int,
    depth: int,
    lock_top_n: int,
) -> dict[str, dict[str, list[str]]]:
    submission: dict[str, dict[str, list[str]]] = {}
    for qid in qids:
        step5_docs = step5[qid]["docs"]
        sbert_docs = sbert.get(qid, [])
        fused_docs = rrf_fuse(
            step5_docs,
            sbert_docs,
            step5_weight=step5_weight,
            sbert_weight=sbert_weight,
            rrf_k=rrf_k,
            depth=depth,
        )
        locked = step5_docs[:lock_top_n]
        final_docs = unique_keep_order(locked + fused_docs + step5_docs)
        submission[qid] = {"answer": final_docs[:5]}
    return submission


def validate_submission(submission: dict[str, Any], qids: list[str]) -> dict[str, Any]:
    issues = []
    length_dist: dict[str, int] = {}
    qid_set = set(qids)
    sub_set = set(submission.keys())
    if qid_set != sub_set:
        issues.append(
            {
                "type": "query_id_mismatch",
                "missing": sorted(qid_set - sub_set)[:10],
                "extra": sorted(sub_set - qid_set)[:10],
            }
        )
    for qid in qids:
        item = submission.get(qid, {})
        answer = item.get("answer") if isinstance(item, dict) else None
        if not isinstance(answer, list):
            issues.append({"type": "answer_not_list", "qid": qid})
            continue
        length_dist[str(len(answer))] = length_dist.get(str(len(answer)), 0) + 1
        if not (1 <= len(answer) <= 5):
            issues.append({"type": "bad_answer_length", "qid": qid, "length": len(answer)})
        if len(set(answer)) != len(answer):
            issues.append({"type": "duplicate_doc_id", "qid": qid})
        if any(not isinstance(x, str) for x in answer):
            issues.append({"type": "non_string_doc_id", "qid": qid})
    return {
        "num_public_queries": len(qids),
        "num_submission_queries": len(submission),
        "answer_length_distribution": dict(sorted(length_dist.items())),
        "num_errors": len(issues),
        "issues": issues[:50],
    }


def zip_submission(submission_json: Path, submission_zip: Path) -> None:
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="submission.json")
    with zipfile.ZipFile(submission_zip, "r") as zf:
        names = zf.namelist()
    if names != ["submission.json"]:
        raise ValueError(f"Bad zip entries: {names}")


def audit_candidate(
    qids: list[str],
    submission: dict[str, Any],
    step5: dict[str, dict[str, Any]],
    sbert: dict[str, list[str]],
) -> dict[str, Any]:
    changed_vs_step5 = 0
    changed_vs_sbert = 0
    overlap_step5 = 0
    overlap_sbert = 0
    rank_counts: dict[str, int] = {}
    for qid in qids:
        docs = submission[qid]["answer"]
        step5_top5 = step5[qid]["docs"][:5]
        sbert_top5 = sbert.get(qid, [])[:5]
        changed_vs_step5 += docs != step5_top5
        changed_vs_sbert += docs != sbert_top5
        overlap_step5 += len(set(docs) & set(step5_top5))
        overlap_sbert += len(set(docs) & set(sbert_top5))
        step5_rank = {doc_id: rank for rank, doc_id in enumerate(step5[qid]["docs"], start=1)}
        for doc_id in docs:
            rank = step5_rank.get(doc_id)
            key = str(rank) if rank is not None and rank <= 20 else ">20_or_missing"
            rank_counts[key] = rank_counts.get(key, 0) + 1
    return {
        "changed_queries_vs_step5": changed_vs_step5,
        "changed_queries_vs_sbert": changed_vs_sbert,
        "avg_top5_overlap_step5": overlap_step5 / len(qids),
        "avg_top5_overlap_sbert": overlap_sbert / len(qids),
        "step5_original_rank_counts_in_output_top5": {
            str(i): rank_counts.get(str(i), 0) for i in range(1, 21)
        }
        | {">20_or_missing": rank_counts.get(">20_or_missing", 0)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--depth", type=int, default=20)
    args = parser.parse_args()

    root = Path(".")
    public_file = root / "task1/public-official.json"
    step5_file = root / "task1/pipeline/step5/step5/rankings/public_rankings_step5_fused.jsonl"
    sbert_file = root / "other_research/kaggle_output/legalir_sbert_cl/public_submission/public_ranked_contexts.csv"
    if not sbert_file.exists():
        sbert_file = root / "task1/pipeline/step6+sbertft/step6+sbertft/sbertft/public_submission/public_ranked_contexts.csv"
    output_root = root / "task1/pipeline/step6+sbertft/outputs/candidates"

    qids = public_qids(public_file)
    step5 = load_step5_rankings(step5_file)
    sbert = load_sbert_rankings(sbert_file)
    missing = [qid for qid in qids if qid not in step5 or qid not in sbert]
    if missing:
        raise ValueError({"missing_qids": missing[:20], "num_missing": len(missing)})

    configs = [
        {"name": "step5_sbertft_rrf_w0p15", "sbert_weight": 0.15, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w0p25", "sbert_weight": 0.25, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w0p40", "sbert_weight": 0.40, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w0p60", "sbert_weight": 0.60, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w0p80", "sbert_weight": 0.80, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w1p00", "sbert_weight": 1.00, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w1p25", "sbert_weight": 1.25, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w1p50", "sbert_weight": 1.50, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w2p00", "sbert_weight": 2.00, "lock_top_n": 0},
        {"name": "step5_sbertft_rrf_w0p40_lock1", "sbert_weight": 0.40, "lock_top_n": 1},
        {"name": "step5_sbertft_rrf_w0p60_lock1", "sbert_weight": 0.60, "lock_top_n": 1},
        {"name": "step5_sbertft_rrf_w0p60_lock2", "sbert_weight": 0.60, "lock_top_n": 2},
        {"name": "step5_sbertft_rrf_w0p80_lock2", "sbert_weight": 0.80, "lock_top_n": 2},
        {"name": "step5_sbertft_rrf_w1p00_lock2", "sbert_weight": 1.00, "lock_top_n": 2},
        {"name": "step5_sbertft_rrf_w1p25_lock2", "sbert_weight": 1.25, "lock_top_n": 2},
    ]

    reports = []
    for config in configs:
        output_dir = output_root / config["name"]
        submission = make_candidate(
            qids,
            step5,
            sbert,
            step5_weight=1.0,
            sbert_weight=float(config["sbert_weight"]),
            rrf_k=args.rrf_k,
            depth=args.depth,
            lock_top_n=int(config["lock_top_n"]),
        )
        validation = validate_submission(submission, qids)
        if validation["num_errors"]:
            raise ValueError({"candidate": config["name"], "validation": validation})

        submission_json = output_dir / "submission.json"
        submission_zip = output_dir / "submission.zip"
        write_json(submission_json, submission)
        write_json(output_dir / "submission_validation.json", validation)
        zip_submission(submission_json, submission_zip)

        audit = audit_candidate(qids, submission, step5, sbert)
        report = {
            "candidate": config["name"],
            "method": "weighted_rrf_step5_plus_sbertft_public_ranking",
            "step5_weight": 1.0,
            "sbert_weight": config["sbert_weight"],
            "lock_top_n": config["lock_top_n"],
            "rrf_k": args.rrf_k,
            "depth": args.depth,
            "validation": validation,
            "audit": audit,
            "submission_zip": str(submission_zip),
        }
        write_json(output_dir / "candidate_info.json", report)
        reports.append(report)

    run_report = {
        "status": "ok",
        "description": "Fuse main Step5 public rankings with existing SBERT-FT public rankings; no model inference.",
        "inputs": {
            "public_file": str(public_file),
            "step5_file": str(step5_file),
            "sbert_file": str(sbert_file),
        },
        "num_queries": len(qids),
        "candidate_audit": reports,
    }
    report_path = output_root / "step5_sbertft_fusion_report.json"
    write_json(report_path, run_report)
    print(json.dumps(run_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
