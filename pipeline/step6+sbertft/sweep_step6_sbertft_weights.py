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


def load_step6_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        qid = str(row["query_id"])
        rankings[qid] = unique_keep_order([str(x) for x in row["fused_doc_ids"]])
    return rankings


def load_sbert_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"])
            docs = json.loads(row["top20_rerank"] or row["answer"])
            rankings[qid] = unique_keep_order([str(x) for x in docs])
    return rankings


def weighted_rrf(
    step6_docs: list[str],
    sbert_docs: list[str],
    *,
    step6_weight: float,
    sbert_weight: float,
    rrf_k: int,
    step6_depth: int | None,
    sbert_depth: int | None,
) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    step6_slice = step6_docs[:step6_depth] if step6_depth else step6_docs
    sbert_slice = sbert_docs[:sbert_depth] if sbert_depth else sbert_docs
    for source_idx, (docs, weight) in enumerate(
        [(step6_slice, step6_weight), (sbert_slice, sbert_weight)]
    ):
        for rank, doc_id in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            first_seen.setdefault(doc_id, (source_idx, rank))
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id], doc_id))


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
    step6: dict[str, list[str]],
    sbert: dict[str, list[str]],
) -> dict[str, Any]:
    changed_vs_step6 = 0
    changed_vs_sbert = 0
    overlap_step6 = 0
    overlap_sbert = 0
    top1_same_step6 = 0
    rank_counts: dict[str, int] = {}
    for qid in qids:
        docs = submission[qid]["answer"]
        step6_top5 = step6[qid][:5]
        sbert_top5 = sbert[qid][:5]
        changed_vs_step6 += docs != step6_top5
        changed_vs_sbert += docs != sbert_top5
        overlap_step6 += len(set(docs) & set(step6_top5))
        overlap_sbert += len(set(docs) & set(sbert_top5))
        top1_same_step6 += docs[0] == step6_top5[0]
        step6_rank = {doc_id: rank for rank, doc_id in enumerate(step6[qid], start=1)}
        for doc_id in docs:
            rank = step6_rank.get(doc_id)
            key = str(rank) if rank is not None and rank <= 20 else ">20_or_missing"
            rank_counts[key] = rank_counts.get(key, 0) + 1
    return {
        "changed_queries_vs_step6": changed_vs_step6,
        "changed_queries_vs_sbert": changed_vs_sbert,
        "avg_top5_overlap_step6": overlap_step6 / len(qids),
        "avg_top5_overlap_sbert": overlap_sbert / len(qids),
        "top1_same_as_step6": top1_same_step6,
        "step6_original_rank_counts_in_output_top5": {
            str(i): rank_counts.get(str(i), 0) for i in range(1, 21)
        }
        | {">20_or_missing": rank_counts.get(">20_or_missing", 0)},
    }


def weight_name(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p").rstrip("0").rstrip("p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--step6-depth", type=int, default=0, help="0 means use the full Step6 ranking.")
    parser.add_argument("--sbert-depth", type=int, default=20)
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50, 2.00],
    )
    args = parser.parse_args()

    root = Path(".")
    input_root = root / "task1/pipeline/step6+sbertft/step6+sbertft"
    public_file = input_root / "public-official.json"
    step6_file = input_root / "step6/rankings/public_rankings_step6_fused.jsonl"
    sbert_file = input_root / "sbertft/public_submission/public_ranked_contexts.csv"
    output_root = root / "task1/pipeline/step6+sbertft/outputs/candidates"

    qids = public_qids(public_file)
    step6 = load_step6_rankings(step6_file)
    sbert = load_sbert_rankings(sbert_file)
    missing = [qid for qid in qids if qid not in step6 or qid not in sbert]
    if missing:
        raise ValueError({"missing_qids": missing[:20], "num_missing": len(missing)})

    reports = []
    for weight in args.weights:
        candidate = f"rrf_sbert{weight_name(weight)}"
        output_dir = output_root / candidate
        submission: dict[str, dict[str, list[str]]] = {}
        for qid in qids:
            docs = weighted_rrf(
                step6[qid],
                sbert[qid],
                step6_weight=1.0,
                sbert_weight=weight,
                rrf_k=args.rrf_k,
                step6_depth=args.step6_depth or None,
                sbert_depth=args.sbert_depth or None,
            )
            submission[qid] = {"answer": docs[:5]}

        validation = validate_submission(submission, qids)
        if validation["num_errors"]:
            raise ValueError({"candidate": candidate, "validation": validation})

        submission_json = output_dir / "submission.json"
        submission_zip = output_dir / "submission.zip"
        write_json(submission_json, submission)
        write_json(output_dir / "submission_validation.json", validation)
        zip_submission(submission_json, submission_zip)

        report = {
            "candidate": candidate,
            "method": "weighted_rrf_step6_plus_sbertft_public_ranking",
            "step6_weight": 1.0,
            "sbert_weight": weight,
            "rrf_k": args.rrf_k,
            "step6_depth": args.step6_depth or "full",
            "sbert_depth": args.sbert_depth or "full",
            "validation": validation,
            "audit": audit_candidate(qids, submission, step6, sbert),
            "submission_zip": str(submission_zip),
        }
        write_json(output_dir / "candidate_info.json", report)
        reports.append(report)

    run_report = {
        "status": "ok",
        "description": "High-weight sweep for Step6 fused ranking plus existing SBERT-FT public ranking; no model inference.",
        "inputs": {
            "public_file": str(public_file),
            "step6_file": str(step6_file),
            "sbert_file": str(sbert_file),
        },
        "num_queries": len(qids),
        "candidate_audit": reports,
    }
    report_path = output_root / "step6_sbertft_high_weight_sweep_report.json"
    write_json(report_path, run_report)
    print(json.dumps(run_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
