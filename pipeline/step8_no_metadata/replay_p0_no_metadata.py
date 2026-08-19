"""Replay P0 no-metadata retrieval/fusion from saved Step 4/5 artifacts.

This script does not run model inference. It reuses existing Step 4 BM25/BGE-M3
rankings and Step 5 fine-tuned dense rankings, removes the Step 4 metadata
branch, then writes clean Step 4/5 no-metadata artifacts for the next GPU
reranker pass.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("task1/pipeline/step8_no_metadata")
DEFAULT_STEP4_DEV = Path("task1/pipeline/step4/step4/rankings/dev_rankings_rrf.jsonl")
DEFAULT_STEP4_PUBLIC = Path("task1/pipeline/step4/step4/rankings/public_rankings_rrf.jsonl")
DEFAULT_FT_DEV = Path("task1/pipeline/step5/step5/rankings/dev_rankings_ft_dense.jsonl")
DEFAULT_FT_PUBLIC = Path("task1/pipeline/step5/step5/rankings/public_rankings_ft_dense.jsonl")
MAX_SUBMISSION_DOCS = 5


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def doc_ids_from_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [str(row["doc_id"]) for row in rows if row.get("doc_id") is not None]


def load_ranked_doc_ids(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        qid = str(row["query_id"])
        if "fused_doc_ids" in row:
            rankings[qid] = [str(x) for x in row["fused_doc_ids"]]
        elif "dense_top_docs" in row:
            rankings[qid] = doc_ids_from_rows(row.get("dense_top_docs", []))
        elif "top_docs" in row:
            rankings[qid] = doc_ids_from_rows(row.get("top_docs", []))
        else:
            raise KeyError(f"Cannot find ranking field in {path}: query_id={qid}")
    return rankings


def load_eval_payload_from_rankings(path: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = str(row["query_id"])
        gold = row.get("gold")
        payload[qid] = {
            "question": row.get("question", ""),
            "answer": [str(x) for x in gold] if isinstance(gold, list) else [],
        }
    return payload


def rrf_fuse(
    branches: list[tuple[list[str], float]],
    *,
    rrf_k: int,
    top_docs: int = 100,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    for branch_idx, (doc_ids, weight) in enumerate(branches):
        if weight <= 0:
            continue
        for rank, doc_id in enumerate(doc_ids, start=1):
            doc_id = str(doc_id)
            scores[doc_id] += weight / (rrf_k + rank)
            first_seen.setdefault(doc_id, (branch_idx, rank))
    return [
        doc_id
        for doc_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]][0], first_seen[item[0]][1], item[0]),
        )[:top_docs]
    ]


def ranking_metrics_for_query(ranked_doc_ids: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = {str(doc_id) for doc_id in gold}
    if not gold_set:
        return {
            "precision@5": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "recall@50": 0.0,
            "recall@90": 0.0,
            "recall@100": 0.0,
            "hit@1": 0.0,
            "hit@5": 0.0,
            "hit@20": 0.0,
            "exist@90": 0.0,
            "mrr": 0.0,
        }

    def recall_at(k: int) -> float:
        return len(set(ranked_doc_ids[:k]) & gold_set) / len(gold_set)

    def hit_at(k: int) -> float:
        return 1.0 if set(ranked_doc_ids[:k]) & gold_set else 0.0

    top5 = ranked_doc_ids[:MAX_SUBMISSION_DOCS]
    first_rank = next((idx + 1 for idx, doc_id in enumerate(ranked_doc_ids) if doc_id in gold_set), None)
    return {
        "precision@5": len(set(top5) & gold_set) / MAX_SUBMISSION_DOCS,
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@20": recall_at(20),
        "recall@50": recall_at(50),
        "recall@90": recall_at(90),
        "recall@100": recall_at(100),
        "hit@1": hit_at(1),
        "hit@5": hit_at(5),
        "hit@20": hit_at(20),
        "exist@90": hit_at(90),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def evaluate_rankings(rankings: dict[str, list[str]], payload: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sums: Counter[str] = Counter()
    by_gold_count: dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_gold_count: Counter[str] = Counter()
    per_query = {}

    for qid, row in payload.items():
        gold = row.get("answer", [])
        metrics = ranking_metrics_for_query(rankings.get(str(qid), []), gold)
        per_query[str(qid)] = metrics
        sums.update(metrics)
        gold_count_key = str(len(gold))
        by_gold_count[gold_count_key].update(metrics)
        counts_by_gold_count[gold_count_key] += 1

    n = len(payload)
    macro = {key: (value / n if n else 0.0) for key, value in sorted(sums.items())}
    breakdown = {
        key: {metric: value / counts_by_gold_count[key] for metric, value in sorted(counter.items())}
        for key, counter in sorted(by_gold_count.items(), key=lambda item: int(item[0]))
    }
    return {"num_queries": n, "macro": macro, "by_gold_count": breakdown, "per_query": per_query}


def compare_rankings(
    before: dict[str, list[str]],
    after: dict[str, list[str]],
    payload: dict[str, dict[str, Any]],
    *,
    metric: str = "recall@5",
) -> dict[str, Any]:
    improved: list[str] = []
    degraded: list[str] = []
    unchanged: list[str] = []
    deltas: dict[str, float] = {}
    for qid, row in payload.items():
        gold = row.get("answer", [])
        before_value = ranking_metrics_for_query(before.get(qid, []), gold)[metric]
        after_value = ranking_metrics_for_query(after.get(qid, []), gold)[metric]
        delta = after_value - before_value
        deltas[qid] = delta
        if delta > 0:
            improved.append(qid)
        elif delta < 0:
            degraded.append(qid)
        else:
            unchanged.append(qid)
    return {
        "metric": metric,
        "num_improved": len(improved),
        "num_degraded": len(degraded),
        "num_unchanged": len(unchanged),
        "improved_query_ids": improved,
        "degraded_query_ids": degraded,
        "deltas": deltas,
    }


def oracle_union_recall(
    branches: list[dict[str, list[str]]],
    payload: dict[str, dict[str, Any]],
    *,
    depth: int,
) -> float:
    total = 0.0
    for qid, row in payload.items():
        gold_set = {str(x) for x in row.get("answer", [])}
        if not gold_set:
            continue
        union: set[str] = set()
        for branch in branches:
            union.update(branch.get(qid, [])[:depth])
        total += len(union & gold_set) / len(gold_set)
    return total / len(payload) if payload else 0.0


def make_submission(rankings: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    return {qid: {"answer": [str(doc_id) for doc_id in docs[:MAX_SUBMISSION_DOCS]]} for qid, docs in rankings.items()}


def validate_submission(submission: dict[str, Any], expected_qids: set[str]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    submission_qids = set(submission)
    for qid in sorted(expected_qids - submission_qids)[:50]:
        issues.append({"severity": "error", "kind": "missing_query", "query_id": qid})
    for qid in sorted(submission_qids - expected_qids)[:50]:
        issues.append({"severity": "error", "kind": "extra_query", "query_id": qid})

    answer_lengths: Counter[int] = Counter()
    for qid, row in submission.items():
        if not isinstance(row, dict):
            issues.append({"severity": "error", "kind": "row_not_object", "query_id": qid})
            continue
        answer = row.get("answer")
        if not isinstance(answer, list):
            issues.append({"severity": "error", "kind": "answer_not_array", "query_id": qid})
            continue
        answer_lengths[len(answer)] += 1
        if not (1 <= len(answer) <= MAX_SUBMISSION_DOCS):
            issues.append({"severity": "error", "kind": "answer_count", "query_id": qid})
        if any(not isinstance(doc_id, str) for doc_id in answer):
            issues.append({"severity": "error", "kind": "doc_id_not_string", "query_id": qid})
        if len(answer) != len(set(answer)):
            issues.append({"severity": "error", "kind": "duplicate_doc_id", "query_id": qid})

    return {
        "num_public_queries": len(expected_qids),
        "num_submission_queries": len(submission_qids),
        "answer_length_distribution": dict(sorted(answer_lengths.items())),
        "num_errors": sum(issue["severity"] == "error" for issue in issues),
        "num_warnings": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
    }


def write_submission_zip(submission_json: Path, submission_zip: Path) -> None:
    submission_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="submission.json")


def replay_step4(
    *,
    source_path: Path,
    output_path: Path,
    bm25_weight: float,
    dense_weight: float,
    rrf_k: int,
    top_docs: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for row in iter_jsonl(source_path):
            qid = str(row["query_id"])
            bm25_doc_ids = doc_ids_from_rows(row.get("bm25_top_docs", []))
            dense_doc_ids = doc_ids_from_rows(row.get("dense_top_docs", []))
            fused_doc_ids = rrf_fuse(
                [(bm25_doc_ids, bm25_weight), (dense_doc_ids, dense_weight)],
                rrf_k=rrf_k,
                top_docs=top_docs,
            )
            rankings[qid] = fused_doc_ids
            yield {
                "query_id": qid,
                "question": row.get("question", ""),
                "gold": row.get("gold"),
                "bm25_doc_ids": bm25_doc_ids,
                "dense_doc_ids": dense_doc_ids,
                "metadata_ranked_doc_ids": [],
                "fused_doc_ids": fused_doc_ids,
            }

    write_jsonl(output_path, rows())
    return rankings


def replay_step5(
    *,
    step4_rankings: dict[str, list[str]],
    ft_dense_path: Path,
    output_path: Path,
    step4_weight: float,
    ft_dense_weight: float,
    rrf_k: int,
    top_docs: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for row in iter_jsonl(ft_dense_path):
            qid = str(row["query_id"])
            ft_dense_doc_ids = doc_ids_from_rows(row.get("dense_top_docs", []))
            fused_doc_ids = rrf_fuse(
                [(step4_rankings.get(qid, []), step4_weight), (ft_dense_doc_ids, ft_dense_weight)],
                rrf_k=rrf_k,
                top_docs=top_docs,
            )
            rankings[qid] = fused_doc_ids
            yield {
                "query_id": qid,
                "question": row.get("question", ""),
                "gold": row.get("gold"),
                "step4_doc_ids": step4_rankings.get(qid, []),
                "ft_dense_doc_ids": ft_dense_doc_ids,
                "fused_doc_ids": fused_doc_ids,
            }

    write_jsonl(output_path, rows())
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step4-dev", type=Path, default=DEFAULT_STEP4_DEV)
    parser.add_argument("--step4-public", type=Path, default=DEFAULT_STEP4_PUBLIC)
    parser.add_argument("--ft-dev", type=Path, default=DEFAULT_FT_DEV)
    parser.add_argument("--ft-public", type=Path, default=DEFAULT_FT_PUBLIC)
    parser.add_argument("--step4-rrf-k", type=int, default=60)
    parser.add_argument("--step4-bm25-weight", type=float, default=1.0)
    parser.add_argument("--step4-dense-weight", type=float, default=1.0)
    parser.add_argument("--step5-rrf-k", type=int, default=20)
    parser.add_argument("--step5-step4-weight", type=float, default=1.2)
    parser.add_argument("--step5-ft-weight", type=float, default=0.4)
    parser.add_argument("--top-docs", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    out = args.output_dir
    rankings_dir = out / "rankings"
    metrics_dir = out / "metrics"
    reports_dir = out / "reports"
    predictions_dir = out / "predictions"
    submission_dir = out / "submission" / "step5_no_metadata"
    configs_dir = out / "configs"

    for path in [args.step4_dev, args.step4_public, args.ft_dev, args.ft_public]:
        if not path.exists():
            raise FileNotFoundError(path)

    dev_payload = load_eval_payload_from_rankings(args.step4_dev)
    write_json(configs_dir / "canonical_dev_ids.json", list(dev_payload.keys()))

    old_step4_dev = load_ranked_doc_ids(args.step4_dev)
    old_step5_dev = load_ranked_doc_ids(Path("task1/pipeline/step5/step5/rankings/dev_rankings_step5_fused.jsonl"))
    ft_dev = load_ranked_doc_ids(args.ft_dev)

    step4_dev = replay_step4(
        source_path=args.step4_dev,
        output_path=rankings_dir / "dev_rankings_step4_no_metadata.jsonl",
        bm25_weight=args.step4_bm25_weight,
        dense_weight=args.step4_dense_weight,
        rrf_k=args.step4_rrf_k,
        top_docs=args.top_docs,
    )
    step4_public = replay_step4(
        source_path=args.step4_public,
        output_path=rankings_dir / "public_rankings_step4_no_metadata.jsonl",
        bm25_weight=args.step4_bm25_weight,
        dense_weight=args.step4_dense_weight,
        rrf_k=args.step4_rrf_k,
        top_docs=args.top_docs,
    )
    step5_dev = replay_step5(
        step4_rankings=step4_dev,
        ft_dense_path=args.ft_dev,
        output_path=rankings_dir / "dev_rankings_step5_no_metadata_fused.jsonl",
        step4_weight=args.step5_step4_weight,
        ft_dense_weight=args.step5_ft_weight,
        rrf_k=args.step5_rrf_k,
        top_docs=args.top_docs,
    )
    step5_public = replay_step5(
        step4_rankings=step4_public,
        ft_dense_path=args.ft_public,
        output_path=rankings_dir / "public_rankings_step5_no_metadata_fused.jsonl",
        step4_weight=args.step5_step4_weight,
        ft_dense_weight=args.step5_ft_weight,
        rrf_k=args.step5_rrf_k,
        top_docs=args.top_docs,
    )

    metrics = {
        "step4_current_with_metadata": evaluate_rankings(old_step4_dev, dev_payload),
        "step4_no_metadata": evaluate_rankings(step4_dev, dev_payload),
        "step5_current_with_metadata": evaluate_rankings(old_step5_dev, dev_payload),
        "step5_no_metadata": evaluate_rankings(step5_dev, dev_payload),
    }
    for name, payload in metrics.items():
        write_json(metrics_dir / f"{name}.json", payload)

    write_json(
        metrics_dir / "delta_report.json",
        {
            "step4_no_metadata_vs_current": compare_rankings(old_step4_dev, step4_dev, dev_payload),
            "step5_no_metadata_vs_current": compare_rankings(old_step5_dev, step5_dev, dev_payload),
            "oracle_union_recall": {
                "step4_bm25_dense_depth5": oracle_union_recall([step4_dev], dev_payload, depth=5),
                "step5_step4_ft_depth5": oracle_union_recall([step4_dev, ft_dev], dev_payload, depth=5),
                "step5_step4_ft_depth10": oracle_union_recall([step4_dev, ft_dev], dev_payload, depth=10),
                "step5_step4_ft_depth20": oracle_union_recall([step4_dev, ft_dev], dev_payload, depth=20),
                "step5_step4_ft_depth50": oracle_union_recall([step4_dev, ft_dev], dev_payload, depth=50),
                "step5_step4_ft_depth100": oracle_union_recall([step4_dev, ft_dev], dev_payload, depth=100),
            },
        },
    )

    write_json(predictions_dir / "dev_predictions_top5_step4_no_metadata.json", {qid: docs[:5] for qid, docs in step4_dev.items()})
    write_json(predictions_dir / "dev_predictions_top5_step5_no_metadata.json", {qid: docs[:5] for qid, docs in step5_dev.items()})

    submission = make_submission(step5_public)
    write_json(submission_dir / "submission.json", submission)
    write_submission_zip(submission_dir / "submission.json", submission_dir / "submission.zip")
    validation = validate_submission(submission, set(step5_public.keys()))
    write_json(submission_dir / "submission_validation.json", validation)

    config = {
        "step4": {
            "bm25_weight": args.step4_bm25_weight,
            "dense_weight": args.step4_dense_weight,
            "metadata_weight": 0.0,
            "rrf_k": args.step4_rrf_k,
            "top_docs": args.top_docs,
        },
        "step5": {
            "step4_weight": args.step5_step4_weight,
            "ft_dense_weight": args.step5_ft_weight,
            "rrf_k": args.step5_rrf_k,
            "top_docs": args.top_docs,
        },
    }
    write_json(configs_dir / "p0_no_metadata_config.json", config)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 3),
        "purpose": "P0 no-metadata replay; no model inference was run.",
        "inputs": {
            "step4_dev": str(args.step4_dev),
            "step4_public": str(args.step4_public),
            "ft_dev": str(args.ft_dev),
            "ft_public": str(args.ft_public),
        },
        "config": config,
        "dev_macro": {
            "step4_current_with_metadata": metrics["step4_current_with_metadata"]["macro"],
            "step4_no_metadata": metrics["step4_no_metadata"]["macro"],
            "step5_current_with_metadata": metrics["step5_current_with_metadata"]["macro"],
            "step5_no_metadata": metrics["step5_no_metadata"]["macro"],
        },
        "outputs": {
            "canonical_dev_ids": str(configs_dir / "canonical_dev_ids.json"),
            "step4_dev_rankings": str(rankings_dir / "dev_rankings_step4_no_metadata.jsonl"),
            "step4_public_rankings": str(rankings_dir / "public_rankings_step4_no_metadata.jsonl"),
            "step5_dev_rankings": str(rankings_dir / "dev_rankings_step5_no_metadata_fused.jsonl"),
            "step5_public_rankings": str(rankings_dir / "public_rankings_step5_no_metadata_fused.jsonl"),
            "step5_submission_zip": str(submission_dir / "submission.zip"),
            "submission_validation": str(submission_dir / "submission_validation.json"),
        },
        "next_gpu_step": {
            "needed": True,
            "reason": "AITeamVN Step 6 reranking needs GPU/model inference on the new Step 5 no-metadata top-50 candidates.",
            "input_rankings": [
                str(rankings_dir / "dev_rankings_step5_no_metadata_fused.jsonl"),
                str(rankings_dir / "public_rankings_step5_no_metadata_fused.jsonl"),
            ],
        },
    }
    write_json(reports_dir / "run_report.json", summary)
    print(json.dumps(summary["dev_macro"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote P0 no-metadata artifacts to {out}")


if __name__ == "__main__":
    main()
