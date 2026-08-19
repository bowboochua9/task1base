"""Run Step 6 AITeamVN reranking on P0 no-metadata Step 5 candidates.

Use this on Kaggle/GPU after uploading the `step8_no_metadata` artifacts created
by `replay_p0_no_metadata.py`. It reuses a saved fine-tuned
AITeamVN/Vietnamese_Reranker model; it does not train a new model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def write_submission_zip(submission_json: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_json, arcname="submission.json")


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", strip_accents(text.lower()))


def load_rankings_and_payload(path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    rankings: dict[str, list[str]] = {}
    payload: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qid = str(row["query_id"])
        rankings[qid] = [str(x) for x in row["fused_doc_ids"]]
        gold = row.get("gold")
        payload[qid] = {
            "question": row.get("question", ""),
            "answer": [str(x) for x in gold] if isinstance(gold, list) else [],
        }
    return rankings, payload


def load_chunks(chunks_file: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]], set[str]]:
    chunks: list[dict[str, Any]] = []
    doc_to_chunk_indices: dict[str, list[int]] = defaultdict(list)
    valid_doc_ids: set[str] = set()
    for idx, row in enumerate(iter_jsonl(chunks_file)):
        doc_id = str(row.get("doc_id", ""))
        chunk = {
            "chunk_idx": idx,
            "chunk_id": str(row.get("chunk_id", idx)),
            "doc_id": doc_id,
            "text": str(row.get("text", "")),
            "heading": str(row.get("heading", "")),
            "word_count": int(row.get("word_count") or 0),
        }
        chunks.append(chunk)
        if doc_id:
            doc_to_chunk_indices[doc_id].append(idx)
            valid_doc_ids.add(doc_id)
        if (idx + 1) % 50000 == 0:
            print(f"loaded {idx + 1:,} chunks")
    return chunks, doc_to_chunk_indices, valid_doc_ids


def pick_evidence_text(
    question: str,
    doc_id: str,
    *,
    chunks: list[dict[str, Any]],
    doc_to_chunk_indices: dict[str, list[int]],
    max_chunks: int,
    max_chars: int,
) -> tuple[str, list[str]]:
    q = Counter(tokenize(question))
    q_set = set(q)
    scored: list[tuple[float, int, int]] = []
    for chunk_idx in doc_to_chunk_indices.get(str(doc_id), []):
        chunk = chunks[chunk_idx]
        toks = tokenize((chunk["heading"] + " " + chunk["text"])[:7000])
        cc = Counter(toks)
        overlap = sum(min(q[tok], cc[tok]) for tok in q_set)
        heading_bonus = 0.25 if any(tok in strip_accents(chunk["heading"].lower()) for tok in q_set) else 0.0
        score = overlap / max(1, len(q_set)) + heading_bonus
        scored.append((score, -abs(chunk["word_count"] - 320), chunk_idx))
    scored.sort(reverse=True)
    parts: list[str] = []
    chunk_ids: list[str] = []
    for _, _, chunk_idx in scored[:max_chunks]:
        chunk = chunks[chunk_idx]
        chunk_ids.append(chunk["chunk_id"])
        parts.append((chunk["heading"] + "\n" + chunk["text"]).strip())
    text = "\n\n".join(parts)
    return (text[:max_chars] if len(text) > max_chars else text), chunk_ids


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def logits_to_relevance(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return logits.view(-1)
    if logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits.max(dim=-1).values


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def load_model(model_path: Path, *, max_seq_length: int, device: torch.device) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    effective_max_length = min(max_seq_length, max_positions - 2) if isinstance(max_positions, int) and max_positions > 2 else max_seq_length
    model.to(device)
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "model_vocab_size": model.get_input_embeddings().num_embeddings,
        "max_length": effective_max_length,
        "model_info": {
            "model_path": str(model_path),
            "tokenizer_len": len(tokenizer),
            "model_vocab_size": model.get_input_embeddings().num_embeddings,
            "max_position_embeddings": max_positions,
            "effective_max_seq_length": effective_max_length,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": str(device),
        },
    }


def validate_batch_token_ids(batch: dict[str, torch.Tensor], *, model_vocab_size: int, tokenizer_len: int, where: str) -> None:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        return
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0 or max_id >= model_vocab_size:
        raise ValueError(
            f"Invalid token id at {where}: min={min_id}, max={max_id}, "
            f"model_vocab_size={model_vocab_size}, tokenizer_len={tokenizer_len}"
        )


@torch.inference_mode()
def score_pairs(bundle: dict[str, Any], pair_rows: list[dict[str, Any]], *, batch_size: int, device: torch.device) -> list[float]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    scores: list[float] = []
    for start in range(0, len(pair_rows), batch_size):
        batch_rows = pair_rows[start : start + batch_size]
        encoded = tokenizer(
            [r["question"] for r in batch_rows],
            [r["text"] for r in batch_rows],
            truncation="only_second",
            padding=True,
            max_length=bundle["max_length"],
            return_tensors="pt",
        )
        validate_batch_token_ids(
            encoded,
            model_vocab_size=bundle["model_vocab_size"],
            tokenizer_len=len(tokenizer),
            where=f"score batch {start}",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = logits_to_relevance(model(**encoded).logits).detach().cpu().numpy().tolist()
        scores.extend([sigmoid(x) for x in logits])
    return scores


def rerank_payload(
    *,
    bundle: dict[str, Any],
    payload: dict[str, dict[str, Any]],
    base_rankings: dict[str, list[str]],
    chunks: list[dict[str, Any]],
    doc_to_chunk_indices: dict[str, list[int]],
    output_file: Path,
    rerank_top_docs: int,
    evidence_chunks: int,
    evidence_max_chars: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    all_rankings: dict[str, list[str]] = {}
    scored_rows_by_qid: dict[str, list[dict[str, Any]]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for idx, (qid, row) in enumerate(payload.items(), start=1):
            question = row.get("question", "")
            candidates = base_rankings.get(qid, [])[:rerank_top_docs]
            pair_rows = []
            for rank, doc_id in enumerate(candidates, start=1):
                evidence_text, chunk_ids = pick_evidence_text(
                    question,
                    doc_id,
                    chunks=chunks,
                    doc_to_chunk_indices=doc_to_chunk_indices,
                    max_chunks=evidence_chunks,
                    max_chars=evidence_max_chars,
                )
                pair_rows.append(
                    {
                        "query_id": qid,
                        "question": question,
                        "doc_id": doc_id,
                        "text": evidence_text,
                        "retrieval_rank": rank,
                        "chunk_ids": chunk_ids,
                    }
                )
            reranker_scores = score_pairs(bundle, pair_rows, batch_size=batch_size, device=device) if pair_rows else []
            scored = []
            for pair, score in zip(pair_rows, reranker_scores):
                scored.append(
                    {
                        "doc_id": pair["doc_id"],
                        "reranker_score": score,
                        "retrieval_rank": pair["retrieval_rank"],
                        "chunk_ids": pair["chunk_ids"],
                    }
                )
            scored.sort(key=lambda r: r["reranker_score"], reverse=True)
            reranked = [r["doc_id"] for r in scored]
            tail = [doc_id for doc_id in base_rankings.get(qid, []) if doc_id not in set(reranked)]
            all_rankings[qid] = reranked + tail
            scored_rows_by_qid[qid] = scored
            if idx % 100 == 0:
                print(f"reranked {idx:,}/{len(payload):,}")
            yield {
                "query_id": qid,
                "question": question,
                "gold": row.get("answer"),
                "reranked_top_docs": scored,
                "reranked_doc_ids": all_rankings[qid],
            }

    write_jsonl(output_file, rows())
    return all_rankings, scored_rows_by_qid


def fuse_with_scores(
    base_rankings: dict[str, list[str]],
    reranked_scores: dict[str, list[dict[str, Any]]],
    *,
    retrieval_weight: float,
    reranker_weight: float,
    rerank_top_docs: int,
) -> dict[str, list[str]]:
    fused = {}
    for qid, base_docs in base_rankings.items():
        top_docs = base_docs[:rerank_top_docs]
        retrieval_raw = [1.0 / rank for rank in range(1, len(top_docs) + 1)]
        retrieval_norm = dict(zip(top_docs, minmax(retrieval_raw)))
        reranker_raw = {row["doc_id"]: row["reranker_score"] for row in reranked_scores.get(qid, [])}
        reranker_norm = dict(zip(reranker_raw.keys(), minmax(list(reranker_raw.values()))))
        scores = {}
        first_seen = {}
        for idx, doc_id in enumerate(top_docs):
            first_seen.setdefault(doc_id, idx)
            scores[doc_id] = retrieval_weight * retrieval_norm.get(doc_id, 0.0) + reranker_weight * reranker_norm.get(doc_id, 0.0)
        for idx, doc_id in enumerate(base_docs[rerank_top_docs:], start=len(top_docs)):
            first_seen.setdefault(doc_id, idx)
            scores.setdefault(doc_id, -idx * 1e-6)
        fused[qid] = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))[:100]]
    return fused


def ranking_metrics_for_query(ranked_doc_ids: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = {str(doc_id) for doc_id in gold}
    if not gold_set:
        return {k: 0.0 for k in ["precision@5", "recall@1", "recall@5", "recall@10", "recall@20", "recall@50", "recall@90", "recall@100", "hit@1", "hit@5", "hit@20", "exist@90", "mrr"]}

    def recall_at(k: int) -> float:
        return len(set(ranked_doc_ids[:k]) & gold_set) / len(gold_set)

    def hit_at(k: int) -> float:
        return 1.0 if set(ranked_doc_ids[:k]) & gold_set else 0.0

    first_rank = next((idx + 1 for idx, doc_id in enumerate(ranked_doc_ids) if doc_id in gold_set), None)
    top5 = ranked_doc_ids[:MAX_SUBMISSION_DOCS]
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
    per_query = {}
    for qid, row in payload.items():
        metrics = ranking_metrics_for_query(rankings.get(qid, []), row.get("answer", []))
        per_query[qid] = metrics
        sums.update(metrics)
    n = len(payload)
    return {"num_queries": n, "macro": {k: v / n for k, v in sorted(sums.items())}, "per_query": per_query}


def make_submission(rankings: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    return {qid: {"answer": [str(doc_id) for doc_id in docs[:MAX_SUBMISSION_DOCS]]} for qid, docs in rankings.items()}


def validate_submission(submission: dict[str, Any], expected_qids: set[str], valid_doc_ids: set[str]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    answer_lengths: Counter[int] = Counter()
    for qid in sorted(expected_qids - set(submission))[:50]:
        issues.append({"severity": "error", "kind": "missing_query", "query_id": qid})
    for qid, row in submission.items():
        answer = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(answer, list):
            issues.append({"severity": "error", "kind": "answer_not_array", "query_id": qid})
            continue
        answer_lengths[len(answer)] += 1
        if not (1 <= len(answer) <= MAX_SUBMISSION_DOCS):
            issues.append({"severity": "error", "kind": "answer_count", "query_id": qid})
        if len(answer) != len(set(answer)):
            issues.append({"severity": "error", "kind": "duplicate_doc_id", "query_id": qid})
        for doc_id in answer:
            if not isinstance(doc_id, str):
                issues.append({"severity": "error", "kind": "doc_id_not_string", "query_id": qid})
            elif doc_id not in valid_doc_ids:
                issues.append({"severity": "error", "kind": "unknown_doc_id", "query_id": qid, "doc_id": doc_id})
    return {
        "num_public_queries": len(expected_qids),
        "num_submission_queries": len(submission),
        "answer_length_distribution": dict(sorted(answer_lengths.items())),
        "num_errors": sum(issue["severity"] == "error" for issue in issues),
        "num_warnings": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dev-rankings", type=Path, required=True)
    parser.add_argument("--public-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/step8_no_metadata_step6"))
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--rerank-top-docs", type=int, default=50)
    parser.add_argument("--evidence-chunks", type=int, default=3)
    parser.add_argument("--evidence-max-chars", type=int, default=1800)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-weights", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--reranker-weights", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir
    rankings_dir = output_dir / "rankings" / "aiteamvn_vietnamese_reranker_finetuned_no_metadata"
    metrics_dir = output_dir / "metrics" / "aiteamvn_vietnamese_reranker_finetuned_no_metadata"
    submission_dir = output_dir / "submission" / "aiteamvn_vietnamese_reranker_finetuned_no_metadata"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available. This run may be very slow.")

    dev_rankings_base, dev_payload = load_rankings_and_payload(args.dev_rankings)
    public_rankings_base, public_payload = load_rankings_and_payload(args.public_rankings)
    chunks, doc_to_chunk_indices, valid_doc_ids = load_chunks(args.chunks_file)
    bundle = load_model(args.model_path, max_seq_length=args.max_seq_length, device=device)

    dev_reranked, dev_scores = rerank_payload(
        bundle=bundle,
        payload=dev_payload,
        base_rankings=dev_rankings_base,
        chunks=chunks,
        doc_to_chunk_indices=doc_to_chunk_indices,
        output_file=rankings_dir / "dev_rankings_reranker.jsonl",
        rerank_top_docs=args.rerank_top_docs,
        evidence_chunks=args.evidence_chunks,
        evidence_max_chars=args.evidence_max_chars,
        batch_size=args.rerank_batch_size,
        device=device,
    )
    reranker_only_metrics = evaluate_rankings(dev_reranked, dev_payload)
    write_json(metrics_dir / "dev_metrics_reranker_only.json", reranker_only_metrics)

    trials = []
    for retrieval_weight in args.retrieval_weights:
        for reranker_weight in args.reranker_weights:
            fused = fuse_with_scores(
                dev_rankings_base,
                dev_scores,
                retrieval_weight=retrieval_weight,
                reranker_weight=reranker_weight,
                rerank_top_docs=args.rerank_top_docs,
            )
            trials.append(
                {
                    "retrieval_weight": retrieval_weight,
                    "reranker_weight": reranker_weight,
                    "metrics": evaluate_rankings(fused, dev_payload),
                }
            )
    trials.sort(key=lambda row: (row["metrics"]["macro"]["recall@5"], row["metrics"]["macro"]["precision@5"]), reverse=True)
    best_fusion = {k: trials[0][k] for k in ["retrieval_weight", "reranker_weight"]}
    write_json(metrics_dir / "fusion_trials.json", trials)
    write_json(metrics_dir / "best_fusion_config.json", best_fusion)

    dev_fused = fuse_with_scores(
        dev_rankings_base,
        dev_scores,
        retrieval_weight=best_fusion["retrieval_weight"],
        reranker_weight=best_fusion["reranker_weight"],
        rerank_top_docs=args.rerank_top_docs,
    )
    dev_fused_metrics = evaluate_rankings(dev_fused, dev_payload)
    write_json(metrics_dir / "dev_metrics_step6_fused.json", dev_fused_metrics)
    write_jsonl(
        rankings_dir / "dev_rankings_step6_fused.jsonl",
        (
            {
                "query_id": qid,
                "question": dev_payload[qid].get("question", ""),
                "gold": dev_payload[qid].get("answer"),
                "base_doc_ids": dev_rankings_base.get(qid, []),
                "fused_doc_ids": dev_fused.get(qid, []),
            }
            for qid in dev_payload
        ),
    )

    public_reranked, public_scores = rerank_payload(
        bundle=bundle,
        payload=public_payload,
        base_rankings=public_rankings_base,
        chunks=chunks,
        doc_to_chunk_indices=doc_to_chunk_indices,
        output_file=rankings_dir / "public_rankings_reranker.jsonl",
        rerank_top_docs=args.rerank_top_docs,
        evidence_chunks=args.evidence_chunks,
        evidence_max_chars=args.evidence_max_chars,
        batch_size=args.rerank_batch_size,
        device=device,
    )
    _ = public_reranked
    public_fused = fuse_with_scores(
        public_rankings_base,
        public_scores,
        retrieval_weight=best_fusion["retrieval_weight"],
        reranker_weight=best_fusion["reranker_weight"],
        rerank_top_docs=args.rerank_top_docs,
    )
    write_jsonl(
        rankings_dir / "public_rankings_step6_fused.jsonl",
        (
            {
                "query_id": qid,
                "question": public_payload[qid].get("question", ""),
                "gold": None,
                "base_doc_ids": public_rankings_base.get(qid, []),
                "fused_doc_ids": public_fused.get(qid, []),
            }
            for qid in public_payload
        ),
    )

    submission = make_submission(public_fused)
    write_json(submission_dir / "submission.json", submission)
    write_submission_zip(submission_dir / "submission.json", submission_dir / "submission.zip")
    validation = validate_submission(submission, set(public_payload), valid_doc_ids)
    write_json(submission_dir / "submission_validation.json", validation)

    run_report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 3),
        "device": str(device),
        "model_info": bundle["model_info"],
        "inputs": {
            "chunks_file": str(args.chunks_file),
            "model_path": str(args.model_path),
            "dev_rankings": str(args.dev_rankings),
            "public_rankings": str(args.public_rankings),
        },
        "config": {
            "max_seq_length": args.max_seq_length,
            "rerank_top_docs": args.rerank_top_docs,
            "evidence_chunks": args.evidence_chunks,
            "evidence_max_chars": args.evidence_max_chars,
            "rerank_batch_size": args.rerank_batch_size,
            "retrieval_weights": args.retrieval_weights,
            "reranker_weights": args.reranker_weights,
        },
        "best_fusion": best_fusion,
        "reranker_only_dev_macro": reranker_only_metrics["macro"],
        "fused_dev_macro": dev_fused_metrics["macro"],
        "outputs": {
            "dev_rankings": str(rankings_dir / "dev_rankings_step6_fused.jsonl"),
            "public_rankings": str(rankings_dir / "public_rankings_step6_fused.jsonl"),
            "submission_zip": str(submission_dir / "submission.zip"),
            "submission_validation": str(submission_dir / "submission_validation.json"),
        },
    }
    write_json(output_dir / "reports" / "run_report.json", run_report)
    print(json.dumps(run_report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
