from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyvi import ViTokenizer
from transformers import AutoModel, AutoTokenizer


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


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


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
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


def mean_pool_normalized(model, tokenizer, texts: list[str], *, batch_size: int, max_length: int) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu().numpy().astype("float32"))
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 20) == 0:
            print(f"encoded {done:,}/{len(texts):,}", flush=True)
    return np.vstack(embeddings) if embeddings else np.zeros((0, 768), dtype="float32")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(".")
    public_file = root / "task1/public-official.json"
    chunks_file = root / "task1/pipeline/step1/outputs/chunks.jsonl"
    step5_rankings_file = root / "task1/pipeline/step5/step5/rankings/public_rankings_step5_fused.jsonl"
    model_dir = root / "other_research/kaggle_output/legalir_sbert_cl/best_model"
    output_name = args.output_name or f"step5_sbertft_model_rerank_top{args.depth}"
    output_dir = root / "task1/pipeline/step6+sbertft/outputs/candidates" / output_name
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"output_dir": str(output_dir), "depth": args.depth}, indent=2), flush=True)

    qids = public_qids(public_file)
    rankings = []
    selected_docs = set()
    for row in iter_jsonl(step5_rankings_file):
        qid = str(row["query_id"])
        docs = unique_keep_order([str(x) for x in row["fused_doc_ids"]])
        rankings.append({"query_id": qid, "question": str(row.get("question", "")), "docs": docs})
        selected_docs.update(docs[: args.depth])
    print(f"loaded rankings={len(rankings):,}; selected_docs={len(selected_docs):,}", flush=True)

    t0 = time.time()
    doc_chunks: dict[str, list[tuple[str, str, set[str]]]] = defaultdict(list)
    for row in iter_jsonl(chunks_file):
        doc_id = str(row.get("doc_id", ""))
        if doc_id not in selected_docs:
            continue
        text = (str(row.get("heading") or "") + "\n" + str(row.get("text") or "")).strip()
        if not text:
            continue
        doc_chunks[doc_id].append((str(row.get("chunk_id")), text, tokens(text)))
    print(
        f"loaded candidate chunks={sum(len(v) for v in doc_chunks.values()):,}; "
        f"docs_with_chunks={len(doc_chunks):,}; seconds={time.time() - t0:.1f}",
        flush=True,
    )

    pair_rows = []
    chunk_text_by_id: dict[str, str] = {}
    for row in rankings:
        qid = row["query_id"]
        qset = tokens(row["question"])
        for rank, doc_id in enumerate(row["docs"][: args.depth], start=1):
            best_score = -1.0
            best_chunk_id = None
            best_text = None
            for chunk_id, text, chunk_tokens in doc_chunks.get(doc_id, []):
                score = len(qset & chunk_tokens) / max(1, len(qset))
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk_id
                    best_text = text
            if best_chunk_id is None:
                continue
            chunk_text_by_id.setdefault(best_chunk_id, best_text or "")
            pair_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "retrieval_rank": rank,
                    "chunk_id": best_chunk_id,
                    "lexical_score": best_score,
                }
            )
    print(f"selected pairs={len(pair_rows):,}; unique_chunks={len(chunk_text_by_id):,}", flush=True)

    pair_path = cache_dir / "selected_pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as f:
        for row in pair_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(cache_dir / "selected_chunk_ids.json", list(chunk_text_by_id.keys()))

    print("loading SBERT-FT model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    query_ids = [row["query_id"] for row in rankings]
    query_texts = [ViTokenizer.tokenize(row["question"]) for row in rankings]
    chunk_ids = list(chunk_text_by_id.keys())
    chunk_texts = [ViTokenizer.tokenize(chunk_text_by_id[cid]) for cid in chunk_ids]

    print("encoding queries", flush=True)
    query_emb = mean_pool_normalized(
        model,
        tokenizer,
        query_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    np.save(cache_dir / "query_embeddings.npy", query_emb)
    write_json(cache_dir / "query_ids.json", query_ids)

    print("encoding selected chunks", flush=True)
    chunk_emb = mean_pool_normalized(
        model,
        tokenizer,
        chunk_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    np.save(cache_dir / "chunk_embeddings.npy", chunk_emb)
    write_json(cache_dir / "chunk_ids.json", chunk_ids)

    query_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    chunk_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
    scored_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        qi = query_idx[row["query_id"]]
        ci = chunk_idx[row["chunk_id"]]
        score = float(np.dot(query_emb[qi], chunk_emb[ci]))
        scored = dict(row)
        scored["sbertft_score"] = score
        scored_by_qid[row["query_id"]].append(scored)

    submission = {}
    ranking_audit_path = output_dir / "rankings_sbertft_reranked.jsonl"
    with ranking_audit_path.open("w", encoding="utf-8") as f:
        for row in rankings:
            qid = row["query_id"]
            scored = scored_by_qid.get(qid, [])
            scored.sort(key=lambda x: (-float(x["sbertft_score"]), int(x["retrieval_rank"])))
            reranked_docs = unique_keep_order([x["doc_id"] for x in scored])
            tail = [doc_id for doc_id in row["docs"] if doc_id not in set(reranked_docs)]
            final_docs = unique_keep_order(reranked_docs + tail)
            submission[qid] = {"answer": final_docs[:5]}
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "question": row["question"],
                        "reranked_top_docs": scored,
                        "reranked_doc_ids": final_docs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    validation = validate_submission(submission, qids)
    if validation["num_errors"]:
        raise ValueError(validation)
    write_json(output_dir / "submission.json", submission)
    write_json(output_dir / "submission_validation.json", validation)
    zip_submission(output_dir / "submission.json", output_dir / "submission.zip")

    report = {
        "status": "ok",
        "candidate": output_name,
        "description": "Main Step1-5 candidates reranked with SBERT-FT model from other_research; no Step6 cross-encoder.",
        "depth": args.depth,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "inputs": {
            "public_file": str(public_file),
            "chunks_file": str(chunks_file),
            "step5_rankings_file": str(step5_rankings_file),
            "model_dir": str(model_dir),
        },
        "num_pairs": len(pair_rows),
        "num_unique_chunks": len(chunk_ids),
        "validation": validation,
        "submission_zip": str(output_dir / "submission.zip"),
    }
    write_json(output_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
