"""Step 4 BGE-M3 dense retrieval + metadata branch + RRF.

Designed for Kaggle T4x2:
- downloads/loads BAAI/bge-m3 locally through FlagEmbedding
- encodes Step 1 chunks into cached fp16 dense vectors
- retrieves top dense chunks for dev/public queries
- aggregates chunks to document candidates
- fuses BM25 + dense + optional metadata branch via Reciprocal Rank Fusion
- evaluates on dev and can package public submission.zip

No hosted inference/API is used. Model weights are loaded into the notebook
runtime and all inference is local.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
STEP2_DIR = THIS_DIR.parent / "step2"
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from legalir_step2_bm25 import (  # noqa: E402
    MAX_SUBMISSION_DOCS,
    append_jsonl,
    evaluate_rankings,
    find_public_file,
    make_submission,
    read_json,
    validate_submission_payload,
    write_json,
    write_submission_zip,
)


DEFAULT_DATA_ROOT = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26")
DEFAULT_STEP4_INPUT_ROOT = DEFAULT_DATA_ROOT / "step4"
DEFAULT_OUTPUT = (
    Path("/kaggle/working/step4")
    if Path("/kaggle").exists()
    else Path("task1/pipeline/step4/outputs")
)
DEFAULT_PUBLIC_FILE = Path(
    "/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json"
)


@dataclass(frozen=True)
class Step4Config:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 12
    query_batch_size: int = 32
    max_length: int = 512
    use_fp16: bool = True
    dense_top_chunks: int = 300
    dense_top_docs: int = 100
    evidence_per_doc: int = 3
    aggregate_mean_top3_weight: float = 0.20
    aggregate_support_weight: float = 0.05
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    metadata_weight: float = 0.20
    fused_top_docs: int = 100
    search_block_size: int = 32768


def find_step4_inputs(
    *,
    data_root: Path | None,
    chunks_file: Path | None,
    train_split_file: Path | None,
    dev_split_file: Path | None,
    bm25_dev_rankings_file: Path | None,
    best_config_file: Path | None,
) -> tuple[Path, Path, Path, Path, Path | None]:
    roots = []
    if data_root is not None:
        roots.append(data_root)
    roots.extend(
        [
            DEFAULT_STEP4_INPUT_ROOT,
            DEFAULT_DATA_ROOT,
            Path("task1/pipeline/step4"),
            Path("task1/pipeline/step1/outputs"),
            Path("."),
        ]
    )

    resolved_chunks = chunks_file
    resolved_train = train_split_file
    resolved_dev = dev_split_file
    resolved_bm25 = bm25_dev_rankings_file
    resolved_best_config = best_config_file

    for root in roots:
        if resolved_chunks is None:
            resolved_chunks = next(
                (
                    p
                    for p in [
                        root / "step1" / "chunks.jsonl",
                        root / "chunks.jsonl",
                        root / "corpus" / "chunks.jsonl",
                    ]
                    if p.exists()
                ),
                None,
            )
        if resolved_train is None:
            resolved_train = next(
                (
                    p
                    for p in [
                        root / "step1" / "train_split.json",
                        root / "train_split.json",
                        root / "splits" / "train_split.json",
                    ]
                    if p.exists()
                ),
                None,
            )
        if resolved_dev is None:
            resolved_dev = next(
                (
                    p
                    for p in [
                        root / "step1" / "dev_split.json",
                        root / "dev_split.json",
                        root / "splits" / "dev_split.json",
                    ]
                    if p.exists()
                ),
                None,
            )
        if resolved_bm25 is None:
            resolved_bm25 = next(
                (
                    p
                    for p in [
                        root / "step3" / "dev_rankings_best.jsonl",
                        root / "dev_rankings_best.jsonl",
                        root / "rankings" / "dev_rankings_best.jsonl",
                        Path("task1/pipeline/step3/outputs/rankings/dev_rankings_best.jsonl"),
                    ]
                    if p.exists()
                ),
                None,
            )
        if resolved_best_config is None:
            resolved_best_config = next(
                (
                    p
                    for p in [
                        root / "step3" / "best_config.json",
                        root / "best_config.json",
                        root / "configs" / "best_config.json",
                        Path("task1/pipeline/step3/outputs/configs/best_config.json"),
                    ]
                    if p.exists()
                ),
                None,
            )

    missing = []
    if resolved_chunks is None or not resolved_chunks.exists():
        missing.append("chunks.jsonl")
    if resolved_train is None or not resolved_train.exists():
        missing.append("train_split.json")
    if resolved_dev is None or not resolved_dev.exists():
        missing.append("dev_split.json")
    if resolved_bm25 is None or not resolved_bm25.exists():
        missing.append("dev_rankings_best.jsonl")
    if missing:
        raise FileNotFoundError("Missing Step 4 input(s): " + ", ".join(missing))
    return resolved_chunks, resolved_train, resolved_dev, resolved_bm25, resolved_best_config


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_chunks(chunks_file: Path, *, limit_chunks: int = 0) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    chunks = []
    doc_metadata: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(iter_jsonl(chunks_file)):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        chunk = {
            "chunk_idx": idx,
            "chunk_id": str(row.get("chunk_id", idx)),
            "doc_id": str(row.get("doc_id", "")),
            "text": str(row.get("text", "")),
            "heading": str(row.get("heading", "")),
            "word_count": int(row.get("word_count") or 0),
            "metadata": metadata,
        }
        chunks.append(chunk)
        if chunk["doc_id"] and chunk["doc_id"] not in doc_metadata:
            doc_metadata[chunk["doc_id"]] = metadata | {"heading": chunk["heading"]}
        if limit_chunks and idx + 1 >= limit_chunks:
            break
    return chunks, doc_metadata


def load_bm25_rankings(path: Path, *, limit_queries: int = 0) -> dict[str, list[dict[str, Any]]]:
    rankings = {}
    for idx, row in enumerate(iter_jsonl(path)):
        rankings[str(row["query_id"])] = row.get("top_docs", [])
        if limit_queries and idx + 1 >= limit_queries:
            break
    return rankings


def import_bge_model() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:  # pragma: no cover - depends on Kaggle env
        raise ImportError(
            "FlagEmbedding is required. In Kaggle run: pip install -U FlagEmbedding"
        ) from exc
    return BGEM3FlagModel


def encode_texts(model: Any, texts: list[str], *, batch_size: int, max_length: int) -> np.ndarray:
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = output["dense_vecs"] if isinstance(output, dict) else output
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    return vectors


def build_or_load_embeddings(
    *,
    chunks: list[dict[str, Any]],
    output_dir: Path,
    config: Step4Config,
    force_rebuild: bool,
) -> tuple[np.ndarray, Path]:
    emb_path = output_dir / "embeddings" / "chunk_embeddings_fp16.npy"
    meta_path = output_dir / "embeddings" / "chunk_embedding_meta.json"
    if emb_path.exists() and meta_path.exists() and not force_rebuild:
        print(f"loading cached embeddings: {emb_path}")
        return np.load(emb_path, mmap_mode="r"), emb_path

    BGEM3FlagModel = import_bge_model()
    model = BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16)
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    all_vecs = []
    started = time.time()
    for start in range(0, len(chunks), config.batch_size):
        batch = chunks[start : start + config.batch_size]
        vecs = encode_texts(
            model,
            [row["text"] for row in batch],
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        all_vecs.append(vecs.astype(np.float16))
        if (start // config.batch_size + 1) % 100 == 0:
            print(f"encoded {min(start + config.batch_size, len(chunks)):,}/{len(chunks):,} chunks")
    embeddings = np.vstack(all_vecs)
    np.save(emb_path, embeddings)
    write_json(
        meta_path,
        {
            "model_name": config.model_name,
            "num_chunks": len(chunks),
            "dim": int(embeddings.shape[1]),
            "dtype": "float16",
            "max_length": config.max_length,
            "seconds": round(time.time() - started, 3),
        },
    )
    return np.load(emb_path, mmap_mode="r"), emb_path


def dense_search_torch(
    *,
    chunk_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
    block_size: int,
) -> list[list[tuple[int, float]]]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_tensor = torch.tensor(query_embeddings, dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
    results: list[list[tuple[int, float]]] = []
    for q_idx in range(query_tensor.shape[0]):
        top_scores = None
        top_indices = None
        q = query_tensor[q_idx : q_idx + 1].T
        for start in range(0, chunk_embeddings.shape[0], block_size):
            block_np = np.asarray(chunk_embeddings[start : start + block_size], dtype=np.float16 if device == "cuda" else np.float32)
            block = torch.tensor(block_np, device=device)
            scores = (block @ q).squeeze(1)
            block_k = min(top_k, scores.numel())
            vals, idxs = torch.topk(scores, k=block_k)
            idxs = idxs + start
            if top_scores is None:
                top_scores, top_indices = vals, idxs
            else:
                top_scores = torch.cat([top_scores, vals])
                top_indices = torch.cat([top_indices, idxs])
                vals2, order = torch.topk(top_scores, k=min(top_k, top_scores.numel()))
                top_indices = top_indices[order]
                top_scores = vals2
        results.append(
            [
                (int(idx), float(score))
                for idx, score in zip(top_indices.detach().cpu().tolist(), top_scores.detach().cpu().tolist())
            ]
        )
    return results


def aggregate_dense_docs(
    chunks: list[dict[str, Any]],
    chunk_hits: list[tuple[int, float]],
    config: Step4Config,
) -> list[dict[str, Any]]:
    per_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for chunk_idx, score in chunk_hits[: config.dense_top_chunks]:
        doc_id = chunks[chunk_idx]["doc_id"]
        if doc_id:
            per_doc[doc_id].append((score, chunk_idx))

    rows = []
    for doc_id, scored in per_doc.items():
        scored.sort(reverse=True)
        scores = [s for s, _ in scored]
        max_score = scores[0]
        mean_top3 = sum(scores[:3]) / min(3, len(scores))
        support_count = len(scored)
        doc_score = (
            max_score
            + config.aggregate_mean_top3_weight * mean_top3
            + config.aggregate_support_weight * support_count
        )
        evidence = []
        for score, chunk_idx in scored[: config.evidence_per_doc]:
            row = chunks[chunk_idx]
            evidence.append(
                {
                    "chunk_id": row["chunk_id"],
                    "score": score,
                    "heading": row["heading"],
                    "word_count": row["word_count"],
                    "is_empty_passage_fallback": bool(row["metadata"].get("is_empty_passage_fallback")),
                }
            )
        rows.append(
            {
                "doc_id": doc_id,
                "score": doc_score,
                "max_chunk_score": max_score,
                "mean_top3_chunk_score": mean_top3,
                "support_count": support_count,
                "evidence": evidence,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[: config.dense_top_docs]


def extract_query_metadata_signals(question: str) -> dict[str, set[str]]:
    q = question.lower()
    years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", q))
    doc_types = set()
    for label, patterns in {
        "luat": ["luật", "bộ luật"],
        "nghi_dinh": ["nghị định"],
        "thong_tu": ["thông tư"],
        "quyet_dinh": ["quyết định"],
        "qcvn": ["qcvn", "quy chuẩn"],
        "tcvn": ["tcvn", "tiêu chuẩn"],
    }.items():
        if any(p in q for p in patterns):
            doc_types.add(label)
    numbers = set(re.findall(r"\b\d{1,5}(?:/\d{4})?(?:/[a-zA-ZĐđ0-9.-]+)?\b", question))
    return {"years": years, "doc_types": doc_types, "numbers": numbers}


def metadata_rank(
    question: str,
    candidate_doc_ids: list[str],
    doc_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    signals = extract_query_metadata_signals(question)
    scored = []
    for doc_id in candidate_doc_ids:
        meta = doc_metadata.get(doc_id, {})
        score = 0.0
        years = set(str(y) for y in meta.get("years", []))
        if signals["years"] & years:
            score += 2.0 * len(signals["years"] & years)
        if meta.get("doc_type") in signals["doc_types"]:
            score += 2.0
        hay = " ".join(str(meta.get(key, "")) for key in ["issue_number", "heading"]).lower()
        score += sum(1.0 for number in signals["numbers"] if number and number.lower() in hay)
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in scored]


def rrf_fuse(
    branches: list[tuple[str, list[str], float]],
    *,
    rrf_k: int,
    top_docs: int,
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for _, doc_ids, weight in branches:
        for rank, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] += weight / (rrf_k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return [
        doc_id
        for doc_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]]),
        )[:top_docs]
    ]


def run_queries(
    *,
    model: Any,
    chunk_embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    doc_metadata: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    bm25_rankings: dict[str, list[dict[str, Any]]] | None,
    output_rankings_file: Path,
    config: Step4Config,
    limit_queries: int = 0,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    query_items = list(payload.items())
    if limit_queries:
        query_items = query_items[:limit_queries]
    questions = [row.get("question", "") if isinstance(row, dict) else "" for _, row in query_items]
    print(f"encoding {len(questions):,} queries")
    query_embeddings = encode_texts(
        model,
        questions,
        batch_size=config.query_batch_size,
        max_length=config.max_length,
    )
    print("dense searching queries")
    dense_hits = dense_search_torch(
        chunk_embeddings=chunk_embeddings,
        query_embeddings=query_embeddings,
        top_k=config.dense_top_chunks,
        block_size=config.search_block_size,
    )
    rankings: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for idx, ((qid, row), chunk_hits) in enumerate(zip(query_items, dense_hits), start=1):
            question = row.get("question", "") if isinstance(row, dict) else ""
            dense_docs = aggregate_dense_docs(chunks, chunk_hits, config)
            dense_ids = [doc["doc_id"] for doc in dense_docs]
            bm25_docs = (bm25_rankings or {}).get(str(qid), [])
            bm25_ids = [str(doc["doc_id"]) for doc in bm25_docs]
            union = list(dict.fromkeys(bm25_ids + dense_ids))
            metadata_ids = metadata_rank(question, union, doc_metadata)
            fused_ids = rrf_fuse(
                [
                    ("bm25", bm25_ids, config.bm25_weight),
                    ("dense", dense_ids, config.dense_weight),
                    ("metadata", metadata_ids, config.metadata_weight),
                ],
                rrf_k=config.rrf_k,
                top_docs=config.fused_top_docs,
            )
            rankings[str(qid)] = fused_ids
            predictions[str(qid)] = fused_ids[:MAX_SUBMISSION_DOCS]
            if idx % 100 == 0:
                print(f"fused {idx:,}/{len(query_items):,} queries")
            yield {
                "query_id": str(qid),
                "question": question,
                "gold": row.get("answer", []) if isinstance(row, dict) else [],
                "bm25_top_docs": bm25_docs[: config.fused_top_docs],
                "dense_top_docs": dense_docs,
                "metadata_ranked_doc_ids": metadata_ids,
                "fused_doc_ids": fused_ids,
            }

    append_jsonl(output_rankings_file, rows())
    return rankings, predictions


def run_step4(args: argparse.Namespace) -> dict[str, Any]:
    chunks_file, train_file, dev_file, bm25_dev_file, best_config_file = find_step4_inputs(
        data_root=args.data_root,
        chunks_file=args.chunks_file,
        train_split_file=args.train_split_file,
        dev_split_file=args.dev_split_file,
        bm25_dev_rankings_file=args.bm25_dev_rankings_file,
        best_config_file=args.best_config_file,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Step4Config(
        model_name=args.model_name,
        batch_size=args.batch_size,
        query_batch_size=args.query_batch_size,
        max_length=args.max_length,
        use_fp16=not args.no_fp16,
        dense_top_chunks=args.dense_top_chunks,
        dense_top_docs=args.dense_top_docs,
        evidence_per_doc=args.evidence_per_doc,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        metadata_weight=args.metadata_weight,
        fused_top_docs=args.fused_top_docs,
        search_block_size=args.search_block_size,
    )
    write_json(output_dir / "configs" / "step4_config.json", asdict(config))

    print("loading chunks")
    chunks, doc_metadata = load_chunks(chunks_file, limit_chunks=args.limit_chunks)
    print(f"loaded {len(chunks):,} chunks; docs={len(doc_metadata):,}")
    embeddings, emb_path = build_or_load_embeddings(
        chunks=chunks,
        output_dir=output_dir,
        config=config,
        force_rebuild=args.force_rebuild_embeddings,
    )

    BGEM3FlagModel = import_bge_model()
    model = BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16)

    dev_payload = read_json(dev_file)
    bm25_dev = load_bm25_rankings(bm25_dev_file, limit_queries=args.limit_queries)
    dev_rankings, dev_predictions = run_queries(
        model=model,
        chunk_embeddings=embeddings,
        chunks=chunks,
        doc_metadata=doc_metadata,
        payload=dev_payload,
        bm25_rankings=bm25_dev,
        output_rankings_file=output_dir / "rankings" / "dev_rankings_rrf.jsonl",
        config=config,
        limit_queries=args.limit_queries,
    )
    dev_eval_payload = dict(list(dev_payload.items())[: args.limit_queries]) if args.limit_queries else dev_payload
    dev_metrics = evaluate_rankings(dev_rankings, dev_eval_payload)
    write_json(output_dir / "metrics" / "dev_metrics_rrf.json", dev_metrics)
    write_json(output_dir / "predictions" / "dev_predictions_top5_rrf.json", dev_predictions)

    train_metrics = None
    if args.eval_train:
        train_payload = read_json(train_file)
        bm25_train = (
            load_bm25_rankings(args.bm25_train_rankings_file, limit_queries=args.limit_queries)
            if args.bm25_train_rankings_file
            else None
        )
        train_rankings, train_predictions = run_queries(
            model=model,
            chunk_embeddings=embeddings,
            chunks=chunks,
            doc_metadata=doc_metadata,
            payload=train_payload,
            bm25_rankings=bm25_train,
            output_rankings_file=output_dir / "rankings" / "train_rankings_rrf.jsonl",
            config=config,
            limit_queries=args.limit_queries,
        )
        train_eval_payload = dict(list(train_payload.items())[: args.limit_queries]) if args.limit_queries else train_payload
        train_metrics = evaluate_rankings(train_rankings, train_eval_payload)
        write_json(output_dir / "metrics" / "train_metrics_rrf.json", train_metrics)
        write_json(output_dir / "predictions" / "train_predictions_top5_rrf.json", train_predictions)

    public_outputs = None
    if args.predict_public:
        public_file = find_public_file(args.public_file, args.data_root) or DEFAULT_PUBLIC_FILE
        if not public_file.exists():
            raise FileNotFoundError(f"Cannot locate public file: {public_file}")
        public_payload = read_json(public_file)
        # For public we do not have BM25 public rankings in the Step 4 minimal
        # input set, so dense+metadata is used unless --bm25-public-rankings-file is passed.
        bm25_public = (
            load_bm25_rankings(args.bm25_public_rankings_file, limit_queries=args.limit_queries)
            if args.bm25_public_rankings_file
            else None
        )
        public_rankings, public_predictions = run_queries(
            model=model,
            chunk_embeddings=embeddings,
            chunks=chunks,
            doc_metadata=doc_metadata,
            payload=public_payload,
            bm25_rankings=bm25_public,
            output_rankings_file=output_dir / "rankings" / "public_rankings_rrf.jsonl",
            config=config,
            limit_queries=args.limit_queries,
        )
        submission = make_submission(public_predictions)
        public_eval_payload = dict(list(public_payload.items())[: args.limit_queries]) if args.limit_queries else public_payload
        validation = validate_submission_payload(
            submission,
            public_eval_payload,
            {chunk["doc_id"] for chunk in chunks},
        )
        submission_dir = output_dir / "submission"
        write_json(submission_dir / "submission.json", submission)
        write_json(submission_dir / "submission_validation.json", validation)
        if validation["num_errors"]:
            raise ValueError(f"Submission validation failed: {validation['num_errors']} errors")
        write_submission_zip(submission_dir / "submission.json", submission_dir / "submission.zip")
        public_outputs = {
            "rankings": "rankings/public_rankings_rrf.jsonl",
            "submission_zip": "submission/submission.zip",
            "submission_validation": "submission/submission_validation.json",
        }

    report = {
        "inputs": {
            "chunks_file": str(chunks_file),
            "train_split_file": str(train_file),
            "dev_split_file": str(dev_file),
            "bm25_dev_rankings_file": str(bm25_dev_file),
            "best_config_file": str(best_config_file) if best_config_file else None,
        },
        "config": asdict(config),
        "embeddings_file": str(emb_path),
        "dev_macro": dev_metrics["macro"],
        "train_macro": train_metrics["macro"] if train_metrics else None,
        "public_outputs": public_outputs,
        "next_gpu_steps": {
            "step5_biencoder_finetune": [
                "chunks.jsonl",
                "train_split.json",
                "dev_split.json",
                "rankings/dev_rankings_rrf.jsonl",
                "optional rankings/train_rankings_rrf.jsonl if mined on train",
            ]
        },
        "sources": {
            "bge_m3_model_card": "https://huggingface.co/BAAI/bge-m3"
        },
    }
    write_json(output_dir / "reports" / "run_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--chunks-file", type=Path, default=None)
    parser.add_argument("--train-split-file", type=Path, default=None)
    parser.add_argument("--dev-split-file", type=Path, default=None)
    parser.add_argument("--bm25-dev-rankings-file", type=Path, default=None)
    parser.add_argument("--bm25-train-rankings-file", type=Path, default=None)
    parser.add_argument("--bm25-public-rankings-file", type=Path, default=None)
    parser.add_argument("--best-config-file", type=Path, default=None)
    parser.add_argument("--public-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dense-top-chunks", type=int, default=300)
    parser.add_argument("--dense-top-docs", type=int, default=100)
    parser.add_argument("--evidence-per-doc", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--metadata-weight", type=float, default=0.20)
    parser.add_argument("--fused-top-docs", type=int, default=100)
    parser.add_argument("--search-block-size", type=int, default=32768)
    parser.add_argument("--predict-public", action="store_true")
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Debug only.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Debug only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_step4(args)
    print(json.dumps(report["dev_macro"], ensure_ascii=False, indent=2))
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
