"""Step 2 Chunk-BM25 baseline for UIT DSC 2026 LegalIR.

Inputs are the data artifacts from Step 1:
- chunks.jsonl
- train_split.json
- dev_split.json

The implementation is dependency-free and CPU-only. It builds a BM25 inverted
index over chunks, retrieves top chunks, aggregates chunk scores to document
scores, then evaluates the document ranking on the dev split.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import statistics
import time
import zipfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26")
DEFAULT_KAGGLE_PUBLIC_FILE = Path(
    "/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json"
)
DEFAULT_OUTPUT = (
    Path("/kaggle/working/step2")
    if Path("/kaggle").exists()
    else Path("task1/pipeline/step2/outputs")
)
MAX_SUBMISSION_DOCS = 5


STOPWORDS = {
    "a",
    "an",
    "anh",
    "ay",
    "bi",
    "boi",
    "cac",
    "can",
    "cho",
    "co",
    "con",
    "cua",
    "duoc",
    "da",
    "de",
    "den",
    "di",
    "do",
    "doi",
    "duoi",
    "gi",
    "hay",
    "hoac",
    "khi",
    "la",
    "lai",
    "lam",
    "mot",
    "nay",
    "neu",
    "nhu",
    "nhung",
    "o",
    "phai",
    "qua",
    "quy",
    "rieng",
    "sau",
    "se",
    "thi",
    "the",
    "theo",
    "thi",
    "trong",
    "tu",
    "va",
    "ve",
    "viec",
    "voi",
}


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_chunks: int = 300
    top_docs: int = 100
    evidence_per_doc: int = 3
    aggregate_mean_top3_weight: float = 0.20
    aggregate_support_weight: float = 0.05
    heading_weight: float = 2.0
    use_deaccent: bool = True
    use_stopwords: bool = True


@dataclass
class ChunkMeta:
    chunk_id: str
    doc_id: str
    heading: str
    word_count: int
    is_empty_passage_fallback: bool


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_submission_zip(submission_json: Path, submission_zip: Path) -> None:
    submission_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        submission_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write(submission_json, arcname="submission.json")


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
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str, *, use_deaccent: bool, use_stopwords: bool) -> list[str]:
    if use_deaccent:
        text = strip_accents(text)
    text = text.lower()
    tokens = re.findall(r"[0-9a-zA-Z_]+", text)
    cleaned = []
    for token in tokens:
        if len(token) > 40:
            continue
        if use_stopwords and token in STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def split_heading_body(text: str) -> tuple[str, str]:
    if "\n" not in text:
        return "", text
    heading, body = text.split("\n", 1)
    return heading.strip(), body.strip()


def find_step2_inputs(
    data_root: Path | None,
    chunks_file: Path | None,
    train_split_file: Path | None,
    dev_split_file: Path | None,
) -> tuple[Path, Path, Path]:
    root_candidates = []
    if data_root is not None:
        root_candidates.append(data_root)
    root_candidates.extend(
        [
            DEFAULT_KAGGLE_DATA_ROOT,
            Path("task1/pipeline/step1/outputs"),
            Path("task1/pipeline/step1/outputs/corpus"),
            Path("."),
        ]
    )

    resolved_chunks = chunks_file
    resolved_train = train_split_file
    resolved_dev = dev_split_file

    for root in root_candidates:
        chunk_candidates = [
            root / "chunks.jsonl",
            root / "corpus" / "chunks.jsonl",
        ]
        train_candidates = [
            root / "train_split.json",
            root / "splits" / "train_split.json",
        ]
        dev_candidates = [
            root / "dev_split.json",
            root / "splits" / "dev_split.json",
        ]
        if resolved_chunks is None:
            resolved_chunks = next((p for p in chunk_candidates if p.exists()), None)
        if resolved_train is None:
            resolved_train = next((p for p in train_candidates if p.exists()), None)
        if resolved_dev is None:
            resolved_dev = next((p for p in dev_candidates if p.exists()), None)

    missing = []
    if resolved_chunks is None or not resolved_chunks.exists():
        missing.append("chunks.jsonl")
    if resolved_train is None or not resolved_train.exists():
        missing.append("train_split.json")
    if resolved_dev is None or not resolved_dev.exists():
        missing.append("dev_split.json")
    if missing:
        raise FileNotFoundError(
            "Cannot locate Step 2 input(s): "
            + ", ".join(missing)
            + ". Pass --data-root or explicit file paths."
        )
    return resolved_chunks, resolved_train, resolved_dev


def find_public_file(public_file: Path | None, data_root: Path | None) -> Path | None:
    if public_file is not None:
        if not public_file.exists():
            raise FileNotFoundError(f"Cannot locate public file: {public_file}")
        return public_file

    candidates = []
    if data_root is not None:
        candidates.extend([data_root / "public-official.json", data_root / "public.json"])
    candidates.extend(
        [
            DEFAULT_KAGGLE_PUBLIC_FILE,
            Path("task1/public-official.json"),
            Path("public-official.json"),
        ]
    )
    return next((path for path in candidates if path.exists()), None)


class BM25ChunkIndex:
    def __init__(self, config: BM25Config) -> None:
        self.config = config
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.doc_freq: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.chunk_lengths: list[float] = []
        self.chunk_meta: list[ChunkMeta] = []
        self.avgdl = 0.0

    def _chunk_token_weights(self, row: dict[str, Any]) -> Counter[str]:
        text = row.get("text", "")
        heading, body = split_heading_body(text if isinstance(text, str) else "")
        counter: Counter[str] = Counter(
            tokenize(body, use_deaccent=self.config.use_deaccent, use_stopwords=self.config.use_stopwords)
        )
        if heading:
            heading_counter = Counter(
                tokenize(
                    heading,
                    use_deaccent=self.config.use_deaccent,
                    use_stopwords=self.config.use_stopwords,
                )
            )
            for token, count in heading_counter.items():
                counter[token] += count * self.config.heading_weight
        return counter

    def build(
        self,
        chunks_file: Path,
        *,
        progress_every: int = 25000,
        limit_chunks: int = 0,
    ) -> dict[str, Any]:
        started = time.time()
        fallback_chunks = 0
        with chunks_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                is_fallback = bool(metadata.get("is_empty_passage_fallback"))
                fallback_chunks += int(is_fallback)
                self.chunk_meta.append(
                    ChunkMeta(
                        chunk_id=str(row.get("chunk_id", idx)),
                        doc_id=str(row.get("doc_id", "")),
                        heading=str(row.get("heading", "")),
                        word_count=int(row.get("word_count") or 0),
                        is_empty_passage_fallback=is_fallback,
                    )
                )
                token_weights = self._chunk_token_weights(row)
                self.chunk_lengths.append(float(sum(token_weights.values())))
                for token, tf in token_weights.items():
                    self.postings[token].append((idx, float(tf)))
                if progress_every and (idx + 1) % progress_every == 0:
                    print(f"indexed {idx + 1:,} chunks; vocab={len(self.postings):,}")
                if limit_chunks and idx + 1 >= limit_chunks:
                    break

        num_chunks = len(self.chunk_meta)
        self.avgdl = statistics.fmean(self.chunk_lengths) if self.chunk_lengths else 0.0
        self.doc_freq = {token: len(posting) for token, posting in self.postings.items()}
        self.idf = {
            token: math.log(1.0 + (num_chunks - df + 0.5) / (df + 0.5))
            for token, df in self.doc_freq.items()
        }
        return {
            "num_chunks": num_chunks,
            "num_docs": len({meta.doc_id for meta in self.chunk_meta}),
            "num_terms": len(self.postings),
            "avgdl": self.avgdl,
            "fallback_chunks": fallback_chunks,
            "build_seconds": round(time.time() - started, 3),
        }

    def score_query_chunks(self, query: str) -> dict[int, float]:
        query_terms = Counter(
            tokenize(
                query,
                use_deaccent=self.config.use_deaccent,
                use_stopwords=self.config.use_stopwords,
            )
        )
        scores: defaultdict[int, float] = defaultdict(float)
        if not query_terms or not self.avgdl:
            return {}

        k1 = self.config.k1
        b = self.config.b
        for token, qtf in query_terms.items():
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf[token]
            for chunk_idx, tf in posting:
                dl = self.chunk_lengths[chunk_idx]
                denom = tf + k1 * (1.0 - b + b * dl / self.avgdl)
                scores[chunk_idx] += qtf * idf * (tf * (k1 + 1.0) / denom)
        return scores

    def rank(self, query: str) -> list[dict[str, Any]]:
        chunk_scores = self.score_query_chunks(query)
        if not chunk_scores:
            return []

        top_chunk_items = heapq.nlargest(
            self.config.top_chunks, chunk_scores.items(), key=lambda item: item[1]
        )
        per_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for chunk_idx, score in top_chunk_items:
            doc_id = self.chunk_meta[chunk_idx].doc_id
            if doc_id:
                per_doc[doc_id].append((score, chunk_idx))

        doc_rows = []
        for doc_id, scored_chunks in per_doc.items():
            scored_chunks.sort(reverse=True)
            scores = [score for score, _ in scored_chunks]
            max_score = scores[0]
            mean_top3 = statistics.fmean(scores[:3])
            support_count = len(scored_chunks)
            doc_score = (
                max_score
                + self.config.aggregate_mean_top3_weight * mean_top3
                + self.config.aggregate_support_weight * support_count
            )
            evidence = []
            for score, chunk_idx in scored_chunks[: self.config.evidence_per_doc]:
                meta = self.chunk_meta[chunk_idx]
                evidence.append(
                    {
                        "chunk_id": meta.chunk_id,
                        "score": score,
                        "heading": meta.heading,
                        "word_count": meta.word_count,
                        "is_empty_passage_fallback": meta.is_empty_passage_fallback,
                    }
                )
            doc_rows.append(
                {
                    "doc_id": doc_id,
                    "score": doc_score,
                    "max_chunk_score": max_score,
                    "mean_top3_chunk_score": mean_top3,
                    "support_count": support_count,
                    "evidence": evidence,
                }
            )

        doc_rows.sort(key=lambda row: row["score"], reverse=True)
        return doc_rows[: self.config.top_docs]


def ranking_metrics_for_query(ranked_doc_ids: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = {str(doc_id) for doc_id in gold}
    if not gold_set:
        return {
            "precision@5": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
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
    precision5 = len(set(top5) & gold_set) / MAX_SUBMISSION_DOCS
    first_rank = next(
        (idx + 1 for idx, doc_id in enumerate(ranked_doc_ids) if doc_id in gold_set),
        None,
    )
    return {
        "precision@5": precision5,
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
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


def evaluate_rankings(
    rankings: dict[str, list[str]],
    query_payload: dict[str, Any],
) -> dict[str, Any]:
    sums: Counter[str] = Counter()
    by_gold_count: dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_gold_count: Counter[str] = Counter()
    per_query = {}

    for qid, row in query_payload.items():
        gold = row.get("answer", []) if isinstance(row, dict) else []
        metrics = ranking_metrics_for_query(rankings.get(str(qid), []), gold)
        per_query[str(qid)] = metrics
        sums.update(metrics)
        gold_count_key = str(len(gold))
        by_gold_count[gold_count_key].update(metrics)
        counts_by_gold_count[gold_count_key] += 1

    n = len(query_payload)
    macro = {key: (value / n if n else 0.0) for key, value in sorted(sums.items())}
    breakdown = {
        key: {
            metric: value / counts_by_gold_count[key]
            for metric, value in sorted(counter.items())
        }
        for key, counter in sorted(by_gold_count.items(), key=lambda item: int(item[0]))
    }
    return {
        "num_queries": n,
        "macro": macro,
        "by_gold_count": breakdown,
        "per_query": per_query,
    }


def make_submission(predictions_top5: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    return {
        str(qid): {"answer": [str(doc_id) for doc_id in doc_ids[:MAX_SUBMISSION_DOCS]]}
        for qid, doc_ids in predictions_top5.items()
    }


def validate_submission_payload(
    submission: Any,
    public_payload: dict[str, Any],
    valid_doc_ids: set[str],
) -> dict[str, Any]:
    issues = []
    if not isinstance(submission, dict):
        return {"num_errors": 1, "num_warnings": 0, "issues": ["root is not object"]}

    public_ids = {str(qid) for qid in public_payload}
    submission_ids = {str(qid) for qid in submission}
    for qid in sorted(public_ids - submission_ids)[:50]:
        issues.append({"severity": "error", "kind": "missing_query", "query_id": qid})
    for qid in sorted(submission_ids - public_ids)[:50]:
        issues.append({"severity": "error", "kind": "extra_query", "query_id": qid})

    answer_lengths: Counter[int] = Counter()
    for qid, row in submission.items():
        if not isinstance(row, dict):
            issues.append({"severity": "error", "kind": "row_not_object", "query_id": str(qid)})
            continue
        answer = row.get("answer")
        if not isinstance(answer, list):
            issues.append({"severity": "error", "kind": "answer_not_array", "query_id": str(qid)})
            continue
        answer_lengths[len(answer)] += 1
        if not (1 <= len(answer) <= MAX_SUBMISSION_DOCS):
            issues.append(
                {
                    "severity": "error",
                    "kind": "answer_count",
                    "query_id": str(qid),
                    "message": f"answer must contain 1-{MAX_SUBMISSION_DOCS} document IDs",
                }
            )
        normalized = [str(doc_id) for doc_id in answer]
        if any(not isinstance(doc_id, str) for doc_id in answer):
            issues.append({"severity": "error", "kind": "doc_id_not_string", "query_id": str(qid)})
        if len(normalized) != len(set(normalized)):
            issues.append({"severity": "error", "kind": "duplicate_doc_id", "query_id": str(qid)})
        for doc_id in normalized:
            if doc_id not in valid_doc_ids:
                issues.append(
                    {
                        "severity": "error",
                        "kind": "unknown_doc_id",
                        "query_id": str(qid),
                        "doc_id": doc_id,
                    }
                )

    return {
        "num_public_queries": len(public_ids),
        "num_submission_queries": len(submission_ids),
        "answer_length_distribution": dict(sorted(answer_lengths.items())),
        "num_errors": sum(issue["severity"] == "error" for issue in issues),
        "num_warnings": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
    }


def run_queries(
    index: BM25ChunkIndex,
    payload: dict[str, Any],
    *,
    output_rankings_file: Path,
    limit_queries: int = 0,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rankings: dict[str, list[str]] = {}
    predictions_top5: dict[str, list[str]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        started = time.time()
        query_items = list(payload.items())
        if limit_queries:
            query_items = query_items[:limit_queries]
        for pos, (qid, row) in enumerate(query_items, start=1):
            question = row.get("question", "") if isinstance(row, dict) else ""
            ranking_rows = index.rank(question)
            doc_ids = [item["doc_id"] for item in ranking_rows]
            rankings[str(qid)] = doc_ids
            predictions_top5[str(qid)] = doc_ids[:MAX_SUBMISSION_DOCS]
            if pos % 100 == 0:
                print(f"ranked {pos:,}/{len(query_items):,} queries in {time.time() - started:.1f}s")
            yield {
                "query_id": str(qid),
                "question": question,
                "gold": row.get("answer", []) if isinstance(row, dict) else [],
                "top_docs": ranking_rows,
            }

    append_jsonl(output_rankings_file, rows())
    return rankings, predictions_top5


def run_step2(args: argparse.Namespace) -> dict[str, Any]:
    chunks_file, train_split_file, dev_split_file = find_step2_inputs(
        args.data_root, args.chunks_file, args.train_split_file, args.dev_split_file
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BM25Config(
        k1=args.k1,
        b=args.b,
        top_chunks=args.top_chunks,
        top_docs=args.top_docs,
        evidence_per_doc=args.evidence_per_doc,
        aggregate_mean_top3_weight=args.mean_top3_weight,
        aggregate_support_weight=args.support_weight,
        heading_weight=args.heading_weight,
        use_deaccent=not args.no_deaccent,
        use_stopwords=not args.no_stopwords,
    )

    print("loading query splits")
    train_payload = read_json(train_split_file)
    dev_payload = read_json(dev_split_file)
    print("building BM25 chunk index")
    index = BM25ChunkIndex(config)
    index_stats = index.build(
        chunks_file,
        progress_every=args.progress_every,
        limit_chunks=args.limit_chunks,
    )

    print("ranking dev queries")
    dev_rankings, dev_top5 = run_queries(
        index,
        dev_payload,
        output_rankings_file=output_dir / "rankings" / "dev_rankings.jsonl",
        limit_queries=args.limit_queries,
    )
    dev_eval_payload = dict(list(dev_payload.items())[: args.limit_queries]) if args.limit_queries else dev_payload
    dev_metrics = evaluate_rankings(dev_rankings, dev_eval_payload)
    write_json(output_dir / "predictions" / "dev_predictions_top5.json", dev_top5)
    write_json(output_dir / "metrics" / "dev_metrics.json", dev_metrics)

    train_metrics = None
    if args.eval_train:
        print("ranking train queries")
        train_rankings, train_top5 = run_queries(
            index,
            train_payload,
            output_rankings_file=output_dir / "rankings" / "train_rankings.jsonl",
            limit_queries=args.limit_queries,
        )
        train_eval_payload = dict(list(train_payload.items())[: args.limit_queries]) if args.limit_queries else train_payload
        train_metrics = evaluate_rankings(train_rankings, train_eval_payload)
        write_json(output_dir / "predictions" / "train_predictions_top5.json", train_top5)
        write_json(output_dir / "metrics" / "train_metrics.json", train_metrics)

    public_outputs = None
    if args.predict_public:
        public_file = find_public_file(args.public_file, args.data_root)
        if public_file is None:
            raise FileNotFoundError(
                "Cannot locate public-official.json. Pass --public-file when using --predict-public."
            )
        print("ranking public queries")
        public_payload = read_json(public_file)
        _, public_top5 = run_queries(
            index,
            public_payload,
            output_rankings_file=output_dir / "rankings" / "public_rankings.jsonl",
            limit_queries=args.limit_queries,
        )
        public_eval_payload = (
            dict(list(public_payload.items())[: args.limit_queries])
            if args.limit_queries
            else public_payload
        )
        submission = make_submission(public_top5)
        submission_json = output_dir / "submission" / "submission.json"
        submission_zip = output_dir / "submission" / args.submission_zip_name
        write_json(submission_json, submission)
        submission_report = validate_submission_payload(
            submission,
            public_eval_payload,
            {meta.doc_id for meta in index.chunk_meta},
        )
        write_json(output_dir / "submission" / "submission_validation.json", submission_report)
        if submission_report["num_errors"]:
            raise ValueError(
                f"Submission validation failed with {submission_report['num_errors']} errors. "
                f"See {output_dir / 'submission' / 'submission_validation.json'}"
            )
        write_submission_zip(submission_json, submission_zip)
        public_outputs = {
            "public_file": str(public_file),
            "public_rankings": "rankings/public_rankings.jsonl",
            "submission_json": "submission/submission.json",
            "submission_zip": f"submission/{args.submission_zip_name}",
            "submission_validation": "submission/submission_validation.json",
        }

    run_report = {
        "inputs": {
            "chunks_file": str(chunks_file),
            "train_split_file": str(train_split_file),
            "dev_split_file": str(dev_split_file),
            "output_dir": str(output_dir),
        },
        "config": asdict(config),
        "index": index_stats,
        "dev_macro": dev_metrics["macro"],
        "train_macro": train_metrics["macro"] if train_metrics else None,
        "outputs": {
            "dev_rankings": "rankings/dev_rankings.jsonl",
            "dev_predictions_top5": "predictions/dev_predictions_top5.json",
            "dev_metrics": "metrics/dev_metrics.json",
        },
        "public_outputs": public_outputs,
    }
    write_json(output_dir / "reports" / "run_report.json", run_report)
    return run_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--chunks-file", type=Path, default=None)
    parser.add_argument("--train-split-file", type=Path, default=None)
    parser.add_argument("--dev-split-file", type=Path, default=None)
    parser.add_argument("--public-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--top-chunks", type=int, default=300)
    parser.add_argument("--top-docs", type=int, default=100)
    parser.add_argument("--evidence-per-doc", type=int, default=3)
    parser.add_argument("--mean-top3-weight", type=float, default=0.20)
    parser.add_argument("--support-weight", type=float, default=0.05)
    parser.add_argument("--heading-weight", type=float, default=2.0)
    parser.add_argument("--no-deaccent", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--predict-public", action="store_true")
    parser.add_argument("--submission-zip-name", default="submission.zip")
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--limit-chunks", type=int, default=0, help="Debug only.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Debug only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_step2(args)
    print(json.dumps(report["dev_macro"], ensure_ascii=False, indent=2))
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
