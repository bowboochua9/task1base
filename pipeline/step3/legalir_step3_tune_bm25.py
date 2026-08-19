"""Step 3 Chunk-BM25 tuning for UIT DSC 2026 LegalIR.

This step follows plan_pipeline.md:
- tune Chunk-BM25 before moving to dense retrieval
- compare retrieval depth and aggregation weights on dev
- keep a best BM25 candidate artifact for Step 4 RRF/dense work

The script imports the Step 2 dependency-free BM25 implementation and adds an
efficient ablation runner. For each scoring group (k1, b, heading_weight), BM25
chunk scores are computed once per query, then multiple aggregation configs are
evaluated from the same top chunk pool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
STEP2_DIR = THIS_DIR.parent / "step2"
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))

from legalir_step2_bm25 import (  # noqa: E402
    BM25ChunkIndex,
    BM25Config,
    DEFAULT_KAGGLE_DATA_ROOT,
    DEFAULT_KAGGLE_PUBLIC_FILE,
    MAX_SUBMISSION_DOCS,
    append_jsonl,
    evaluate_rankings,
    find_public_file,
    find_step2_inputs,
    make_submission,
    read_json,
    validate_submission_payload,
    write_json,
    write_submission_zip,
)


DEFAULT_OUTPUT = (
    Path("/kaggle/working/step3")
    if Path("/kaggle").exists()
    else Path("task1/pipeline/step3/outputs")
)


@dataclass(frozen=True)
class TrialConfig:
    name: str
    k1: float = 1.5
    b: float = 0.75
    heading_weight: float = 2.0
    top_chunks: int = 300
    top_docs: int = 100
    evidence_per_doc: int = 3
    mean_top3_weight: float = 0.20
    support_weight: float = 0.05
    use_deaccent: bool = True
    use_stopwords: bool = True

    def scoring_key(self) -> tuple[float, float, float, bool, bool]:
        return (
            self.k1,
            self.b,
            self.heading_weight,
            self.use_deaccent,
            self.use_stopwords,
        )

    def to_bm25_config(self, *, top_chunks: int | None = None) -> BM25Config:
        return BM25Config(
            k1=self.k1,
            b=self.b,
            top_chunks=self.top_chunks if top_chunks is None else top_chunks,
            top_docs=self.top_docs,
            evidence_per_doc=self.evidence_per_doc,
            aggregate_mean_top3_weight=self.mean_top3_weight,
            aggregate_support_weight=self.support_weight,
            heading_weight=self.heading_weight,
            use_deaccent=self.use_deaccent,
            use_stopwords=self.use_stopwords,
        )


def default_trials(include_heading_ablation: bool = False) -> list[TrialConfig]:
    trials = [
        TrialConfig("baseline"),
        TrialConfig("top500", top_chunks=500),
        TrialConfig("top800", top_chunks=800),
        TrialConfig("support0", support_weight=0.0),
        TrialConfig("support002", support_weight=0.02),
        TrialConfig("support010", support_weight=0.10),
        TrialConfig("mean010", mean_top3_weight=0.10),
        TrialConfig("mean030", mean_top3_weight=0.30),
        TrialConfig("k1_1p2", k1=1.2),
        TrialConfig("k1_1p8", k1=1.8),
        TrialConfig("b_0p55", b=0.55),
        TrialConfig("b_0p90", b=0.90),
    ]
    if include_heading_ablation:
        trials.extend(
            [
                TrialConfig("heading1", heading_weight=1.0),
                TrialConfig("heading3", heading_weight=3.0),
            ]
        )
    return trials


def parse_trial_config(raw: str) -> TrialConfig:
    payload = json.loads(raw)
    return TrialConfig(**payload)


def load_trials(args: argparse.Namespace) -> list[TrialConfig]:
    if args.best_config_file:
        payload = read_json(args.best_config_file)
        if "best_config" in payload and isinstance(payload["best_config"], dict):
            payload = payload["best_config"]
        if "config" in payload and isinstance(payload["config"], dict):
            payload = payload["config"]
        return [TrialConfig(**payload)]
    if args.trials_file:
        payload = read_json(args.trials_file)
        if not isinstance(payload, list):
            raise ValueError("--trials-file must contain a JSON list.")
        return [TrialConfig(**item) for item in payload]
    if args.trial_json:
        return [parse_trial_config(raw) for raw in args.trial_json]
    return default_trials(include_heading_ablation=args.include_heading_ablation)


def aggregate_chunk_scores(
    index: BM25ChunkIndex,
    chunk_scores: dict[int, float],
    config: TrialConfig,
) -> list[dict[str, Any]]:
    if not chunk_scores:
        return []
    max_pool = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[
        : config.top_chunks
    ]
    per_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for chunk_idx, score in max_pool:
        doc_id = index.chunk_meta[chunk_idx].doc_id
        if doc_id:
            per_doc[doc_id].append((score, chunk_idx))

    rows = []
    for doc_id, scored_chunks in per_doc.items():
        scored_chunks.sort(reverse=True)
        scores = [score for score, _ in scored_chunks]
        max_score = scores[0]
        mean_top3 = sum(scores[:3]) / min(3, len(scores))
        support_count = len(scored_chunks)
        doc_score = (
            max_score
            + config.mean_top3_weight * mean_top3
            + config.support_weight * support_count
        )
        evidence = []
        for score, chunk_idx in scored_chunks[: config.evidence_per_doc]:
            meta = index.chunk_meta[chunk_idx]
            evidence.append(
                {
                    "chunk_id": meta.chunk_id,
                    "score": score,
                    "heading": meta.heading,
                    "word_count": meta.word_count,
                    "is_empty_passage_fallback": meta.is_empty_passage_fallback,
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
    return rows[: config.top_docs]


def metrics_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    macro = item["metrics"]["macro"]
    return (
        macro.get("recall@5", 0.0),
        macro.get("precision@5", 0.0),
        macro.get("recall@20", 0.0),
        macro.get("mrr", 0.0),
    )


def evaluate_trial_group(
    *,
    index: BM25ChunkIndex,
    query_payload: dict[str, Any],
    trials: list[TrialConfig],
    limit_queries: int = 0,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]], dict[str, list[dict[str, Any]]]]:
    query_items = list(query_payload.items())
    if limit_queries:
        query_items = query_items[:limit_queries]

    rankings_by_trial: dict[str, dict[str, list[str]]] = {
        trial.name: {} for trial in trials
    }
    top_docs_by_trial: dict[str, list[dict[str, Any]]] = {trial.name: [] for trial in trials}

    started = time.time()
    for pos, (qid, row) in enumerate(query_items, start=1):
        question = row.get("question", "") if isinstance(row, dict) else ""
        chunk_scores = index.score_query_chunks(question)
        for trial in trials:
            top_docs = aggregate_chunk_scores(index, chunk_scores, trial)
            doc_ids = [item["doc_id"] for item in top_docs]
            rankings_by_trial[trial.name][str(qid)] = doc_ids
            top_docs_by_trial[trial.name].append(
                {
                    "query_id": str(qid),
                    "question": question,
                    "gold": row.get("answer", []) if isinstance(row, dict) else [],
                    "top_docs": top_docs,
                }
            )
        if progress_every and pos % progress_every == 0:
            print(f"ranked {pos:,}/{len(query_items):,} queries in {time.time() - started:.1f}s")

    eval_payload = dict(query_items)
    summaries = []
    for trial in trials:
        metrics = evaluate_rankings(rankings_by_trial[trial.name], eval_payload)
        summaries.append(
            {
                "name": trial.name,
                "config": asdict(trial),
                "metrics": metrics,
            }
        )
    summaries.sort(key=metrics_sort_key, reverse=True)
    return summaries, rankings_by_trial, top_docs_by_trial


def write_top_docs_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    append_jsonl(path, rows)


def write_predictions(path: Path, rankings: dict[str, list[str]]) -> None:
    write_json(
        path,
        {
            qid: doc_ids[:MAX_SUBMISSION_DOCS]
            for qid, doc_ids in sorted(rankings.items(), key=lambda item: item[0])
        },
    )


def run_best_on_payload(
    *,
    index: BM25ChunkIndex,
    payload: dict[str, Any],
    config: TrialConfig,
    output_rankings_file: Path,
    limit_queries: int = 0,
    progress_every: int = 100,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    query_items = list(payload.items())
    if limit_queries:
        query_items = query_items[:limit_queries]
    rankings: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}

    def rows() -> Iterable[dict[str, Any]]:
        started = time.time()
        for pos, (qid, row) in enumerate(query_items, start=1):
            question = row.get("question", "") if isinstance(row, dict) else ""
            top_docs = aggregate_chunk_scores(index, index.score_query_chunks(question), config)
            doc_ids = [item["doc_id"] for item in top_docs]
            rankings[str(qid)] = doc_ids
            predictions[str(qid)] = doc_ids[:MAX_SUBMISSION_DOCS]
            if progress_every and pos % progress_every == 0:
                print(f"ranked {pos:,}/{len(query_items):,} best queries in {time.time() - started:.1f}s")
            yield {
                "query_id": str(qid),
                "question": question,
                "gold": row.get("answer", []) if isinstance(row, dict) else [],
                "top_docs": top_docs,
            }

    append_jsonl(output_rankings_file, rows())
    return rankings, predictions


def run_step3(args: argparse.Namespace) -> dict[str, Any]:
    chunks_file, train_split_file, dev_split_file = find_step2_inputs(
        args.data_root, args.chunks_file, args.train_split_file, args.dev_split_file
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = read_json(train_split_file)
    dev_payload = read_json(dev_split_file)
    trials = load_trials(args)
    if len({trial.name for trial in trials}) != len(trials):
        raise ValueError("Trial names must be unique.")

    write_json(output_dir / "configs" / "trial_configs.json", [asdict(trial) for trial in trials])

    all_summaries = []
    best_payload = None
    best_index = None
    best_trial = None
    best_rankings = None
    best_top_docs = None

    groups: dict[tuple[float, float, float, bool, bool], list[TrialConfig]] = defaultdict(list)
    for trial in trials:
        groups[trial.scoring_key()].append(trial)

    for group_idx, (scoring_key, group_trials) in enumerate(groups.items(), start=1):
        representative = group_trials[0]
        max_top_chunks = max(trial.top_chunks for trial in group_trials)
        build_config = representative.to_bm25_config(top_chunks=max_top_chunks)
        print(
            f"building index group {group_idx}/{len(groups)} "
            f"k1={representative.k1} b={representative.b} "
            f"heading={representative.heading_weight} trials={len(group_trials)}"
        )
        index = BM25ChunkIndex(build_config)
        index_stats = index.build(
            chunks_file,
            progress_every=args.progress_every_chunks,
            limit_chunks=args.limit_chunks,
        )
        print("evaluating dev group")
        summaries, rankings_by_trial, top_docs_by_trial = evaluate_trial_group(
            index=index,
            query_payload=dev_payload,
            trials=group_trials,
            limit_queries=args.limit_queries,
            progress_every=args.progress_every_queries,
        )
        for summary in summaries:
            summary["index"] = index_stats
            summary["scoring_key"] = list(scoring_key)
        all_summaries.extend(summaries)
        group_best = summaries[0]
        if best_payload is None or metrics_sort_key(group_best) > metrics_sort_key(best_payload):
            best_payload = group_best
            best_index = index
            best_trial = next(trial for trial in group_trials if trial.name == group_best["name"])
            best_rankings = rankings_by_trial[best_trial.name]
            best_top_docs = top_docs_by_trial[best_trial.name]

    all_summaries.sort(key=metrics_sort_key, reverse=True)
    write_json(output_dir / "metrics" / "ablation_summary.json", all_summaries)
    write_json(output_dir / "configs" / "best_config.json", asdict(best_trial))
    write_json(output_dir / "metrics" / "best_trial_summary.json", best_payload)

    if best_trial is None or best_index is None or best_rankings is None or best_top_docs is None:
        raise RuntimeError("No best trial was selected.")

    write_json(output_dir / "metrics" / "dev_metrics_best.json", best_payload["metrics"])
    write_predictions(output_dir / "predictions" / "dev_predictions_top5_best.json", best_rankings)
    write_top_docs_jsonl(output_dir / "rankings" / "dev_rankings_best.jsonl", best_top_docs)

    train_metrics = None
    if args.eval_train:
        print("ranking train with best config")
        train_rankings, train_predictions = run_best_on_payload(
            index=best_index,
            payload=train_payload,
            config=best_trial,
            output_rankings_file=output_dir / "rankings" / "train_rankings_best.jsonl",
            limit_queries=args.limit_queries,
            progress_every=args.progress_every_queries,
        )
        train_eval_payload = (
            dict(list(train_payload.items())[: args.limit_queries])
            if args.limit_queries
            else train_payload
        )
        train_metrics = evaluate_rankings(train_rankings, train_eval_payload)
        write_json(output_dir / "metrics" / "train_metrics_best.json", train_metrics)
        write_predictions(output_dir / "predictions" / "train_predictions_top5_best.json", train_rankings)

    public_outputs = None
    if args.predict_public:
        public_file = find_public_file(args.public_file, args.data_root)
        if public_file is None:
            raise FileNotFoundError(
                "Cannot locate public-official.json. Pass --public-file when using --predict-public."
            )
        print("ranking public with best config")
        public_payload = read_json(public_file)
        public_rankings, public_predictions = run_best_on_payload(
            index=best_index,
            payload=public_payload,
            config=best_trial,
            output_rankings_file=output_dir / "rankings" / "public_rankings_best.jsonl",
            limit_queries=args.limit_queries,
            progress_every=args.progress_every_queries,
        )
        submission = make_submission(public_predictions)
        public_eval_payload = (
            dict(list(public_payload.items())[: args.limit_queries])
            if args.limit_queries
            else public_payload
        )
        validation = validate_submission_payload(
            submission,
            public_eval_payload,
            {meta.doc_id for meta in best_index.chunk_meta},
        )
        submission_dir = output_dir / "submission"
        write_json(submission_dir / "submission.json", submission)
        write_json(submission_dir / "submission_validation.json", validation)
        if validation["num_errors"]:
            raise ValueError(
                f"Submission validation failed with {validation['num_errors']} errors. "
                f"See {submission_dir / 'submission_validation.json'}"
            )
        write_submission_zip(submission_dir / "submission.json", submission_dir / args.submission_zip_name)
        public_outputs = {
            "public_file": str(public_file),
            "public_rankings_best": "rankings/public_rankings_best.jsonl",
            "submission_json": "submission/submission.json",
            "submission_zip": f"submission/{args.submission_zip_name}",
            "submission_validation": "submission/submission_validation.json",
        }

    report = {
        "inputs": {
            "chunks_file": str(chunks_file),
            "train_split_file": str(train_split_file),
            "dev_split_file": str(dev_split_file),
            "output_dir": str(output_dir),
        },
        "num_trials": len(trials),
        "num_scoring_groups": len(groups),
        "best_trial": best_trial.name,
        "best_config": asdict(best_trial),
        "best_dev_macro": best_payload["metrics"]["macro"],
        "train_macro": train_metrics["macro"] if train_metrics else None,
        "public_outputs": public_outputs,
        "next_step_gpu_inputs": {
            "step4_bge_rrf": [
                "Step 1 data: chunks.jsonl, train_split.json, dev_split.json",
                "Step 3 BM25 candidates: rankings/dev_rankings_best.jsonl",
                "Optional for training/mining later: rankings/train_rankings_best.jsonl",
            ]
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
    parser.add_argument("--public-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-config-file", type=Path, default=None)
    parser.add_argument("--trials-file", type=Path, default=None)
    parser.add_argument("--trial-json", action="append", default=[])
    parser.add_argument("--include-heading-ablation", action="store_true")
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--predict-public", action="store_true")
    parser.add_argument("--submission-zip-name", default="submission.zip")
    parser.add_argument("--progress-every-chunks", type=int, default=25000)
    parser.add_argument("--progress-every-queries", type=int, default=100)
    parser.add_argument("--limit-chunks", type=int, default=0, help="Debug only.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Debug only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_step3(args)
    print(json.dumps(report["best_dev_macro"], ensure_ascii=False, indent=2))
    print(f"Best trial: {report['best_trial']}")
    print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
