from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SEED = 42
FAST_STACKER_QUERIES = 1000
N_FOLDS = 5
INCLUDE_MANIFEST_INPUT_FILES = [
    "split_manifest.json",
    "feature_contract.json",
    "preflight_report.json",
    "gpu_notebook_static_audit_report.json",
]
REQUIRED_PHASES = {"fast", "oof_fold", "fulltrain_dev", "merge"}
FORBIDDEN_NOTEBOOK_PATTERNS = [
    "other_research",
    "DSC26_weight_report",
    "CODE_ROOT",
    "/kaggle/input/dscuit2026/dscuit2026",
    "train_rankings_step5_fused.jsonl",
    "dev_rankings_step5_fused.jsonl",
    "public_rankings_step5_fused.jsonl",
    "train_rankings_step6_fused.jsonl",
    "hf_hub_download",
    "requests.",
    "http://",
    "https://",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qid_sort_key(qid: str) -> tuple[int, Any]:
    text = str(qid)
    return (0, int(text)) if text.isdigit() else (1, text)


def unique_ids(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_split(path: Path) -> dict[str, dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Split must be a dict keyed by query_id: {path}")
    out: dict[str, dict[str, Any]] = {}
    for qid, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid split row for query_id={qid!r}")
        gold = item.get("answer", item.get("gold"))
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"Missing non-empty answer/gold list for query_id={qid!r}")
        out[str(qid)] = {"question": str(item.get("question", "")), "gold": unique_ids(gold)}
    return out


def default_audit_root(workspace: Path) -> Path:
    return workspace.resolve() / "task1" / "pipeline" / "step9_p1_audit"


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(2026, 8, 18, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def stratify_labels(split: dict[str, dict[str, Any]], qids: list[str]) -> list[int]:
    return [min(len(split[qid]["gold"]), 4) for qid in qids]


def make_fast_holdout(split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    qids = sorted(split, key=qid_sort_key)
    labels = stratify_labels(split, qids)
    base_train, stacker = train_test_split(
        qids,
        test_size=FAST_STACKER_QUERIES,
        random_state=SEED,
        stratify=labels,
    )
    return {
        "seed": SEED,
        "protocol": "fast_holdout",
        "base_train_query_ids": sorted(map(str, base_train), key=qid_sort_key),
        "stacker_train_query_ids": sorted(map(str, stacker), key=qid_sort_key),
        "num_base_train_queries": len(base_train),
        "num_stacker_train_queries": len(stacker),
    }


def make_oof_folds(split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    qids = sorted(split, key=qid_sort_key)
    labels = stratify_labels(split, qids)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds: list[dict[str, Any]] = []
    for fold_idx, (train_idx, heldout_idx) in enumerate(skf.split(qids, labels), start=1):
        train_qids = [qids[i] for i in train_idx]
        heldout_qids = [qids[i] for i in heldout_idx]
        folds.append(
            {
                "fold": fold_idx,
                "base_train_query_ids": sorted(train_qids, key=qid_sort_key),
                "heldout_query_ids": sorted(heldout_qids, key=qid_sort_key),
                "num_base_train_queries": len(train_qids),
                "num_heldout_queries": len(heldout_qids),
            }
        )
    return {"seed": SEED, "protocol": "5fold_oof", "n_folds": N_FOLDS, "folds": folds}


def make_feature_contract() -> dict[str, Any]:
    return {
        "status": "contract_only_not_final_manifest",
        "final_manifest_name": "manifest.json",
        "final_manifest_rule": "Only the GPU/merge job that writes all required parquet artifacts may create manifest.json with status=ok.",
        "required_outputs": {
            "fast_holdout_features.parquet": {
                "scope": "Features for the 1000 fast-holdout stacker-train queries, produced by base models trained on the disjoint 5000-query base-train split.",
                "required_columns": [
                    "query_id",
                    "doc_id",
                    "label",
                    "source_split",
                    "feature_protocol",
                    "step4_rank",
                    "step5_oof_rank",
                    "step5_oof_score",
                    "step6_oof_rank",
                    "step6_oof_score",
                ],
            },
            "train_oof_features.parquet": {
                "scope": "OOF features for all 6000 canonical-train queries; learned branch features for each query must come from a checkpoint that did not train on that query.",
                "required_columns": [
                    "query_id",
                    "doc_id",
                    "label",
                    "fold",
                    "source_split",
                    "feature_protocol",
                    "step4_rank",
                    "step5_oof_rank",
                    "step5_oof_score",
                    "step6_oof_rank",
                    "step6_oof_score",
                ],
            },
            "dev_features.parquet": {
                "scope": "Canonical dev features. Dev labels may be included for final evaluation only, never for model or config selection.",
                "required_columns": [
                    "query_id",
                    "doc_id",
                    "label",
                    "source_split",
                    "feature_protocol",
                    "step4_rank",
                    "step5_fulltrain_rank",
                    "step5_fulltrain_score",
                    "step6_fulltrain_rank",
                    "step6_fulltrain_score",
                ],
            },
            "manifest.json": {
                "scope": "Final provenance manifest for the three parquet artifacts.",
                "required_fields": [
                    "status",
                    "protocol",
                    "created_by_notebook",
                    "input_paths",
                    "output_paths",
                    "checksums",
                    "model_manifests",
                    "leakage_guards",
                ],
            },
        },
        "rejected_inputs": [
            "task1/pipeline/step5/step5/rankings/train_rankings_step5_fused.jsonl as learned Step5 train feature, because it is in-sample for canonical train.",
            "Any Step6 train ranking produced by a reranker checkpoint trained on the same query.",
            "Any artifact from DSC2026_baseline for final compliant P1.2, because P1.0 is blocked and split differs.",
        ],
    }


def cmd_prepare_manifest(args: argparse.Namespace) -> None:
    audit_root = default_audit_root(args.workspace)
    pipeline_root = audit_root.parent
    train_split_path = pipeline_root / "step1" / "outputs" / "train_split.json"
    dev_split_path = pipeline_root / "step1" / "outputs" / "dev_split.json"
    output_dir = args.output_dir or (audit_root / "p1_2_clean_base_features")

    missing = [str(path) for path in [train_split_path, dev_split_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(json.dumps({"missing_required_inputs": missing}, ensure_ascii=False, indent=2))

    train_split = load_split(train_split_path)
    dev_split = load_split(dev_split_path)
    if set(train_split) & set(dev_split):
        raise ValueError("Canonical train/dev query IDs overlap; refusing to write P1.2 split manifests.")

    split_manifest = {
        "status": "ok",
        "purpose": "P1.2 clean base-feature generation split manifest. This is not the final feature manifest.",
        "seed": SEED,
        "num_canonical_train_queries": len(train_split),
        "num_canonical_dev_queries": len(dev_split),
        "train_split": {"path": str(train_split_path), "sha256": sha256_file(train_split_path)},
        "dev_split": {"path": str(dev_split_path), "sha256": sha256_file(dev_split_path)},
        "fast_holdout": make_fast_holdout(train_split),
        "oof": make_oof_folds(train_split),
        "canonical_dev_rule": "Canonical dev is not used in base-model training, XGB training, validation, grid selection, or fold selection; use only for final locked evaluation.",
    }
    preflight = {
        "status": "pending_gpu_feature_generation",
        "output_dir": str(output_dir),
        "created_files": {
            "split_manifest": str(output_dir / "split_manifest.json"),
            "feature_contract": str(output_dir / "feature_contract.json"),
        },
        "missing_final_outputs": [
            str(output_dir / "fast_holdout_features.parquet"),
            str(output_dir / "train_oof_features.parquet"),
            str(output_dir / "dev_features.parquet"),
            str(output_dir / "manifest.json"),
        ],
        "next_gpu_notebook": "task1/pipeline/step9_p1_audit/kaggle_p1_2_generate_clean_base_features_gpu.ipynb",
    }
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "feature_contract.json", make_feature_contract())
    write_json(output_dir / "preflight_report.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))


def notebook_text(path: Path) -> tuple[str, int, bool]:
    nb = read_json(path)
    if nb.get("nbformat") != 4:
        raise ValueError(f"Unexpected notebook format for {path}: {nb.get('nbformat')}")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"Notebook has no cells: {path}")
    text_parts: list[str] = []
    all_have_ids = True
    for cell in cells:
        if "id" not in cell:
            all_have_ids = False
        source = cell.get("source", "")
        text_parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(text_parts), len(cells), all_have_ids


def assert_disjoint(name: str, left: Iterable[str], right: Iterable[str]) -> None:
    overlap = set(map(str, left)) & set(map(str, right))
    if overlap:
        head = sorted(overlap, key=qid_sort_key)[:20]
        raise ValueError(f"{name}: expected disjoint query ids, found {len(overlap)} overlap; head={head}")


def audit_splits(split_manifest: dict[str, Any], train_split: dict[str, Any], dev_split: dict[str, Any]) -> dict[str, Any]:
    if split_manifest.get("status") != "ok":
        raise ValueError(f"split_manifest status must be ok, got {split_manifest.get('status')!r}")
    train_qids = set(train_split)
    dev_qids = set(dev_split)
    assert_disjoint("canonical train/dev", train_qids, dev_qids)

    fast = split_manifest["fast_holdout"]
    fast_base = set(map(str, fast["base_train_query_ids"]))
    fast_stacker = set(map(str, fast["stacker_train_query_ids"]))
    assert_disjoint("fast base/stacker", fast_base, fast_stacker)
    if fast_base | fast_stacker != train_qids:
        raise ValueError("fast split does not cover canonical train exactly")

    heldout_all: list[str] = []
    for fold in split_manifest["oof"]["folds"]:
        base = set(map(str, fold["base_train_query_ids"]))
        heldout = set(map(str, fold["heldout_query_ids"]))
        assert_disjoint(f"oof fold {fold['fold']} base/heldout", base, heldout)
        if base | heldout != train_qids:
            raise ValueError(f"oof fold {fold['fold']} does not cover canonical train exactly")
        heldout_all.extend(heldout)
    if sorted(heldout_all, key=qid_sort_key) != sorted(train_qids, key=qid_sort_key):
        raise ValueError("OOF heldout folds do not cover each canonical train query exactly once")

    return {
        "canonical_train_queries": len(train_qids),
        "canonical_dev_queries": len(dev_qids),
        "fast_base_queries": len(fast_base),
        "fast_stacker_queries": len(fast_stacker),
        "oof_folds": len(split_manifest["oof"]["folds"]),
        "oof_heldout_total": len(heldout_all),
    }


def audit_notebook_payload(notebook_body: str, *, ipynb_cells: int, ipynb_all_have_ids: bool) -> dict[str, Any]:
    missing_phases = sorted(phase for phase in REQUIRED_PHASES if phase not in notebook_body)
    if missing_phases:
        raise ValueError(f"Notebook missing required phase strings: {missing_phases}")
    forbidden_hits = [pattern for pattern in FORBIDDEN_NOTEBOOK_PATTERNS if pattern in notebook_body]
    if forbidden_hits:
        raise ValueError(f"Forbidden notebook patterns found: {forbidden_hits}")
    required_snippets = [
        'MANIFEST_ROOT = Path(',
        '"/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_clean_base_features_input"',
        'SPLIT_MANIFEST_PATH = MANIFEST_ROOT / "split_manifest.json"',
        'FEATURE_CONTRACT_PATH = MANIFEST_ROOT / "feature_contract.json"',
        '"P1_2_TRAIN_RANKINGS_PATH"',
        '"/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl"',
        'BASE_MODEL_SOURCE = os.environ.get("P1_2_BASE_MODEL_SOURCE", "hf_download")',
        "def resolve_base_model_dir(repo_id: str) -> Path:",
        "snapshot_download(",
        "MODEL_REVISIONS = {",
        'BASE_BI_ENCODER_DIR = resolve_base_model_dir("bkai-foundation-models/vietnamese-bi-encoder")',
        'BASE_RERANKER_DIR = resolve_base_model_dir("AITeamVN/Vietnamese_Reranker")',
        "SentenceTransformer(str(BASE_BI_ENCODER_DIR)",
        "AutoTokenizer.from_pretrained(str(BASE_RERANKER_DIR)",
        "AutoModelForSequenceClassification.from_pretrained(str(BASE_RERANKER_DIR)",
        'DATA_ROOT = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26")',
        "set(train_split) & set(dev_split)",
        "fast_holdout_features.parquet",
        "train_oof_features.parquet",
        "dev_features.parquet",
        "manifest.json",
    ]
    missing_snippets = [snippet for snippet in required_snippets if snippet not in notebook_body]
    if missing_snippets:
        raise ValueError(f"Required notebook snippets missing: {missing_snippets}")
    if ipynb_cells < 10:
        raise ValueError(f"GPU notebook unexpectedly small: {ipynb_cells} cells")
    if not ipynb_all_have_ids:
        raise ValueError("GPU notebook has cells without ids")
    check_idx = notebook_body.find("missing = {name: str(path) for name, path in PATHS.items()")
    model_idx = notebook_body.find('BASE_BI_ENCODER_DIR = resolve_base_model_dir("bkai-foundation-models/vietnamese-bi-encoder")')
    if check_idx < 0 or model_idx < 0 or check_idx > model_idx:
        raise ValueError("GPU notebook must validate required dataset inputs before downloading/loading base models")
    return {
        "ipynb_cells": ipynb_cells,
        "ipynb_cells_have_ids": ipynb_all_have_ids,
        "required_phases": sorted(REQUIRED_PHASES),
        "forbidden_pattern_hits": [],
    }


def audit_contract(contract: dict[str, Any]) -> dict[str, Any]:
    required_outputs = contract.get("required_outputs", {})
    expected_outputs = {
        "fast_holdout_features.parquet",
        "train_oof_features.parquet",
        "dev_features.parquet",
        "manifest.json",
    }
    missing = sorted(expected_outputs - set(required_outputs))
    if missing:
        raise ValueError(f"feature_contract missing outputs: {missing}")
    if contract.get("status") != "contract_only_not_final_manifest":
        raise ValueError(f"Unexpected feature_contract status: {contract.get('status')!r}")
    return {"status": contract["status"], "required_outputs": sorted(required_outputs)}


def cmd_audit_notebook(args: argparse.Namespace) -> None:
    audit_root = default_audit_root(args.workspace)
    pipeline_root = audit_root.parent
    feature_dir = args.feature_dir or (audit_root / "p1_2_clean_base_features")
    output_path = feature_dir / "gpu_notebook_static_audit_report.json"
    paths = {
        "notebook": audit_root / "kaggle_p1_2_generate_clean_base_features_gpu.ipynb",
        "split_manifest": feature_dir / "split_manifest.json",
        "feature_contract": feature_dir / "feature_contract.json",
        "train_split": pipeline_root / "step1" / "outputs" / "train_split.json",
        "dev_split": pipeline_root / "step1" / "outputs" / "dev_split.json",
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps({"missing_required_inputs": missing}, ensure_ascii=False, indent=2))

    ipynb_text, ipynb_cells, ipynb_all_have_ids = notebook_text(paths["notebook"])
    split_manifest = read_json(paths["split_manifest"])
    contract = read_json(paths["feature_contract"])
    train_split = load_split(paths["train_split"])
    dev_split = load_split(paths["dev_split"])
    report = {
        "status": "ok",
        "purpose": "Static audit for P1.2 GPU notebook before Kaggle GPU execution.",
        "paths": {name: str(path) for name, path in paths.items()},
        "notebook": audit_notebook_payload(
            ipynb_text,
            ipynb_cells=ipynb_cells,
            ipynb_all_have_ids=ipynb_all_have_ids,
        ),
        "split_protocol": audit_splits(split_manifest, train_split, dev_split),
        "feature_contract": audit_contract(contract),
        "remaining_runtime_requirement": "Run GPU phases on Kaggle T4x2; local static audit does not prove GPU runtime success.",
    }
    write_json(output_path, report, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_package_manifest_input(args: argparse.Namespace) -> None:
    audit_root = default_audit_root(args.workspace)
    feature_dir = args.feature_dir or (audit_root / "p1_2_clean_base_features")
    zip_path = args.zip_path or (audit_root / "p1_2_clean_base_features_input.zip")
    manifest_path = feature_dir / "input_zip_manifest.json"
    missing = [str(feature_dir / name) for name in INCLUDE_MANIFEST_INPUT_FILES if not (feature_dir / name).exists()]
    if missing:
        raise FileNotFoundError(json.dumps({"missing_required_inputs": missing}, ensure_ascii=False, indent=2))
    if read_json(feature_dir / "split_manifest.json").get("status") != "ok":
        raise ValueError("split_manifest status must be ok")
    if read_json(feature_dir / "feature_contract.json").get("status") != "contract_only_not_final_manifest":
        raise ValueError("feature_contract status unexpected")
    if read_json(feature_dir / "gpu_notebook_static_audit_report.json").get("status") != "ok":
        raise ValueError("gpu_notebook_static_audit_report status unexpected")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE_MANIFEST_INPUT_FILES:
            zf.writestr(fixed_zip_info(name), (feature_dir / name).read_bytes())
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
    if namelist != INCLUDE_MANIFEST_INPUT_FILES:
        raise ValueError(f"Unexpected zip namelist/order: {namelist}")
    if any("\\" in name or name.startswith("/") for name in namelist):
        raise ValueError(f"ZIP arcnames must be relative forward-slash paths only: {namelist}")
    payload = {
        "status": "ok",
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_namelist": namelist,
        "kaggle_location_current": "/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_clean_base_features_input",
        "notebook_manifest_root_default": "/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_clean_base_features_input",
        "contains_helper_py": False,
        "contains_model_or_ranking_artifact": False,
    }
    write_json(manifest_path, payload, sort_keys=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_package_train_rankings(args: argparse.Namespace) -> None:
    audit_root = default_audit_root(args.workspace)
    source_path = args.source or (audit_root.parent / "step3" / "outputs" / "rankings" / "train_rankings_best.jsonl")
    zip_path = args.zip_path or (audit_root / "p1_2_step3_train_rankings_input.zip")
    manifest_path = audit_root / "p1_2_step3_train_rankings_input_manifest.json"
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    lines = 0
    qids: set[str] = set()
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines += 1
                qids.add(str(json.loads(line).get("query_id")))
    if len(qids) != 6000:
        raise ValueError({"expected_qids": 6000, "actual_unique_qids": len(qids), "lines": lines})

    arcname = "step3/outputs/rankings/train_rankings_best.jsonl"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(fixed_zip_info(arcname), source_path.read_bytes())
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
    bad_arcnames = [name for name in namelist if "\\" in name or name.startswith("/")]
    payload = {
        "status": "ok",
        "source": str(source_path),
        "source_size": source_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        "lines": lines,
        "unique_qids": len(qids),
        "zip_path": str(zip_path),
        "zip_size": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "zip_namelist": namelist,
        "bad_arcnames": bad_arcnames,
        "expected_kaggle_path": "/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl",
    }
    if bad_arcnames:
        raise ValueError(json.dumps(payload, ensure_ascii=False, indent=2))
    write_json(manifest_path, payload, sort_keys=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def required_columns(contract: dict[str, Any], output_name: str) -> list[str]:
    return list(contract["required_outputs"][output_name]["required_columns"])


def validate_frame(
    df: Any,
    *,
    name: str,
    expected_qids: list[str],
    required_cols: list[str],
    expected_folds: list[int] | None = None,
) -> dict[str, Any]:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing required columns: {missing_cols}")
    got_qids = set(map(str, df["query_id"].unique()))
    expected = set(map(str, expected_qids))
    if got_qids != expected:
        raise ValueError(
            json.dumps(
                {
                    "artifact": name,
                    "qid_mismatch": {
                        "missing_count": len(expected - got_qids),
                        "extra_count": len(got_qids - expected),
                        "missing_head": sorted(expected - got_qids, key=qid_sort_key)[:20],
                        "extra_head": sorted(got_qids - expected, key=qid_sort_key)[:20],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if expected_folds is not None:
        got_folds = sorted(int(x) for x in df["fold"].unique())
        if got_folds != expected_folds:
            raise ValueError(f"{name}: expected folds {expected_folds}, got {got_folds}")
    positives = df.groupby("query_id", sort=False)["label"].sum()
    no_positive = [str(qid) for qid, value in positives.items() if value <= 0]
    if no_positive:
        raise ValueError(f"{name}: {len(no_positive)} queries have no positive candidate; head={no_positive[:20]}")
    return {"rows": int(len(df)), "queries": int(df["query_id"].nunique()), "positive_rows": int(df["label"].sum())}


def cmd_validate_merge(args: argparse.Namespace) -> None:
    import pandas as pd

    audit_root = default_audit_root(args.workspace)
    pipeline_root = audit_root.parent
    feature_dir = args.feature_dir or (audit_root / "p1_2_clean_base_features")
    parts_dir = feature_dir / "parts"
    split_manifest_path = feature_dir / "split_manifest.json"
    contract_path = feature_dir / "feature_contract.json"
    train_split_path = pipeline_root / "step1" / "outputs" / "train_split.json"
    dev_split_path = pipeline_root / "step1" / "outputs" / "dev_split.json"
    required_inputs = [split_manifest_path, contract_path, train_split_path, dev_split_path]
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(json.dumps({"missing_required_inputs": missing_inputs}, ensure_ascii=False, indent=2))

    split_manifest = read_json(split_manifest_path)
    contract = read_json(contract_path)
    train_split = load_split(train_split_path)
    dev_split = load_split(dev_split_path)
    if set(train_split) & set(dev_split):
        raise ValueError("Canonical train/dev query IDs overlap; refusing to validate P1.2 features.")

    part_paths = {
        "fast": parts_dir / "fast_holdout_features.parquet",
        "dev": parts_dir / "dev_features.parquet",
        **{f"oof_fold_{i}": parts_dir / f"oof_fold_{i}_features.parquet" for i in range(1, N_FOLDS + 1)},
    }
    missing_parts = {name: str(path) for name, path in part_paths.items() if not path.exists()}
    report_path = feature_dir / "merge_validation_report.json"
    if missing_parts:
        report = {
            "status": "blocked_missing_gpu_outputs",
            "missing_parts": missing_parts,
            "feature_dir": str(feature_dir),
            "note": "Run kaggle_p1_2_generate_clean_base_features_gpu.ipynb phases fast, oof_fold=1..5, and fulltrain_dev before merge.",
        }
        write_json(report_path, report, sort_keys=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    fast_df = pd.read_parquet(part_paths["fast"])
    oof_parts = [pd.read_parquet(part_paths[f"oof_fold_{i}"]) for i in range(1, N_FOLDS + 1)]
    oof_df = pd.concat(oof_parts, ignore_index=True)
    dev_df = pd.read_parquet(part_paths["dev"])
    fast_qids = list(map(str, split_manifest["fast_holdout"]["stacker_train_query_ids"]))
    train_qids = sorted(train_split, key=qid_sort_key)
    dev_qids = sorted(dev_split, key=qid_sort_key)
    stats = {
        "fast_holdout_features.parquet": validate_frame(
            fast_df,
            name="fast_holdout_features.parquet",
            expected_qids=fast_qids,
            required_cols=required_columns(contract, "fast_holdout_features.parquet"),
        ),
        "train_oof_features.parquet": validate_frame(
            oof_df,
            name="train_oof_features.parquet",
            expected_qids=train_qids,
            required_cols=required_columns(contract, "train_oof_features.parquet"),
            expected_folds=list(range(1, N_FOLDS + 1)),
        ),
        "dev_features.parquet": validate_frame(
            dev_df,
            name="dev_features.parquet",
            expected_qids=dev_qids,
            required_cols=required_columns(contract, "dev_features.parquet"),
        ),
    }
    final_paths = {
        "fast_holdout_features.parquet": feature_dir / "fast_holdout_features.parquet",
        "train_oof_features.parquet": feature_dir / "train_oof_features.parquet",
        "dev_features.parquet": feature_dir / "dev_features.parquet",
    }
    fast_df.to_parquet(final_paths["fast_holdout_features.parquet"], index=False)
    oof_df.to_parquet(final_paths["train_oof_features.parquet"], index=False)
    dev_df.to_parquet(final_paths["dev_features.parquet"], index=False)
    manifest = {
        "status": "ok",
        "protocol": "p1_2_clean_base_features_fast_holdout_and_5fold_oof",
        "created_by": "p1_2_tools.py validate-merge",
        "created_from_gpu_parts": {name: str(path) for name, path in part_paths.items()},
        "output_paths": {name: str(path) for name, path in final_paths.items()},
        "checksums": {name: sha256_file(path) for name, path in final_paths.items()},
        "input_checksums": {
            "split_manifest.json": sha256_file(split_manifest_path),
            "feature_contract.json": sha256_file(contract_path),
        },
        "stats": stats,
        "leakage_guards": {
            "canonical_train_dev_query_overlap": False,
            "fast_holdout_target_queries_excluded_from_base_training": True,
            "oof_target_queries_excluded_from_corresponding_base_training": True,
            "canonical_dev_used_for_training_or_selection": False,
            "no_external_DSC2026_baseline_features": True,
            "no_in_sample_step5_or_step6_train_rankings_used_as_stacker_features": True,
        },
    }
    write_json(feature_dir / "manifest.json", manifest, sort_keys=True)
    report = {"status": "ok", "manifest": str(feature_dir / "manifest.json"), "stats": stats}
    write_json(report_path, report, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1.2 local CPU maintenance tools.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--output-dir", type=Path, default=None)
    prepare.set_defaults(func=cmd_prepare_manifest)

    audit = sub.add_parser("audit-notebook")
    audit.add_argument("--feature-dir", type=Path, default=None)
    audit.set_defaults(func=cmd_audit_notebook)

    package_manifest = sub.add_parser("package-manifest-input")
    package_manifest.add_argument("--feature-dir", type=Path, default=None)
    package_manifest.add_argument("--zip-path", type=Path, default=None)
    package_manifest.set_defaults(func=cmd_package_manifest_input)

    package_rankings = sub.add_parser("package-train-rankings")
    package_rankings.add_argument("--source", type=Path, default=None)
    package_rankings.add_argument("--zip-path", type=Path, default=None)
    package_rankings.set_defaults(func=cmd_package_train_rankings)

    validate = sub.add_parser("validate-merge")
    validate.add_argument("--feature-dir", type=Path, default=None)
    validate.set_defaults(func=cmd_validate_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
