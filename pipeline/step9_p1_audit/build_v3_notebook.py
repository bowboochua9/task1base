"""Build clean P1.2 GPU notebook with proper cell structure."""
import json
import os

OUTPUT_PATH = r"D:\ds\dscuit2026\task1\pipeline\step9_p1_audit\kaggle_p1_2_generate_clean_base_features_gpu_v3.ipynb"

# --- Helper: build a cell ---
def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").split("\n")
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").split("\n")
    }

cells = []

# ============================================================
# CELL 1: Header / overview
# ============================================================
cells.append(md("""
# P1.2 Clean Base Feature Generation — GPU

**Phase notebook for generating the missing `p1_2_clean_base_features` artifacts.**

## Phases

| Phase | Purpose | Time |
|-------|---------|------|
| `fast` | Train Step5/Step6 on 5000-query base split, score 1000-query stacker split | ~3h |
| `oof_fold_<N>` | Train on 4 folds, score held-out fold N (N=1..5) | ~1h each |
| `fulltrain_dev` | Train on all canonical train, score canonical dev | ~2h |
| `merge` | Validate GPU parts and write final parquet + manifest | ~10 min |
| `all` | Run all 7 subphases sequentially in one session | ~9-10h total |

> **Default: `all`** — set `P1_2_GPU_PHASE` environment variable to override.

## Hard constraints (validated at runtime)

1. No train query receives learned Step5/Step6 features from a checkpoint that trained on that query.
2. Base models always download from HuggingFace (internet enabled) — never from a full-train Step5/Step6 checkpoint.
3. Final `manifest.json` is only created by the merge phase after all parts pass validation.
"""))

# ============================================================
# CELL 2: Install dependencies
# ============================================================
cells.append(code("""
import sys, subprocess

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "sentence-transformers", "transformers", "accelerate",
    "safetensors", "sentencepiece", "pyarrow", "huggingface-hub",
], check=True)
print("✓ dependencies installed")
"""))

# ============================================================
# CELL 3: Imports + config
# ============================================================
cells.append(code("""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc, hashlib, json, math, random, re, time, unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---- Runtime config (env vars override defaults) ----
RUN_PHASE   = os.environ.get("P1_2_GPU_PHASE", "all").strip().lower()   # fast|oof_fold|fulltrain_dev|merge|all
OOF_FOLD    = int(os.environ.get("P1_2_OOF_FOLD", "1"))                # 1..5 when phase=oof_fold
DRY_RUN     = os.environ.get("P1_2_DRY_RUN", "0").strip() == "1"        # 1=subsample for smoke test

# Hardcoded to merged-input path; internet is enabled so we always hf_download
DATA_ROOT        = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input")
SPLIT_MANIFEST   = DATA_ROOT / "split_manifest.json"
FEATURE_CONTRACT = DATA_ROOT / "feature_contract.json"
TRAIN_RANKINGS   = DATA_ROOT / "step3" / "outputs" / "rankings" / "train_rankings_best.jsonl"
TRAIN_SPLIT      = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/train_split.json")
DEV_SPLIT        = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json")
CHUNKS           = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl")
DEV_RANKINGS     = Path("/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_rankings_best.jsonl")

OUTPUT_DIR  = Path("/kaggle/working/p1_2_clean_base_features")
PARTS_DIR   = OUTPUT_DIR / "parts"
MODELS_DIR  = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
HF_CACHE    = Path("/kaggle/working/p1_2_hf_models")

BASE_BI_ENCODER_ID = "bkai-foundation-models/vietnamese-bi-encoder"
BASE_BI_ENCODER_REV = "84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0"
BASE_RERANKER_ID    = "AITeamVN/Vietnamese_Reranker"
BASE_RERANKER_REV   = "f536976248403314225d7fdfdbc87f0e9516a54e"

ALLOWED_MODELS = {BASE_BI_ENCODER_ID, BASE_RERANKER_ID}
MAX_DOCS, MISSING_RANK = 200, 999

if RUN_PHASE not in {"fast", "oof_fold", "fulltrain_dev", "merge", "all"}:
    raise ValueError(f"Invalid P1_2_GPU_PHASE={RUN_PHASE!r}")
if RUN_PHASE == "oof_fold" and not 1 <= OOF_FOLD <= 5:
    raise ValueError(f"OOF_FOLD must be in [1,5], got {OOF_FOLD}")

print(json.dumps({
    "run_phase": RUN_PHASE,
    "oof_fold": OOF_FOLD if RUN_PHASE == "oof_fold" else None,
    "dry_run": DRY_RUN,
    "data_root": str(DATA_ROOT),
    "output_dir": str(OUTPUT_DIR),
    "hf_download": True,
}, indent=2))
"""))

# ============================================================
# CELL 4: Validate inputs + download HF models
# ============================================================
cells.append(code("""
from huggingface_hub import snapshot_download

missing = [p for p in [SPLIT_MANIFEST, FEATURE_CONTRACT, TRAIN_RANKINGS, TRAIN_SPLIT, DEV_SPLIT, CHUNKS, DEV_RANKINGS] if not p.exists()]
if missing:
    raise FileNotFoundError("Missing inputs: " + ", ".join(str(p) for p in missing))
print("✓ all required input files present")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)

# Download base models from HuggingFace (internet is enabled)
print("Downloading bkai-foundation-models/vietnamese-bi-encoder ...")
BASE_BI_ENCODER_DIR = snapshot_download(
    repo_id=BASE_BI_ENCODER_ID, repo_type="model", revision=BASE_BI_ENCODER_REV,
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bpe.codes"],
    local_dir=str(HF_CACHE / BASE_BI_ENCODER_ID.replace("/", "--")),
)
print(f"  → {BASE_BI_ENCODER_DIR}")

print("Downloading AITeamVN/Vietnamese_Reranker ...")
BASE_RERANKER_DIR = snapshot_download(
    repo_id=BASE_RERANKER_ID, repo_type="model", revision=BASE_RERANKER_REV,
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer.json"],
    local_dir=str(HF_CACHE / BASE_RERANKER_ID.replace("/", "--")),
)
print(f"  → {BASE_RERANKER_DIR}")

print("✓ base models cached locally")
"""))

# ============================================================
# CELL 5: Utility functions
# ============================================================
cells.append(code("""
def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True); f.write("\n")

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def append_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def qid_sort_key(qid):
    t = str(qid)
    return (0, int(t)) if t.isdigit() else (1, t)

def unique_ids(values):
    out, seen = [], set()
    for v in values:
        s = str(v)
        if s not in seen: seen.add(s); out.append(s)
    return out

def load_split(path: Path):
    raw = read_json(path)
    out = {}
    for qid, row in raw.items():
        gold = row.get("answer", row.get("gold"))
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"missing gold for {qid}")
        out[str(qid)] = {"question": str(row.get("question", "")), "gold": unique_ids(gold)}
    return out

def load_rankings(path: Path, qids=None):
    rows = {}
    for row in iter_jsonl(path):
        qid = str(row.get("query_id"))
        if qids is None or qid in qids: rows[qid] = row
    return rows

def top_doc_ids(row, preferred):
    for k in preferred:
        v = row.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], dict):
                return unique_ids(it.get("doc_id") for it in v if it.get("doc_id"))
            return unique_ids(v)
    return []

def scored_docs(row, preferred):
    for k in preferred:
        v = row.get(k)
        if isinstance(v, list) and v:
            if isinstance(v[0], dict):
                return [{**it, "doc_id": str(it.get("doc_id"))} for it in v if it.get("doc_id")]
            return [{"doc_id": str(d), "score": 0.0} for d in v]
    return []

# Load inputs now (used by all later cells)
SPLIT_MANIFEST_DATA = read_json(SPLIT_MANIFEST)
CONTRACT = read_json(FEATURE_CONTRACT)
TRAIN_SPLIT_DATA = load_split(TRAIN_SPLIT)
DEV_SPLIT_DATA   = load_split(DEV_SPLIT)

if set(TRAIN_SPLIT_DATA) & set(DEV_SPLIT_DATA):
    raise ValueError("canonical train/dev overlap")

print(f"✓ train={len(TRAIN_SPLIT_DATA)} dev={len(DEV_SPLIT_DATA)}")
print(f"  fast base={len(SPLIT_MANIFEST_DATA['fast_holdout']['base_train_query_ids'])}")
print(f"  fast target={len(SPLIT_MANIFEST_DATA['fast_holdout']['stacker_train_query_ids'])}")
"""))

# ============================================================
# CELL 6: Chunks + tokenize helpers
# ============================================================
cells.append(code("""
TOKEN_RE = re.compile(r"\\b\\w+\\b", flags=re.UNICODE)
STOPWORDS = {"a","an","anh","ay","bi","boi","cac","can","cho","co","con","cua","duoc","da","de","den","di","do","doi","duoi","gi","hay","hoac","khi","la","lai","lam","mot","nay","neu","nhu","nhung","o","phai","qua","quy","rieng","sau","se","thi","theo","thuoc","toi","trong","tu","va","ve","vi","voi"}

def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).replace("\\u0111", "d").replace("\\u0110", "D")

def tokenize(text):
    text = strip_accents(text.lower())
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1 and t not in STOPWORDS]

print("loading chunks (this may take a minute)...")
_chunks, _doc2idx, _valid_docs = [], defaultdict(list), set()
for idx, row in enumerate(iter_jsonl(CHUNKS)):
    doc_id = str(row.get("doc_id", ""))
    _chunks.append({
        "chunk_idx": idx,
        "chunk_id": str(row.get("chunk_id", idx)),
        "doc_id": doc_id,
        "text": str(row.get("text", "")),
        "heading": str(row.get("heading", "")),
        "word_count": int(row.get("word_count") or 0),
    })
    if doc_id: _doc2idx[doc_id].append(idx); _valid_docs.add(doc_id)
    if (idx+1) % 50000 == 0: print(f"  loaded {idx+1:,}")
print(f"✓ loaded {len(_chunks):,} chunks, {len(_valid_docs):,} unique docs")

def pick_evidence_text(question, doc_id, max_chunks=3, max_chars=1800):
    q = Counter(tokenize(question)); qset = set(q)
    scored = []
    for ci in _doc2idx.get(str(doc_id), []):
        ch = _chunks[ci]
        cc = Counter(tokenize((ch["heading"] + " " + ch["text"])[:7000]))
        overlap = sum(min(q[t], cc[t]) for t in qset)
        hbonus = 0.25 if any(t in strip_accents(ch["heading"].lower()) for t in qset) else 0.0
        scored.append((overlap/max(1,len(qset)) + hbonus, -abs(ch["word_count"]-320), ci))
    scored.sort(reverse=True)
    parts, cids = [], []
    for _, _, ci in scored[:max_chunks]:
        ch = _chunks[ci]
        cids.append(ch["chunk_id"])
        parts.append((ch["heading"] + "\\n" + ch["text"]).strip())
    text = "\\n\\n".join(parts)
    if len(text) > max_chars: text = text[:max_chars]
    return text, cids

def select_positive_chunks(question, gold_doc_id, top_n=2):
    q = Counter(tokenize(question)); qset = set(q)
    qdeacc = strip_accents(question.lower())
    scored = []
    for ci in _doc2idx.get(str(gold_doc_id), []):
        ch = _chunks[ci]
        cc = Counter(tokenize(ch["text"][:6000]))
        overlap = sum(min(q[t], cc[t]) for t in qset)
        hbonus = 0.25 if any(t in strip_accents(ch["heading"].lower()) for t in qset) else 0.0
        ebonus = 0.5 if qdeacc[:80] and qdeacc[:80] in strip_accents(ch["text"].lower()) else 0.0
        scored.append((overlap/max(1,len(qset)) + hbonus + ebonus, -abs(ch["word_count"]-320), ci))
    scored.sort(reverse=True)
    return [_chunks[ci] for _,_,ci in scored[:top_n] if _chunks[ci]["text"].strip()]

print("✓ evidence + positive selection helpers ready")
"""))

# ============================================================
# CELL 7: Configs (dataclasses)
# ============================================================
cells.append(code("""
@dataclass(frozen=True)
class Step5Config:
    positives_per_gold: int = 2
    max_train_examples: int = 16000
    train_batch_size: int = 12
    epochs: int = 1
    learning_rate: float = 4e-5
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    max_seq_length: int = 256
    encode_batch_size: int = 96
    query_batch_size: int = 128
    dense_top_chunks: int = 300
    dense_top_docs: int = 100
    aggregate_mean_top3_weight: float = 0.20
    aggregate_support_weight: float = 0.05
    search_block_size: int = 32768

@dataclass(frozen=True)
class Step6Config:
    max_seq_length: int = 384
    train_top_candidates: int = 80
    rerank_top_docs: int = 50
    negatives_per_positive: int = 10
    max_train_queries: int = 6000
    max_train_pairs: int = 80000
    train_batch_size: int = 8
    grad_accum_steps: int = 2
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    evidence_chunks: int = 3
    evidence_max_chars: int = 1800
    rerank_batch_size: int = 32

S5 = Step5Config()
S6 = Step6Config()
write_json(OUTPUT_DIR / "step5_config.json", asdict(S5))
write_json(OUTPUT_DIR / "step6_config.json", asdict(S6))
print("✓ configs saved")
"""))

# ============================================================
# CELL 8: Step5 biencoder training + dense ranking
# ============================================================
cells.append(code("""
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses

def build_step5_pairs(train_payload, train_qids, run_dir: Path):
    pairs, missing = [], 0
    for qid in train_qids:
        row = train_payload[qid]
        for doc_id in row["gold"]:
            pos = select_positive_chunks(row["question"], doc_id, top_n=S5.positives_per_gold)
            if not pos:
                missing += 1; continue
            for ch in pos:
                pairs.append({"query_id": str(qid), "doc_id": str(doc_id),
                              "chunk_id": ch["chunk_id"], "question": row["question"],
                              "positive_text": ch["text"]})
    random.Random(SEED).shuffle(pairs)
    if S5.max_train_examples and len(pairs) > S5.max_train_examples:
        pairs = pairs[:S5.max_train_examples]
    rep = {"num_pairs": len(pairs), "missing_gold_docs": missing, "num_train_queries": len(train_qids)}
    write_json(run_dir / "step5_train_pair_report.json", rep)
    append_jsonl(run_dir / "step5_train_pairs.jsonl", pairs)
    print(json.dumps(rep, indent=2))
    return pairs

def train_step5(train_payload, train_qids, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_step5_pairs(train_payload, train_qids, run_dir)
    if DRY_RUN: pairs = pairs[:64]
    examples = [InputExample(texts=[r["question"], r["positive_text"]]) for r in pairs]
    if len(examples) < S5.train_batch_size:
        raise ValueError(f"too few Step5 examples: {len(examples)}")
    loader = DataLoader(examples, shuffle=True, batch_size=S5.train_batch_size, drop_last=True)
    model = SentenceTransformer(BASE_BI_ENCODER_DIR, trust_remote_code=True)
    model.max_seq_length = min(S5.max_seq_length, 256)
    write_json(run_dir / "step5_model_manifest.json", {
        "model_id": BASE_BI_ENCODER_ID, "revision": BASE_BI_ENCODER_REV,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "local_training_and_inference": True, "no_hosted_inference_or_api": True,
    })
    out = run_dir / "step5_biencoder_mnrl"
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
              epochs=S5.epochs, warmup_steps=math.ceil(len(loader)*S5.epochs*S5.warmup_ratio),
              optimizer_params={"lr": S5.learning_rate}, weight_decay=S5.weight_decay,
              output_path=str(out), save_best_model=False, use_amp=True, show_progress_bar=True)
    saved = SentenceTransformer(str(out), trust_remote_code=True)
    return saved

def encode_chunks(model, chunks, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): return np.load(out, mmap_mode="r")
    probe = model.encode([chunks[0]["text"]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    dim = int(probe.shape[1])
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(chunks), dim))
    for start in range(0, len(chunks), S5.encode_batch_size):
        batch = chunks[start:start+S5.encode_batch_size]
        vecs = model.encode([b["text"] for b in batch], batch_size=S5.encode_batch_size,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        arr[start:start+len(batch)] = vecs.astype(np.float16)
    arr.flush()
    return np.load(out, mmap_mode="r")

def dense_search(emb, qemb, top_k, block):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qt = torch.tensor(qemb, dtype=torch.float16 if device=="cuda" else torch.float32, device=device)
    out = []
    for i in range(qt.shape[0]):
        q = qt[i:i+1].T; ts=None; ti=None
        for s in range(0, emb.shape[0], block):
            blk = torch.tensor(np.asarray(emb[s:s+block], dtype=np.float16 if device=="cuda" else np.float32), device=device)
            sc = (blk @ q).squeeze(1)
            v, ix = torch.topk(sc, k=min(top_k, sc.numel())); ix = ix + s
            if ts is None: ts, ti = v, ix
            else:
                ms = torch.cat([ts, v]); mi = torch.cat([ti, ix])
                ts, o = torch.topk(ms, k=min(top_k, ms.numel())); ti = mi[o]
        out.append([(int(idx), float(s)) for idx, s in zip(ti.detach().cpu().tolist(), ts.detach().cpu().tolist())])
    return out

def aggregate_docs(hits, chunks):
    per = defaultdict(list)
    for ci, sc in hits[:S5.dense_top_chunks]:
        d = chunks[ci]["doc_id"]
        if d: per[d].append((sc, ci))
    rows = []
    for d, sc_list in per.items():
        sc_list.sort(reverse=True)
        scores = [s for s,_ in sc_list]
        m = scores[0]; mt = sum(scores[:3])/min(3,len(scores))
        doc_sc = m + S5.aggregate_mean_top3_weight*mt + S5.aggregate_support_weight*math.log1p(len(sc_list))
        rows.append({"doc_id": d, "score": float(doc_sc), "max_chunk_score": float(m),
                     "mean_top3_chunk_score": float(mt), "support_count": int(len(sc_list)),
                     "evidence": [{"chunk_id": chunks[ci]["chunk_id"], "score": float(s),
                                   "heading": chunks[ci]["heading"], "word_count": chunks[ci]["word_count"]}
                                  for s,ci in sc_list[:3]]})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:S5.dense_top_docs]

def dense_rank(model, payload, qids, run_dir: Path, name: str):
    out = run_dir / f"{name}.jsonl"
    if out.exists(): return load_rankings(out)
    emb = encode_chunks(model, _chunks, run_dir / "chunk_embeddings_fp16.npy")
    sqids = list(map(str, qids))
    if DRY_RUN: sqids = sqids[:10]
    rows = {}
    def gen():
        for start in range(0, len(sqids), S5.query_batch_size):
            bq = sqids[start:start+S5.query_batch_size]
            qe = model.encode([payload[q]["question"] for q in bq], batch_size=S5.query_batch_size,
                              convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            hits = dense_search(emb, qe, top_k=S5.dense_top_chunks, block=S5.search_block_size)
            for qid, h in zip(bq, hits):
                docs = aggregate_docs(h, _chunks)
                row = {"query_id": qid, "question": payload[qid]["question"], "gold": payload[qid].get("gold"), "dense_top_docs": docs}
                rows[qid] = row; yield row
            print(f"  {name}: {min(start+S5.query_batch_size,len(sqids)):,}/{len(sqids):,}")
    append_jsonl(out, gen())
    return rows
"""))

# ============================================================
# CELL 9: Step6 reranker training + scoring
# ============================================================
cells.append(code("""
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

def logits_to_rel(logits):
    if logits.ndim==1 or logits.shape[-1]==1: return logits.view(-1)
    if logits.shape[-1]==2: return logits[:,1]-logits[:,0]
    return logits.max(dim=-1).values

class PairDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["question"], r["text"], truncation="only_second", padding="max_length",
                       max_length=self.max_len, return_tensors=None)
        item = {k: torch.tensor(v, dtype=torch.long) for k,v in enc.items()}
        item["labels"] = torch.tensor(float(r["label"]), dtype=torch.float32)
        return item

def load_reranker(device, run_dir: Path):
    tok = AutoTokenizer.from_pretrained(BASE_RERANKER_DIR, use_fast=False, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_RERANKER_DIR, trust_remote_code=True).to(device)
    if len(tok) > model.get_input_embeddings().num_embeddings:
        raise ValueError("vocab mismatch")
    max_len = min(S6.max_seq_length, getattr(model.config, "max_position_embeddings", 512)-2)
    write_json(run_dir / "step6_model_manifest.json", {
        "model_id": BASE_RERANKER_ID, "revision": BASE_RERANKER_REV,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "local_training_and_inference": True, "no_hosted_inference_or_api": True,
        "effective_max_length": max_len,
    })
    return {"model": model, "tokenizer": tok, "max_length": max_len, "device": device}

def train_step6(train_rows, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_reranker(device, run_dir)
    rows = train_rows[:512] if DRY_RUN else train_rows
    if len(rows) < S6.train_batch_size: raise ValueError("too few rows")
    loader = DataLoader(PairDataset(rows, bundle["tokenizer"], bundle["max_length"]),
                        batch_size=S6.train_batch_size, shuffle=True)
    opt = torch.optim.AdamW(bundle["model"].parameters(), lr=S6.learning_rate, weight_decay=S6.weight_decay)
    total = math.ceil(len(loader)/S6.grad_accum_steps)*S6.epochs
    sch = get_linear_schedule_with_warmup(opt, warmup_steps=math.ceil(total*S6.warmup_ratio), num_training_steps=max(1,total))
    scaler = torch.cuda.amp.GradScaler(s, enabled=torch.cuda.is_available())
    loss_fn = torch.nn.BCEWithLogitsLoss()
    bundle["model"].train(); opt.zero_grad(set_to_none=True)
    for ep in range(S6.epochs):
        run = []
        for step, batch in enumerate(loader, 1):
            lab = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k,v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = logits_to_rel(bundle["model"](**batch).logits)
                loss = loss_fn(logits, lab) / S6.grad_accum_steps
            scaler.scale(loss).backward()
            if step % S6.grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(bundle["model"].parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
            run.append(float(loss.detach().cpu())*S6.grad_accum_steps)
            if step % 100 == 0: print(f"  ep{ep+1} step{step} loss={np.mean(run[-100:]):.4f}")
    out = run_dir / "step6_aiteamvn_reranker"
    bundle["model"].save_pretrained(out); bundle["tokenizer"].save_pretrained(out)
    return bundle

@torch.inference_mode()
def score_pairs(bundle, pair_rows):
    model = bundle["model"]; tok = bundle["tokenizer"]; dev = bundle["device"]
    model.eval(); scores = []
    for start in range(0, len(pair_rows), S6.rerank_batch_size):
        br = pair_rows[start:start+S6.rerank_batch_size]
        enc = tok([r["question"] for r in br], [r["text"] for r in br],
                  truncation="only_second", padding=True, max_length=bundle["max_length"], return_tensors="pt")
        enc = {k: v.to(dev) for k,v in enc.items()}
        lg = logits_to_rel(model(**enc).logits).detach().cpu().numpy().tolist()
        scores.extend([1.0/(1.0+math.exp(-float(x))) for x in lg])
    return scores

def rrf_fuse(s4_docs, s5_docs, k=60, w4=1.0, w5=0.7):
    sc = defaultdict(float)
    for r, d in enumerate(s4_docs, 1): sc[str(d)] += w4/(k+r)
    for r, d in enumerate(s5_docs, 1): sc[str(d)] += w5/(k+r)
    return [d for d,_ in sorted(sc.items(), key=lambda kv:(-kv[1], kv[0]))]

def build_step6_train_rows(payload, qids, step4_rows, step5_rows, run_dir: Path):
    cache = run_dir / "step6_train_pairs.jsonl"
    if cache.exists():
        rows = list(iter_jsonl(cache)); print(f"  cached {len(rows)} step6 train rows"); return rows
    rows = []; stats = Counter(); rng = random.Random(SEED)
    qs = list(map(str, qids))[:S6.max_train_queries]
    if DRY_RUN: qs = qs[:20]
    for qid in qs:
        gold = set(str(g) for g in payload[qid]["gold"])
        s4d = top_doc_ids(step4_rows[qid], ["fused_doc_ids","doc_ids","top_docs"])[:MAX_DOCS]
        s5d = top_doc_ids(step5_rows[qid], ["dense_top_docs","ft_dense_doc_ids","fused_doc_ids"])[:MAX_DOCS]
        cands = rrf_fuse(s4d, s5d)
        neg_pool = []; seen=set()
        for d in cands[:S6.train_top_candidates]:
            if d in gold or d in seen: continue
            seen.add(d); neg_pool.append(d)
        semi = neg_pool[5:min(len(neg_pool),60)]
        if not semi: stats["no_semi_hard_pool"]+=1; continue
        for gd in gold:
            if gd not in _valid_docs: stats["missing_gold_doc"]+=1; continue
            pt, pc = pick_evidence_text(payload[qid]["question"], gd)
            if not pt: stats["empty_positive"]+=1; continue
            rows.append({"query_id":qid,"question":payload[qid]["question"],"doc_id":gd,"text":pt,"label":1.0,"chunk_ids":pc})
            for nd in rng.sample(semi, k=min(S6.negatives_per_positive,len(semi))):
                nt, nc = pick_evidence_text(payload[qid]["question"], nd)
                if not nt: stats["empty_negative"]+=1; continue
                rows.append({"query_id":qid,"question":payload[qid]["question"],"doc_id":nd,"text":nt,"label":0.0,"chunk_ids":nc})
    rng.shuffle(rows)
    if S6.max_train_pairs and len(rows)>S6.max_train_pairs: rows = rows[:S6.max_train_pairs]
    stats["num_rows"]=len(rows); stats["num_positive"]=sum(1 for r in rows if r["label"]==1.0); stats["num_negative"]=sum(1 for r in rows if r["label"]==0.0)
    write_json(run_dir / "step6_training_pair_report.json", dict(stats))
    append_jsonl(cache, rows)
    print(json.dumps(dict(stats), indent=2))
    return rows

def rerank(bundle, payload, qids, step4_rows, step5_rows, run_dir: Path, name: str):
    out = run_dir / f"{name}.jsonl"
    if out.exists(): return load_rankings(out)
    sqids = list(map(str, qids))
    if DRY_RUN: sqids = sqids[:10]
    rows = {}
    def gen():
        for idx, qid in enumerate(sqids, 1):
            q = payload[qid]["question"]
            s4d = top_doc_ids(step4_rows[qid], ["fused_doc_ids","doc_ids","top_docs"])[:MAX_DOCS]
            s5d = top_doc_ids(step5_rows[qid], ["dense_top_docs","ft_dense_doc_ids","fused_doc_ids"])[:MAX_DOCS]
            cands = rrf_fuse(s4d, s5d)[:S6.rerank_top_docs]
            pr = []
            for rank, d in enumerate(cands, 1):
                et, ec = pick_evidence_text(q, d)
                pr.append({"query_id":qid,"question":q,"doc_id":d,"text":et,"retrieval_rank":rank,"chunk_ids":ec})
            sc = score_pairs(bundle, pr) if pr else []
            scored = [{"doc_id":p["doc_id"], "reranker_score":float(s), "retrieval_rank":int(p["retrieval_rank"]), "chunk_ids":p["chunk_ids"]}
                      for p,s in zip(pr,sc)]
            scored.sort(key=lambda r: r["reranker_score"], reverse=True)
            row = {"query_id":qid,"question":q,"gold":payload[qid].get("gold"),"reranked_top_docs":scored}
            rows[qid] = row; yield row
            if idx%100==0: print(f"  {name}: {idx:,}/{len(sqids):,}")
    append_jsonl(out, gen())
    return rows
"""))

# ============================================================
# CELL 10: Feature frame builder + phase spec
# ============================================================
cells.append(code("""
def make_feature_frame(payload, qids, step4_rows, step5_rows, step6_rows, *, source_split, feature_protocol, fold=None):
    rows = []
    for qid in map(str, qids):
        gold = set(payload[qid].get("gold", []))
        bm_docs = scored_docs(step4_rows[qid], ["bm25_top_docs","top_docs"])[:MAX_DOCS]
        s4_docs = top_doc_ids(step4_rows[qid], ["fused_doc_ids","doc_ids","top_docs"])[:MAX_DOCS]
        s5_docs = scored_docs(step5_rows[qid], ["dense_top_docs"])[:MAX_DOCS]
        s6_docs = scored_docs(step6_rows[qid], ["reranked_top_docs"])[:MAX_DOCS]
        smap = {r["doc_id"]:(i+1,r) for i,r in enumerate(bm_docs)}
        s4map = {d:i+1 for i,d in enumerate(s4_docs)}
        s5map = {r["doc_id"]:(i+1,r) for i,r in enumerate(s5_docs)}
        s6map = {r["doc_id"]:(i+1,r) for i,r in enumerate(s6_docs)}
        cands = unique_ids(s4_docs + [r["doc_id"] for r in s5_docs] + [r["doc_id"] for r in s6_docs])[:MAX_DOCS]
        if source_split != "canonical_dev":
            for g in gold:
                if g not in cands: cands.append(g)
        cands = cands[:MAX_DOCS]
        for doc_id in cands:
            sr, srow = smap.get(doc_id, (MISSING_RANK, {}))
            s4r = s4map.get(doc_id, MISSING_RANK)
            s5r, s5row = s5map.get(doc_id, (MISSING_RANK, {}))
            s6r, s6row = s6map.get(doc_id, (MISSING_RANK, {}))
            base = {
                "query_id": qid, "doc_id": doc_id, "label": int(doc_id in gold),
                "source_split": source_split, "feature_protocol": feature_protocol,
                "fold": -1 if fold is None else int(fold),
                "score_rank": int(sr), "score_rr": 0.0 if sr==MISSING_RANK else 1.0/sr,
                "score": float(srow.get("score",0.0)),
                "max_chunk_score": float(srow.get("max_chunk_score",0.0)),
                "mean_top3_chunk_score": float(srow.get("mean_top3_chunk_score",0.0)),
                "support_count": float(srow.get("support_count",0.0)),
                "step4_rank": int(s4r),
                "step4_rr": 0.0 if s4r==MISSING_RANK else 1.0/s4r,
                "step5_rank": int(s5r),
                "step5_score": float(s5row.get("score",0.0)),
                "step6_rank": int(s6r),
                "step6_score": float(s6row.get("reranker_score", s6row.get("score",0.0))),
                "agreement_top5": int(s4r<=5)+int(s5r<=5)+int(s6r<=5),
                "agreement_top20": int(s4r<=20)+int(s5r<=20)+int(s6r<=20),
            }
            if source_split == "canonical_dev":
                base.update({"step5_fulltrain_rank": base["step5_rank"],
                             "step5_fulltrain_score": base["step5_score"],
                             "step6_fulltrain_rank": base["step6_rank"],
                             "step6_fulltrain_score": base["step6_score"]})
            else:
                base.update({"step5_oof_rank": base["step5_rank"],
                             "step5_oof_score": base["step5_score"],
                             "step6_oof_rank": base["step6_rank"],
                             "step6_oof_score": base["step6_score"]})
            rows.append(base)
    df = pd.DataFrame(rows).sort_values(
        ["query_id","step4_rank","step5_rank","step6_rank"],
        key=lambda c: c.map(qid_sort_key) if c.name=="query_id" else c,
    ).reset_index(drop=True)
    return df

def required_cols(name):
    return CONTRACT["required_outputs"][name]["required_columns"]

def validate_frame(df, expected_qids, *, name):
    expected = set(map(str, expected_qids))
    got = set(map(str, df["query_id"].unique())) if len(df) else set()
    if DRY_RUN: expected = set(list(expected)[:len(got)])
    if got != expected:
        raise ValueError(f"{name}: qid mismatch (missing={len(expected-got)}, extra={len(got-expected)})")
    req = required_cols(name)
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"{name}: missing cols {miss}")
    if df.groupby("query_id")["label"].sum().min() <= 0:
        raise ValueError(f"{name}: at least one query has no positive candidate")
    return {"rows": int(len(df)), "queries": int(df["query_id"].nunique()), "positive_rows": int(df["label"].sum())}

def phase_spec(phase, oof_fold):
    sm = SPLIT_MANIFEST_DATA
    if phase == "fast":
        fh = sm["fast_holdout"]
        return {"name":"fast", "base_train_qids":list(map(str, fh["base_train_query_ids"])),
                "target_qids":list(map(str, fh["stacker_train_query_ids"])),
                "target_payload":TRAIN_SPLIT_DATA, "source_split":"canonical_train_fast_holdout",
                "feature_protocol":"fast_holdout_oof", "fold":None,
                "part_path":PARTS_DIR/"fast_holdout_features.parquet"}
    if phase == "oof_fold":
        folds = {int(f["fold"]):f for f in sm["oof"]["folds"]}
        f = folds[oof_fold]
        return {"name":f"oof_fold_{oof_fold}", "base_train_qids":list(map(str, f["base_train_query_ids"])),
                "target_qids":list(map(str, f["heldout_query_ids"])),
                "target_payload":TRAIN_SPLIT_DATA, "source_split":"canonical_train_oof",
                "feature_protocol":"5fold_oof", "fold":oof_fold,
                "part_path":PARTS_DIR/f"oof_fold_{oof_fold}_features.parquet"}
    if phase == "fulltrain_dev":
        return {"name":"fulltrain_dev",
                "base_train_qids":sorted(TRAIN_SPLIT_DATA, key=qid_sort_key),
                "target_qids":sorted(DEV_SPLIT_DATA, key=qid_sort_key),
                "target_payload":DEV_SPLIT_DATA, "source_split":"canonical_dev",
                "feature_protocol":"fulltrain_dev", "fold":None,
                "part_path":PARTS_DIR/"dev_features.parquet"}
    raise ValueError(phase)
"""))

# ============================================================
# CELL 11: Run a single training phase
# ============================================================
cells.append(code("""
def run_phase(spec):
    run_dir = MODELS_DIR / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    base_qids = list(map(str, spec["base_train_qids"]))
    tqids = list(map(str, spec["target_qids"]))
    if spec["source_split"] != "canonical_dev" and (set(base_qids) & set(tqids)):
        raise ValueError("leakage guard: base/target train qid overlap")
    if spec["source_split"] == "canonical_dev" and (set(TRAIN_SPLIT_DATA) & set(tqids)):
        raise ValueError("dev overlaps canonical train")

    train_s4 = load_rankings(TRAIN_RANKINGS, set(base_qids))
    t_s4_path = DEV_RANKINGS if spec["source_split"]=="canonical_dev" else TRAIN_RANKINGS
    t_s4 = load_rankings(t_s4_path, set(tqids))
    if set(train_s4) != set(base_qids):
        miss = sorted(set(base_qids)-set(train_s4), key=qid_sort_key)
        raise ValueError(f"missing train rankings: {len(miss)} (need full Step3 rankings)")
    if set(t_s4) != set(tqids):
        raise ValueError(f"missing target rankings: {len(set(tqids)-set(t_s4))}")

    s5_model = train_step5(TRAIN_SPLIT_DATA, base_qids, run_dir)
    s5_train = dense_rank(s5_model, TRAIN_SPLIT_DATA, base_qids, run_dir, "step5_base_train_dense")
    s5_target = dense_rank(s5_model, spec["target_payload"], tqids, run_dir, "step5_target_dense")
    del s5_model; gc.collect()

    s6_train_cands = {qid: rrf_fuse(top_doc_ids(train_s4[qid],["fused_doc_ids","doc_ids","top_docs"])[:MAX_DOCS],
                                    top_doc_ids(s5_train[qid],["dense_top_docs","ft_dense_doc_ids","fused_doc_ids"])[:MAX_DOCS])[:MAX_DOCS]
                      for qid in map(str, base_qids)}
    s6_train_rows = build_step6_train_rows(TRAIN_SPLIT_DATA, base_qids, train_s4, s5_train, run_dir)
    s6_bundle = train_step6(s6_train_rows, run_dir)
    s6_target = rerank(s6_bundle, spec["target_payload"], tqids, t_s4, s5_target, run_dir, "step6_target_reranker")

    df = make_feature_frame(spec["target_payload"], tqids, t_s4, s5_target, s6_target,
                            source_split=spec["source_split"], feature_protocol=spec["feature_protocol"], fold=spec.get("fold"))
    validate_frame(df, tqids, name=spec["part_path"].name)
    df.to_parquet(spec["part_path"], index=False)
    write_json(REPORTS_DIR / f"{spec['name']}_report.json", {
        "status":"ok", "phase":spec["name"],
        "num_base_train_queries":len(base_qids), "num_target_queries":len(tqids),
        "num_feature_rows":int(len(df)), "part_path":str(spec["part_path"]),
        "part_sha256":sha256_file(spec["part_path"]),
        "leakage_guards":{"target_train_queries_excluded_from_base_training": spec["source_split"]!="canonical_dev",
                          "canonical_dev_used_for_training": False},
    })
    print(json.dumps({"wrote":str(spec["part_path"]),"rows":len(df)}, indent=2))
"""))

# ============================================================
# CELL 12: Merge phase
# ============================================================
cells.append(code("""
def run_merge():
    def req(p):
        if not p.exists(): raise FileNotFoundError(f"missing part: {p}")
        return p
    fast = req(PARTS_DIR/"fast_holdout_features.parquet")
    folds = [req(PARTS_DIR/f"oof_fold_{i}_features.parquet") for i in range(1,6)]
    dev   = req(PARTS_DIR/"dev_features.parquet")
    fast_df = pd.read_parquet(fast)
    oof_df  = pd.concat([pd.read_parquet(p) for p in folds], ignore_index=True)
    dev_df  = pd.read_parquet(dev)
    fast_qids = list(map(str, SPLIT_MANIFEST_DATA["fast_holdout"]["stacker_train_query_ids"]))
    train_qids = sorted(TRAIN_SPLIT_DATA, key=qid_sort_key)
    dev_qids   = sorted(DEV_SPLIT_DATA, key=qid_sort_key)
    stats = {
        "fast_holdout_features.parquet": validate_frame(fast_df, fast_qids, name="fast_holdout_features.parquet"),
        "train_oof_features.parquet":    validate_frame(oof_df, train_qids, name="train_oof_features.parquet"),
        "dev_features.parquet":          validate_frame(dev_df, dev_qids,   name="dev_features.parquet"),
    }
    # dev folds should be [-1]
    got_folds = sorted(int(x) for x in dev_df["fold"].unique())
    if got_folds != [-1]: raise ValueError(f"dev folds must be [-1], got {got_folds}")
    # oof folds must be 1..5
    got_folds = sorted(int(x) for x in oof_df["fold"].unique())
    if got_folds != [1,2,3,4,5]: raise ValueError(f"oof folds must be [1..5], got {got_folds}")

    final_fast = OUTPUT_DIR/"fast_holdout_features.parquet"
    final_oof  = OUTPUT_DIR/"train_oof_features.parquet"
    final_dev  = OUTPUT_DIR/"dev_features.parquet"
    fast_df.to_parquet(final_fast, index=False)
    oof_df.to_parquet(final_oof, index=False)
    dev_df.to_parquet(final_dev, index=False)

    manifest = {
        "status":"ok", "protocol":"p1_2_clean_base_features_fast_holdout_and_5fold_oof",
        "created_by_notebook":"task1/pipeline/step9_p1_audit/kaggle_p1_2_generate_clean_base_features_gpu_v3.ipynb",
        "input_paths":{"split_manifest":str(SPLIT_MANIFEST), "feature_contract":str(FEATURE_CONTRACT),
                       "train_rankings":str(TRAIN_RANKINGS), "train_split":str(TRAIN_SPLIT),
                       "dev_split":str(DEV_SPLIT), "chunks":str(CHUNKS), "dev_rankings":str(DEV_RANKINGS),
                       "hf_bi_encoder":f"{BASE_BI_ENCODER_ID}@{BASE_BI_ENCODER_REV}",
                       "hf_reranker":f"{BASE_RERANKER_ID}@{BASE_RERANKER_REV}"},
        "output_paths":{"fast_holdout_features.parquet":str(final_fast),
                        "train_oof_features.parquet":str(final_oof),
                        "dev_features.parquet":str(final_dev)},
        "checksums":{
            "fast_holdout_features.parquet":sha256_file(final_fast),
            "train_oof_features.parquet":sha256_file(final_oof),
            "dev_features.parquet":sha256_file(final_dev),
            "split_manifest.json":sha256_file(SPLIT_MANIFEST),
            "feature_contract.json":sha256_file(FEATURE_CONTRACT),
        },
        "model_manifests": sorted(str(p) for p in MODELS_DIR.glob("*/step*_model_manifest.json")),
        "stats": stats,
        "leakage_guards":{
            "canonical_train_dev_query_overlap": False,
            "fast_holdout_target_queries_excluded_from_base_training": True,
            "oof_target_queries_excluded_from_corresponding_base_training": True,
            "canonical_dev_used_for_training_or_selection": False,
            "no_external_DSC2026_baseline_features": True,
            "no_in_sample_step5_or_step6_train_rankings_used_as_stacker_features": True,
        },
    }
    for fld in CONTRACT["required_outputs"]["manifest.json"]["required_fields"]:
        if fld not in manifest: raise ValueError(f"manifest missing field {fld}")
    write_json(OUTPUT_DIR/"manifest.json", manifest)
    print(json.dumps({"status":"ok","manifest":str(OUTPUT_DIR/"manifest.json"),"stats":stats}, indent=2))
"""))

# ============================================================
# CELL 13: Dispatcher — run selected phase
# ============================================================
cells.append(code("""
phases_to_run = []
if RUN_PHASE == "all":
    phases_to_run = ["fast", "oof_fold:1", "oof_fold:2", "oof_fold:3", "oof_fold:4", "oof_fold:5", "fulltrain_dev", "merge"]
elif RUN_PHASE == "oof_fold":
    phases_to_run = [f"oof_fold:{OOF_FOLD}"]
else:
    phases_to_run = [RUN_PHASE]

for p in phases_to_run:
    t0 = time.time()
    print(f"\\n{'='*70}\\n>>> PHASE: {p}\\n{'='*70}")
    if p == "merge":
        run_merge()
    else:
        if ":" in p:
            base, fold = p.split(":"); spec = phase_spec(base, int(fold))
        else:
            spec = phase_spec(p, OOF_FOLD)
        print(json.dumps({"name":spec["name"],"n_base":len(spec["base_train_qids"]),
                          "n_target":len(spec["target_qids"]),"source_split":spec["source_split"]}, indent=2))
        run_phase(spec)
    elapsed = time.time() - t0
    print(f"--- phase {p} done in {elapsed/3600:.2f}h ({elapsed:.0f}s) ---")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\\n✓ all requested phases completed")
print(f"output dir: {OUTPUT_DIR}")
if (OUTPUT_DIR/'manifest.json').exists():
    print(f"manifest:   {OUTPUT_DIR/'manifest.json'}")
"""))

# ============================================================
# CELL 14: Verification (post-run)
# ============================================================
cells.append(md("""
## Verification

After the notebook completes, verify outputs:

```python
import json, pandas as pd
from pathlib import Path
OUT = Path("/kaggle/working/p1_2_clean_base_features")
print(json.loads((OUT/'manifest.json').read_text())['stats'])
for n in ['fast_holdout_features.parquet','train_oof_features.parquet','dev_features.parquet']:
    df = pd.read_parquet(OUT/n)
    print(f"{n}: rows={len(df):,} queries={df.query_id.nunique():,} positives={df.label.sum():,}")
```

## Download these artifacts back to local

- `manifest.json`
- `fast_holdout_features.parquet`, `train_oof_features.parquet`, `dev_features.parquet`
- `models/{phase}/step{5,6}_model_manifest.json` (provenance only; checkpoints not needed downstream)

Save locally to `task1/pipeline/step9_p1_audit/p1_2_clean_base_features/` (overwriting staged JSONs).
"""))

# ============================================================
# Save notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Validate JSON before writing
json.dumps(notebook)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✓ wrote {OUTPUT_PATH}")
print(f"  cells: {len(cells)}")
print(f"  size: {os.path.getsize(OUTPUT_PATH):,} bytes")