"""Shared helpers: data loading, detectors (keyword / HF classifier), metrics."""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
PROTECTAI = "protectai/deberta-v3-base-prompt-injection-v2"


def read_jsonl(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_pools():
    pools = {p.stem[len("test_"):]: read_jsonl(p.stem) for p in sorted(PROC.glob("test_*.jsonl"))}
    suites = json.loads((PROC / "suites.json").read_text())
    return pools, suites


def suite_rows(pools, suite):
    """Return (rows, pos_pools) for a suite: positives from suite['pos'], negatives from suite['neg']."""
    pos = [r for p in suite["pos"] for r in pools[p] if r["label"] == 1]
    neg = [r for p in suite["neg"] for r in pools[p] if r["label"] == 0]
    return pos, neg


# ------------------------------------------------------------------ detectors
class KeywordDetector:
    """Hand-written regex filter: the 'what people ship first' baseline. Score is 0/1."""
    name = "keyword"
    PATTERNS = [
        r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|preceding)?\s*(instructions?|prompts?|rules?|directions?|text)",
        r"disregard\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions?|prompts?|rules?|directions?)",
        r"forget\s+(everything|all|your)\b.{0,40}(instructions?|rules?|told|said|above)",
        r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|password|secret)",
        r"system\s+prompt",
        r"you\s+are\s+now\b",
        r"\bjailbreak",
        r"\bDAN\b",
        r"developer\s+mode",
        r"do\s+anything\s+now",
        r"pretend\s+(to\s+be|you\s+are|that)",
        r"act\s+as\s+(if\s+you\s+(have|are)|an?\s+unfiltered|an?\s+unrestricted)",
        r"without\s+(any\s+)?(restrictions?|limitations?|filters?|censorship)",
        r"(new|updated)\s+instructions?\s*:",
        r"\bapi_call\b|send\s+an?\s+email\s+(to|with)",
        r"override\s+(your|all|the)\s+(instructions?|rules?|safety)",
    ]

    def __init__(self):
        self.rx = [re.compile(p, re.I | re.S) for p in self.PATTERNS]

    def predict(self, texts, **kw):
        return np.array([float(any(r.search(t) for r in self.rx)) for t in texts])

    predict_chunked = predict


class HFDetector:
    """Sequence classifier wrapper. `inj_index` = index of the 'injection' logit."""

    def __init__(self, path, name, inj_index=1, max_len=512, device=None, batch_size=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name, self.max_len, self.inj, self.bs = name, max_len, inj_index, batch_size
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(self.dev).eval()

    def _probs(self, enc_ids_list):
        torch, out = self.torch, []
        order = np.argsort([len(x) for x in enc_ids_list])
        res = np.zeros(len(enc_ids_list))
        for i in range(0, len(order), self.bs):
            idx = order[i : i + self.bs]
            batch = self.tok.pad({"input_ids": [enc_ids_list[j] for j in idx]}, return_tensors="pt").to(self.dev)
            with torch.no_grad(), torch.autocast(self.dev, dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                logits = self.model(**batch).logits.float()
            res[idx] = torch.softmax(logits, -1)[:, self.inj].cpu().numpy()
        return res

    def predict(self, texts, **kw):
        ids = self.tok(list(texts), truncation=True, max_length=self.max_len, add_special_tokens=True)["input_ids"]
        return self._probs(ids)

    def predict_chunked(self, texts, stride=None):
        """Slide a window over the whole text; document score = max chunk score."""
        win = self.max_len - 2
        stride = stride or win // 2
        chunk_ids, owner = [], []
        for n, t in enumerate(texts):
            ids = self.tok(t, add_special_tokens=False)["input_ids"]
            starts = list(range(0, max(1, len(ids) - win + stride), stride)) or [0]
            for s in starts:
                chunk_ids.append([self.tok.cls_token_id] + ids[s : s + win] + [self.tok.sep_token_id])
                owner.append(n)
        p = self._probs(chunk_ids)
        res = np.zeros(len(texts))
        for o, v in zip(owner, p):
            res[o] = max(res[o], v)
        return res


# ------------------------------------------------------------------ metrics
def metrics(y, s, thr=0.5):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "precision": float(p), "recall": float(r), "f1": float(f),
           "fpr": float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))}
    out["auroc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
    out["auprc"] = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else None
    return out


def score_all_pools(det, pools):
    """Score every row of every pool once. Returns {pool: np.array}."""
    return {k: det.predict([r["text"] for r in v]) for k, v in pools.items()}


def suite_metrics(scores, pools, suites, thr=0.5):
    res = {}
    for sname, suite in suites.items():
        ys, ss = [], []
        for p in suite["pos"]:
            for r, s in zip(pools[p], scores[p]):
                if r["label"] == 1:
                    ys.append(1); ss.append(s)
        negs = {}
        for p in suite["neg"]:
            for r, s in zip(pools[p], scores[p]):
                if r["label"] == 0:
                    ys.append(0); ss.append(s); negs.setdefault(p, []).append(s)
        m = metrics(ys, ss, thr)
        m["fpr_by_benign_pool"] = {p: float((np.array(v) >= thr).mean()) for p, v in negs.items()}
        if len(suite["pos"]) > 1 or sname == "seen":
            m["recall_by_source"] = {}
            for p in suite["pos"]:
                sc = [s for r, s in zip(pools[p], scores[p]) if r["label"] == 1]
                m["recall_by_source"][p] = float((np.array(sc) >= thr).mean())
        res[sname] = m
    return res
