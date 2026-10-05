"""Fine-tune microsoft/deberta-v3-small as a binary prompt-injection classifier.

python train.py --seed 0 --out models/deberta-small-s0
Model selection uses the in-distribution validation split ONLY (never the held-out sources).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, read_jsonl

BASE = "microsoft/deberta-v3-small"


def encode(tok, rows, max_len):
    return tok([r["text"] for r in rows], truncation=True, max_length=max_len)["input_ids"]


def batches(ids, labels, bs, shuffle, rng):
    order = rng.permutation(len(ids)) if shuffle else np.argsort([len(x) for x in ids])
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        yield idx, [ids[j] for j in idx], torch.tensor([labels[j] for j in idx])


@torch.no_grad()
def predict(model, tok, ids, dev, bs=64):
    model.eval()
    out = np.zeros(len(ids))
    for idx, b, _ in batches(ids, [0] * len(ids), bs, False, None):
        enc = tok.pad({"input_ids": b}, return_tensors="pt").to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            out[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true", help="smoke-test mode")
    ap.add_argument("--limit", type=int, default=None, help="use only N train/val rows (smoke test)")
    a = ap.parse_args()
    out = Path(a.out or ROOT / "models" / f"deberta-small-s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, dtype=torch.float32).to(dev)
    tr, va = read_jsonl("train"), read_jsonl("val")
    if a.limit:
        tr, va = tr[: a.limit], va[: a.limit]
    tr_ids, tr_y = encode(tok, tr, a.max_len), [r["label"] for r in tr]
    va_ids, va_y = encode(tok, va, a.max_len), [r["label"] for r in va]

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr_ids) + a.bs - 1) // a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, log, t0 = -1, [], time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0
        for idx, b, y in batches(tr_ids, tr_y, a.bs, True, rng):
            enc = tok.pad({"input_ids": b}, return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item() * len(idx)
        p = predict(model, tok, va_ids, dev)
        f1 = f1_score(va_y, p >= 0.5)
        log.append({"epoch": ep + 1, "train_loss": tot / len(tr_ids), "val_f1": float(f1), "secs": round(time.time() - t0)})
        print(log[-1], flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(out); tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log, "best_val_f1": float(best)}, indent=1))


if __name__ == "__main__":
    main()
