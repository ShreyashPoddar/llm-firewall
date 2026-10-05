"""Full evaluation: held-out generalization, FPR, perturbation robustness, injections inside long documents,
failure cases. Compares our fine-tuned deberta-v3-small (mean +- std over seeds) with the two baselines.

python eval.py                (needs models/deberta-small-s* from train.py, and runs baselines internally)
Outputs: results/eval.json, results/failure_cases.json, results/*.png
"""
import json
import random
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from common import PROTECTAI, RES, ROOT, HFDetector, KeywordDetector, load_pools, metrics, score_all_pools, suite_metrics

rng = random.Random(7)
UNSEEN = ["unseen_gandalf", "unseen_multilingual", "unseen_llmail"]
SUITES_ALL = ["seen"] + UNSEEN
COL = {"keyword": "#9aa0a6", "protectai": "#e8710a", "ours": "#1a73e8"}
LAB = {"keyword": "regex keyword", "protectai": "protectai v2 (deberta-base)", "ours": "ours (deberta-small, 3 seeds)"}


# ------------------------------------------------------------------ perturbations (applied to existing text)
TRIGGERS = ["ignore", "instructions", "instruction", "previous", "system", "prompt", "password", "secret", "forget", "reveal", "disregard"]
SYN = {"ignore": "disregard", "previous": "prior", "instructions": "directions", "instruction": "direction", "tell": "inform",
       "reveal": "disclose", "password": "passcode", "secret": "confidential detail", "forget": "set aside", "ignoring": "overlooking",
       "all": "every one of the", "above": "earlier", "now": "from this point", "send": "forward", "please": "kindly"}


def _swap_words(t, fn):
    return re.sub(r"[A-Za-z]+", lambda m: fn(m.group(0)) if m.group(0).lower() in TRIGGERS else m.group(0), t)


def p_lower(t): return t.lower()
def p_upper(t): return t.upper()
def p_extra_space(t): return re.sub(r" ", "  ", t)
def p_letter_space(t): return _swap_words(t, lambda w: " ".join(w))
def p_zero_width(t): return _swap_words(t, lambda w: w[:2] + "​" + w[2:])
def p_leet(t): return _swap_words(t, lambda w: w.translate(str.maketrans("aeio", "4310")))
def p_synonym(t):
    return re.sub(r"[A-Za-z]+", lambda m: (SYN[m.group(0).lower()] if m.group(0).lower() in SYN else m.group(0)), t)


def make_padders(pad_text):
    return {
        "pad_prefix": lambda t: pad_text + "\n\n" + t,
        "pad_suffix": lambda t: t + "\n\n" + pad_text,
        "pad_both": lambda t: pad_text + "\n\n" + t + "\n\n" + pad_text,
    }


# ------------------------------------------------------------------ long documents
def build_docs(attacks, benign_paras, words, pos):
    """Benign email-like carrier of ~`words` words with an attack block inserted at relative position pos (0..1)."""
    out = []
    for i, atk in enumerate(attacks):
        paras, n = [], 0
        while n < words:
            p = rng.choice(benign_paras)
            paras.append(p)
            n += len(p.split())
        cut = int(round(pos * len(paras)))
        paras = paras[:cut] + [atk] + paras[cut:]
        out.append("Subject: Fwd: weekly digest\n\n" + "\n\n".join(paras))
    return out


def build_benign_docs(n, benign_paras, words):
    out = []
    for _ in range(n):
        paras, c = [], 0
        while c < words:
            p = rng.choice(benign_paras)
            paras.append(p)
            c += len(p.split())
        out.append("Subject: Fwd: weekly digest\n\n" + "\n\n".join(paras))
    return out


def rate(s, thr=0.5):
    return float((np.asarray(s) >= thr).mean())


def evaluate_detector(det, pools, suites, pert_sets, long_sets):
    sc = score_all_pools(det, pools)
    res = {"suites": suite_metrics(sc, pools, suites)}
    # perturbations: recall on perturbed positives, FPR on perturbed benign
    res["perturb"] = {}
    for name, (pos_texts, neg_texts, fns) in pert_sets.items():
        base = {"recall": rate(det.predict(pos_texts)), "fpr": rate(det.predict(neg_texts))}
        r = {"none": base}
        for pn, fn in fns.items():
            r[pn] = {"recall": rate(det.predict([fn(t) for t in pos_texts])), "fpr": rate(det.predict([fn(t) for t in neg_texts]))}
        res["perturb"][name] = r
    # long documents
    res["long"] = {}
    for (words, pos), (docs, _) in long_sets["attack"].items():
        key = f"{words}w_{pos}"
        res["long"][key] = {"trunc_recall": rate(det.predict(docs)), "chunk_recall": rate(det.predict_chunked(docs))}
    for words, docs in long_sets["benign"].items():
        res["long"][f"{words}w_benign"] = {"trunc_fpr": rate(det.predict(docs)), "chunk_fpr": rate(det.predict_chunked(docs))}
    return res, sc


def agg(list_of_dicts):
    """Recursively mean/std across seeds for nested dicts of floats."""
    first = list_of_dicts[0]
    if isinstance(first, dict):
        return {k: agg([d[k] for d in list_of_dicts]) for k in first}
    if first is None:
        return None
    a = np.array(list_of_dicts, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def mv(x):  # mean value of an agg node or plain float
    return x["mean"] if isinstance(x, dict) else x


def main():
    pools, suites = load_pools()
    web = [r["text"] for r in pools["benign_web"]]
    benign_paras = [t for t in web if 40 <= len(t.split()) <= 200] or web
    pad_text = web[0]

    # evaluation sets for perturbations (positives from each unseen source, 400 each; benign matched)
    def pick(pool, n):
        rows = [r["text"] for r in pools[pool] if r["label"] == 1]
        return rng.sample(rows, min(n, len(rows)))

    def benign(pool, n):
        rows = [r["text"] for r in pools[pool]]
        return rng.sample(rows, min(n, len(rows)))

    fns = {"lowercase": p_lower, "uppercase": p_upper, "extra_spaces": p_extra_space, "letter_spaced_triggers": p_letter_space,
           "zero_width_in_triggers": p_zero_width, "leetspeak_triggers": p_leet, "synonym_swap": p_synonym, **make_padders(pad_text)}
    pert_sets = {
        "gandalf": (pick("unseen_gandalf_pos", 400), benign("benign_prompts", 400), fns),
        "multilingual": (pick("unseen_multi_pos", 400), benign("benign_multi", 400), fns),
        "llmail": (pick("unseen_llmail_pos", 400), benign("benign_email", 400), fns),
    }
    # long docs: attacks = short-ish llmail emails
    atk_pool = [t for t in pick("unseen_llmail_pos", 2000) if len(t) < 1500][:150]
    lengths, positions = [100, 400, 1500, 4000], [0.0, 0.5, 1.0]
    long_sets = {"attack": {(w, p): (build_docs(atk_pool, benign_paras, w, p), None) for w in lengths for p in positions},
                 "benign": {w: build_benign_docs(150, benign_paras, w) for w in lengths}}

    dets = {"keyword": [KeywordDetector()], "protectai": [HFDetector(PROTECTAI, "protectai")]}
    model_dirs = sorted((ROOT / "models").glob("deberta-small-s*"))
    assert model_dirs, "run train.py first"
    results, all_scores = {}, {}
    for name, ds in list(dets.items()):
        r, sc = evaluate_detector(ds[0], pools, suites, pert_sets, long_sets)
        results[name], all_scores[name] = [r], [sc]
    ours_r, ours_sc = [], []
    for d in model_dirs:
        det = HFDetector(str(d), d.name)
        r, sc = evaluate_detector(det, pools, suites, pert_sets, long_sets)
        ours_r.append(r); ours_sc.append(sc)
        del det
    results["ours"], all_scores["ours"] = ours_r, ours_sc

    summary = {k: agg(v) for k, v in results.items()}
    summary["n_seeds_ours"] = len(model_dirs)

    # --- recall breakdown by source/scenario for ours (seed 0)
    breakdown = {}
    for p in ["unseen_llmail_pos", "unseen_gandalf_pos", "unseen_multi_pos"]:
        by = {}
        for row, s in zip(pools[p], ours_sc[0][p]):
            by.setdefault(f"{row['source']}:{row['group']}" if p == "unseen_llmail_pos" else row["source"], []).append(s >= 0.5)
        breakdown[p] = {k: {"n": len(v), "recall": float(np.mean(v))} for k, v in sorted(by.items())}
    summary["ours_seed0_recall_breakdown"] = breakdown

    # --- failure cases (ours, seed 0, thr 0.5)
    fails = {}
    for sname, suite in suites.items():
        fn_rows, fp_rows = [], []
        for p in suite["pos"]:
            fn_rows += [(r, s) for r, s in zip(pools[p], ours_sc[0][p]) if r["label"] == 1 and s < 0.5]
        for p in suite["neg"]:
            fp_rows += [(r, s) for r, s in zip(pools[p], ours_sc[0][p]) if r["label"] == 0 and s >= 0.5]
        rng.shuffle(fn_rows); rng.shuffle(fp_rows)
        fmt = lambda rows: [{"score": round(float(s), 3), "source": r["source"], "text": r["text"][:300]} for r, s in rows[:8]]
        fails[sname] = {"n_false_negatives": len(fn_rows), "n_false_positives": len(fp_rows),
                        "sample_false_negatives": fmt(fn_rows), "sample_false_positives": fmt(fp_rows)}
    (RES / "failure_cases.json").write_text(json.dumps(fails, indent=1, ensure_ascii=False))

    # --- threshold sweep + recall at 1% FPR on pooled unseen (ours seed-avg)
    sweep = {}
    thr_grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for name in dets.keys() | {"ours"}:
        sweep[name] = {}
        for sname in UNSEEN:
            suite = suites[sname]
            rows = []
            for k, sc in enumerate(all_scores[name]):
                ys = [1] * sum(len(pools[p]) for p in suite["pos"]) + [0] * sum(len(pools[p]) for p in suite["neg"])
                ss = np.concatenate([sc[p] for p in suite["pos"]] + [sc[p] for p in suite["neg"]])
                rows.append((np.array(ys), ss))
            ys = rows[0][0]
            ss = np.mean([r[1] for r in rows], axis=0)
            fpr, tpr, thr = roc_curve(ys, ss)
            ok = fpr <= 0.01
            sweep[name][sname] = {"recall_at_fpr_1pct": float(tpr[ok].max()) if ok.any() else 0.0,
                                  "by_threshold": {str(t): {"recall": float((ss[ys == 1] >= t).mean()), "fpr": float((ss[ys == 0] >= t).mean())} for t in thr_grid}}
    summary["threshold_sweep_seed_averaged_scores"] = sweep

    (RES / "eval.json").write_text(json.dumps(summary, indent=1))
    make_plots(summary, all_scores, pools, suites, sweep)
    print_table(summary)


def print_table(S):
    print("\nsuite                 model       P      R      F1     AUROC  FPR")
    for sn in SUITES_ALL:
        for m in ["keyword", "protectai", "ours"]:
            x = S[m]["suites"][sn]
            print(f"{sn:21s} {m:10s} {mv(x['precision']):.3f}  {mv(x['recall']):.3f}  {mv(x['f1']):.3f}  {mv(x['auroc']):.3f}  {mv(x['fpr']):.3f}")


# ------------------------------------------------------------------ plots
def make_plots(S, all_scores, pools, suites, sweep):
    models = ["keyword", "protectai", "ours"]
    # 1) held-out F1 & AUROC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, met, title in zip(axes, ["f1", "auroc"], ["F1 @ threshold 0.5", "AUROC"]):
        w = 0.26
        for i, m in enumerate(models):
            vals = [mv(S[m]["suites"][s][met]) for s in SUITES_ALL]
            errs = [S[m]["suites"][s][met]["std"] if isinstance(S[m]["suites"][s][met], dict) else 0 for s in SUITES_ALL]
            ax.bar(np.arange(4) + (i - 1) * w, vals, w, yerr=errs, label=LAB[m], color=COL[m], capsize=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(["seen sources\n(in-dist.)", "Gandalf\n(unseen)", "multilingual\n(unseen)", "LLMail-Inject\n(unseen)"])
        ax.set_ylim(0, 1.05); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(RES / "heldout_f1_auroc.png", dpi=140); plt.close()

    # 2) ROC curves on unseen suites
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sn in zip(axes, UNSEEN):
        suite = suites[sn]
        ys = np.array([1] * sum(len(pools[p]) for p in suite["pos"]) + [0] * sum(len(pools[p]) for p in suite["neg"]))
        for m in models:
            ss = np.mean([np.concatenate([sc[p] for p in suite["pos"]] + [sc[p] for p in suite["neg"]]) for sc in all_scores[m]], axis=0)
            f, t, _ = roc_curve(ys, ss)
            ax.plot(f, t, color=COL[m], label=LAB[m])
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_title(sn.replace("unseen_", "") + " (unseen)")
        ax.set_xlabel("false positive rate (symlog)"); ax.set_ylabel("true positive rate"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(RES / "roc_unseen.png", dpi=140); plt.close()

    # 3) perturbation robustness (recall), mean over unseen sources
    names = [k for k in S["ours"]["perturb"]["llmail"].keys()]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.26
    for i, m in enumerate(models):
        vals = [np.mean([mv(S[m]["perturb"][src][n]["recall"]) for src in ["gandalf", "multilingual", "llmail"]]) for n in names]
        ax.bar(np.arange(len(names)) + (i - 1) * w, vals, w, label=LAB[m], color=COL[m])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("recall @0.5 (mean of 3 unseen sources)"); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Robustness to simple perturbations of held-out attacks")
    plt.tight_layout(); plt.savefig(RES / "perturbations.png", dpi=140); plt.close()

    # 4) long-document recall by position/length: truncated vs chunked
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    lengths, positions = [100, 400, 1500, 4000], ["0.0", "0.5", "1.0"]
    ls = {"0.0": "-", "0.5": "--", "1.0": ":"}
    for ax, mode, title in zip(axes, ["trunc_recall", "chunk_recall"], ["whole document, truncated at 512 tokens", "chunked scan (512-token windows, max score)"]):
        for m in ["protectai", "ours"]:
            for p in positions:
                ax.plot(range(4), [mv(S[m]["long"][f"{w}w_{p}"][mode]) for w in lengths], ls[p], color=COL[m], marker="o", label=f"{m}, attack at {'start' if p=='0.0' else 'middle' if p=='0.5' else 'end'}")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{w} words" for w in lengths]); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    axes[0].set_ylabel("recall of embedded LLMail attack"); axes[1].legend(fontsize=6.5, loc="lower left")
    plt.tight_layout(); plt.savefig(RES / "long_document.png", dpi=140); plt.close()

    # 5) threshold sweep for ours
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sn in zip(axes, UNSEEN):
        bt = sweep["ours"][sn]["by_threshold"]
        th = [float(t) for t in bt]
        ax.plot(th, [bt[str(t)]["recall"] for t in th], label="recall", color=COL["ours"])
        ax.plot(th, [bt[str(t)]["fpr"] for t in th], label="FPR", color="#d93025")
        ax.set_title(sn.replace("unseen_", "") + " (ours)"); ax.set_xlabel("decision threshold"); ax.grid(alpha=.3)
    axes[0].legend()
    plt.tight_layout(); plt.savefig(RES / "threshold_sweep.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
