"""Build source-held-out train/val/test splits for the prompt-injection detector.

Usage:  python data/prepare.py            (writes data/processed/*.jsonl + results/data_stats.json)

All attack text comes from the public, license-cleared datasets listed in README.md.
Nothing here authors new attack text; the only text written by us is a small set of
*benign* email templates (used as held-out benign emails).

Design (see README "Splits"):
  TRAIN sources   : deepset (train), jackhhao (train), SPML (train system-prompt groups),
                    benign dolly instructions, benign cnn_dailymail news text
  VAL             : random 10% of the train sources
  TEST (seen)     : deepset test, jackhhao test, SPML held-out system prompts, held-out dolly/cnn
  TEST (unseen)   : gandalf    (user-written "ignore instructions" attacks, never trained on)
                    multilingual (yanismiraoui, FR/DE/ES/IT/PT/RO attacks, never trained on)
                    llmail     (indirect injections inside emails, never trained on)
  Every unseen-source positive set is paired with a benign pool that is also unseen
  (held-out dolly prompts / multilingual Wikipedia snippets / benign emails).
"""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 13
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "processed"
RES = ROOT.parent / "results"
rng = random.Random(SEED)


def norm_key(t: str) -> str:
    t = re.sub(r"[^\w]+", " ", t.lower(), flags=re.UNICODE).strip()
    return hashlib.md5(t.encode()).hexdigest()


def rec(text, label, source, pool, group=None):
    return {"text": (text or "").strip(), "label": int(label), "source": source, "pool": pool, "group": group}


# ---------------------------------------------------------------- positive/mixed sources
def load_deepset():
    d = load_dataset("deepset/prompt-injections")
    tr = [rec(r["text"], r["label"], "deepset", "seen") for r in d["train"]]
    te = [rec(r["text"], r["label"], "deepset", "seen") for r in d["test"]]
    return tr, te


def load_jackhhao():
    d = load_dataset("jackhhao/jailbreak-classification")
    f = lambda r: rec(r["prompt"], r["type"] == "jailbreak", "jackhhao", "seen")
    return [f(r) for r in d["train"]], [f(r) for r in d["test"]]


def load_spml(max_train=3000, heldout_frac=0.15):
    d = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")["train"]
    rows = [(r["System Prompt"], r["User Prompt"], int(r["Prompt injection"])) for r in d]
    systems = sorted({s for s, _, _ in rows})
    rng.shuffle(systems)
    held = set(systems[: int(len(systems) * heldout_frac)])
    tr, te = [], []
    for s, u, y in rows:
        (te if s in held else tr).append(rec(u, y, "spml", "seen", group=norm_key(s)))
    rng.shuffle(tr)
    return tr, te, max_train


def load_gandalf():
    d = load_dataset("Lakera/gandalf_ignore_instructions")
    return [rec(r["text"], 1, "gandalf", "gandalf_pos") for sp in d for r in d[sp]]


def load_multilingual():
    d = load_dataset("yanismiraoui/prompt_injections")["train"]
    return [rec(r["prompt_injections"], 1, "multilingual", "multi_pos") for r in d]


def load_llmail(n_phase1=1200, n_phase2=1200):
    d = load_dataset("microsoft/llmail-inject-challenge")
    out = []
    for phase, n in (("Phase1", n_phase1), ("Phase2", n_phase2)):
        seen, by_scn = set(), {}
        for r in d[phase]:
            body = (r["body"] or "").strip()
            if len(body) < 20:
                continue
            text = f"Subject: {(r['subject'] or '').strip()}\n\n{body}"
            k = norm_key(text)
            if k in seen:
                continue
            seen.add(k)
            by_scn.setdefault(r["scenario"], []).append(text)
        # stratify by scenario (level) so one easy scenario does not dominate
        scns = sorted(by_scn)
        for s in scns:
            rng.shuffle(by_scn[s])
        picked = []
        i = 0
        while len(picked) < n and any(by_scn.values()):
            for s in scns:
                if by_scn[s] and len(picked) < n:
                    picked.append((s, by_scn[s].pop()))
        out += [rec(t, 1, f"llmail_{phase.lower()}", "llmail_pos", group=s) for s, t in picked]
    return out


# ---------------------------------------------------------------- benign sources
def trunc_words(t, lo, hi):
    w = t.split()
    n = rng.randint(lo, hi)
    if len(w) <= n:
        return " ".join(w)
    s = rng.randint(0, len(w) - n)
    return " ".join(w[s : s + n])


def load_dolly():
    d = load_dataset("databricks/databricks-dolly-15k")["train"]
    out = []
    for r in d:
        t = r["instruction"].strip()
        if r["context"]:
            t += "\n\n" + trunc_words(r["context"], 30, 250)
        out.append(rec(t, 0, "dolly", "benign_prompts"))
    rng.shuffle(out)
    return out


def load_cnn(n=2400):
    d = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    out = []
    for i, r in enumerate(d):
        if i >= n:
            break
        out.append(rec(trunc_words(r["article"], 40, 400), 0, "cnn_dailymail", "benign_web"))
    return out


def load_wiki_multi(per_lang=100):
    out = []
    for lang in ["fr", "de", "es", "it", "pt", "ro"]:
        d = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        c = 0
        for i, r in enumerate(d):
            if i % 7:  # skip ahead a bit so we do not take only the first alphabetical articles
                continue
            t = trunc_words(r["text"], 20, 120)
            if len(t.split()) < 20:
                continue
            out.append(rec(t, 0, f"wiki_{lang}", "benign_multi"))
            c += 1
            if c >= per_lang:
                break
    return out


NAMES = ["Priya", "Daniel", "Mei", "Carlos", "Amara", "Jonas", "Fatima", "Oliver", "Yuki", "Sofia"]
TOPICS = ["the Q3 budget review", "the vendor contract renewal", "next week's design sync",
          "the onboarding checklist", "the quarterly security training", "the conference travel plan",
          "the customer feedback summary", "the release schedule", "the office move", "the hiring plan"]
EMAIL_TEMPLATES = [
    "Hi {a},\n\nCould we move {t} to {day}? I have a conflict at the original time. Let me know what works for you and I will update the invite.\n\nThanks,\n{b}",
    "Hello {a},\n\nAttached is the latest draft for {t}. Please review sections 2 and 3 and send me your comments by {day}. No rush on the appendix.\n\nBest regards,\n{b}",
    "Hi team,\n\nA quick reminder that {t} is due on {day}. If you are blocked on anything, reply here and I will help unblock you.\n\nCheers,\n{b}",
    "Dear {a},\n\nThank you for your invoice. Our finance team has approved it for payment and it should arrive within 10 business days. Please tell me if anything looks wrong on your side.\n\nKind regards,\n{b}",
    "{a},\n\nFollowing up on our call about {t}. My notes: we agreed on the scope, the owner is {b}, and the first checkpoint is {day}. Shout if I missed something.\n\n{b}",
    "Hi {a},\n\nJust confirming lunch on {day}. I booked a table for four near the office. Let me know about any dietary requirements.\n\nSee you then,\n{b}",
    "Hello,\n\nYour order has shipped and is expected to arrive on {day}. You can track the package with the link in your account page. Thank you for shopping with us.\n\nCustomer Care",
    "Hi {a},\n\nWelcome aboard! Your laptop will be ready on {day}. Your first week schedule covers {t} and a tour of the office. Reach out to me with any questions.\n\n{b}",
    "Hi all,\n\nMinutes from today's meeting on {t}: decisions were recorded in the shared doc, action items are assigned, and the next meeting is on {day}. Thanks for the productive discussion.\n\n{b}",
    "Dear {a},\n\nI am writing to ask about the status of {t}. Could you share a short update, even if it is only a rough estimate? Thanks in advance for your help.\n\nSincerely,\n{b}",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "the 14th", "the end of the month"]


def make_benign_emails(n=500):
    out, seen = [], set()
    while len(out) < n:
        a, b = rng.sample(NAMES, 2)
        subj = rng.choice(["Re: ", "Fwd: ", ""]) + rng.choice(TOPICS).replace("the ", "", 1).capitalize()
        body = rng.choice(EMAIL_TEMPLATES).format(a=a, b=b, t=rng.choice(TOPICS), day=rng.choice(DAYS))
        t = f"Subject: {subj}\n\n{body}"
        if t in seen:
            continue
        seen.add(t)
        out.append(rec(t, 0, "synthetic_email", "benign_email"))
    return out


# ---------------------------------------------------------------- assembly
def dedupe(rows, taken):
    out = []
    for r in rows:
        if len(r["text"]) < 3 or r["label"] not in (0, 1):
            continue
        k = norm_key(r["text"])
        if k in taken:
            continue
        taken.add(k)
        out.append(r)
    return out


def write(name, rows):
    OUT.mkdir(exist_ok=True, parents=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ds_tr, ds_te = load_deepset()
    jb_tr, jb_te = load_jackhhao()
    sp_tr, sp_te, sp_cap = load_spml()
    dolly, cnn = load_dolly(), load_cnn()
    gandalf, multi, llmail = load_gandalf(), load_multilingual(), load_llmail()
    wiki, emails = load_wiki_multi(), make_benign_emails()

    taken = set()  # global dedupe, priority order: TEST pools first, so train never contains test text
    test = {}
    test["seen_deepset"] = dedupe(ds_te, taken)
    test["seen_jackhhao"] = dedupe(jb_te, taken)
    test["seen_spml"] = dedupe(sp_te, taken)
    test_dolly = dedupe(dolly[:1200], taken)
    test_cnn = dedupe(cnn[:400], taken)
    test["unseen_gandalf_pos"] = dedupe(gandalf, taken)
    test["unseen_multi_pos"] = dedupe(multi, taken)
    test["unseen_llmail_pos"] = dedupe(llmail, taken)
    test["benign_prompts"] = test_dolly[:800]
    test["benign_web"] = test_cnn
    test["benign_multi"] = dedupe(wiki, taken)
    test["benign_email"] = dedupe(emails, taken)
    # train pool (anything whose normalized text is in a test set was already removed)
    train = []
    train += dedupe(ds_tr, taken) + dedupe(jb_tr, taken) + dedupe(sp_tr, taken)[:sp_cap]
    train += dedupe(dolly[1200:4200], taken) + dedupe(cnn[400:1900], taken)
    rng.shuffle(train)
    n_val = len(train) // 10
    val, train = train[:n_val], train[n_val:]

    write("train", train)
    write("val", val)
    for k, v in test.items():
        write(f"test_{k}", v)

    # named evaluation suites: (positives pool(s), matched benign pool(s))
    suites = {
        "seen": {"pos": ["seen_deepset", "seen_jackhhao", "seen_spml"], "neg": ["seen_deepset", "seen_jackhhao", "seen_spml", "benign_prompts", "benign_web"]},
        "unseen_gandalf": {"pos": ["unseen_gandalf_pos"], "neg": ["benign_prompts"]},
        "unseen_multilingual": {"pos": ["unseen_multi_pos"], "neg": ["benign_multi"]},
        "unseen_llmail": {"pos": ["unseen_llmail_pos"], "neg": ["benign_email"]},
    }
    (OUT / "suites.json").write_text(json.dumps(suites, indent=1))

    stats = {"seed": SEED, "train": dict(Counter((r["source"], r["label"]) .__str__() for r in train)),
             "val_n": len(val), "train_n": len(train),
             "train_pos_frac": round(sum(r["label"] for r in train) / len(train), 3),
             "test_sizes": {k: {"n": len(v), "pos": sum(r["label"] for r in v)} for k, v in test.items()}}
    RES.mkdir(exist_ok=True)
    (RES / "data_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
