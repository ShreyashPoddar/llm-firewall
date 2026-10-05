"""Build examples/scenarios.json for the demo's simulated-agent tab.
Injected text is taken verbatim from the held-out public test pools (LLMail-Inject, Gandalf); benign carriers are
held-out news text / synthetic benign emails. Nothing is authored by us."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_jsonl  # noqa: E402

rng = random.Random(3)
llmail = [r["text"] for r in read_jsonl("test_unseen_llmail_pos") if r["source"] == "llmail_phase2" and 80 < len(r["text"]) < 700]
gandalf = [r["text"] for r in read_jsonl("test_unseen_gandalf_pos") if 30 < len(r["text"]) < 200]
news = [r["text"] for r in read_jsonl("test_benign_web") if 90 < len(r["text"].split()) < 160]
mails = [r["text"] for r in read_jsonl("test_benign_email")]

web_a, web_b = rng.sample(news, 2)
sc = [
    {"name": "Inbox summarizer: poisoned email (LLMail-Inject attack)", "kind": "email", "poisoned": True,
     "task": "Summarize my unread email.",
     "document": mails[0] + "\n\n---\n\n" + rng.choice(llmail) + "\n\n---\n\n" + mails[1]},
    {"name": "Web browsing agent: injected line inside a news page (Gandalf-style)", "kind": "web page", "poisoned": True,
     "task": "Summarize this web page for me.",
     "document": web_a + "\n\n" + rng.choice(gandalf) + "\n\n" + web_b},
    {"name": "Control: clean newsletter email", "kind": "email", "poisoned": False,
     "task": "Summarize my unread email.", "document": mails[2] + "\n\n---\n\n" + mails[3]},
    {"name": "Control: clean news page", "kind": "web page", "poisoned": False,
     "task": "Summarize this web page for me.", "document": web_a + "\n\n" + web_b},
]
out = Path(__file__).resolve().parent / "scenarios.json"
out.write_text(json.dumps(sc, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
