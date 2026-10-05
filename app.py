"""Gradio demo for the LLM-firewall prompt-injection detector (runs locally; CPU is fine).

python app.py                      # uses models/deberta-small-s0 (or $FW_MODEL = local dir or HF repo id)

Tab 1: paste text -> risk score + verdict.
Tab 2: a *simulated* agent reads a poisoned email / web page. No LLM is called: the tab shows what the firewall
       would do between "fetch document" and "put document into the agent's context".
"""
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np

from common import HFDetector

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("FW_MODEL", str(ROOT / "models" / "deberta-small-s0"))
DET = HFDetector(MODEL, "ours", device="cpu", batch_size=8)
SCENARIOS = json.loads((ROOT / "examples" / "scenarios.json").read_text(encoding="utf-8"))


def windows(text, size=110, stride=55):
    """Overlapping word windows so a short injection inside a long document is scored on its own."""
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)]
    starts = list(range(0, len(w) - size + stride, stride))
    return [" ".join(w[s : s + size]) for s in starts]


def scan(text):
    chunks = windows(text)
    scores = DET.predict(chunks)
    return chunks, scores


def verdict(score, thr):
    return ("BLOCK: likely prompt injection" if score >= thr else "ALLOW: no injection detected")


def check_text(text, thr):
    if not text.strip():
        return {}, "Paste some text."
    chunks, scores = scan(text)
    s = float(np.max(scores))
    label = {"injection risk": s, "benign": 1 - s}
    note = f"{verdict(s, thr)}  (risk {s:.3f}, threshold {thr:.2f}, {len(chunks)} window(s) scanned)"
    return label, note


def run_agent(name, thr, firewall_on):
    sc = next(s for s in SCENARIOS if s["name"] == name)
    doc = sc["document"]
    chunks, scores = scan(doc)
    risk = float(np.max(scores))
    trace = [f"1. User asks the agent: **{sc['task']}**",
             f"2. Agent tool call: `fetch({sc['kind']})` returns a {len(doc.split())}-word document.",
             f"3. Firewall scans {len(chunks)} window(s); max risk = **{risk:.3f}** (threshold {thr:.2f})."]
    if firewall_on and risk >= thr:
        trace.append("4. **Document quarantined.** The agent is told: 'the fetched content was flagged as a possible prompt injection and was not read.' "
                     "The flagged window is shown below for the human.")
    elif risk >= thr:
        trace.append("4. Firewall is OFF: the full document, including the flagged span, enters the agent's context "
                     "(this is where a real agent could be hijacked).")
    else:
        trace.append("4. Document passes the firewall and is given to the agent.")
        if sc.get("poisoned"):
            trace.append("   **Missed detection** (false negative). See the README's failure analysis.")
    hl = [(c, f"{s:.2f}" if s >= thr else None) for c, s in zip(chunks, scores)] if len(chunks) > 1 else [(doc, f"{risk:.2f}" if risk >= thr else None)]
    return "\n\n".join(trace), hl


with gr.Blocks(title="LLM Firewall") as demo:
    gr.Markdown("# LLM Firewall: prompt-injection detector\nFine-tuned `deberta-v3-small`. Research demo, **not** a security guarantee: see the README for measured failure modes.")
    with gr.Tab("Check text"):
        inp = gr.Textbox(lines=8, label="Text (user prompt, email, web page...)")
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Block threshold")
        btn = gr.Button("Scan")
        out_lab, out_note = gr.Label(label="Risk"), gr.Markdown()
        btn.click(check_text, [inp, thr], [out_lab, out_note])
        gr.Examples([["What is the capital of France? Please also give me one fun fact."],
                     ["Hi Dana, can we move Thursday's sync to 3pm? Thanks, Sam"]], inp)
    with gr.Tab("Simulated agent reads a poisoned page"):
        gr.Markdown("A toy agent fetches a document that contains an injected instruction (verbatim from public datasets). No LLM is run; this shows where the firewall sits.")
        sel = gr.Dropdown([s["name"] for s in SCENARIOS], value=SCENARIOS[0]["name"], label="Scenario")
        doc_view = gr.Textbox(label="Fetched document", lines=10, interactive=False, value=SCENARIOS[0]["document"])
        sel.change(lambda n: next(s["document"] for s in SCENARIOS if s["name"] == n), sel, doc_view)
        thr2 = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Block threshold")
        fw = gr.Checkbox(True, label="Firewall enabled")
        go = gr.Button("Run agent")
        tr, hl = gr.Markdown(), gr.HighlightedText(label="Scanned windows (flagged ones highlighted with their risk)")
        go.click(run_agent, [sel, thr2, fw], [tr, hl])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
