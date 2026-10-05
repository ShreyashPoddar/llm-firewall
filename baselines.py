"""Evaluate the two baselines on the held-out splits: regex keyword filter and protectai/deberta-v3-base-prompt-injection-v2.

Writes results/baselines.json and results/scores_{keyword,protectai}.json (raw per-row scores).
NOTE: the protectai model was trained on data that may overlap some of our *seen* sources (it lists deepset and
others in its card), so only the *unseen_* suites are a fair comparison; see README.
"""
import json

from common import PROTECTAI, RES, HFDetector, KeywordDetector, load_pools, score_all_pools, suite_metrics


def main():
    pools, suites = load_pools()
    RES.mkdir(exist_ok=True)
    out = {}
    for det in [KeywordDetector(), HFDetector(PROTECTAI, "protectai", inj_index=1)]:
        sc = score_all_pools(det, pools)
        (RES / f"scores_{det.name}.json").write_text(json.dumps({k: [round(float(x), 5) for x in v] for k, v in sc.items()}))
        out[det.name] = suite_metrics(sc, pools, suites)
        for s, m in out[det.name].items():
            print(f"{det.name:10s} {s:20s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} AUROC={m['auroc']:.3f} FPR={m['fpr']:.3f}")
    (RES / "baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
