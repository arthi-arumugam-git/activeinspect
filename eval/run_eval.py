"""Benchmark: single-shot baseline vs software-enhanced baseline vs active-perception agent.
Held-out seeds 10_000+ (the policy was tuned on seeds 0-39). Writes eval/results.json,
eval/RESULTS.md and failure-case images in eval/failures/."""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path
import cv2
from inspect_agent.camera import Scene, CONDITIONS, DEFECTS, CameraParams, capture
from inspect_agent import vision
from inspect_agent.agent import inspect, REJECT_AT

OUT = Path(__file__).parent


def baseline_clahe(scene):
    img = vision.enhance(capture(scene, CameraParams(), shot=1))
    return {"verdict": "REJECT" if vision.detect(img)["defect_score"] >= REJECT_AT else "PASS", "shots": 1}


def metrics(rows):
    tp = sum(r["gt"] and r["v"] == "REJECT" for r in rows); fn = sum(r["gt"] and r["v"] == "PASS" for r in rows)
    fp = sum((not r["gt"]) and r["v"] == "REJECT" for r in rows); tn = sum((not r["gt"]) and r["v"] == "PASS" for r in rows)
    esc = sum(r["v"] == "ESCALATE" for r in rows); n = len(rows)
    auto = n - esc
    return {"n": n, "recall": round(tp / max(1, tp + fn + sum(r["gt"] and r["v"] == "ESCALATE" for r in rows)), 3),
            "escaped_defects": fn, "false_rejects": fp, "escalated": esc,
            "auto_accuracy": round((tp + tn) / max(1, auto), 3),
            "mean_shots": round(sum(r["shots"] for r in rows) / n, 2), "mean_ms": round(sum(r["ms"] for r in rows) / n, 1)}


def main(seeds=30):
    rows = defaultdict(list)
    fails = OUT / "failures"; fails.mkdir(exist_ok=True)
    for f in fails.glob("*.png"):
        f.unlink()
    for k in range(seeds):
        seed = 10_000 + k
        for cond in CONDITIONS:
            for d in DEFECTS:
                sc = Scene.make(seed, d, cond)
                gt = d != "none"
                for name, fn in (("single_shot", lambda s: inspect(s, agentic=False)), ("single_shot_clahe", baseline_clahe), ("agent", inspect)):
                    t0 = time.perf_counter(); r = fn(sc); ms = (time.perf_counter() - t0) * 1000
                    rows[name].append({"cond": cond, "defect": d, "gt": gt, "v": r["verdict"], "shots": r["shots"], "ms": ms})
                    if name == "agent" and ((gt and r["verdict"] == "PASS") or (not gt and r["verdict"] == "REJECT")) and len(list(fails.glob("*.png"))) < 12:
                        cv2.imwrite(str(fails / f"{cond}_{d}_{seed}_{r['verdict']}.png"), vision.annotate(r["frame"], r["det"], r["verdict"]))
    res = {name: {"overall": metrics(rs), "by_condition": {c: metrics([r for r in rs if r["cond"] == c]) for c in CONDITIONS}} for name, rs in rows.items()}
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    lines = ["# Evaluation results", "", f"Held-out scenes: {seeds} seeds × {len(CONDITIONS)} conditions × {len(DEFECTS)} defect types = {seeds*len(CONDITIONS)*len(DEFECTS)} parts (2/3 defective).", "",
             "| System | Defect recall | Escaped defects | False rejects | Escalated to human | Accuracy on auto-decisions | Captures/part | ms/part |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        o = r["overall"]
        lines.append(f"| {name} | {o['recall']:.1%} | {o['escaped_defects']} | {o['false_rejects']} | {o['escalated']} | {o['auto_accuracy']:.1%} | {o['mean_shots']} | {o['mean_ms']} |")
    lines += ["", "## Recall by condition", "", "| Condition | single_shot | single_shot_clahe | agent | agent escalations |", "|---|---|---|---|---|"]
    for c in CONDITIONS:
        lines.append(f"| {c} | {res['single_shot']['by_condition'][c]['recall']:.0%} | {res['single_shot_clahe']['by_condition'][c]['recall']:.0%} | {res['agent']['by_condition'][c]['recall']:.0%} | {res['agent']['by_condition'][c]['escalated']} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
