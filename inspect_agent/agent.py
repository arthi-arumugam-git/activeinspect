"""Active-perception inspection agent.

Perception -> decision -> action loop. Every step the agent looks at OpenCV
measurements of the latest frame and picks the next tool: change exposure,
engage the polariser, probe focus, zoom into a candidate, or finish with
PASS / REJECT / ESCALATE (human review). Visual evidence therefore decides
what the camera does next, not just what the report says.

Planner: deterministic policy by default (mock mode). If BEDROCK_MODEL_ID is set
and AWS credentials are available, an LLM on Amazon Bedrock (Converse API,
tool use) chooses the action; any invalid choice falls back to the policy.
"""
from __future__ import annotations
import json, os, time
from dataclasses import asdict, replace
from inspect_agent.camera import Scene, CameraParams, capture
from inspect_agent import vision

TOOLS = ["adjust_exposure", "enable_polarizer", "probe_focus", "average_frames", "zoom_verify", "finish"]
MAX_STEPS = 7
REJECT_AT, CLEAN_BELOW = 0.6, 0.15


class Session:
    def __init__(self, scene: Scene):
        self.scene, self.params, self.trace, self.shots = scene, CameraParams(), [], 0
        self.focus_checked = False
        self.verified = None
        self.averaged = False
        self.frame = self._shoot()

    def _shoot(self, params: CameraParams | None = None):
        self.shots += 1
        return capture(self.scene, params or self.params, shot=self.shots)

    def observe(self) -> dict:
        q = vision.quality(self.frame)
        d = vision.detect(self.frame)
        self.last_det = d
        return {"quality": {k: round(v, 3) for k, v in q.items()},
                "defect_score": d["defect_score"], "best": d["best"],
                "params": asdict(self.params), "focus_checked": self.focus_checked,
                "verified": self.verified, "averaged": self.averaged, "shots": self.shots}

    # ---- tools ---------------------------------------------------------------
    def adjust_exposure(self, obs):
        b = max(obs["quality"]["brightness"], 1.0)
        self.params = replace(self.params, exposure=round(min(12.0, self.params.exposure * 150.0 / b), 2))
        self.frame = self._shoot()
        return f"exposure -> {self.params.exposure}"

    def enable_polarizer(self, obs):
        self.params = replace(self.params, polarizer=True)
        self.frame = self._shoot()
        return "cross-polariser on"

    def probe_focus(self, obs):
        self.focus_checked = True
        base = vision.quality(self.frame)["sharpness"]
        trial = self._shoot(replace(self.params, focus=1.0))
        s = vision.quality(trial)["sharpness"]
        if s > base * 1.15:
            self.params = replace(self.params, focus=1.0)
            self.frame = trial
            return f"refocused (sharpness {base:.1f} -> {s:.1f})"
        return f"focus already optimal (sharpness {base:.1f} vs {s:.1f})"

    def average_frames(self, obs, n: int = 6):
        """Faint-defect check: average n frames (noise falls ~1/sqrt(n))."""
        import numpy as np
        self.averaged = True
        stack = [self._shoot().astype(np.float32) for _ in range(n)]
        self.frame = np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)
        return f"averaged {n} frames"

    def zoom_verify(self, obs):
        x, y, w, h = obs["best"]["bbox"]
        cx, cy, side = x + w // 2, y + h // 2, max(64, int(max(w, h) * 1.6))
        side = min(side, 256)
        rx, ry = int(min(max(0, cx - side // 2), 256 - side)), int(min(max(0, cy - side // 2), 256 - side))
        z = self._shoot(replace(self.params, roi=(rx, ry, side, side)))
        dz = vision.detect(z, zoom=256 / side)
        self.verified = dz["defect_score"] >= 0.5
        self.zoom = (z, dz)
        return f"zoom ROI {(rx, ry, side, side)}: defect_score {dz['defect_score']}"

    # ---- policy --------------------------------------------------------------
    def policy(self, obs) -> str:
        q, p = obs["quality"], obs["params"]
        if q["saturated"] > 0.01 and not p["polarizer"]:
            return "enable_polarizer"
        if (q["brightness"] < 90 or q["brightness"] > 215) and abs(q["brightness"] - 150) > 25 and p["exposure"] < 12:
            return "adjust_exposure"
        if not obs["focus_checked"]:
            return "probe_focus"
        if obs["best"] and obs["defect_score"] >= CLEAN_BELOW and obs["verified"] is None:
            return "zoom_verify"
        if obs["defect_score"] < REJECT_AT and not obs.get("averaged"):
            return "average_frames"
        return "finish"


def llm_choose(obs: dict, allowed: list[str]) -> str | None:
    model = os.getenv("BEDROCK_MODEL_ID")
    if not model or os.getenv("INSPECT_MOCK") == "1":
        return None
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        prompt = ("You control an industrial inspection camera. Pick the single next tool to reduce uncertainty "
                  f"about whether the part has a surface defect. Allowed: {allowed}. Observation (OpenCV metrics): "
                  f"{json.dumps(obs)}. Reply with JSON {{\"tool\": <name>, \"reason\": <short>}}.")
        r = client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                            inferenceConfig={"maxTokens": 120, "temperature": 0})
        tool = json.loads(r["output"]["message"]["content"][0]["text"]).get("tool")
        return tool if tool in allowed else None
    except Exception:
        return None


def decide(obs: dict, verified) -> tuple[str, str]:
    s = obs["defect_score"]
    q = obs["quality"]
    in_spec = q["saturated"] <= 0.01 and 90 <= q["brightness"] <= 215
    if verified is True:
        return "REJECT", "defect confirmed at zoom"
    if not in_spec:
        return "ESCALATE", "image quality could not be brought into spec: human review"
    if verified is False and s < REJECT_AT:
        return "PASS", "candidate not confirmed at zoom"
    if verified is False:
        return "ESCALATE", "strong candidate not confirmed at zoom: human review"
    if s < REJECT_AT:
        return "PASS", "image in spec and no confirmed candidate"
    return "ESCALATE", "strong candidate without zoom confirmation: human review"


def inspect(scene: Scene, agentic: bool = True) -> dict:
    t0 = time.perf_counter()
    s = Session(scene)
    obs = s.observe()
    if not agentic:  # single-shot baseline
        v = "REJECT" if obs["defect_score"] >= REJECT_AT else "PASS"
        return {"verdict": v, "reason": "single shot", "trace": [], "shots": 1, "obs": obs, "frame": s.frame,
                "det": s.last_det, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    for step in range(MAX_STEPS):
        tool = s.policy(obs)
        chosen = llm_choose(obs, [tool, "finish"]) if tool != "finish" else None  # LLM may stop early, never act outside policy-safe set
        tool = chosen or tool
        if tool == "finish":
            break
        result = getattr(s, tool)(obs)
        obs = s.observe()
        s.trace.append({"step": step + 1, "tool": tool, "result": result, "defect_score": obs["defect_score"],
                        "brightness": obs["quality"]["brightness"], "saturated": obs["quality"]["saturated"]})
    verdict, reason = decide(obs, s.verified)
    return {"verdict": verdict, "reason": reason, "trace": s.trace, "shots": s.shots, "obs": obs, "frame": s.frame,
            "det": s.last_det, "params": asdict(s.params), "ms": round((time.perf_counter() - t0) * 1000, 1)}
