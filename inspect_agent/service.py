"""Shared service layer: used by the AWS Lambda handler and the local FastAPI server.
AWS pieces (all optional, activated by env vars):
  RESULTS_BUCKET  -> annotated frames + trace JSON stored in S3
  AUDIT_TABLE     -> every decision and every human review written to DynamoDB
  BEDROCK_MODEL_ID-> LLM planner on Amazon Bedrock (see agent.llm_choose)
"""
from __future__ import annotations
import base64, json, os, time, uuid
import cv2
from inspect_agent.camera import Scene, CameraParams, capture
from inspect_agent import vision
from inspect_agent.agent import inspect


def _b64png(img) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def _aws(name):
    import boto3
    return boto3.client(name)


def run_inspection(seed: int, defect: str, condition: str) -> dict:
    scene = Scene.make(int(seed), defect, condition)
    first = capture(scene, CameraParams(), shot=1)
    r = inspect(scene)
    rid = uuid.uuid4().hex[:12]
    out = {"id": rid, "verdict": r["verdict"], "reason": r["reason"], "trace": r["trace"], "shots": r["shots"],
           "params": r["params"], "ms": r["ms"], "ground_truth": defect, "condition": condition, "seed": seed,
           "first_frame": _b64png(vision.annotate(first, vision.detect(first), "1st shot")),
           "final_frame": _b64png(vision.annotate(r["frame"], r["det"], r["verdict"])),
           "planner": "bedrock" if os.getenv("BEDROCK_MODEL_ID") and os.getenv("INSPECT_MOCK") != "1" else "policy"}
    if os.getenv("RESULTS_BUCKET"):
        s3 = _aws("s3"); b = os.environ["RESULTS_BUCKET"]
        s3.put_object(Bucket=b, Key=f"inspections/{rid}/final.png", Body=base64.b64decode(out["final_frame"]), ContentType="image/png")
        s3.put_object(Bucket=b, Key=f"inspections/{rid}/trace.json", Body=json.dumps({k: v for k, v in out.items() if "frame" not in k}).encode())
    audit({"id": rid, "event": "decision", "verdict": r["verdict"], "reason": r["reason"], "shots": r["shots"]})
    return out


def human_review(rid: str, decision: str, reviewer: str = "operator") -> dict:
    if decision not in {"PASS", "REJECT"}:
        raise ValueError("decision must be PASS or REJECT")
    return audit({"id": rid, "event": "human_review", "verdict": decision, "reviewer": reviewer})


_LOCAL_AUDIT: list[dict] = []


def audit(item: dict) -> dict:
    item = {**item, "ts": int(time.time() * 1000)}
    if os.getenv("AUDIT_TABLE"):
        import boto3
        boto3.resource("dynamodb").Table(os.environ["AUDIT_TABLE"]).put_item(Item={"pk": item["id"], "sk": f"{item['ts']}#{item['event']}", **{k: str(v) for k, v in item.items()}})
    else:
        _LOCAL_AUDIT.append(item)
    return item
