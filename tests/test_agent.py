import json, os
os.environ["INSPECT_MOCK"] = "1"
import cv2
from inspect_agent.camera import Scene, CameraParams, capture
from inspect_agent import vision
from inspect_agent.agent import inspect
from inspect_agent.lambda_handler import handler


def test_opencv5():
    assert cv2.__version__.startswith("5.")


def test_quality_metrics_reflect_conditions():
    nom = vision.quality(capture(Scene.make(1, "none", "nominal"), CameraParams()))
    dim = vision.quality(capture(Scene.make(1, "none", "dim"), CameraParams()))
    glare = vision.quality(capture(Scene.make(1, "none", "glare"), CameraParams()))
    assert dim["brightness"] < 60 < nom["brightness"] and glare["saturated"] > 0.01 >= nom["saturated"]


def test_baseline_blind_in_dim_light_agent_recovers():
    sc = Scene.make(7, "scratch", "dim")
    assert inspect(sc, agentic=False)["verdict"] == "PASS"     # baseline misses it
    r = inspect(sc)
    assert r["verdict"] == "REJECT" and r["trace"][0]["tool"] == "adjust_exposure"


def test_visual_evidence_changes_next_action():
    assert inspect(Scene.make(7, "pit", "glare"))["trace"][0]["tool"] == "enable_polarizer"
    tools = [t["tool"] for t in inspect(Scene.make(7, "pit", "defocus"))["trace"]]
    assert "probe_focus" in tools and "zoom_verify" in tools


def test_clean_part_passes_and_impossible_light_escalates():
    assert inspect(Scene.make(7, "none", "nominal"))["verdict"] == "PASS"
    assert inspect(Scene.make(7, "none", "very_dark"))["verdict"] == "ESCALATE"


def test_step_budget():
    for c in ("dim+defocus", "glare+defocus", "very_dark"):
        assert len(inspect(Scene.make(3, "pit", c))["trace"]) <= 7


def test_lambda_routes():
    ui = handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/"}, None)
    assert ui["statusCode"] == 200 and "ActiveInspect" in ui["body"]
    r = handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/inspect",
                 "body": json.dumps({"seed": 7, "defect": "scratch", "condition": "dim"})}, None)
    body = json.loads(r["body"])
    assert body["verdict"] == "REJECT" and body["final_frame"]
    rv = handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/review",
                  "body": json.dumps({"id": body["id"], "decision": "MAYBE"})}, None)
    assert rv["statusCode"] == 400
