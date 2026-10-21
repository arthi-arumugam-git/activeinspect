"""AWS Lambda (container image, arm64/Graviton) behind a Function URL."""
import json
from pathlib import Path
from inspect_agent.service import run_inspection, human_review

UI = (Path(__file__).parent / "ui.html").read_text()


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    if method == "GET" and path in ("/", "/index.html"):
        return {"statusCode": 200, "headers": {"Content-Type": "text/html"}, "body": UI}
    body = json.loads(event.get("body") or "{}")
    try:
        if path == "/inspect":
            res = run_inspection(body.get("seed", 1), body.get("defect", "scratch"), body.get("condition", "dim"))
        elif path == "/review":
            res = human_review(body["id"], body["decision"], body.get("reviewer", "operator"))
        else:
            return {"statusCode": 404, "body": "not found"}
    except (KeyError, ValueError) as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(res)}
