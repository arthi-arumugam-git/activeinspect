"""Local dev server with the same routes as the Lambda Function URL."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from inspect_agent.service import run_inspection, human_review

app = FastAPI(title="ActiveInspect")


class InspectReq(BaseModel):
    seed: int = 1
    defect: str = "scratch"
    condition: str = "dim"


class ReviewReq(BaseModel):
    id: str
    decision: str
    reviewer: str = "operator"


@app.get("/", response_class=HTMLResponse)
def ui():
    return (Path(__file__).parent / "ui.html").read_text()


@app.post("/inspect")
def inspect_(r: InspectReq):
    return run_inspection(r.seed, r.defect, r.condition)


@app.post("/review")
def review(r: ReviewReq):
    return human_review(r.id, r.decision, r.reviewer)
