# ActiveInspect: technical report
*OpenCV AI Competition 2026. Overall track plus the **Agentic Vision** special award.*

## 1. Problem and users
Automated visual inspection on small and mid-size production lines (machined metal, moulded plastic, painted panels) fails quietly when conditions drift: a lamp ages, a part arrives at a slightly different height (defocus), or a shiny batch throws specular glare. A fixed single-shot pipeline keeps answering "PASS" because it can't see the defect, and defective parts escape. **Users:** line QA engineers and operators at SMEs that can't afford a vision integrator for every product change. **Beneficiaries:** downstream customers and safety (escaped defects), plus less scrap from over-rejection.

## 2. Approach: active perception
Instead of judging whatever frame it was given, the agent **measures image quality with OpenCV 5 and changes how it looks** until the evidence is good enough. If it can't get there, it hands the part to a human.

Loop (at most 7 steps): `observe` (OpenCV metrics) → `choose tool` → `act on camera` → `re-observe` → `PASS / REJECT / ESCALATE`.

| Visual evidence (OpenCV) | Next action |
|---|---|
| >1% saturated pixels (`img>=250`) | `enable_polarizer` (cross-polarised capture) |
| mean brightness outside 90–215 | `adjust_exposure` (exposure × 150/brightness, capped at 12×) |
| always once | `probe_focus`: trial capture at +focus, keep it if Laplacian variance improves >15% |
| candidate with score ≥ 0.15 | `zoom_verify`: optical zoom on the candidate's bbox, re-detect with scale-adapted parameters |
| no confirmed candidate yet | `average_frames`: average 6 frames (noise ~1/√6) to reveal faint defects |
| quality still out of spec | **ESCALATE** to the operator (UI review buttons, logged to DynamoDB) |

The **visual result changes the next tool call and the camera configuration**, which is the Agentic Vision qualifying requirement. An optional **Amazon Bedrock** planner (Converse API) can replace the deterministic policy. For safety it may only choose between the policy's proposed action and `finish`. Invalid output falls back to the policy.

## 3. OpenCV 5 implementation (`inspect_agent/vision.py`)
- **Quality:** `GaussianBlur` + `Laplacian(CV_64F).var()` (sharpness), mean brightness, saturation fraction, 1–99 percentile dynamic range.
- **Detection:** `medianBlur` background estimate (kernel scaled with zoom), dark residual, robust MAD noise estimate, threshold at 4σ, `morphologyEx` open/close with elliptical `getStructuringElement`, `connectedComponentsWithStats`, `minAreaRect` elongation to classify **scratch vs pit**, border-artefact suppression, and a defect score from strength margin × area.
- **Enhancement / visualisation:** `createCLAHE` (software baseline), `resize(INTER_CUBIC)`, `rectangle`/`putText` annotation, `imencode` for the UI and S3.
- Pinned `opencv-python-headless==5.0.0.93` (`cv2.__version__ == 5.0.0`), with a unit test asserting it.

## 4. AWS deployment (`aws/`)
- **AWS Lambda container image on arm64 (Graviton)** behind a **Function URL** serves the UI plus `/inspect` and `/review`.
- **Amazon S3** stores annotated final frames and trace JSON (30-day lifecycle, public access blocked).
- **Amazon DynamoDB** (on-demand) is the audit log of every decision and every human review.
- **Amazon Bedrock** is the optional LLM planner (`BEDROCK_MODEL_ID`).
- Infrastructure as code: `aws/template.yaml` (SAM). Deploy with `aws/deploy.sh`.

## 5. Evaluation (`eval/run_eval.py`, results in `eval/RESULTS.md`)
**Data:** a procedural part-surface generator with ground-truth defects (none / scratch / pit, with contrast drawn from a range that includes faint defects), rendered through a sensor model with exposure-dependent shot noise, read noise, 8-bit clipping, Gaussian defocus and a specular glare lobe. The policy was tuned on seeds 0–39. Results are on **held-out seeds 10000–10029**: 630 parts across 7 conditions.

| System | Defect recall | Escaped defects | False rejects | Escalated | Auto-decision accuracy | Captures/part | ms/part (x86 dev box) |
|---|---|---|---|---|---|---|---|
| Single shot | 9.8% | 379 | 0 | 0 | 39.8% | 1.0 | 8 |
| Single shot + CLAHE | 9.0% | 382 | 0 | 0 | 39.4% | 1.0 | 7 |
| **ActiveInspect agent** | **90.7%** | **8** | **0** | 61 | **98.6%** | 7.2 | 46 |

Recall by condition: nominal 68% → 97%. Dim, defocus, glare and the compound conditions go from 0% → 97–98%. In **very_dark** (light too low even at maximum exposure), recall is 52%, and the agent **escalates 59 of 90 parts instead of guessing**.

**Failure cases** (`eval/failures/`):
1. Very faint pits (contrast near the lower end of the range) stay below the threshold even after averaging. They are passed, so 8 defects escape. Mitigation: more frames, adaptive threshold per product, or a learned detector.
2. very_dark: the image can't be brought into spec, so the agent escalates. This is correct but costs operator time. The fix is physical (lighting), and the trace says so.
3. Cost: 7 captures per part vs 1. That is acceptable at about 50 ms of compute, but camera time dominates on a real line. The policy skips `zoom_verify` when nothing is found, and `average_frames` could be gated on product risk.

**Limitations (honest):** results come from a simulator we built, so they show the *control logic* works and are not a field accuracy figure. The detector is classical (no learned model) and only finds dark anomalies. The camera actions map to real hardware (exposure, liquid-lens focus, polariser, PTZ/zoom), but the driver is simulated. Latency was measured on an x86 dev box; Lambda arm64 timing should be measured after deploy.

## 6. Responsible use
No people and no personal data are imaged. Every automated decision is explainable (trace of tools and metrics) and audited. Uncertain cases go to a human by design. The Bedrock planner can't take actions outside the safe set. The tool shouldn't be the sole gate for safety-critical parts without validation on real parts.

## 7. Reproduce
```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q                 # 7 tests
PYTHONPATH=. python -m eval.run_eval 30
PYTHONPATH=. uvicorn inspect_agent.server:app --port 8000   # UI at http://localhost:8000
bash aws/deploy.sh                     # AWS (needs credentials)
```
