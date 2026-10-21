# Devpost submission: OpenCV AI Competition 2026
**Deadline: Oct 27, 2026, 12:15 IST** (Devpost shows Oct 26, 11:45 pm PDT; the overview says 11:59 pm PT, so use the earlier time). Submit by about 09:00 IST.
**Eligibility:** open to entrants above the age of majority in all countries except the standard exceptions (Devpost page). India is eligible. The rules' only other exclusions are organiser/judge staff.

**Project name:** ActiveInspect

**Tagline:** An OpenCV 5 inspection agent that changes how it looks (exposure, polariser, focus, zoom) before deciding, and asks a human when it still can't see.

**Featured path:** Agentic Vision (yes). COOL: no.

## Description
### Problem and users
Single-shot inspection fails quietly when the light dims, focus drifts or glare appears. In our benchmark, a fixed pipeline caught only 9.8% of defects across realistic condition drift. Users: QA engineers and operators on small and mid-size production lines.

### What it does
For each part, the agent captures a frame and measures it with OpenCV 5 (brightness, saturation, Laplacian sharpness, anomaly candidates). It then picks the next camera action: cross-polariser for glare, an exposure change for dim light, a focus probe, 6-frame averaging for faint defects, and zoom-to-verify on a candidate. It finishes with PASS / REJECT, or ESCALATE to an operator, who decides in the UI. Every step is traced and audited.

### How we built it
- **OpenCV 5.0:** medianBlur background model, MAD thresholding, morphology, connectedComponentsWithStats, minAreaRect scratch/pit classification, Laplacian focus measure, CLAHE, annotation.
- **AWS:** Lambda container on arm64/Graviton with a Function URL, S3 (frames and traces), DynamoDB (audit log of decisions and human reviews), optional Bedrock planner (Converse API) restricted to policy-safe actions. All deployed with SAM.

### Evaluation
630 held-out parts across 7 conditions. Recall 9.8% → 90.7%, escaped defects 379 → 8, 0 false rejects, 98.6% accuracy on automatic decisions, 61 escalations (59 of them in the "very dark" condition, where a human should look). The report covers the failure cases (very faint pits), the cost (7.2 captures per part) and the limitation that the data is simulated.

### Responsible use
No personal data is imaged. Decisions are explainable, a human stays in the loop for uncertain parts, and the LLM can't take actions outside the safe set.

## Links and files (fill in)
- Code repo (public or judge-accessible): `https://github.com/arthi-arumugam-git/activeinspect` ← push
- Working endpoint: Lambda Function URL from `aws/deploy.sh` ← deploy
- Video (≤5 min, public/unlisted, must show the team): ← record
- Technical report: `REPORT.md` (export to PDF if the form asks for a file: `pandoc REPORT.md -o report.pdf`)
- Architecture diagram: `docs/screenshots/architecture.png`
- Team: Arthi Arumugam (solo). Bio: AI engineer in India; 26 merged OSS PRs across UK AISI inspect_evals, DeepEval, LiveKit Agents, pipecat, Haystack and TruLens.
- Built with: opencv, python, numpy, aws-lambda, graviton, amazon-s3, amazon-dynamodb, amazon-bedrock, aws-sam, fastapi
