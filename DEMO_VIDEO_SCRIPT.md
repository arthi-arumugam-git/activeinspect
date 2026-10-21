# Demo video script: ActiveInspect (target 3:30, limit 5:00)
The rules require the video to show **the team**, the app working, the architecture and the principal results. Upload to YouTube as Unlisted or Public.

**0:00–0:20 Team + hook** (face cam): "I'm Arthi, an AI engineer from India. I build evaluation tools for AI systems. Inspection cameras fail quietly: when the light dims, they don't say 'I can't see', they say PASS."

**0:20–0:50 Problem.** Open the deployed **Lambda Function URL**. Set Condition = dim, Defect = scratch, seed 7. Point at the first shot: almost black. "The single-shot baseline passes this part. Across our benchmark it caught under 10% of defects once conditions drifted."

**0:50–1:50 Agent at work.** Press **Inspect part** and walk through the trace: `adjust_exposure` (brightness was 20, so it raises exposure) → `probe_focus` (sharpness didn't improve, so it keeps the focus) → `average_frames` → `zoom_verify` (score 0.99) → REJECT, with the box drawn. "Every step was chosen because of what OpenCV measured in the last frame."
Then glare + pit: the first action is `enable_polarizer` because 17% of pixels were saturated. Then defocus: `probe_focus` refocuses (sharpness ↑).

**1:50–2:20 Human in the loop.** Condition = very_dark → ESCALATE: "Image quality couldn't be brought into spec." Click **Reject part**, then show the DynamoDB item in the AWS console (decision + human_review) and the S3 objects (final.png, trace.json).

**2:20–2:50 Architecture.** Show `docs/screenshots/architecture.png`: OpenCV 5 tools, the agent loop, Lambda arm64, S3, DynamoDB, the optional Bedrock planner restricted to safe actions, and the SAM template.

**2:50–3:20 Results.** Show `eval/RESULTS.md`: recall 9.8% → 90.7%, escaped 379 → 8, 0 false rejects. Show two failure images from `eval/failures/` (faint pits) and say plainly: simulated data, 7 captures per part.

**3:20–3:30 Close.** "ActiveInspect: look again before you decide, and ask when you can't."
