# Backend walkthrough for a project demo

This guide documents the existing Vehicle Damage AI backend flow for a live demo. It does not configure model weights or claim that real model inference has been validated.

## One-minute architecture

The React frontend sends HTTP requests to a FastAPI backend. The backend validates inputs, coordinates inspection services, stores structured records in PostgreSQL, and stores uploaded image files under the configured local storage root. Optional local models provide damage detections and vision-language reports.

```text
Browser → FastAPI routes → inspection/report workflow
                              ├─ validated image files
                              ├─ optional local YOLO detections
                              ├─ local Qwen-VL report generation
                              ├─ claim verification
                              └─ deterministic claim policy
                         → PostgreSQL + local image storage
```

## Walkthrough order

1. Open the API docs at `/docs` and show the versioned API. The documented report endpoints include `POST /api/v1/inspections/{id}/reports`, `GET /api/v1/reports/{id}`, and `POST /api/v1/reports/{id}/verifications`. Use the live OpenAPI page as the source of truth for route details on the running checkout.
2. Open `backend/app/services/inspection_workflow.py`. The coordinator marks an inspection as processing, runs detection, generates a report, verifies saved claims, and sets the final status. Errors mark the workflow failed; the underlying stage runs remain available for diagnosis.
3. Open `backend/app/services/report_service.py`. It gathers inspection images and the latest detection runs into an evidence bundle, calls the report generator, validates citations, and stores the report and its individual claims.
4. Open `backend/app/services/vlm/prompts.py`. The generator is prompted to return JSON claims about visible damage and cite only the image and detection aliases it was given. The prompt says not to assert repair costs, causes, hidden damage, or repair recommendations from photos.
5. Open `backend/app/services/verification_service.py` and then `backend/app/services/claim_policy.py`. The verifier checks claims against associated image evidence; the deterministic policy handles evidence integrity, claim type, detector conflicts, verifier failures, and review escalation.

## What “comparator” means here

There is no separate model named Comparator. The configured local Qwen-VL runner supplies both the structured report generator and the claim verifier. The optional YOLO detector is a separate detection component, not a report-comparison model.

The backend combines the verifier's raw answer with evidence and detector facts using deterministic policy code. A missing detection is not treated as proof that damage is absent. A detector conflict, invalid or missing evidence, unsupported claim type, or unusable verifier result can require human review. When the generator and verifier use the same model weights, a supported answer is not treated as independent confirmation and also requires review.

A separate cross-run consistency check can compare a claim's current raw verifier verdict with earlier completed runs. Disagreement adds review; agreement does not add trust.

## What the AI does and does not establish

The vision-language model can produce structured descriptions and compare a textual claim with photographs. A YOLO detector can provide a damage class, confidence, and location when configured with local weights. The deterministic policy applies explicit decision rules; it is not another trained model.

A photo-based system cannot establish hidden structural damage, repair cost, cause of damage, or severity without the required validated source or human confirmation. The application is a review aid, not an autonomous insurance approval system.

## Validation caveat

The repository's model configuration requires local model files. Do not present a code path or a successful API response from a stub as proof that real Qwen inference works. The project README records that model weights are not committed and real inference has not been run. Any live model demo should be described with the exact model, weights, and validation evidence used.

## Useful source files

- `backend/app/services/inspection_workflow.py`
- `backend/app/services/report_service.py`
- `backend/app/services/verification_service.py`
- `backend/app/services/claim_policy.py`
- `backend/app/services/vlm/prompts.py`
- `backend/app/services/vlm/provider.py`
