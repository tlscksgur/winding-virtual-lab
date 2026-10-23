# Project handoff

### Project information
- Project: winding-virtual-lab
- Updated: 2026-10-23 (Asia/Seoul)
- AI: Codex

### Current goal
Publish the latest source-only snapshot from the local working project.

### Completed
Latest implementation synchronized; credentials, private deployment details,
training data, models and runtime outputs excluded.

### In progress
Snapshot validated; publication targets the existing main branch.

### Changed files
See the associated Git commit diff.

### Validation
Python unittest: 26 passed with PYTHONUTF8=1. Current served JavaScript syntax checks passed. sim-scene-body.js is a build template fragment, not a standalone script. Secret/internal address scan: no issues.

### Known limitations
External GPU/model services or factory calibration are not validated by source tests.

### Next steps
Configure a Python environment and deployment settings using README.md.

### Decisions
Keep this sanitized source repository separate from operational data and credentials.
