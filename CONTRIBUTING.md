# Contributing to Photo Enhancer

Thank you for your interest in contributing!

## Getting started

1. Fork the repository and clone locally.
2. Set up the backend: `cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
3. Set up the frontend: `cd frontend && npm install && npm run build`
4. Run dev mode (see README) or `python launcher.py`.

## Pull requests

- Keep changes focused — one feature or fix per PR when possible.
- Match existing code style (Python: services in `backend/services/`, React + Tailwind in `frontend/src/`).
- Do not commit secrets, `.env`, model weights, `venv/`, `node_modules/`, or `dist/`.
- Test: `npm run build` in `frontend/` and verify backend imports (`python -c "import main"` from `backend/`).

## Reporting issues

Include OS version, Python/Node versions, steps to reproduce, and whether you're using the browser launcher or packaged `.exe`.

## AGPL note

Changes touching SAM/YOLO/Ultralytics should mention AGPL implications in the PR description if behavior or bundling changes.

## Code of conduct

Be respectful and constructive. We welcome bug fixes, docs improvements, and local-feasible feature additions aligned with the project's privacy-first scope.
