# Repository Guidelines

## Project Structure & Module Organization
- `src/flux/`: custom FLUX.1-Kontext pipeline/attention/transformer extensions.
- `src/model/`: training and inference logic (`train.py`, `generate.py`, `model.py`, dataset/callbacks).
- `src/utils/`: shared file, text, and image utilities.
- `inference.py`: main CLI entry for demo/custom generation.
- `train/config/config.yaml`: primary training configuration.
- `gui/backend/app.py`: Flask backend for interactive GUI.
- `gui/frontend/`: Vite + React frontend (`src/components`, `src/api`, `src/theme`).
- `bench/`: benchmark and sample evaluation scripts.
- `images/input` and `images/output`: example assets and generated outputs.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `python inference.py --input_dir ./images/input --output_dir ./images/output --num_samples 4 --image_size 768`: run inference.
- `python src/model/train.py`: start training (after editing `train/config/config.yaml`).
- `python gui/backend/app.py`: run GUI backend on port `5000`.
- `cd gui/frontend && npm install && npm run dev`: run frontend dev server on port `5173`.
- `cd gui/frontend && npm run build`: production frontend build.
- `cd gui/frontend && npm run lint`: run ESLint checks.

## Coding Style & Naming Conventions
- Python: follow PEP 8 style, 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- React/JS: components in `PascalCase` (for example `CanvasEditor.jsx`), hooks/utilities in `camelCase`.
- Keep module names descriptive and task-oriented (for example `pipeline_preprocess.py`, `image_process.py`).
- Read runtime paths/secrets from `.env` (see `.env_template`); never hardcode credentials.

## Testing Guidelines
- There is no dedicated automated test suite yet (`pytest`/`vitest` not configured).
- Minimum validation before PR:
  - `cd gui/frontend && npm run lint`
  - run one inference command and verify output under `images/output/`
  - for training-related changes, run a short sanity training pass with adjusted config.
- If adding tests, place Python tests under `tests/` with `test_*.py` naming.

## Commit & Pull Request Guidelines
- Existing history uses short, imperative messages (for example `add gui demo`, `update readme`).
- Prefer: `<scope>: <imperative summary>` (for example `gui: fix canvas mask export`).
- PRs should include:
  - clear problem/solution summary,
  - linked issue (if any),
  - exact reproduction/verification commands,
  - screenshots/GIFs for GUI changes.
