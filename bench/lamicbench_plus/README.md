# LAMICBench++ Evaluation

## Benchmark package (`lamicbench_plus_files.tar.gz`)

We provide `bench/lamicbench_plus/lamicbench_plus_files.tar.gz` for evaluation setup.

This package contains:

- **benchmark metadata** (including `lamicbench_plus.json`)
- **reference images** (and related benchmark assets used by evaluation)

## What this folder provides

- `eval_lamicbench.py`: computes per-sample metrics and writes `metrics.json`
- `print_score.py`: aggregates `metrics.json` under `bench/output/*` and prints score tables

## 1) Install dependencies

From repo root:

```bash
pip install -r bench/lamicbench_plus/requirements.txt
```

## 2) Prepare environment variables

Create `bench/lamicbench_plus/.env` and set:

- `FACE_MODEL_ROOT`
- `CLIP_MODEL_ROOT`
- `DINO_MODEL_ROOT`
- `SAM2_CKPT`
- `GROUNDINGDINO_CKPT`
- `IMPROVED_AES_MODEL_PATH`
- `VQA_MODEL_ROOT`

## 3) Run evaluation

From repo root:

```bash
python bench/lamicbench_plus/eval_lamicbench.py
```

Default paths:

- generated images: `bench/output/lamicbench_plus/images`
- benchmark metadata: `bench/lamicbench_plus/lamicbench_plus_files/lamicbench_plus.json`
- output metrics: `bench/output/lamicbench_plus/metrics.json`

## 4) Print score tables

From repo root:

```bash
python bench/lamicbench_plus/print_score.py
```

This scans model folders under `bench/output/` and reads each `<model>/metrics.json`.
