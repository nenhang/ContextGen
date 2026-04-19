# LayoutSAM sample pipeline

## Fast reproduction (recommended)

If your goal is to reproduce the released numbers in `bench/sample_scripts/README.md`, use the packaged references first.

```bash
cd bench/sample_scripts/layoutsam
# unpack the released pack beside this script
tar -xzf layoutsam_reference_pack.tar.gz
# run with default arguments
python run_sample_bench_from_pack.py
```

This uses default `./layoutsam_reference_pack` and writes outputs under `bench/output/layoutsam/`.

Then run your evaluation pipeline to obtain final LayoutSAM metrics.

## Full pipeline (rebuild references from scratch)

```bash
cd bench/sample_scripts/layoutsam
python run_layoutsam.py --help
```

Run order:
1. `gen-segments`
2. `valid-bbox`
3. `filter-benchmark`
4. `sample-bench`

## Configuration

Defaults live in `layoutsam_config.py` (`USER CONFIG` block):

- `DATA_DIR`: benchmark JSON + generated intermediate files.
- `FLUX_MODEL_PATH`: model path/id for `gen-segments`.
- `FLUX_TURBO_LORA_PATH`: optional turbo LoRA for fast segment generation.
- `GROUNDING_DINO_MODEL`: model path/id for `valid-bbox`.
- `GENERATION_REPO`: project root containing `src/`.
- `KONTEXT_MODEL_PATH`, `ADAPTER_PATH`: model paths for `sample-bench`.

You can override config fields via CLI flags (for example `--data-dir`, `--kontext-model-path`, `--adapter-path`).

## Key files

- `run_sample_bench_from_pack.py`: direct reproduction from unpacked pack.
- `run_layoutsam.py`: full CLI pipeline.
- `pipeline/reference.py`: segment generation, bbox validation, benchmark filtering.
- `pipeline/sample_bench.py`: full-scene sampling.
