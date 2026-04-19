# COCO-MIG sample pipeline

## Fast reproduction (recommended)

If you only want to reproduce the released numbers in `bench/sample_scripts/README.md`, use the packaged references directly.

```bash
cd bench/sample_scripts/cocomig
# unpack the released pack beside this script
tar -xzf cocomig_reference_pack.tar.gz
# run with default arguments
python run_sample_bench_from_pack.py
```

This command uses the default pack folder `./cocomig_reference_pack` and writes outputs under `bench/output/cocomig/`.

Then run your evaluation pipeline to get final COCO-MIG metrics.

## Full pipeline (rebuild references from scratch)

```bash
cd bench/sample_scripts/cocomig
python run_cocomig.py --help
```

Run order:
1. `build-categories`
2. `gen-reference`
3. `crop-reference`
4. `filter mask`
5. `filter sync`
6. `sample-bench`

## Configuration

All default paths are in `cocomig_config.py` (`USER CONFIG` block):

- `REPO_ROOT`: ContextGen repo root.
- `DATA_DIR`: benchmark JSON and intermediate assets.
- `FLUX_MODEL_PATH`: model path for `gen-reference`.
- `GROUNDING_DINO_MODEL`: model path/id for `crop-reference`.
- `GENERATION_REPO`: project root containing `src/`.
- `TRAIN_CONFIG_YAML`: optional, not required by `sample-bench`.

CLI flags can override config values without editing the file (for example `--data-dir`, `--generation-repo`).

## Key files

- `run_sample_bench_from_pack.py`: direct reproduction from unpacked pack.
- `run_cocomig.py`: full CLI pipeline.
- `pipeline/reference.py`: category build, reference generation, crop, filtering.
- `pipeline/sample_bench.py`: full-scene sampling.
- `pipeline/io.py`: benchmark readers and mask path helpers.
