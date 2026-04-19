# LAMICBench++ sample pipeline

## Fast reproduction (recommended)

If you want to reproduce the released LAMICBench++ numbers in `bench/sample_scripts/README.md`, start here.

```bash
cd bench/sample_scripts/lamicbench_plus
# unpack the released layout pack beside this script
tar -xzf lamicbench_layout_pack.tar.gz
# run with default arguments
python sample_bench.py
```

With defaults, generated images are written to `bench/output/lamicbench_plus/images`.

After generation, evaluate with:

```bash
python bench/lamicbench_plus/eval_lamicbench.py
python bench/lamicbench_plus/print_score.py
```

## Main script and defaults

`sample_bench.py` defaults:

- layout JSON: `./lamicbench_layout_pack/lamicbench_layout.json`
- layout images: `./lamicbench_layout_pack/layout_images/`
- reference images: `bench/lamicbench_plus/lamicbench_plus_files/reference_images/`
- reference masks: `bench/lamicbench_plus/lamicbench_plus_files/reference_masks/`
- output: `bench/output/lamicbench_plus/`

Model paths are read from CLI or env vars:

- `KONTEXT_MODEL_PATH`
- `ADAPTER_PATH`

## Optional overrides

Use `python sample_bench.py --help` to override paths, seeds, width, batch size, and sample count.
