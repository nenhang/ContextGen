# Benchmark Sample Scripts

## Important note on paper vs open-source results

The model used in the paper was trained with a **512-focused setup**.  
The released open-source model is a **generalized checkpoint across 512-1024 resolutions**.  
Because of this, benchmark numbers from the open-source release are expected to differ from the paper tables.

We reran all three benchmarks with the released model. Results are listed below.

## Re-tested results (released model)

### LAMICBench++

Fewer Instances:
| ITC | AES | IDS | IPS | AVG |
|---|---|---|---|---|
| 58.51 | 91.56 | 40.16 | 82.32 | 68.14 |

More Instances:
| ITC | AES | IDS | IPS | AVG |
|---|---|---|---|---|
| 59.73 | 83.04 | 38.02 | 76.34 | 64.28 |

### COCO-MIG

| SR | I-SR | mIoU | G-C | L-C |
|---|---|---|---|---|
| 28.00 | 64.91 | 60.25 | 25.82 | 21.59 |

### LayoutSAM

| Spatial | Color | Texture | Shape | CLIP | Pick |
|---|---|---|---|---|---|
| 94.13 | 86.41 | 88.11 | 87.57 | 27.75 | 22.88 |

## Reproduction entry points

If your goal is fast reproduction, start from these three READMEs.  
Each one begins with the shortest path: unpack the provided `tar.gz` and run the script with default args.

- `COCO-MIG`: [`bench/sample_scripts/cocomig/README.md`](cocomig/README.md)
- `LayoutSAM`: [`bench/sample_scripts/layoutsam/README.md`](layoutsam/README.md)
- `LAMICBench++`: [`bench/sample_scripts/lamicbench_plus/README.md`](lamicbench_plus/README.md)
