# Z-Stack Image Alignment

Drift correction for time-series 3D fluorescence microscopy. Aligns a folder of `.tif` z-stacks to a common reference (the middle one in time) using 2D phase cross-correlation on max-projections.

## How it works

1. Load all `.tif` files, de-interleaving channels.
2. Pick the middle stack as the reference.
3. Project each stack onto XY, XZ, YZ max-projections (normalized).
4. Register each projection of every moving stack to the reference.
5. Fuse the 3 per-axis shifts into one 3D shift (confidence-weighted).
6. Validate with NCC fitness; reject shifts that don't beat baseline.
7. Save aligned stacks, CSV report, and before/after plot.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Configure `INPUT_FOLDER`, `OUTPUT_FOLDER`, `ALIGNMENT_CHANNEL` at the top of `main.py`.

## Input

- Multi-page `.tif` files with interleaved channels (C0, C1, C0, C1, …).
- All files must share the same shape after de-interleaving (others are skipped).
- Filenames determine time order.

## Output

- `<name>_aligned.tif` — one per input.
- `stack_shifts.csv` — per-stack shifts (dz, dy, dx), baseline and post-alignment NCC, improvement flag.
- `alignment_plot.png` — fitness before vs after.
