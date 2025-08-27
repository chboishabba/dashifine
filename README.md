# dashifine

This project searches a procedurally defined 4D color field for interesting 2D slices and renders them as images.  It evaluates candidate slices, refines the best result, and writes the chosen slice along with several rotated variants.  The fourth dimension `w` represents normalised time, so the renderer can step through time to produce sequences of slices.

## Requirements
- Python 3.10+
- numpy
- matplotlib
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage
Run the main script to perform the search and generate slices:

```bash
python Main_with_rotation.py
```

All output images are written to `/mnt/data`, including a coarse density map and PNG files for the origin slice and each rotation.

`PATCH_DROPIN_SUGGESTED.py` also supports temporal rendering.  Passing `--num_time N` steps the slice origin through N normalised
time values (0 to 1), writing files like `slice_t0_rot_0deg.png` for each time step and rotation.

### Palette options

Use `--palette` to choose how class weights map to colour:

* `cmy` (default) – map the first three classes to cyan, magenta and yellow.
* `eigen` – visualise weights via the leading eigenvectors (grayscale placeholder).
* `lineage` – assign hues by lineage using an HSV mapping.

Example commands:

```bash
python Main_with_rotation.py --output_dir examples --palette cmy
python Main_with_rotation.py --output_dir examples --palette eigen
python Main_with_rotation.py --output_dir examples --palette lineage
```

### P-adic palette

The `render` function accepts two 2D arrays per pixel:

* `addresses` – integer p-adic addresses.
* `depth` – floating-point depth values.

Setting `palette="p_adic"` maps `addresses` to hue and `depth` to saturation,
producing an RGB image via HSV conversion.

## Configuration
`Main_with_rotation.py` exposes several constants at the top of the file that control behavior, such as:

```python
RES_HI = 420        # high-resolution output size
RES_COARSE = 56     # coarse search resolution
SHARPNESS = 2.8     # slice sharpness
ROT_BASE_DEG = 10.0 # rotation step size
NUM_ROTATED = 10    # number of rotated slices
```

Adjust these values (and others like search ranges or random seed) to experiment with different results.

## License
Distributed under the terms of the [Mozilla Public License 2.0](LICENSE).



further info:


---

dashifine

dashifine is a visual exploration tool for slicing and inspecting high-dimensional scalar fields — designed here for a 4D CMYK colour field — with an adaptive, two-phase search that balances speed and detail.

Overview

Most visualisation approaches either:

Render the entire high-dimensional field (which is prohibitively expensive), or

Choose arbitrary slices that may miss the most “interesting” regions.


dashifine instead:

1. Coarsely scans the space in fast, low-precision int8 to locate promising slices.


2. Refines only the most promising slice in full-precision float32.


3. Expands the view by rotating that slice to generate multiple perspectives.



This lets you see the “shape” of your data while keeping computation tractable.


---

How it works

1. Field definition

We define a continuous 4D field where each point’s CMYK weights are based on its Euclidean distance to predefined class centres. A GELU activation softens the edges.

2. Coarse int8 search

Instead of evaluating every slice in float32:

We grid-search over origin positions (z0, w0) and directional slopes in both axes.

We compute the field in 8-bit integers (uint8), which is much faster.

We score each candidate slice by combining field activity and colour variance.


3. Bound refinement

The best coarse slice is refined by testing upper and lower bounds based on the resolution step sizes in (z, w) and slope space.
We keep whichever bound scores higher in float32.

4. Perpendicular rotations

Once we have the “best” slice:

We compute a perpendicular axis in 4D space.

We rotate the slice plane around that axis to produce a fan of additional views.

In this demo, we generate 10 evenly-spaced angles, giving a sense of the structure around the best slice.



---
## Example Output

Sample output from a run of `Main_with_rotation.py`:

![Coarse Density Map](examples/coarse_density_map.png)
![Origin Slice](examples/slice_origin.png)
![Rotated Slice](examples/slice_rot_10deg.png)

---

## Manual QA

Use this checklist to verify the renderer behaves as expected:

- [ ] Run `python Main_with_rotation.py --output_dir examples --num_rotated 8` and confirm images change with angle.
- [ ] Verify low-density regions fade transparent ("inverse Swiss cheese").
- [ ] Move centres to see soft vs. crisp class boundaries.

Why not just use float32 everywhere?

Because the space of possible slices is huge. Even in 4D:

Brute-forcing all origins and directions at high resolution would take orders of magnitude more time and memory.

The coarse int8 phase can skip 99% of the search space.

The bound refinement step bridges the precision gap without losing much accuracy.



---

Usage

python dashifine.py

Outputs:

coarse_density_map.png

slice_origin_<...>.png

slice_rot_<angle>.png × N


Tune parameters at the top of the file:

RES_HI, RES_COARSE — resolution of refined/coarse passes

Z0_RANGE, Z0_STEPS, W0_RANGE, W0_STEPS — search bounds in the fixed dims

SLOPES — slopes for slice plane directions

NUM_ROTATED, ROT_BASE_DEG — how many rotated views, and angular spacing



---

Roadmap

Generalise to N-dimensional fields

Plug-in field definitions (not just CMYK)

Interactive viewer for navigating slices in real-time

Optional GPU acceleration



---

Do you want me to now also include a rendered example set of the 10 slices in the README so GitHub visitors can immediately see the output without running the code? That would make the repo more compelling visually.

