# dashifine

| <p align="center">![ezgif-3f0c8b20812b0d](https://github.com/user-attachments/assets/58af7f55-ac1c-406b-901e-95fd49ca3ed4)</p> 
| Image 1 | Image 2 | Image 3 |
| :---: | :---: | :---: |
| <img width="300" alt="1d44df91-9532-4698-b651-052d0916f1f6" src="https://github.com/user-attachments/assets/f6817717-261a-43d7-a7e5-5d7c3da7853f" /> | <img width="300" alt="addc21ad-031c-4a17-a5e7-6c0be56acd7d" src="https://github.com/user-attachments/assets/35dbed02-8545-4537-9ff6-dc9a06e534b1" /> | <img width="300" alt="pants_example_fixed" src="https://github.com/user-attachments/assets/e004af2d-d85c-4b4b-a650-1e9bed3633cc" /> |
| <img width="300" alt="output" src="https://github.com/user-attachments/assets/5e69925c-bb19-4fca-abd6-6247b26bb5f8" /> | <img width="300" alt="output(1)" src="https://github.com/user-attachments/assets/8a62cb7b-a0dc-4729-8eba-e315d8b27540" /> | <img width="300" alt="generalized_pants_nw1_nl2" src="https://github.com/user-attachments/assets/7c445993-2916-402d-b366-f546ef0aba6a" /> |
| <img width="300" alt="n_pants_with_seams" src="https://github.com/user-attachments/assets/a4336306-d4b8-4fbc-aeee-b79c683a10e5" /> | <img width="300" alt="nwaists_nlegs_pants" src="https://github.com/user-attachments/assets/73237f25-1eae-4e19-aae9-b1d69122dca5" /> | <img width="300" alt="output(3)" src="https://github.com/user-attachments/assets/bed9765d-6aa3-4d79-ac13-d6437e53498a" /> |
| <img width="300" alt="output(4)" src="https://github.com/user-attachments/assets/87802c0e-5930-48a0-b2e2-64fea7f3a04d" /> | <img width="300" alt="output(5)" src="https://github.com/user-attachments/assets/634498ea-17df-427a-bbcc-b9488f3a59a2" /> | <img width="300" alt="output(6)" src="https://github.com/user-attachments/assets/b46d8a3b-89ae-42fc-be3d-4d13f26372b1" /> |
| <img width="300" alt="output(7)" src="https://github.com/user-attachments/assets/25db6e56-ced7-4830-bae4-086926235191" /> | <img width="300" alt="output(8)" src="https://github.com/user-attachments/assets/49dfda5b-d200-4203-867b-7887df988b16" /> | <img width="300" alt="output(9)" src="https://github.com/user-attachments/assets/37cc9322-a19c-4dc1-8269-e3e0b780ef24" /> |
| <img width="300" alt="output(10)" src="https://github.com/user-attachments/assets/8a9a22d4-af20-4f77-af1b-d9146d94eaef" /> | <img width="300" alt="output(11)" src="https://github.com/user-attachments/assets/00566baa-6589-4712-9e5a-d776b1771aa5" /> | <img width="300" alt="output(12)" src="https://github.com/user-attachments/assets/b9ad1b98-0359-415d-9780-8f41d735320f" /> |
| <img width="1000" height="600" alt="Figure_0" src="https://github.com/user-attachments/assets/3fd94cf4-8d8f-4a52-9674-565d5ddcb6ec" /> | <img width="1000" height="1000" alt="Figure_1" src="https://github.com/user-attachments/assets/91f3e4ca-8e80-4956-896f-5b88df644d17" /> | |


---



<img width="6715" height="10737" alt="NotebookLM Mind Map(3)" src="https://github.com/user-attachments/assets/5ed0fefa-29cc-4747-a918-e9178f579a81" />



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
# default CMY palette
python Main_with_rotation.py --output_dir examples --palette cmy

# eigen palette
python Main_with_rotation.py --output_dir examples --palette eigen

# lineage palette
python Main_with_rotation.py --output_dir examples --palette lineage
```

All output images are written to `/mnt/data`, including a coarse density map and PNG files for the origin slice and each rotation.

`PATCH_DROPIN_SUGGESTED.py` also supports temporal rendering.  Passing `--num_time N` steps the slice origin through N normalised time values (0 to 1), writing files like `slice_t0_rot_0deg.png` for each time step and rotation.

### P-adic palette

The `render` function accepts two 2D arrays per pixel:

* `addresses` – integer p-adic addresses.
* `depth` – floating-point depth values.

Setting `palette="p_adic"` maps `addresses` to hue and `depth` to saturation,
producing an RGB image via HSV conversion.

### Palette options

`Main_with_rotation.py` exposes a `--palette` flag to control colouring. The
available choices are `cmy`, `lineage`, and `eigen`. For example, to render
using the lineage palette:

```bash
python Main_with_rotation.py --output_dir examples --palette lineage
```

The default `cmy` palette blends cyan, magenta, and yellow.  `lineage` assigns a
stable hue based on each centre's address, while `eigen` currently falls back to
grayscale until a PCA-based colouring is implemented.


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

- Run `python Main_with_rotation.py --output_dir examples --num_rotated 8` and confirm successive images differ.
- Observe translucent voids where density is low.
- Move centres to compare soft vs. crisp class transitions.

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

