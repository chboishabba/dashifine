# Dashifine

This project searches a procedurally defined 4D color field for interesting 2D slices and renders them as images.  It evaluates candidate slices, refines the best result, and writes the chosen slice along with several rotated variants.

## Requirements
- Python 3.10+
- numpy
- matplotlib
- scipy

Install dependencies with:

```bash
pip install numpy matplotlib scipy
```

## Usage
Run the main script to perform the search and generate slices:

```bash
python Main_with_rotation.py
```

All output images are written to `/mnt/data`, including a coarse density map and PNG files for the origin slice and each rotation.

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
