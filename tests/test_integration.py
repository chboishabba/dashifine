import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashifine.Main_with_rotation import (
    gelu,
    main,
    orthonormalize,
)


def test_main_creates_outputs(tmp_path):
    summary = main(
        output_dir=tmp_path,
        res_hi=4,
        res_coarse=2,
        num_rotated=1,
        z0_steps=1,
        w0_steps=1,
        slopes=np.array([0.0], dtype=np.float32),
    )
    for p in summary["paths"].values():
        assert Path(p).exists()
