"""Top-level wrapper to execute the rotation demo script.

This shim keeps the repository root entrypoint stable for CI while delegating
to the actual implementation under ``dashifine/Main_with_rotation.py``.
"""

from __future__ import annotations

import argparse

from dashifine.Main_with_rotation import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="examples")
    parser.add_argument("--res_hi", type=int, default=4)
    parser.add_argument("--res_coarse", type=int, default=2)
    parser.add_argument("--num_rotated", type=int, default=1)
    parser.add_argument("--z0_steps", type=int, default=1)
    parser.add_argument("--w0_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_dir=args.output_dir,
        res_hi=args.res_hi,
        res_coarse=args.res_coarse,
        num_rotated=args.num_rotated,
        z0_steps=args.z0_steps,
        w0_steps=args.w0_steps,
    )
