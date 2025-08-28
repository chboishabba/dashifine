"""Render a central RGBA slice of the 4-D CMYK field.

This script uses :func:`demo_rgba.cmyk_slice_rgba` to evaluate the slice at
``(z0=0.125, w0=-0.05)``, approximately the mean of the class centres in the
``(z, w)`` dimensions.  The resulting image is saved with an alpha channel that
encodes overall field strength.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from demo_rgba import cmyk_slice_rgba


def main() -> None:
    """Generate and save the central RGBA slice."""
    z0 = 0.125
    w0 = -0.05
    img = cmyk_slice_rgba(z0, w0)

    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title("CMYK 4D \u2192 2D Slice with Alpha (z0=0.125, w0=-0.05)")
    plt.tight_layout()

    out_path = "cmyk_4d_slice_rgba_center.png"
    plt.imsave(out_path, img)
    plt.show()
    print(out_path)


if __name__ == "__main__":
    main()
