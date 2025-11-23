# Lean project notes

- Continuous integration builds Lean using the `leanprovercommunity/lean4:latest` container image.
- The Lean version for the project is pinned via [`lean-toolchain`](lean-toolchain); update that file to switch versions.
- For local development, install [`elan`](https://github.com/leanprover/elan) and run `lake build` from this directory.
