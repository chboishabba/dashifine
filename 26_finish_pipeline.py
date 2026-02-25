#!/usr/bin/env python3
# 26_finish_pipeline.py
import argparse, subprocess, os

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run(["python", "26_operator_jacobian_v2.py",
         "--beta-root", args.beta_root,
         "--out", args.out,
         "--tail", "5"])
    run(["python", "26_quadratic_fit_v2.py",
         "--J", os.path.join(args.out, "J_global.npy"),
         "--out", args.out])

if __name__ == "__main__":
    main()
