import numpy as np
import pandas as pd
from pathlib import Path
from ripser import ripser


# -----------------------------
# Helpers
# -----------------------------

def ternarize(X, tau):
    T = np.zeros_like(X)
    T[X > tau] = 1
    T[X < -tau] = -1
    return T


def max_persistence(diagram):
    if len(diagram) == 0:
        return 0.0
    pers = diagram[:, 1] - diagram[:, 0]
    return float(np.max(pers))


def compute_persistence(T):
    dgms = ripser(T, maxdim=2)["dgms"]
    H0 = max_persistence(dgms[0])
    H1 = max_persistence(dgms[1])
    H2 = max_persistence(dgms[2])
    return H0, H1, H2


def null_permute(T):
    Tnull = T.copy()
    for j in range(T.shape[1]):
        np.random.shuffle(Tnull[:, j])
    return Tnull


# -----------------------------
# Main
# -----------------------------

def main():
    inp = Path("hepdata_to_dashi")

    if not inp.exists():
        raise RuntimeError("Directory hepdata_to_dashi not found.")

    subdirs = [d for d in inp.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError("No observable subdirectories found.")

    all_rows = []

    for d in subdirs:
        csvs = list(d.glob("*.csv"))
        if not csvs:
            continue

        for f in csvs:
            if "GLOBAL" in f.name.upper():
                continue

            df = pd.read_csv(f)

            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 10:
                continue

            # Drop first numeric column if it's bin/index
            first_col = numeric_cols[0].lower()
            if first_col in ["bin", "index"]:
                numeric_cols = numeric_cols[1:]

            # Take first 10 numeric columns as lens dimensions
            lens_cols = list(numeric_cols[:10])

            if len(lens_cols) != 10:
                continue

            all_rows.append(df[lens_cols].values)

    if len(all_rows) == 0:
        raise RuntimeError("No lens columns found in any subdirectory.")

    X = np.vstack(all_rows)

    print("\n==============================")
    print("Total ternary points:", X.shape[0])
    print("Lens dimension:", X.shape[1])
    print("==============================\n")

    # -----------------------------
    # τ Sweep
    # -----------------------------
    taus = [0.15, 0.25, 0.35, 0.45]

    print("=== τ Sweep (real data) ===")
    for tau in taus:
        T = ternarize(X, tau)
        H0, H1, H2 = compute_persistence(T)
        print(
            f"tau={tau:.2f} | "
            f"H1_max={H1:.4f} | "
            f"H2_max={H2:.4f}"
        )

    # -----------------------------
    # Null control
    # -----------------------------
    print("\n=== Null Control (tau=0.25) ===")

    tau = 0.25
    T = ternarize(X, tau)
    real_H1 = compute_persistence(T)[1]

    null_vals = []
    for _ in range(10):
        Tn = null_permute(T)
        _, H1n, _ = compute_persistence(Tn)
        null_vals.append(H1n)

    null_vals = np.array(null_vals)

    print("Real H1:", round(real_H1, 4))
    print("Null H1 values:", np.round(null_vals, 4))
    print("Mean Null H1:", round(float(null_vals.mean()), 4))
    print("Std Null H1:", round(float(null_vals.std()), 4))

    if real_H1 > null_vals.mean() + 2 * null_vals.std():
        print("\nResult: H1 is statistically above null (structured loop).")
    else:
        print("\nResult: H1 not clearly above null (possibly discretization).")


if __name__ == "__main__":
    main()
