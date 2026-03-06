#!/usr/bin/env python3
"""
Make alpha–beta contour plots for all polars + derivatives in a CSV.

- X axis: beta
- Y axis: alpha
- Outputs: one multi-page PDF
- Uses plain-English titles (with the original coefficient in parentheses)

Example:
  python make_alpha_beta_contours.py aero_database_2d.csv -o contours.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---- Plain-English labels ----
LABEL_MAP = {
    "airspeed": "Airspeed",
    "CL": "Lift coefficient",
    "CD": "Drag coefficient",
    "CY": "Side force coefficient",
    "Cl": "Rolling moment coefficient",
    "Cm": "Pitching moment coefficient",
    "Cn": "Yawing moment coefficient",
    # Angle-of-attack derivatives (alpha)
    "CLa": "Lift change with angle of attack",
    "CDa": "Drag change with angle of attack",
    "CYa": "Side force change with angle of attack",
    "Cla": "Rolling moment change with angle of attack",
    "Cma": "Pitch stability (static)",
    "Cna": "Yawing moment change with angle of attack",
    # Sideslip derivatives (beta)
    "CLb": "Lift change with sideslip",
    "CDb": "Drag change with sideslip",
    "CYb": "Side force change with sideslip",
    "Clb": "Dihedral effect",
    "Cmb": "Pitching moment change with sideslip",
    "Cnb": "Weathercock stability",
    # Roll-rate derivatives (p)
    "CLp": "Lift change with roll rate",
    "CDp": "Drag change with roll rate",
    "CYp": "Side force change with roll rate",
    "Clp": "Roll damping",
    "Cmp": "Pitching moment change with roll rate",
    "Cnp": "Yawing moment change with roll rate",
    # Pitch-rate derivatives (q)
    "CLq": "Lift change with pitch rate",
    "CDq": "Drag change with pitch rate",
    "CYq": "Side force change with pitch rate",
    "Clq": "Rolling moment change with pitch rate",
    "Cmq": "Pitch damping",
    "Cnq": "Yawing moment change with pitch rate",
    # Yaw-rate derivatives (r)
    "CLr": "Lift change with yaw rate",
    "CDr": "Drag change with yaw rate",
    "CYr": "Side force change with yaw rate",
    "Clr": "Rolling moment change with yaw rate",
    "Cmr": "Pitching moment change with yaw rate",
    "Cnr": "Yaw damping",
}


def infer_unit(vals: np.ndarray) -> str:
    """
    Crude heuristic:
      if max(|value|) > ~3 -> degrees
      else -> radians
    """
    vmax = np.nanmax(np.abs(vals))
    return "deg" if vmax > 3 else "rad"


def get_label(var: str) -> str:
    base = LABEL_MAP.get(var, var)
    # include original variable in parentheses for clarity
    if var in LABEL_MAP:
        return f"{base} ({var})"
    return base


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv", type=str, help="Path to CSV with alpha, beta, and coefficients"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default="alpha_beta_contours.pdf",
        help="Output PDF filename",
    )
    ap.add_argument("--alpha-col", type=str, default="alpha", help="Alpha column name")
    ap.add_argument("--beta-col", type=str, default="beta", help="Beta column name")
    ap.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Extra columns to exclude from plotting",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=60,
        help="Number of contour levels (default 60)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    alpha_col = args.alpha_col
    beta_col = args.beta_col

    if alpha_col not in df.columns or beta_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{alpha_col}' and '{beta_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    # units + axes labels
    alpha_unit = infer_unit(df[alpha_col].to_numpy(float))
    beta_unit = infer_unit(df[beta_col].to_numpy(float))
    x_label = f"Beta ({beta_unit})"
    y_label = f"Alpha ({alpha_unit})"

    # choose numeric variables to plot
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_to_plot = [c for c in numeric_cols if c not in [alpha_col, beta_col]]
    vars_to_plot = [c for c in vars_to_plot if c not in set(args.exclude)]

    if not vars_to_plot:
        raise ValueError("No numeric variables found to plot.")

    output_pdf = Path(args.output)

    with PdfPages(output_pdf) as pdf:
        for var in vars_to_plot:
            # Build an alpha x beta grid via pivot
            piv = df.pivot_table(
                index=alpha_col, columns=beta_col, values=var, aggfunc="mean"
            )
            piv = piv.sort_index(axis=0).sort_index(axis=1)

            xg = piv.columns.to_numpy(dtype=float)  # beta grid
            yg = piv.index.to_numpy(dtype=float)  # alpha grid
            Z = piv.to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            title = get_label(var)

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            # handle constant / NaNs
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)

            if not np.isfinite(zmin) or not np.isfinite(zmax):
                ax.text(
                    0.5,
                    0.5,
                    "Non-finite values only",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            elif np.isclose(zmin, zmax):
                # constant surface -> just show points + value
                X, Y = np.meshgrid(xg, yg)
                ax.scatter(X.ravel(), Y.ravel(), s=5)
                ax.text(
                    0.5,
                    0.08,
                    f"Constant value: {float(zmin):.6g}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                X, Y = np.meshgrid(xg, yg)
                # default matplotlib colormap (no manual colors)
                cs = ax.contourf(X, Y, Z, levels=args.levels)
                fig.colorbar(cs, ax=ax, label=title)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {output_pdf.resolve()}")


if __name__ == "__main__":
    main()
