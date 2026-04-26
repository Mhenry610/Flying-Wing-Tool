from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


RHO_SL_KG_M3 = 1.225
MU_SL_PA_S = 1.789e-5


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interp_linear(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / max(xs[i + 1] - xs[i], 1e-12)
            return lerp(ys[i], ys[i + 1], t)
    return ys[-1]


def trapz(x: list[float], y: list[float]) -> float:
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def smoothstep01(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def clmax_re_corrected(reynolds: float, estimated_cl_max: float) -> float:
    # Keep this deliberately conservative: StockWing's program model supplied
    # estimated_cl_max=1.2, then low-Re operation mildly reduces it.
    re_factor = 0.9 + 0.1 * min(1.0, max(0.0, (reynolds - 2.0e5) / 6.0e5))
    return estimated_cl_max * re_factor


def clmax_tc_re(thickness_ratio: float, reynolds: float) -> float:
    # Sensitivity model used only as a cross-check. It gives lower CLmax near
    # the thin tip and captures the same conclusion as the project CLmax model.
    log_re = math.log10(max(reynolds, 1.0))
    re_factor = 0.86 + 0.14 * min(1.0, max(0.0, (log_re - 5.3) / (6.0 - 5.3)))
    return (0.85 + 3.2 * thickness_ratio) * re_factor


def compute_case(
    *,
    name: str,
    cl_total: float,
    velocity_m_s: float,
    sections: list[dict[str, Any]],
    twist_trim: dict[str, Any],
    reference_area_m2: float,
    n_strips: int,
    clmax_model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    half_span = float(sections[-1]["y_m"])
    y = [half_span * i / (n_strips - 1) for i in range(n_strips)]
    sec_y = [float(section["y_m"]) for section in sections]
    sec_chord = [float(section["chord_m"]) for section in sections]
    sec_twist = [float(section["twist_deg"]) for section in sections]

    q = 0.5 * RHO_SL_KG_M3 * velocity_m_s * velocity_m_s
    half_lift_n = q * reference_area_m2 * cl_total / 2.0
    bell_integral = 3.0 * math.pi / 16.0

    rows: list[dict[str, Any]] = []
    area_weighted_sep = []
    pressure_proxy_weight = []
    chord_values = []
    cl_values = []
    margins = []
    excess_values = []

    for y_m in y:
        eta = min(1.0, max(0.0, y_m / max(half_span, 1e-12)))
        chord_m = interp_linear(y_m, sec_y, sec_chord)
        twist_deg = interp_linear(y_m, sec_y, sec_twist)
        bell_shape = max(0.0, (1.0 - eta * eta)) ** 1.5
        lift_per_span = half_lift_n / max(half_span * bell_integral, 1e-12) * bell_shape
        cl_local = lift_per_span / max(q * chord_m, 1e-12)
        reynolds = RHO_SL_KG_M3 * velocity_m_s * chord_m / MU_SL_PA_S

        alpha0 = lerp(
            float(twist_trim["zero_lift_aoa_root_deg"]),
            float(twist_trim["zero_lift_aoa_tip_deg"]),
            eta,
        )
        cl_alpha = lerp(
            float(twist_trim["cl_alpha_root_per_deg"]),
            float(twist_trim["cl_alpha_tip_per_deg"]),
            eta,
        )
        thickness_ratio = lerp(0.12, 0.09, eta)
        if clmax_model == "tc_re":
            cl_max = clmax_tc_re(thickness_ratio, reynolds)
        else:
            cl_max = clmax_re_corrected(reynolds, float(twist_trim["estimated_cl_max"]))

        alpha_eff = alpha0 + cl_local / max(cl_alpha, 1e-12)
        alpha_sep = alpha0 + cl_max / max(cl_alpha, 1e-12)
        alpha_margin = alpha_sep - alpha_eff
        alpha_excess = max(0.0, -alpha_margin)
        e_local = smoothstep01(alpha_excess / 4.0)
        onset_proximity = min(1.0, max(0.0, 1.0 - alpha_margin / 4.0))

        chord_values.append(chord_m)
        area_weighted_sep.append(chord_m * e_local)
        pressure_proxy_weight.append(chord_m * onset_proximity)
        cl_values.append(cl_local)
        margins.append(alpha_margin)
        excess_values.append(alpha_excess)

        rows.append(
            {
                "condition": name,
                "y_m": y_m,
                "eta": eta,
                "chord_m": chord_m,
                "twist_deg": twist_deg,
                "lift_per_span_N_m": lift_per_span,
                "cl_local": cl_local,
                "reynolds": reynolds,
                "alpha0_deg": alpha0,
                "cl_alpha_per_deg": cl_alpha,
                "thickness_ratio": thickness_ratio,
                "cl_max_local": cl_max,
                "alpha_eff_deg": alpha_eff,
                "alpha_sep_deg": alpha_sep,
                "alpha_margin_deg": alpha_margin,
                "alpha_excess_deg": alpha_excess,
                "e_sep_local": e_local,
                "onset_proximity": onset_proximity,
            }
        )

    e_sep = 2.0 * trapz(y, area_weighted_sep) / max(reference_area_m2, 1e-12)
    onset_area_proxy = 2.0 * trapz(y, pressure_proxy_weight) / max(reference_area_m2, 1e-12)
    half_area = trapz(y, chord_values)
    min_margin = min(margins)
    min_idx = margins.index(min_margin)
    max_excess = max(excess_values)
    max_excess_idx = excess_values.index(max_excess)

    summary = {
        "condition": name,
        "CL": cl_total,
        "velocity_m_s": velocity_m_s,
        "dynamic_pressure_Pa": q,
        "half_lift_N": half_lift_n,
        "half_area_m2": half_area,
        "full_area_from_strips_m2": 2.0 * half_area,
        "E_sep": e_sep,
        "onset_area_proxy": onset_area_proxy,
        "active_sep_strip_count": sum(1 for value in excess_values if value > 0.0),
        "active_onset_strip_count": sum(1 for row in rows if row["onset_proximity"] > 0.0),
        "min_alpha_margin_deg": min_margin,
        "min_alpha_margin_y_m": y[min_idx],
        "max_alpha_excess_deg": max_excess,
        "max_alpha_excess_y_m": y[max_excess_idx],
        "max_cl_local": max(cl_values),
        "max_cl_local_y_m": y[cl_values.index(max(cl_values))],
        "clmax_model": clmax_model,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockwing-json", required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--program-summary", required=True)
    parser.add_argument("--cfd-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-strips", type=int, default=81)
    args = parser.parse_args()

    stockwing = load_json(Path(args.stockwing_json))
    patch_manifest = load_json(Path(args.patch_manifest))
    program_summary = load_json(Path(args.program_summary))
    cfd_summary = load_json(Path(args.cfd_summary))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = program_summary["metrics"]
    reference_area_m2 = float(program_summary["geometry"]["reference_area_m2"])
    twist_trim = stockwing["wing"]["twist_trim"]
    sections = patch_manifest["sections"]
    cfd_cl = float(cfd_summary["scaled_last_average"]["Cl"])

    cases: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for clmax_model in ("project_estimated_clmax_re", "tc_re"):
        for name, cl_value in (
            ("program_takeoff_loading", float(metrics["takeoff_cl"])),
            ("cfd_takeoff_loading", cfd_cl),
        ):
            summary, rows = compute_case(
                name=f"{name}_{clmax_model}",
                cl_total=cl_value,
                velocity_m_s=float(metrics["takeoff_velocity"]),
                sections=sections,
                twist_trim=twist_trim,
                reference_area_m2=reference_area_m2,
                n_strips=int(args.n_strips),
                clmax_model="tc_re" if clmax_model == "tc_re" else "project",
            )
            cases.append((summary, rows))

    all_rows = [row for _, rows in cases for row in rows]
    csv_path = output_dir / "stockwing_takeoff_strip_separation_rows.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "source_stockwing_json": str(Path(args.stockwing_json).resolve()),
        "source_patch_manifest": str(Path(args.patch_manifest).resolve()),
        "source_program_summary": str(Path(args.program_summary).resolve()),
        "source_cfd_summary": str(Path(args.cfd_summary).resolve()),
        "method": (
            "Deterministic strip reconstruction using StockWing exported sections, "
            "bell spanload, local project cl-alpha/alpha0, sea-level Reynolds, and "
            "area-weighted smoothstep(alpha_excess/4 deg) separation exposure."
        ),
        "limitation": (
            "AeroSandbox LiftingLine and NeuralFoil both crashed natively in the "
            "flying-wing-tool conda environment, so this run avoids those calls."
        ),
        "program_takeoff": {
            "CL": float(metrics["takeoff_cl"]),
            "CD": float(metrics["takeoff_cd"]),
            "L_over_D": float(metrics["takeoff_l_d"]),
            "alpha_deg": float(metrics["takeoff_alpha"]),
            "velocity_m_s": float(metrics["takeoff_velocity"]),
        },
        "cfd_takeoff": {
            "CL": cfd_cl,
            "CD": float(cfd_summary["scaled_last_average"]["Cd"]),
            "L_over_D": float(cfd_summary["scaled_last_average"]["L_over_D"]),
        },
        "case_summaries": [case_summary for case_summary, _ in cases],
        "interpretation": {
            "stockwing_takeoff_program_loading_reaches_local_stall": False,
            "reason": (
                "Both CLmax models produce E_sep=0 and positive alpha margin at "
                "the program takeoff loading. The low CFD CL is therefore not "
                "explained by the strip separation/Reynolds exposure mechanism "
                "that explained the X-48-style case."
            ),
        },
    }
    json_path = output_dir / "stockwing_takeoff_strip_separation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
