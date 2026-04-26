from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baram_case_tools import (
    average_last,
    compute_dynamic_pressure,
    extract_actual_area,
    extract_metrics,
    find_monitor_file,
    infer_geometry_length_scale_to_m,
    load_summary_json,
    normalize_case_dir,
    parse_coefficient_dat,
    parse_force_dat,
    project_force_components,
    read_force_axis_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a BARAM/OpenFOAM-style case bundle.")
    parser.add_argument("--case-root", required=True, help="Path to a .bf bundle or inner case directory.")
    parser.add_argument("--summary-json", help="Optional project summary JSON for cruise-point comparison.")
    parser.add_argument(
        "--condition-name",
        help="Optional condition selector when summary JSON contains multiple conditions (for example 'cruise' or 'takeoff').",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and plots.")
    parser.add_argument("--case-name", help="Case label for titles and output prefixes.")
    parser.add_argument("--reference-area", type=float, help="Override reference area in m^2.")
    parser.add_argument("--reference-density", type=float, default=1.225, help="Reference density in kg/m^3.")
    parser.add_argument("--reference-velocity", type=float, help="Override reference velocity in m/s.")
    parser.add_argument("--average-count", type=int, default=20, help="Number of trailing iterations to average.")
    parser.add_argument("--skip-plots", action="store_true", help="Write JSON only and skip Matplotlib plot generation.")
    return parser.parse_args()


def select_condition_metrics(metrics: dict[str, float], condition_name: str | None) -> tuple[str, dict[str, float]]:
    if condition_name:
        prefix = condition_name.strip().lower()
        required = [f"{prefix}_velocity", f"{prefix}_cd", f"{prefix}_cl", f"{prefix}_l_d"]
        missing = [key for key in required if key not in metrics]
        if missing:
            raise KeyError(f"Condition '{prefix}' is missing required metric keys: {missing}")
        return prefix, {
            "velocity": float(metrics[f"{prefix}_velocity"]),
            "cd": float(metrics[f"{prefix}_cd"]),
            "cl": float(metrics[f"{prefix}_cl"]),
            "l_d": float(metrics[f"{prefix}_l_d"]),
            "alpha_deg": float(metrics.get(f"{prefix}_alpha", 0.0)),
        }

    return "cruise", {
        "velocity": float(metrics["cruise_velocity"]),
        "cd": float(metrics["cruise_cd"]),
        "cl": float(metrics["cruise_cl"]),
        "l_d": float(metrics["cruise_l_d"]),
        "alpha_deg": float(metrics.get("cruise_alpha", 0.0)),
    }


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root).resolve()
    case_dir = normalize_case_dir(case_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = find_monitor_file(case_dir, "force-mon-1", "coefficient.dat")
    force_path = find_monitor_file(case_dir, "force-mon-1_forces", "force.dat")

    coeff_rows = parse_coefficient_dat(coeff_path)
    force_rows = parse_force_dat(force_path)
    if len(coeff_rows) != len(force_rows):
        raise ValueError(
            f"Monitor length mismatch: {len(coeff_rows)} coefficient rows vs {len(force_rows)} force rows."
        )

    metrics = None
    target_label = None
    target_metrics = None
    actual_area = args.reference_area
    reference_velocity = args.reference_velocity
    if args.summary_json:
        summary = load_summary_json(Path(args.summary_json))
        metrics = extract_metrics(summary)
        target_label, target_metrics = select_condition_metrics(metrics, args.condition_name)
        if actual_area is None:
            actual_area = extract_actual_area(summary)
        if reference_velocity is None:
            reference_velocity = float(target_metrics["velocity"])

    if actual_area is None or reference_velocity is None:
        raise ValueError(
            "Reference area and velocity are required. Supply --summary-json or set both overrides directly."
        )

    q = compute_dynamic_pressure(args.reference_density, reference_velocity)
    geometry_scale = infer_geometry_length_scale_to_m(case_dir)
    force_scale = float(geometry_scale["area_scale_to_m2"])
    axis_dirs = read_force_axis_dirs(case_dir)

    combined = []
    for coeff_row, force_row in zip(coeff_rows, force_rows):
        projected = project_force_components(force_row, axis_dirs["drag_dir"], axis_dirs["lift_dir"])
        drag_full = 2.0 * projected["total_drag_half"] * force_scale
        lift_full = 2.0 * projected["total_lift_half"] * force_scale
        pressure_drag_full = 2.0 * projected["pressure_drag_half"] * force_scale
        viscous_drag_full = 2.0 * projected["viscous_drag_half"] * force_scale
        pressure_lift_full = 2.0 * projected["pressure_lift_half"] * force_scale
        viscous_lift_full = 2.0 * projected["viscous_lift_half"] * force_scale
        combined.append(
            {
                "time": coeff_row["time"],
                "Cd_raw": coeff_row["Cd_raw"],
                "Cl_raw": coeff_row["Cl_raw"],
                "Cm_raw": coeff_row["Cm_raw"],
                "drag_full_N": drag_full,
                "lift_full_N": lift_full,
                "pressure_drag_full_N": pressure_drag_full,
                "viscous_drag_full_N": viscous_drag_full,
                "pressure_lift_full_N": pressure_lift_full,
                "viscous_lift_full_N": viscous_lift_full,
                "Cd_scaled": drag_full / (q * actual_area),
                "Cl_scaled": lift_full / (q * actual_area),
                "L_over_D": (lift_full / drag_full) if drag_full else float("inf"),
            }
        )

    avg_count = max(1, min(args.average_count, len(combined)))
    last = combined[-1]

    averaged = {
        "Cd": average_last(combined, "Cd_scaled", avg_count),
        "Cl": average_last(combined, "Cl_scaled", avg_count),
        "L_over_D": average_last(combined, "L_over_D", avg_count),
        "drag_full_N": average_last(combined, "drag_full_N", avg_count),
        "lift_full_N": average_last(combined, "lift_full_N", avg_count),
    }
    drag_breakdown = {
        "total_drag": average_last(combined, "drag_full_N", avg_count),
        "pressure_drag": average_last(combined, "pressure_drag_full_N", avg_count),
        "viscous_drag": average_last(combined, "viscous_drag_full_N", avg_count),
        "pressure_drag_fraction": average_last(combined, "pressure_drag_full_N", avg_count)
        / average_last(combined, "drag_full_N", avg_count),
        "viscous_drag_fraction": average_last(combined, "viscous_drag_full_N", avg_count)
        / average_last(combined, "drag_full_N", avg_count),
    }
    lift_breakdown = {
        "total_lift": average_last(combined, "lift_full_N", avg_count),
        "pressure_lift": average_last(combined, "pressure_lift_full_N", avg_count),
        "viscous_lift": average_last(combined, "viscous_lift_full_N", avg_count),
    }

    result = {
        "source_case": str(case_root),
        "source_case_dir": str(case_dir),
        "source_force_file": str(force_path),
        "source_coefficient_file": str(coeff_path),
        "normalization": {
            "symmetry_plane_present": True,
            "forces_doubled_to_estimate_full_aircraft": True,
            "reference_area_m2": actual_area,
            "reference_density_kg_m3": args.reference_density,
            "reference_velocity_m_s": reference_velocity,
            "reference_dynamic_pressure_Pa": q,
            "drag_dir": axis_dirs["drag_dir"],
            "lift_dir": axis_dirs["lift_dir"],
            "geometry_scale": geometry_scale,
            "average_count": avg_count,
        },
        "raw_last": {
            "time": last["time"],
            "Cd_raw": last["Cd_raw"],
            "Cl_raw": last["Cl_raw"],
            "Cm_raw": last["Cm_raw"],
        },
        "scaled_last_average": averaged,
        "drag_breakdown_last_average_full_aircraft_N": drag_breakdown,
        "lift_breakdown_last_average_full_aircraft_N": lift_breakdown,
    }

    if metrics is not None:
        result["comparison_target"] = {
            "condition": target_label,
            "cd": target_metrics["cd"],
            "cl": target_metrics["cl"],
            "l_d": target_metrics["l_d"],
            "alpha_deg": target_metrics["alpha_deg"],
            "velocity_m_s": target_metrics["velocity"],
        }
        result["comparison_delta"] = {
            "delta_Cd": averaged["Cd"] - float(target_metrics["cd"]),
            "delta_Cl": averaged["Cl"] - float(target_metrics["cl"]),
            "delta_L_over_D": averaged["L_over_D"] - float(target_metrics["l_d"]),
        }
        result["interpretation"] = {
            "drag_mismatch_is_pressure_dominated": drag_breakdown["pressure_drag"] > drag_breakdown["viscous_drag"],
            "note": (
                f"Pressure drag alone exceeds the low-order {target_label} drag when the comparison target is supplied."
                if drag_breakdown["pressure_drag"] > (float(target_metrics["cd"]) * q * actual_area)
                else f"Pressure drag does not exceed the low-order {target_label} drag in this normalization."
            ),
        }

    summary_name = args.case_name or case_root.stem
    summary_path = output_dir / f"{summary_name}_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    if args.skip_plots:
        print(f"Wrote summary: {summary_path}")
        return

    times = [row["time"] for row in combined]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times, [row["Cl_scaled"] for row in combined], label="BARAM Cl", color="tab:blue")
    axes[0].plot(times, [row["Cd_scaled"] for row in combined], label="BARAM Cd", color="tab:red")
    if target_metrics is not None:
        target_x = [min(times), max(times)]
        axes[0].plot(target_x, [float(target_metrics["cl"])] * 2, linestyle="--", color="tab:blue", alpha=0.6, label="Target Cl")
        axes[0].plot(target_x, [float(target_metrics["cd"])] * 2, linestyle="--", color="tab:red", alpha=0.6, label="Target Cd")
    axes[0].set_ylabel("Coefficient")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(times, [row["L_over_D"] for row in combined], label="BARAM L/D", color="tab:green")
    if target_metrics is not None:
        target_x = [min(times), max(times)]
        axes[1].plot(target_x, [float(target_metrics["l_d"])] * 2, linestyle="--", color="tab:green", alpha=0.6, label="Target L/D")
    axes[1].set_xlabel("Iteration / monitor time")
    axes[1].set_ylabel("L/D")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{summary_name}_convergence.png", dpi=200)
    plt.close(fig)

    if metrics is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        labels = ["Cl", "Cd", "L/D"]
        baram_values = [averaged["Cl"], averaged["Cd"], averaged["L_over_D"]]
        target_values = [
            float(target_metrics["cl"]),
            float(target_metrics["cd"]),
            float(target_metrics["l_d"]),
        ]
        x_positions = list(range(len(labels)))
        width = 0.35
        ax.bar([x - width / 2 for x in x_positions], baram_values, width=width, label="BARAM", color="#1f77b4")
        ax.bar([x + width / 2 for x in x_positions], target_values, width=width, label="Target", color="#ff7f0e")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{summary_name}_comparison.png", dpi=200)
        plt.close(fig)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
