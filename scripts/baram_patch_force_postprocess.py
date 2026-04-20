from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baram_case_tools import average_last, find_monitor_file, infer_geometry_length_scale_to_m, normalize_case_dir, parse_force_dat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess split-patch BARAM/OpenFOAM force monitors and compute patch-level drag shares."
    )
    parser.add_argument("--case-root", required=True, help="Path to the solved case bundle or inner case directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and plots.")
    parser.add_argument("--case-name", help="Output prefix. Defaults to the case stem.")
    parser.add_argument("--average-count", type=int, default=20, help="Trailing iteration count to average.")
    return parser.parse_args()


def build_patch_summary(rows: list[dict[str, float]], average_count: int, force_scale: float) -> dict[str, float]:
    avg_count = max(1, min(average_count, len(rows)))
    total_drag = 2.0 * average_last(rows, "total_x_half", avg_count) * force_scale
    pressure_drag = 2.0 * average_last(rows, "pressure_x_half", avg_count) * force_scale
    viscous_drag = 2.0 * average_last(rows, "viscous_x_half", avg_count) * force_scale
    total_lift = 2.0 * average_last(rows, "total_z_half", avg_count) * force_scale
    return {
        "average_count": avg_count,
        "total_drag_N": total_drag,
        "total_drag_magnitude_N": abs(total_drag),
        "pressure_drag_N": pressure_drag,
        "pressure_drag_magnitude_N": abs(pressure_drag),
        "viscous_drag_N": viscous_drag,
        "viscous_drag_magnitude_N": abs(viscous_drag),
        "total_lift_N": total_lift,
        "pressure_drag_fraction": pressure_drag / total_drag if total_drag else 0.0,
        "viscous_drag_fraction": viscous_drag / total_drag if total_drag else 0.0,
    }


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root).resolve()
    case_dir = normalize_case_dir(case_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = args.case_name or case_root.stem

    geometry_scale = infer_geometry_length_scale_to_m(case_dir)
    force_scale = float(geometry_scale["area_scale_to_m2"])
    aggregate_force_path = find_monitor_file(case_dir, "force-mon-1_forces", "force.dat")
    aggregate_rows = parse_force_dat(aggregate_force_path)
    aggregate_summary = build_patch_summary(aggregate_rows, args.average_count, force_scale)

    patch_monitors = {}
    post_root = case_dir / "postProcessing"
    for child in sorted(post_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("force-mon-") or not name.endswith("_forces") or name == "force-mon-1_forces":
            continue
        patch_name = name[len("force-mon-") : -len("_forces")]
        force_path = find_monitor_file(case_dir, name, "force.dat")
        rows = parse_force_dat(force_path)
        patch_summary = build_patch_summary(rows, args.average_count, force_scale)
        patch_summary["source_force_file"] = str(force_path)
        patch_monitors[patch_name] = patch_summary

    total_drag = aggregate_summary["total_drag_N"]
    total_pressure_drag = aggregate_summary["pressure_drag_N"]
    total_drag_magnitude = aggregate_summary["total_drag_magnitude_N"]
    total_pressure_drag_magnitude = aggregate_summary["pressure_drag_magnitude_N"]
    for patch_name, patch_summary in patch_monitors.items():
        patch_summary["drag_share_of_total"] = patch_summary["total_drag_N"] / total_drag if total_drag else 0.0
        patch_summary["drag_share_of_total_magnitude"] = (
            patch_summary["total_drag_magnitude_N"] / total_drag_magnitude if total_drag_magnitude else 0.0
        )
        patch_summary["pressure_drag_share_of_total"] = (
            patch_summary["pressure_drag_N"] / total_pressure_drag if total_pressure_drag else 0.0
        )
        patch_summary["pressure_drag_share_of_total_magnitude"] = (
            patch_summary["pressure_drag_magnitude_N"] / total_pressure_drag_magnitude
            if total_pressure_drag_magnitude
            else 0.0
        )

    body_drag = sum(
        patch_monitors[name]["total_drag_N"]
        for name in ("centerbody", "junction")
        if name in patch_monitors
    )
    body_drag_magnitude = sum(
        patch_monitors[name]["total_drag_magnitude_N"]
        for name in ("centerbody", "junction")
        if name in patch_monitors
    )
    body_pressure_drag = sum(
        patch_monitors[name]["pressure_drag_N"]
        for name in ("centerbody", "junction")
        if name in patch_monitors
    )
    body_pressure_drag_magnitude = sum(
        patch_monitors[name]["pressure_drag_magnitude_N"]
        for name in ("centerbody", "junction")
        if name in patch_monitors
    )
    outer_drag = patch_monitors.get("outer_wing", {}).get("total_drag_N", 0.0)
    outer_drag_magnitude = patch_monitors.get("outer_wing", {}).get("total_drag_magnitude_N", 0.0)
    hypothesis_supported = body_drag_magnitude > outer_drag_magnitude

    hypothesis = {
        "tested_hypothesis": "Most of the excess drag is generated by the centerbody and wing-body junction rather than the outer wing.",
        "supported": hypothesis_supported,
        "body_plus_junction_drag_share": body_drag / total_drag if total_drag else 0.0,
        "body_plus_junction_drag_share_magnitude": (
            body_drag_magnitude / total_drag_magnitude if total_drag_magnitude else 0.0
        ),
        "body_plus_junction_pressure_drag_share": body_pressure_drag / total_pressure_drag if total_pressure_drag else 0.0,
        "body_plus_junction_pressure_drag_share_magnitude": (
            body_pressure_drag_magnitude / total_pressure_drag_magnitude if total_pressure_drag_magnitude else 0.0
        ),
        "body_plus_junction_drag_magnitude_N": body_drag_magnitude,
        "outer_wing_drag_magnitude_N": outer_drag_magnitude,
        "note": (
            "Supported when centerbody + junction drag exceeds outer-wing drag in the patch-resolved force monitors."
            if hypothesis_supported
            else "Not supported when outer-wing drag remains larger than centerbody + junction drag."
        ),
    }

    result = {
        "source_case": str(case_root),
        "source_case_dir": str(case_dir),
        "aggregate_force_file": str(aggregate_force_path),
        "normalization": {
            "symmetry_plane_present": True,
            "forces_doubled_to_estimate_full_aircraft": True,
            "geometry_scale": geometry_scale,
        },
        "aggregate_summary": aggregate_summary,
        "patch_summaries": patch_monitors,
        "hypothesis_evaluation": hypothesis,
    }

    summary_path = output_dir / f"{summary_name}_patch_force_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))

    if patch_monitors:
        patch_names = list(patch_monitors.keys())
        drag_values = [patch_monitors[name]["total_drag_N"] for name in patch_names]
        pressure_values = [patch_monitors[name]["pressure_drag_N"] for name in patch_names]
        viscous_values = [patch_monitors[name]["viscous_drag_N"] for name in patch_names]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        x_positions = range(len(patch_names))
        ax.bar(x_positions, pressure_values, label="Pressure drag", color="#d95f02")
        ax.bar(x_positions, viscous_values, bottom=pressure_values, label="Viscous drag", color="#1b9e77")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(patch_names)
        ax.set_ylabel("Drag [N, full-aircraft estimate]")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{summary_name}_patch_drag_breakdown.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(patch_names, [patch_monitors[name]["drag_share_of_total"] for name in patch_names], color="#4c78a8")
        ax.set_ylabel("Share of total drag")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{summary_name}_patch_drag_share.png", dpi=200)
        plt.close(fig)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
