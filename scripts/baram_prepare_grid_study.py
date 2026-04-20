from __future__ import annotations

import argparse
from pathlib import Path

from baram_grid_study_tools import DEFAULT_MESH_PRESETS, DEFAULT_PRESET_ORDER, dump_json, load_json, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a manifest for a BARAM grid-independence study.")
    parser.add_argument("--source-bm", required=True, help="Path to the source .bm bundle.")
    parser.add_argument(
        "--flow-template-case-root",
        required=True,
        help="Path to a solver-ready case root or bundle containing a reconstructed 0/ directory.",
    )
    parser.add_argument("--summary-json", required=True, help="Project summary JSON with cruise/takeoff metrics.")
    parser.add_argument("--study-root", required=True, help="Directory where the study workspace should live.")
    parser.add_argument("--name", required=True, help="Study name.")
    parser.add_argument("--map-from-case-root", help="Optional solved source case to map fields from before solving.")
    parser.add_argument(
        "--report-output-dir",
        help="Directory where postprocessed summaries and comparison tables should be written.",
    )
    parser.add_argument("--cruise-alpha", type=float, help="Override cruise angle of attack in degrees.")
    parser.add_argument("--cruise-velocity", type=float, help="Override cruise velocity in m/s.")
    parser.add_argument("--takeoff-alpha", type=float, help="Override takeoff angle of attack in degrees.")
    parser.add_argument("--takeoff-velocity", type=float, help="Override takeoff velocity in m/s.")
    parser.add_argument(
        "--solve-end-time",
        type=float,
        default=800.0,
        help="Recommended steady-solver endTime for each variant.",
    )
    parser.add_argument(
        "--write-interval",
        type=float,
        default=100.0,
        help="Recommended steady-solver writeInterval for each variant.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest with the same name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.summary_json).resolve())
    metrics = summary.get("metrics", {})
    if not metrics:
        raise KeyError("Summary JSON does not contain a 'metrics' block.")

    cruise_alpha = float(args.cruise_alpha if args.cruise_alpha is not None else metrics["cruise_alpha"])
    cruise_velocity = float(args.cruise_velocity if args.cruise_velocity is not None else metrics["cruise_velocity"])
    takeoff_alpha = float(args.takeoff_alpha if args.takeoff_alpha is not None else metrics["takeoff_alpha"])
    takeoff_velocity = float(args.takeoff_velocity if args.takeoff_velocity is not None else metrics["takeoff_velocity"])

    study_name = slugify(args.name)
    study_dir = Path(args.study_root).resolve() / study_name
    manifest_path = study_dir / "grid_study_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Pass --overwrite to replace it.")

    report_output_dir = (
        Path(args.report_output_dir).resolve()
        if args.report_output_dir
        else study_dir / "reports"
    )

    conditions = [
        {
            "name": "cruise",
            "alpha_deg": cruise_alpha,
            "velocity_m_s": cruise_velocity,
            "map_from_case_root": str(Path(args.map_from_case_root).resolve()) if args.map_from_case_root else None,
            "map_from_time": "latestTime",
        },
        {
            "name": "takeoff",
            "alpha_deg": takeoff_alpha,
            "velocity_m_s": takeoff_velocity,
            "map_from_case_root": str(Path(args.map_from_case_root).resolve()) if args.map_from_case_root else None,
            "map_from_time": "latestTime",
        },
    ]

    variants = []
    for condition in conditions:
        for mesh_preset in DEFAULT_PRESET_ORDER:
            variant_name = f"{condition['name']}_{mesh_preset}"
            variants.append(
                {
                    "name": variant_name,
                    "condition": condition["name"],
                    "mesh_preset": mesh_preset,
                    "workspace_root": str((study_dir / "variants" / variant_name).resolve()),
                    "bundle_name": Path(args.source_bm).name,
                    "status": "planned",
                }
            )

    manifest = {
        "study_name": study_name,
        "source_bm_bundle": str(Path(args.source_bm).resolve()),
        "flow_template_case_root": str(Path(args.flow_template_case_root).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "report_output_dir": str(report_output_dir),
        "recommended_solver_settings": {
            "end_time": float(args.solve_end_time),
            "write_interval": float(args.write_interval),
        },
        "notes": [
            "Prepared by baram_prepare_grid_study.py",
            "Run baram_run_grid_study.py later to prepare bundles, remesh, assemble flow cases, solve, and postprocess.",
            "The default mesh presets are intended as a pragmatic coarse/medium/fine ladder and may need adjustment if cell counts do not spread enough.",
            "The current template indicates that snappy layer addition is requested but may still add zero cells; treat boundary-layer convergence separately until that is fixed.",
        ],
        "mesh_presets": DEFAULT_MESH_PRESETS,
        "conditions": conditions,
        "variants": variants,
    }

    dump_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
