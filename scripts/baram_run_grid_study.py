from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from baram_case_tools import discover_baram_openfoam_root, normalize_case_dir, run_openfoam_utility
from baram_grid_study_tools import (
    apply_mesh_preset,
    copy_bundle,
    dump_json,
    load_json,
    patch_flow_condition,
)


SAFE_MPI_BY_PRESET: dict[str, dict[str, int]] = {
    "fine": {
        "mesh": 8,
        "solve": 8,
    },
    "extra_fine": {
        "mesh": 12,
        "solve": 12,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, run, and postprocess a BARAM grid-independence study.")
    parser.add_argument("--manifest", required=True, help="Path to a grid_study_manifest.json file.")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant name to run. Repeat to select a subset. Defaults to all variants.",
    )
    parser.add_argument("--prepare", action="store_true", help="Clone source bundles and patch their mesh dictionaries.")
    parser.add_argument("--mesh", action="store_true", help="Run the mesh workflow for each selected variant.")
    parser.add_argument("--assemble-flow", action="store_true", help="Promote remeshed polyMesh and patch flow conditions.")
    parser.add_argument("--solve", action="store_true", help="Run mapFields/decomposePar/solver for each selected variant.")
    parser.add_argument("--postprocess", action="store_true", help="Postprocess each selected variant.")
    parser.add_argument("--compare", action="store_true", help="Run the grid-study comparison script after postprocessing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared variant workspaces.")
    parser.add_argument("--reuse-existing", action="store_true", help="Keep existing prepared bundles and only run later stages.")
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--np", type=int, help="MPI process count override.")
    parser.add_argument("--timeout", type=int, default=18000, help="Solver timeout in seconds.")
    parser.add_argument("--set-end-time", type=float, help="Override solver endTime.")
    parser.add_argument("--set-write-interval", type=float, help="Override solver writeInterval.")
    parser.add_argument("--preview", action="store_true", help="Print planned actions without running them.")
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path, preview: bool) -> subprocess.CompletedProcess[str] | None:
    print("[grid-study] command:", " ".join(command))
    if preview:
        return None
    completed = subprocess.run(command, cwd=str(cwd), check=False, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def select_variants(manifest: dict[str, object], names: list[str] | None) -> list[dict[str, object]]:
    variants = manifest.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Manifest 'variants' field is not a list.")

    records = [item for item in variants if isinstance(item, dict)]
    if not names:
        return records

    wanted = set(names)
    selected = [item for item in records if str(item.get("name")) in wanted]
    found = {str(item.get("name")) for item in selected}
    missing = wanted - found
    if missing:
        raise KeyError(f"Variant(s) not found in manifest: {', '.join(sorted(missing))}")
    return selected


def lookup_condition(manifest: dict[str, object], name: str) -> dict[str, object]:
    conditions = manifest.get("conditions", [])
    if not isinstance(conditions, list):
        raise ValueError("Manifest 'conditions' field is not a list.")
    for item in conditions:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise KeyError(f"Condition not found in manifest: {name}")


def resolve_stage_np(variant: dict[str, object], *, stage: str, override_np: int | None) -> int | None:
    if override_np is not None:
        return override_np

    preset_name = str(variant.get("mesh_preset", ""))
    preset_limits = SAFE_MPI_BY_PRESET.get(preset_name)
    if not preset_limits:
        return None
    return preset_limits.get(stage)


def patch_decompose_subdomains(case_dir: Path, np: int) -> None:
    decompose_path = case_dir / "system" / "decomposeParDict"
    text = decompose_path.read_text()
    updated, count = re.subn(
        r"(^\s*numberOfSubdomains\s+)\d+(\s*;)",
        rf"\g<1>{np}\2",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError(f"Could not patch numberOfSubdomains in {decompose_path}")
    decompose_path.write_text(updated)


def prepare_variant(
    manifest: dict[str, object],
    variant: dict[str, object],
    *,
    overwrite: bool,
    reuse_existing: bool,
    preview: bool,
) -> None:
    source_bm_bundle = Path(str(manifest["source_bm_bundle"])).resolve()
    mesh_presets = manifest.get("mesh_presets", {})
    preset_name = str(variant["mesh_preset"])
    if preset_name not in mesh_presets:
        raise KeyError(f"Mesh preset '{preset_name}' not found in manifest.")
    preset = mesh_presets[preset_name]
    if not isinstance(preset, dict):
        raise ValueError(f"Mesh preset '{preset_name}' is not a dictionary.")

    workspace_root = Path(str(variant["workspace_root"])).resolve()
    bundle_root = workspace_root / str(variant["bundle_name"])
    expected_block_mesh = bundle_root / "case" / "system" / "blockMeshDict"

    print(f"[grid-study] prepare variant={variant['name']}")
    if preview:
        print(f"  source={source_bm_bundle}")
        print(f"  destination={bundle_root}")
        case_dir = bundle_root / "case"
    else:
        if bundle_root.exists():
            if reuse_existing:
                if not expected_block_mesh.exists():
                    copy_bundle(source_bm_bundle, bundle_root, overwrite=True)
            else:
                copy_bundle(source_bm_bundle, bundle_root, overwrite=overwrite)
        else:
            copy_bundle(source_bm_bundle, bundle_root, overwrite=False)

        case_dir = normalize_case_dir(bundle_root)
        if not reuse_existing or not (workspace_root / "mesh_recipe.json").exists():
            mesh_summary = apply_mesh_preset(case_dir, preset)
            dump_json(workspace_root / "mesh_recipe.json", mesh_summary)
            variant["mesh_recipe"] = str((workspace_root / "mesh_recipe.json").resolve())

    variant["bundle_root"] = str(bundle_root)
    variant["case_dir"] = str(case_dir)
    variant["status"] = "prepared"


def run_mesh_stage(
    variant: dict[str, object],
    *,
    baram_openfoam_root: str | None,
    np: int | None,
    preview: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    stage_np = resolve_stage_np(variant, stage="mesh", override_np=np)
    command = [sys.executable, str(script_dir / "baram_run_mesh_case.py"), "--case-root", str(variant["bundle_root"]), "--clean"]
    if baram_openfoam_root:
        command.extend(["--baram-openfoam-root", baram_openfoam_root])
    if stage_np is not None:
        command.extend(["--np", str(stage_np)])
    run_command(command, cwd=script_dir, preview=preview)
    variant["mesh_status"] = "mesh_ok" if not preview else "mesh_planned"
    variant["mesh_summary"] = str(Path(str(variant["case_dir"])) / "mesh_automation_summary.json")
    if stage_np is not None:
        variant["mesh_np"] = stage_np


def run_assemble_flow_stage(
    manifest: dict[str, object],
    variant: dict[str, object],
    *,
    preview: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    flow_template_case_root = str(manifest["flow_template_case_root"])
    command = [
        sys.executable,
        str(script_dir / "baram_build_flow_case_from_mesh.py"),
        "--mesh-case-root",
        str(variant["bundle_root"]),
        "--flow-template-case-root",
        flow_template_case_root,
        "--output-case-root",
        str(variant["bundle_root"]),
    ]
    run_command(command, cwd=script_dir, preview=preview)

    condition = lookup_condition(manifest, str(variant["condition"]))
    if not preview:
        flow_patch_summary = patch_flow_condition(
            Path(str(variant["case_dir"])),
            velocity=float(condition["velocity_m_s"]),
            alpha_deg=float(condition["alpha_deg"]),
        )
        dump_json(Path(str(variant["case_dir"])) / "grid_study_flow_condition.json", flow_patch_summary)
    variant["flow_status"] = "assembled" if not preview else "assemble_planned"


def decompose_time_zero(case_dir: Path, *, openfoam_root: Path, np: int | None, preview: bool) -> None:
    if np is not None and not preview:
        patch_decompose_subdomains(case_dir, np)
    command = ["-case", str(case_dir), "-force", "-time", "0"]
    print("[grid-study] command:", f"decomposePar {' '.join(command)}")
    if preview:
        return
    result = run_openfoam_utility(openfoam_root, "decomposePar", command, cwd=case_dir)
    (case_dir / "grid_study_decomposePar_0_stdout.log").write_text(result.stdout)
    (case_dir / "grid_study_decomposePar_0_stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"decomposePar -time 0 failed for {case_dir}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_solve_stage(
    manifest: dict[str, object],
    variant: dict[str, object],
    *,
    openfoam_root: Path,
    np: int | None,
    timeout: int,
    set_end_time: float | None,
    set_write_interval: float | None,
    preview: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    condition = lookup_condition(manifest, str(variant["condition"]))
    case_dir = Path(str(variant["case_dir"]))

    map_from_case_root = condition.get("map_from_case_root")
    if map_from_case_root:
        map_command = [
            sys.executable,
            str(script_dir / "baram_map_fields.py"),
            "--source-case-root",
            str(map_from_case_root),
            "--target-case-root",
            str(variant["bundle_root"]),
            "--source-time",
            str(condition.get("map_from_time", "latestTime")),
        ]
        run_command(map_command, cwd=script_dir, preview=preview)

    if not preview:
        flow_patch_summary = patch_flow_condition(
            case_dir,
            velocity=float(condition["velocity_m_s"]),
            alpha_deg=float(condition["alpha_deg"]),
        )
        dump_json(case_dir / "grid_study_flow_condition.json", flow_patch_summary)

    stage_np = resolve_stage_np(variant, stage="solve", override_np=np)
    decompose_time_zero(case_dir, openfoam_root=openfoam_root, np=stage_np, preview=preview)

    recommended = manifest.get("recommended_solver_settings", {})
    end_time = float(set_end_time if set_end_time is not None else recommended.get("end_time", 800.0))
    write_interval = float(
        set_write_interval if set_write_interval is not None else recommended.get("write_interval", 100.0)
    )
    run_case_command = [
        sys.executable,
        str(script_dir / "baram_run_case.py"),
        "--case-root",
        str(variant["bundle_root"]),
        "--set-start-from",
        "startTime",
        "--set-end-time",
        str(end_time),
        "--set-write-interval",
        str(write_interval),
        "--timeout",
        str(timeout),
    ]
    if openfoam_root:
        run_case_command.extend(["--baram-openfoam-root", str(openfoam_root)])
    if stage_np is not None:
        run_case_command.extend(["--np", str(stage_np)])
    run_command(run_case_command, cwd=script_dir, preview=preview)
    variant["solve_status"] = "solve_ok" if not preview else "solve_planned"
    if stage_np is not None:
        variant["solve_np"] = stage_np


def run_postprocess_stage(manifest: dict[str, object], variant: dict[str, object], *, preview: bool) -> None:
    script_dir = Path(__file__).resolve().parent
    report_output_dir = Path(str(manifest["report_output_dir"])).resolve()
    command = [
        sys.executable,
        str(script_dir / "baram_postprocess.py"),
        "--case-root",
        str(variant["bundle_root"]),
        "--summary-json",
        str(manifest["summary_json"]),
        "--output-dir",
        str(report_output_dir),
        "--case-name",
        str(variant["name"]),
    ]
    run_command(command, cwd=script_dir, preview=preview)
    variant["postprocess_summary"] = str(report_output_dir / f"{variant['name']}_summary.json")
    variant["postprocess_status"] = "postprocess_ok" if not preview else "postprocess_planned"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    selected = select_variants(manifest, args.variants)

    if not any((args.prepare, args.mesh, args.assemble_flow, args.solve, args.postprocess, args.compare)):
        args.prepare = True
        args.mesh = True
        args.assemble_flow = True
        args.solve = True
        args.postprocess = True
        args.compare = True

    openfoam_root = None
    if args.mesh or args.solve:
        if args.preview:
            openfoam_root = Path(args.baram_openfoam_root).resolve() if args.baram_openfoam_root else Path("preview_openfoam_root")
        else:
            openfoam_root = discover_baram_openfoam_root(args.baram_openfoam_root)

    for variant in selected:
        if args.prepare:
            prepare_variant(
                manifest,
                variant,
                overwrite=args.overwrite,
                reuse_existing=args.reuse_existing,
                preview=args.preview,
            )
        else:
            workspace_root = Path(str(variant["workspace_root"])).resolve()
            bundle_root = workspace_root / str(variant["bundle_name"])
            variant["bundle_root"] = str(bundle_root)
            variant["case_dir"] = str(normalize_case_dir(bundle_root))

        if args.mesh:
            run_mesh_stage(
                variant,
                baram_openfoam_root=str(openfoam_root) if openfoam_root else args.baram_openfoam_root,
                np=args.np,
                preview=args.preview,
            )
        if args.assemble_flow:
            run_assemble_flow_stage(manifest, variant, preview=args.preview)
        if args.solve:
            if openfoam_root is None and not args.preview:
                raise FileNotFoundError("BARAM OpenFOAM root is required for the solve stage.")
            run_solve_stage(
                manifest,
                variant,
                openfoam_root=openfoam_root,
                np=args.np,
                timeout=args.timeout,
                set_end_time=args.set_end_time,
                set_write_interval=args.set_write_interval,
                preview=args.preview,
            )
        if args.postprocess:
            run_postprocess_stage(manifest, variant, preview=args.preview)

    if not args.preview:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.compare:
        script_dir = Path(__file__).resolve().parent
        compare_command = [sys.executable, str(script_dir / "baram_compare_grid_study.py"), "--manifest", str(manifest_path)]
        run_command(compare_command, cwd=script_dir, preview=args.preview)


if __name__ == "__main__":
    main()
