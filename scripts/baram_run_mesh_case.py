from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from baram_case_tools import (
    build_openfoam_env,
    count_processor_dirs,
    discover_baram_openfoam_root,
    list_numeric_time_dirs,
    normalize_case_dir,
    parse_boundary_patch_nfaces,
    resolve_solver_path,
    run_openfoam_utility,
)


STAGE_FLAGS: dict[str, tuple[bool, bool, bool]] = {
    "castellated": (True, False, False),
    "snap": (False, True, False),
    "layers": (False, False, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a headless BARAM/OpenFOAM mesh workflow from a .bm bundle.")
    parser.add_argument("--case-root", required=True, help="Path to a .bm bundle or inner case directory.")
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--np", type=int, help="MPI process count. Defaults to decomposeParDict or existing processors.")
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["castellated", "snap", "layers"],
        help="Stage(s) to run. Defaults to all three in order.",
    )
    parser.add_argument("--skip-blockmesh", action="store_true", help="Skip blockMesh.")
    parser.add_argument("--skip-decompose", action="store_true", help="Skip decomposePar.")
    parser.add_argument("--clean", action="store_true", help="Delete processor directories before meshing.")
    parser.add_argument(
        "--patch-name",
        default="IntendedValidation2CFDmesh_surface",
        help="Surface patch name to track in stage summaries.",
    )
    parser.add_argument("--compare-case-root", help="Optional baseline .bm bundle or inner case directory to compare against.")
    parser.add_argument(
        "--log-prefix",
        default="mesh_automation",
        help="Prefix for stage log files written into the case directory.",
    )
    parser.add_argument("--preview", action="store_true", help="Print planned actions without running them.")
    return parser.parse_args()


def patch_dictionary_entry(openfoam_root: Path, case_dir: Path, dict_relpath: str, entry: str, value: str) -> None:
    result = run_openfoam_utility(
        openfoam_root,
        "foamDictionary",
        [str(case_dir / dict_relpath), "-entry", entry, "-set", value],
        cwd=case_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"foamDictionary failed for {entry}={value}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def patch_stage_start_time(openfoam_root: Path, case_dir: Path, stage: str) -> None:
    if stage == "castellated":
        patch_dictionary_entry(openfoam_root, case_dir, "system/controlDict", "startFrom", "startTime")
        patch_dictionary_entry(openfoam_root, case_dir, "system/controlDict", "startTime", "0")
        return

    patch_dictionary_entry(openfoam_root, case_dir, "system/controlDict", "startFrom", "latestTime")


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


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> int:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    return completed.returncode


def find_stage_boundary(case_dir: Path, stage_time: str, processor_index: int = 0) -> Path:
    return case_dir / f"processor{processor_index}" / stage_time / "polyMesh" / "boundary"


def try_parse_boundary_patch_nfaces(boundary_path: Path, patch_name: str) -> int | None:
    try:
        return parse_boundary_patch_nfaces(boundary_path, patch_name)
    except (FileNotFoundError, ValueError):
        return None


def resolve_latest_mesh_time(case_dir: Path, *, parallel: bool) -> tuple[str, Path] | tuple[None, None]:
    if parallel:
        time_dirs = list_numeric_time_dirs(case_dir / "processor0")
        if not time_dirs:
            return None, None
        latest_time = time_dirs[-1].name
        return latest_time, find_stage_boundary(case_dir, latest_time)

    time_dirs = [path for path in list_numeric_time_dirs(case_dir) if path.name != "0"]
    if not time_dirs:
        return None, None
    latest_time = time_dirs[-1].name
    return latest_time, case_dir / latest_time / "polyMesh" / "boundary"


def main() -> None:
    args = parse_args()
    case_dir = normalize_case_dir(Path(args.case_root))
    openfoam_root = discover_baram_openfoam_root(args.baram_openfoam_root)
    env = build_openfoam_env(openfoam_root)

    stages = args.stages or ["castellated", "snap", "layers"]
    np = args.np or count_processor_dirs(case_dir) or 16
    parallel = np >= 2

    compare_case_dir = normalize_case_dir(Path(args.compare_case_root)) if args.compare_case_root else None
    summary: dict[str, object] = {
        "case_dir": str(case_dir),
        "np": np,
        "parallel": parallel,
        "stages": [],
    }

    planned = {
        "blockMesh": not args.skip_blockmesh,
        "decomposePar": parallel and not args.skip_decompose,
        "clean": args.clean,
        "parallel": parallel,
        "stages": stages,
    }
    print(json.dumps(planned, indent=2))
    if args.preview:
        return

    if args.clean:
        for child in case_dir.iterdir():
            if child.is_dir() and child.name.startswith("processor"):
                shutil.rmtree(child)

    if not args.skip_blockmesh:
        blockmesh = [str(resolve_solver_path(openfoam_root, "blockMesh")), "-case", str(case_dir)]
        code = run_command(
            blockmesh,
            cwd=case_dir,
            env=env,
            stdout_path=case_dir / f"{args.log_prefix}_blockMesh_stdout.log",
            stderr_path=case_dir / f"{args.log_prefix}_blockMesh_stderr.log",
        )
        if code != 0:
            raise SystemExit(code)

    if parallel and not args.skip_decompose:
        patch_decompose_subdomains(case_dir, np)
        decompose = [str(resolve_solver_path(openfoam_root, "decomposePar")), "-case", str(case_dir), "-force"]
        code = run_command(
            decompose,
            cwd=case_dir,
            env=env,
            stdout_path=case_dir / f"{args.log_prefix}_decomposePar_stdout.log",
            stderr_path=case_dir / f"{args.log_prefix}_decomposePar_stderr.log",
        )
        if code != 0:
            raise SystemExit(code)

    snappy_path = resolve_solver_path(openfoam_root, "snappyHexMesh")

    for stage in stages:
        cast_flag, snap_flag, layer_flag = STAGE_FLAGS[stage]
        patch_stage_start_time(openfoam_root, case_dir, stage)
        patch_dictionary_entry(openfoam_root, case_dir, "system/snappyHexMeshDict", "castellatedMesh", str(cast_flag).lower())
        patch_dictionary_entry(openfoam_root, case_dir, "system/snappyHexMeshDict", "snap", str(snap_flag).lower())
        patch_dictionary_entry(openfoam_root, case_dir, "system/snappyHexMeshDict", "addLayers", str(layer_flag).lower())

        if parallel:
            command = ["mpiexec", "-n", str(np), str(snappy_path), "-parallel", "-case", str(case_dir)]
        else:
            command = [str(snappy_path), "-case", str(case_dir)]
        stdout_path = case_dir / f"{args.log_prefix}_{stage}_stdout.log"
        stderr_path = case_dir / f"{args.log_prefix}_{stage}_stderr.log"
        code = run_command(command, cwd=case_dir, env=env, stdout_path=stdout_path, stderr_path=stderr_path)

        stage_record: dict[str, object] = {
            "stage": stage,
            "returncode": code,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

        latest_time, boundary_path = resolve_latest_mesh_time(case_dir, parallel=parallel)
        if latest_time is not None and boundary_path is not None:
            stage_record["time"] = latest_time
            patch_nfaces = try_parse_boundary_patch_nfaces(boundary_path, args.patch_name)
            if parallel:
                stage_record["processor0_patch_nFaces"] = patch_nfaces
            else:
                stage_record["root_patch_nFaces"] = patch_nfaces

            if compare_case_dir is not None:
                compare_parallel = count_processor_dirs(compare_case_dir) >= 2
                _, compare_boundary = resolve_latest_mesh_time(compare_case_dir, parallel=compare_parallel)
                compare_nfaces = try_parse_boundary_patch_nfaces(compare_boundary, args.patch_name) if compare_boundary else None
                if parallel:
                    stage_record["baseline_processor0_patch_nFaces"] = compare_nfaces
                    if patch_nfaces is not None and compare_nfaces is not None:
                        stage_record["matches_baseline_processor0_nFaces"] = (patch_nfaces == compare_nfaces)
                else:
                    stage_record["baseline_root_patch_nFaces"] = compare_nfaces
                    if patch_nfaces is not None and compare_nfaces is not None:
                        stage_record["matches_baseline_root_nFaces"] = (patch_nfaces == compare_nfaces)

        summary["stages"].append(stage_record)
        if code != 0:
            break

    summary_path = case_dir / f"{args.log_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote mesh summary: {summary_path}")
    failed_stage = next((stage for stage in summary["stages"] if int(stage.get("returncode", 0)) != 0), None)
    if failed_stage is not None:
        raise SystemExit(int(failed_stage["returncode"]))


if __name__ == "__main__":
    main()
