from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from baram_case_tools import (
    build_openfoam_env,
    count_processor_dirs,
    discover_baram_openfoam_root,
    normalize_case_dir,
    read_control_dict_application,
    read_decompose_subdomains,
    resolve_solver_path,
    run_openfoam_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BARAM/OpenFOAM case from PowerShell.")
    parser.add_argument("--case-root", required=True, help="Path to a .bf bundle or inner case directory.")
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--solver", help="Override solver name. Defaults to controlDict application.")
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        help="Force parallel execution.",
    )
    parser.add_argument(
        "--serial",
        dest="parallel",
        action="store_false",
        help="Force serial execution.",
    )
    parser.set_defaults(parallel=None)
    parser.add_argument("--np", type=int, help="MPI process count. Defaults to processor dirs or decomposeParDict.")
    parser.add_argument("--dry-run", action="store_true", help="Use the solver's non-writing dry-run mode.")
    parser.add_argument("--post-process", action="store_true", help="Run the solver in postProcess mode.")
    parser.add_argument("--timeout", type=int, help="Optional timeout in seconds.")
    parser.add_argument("--set-start-from", help="Patch controlDict startFrom before running.")
    parser.add_argument("--set-end-time", type=float, help="Patch controlDict endTime before running.")
    parser.add_argument("--set-write-interval", type=float, help="Patch controlDict writeInterval before running.")
    parser.add_argument("--set-application", help="Patch controlDict application before running.")
    parser.add_argument(
        "--log-prefix",
        default="automation",
        help="Prefix for stdout/stderr log files written into the case directory.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the resolved command and exit without launching.",
    )
    return parser.parse_args()


def patch_control_dict(openfoam_root: Path, case_dir: Path, args: argparse.Namespace) -> list[str]:
    control_dict = case_dir / "system" / "controlDict"
    patches: list[tuple[str, str]] = []
    if args.set_start_from is not None:
        patches.append(("startFrom", args.set_start_from))
    if args.set_end_time is not None:
        patches.append(("endTime", str(args.set_end_time)))
    if args.set_write_interval is not None:
        patches.append(("writeInterval", str(args.set_write_interval)))
    if args.set_application is not None:
        patches.append(("application", args.set_application))

    applied: list[str] = []
    for entry, value in patches:
        result = run_openfoam_utility(
            openfoam_root,
            "foamDictionary",
            [str(control_dict), "-entry", entry, "-set", value],
            cwd=case_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"foamDictionary failed for {entry}={value}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        applied.append(f"{entry}={value}")
    return applied


def infer_parallel(case_dir: Path, args: argparse.Namespace) -> tuple[bool, int]:
    processor_count = count_processor_dirs(case_dir)
    decompose_count = read_decompose_subdomains(case_dir / "system" / "decomposeParDict")

    default_parallel = processor_count > 0 or (decompose_count is not None and decompose_count > 1)
    parallel = default_parallel if args.parallel is None else args.parallel
    if not parallel:
        return False, 1

    np = args.np or processor_count or decompose_count or 1
    if np < 2:
        raise ValueError("Parallel execution requested, but no processor count >= 2 could be inferred.")
    return True, np


def build_solver_command(
    case_dir: Path,
    openfoam_root: Path,
    solver_name: str,
    *,
    parallel: bool,
    np: int,
    dry_run: bool,
    post_process: bool,
) -> list[str]:
    solver_path = resolve_solver_path(openfoam_root, solver_name)
    command: list[str]
    if parallel:
        command = ["mpiexec", "-n", str(np), str(solver_path), "-parallel"]
    else:
        command = [str(solver_path)]
    if dry_run:
        command.append("-dry-run")
    if post_process:
        command.append("-postProcess")
    command.extend(["-case", str(case_dir)])
    return command


def main() -> None:
    args = parse_args()
    case_dir = normalize_case_dir(Path(args.case_root))
    openfoam_root = discover_baram_openfoam_root(args.baram_openfoam_root)
    applied_patches = patch_control_dict(openfoam_root, case_dir, args)

    control_dict = case_dir / "system" / "controlDict"
    solver_name = args.solver or read_control_dict_application(control_dict)
    parallel, np = infer_parallel(case_dir, args)
    command = build_solver_command(
        case_dir,
        openfoam_root,
        solver_name,
        parallel=parallel,
        np=np,
        dry_run=args.dry_run,
        post_process=args.post_process,
    )

    print(f"[run] case: {case_dir}")
    print(f"[run] solver: {solver_name}")
    print(f"[run] parallel: {parallel}")
    if parallel:
        print(f"[run] np: {np}")
    if applied_patches:
        print(f"[run] patched: {', '.join(applied_patches)}")
    print(f"[run] command: {shlex.join(command)}")
    if args.preview:
        return

    stdout_path = case_dir / f"{args.log_prefix}_stdout.log"
    stderr_path = case_dir / f"{args.log_prefix}_stderr.log"
    env = build_openfoam_env(openfoam_root)

    completed = subprocess.run(
        command,
        cwd=str(case_dir),
        env=env,
        check=False,
        timeout=args.timeout,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)

    print(f"[run] stdout: {stdout_path}")
    print(f"[run] stderr: {stderr_path}")
    print(f"[run] returncode: {completed.returncode}")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
