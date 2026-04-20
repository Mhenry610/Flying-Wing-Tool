from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a solver-ready case from a remeshed .bm bundle and run a flow startup."
    )
    parser.add_argument("--mesh-case-root", required=True, help="Path to the source .bm bundle or inner case directory.")
    parser.add_argument(
        "--flow-template-case-root",
        required=True,
        help="Path to the source .bf bundle or inner case directory. A reconstructed root 0/ is required.",
    )
    parser.add_argument("--output-case-root", required=True, help="Destination .bm bundle or inner case directory.")
    parser.add_argument("--mesh-time", default="latest", help="Mesh time to promote. Use 'latest' or a numeric time.")
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--map-from-case-root", help="Optional source case to map fields from before solving.")
    parser.add_argument("--map-from-time", default="latestTime", help="Source time for mapFields.")
    parser.add_argument(
        "--map-patch",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="Patch mapping entry in target=source form. Repeat for multiple patches.",
    )
    parser.add_argument(
        "--map-cutting-patch",
        action="append",
        default=[],
        metavar="PATCH",
        help="Patch name to include in cuttingPatches for mapFields. Repeat for multiple patches.",
    )
    parser.add_argument("--end-time", type=float, default=1.0, help="Flow endTime for the startup run.")
    parser.add_argument("--write-interval", type=float, default=1.0, help="Flow writeInterval for the startup run.")
    parser.add_argument("--timeout", type=int, default=300, help="Solver timeout in seconds.")
    parser.add_argument("--preview", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    mesh_case_root = str(Path(args.mesh_case_root).resolve())
    flow_template_case_root = str(Path(args.flow_template_case_root).resolve())
    output_case_root = Path(args.output_case_root).resolve()

    build_command = [
        sys.executable,
        str(script_dir / "baram_build_flow_case_from_mesh.py"),
        "--mesh-case-root",
        mesh_case_root,
        "--flow-template-case-root",
        flow_template_case_root,
        "--output-case-root",
        str(output_case_root),
        "--mesh-time",
        args.mesh_time,
    ]
    if args.baram_openfoam_root:
        build_command.extend(["--baram-openfoam-root", args.baram_openfoam_root])
    run_case_root = str(output_case_root)

    map_command = None
    if args.map_from_case_root:
        map_command = [
            sys.executable,
            str(script_dir / "baram_map_fields.py"),
            "--source-case-root",
            str(Path(args.map_from_case_root).resolve()),
            "--target-case-root",
            str(output_case_root),
            "--source-time",
            args.map_from_time,
        ]
        if args.baram_openfoam_root:
            map_command.extend(["--baram-openfoam-root", args.baram_openfoam_root])
        for item in args.map_patch:
            map_command.extend(["--patch-map", item])
        for item in args.map_cutting_patch:
            map_command.extend(["--cutting-patch", item])

    flow_run_command = [
        sys.executable,
        str(script_dir / "baram_run_case.py"),
        "--case-root",
        run_case_root,
        "--set-start-from",
        "startTime",
        "--set-end-time",
        str(args.end_time),
        "--set-write-interval",
        str(args.write_interval),
        "--timeout",
        str(args.timeout),
    ]
    if args.baram_openfoam_root:
        flow_run_command.extend(["--baram-openfoam-root", args.baram_openfoam_root])

    # decomposePar for time 0 is still a direct OpenFOAM utility call.
    decompose_time0 = [
        "python",
        "-c",
        (
            "import os, subprocess, sys; "
            "baram=os.environ.get('BARAM_OPENFOAM_ROOT_ARG'); "
            "case=os.environ.get('FLOW_CASE_ARG'); "
            "env=dict(os.environ); "
            "env['WM_PROJECT']='OpenFOAM'; "
            "env['WM_PROJECT_VERSION']='v2412'; "
            "env['WM_PROJECT_DIR']=baram; "
            "env['FOAM_ETC']=os.path.join(baram,'etc'); "
            "env['FOAM_SIGFPE']=env.get('FOAM_SIGFPE','1'); "
            "env['PATH']=';'.join([os.path.join(baram,'bin'), os.path.join(baram,'lib'), os.path.join(baram,'lib','msmpi'), r'C:\\Users\\Malik\\AppData\\Local\\Programs\\BARAM\\solvers\\mingw64\\bin', r'C:\\Program Files\\Microsoft MPI\\Bin', env.get('PATH','')]); "
            "cmd=[os.path.join(baram,'bin','decomposePar.exe'), '-case', case, '-force', '-time', '0']; "
            "raise SystemExit(subprocess.run(cmd, env=env).returncode)"
        ),
    ]

    if args.preview:
        print("BUILD:", " ".join(build_command))
        print("DECOMPOSE:", " ".join(decompose_time0))
        if map_command:
            print("MAP:", " ".join(map_command))
            print("DECOMPOSE_AFTER_MAP:", " ".join(decompose_time0))
        print("RUN:", " ".join(flow_run_command))
        return

    build_result = run_command(build_command, script_dir)
    print(build_result.stdout, end="")
    if build_result.stderr:
        print(build_result.stderr, end="", file=sys.stderr)
    if build_result.returncode != 0:
        raise SystemExit(build_result.returncode)

    baram_root = args.baram_openfoam_root or r"C:\Users\Malik\AppData\Local\Programs\BARAM\solvers\openfoam"
    env = dict(os.environ)
    env["BARAM_OPENFOAM_ROOT_ARG"] = str(Path(baram_root).resolve())
    env["FLOW_CASE_ARG"] = str((output_case_root / "case") if output_case_root.name != "case" else output_case_root)
    decompose_result = subprocess.run(decompose_time0, cwd=str(script_dir), env=env, check=False, capture_output=True, text=True)
    print(decompose_result.stdout, end="")
    if decompose_result.stderr:
        print(decompose_result.stderr, end="", file=sys.stderr)
    if decompose_result.returncode != 0:
        raise SystemExit(decompose_result.returncode)

    if map_command:
        map_result = run_command(map_command, script_dir)
        print(map_result.stdout, end="")
        if map_result.stderr:
            print(map_result.stderr, end="", file=sys.stderr)
        if map_result.returncode != 0:
            raise SystemExit(map_result.returncode)

        redecompose_result = subprocess.run(
            decompose_time0,
            cwd=str(script_dir),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        print(redecompose_result.stdout, end="")
        if redecompose_result.stderr:
            print(redecompose_result.stderr, end="", file=sys.stderr)
        if redecompose_result.returncode != 0:
            raise SystemExit(redecompose_result.returncode)

    run_result = run_command(flow_run_command, script_dir)
    print(run_result.stdout, end="")
    if run_result.stderr:
        print(run_result.stderr, end="", file=sys.stderr)
    if run_result.returncode != 0:
        raise SystemExit(run_result.returncode)


if __name__ == "__main__":
    main()
