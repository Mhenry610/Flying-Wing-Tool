from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and postprocess variants from a BARAM study manifest.")
    parser.add_argument("--manifest", required=True, help="Path to study_manifest.json.")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant name to run. Repeat to select specific variants. Defaults to all variants.",
    )
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--solver", help="Override solver name.")
    parser.add_argument("--serial", action="store_true", help="Force serial execution.")
    parser.add_argument("--np", type=int, help="MPI process count override.")
    parser.add_argument("--dry-run", action="store_true", help="Use solver dry-run mode.")
    parser.add_argument("--post-process", action="store_true", help="Run solver in postProcess mode.")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds for each solver run.")
    parser.add_argument("--set-start-from", help="Patch controlDict startFrom before running.")
    parser.add_argument("--set-end-time", type=float, help="Patch controlDict endTime before running.")
    parser.add_argument("--set-write-interval", type=float, help="Patch controlDict writeInterval before running.")
    parser.add_argument("--set-application", help="Patch controlDict application before running.")
    parser.add_argument("--log-prefix", default="automation", help="Prefix for per-case logs.")
    parser.add_argument("--summary-json", help="Project summary JSON for postprocessing comparisons.")
    parser.add_argument("--report-output-dir", help="Directory for generated summaries and plots.")
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip baram_postprocess.py even if summary/report output are supplied.",
    )
    return parser.parse_args()


def run_subprocess(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def select_variants(manifest: dict[str, object], names: list[str] | None) -> list[dict[str, object]]:
    variants = manifest.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Manifest 'variants' field is not a list.")
    if not names:
        return [variant for variant in variants if isinstance(variant, dict)]

    wanted = set(names)
    selected = [
        variant for variant in variants if isinstance(variant, dict) and str(variant.get("name")) in wanted
    ]
    found = {str(variant.get("name")) for variant in selected}
    missing = wanted - found
    if missing:
        raise KeyError(f"Variant(s) not found in manifest: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    script_dir = Path(__file__).resolve().parent
    manifest = json.loads(manifest_path.read_text())
    selected = select_variants(manifest, args.variants)

    report_output_dir = Path(args.report_output_dir).resolve() if args.report_output_dir else None

    for variant in selected:
        case_root = Path(str(variant["bf_bundle"])).resolve()
        variant_name = str(variant["name"])
        print(f"[study] variant={variant_name}")

        run_command = [sys.executable, str(script_dir / "baram_run_case.py"), "--case-root", str(case_root)]
        if args.baram_openfoam_root:
            run_command.extend(["--baram-openfoam-root", args.baram_openfoam_root])
        if args.solver:
            run_command.extend(["--solver", args.solver])
        if args.serial:
            run_command.append("--serial")
        if args.np is not None:
            run_command.extend(["--np", str(args.np)])
        if args.dry_run:
            run_command.append("--dry-run")
        if args.post_process:
            run_command.append("--post-process")
        if args.timeout is not None:
            run_command.extend(["--timeout", str(args.timeout)])
        if args.set_start_from is not None:
            run_command.extend(["--set-start-from", args.set_start_from])
        if args.set_end_time is not None:
            run_command.extend(["--set-end-time", str(args.set_end_time)])
        if args.set_write_interval is not None:
            run_command.extend(["--set-write-interval", str(args.set_write_interval)])
        if args.set_application is not None:
            run_command.extend(["--set-application", args.set_application])
        if args.log_prefix:
            run_command.extend(["--log-prefix", args.log_prefix])

        run_result = run_subprocess(run_command, cwd=script_dir)
        print(run_result.stdout, end="")
        if run_result.stderr:
            print(run_result.stderr, end="", file=sys.stderr)

        variant["run_returncode"] = run_result.returncode
        variant["status"] = "run_ok" if run_result.returncode == 0 else "run_failed"

        if run_result.returncode != 0:
            continue

        if args.skip_postprocess or report_output_dir is None:
            continue

        report_output_dir.mkdir(parents=True, exist_ok=True)
        summary_name = f"{manifest.get('study_name', 'study')}_{variant_name}"
        post_command = [
            sys.executable,
            str(script_dir / "baram_postprocess.py"),
            "--case-root",
            str(case_root),
            "--output-dir",
            str(report_output_dir),
            "--case-name",
            summary_name,
        ]
        if args.summary_json:
            post_command.extend(["--summary-json", args.summary_json])

        post_result = run_subprocess(post_command, cwd=script_dir)
        print(post_result.stdout, end="")
        if post_result.stderr:
            print(post_result.stderr, end="", file=sys.stderr)

        variant["postprocess_returncode"] = post_result.returncode
        if post_result.returncode == 0:
            variant["postprocess_summary"] = str((report_output_dir / f"{summary_name}_summary.json").resolve())
        else:
            variant["status"] = "postprocess_failed"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[study] updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
