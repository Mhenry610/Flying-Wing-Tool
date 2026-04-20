from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless mesh workflows for variants in a BARAM study manifest.")
    parser.add_argument("--manifest", required=True, help="Path to study_manifest.json.")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant name to run. Repeat to select specific variants. Defaults to all variants with .bm bundles.",
    )
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument("--np", type=int, help="MPI process count override.")
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["castellated", "snap", "layers"],
        help="Specific mesh stages to run.",
    )
    parser.add_argument("--skip-blockmesh", action="store_true", help="Skip blockMesh.")
    parser.add_argument("--skip-decompose", action="store_true", help="Skip decomposePar.")
    parser.add_argument("--clean", action="store_true", help="Delete processor directories before meshing.")
    parser.add_argument("--patch-name", default="IntendedValidation2CFDmesh_surface", help="Patch name to track.")
    parser.add_argument("--compare-case-root", help="Optional baseline .bm bundle for comparison.")
    parser.add_argument("--log-prefix", default="mesh_automation", help="Prefix for per-case mesh logs.")
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
    filtered = [variant for variant in variants if isinstance(variant, dict) and variant.get("bm_bundle")]
    if not names:
        return filtered

    wanted = set(names)
    selected = [variant for variant in filtered if str(variant.get("name")) in wanted]
    found = {str(variant.get("name")) for variant in selected}
    missing = wanted - found
    if missing:
        raise KeyError(f"Variant(s) not found in manifest with bm_bundle: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    script_dir = Path(__file__).resolve().parent
    manifest = json.loads(manifest_path.read_text())
    selected = select_variants(manifest, args.variants)

    for variant in selected:
        variant_name = str(variant["name"])
        bm_bundle = Path(str(variant["bm_bundle"])).resolve()
        print(f"[mesh-study] variant={variant_name}")

        command = [sys.executable, str(script_dir / "baram_run_mesh_case.py"), "--case-root", str(bm_bundle)]
        if args.baram_openfoam_root:
            command.extend(["--baram-openfoam-root", args.baram_openfoam_root])
        if args.np is not None:
            command.extend(["--np", str(args.np)])
        if args.stages:
            for stage in args.stages:
                command.extend(["--stage", stage])
        if args.skip_blockmesh:
            command.append("--skip-blockmesh")
        if args.skip_decompose:
            command.append("--skip-decompose")
        if args.clean:
            command.append("--clean")
        if args.patch_name:
            command.extend(["--patch-name", args.patch_name])
        if args.compare_case_root:
            command.extend(["--compare-case-root", args.compare_case_root])
        if args.log_prefix:
            command.extend(["--log-prefix", args.log_prefix])

        result = run_subprocess(command, cwd=script_dir)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        variant["mesh_returncode"] = result.returncode
        variant["mesh_status"] = "mesh_ok" if result.returncode == 0 else "mesh_failed"
        summary_path = bm_bundle / "case" / f"{args.log_prefix}_summary.json"
        if summary_path.exists():
            variant["mesh_summary"] = str(summary_path)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[mesh-study] updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
