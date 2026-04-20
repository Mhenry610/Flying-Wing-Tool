from __future__ import annotations

import argparse
from pathlib import Path

from baram_case_tools import discover_baram_openfoam_root, normalize_case_dir, run_openfoam_utility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map a converged BARAM/OpenFOAM solution onto a target case, with optional patch mapping."
    )
    parser.add_argument("--source-case-root", required=True, help="Path to the source .bf/.bm bundle or inner case dir.")
    parser.add_argument("--target-case-root", required=True, help="Path to the target .bf/.bm bundle or inner case dir.")
    parser.add_argument("--source-time", default="latestTime", help="Source time to map from.")
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root.")
    parser.add_argument(
        "--patch-map",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="Patch mapping entry in target=source form. Repeat for multiple patches.",
    )
    parser.add_argument(
        "--cutting-patch",
        action="append",
        default=[],
        metavar="PATCH",
        help="Patch name to list under cuttingPatches. Repeat for multiple patches.",
    )
    parser.add_argument(
        "--consistent",
        action="store_true",
        help="Use mapFields -consistent instead of writing mapFieldsDict.",
    )
    parser.add_argument(
        "--summary-name",
        default="mapped_field_summary.txt",
        help="Name of the text summary file written into the target case directory.",
    )
    return parser.parse_args()


def write_map_fields_dict(target_case_dir: Path, patch_pairs: list[tuple[str, str]], cutting_patches: list[str]) -> Path:
    system_dir = target_case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    dict_path = system_dir / "mapFieldsDict"

    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        '    object      mapFieldsDict;',
        "}",
        "",
        "patchMap",
        "(",
    ]
    for target_patch, source_patch in patch_pairs:
        lines.append(f"    {target_patch} {source_patch}")
    lines.extend(
        [
            ");",
            "",
            "cuttingPatches",
            "(",
        ]
    )
    for patch_name in cutting_patches:
        lines.append(f"    {patch_name}")
    lines.extend(
        [
            ");",
            "",
        ]
    )
    dict_path.write_text("\n".join(lines))
    return dict_path


def parse_patch_maps(raw_items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --patch-map '{item}'. Use TARGET=SOURCE.")
        target_patch, source_patch = item.split("=", 1)
        target_patch = target_patch.strip()
        source_patch = source_patch.strip()
        if not target_patch or not source_patch:
            raise ValueError(f"Invalid --patch-map '{item}'. Use TARGET=SOURCE.")
        pairs.append((target_patch, source_patch))
    return pairs


def main() -> None:
    args = parse_args()
    openfoam_root = discover_baram_openfoam_root(args.baram_openfoam_root)
    source_case_dir = normalize_case_dir(Path(args.source_case_root))
    target_case_dir = normalize_case_dir(Path(args.target_case_root))

    patch_pairs = parse_patch_maps(args.patch_map)
    summary_lines = [
        f"source_case_dir={source_case_dir}",
        f"target_case_dir={target_case_dir}",
        f"source_time={args.source_time}",
        f"consistent={args.consistent}",
    ]

    command = ["-case", str(target_case_dir), str(source_case_dir), "-sourceTime", str(args.source_time)]
    if args.consistent:
        command.append("-consistent")
    else:
        dict_path = write_map_fields_dict(target_case_dir, patch_pairs, args.cutting_patch)
        summary_lines.append(f"map_fields_dict={dict_path}")
    if patch_pairs:
        summary_lines.append("patch_map=" + ",".join(f"{target}<-{source}" for target, source in patch_pairs))
    if args.cutting_patch:
        summary_lines.append("cutting_patches=" + ",".join(args.cutting_patch))

    result = run_openfoam_utility(
        openfoam_root,
        "mapFields",
        command,
        cwd=target_case_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mapFields failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    stdout_path = target_case_dir / "mapFields_stdout.log"
    stderr_path = target_case_dir / "mapFields_stderr.log"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    summary_lines.append(f"stdout_log={stdout_path}")
    summary_lines.append(f"stderr_log={stderr_path}")

    summary_path = target_case_dir / args.summary_name
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote map summary: {summary_path}")


if __name__ == "__main__":
    main()
