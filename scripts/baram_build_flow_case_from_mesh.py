from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from baram_case_tools import (
    discover_baram_openfoam_root,
    list_numeric_time_dirs,
    normalize_case_dir,
    run_openfoam_utility,
)


FLOW_SYSTEM_FILES = ["controlDict", "decomposeParDict", "fvSchemes", "fvSolution"]
FLOW_CONSTANT_FILES = ["g", "operatingConditions", "thermophysicalProperties", "turbulenceProperties"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a solver-ready flow case by combining a remeshed .bm case with a .bf flow template."
    )
    parser.add_argument("--mesh-case-root", required=True, help="Path to the source .bm bundle or inner case directory.")
    parser.add_argument(
        "--flow-template-case-root",
        required=True,
        help="Path to the source .bf bundle or inner case directory. A reconstructed root 0/ is required.",
    )
    parser.add_argument(
        "--output-case-root",
        help="Optional output .bm bundle or case directory. Defaults to in-place update of --mesh-case-root.",
    )
    parser.add_argument(
        "--mesh-time",
        default="latest",
        help="Mesh time directory to promote into constant/polyMesh. Use 'latest' or a numeric time such as 3.",
    )
    parser.add_argument("--baram-openfoam-root", help="Path to the BARAM OpenFOAM root for reconstructParMesh if needed.")
    parser.add_argument(
        "--decompose-zero",
        action="store_true",
        help="After assembly, run metadata-only preparation by marking that processor 0/ should be redecomposed externally.",
    )
    parser.add_argument("--summary-name", default="synthetic_flow_build_summary.json", help="Summary filename.")
    return parser.parse_args()


def resolve_mesh_time(case_dir: Path, mesh_time: str) -> str:
    if mesh_time != "latest":
        return mesh_time
    times = [path.name for path in list_numeric_time_dirs(case_dir) if path.name != "0"]
    if not times:
        processor0 = case_dir / "processor0"
        if processor0.is_dir():
            times = [path.name for path in list_numeric_time_dirs(processor0) if path.name != "0"]
    if not times:
        raise FileNotFoundError(f"No numeric mesh times found under {case_dir}")
    return times[-1]


def parse_patch_types(boundary_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z0-9_]+)\s*\{\s*type\s+([A-Za-z0-9_]+)\s*;", re.DOTALL)
    for patch, patch_type in pattern.findall(boundary_text):
        result[patch] = patch_type
    return result


def replace_patch_type(boundary_text: str, patch_name: str, patch_type: str) -> str:
    pattern = re.compile(rf"({re.escape(patch_name)}\s*\{{\s*type\s+)([A-Za-z0-9_]+)(\s*;)", re.DOTALL)
    return pattern.sub(rf"\1{patch_type}\3", boundary_text)


def find_named_block(text: str, name: str) -> tuple[int, int]:
    match = re.search(rf"(^|\s){re.escape(name)}\s*\{{", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find block '{name}'")
    name_index = match.start()

    cursor = match.end() - 1
    while cursor > name_index and text[cursor] != "{":
        cursor -= 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        raise ValueError(f"Block '{name}' is not followed by '{{'")

    depth = 0
    end = cursor
    while end < len(text):
        ch = text[end]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return name_index, end + 1
        end += 1
    raise ValueError(f"Unbalanced block '{name}'")


def iter_top_level_entries(block_body: str) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    idx = 0
    length = len(block_body)
    while idx < length:
        while idx < length and block_body[idx].isspace():
            idx += 1
        if idx >= length:
            break

        name_start = idx
        while idx < length and not block_body[idx].isspace() and block_body[idx] != "{":
            idx += 1
        name = block_body[name_start:idx].strip()
        while idx < length and block_body[idx].isspace():
            idx += 1
        if not name or idx >= length or block_body[idx] != "{":
            idx += 1
            continue

        depth = 0
        entry_end = idx
        while entry_end < length:
            ch = block_body[entry_end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    entries.append((name, name_start, entry_end + 1))
                    idx = entry_end + 1
                    break
            entry_end += 1
        else:
            break

    return entries


def rewrite_boundary_field(
    field_text: str,
    *,
    mesh_patch_names: list[str],
    source_patch_name: str | None,
) -> str:
    try:
        block_start, block_end = find_named_block(field_text, "boundaryField")
    except ValueError:
        return field_text

    brace_index = field_text.find("{", block_start)
    block_body = field_text[brace_index + 1 : block_end - 1]
    entries = iter_top_level_entries(block_body)
    if not entries:
        return field_text

    existing_names = [name for name, _, _ in entries]
    pieces: list[str] = []
    source_entry = None
    cursor = 0
    for name, start, end in entries:
        pieces.append(block_body[cursor:start])
        entry_text = block_body[start:end]
        if source_patch_name and name == source_patch_name:
            source_entry = entry_text
        if name in mesh_patch_names:
            pieces.append(entry_text)
        cursor = end
    pieces.append(block_body[cursor:])

    if source_entry:
        for patch_name in mesh_patch_names:
            if patch_name in existing_names:
                continue
            cloned_entry = re.sub(
                rf"^\s*{re.escape(source_patch_name)}\b",
                f"    {patch_name}",
                source_entry,
                count=1,
                flags=re.MULTILINE,
            )
            if not cloned_entry.endswith("\n"):
                cloned_entry += "\n"
            pieces.append(cloned_entry)

    new_block_body = "".join(pieces)
    return field_text[: brace_index + 1] + new_block_body + field_text[block_end - 1 :]


def patch_zero_boundary_fields(zero_dir: Path, mesh_boundary_path: Path, flow_boundary_path: Path) -> dict[str, object]:
    mesh_patch_types = parse_patch_types(mesh_boundary_path.read_text())
    flow_patch_types = parse_patch_types(flow_boundary_path.read_text())

    mesh_patch_names = list(mesh_patch_types.keys())
    template_only = [name for name in flow_patch_types.keys() if name not in mesh_patch_types]
    source_patch_name = None
    generic_boundary_patches = {"xMin", "xMax", "yMin", "yMax", "zMin", "zMax"}
    for name in template_only:
        if name not in generic_boundary_patches and not name.startswith("procBoundary"):
            source_patch_name = name
            break

    patched_files: list[str] = []
    for path in sorted(zero_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            original_text = path.read_text()
        except UnicodeDecodeError:
            continue
        rewritten = rewrite_boundary_field(
            original_text,
            mesh_patch_names=mesh_patch_names,
            source_patch_name=source_patch_name,
        )
        if rewritten != original_text:
            path.write_text(rewritten)
            patched_files.append(str(path))

    return {
        "mesh_patch_names": mesh_patch_names,
        "template_only_patches": template_only,
        "source_wall_patch": source_patch_name,
        "source_wall_patch_type": flow_patch_types.get(source_patch_name) if source_patch_name else None,
        "patched_files": patched_files,
    }


def copy_tree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ensure_reconstructed_mesh_time(case_dir: Path, mesh_time: str, baram_openfoam_root: str | None) -> Path:
    promoted_mesh = case_dir / mesh_time / "polyMesh"
    if promoted_mesh.is_dir():
        return promoted_mesh

    processor_mesh = case_dir / "processor0" / mesh_time / "polyMesh"
    if not processor_mesh.is_dir():
        raise FileNotFoundError(f"Mesh polyMesh directory not found in root or processor0 for time {mesh_time}: {case_dir}")

    openfoam_root = discover_baram_openfoam_root(baram_openfoam_root)
    result = run_openfoam_utility(
        openfoam_root,
        "reconstructParMesh",
        ["-case", str(case_dir), "-time", mesh_time, "-constant"],
        cwd=case_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "reconstructParMesh failed while preparing the remeshed case.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if not promoted_mesh.is_dir():
        raise FileNotFoundError(f"reconstructParMesh completed but {promoted_mesh} was not created.")
    return promoted_mesh


def patch_control_dict_for_split_patches(control_dict_path: Path, split_patch_names: list[str], legacy_patch_name: str | None) -> dict[str, object]:
    text = control_dict_path.read_text()
    patch_list = " ".join(split_patch_names)
    replacements = 0

    if legacy_patch_name:
        text, patch_count = re.subn(
            rf"(patches\s*\()\s*{re.escape(legacy_patch_name)}\s*(\);)",
            rf"\1 {patch_list} \2",
            text,
        )
        replacements += patch_count

        surface_monitor_pattern = re.compile(
            rf"\n\s*surface-mon-1\s*\{{.*?name\s+{re.escape(legacy_patch_name)}\s*;.*?\n\s*\}}",
            flags=re.DOTALL,
        )
        text, removed_count = surface_monitor_pattern.subn("", text)
    else:
        removed_count = 0

    patch_force_monitor_template = None
    template_match = re.search(r"(^\s*force-mon-1_forces\s*\{.*?^\s*\})", text, flags=re.MULTILINE | re.DOTALL)
    if template_match:
        patch_force_monitor_template = template_match.group(1)

    added_patch_monitors = 0
    if patch_force_monitor_template:
        functions_start, functions_end = find_named_block(text, "functions")
        insert_at = functions_end - 1
        insertion = []
        for patch_name in split_patch_names:
            monitor_name = f"force-mon-{patch_name}_forces"
            if monitor_name in text:
                continue
            patch_block = patch_force_monitor_template
            patch_block = re.sub(r"\bforce-mon-1_forces\b", monitor_name, patch_block, count=1)
            patch_block = re.sub(r"(\bpatches\s*\()\s*[^)]*(\);)", rf"\1 {patch_name} \2", patch_block, count=1)
            insertion.append("\n" + patch_block.rstrip() + "\n")
            added_patch_monitors += 1
        if insertion:
            text = text[:insert_at] + "".join(insertion) + text[insert_at:]

    if text != control_dict_path.read_text():
        control_dict_path.write_text(text)

    return {
        "legacy_patch_name": legacy_patch_name,
        "split_patch_names": split_patch_names,
        "patched_force_monitor_entries": replacements,
        "removed_surface_monitors": removed_count,
        "added_patch_force_monitors": added_patch_monitors,
    }


def main() -> None:
    args = parse_args()
    mesh_case_dir = normalize_case_dir(Path(args.mesh_case_root))
    flow_template_dir = normalize_case_dir(Path(args.flow_template_case_root))
    if args.output_case_root:
        raw_output = Path(args.output_case_root).resolve()
        output_bundle_root = raw_output.parent if raw_output.name == "case" else raw_output
        output_case_dir = output_bundle_root / "case"
    else:
        output_bundle_root = mesh_case_dir.parent
        output_case_dir = mesh_case_dir

    mesh_time = resolve_mesh_time(mesh_case_dir, args.mesh_time)
    ensure_reconstructed_mesh_time(mesh_case_dir, mesh_time, args.baram_openfoam_root)

    if output_case_dir != mesh_case_dir:
        if output_bundle_root.exists():
            shutil.rmtree(output_bundle_root)
        shutil.copytree(mesh_case_dir.parent, output_bundle_root)
        output_case_dir = output_bundle_root / "case"

    template_zero = flow_template_dir / "0"
    if not template_zero.is_dir():
        raise FileNotFoundError(
            f"Flow template root 0/ directory not found: {template_zero}. Reconstruct root fields first."
        )

    promoted_mesh = ensure_reconstructed_mesh_time(output_case_dir, mesh_time, args.baram_openfoam_root)

    copy_tree_replace(promoted_mesh, output_case_dir / "constant" / "polyMesh")
    copy_tree_replace(template_zero, output_case_dir / "0")

    for name in FLOW_SYSTEM_FILES:
        shutil.copy2(flow_template_dir / "system" / name, output_case_dir / "system" / name)
    for name in FLOW_CONSTANT_FILES:
        shutil.copy2(flow_template_dir / "constant" / name, output_case_dir / "constant" / name)

    flow_boundary_path = flow_template_dir / "constant" / "polyMesh" / "boundary"
    output_boundary_path = output_case_dir / "constant" / "polyMesh" / "boundary"
    flow_boundary_text = flow_boundary_path.read_text()
    output_boundary_text = output_boundary_path.read_text()
    flow_patch_types = parse_patch_types(flow_boundary_text)

    for patch_name, patch_type in flow_patch_types.items():
        output_boundary_text = replace_patch_type(output_boundary_text, patch_name, patch_type)
    source_wall_patch = None
    for patch_name in flow_patch_types:
        if patch_name not in parse_patch_types(output_boundary_text) and patch_name not in {"xMin", "xMax", "yMin", "yMax", "zMin", "zMax"}:
            source_wall_patch = patch_name
            break
    if source_wall_patch:
        source_wall_type = flow_patch_types.get(source_wall_patch)
        if source_wall_type:
            output_patch_types = parse_patch_types(output_boundary_text)
            for patch_name in output_patch_types:
                if patch_name not in flow_patch_types and output_patch_types.get(patch_name) == "patch":
                    output_boundary_text = replace_patch_type(output_boundary_text, patch_name, source_wall_type)
    output_boundary_path.write_text(output_boundary_text)

    zero_patch_summary = patch_zero_boundary_fields(output_case_dir / "0", output_boundary_path, flow_boundary_path)
    control_dict_monitor_summary = patch_control_dict_for_split_patches(
        output_case_dir / "system" / "controlDict",
        split_patch_names=[name for name in zero_patch_summary["mesh_patch_names"] if name not in {"xMin", "xMax", "yMin", "yMax", "zMin", "zMax"}],
        legacy_patch_name=zero_patch_summary["source_wall_patch"],
    )

    summary = {
        "mesh_case_dir": str(mesh_case_dir),
        "flow_template_dir": str(flow_template_dir),
        "output_case_dir": str(output_case_dir),
        "promoted_mesh_time": mesh_time,
        "flow_system_files": FLOW_SYSTEM_FILES,
        "flow_constant_files": FLOW_CONSTANT_FILES,
        "copied_zero_dir": str(template_zero),
        "zero_boundary_patch_summary": zero_patch_summary,
        "control_dict_monitor_summary": control_dict_monitor_summary,
        "note": (
            "Run decomposePar on time 0 before solving so processor 0/ files receive processor-boundary entries."
            if args.decompose_zero
            else "Assembled solver-ready root case; decomposePar on time 0 is still required before solving."
        ),
    }
    summary_path = output_case_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote build summary: {summary_path}")


if __name__ == "__main__":
    main()
