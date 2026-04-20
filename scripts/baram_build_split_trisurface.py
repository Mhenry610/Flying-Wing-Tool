from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

from baram_grid_study_tools import parse_layer_patch, parse_region_level, parse_surface_levels
from baram_export_cfd_geometry import classify_segment
from core.state import Project
from core.occ_utils.shapes import loft_surface_from_profiles, make_airfoil_wire_spline, loft_solid_from_wires, make_compound, scale_shape
from services.export.geometry_builder import _get_section_profiles_3d
from services.geometry import AeroSandboxService, SpanwiseSection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build split triSurface assets from a project JSON and patch a .bm case's "
            "snappyHexMeshDict to use centerbody / junction / outer-wing wall patches."
        )
    )
    parser.add_argument("--project-json", required=True, help="Path to the project JSON file.")
    parser.add_argument("--mesh-case-root", required=True, help="Path to the .bm bundle or inner case directory.")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000.0,
        help="Uniform scale before STL export. Default: 1000 (m -> mm).",
    )
    parser.add_argument(
        "--linear-deflection",
        type=float,
        default=1.0,
        help="Linear meshing deflection for STL export in exported units. Default: 1.0.",
    )
    parser.add_argument(
        "--angular-deflection",
        type=float,
        default=0.5,
        help="Angular meshing deflection for STL export in radians. Default: 0.5.",
    )
    parser.add_argument(
        "--volume-name",
        default="cfd_volume",
        help="Geometry name for the closed refinement-region STL.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional path for the generated summary JSON. Defaults inside the case directory.",
    )
    return parser.parse_args()


def normalize_case_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and path.name == "case":
        return path
    case_dir = path / "case"
    if case_dir.is_dir():
        return case_dir
    raise FileNotFoundError(f"Could not resolve case directory from {path}")


def build_surface_segment(section_a: SpanwiseSection, section_b: SpanwiseSection):
    upper_a, lower_a = _get_section_profiles_3d(section_a)
    upper_b, lower_b = _get_section_profiles_3d(section_b)
    if upper_a is None or lower_a is None or upper_b is None or lower_b is None:
        raise RuntimeError(f"Failed to derive section surface profiles for segment {section_a.index}->{section_b.index}.")

    upper_surface = loft_surface_from_profiles([upper_a, upper_b], which="upper")
    lower_surface = loft_surface_from_profiles([lower_a, lower_b], which="lower")
    if upper_surface is None or upper_surface.IsNull() or lower_surface is None or lower_surface.IsNull():
        raise RuntimeError(f"Failed to loft OML surfaces for segment {section_a.index}->{section_b.index}.")

    return make_compound([upper_surface, lower_surface])


def build_volume_solid(sections: List[SpanwiseSection]):
    wires = []
    for section in sections:
        upper, lower = _get_section_profiles_3d(section)
        if upper is None or lower is None:
            raise RuntimeError(f"Failed to derive full section profile for section {section.index}.")
        full_profile = upper + list(reversed(lower[1:-1]))
        wire = make_airfoil_wire_spline(full_profile)
        if wire is None:
            raise RuntimeError(f"Failed to build full airfoil wire for section {section.index}.")
        wires.append(wire)

    solid = loft_solid_from_wires(wires)
    if solid is None or solid.IsNull():
        raise RuntimeError("Failed to loft the closed full-volume solid.")
    return solid


def write_stl(shape, path: Path, *, linear_deflection: float, angular_deflection: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesher = BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), True)
    mesher.Perform()
    writer = StlAPI_Writer()
    try:
        writer.SetASCIIMode(False)
    except Exception:
        pass
    ok = writer.Write(shape, str(path))
    if ok is False:
        raise RuntimeError(f"Failed to write STL: {path}")


def find_named_block(text: str, name: str, opener: str, closer: str) -> Tuple[int, int]:
    search_from = 0
    while True:
        name_index = text.find(name, search_from)
        if name_index < 0:
            raise ValueError(f"Could not find block '{name}'")

        cursor = name_index + len(name)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != opener:
            search_from = cursor
            continue

        depth = 0
        end = cursor
        while end < len(text):
            ch = text[end]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end += 1
                    while end < len(text) and text[end].isspace():
                        end += 1
                    if end < len(text) and text[end] == ";":
                        end += 1
                    return name_index, end
            end += 1

        raise ValueError(f"Unbalanced block '{name}'")


def replace_named_block(text: str, name: str, opener: str, closer: str, body: str, *, trailing_semicolon: bool = False) -> str:
    start, end = find_named_block(text, name, opener, closer)
    replacement = f"{name}\n{opener}\n{body.rstrip()}\n{closer}"
    if trailing_semicolon:
        replacement += ";"
    replacement += "\n"
    return text[:start] + replacement + text[end:]


def patch_snappy_dict(
    snappy_path: Path,
    *,
    volume_name: str,
    patch_names: List[str],
) -> Dict[str, object]:
    text = snappy_path.read_text()
    try:
        _, surface_min_level, surface_max_level = parse_surface_levels(text)
    except Exception:
        surface_min_level, surface_max_level = 0, 0
    try:
        _, region_level = parse_region_level(text)
    except Exception:
        region_level = 7
    try:
        _, layer_count = parse_layer_patch(text)
    except Exception:
        layer_count = 3
    layer_style_match = re.search(
        r"[A-Za-z0-9_]+\s*\{\s*nSurfaceLayers\s+\d+\s*;\s*thicknessModel\s+([A-Za-z0-9_]+)\s*;\s*relativeSizes\s+([A-Za-z0-9_]+)\s*;\s*expansionRatio\s+([-+0-9.eE]+)\s*;\s*firstLayerThickness\s+([-+0-9.eE]+)\s*;\s*minThickness\s+([-+0-9.eE]+)\s*;",
        text,
        flags=re.DOTALL,
    )
    if layer_style_match:
        layer_thickness_model = layer_style_match.group(1)
        layer_relative_sizes = layer_style_match.group(2)
        layer_expansion_ratio = layer_style_match.group(3)
        layer_first_thickness = layer_style_match.group(4)
        layer_min_thickness = layer_style_match.group(5)
    else:
        layer_thickness_model = "firstAndExpansion"
        layer_relative_sizes = "on"
        layer_expansion_ratio = "1.2"
        layer_first_thickness = "0.001"
        layer_min_thickness = "0.3"

    geometry_body = "\n".join(
        [
            f"  {volume_name}.stl",
            "  {",
            "    type triSurfaceMesh;",
            f"    name {volume_name};",
            "  }",
            *[
                line
                for patch_name in patch_names
                for line in (
                    f"  {patch_name}.stl",
                    "  {",
                    "    type triSurfaceMesh;",
                    f"    name {patch_name};",
                    "  }",
                )
            ],
        ]
    )
    text = replace_named_block(text, "geometry", "{", "}", geometry_body)

    text = replace_named_block(text, "features", "(", ")", "", trailing_semicolon=True)
    text = text.replace("explicitFeatureSnap true;", "explicitFeatureSnap false;")

    refinement_surfaces_body = "\n".join(
        [
            line
            for patch_name in patch_names
            for line in (
                f"    {patch_name}",
                "    {",
                "      patchInfo",
                "      {",
                "        type patch;",
                "      }",
                "      level",
                "        (",
                f"          {surface_min_level}",
                f"          {surface_max_level}",
                "        );",
                "    }",
            )
        ]
    )
    text = replace_named_block(text, "refinementSurfaces", "{", "}", refinement_surfaces_body)

    refinement_regions_body = "\n".join(
        [
            f"    {volume_name}",
            "    {",
            "      mode inside;",
            "      levels",
            "        (",
            "          (",
            "            1000000000000000.0",
            f"            {region_level}",
            "          )",
            "        );",
            "    }",
        ]
    )
    text = replace_named_block(text, "refinementRegions", "{", "}", refinement_regions_body)

    layers_body = "\n".join(
        [
            line
            for patch_name in patch_names
            for line in (
                f"    {patch_name}",
                "    {",
                f"      nSurfaceLayers {layer_count};",
                f"      thicknessModel {layer_thickness_model};",
                f"      relativeSizes {layer_relative_sizes};",
                f"      expansionRatio {layer_expansion_ratio};",
                f"      firstLayerThickness {layer_first_thickness};",
                f"      minThickness {layer_min_thickness};",
                "    }",
            )
        ]
    )
    text = replace_named_block(text, "layers", "{", "}", layers_body)

    snappy_path.write_text(text)
    return {
        "surface_levels": [int(surface_min_level), int(surface_max_level)],
        "region_level": int(region_level),
        "layer_count": int(layer_count),
        "layer_style": {
            "thicknessModel": layer_thickness_model,
            "relativeSizes": layer_relative_sizes,
            "expansionRatio": layer_expansion_ratio,
            "firstLayerThickness": layer_first_thickness,
            "minThickness": layer_min_thickness,
        },
        "patch_names": list(patch_names),
        "volume_name": volume_name,
    }


def main() -> None:
    args = parse_args()
    project = Project.load(str(Path(args.project_json).resolve()))
    case_dir = normalize_case_dir(Path(args.mesh_case_root))
    tri_surface_dir = case_dir / "constant" / "triSurface"
    snappy_path = case_dir / "system" / "snappyHexMeshDict"

    service = AeroSandboxService(project)
    sections = service.spanwise_sections()
    if len(sections) < 2:
        raise ValueError(f"Need at least two sections, got {len(sections)}.")

    sorted_body_sections = sorted(project.wing.planform.body_sections, key=lambda sec: sec.y_pos)
    body_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else None

    patch_shapes: Dict[str, List[object]] = {}
    segment_summary: List[Dict[str, object]] = []
    for idx in range(len(sections) - 1):
        section_a = sections[idx]
        section_b = sections[idx + 1]
        patch_name = classify_segment(section_a, section_b, body_outer_y)
        shape = build_surface_segment(section_a, section_b)
        patch_shapes.setdefault(patch_name, []).append(shape)
        segment_summary.append(
            {
                "name": f"{patch_name}_seg_{idx:02d}",
                "patch": patch_name,
                "section_a_index": int(section_a.index),
                "section_b_index": int(section_b.index),
                "span_start_m": float(section_a.y_m),
                "span_end_m": float(section_b.y_m),
            }
        )

    patch_names = sorted(patch_shapes.keys())
    volume_shape = build_volume_solid(sections)
    volume_shape = scale_shape(volume_shape, float(args.scale_factor))
    write_stl(
        volume_shape,
        tri_surface_dir / f"{args.volume_name}.stl",
        linear_deflection=args.linear_deflection,
        angular_deflection=args.angular_deflection,
    )

    patch_files: List[Dict[str, object]] = []
    for patch_name in patch_names:
        patch_shape = make_compound(patch_shapes[patch_name])
        patch_shape = scale_shape(patch_shape, float(args.scale_factor))
        stl_path = tri_surface_dir / f"{patch_name}.stl"
        write_stl(
            patch_shape,
            stl_path,
            linear_deflection=args.linear_deflection,
            angular_deflection=args.angular_deflection,
        )
        patch_files.append({"name": patch_name, "path": str(stl_path.resolve())})

    snappy_summary = patch_snappy_dict(snappy_path, volume_name=args.volume_name, patch_names=patch_names)

    summary = {
        "project_json": str(Path(args.project_json).resolve()),
        "case_dir": str(case_dir),
        "tri_surface_dir": str(tri_surface_dir),
        "volume_name": args.volume_name,
        "scale_factor": float(args.scale_factor),
        "linear_deflection": float(args.linear_deflection),
        "angular_deflection": float(args.angular_deflection),
        "patch_files": patch_files,
        "segment_summary": segment_summary,
        "snappy_hex_mesh_dict": str(snappy_path.resolve()),
        "snappy_settings": snappy_summary,
        "notes": [
            "Patch STLs contain upper+lower OML strip surfaces only; they do not include spanwise cap faces.",
            "The volume STL is a closed lofted solid used for the snappyHexMesh inside-region definition.",
            "explicitFeatureSnap is disabled in the patched snappyHexMeshDict to remove the legacy OBJ dependency.",
            "Existing refinement surface levels, region level, and layer count are preserved and replicated across the split wall patches.",
        ],
    }

    summary_path = Path(args.summary_path).resolve() if args.summary_path else case_dir / "split_trisurface_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[baram_build_split_trisurface] Wrote summary: {summary_path}")
    for patch_file in patch_files:
        print(f"[baram_build_split_trisurface] patch {patch_file['name']}: {patch_file['path']}")
    print(f"[baram_build_split_trisurface] volume: {tri_surface_dir / f'{args.volume_name}.stl'}")


if __name__ == "__main__":
    main()
