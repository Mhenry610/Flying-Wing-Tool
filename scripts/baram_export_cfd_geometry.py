from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.state import Project
from core.occ_utils.shapes import make_airfoil_wire_spline, loft_solid_from_wires, make_compound, mirror_y, scale_shape
from services.export.geometry_builder import (
    WingGeometryConfig,
    apply_twist_to_points,
    build_control_surface_geometry,
)
from services.export.step_export import write_step
from services.geometry import AeroSandboxService, SpanwiseSection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export CFD-oriented geometry patches from a project JSON using the repo's "
            "existing geometry stack. The default output is a half-model split into "
            "centerbody, junction, and outer-wing OML solids for BARAM remeshing."
        )
    )
    parser.add_argument("--project-json", required=True, help="Path to the project JSON file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where STEP files and the patch manifest will be written.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1000.0,
        help="Uniform geometry scale before STEP export. Default: 1000 (m -> mm).",
    )
    parser.add_argument(
        "--full-aircraft",
        action="store_true",
        help="Mirror the half-model to export a full-aircraft patch kit.",
    )
    parser.add_argument(
        "--include-control-surfaces",
        action="store_true",
        help=(
            "Also export control-surface solids from the geometry stack. These are written "
            "as optional auxiliaries because they can overlap the main OML split."
        ),
    )
    parser.add_argument(
        "--case-name",
        help="Optional case name used in the manifest. Defaults to the project stem.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unnamed"


def build_section_wire(section: SpanwiseSection):
    if section.airfoil is None or section.airfoil.coordinates is None:
        return None

    coords = np.array(section.airfoil.coordinates)
    chord = float(section.chord_m)
    x = coords[:, 0] * chord + float(section.x_le_m)
    y = np.full_like(x, float(section.y_m))
    z = coords[:, 1] * chord + float(section.z_m)
    points_3d = np.column_stack([x, y, z])

    if abs(float(section.twist_deg)) > 1e-6:
        twisted = apply_twist_to_points(
            [pt for pt in points_3d],
            float(section.twist_deg),
            pivot_x=float(section.x_le_m),
            pivot_z=float(section.z_m),
        )
        points_3d = np.array(twisted)

    return make_airfoil_wire_spline(points_3d)


def classify_segment(
    section_a: SpanwiseSection,
    section_b: SpanwiseSection,
    body_outer_y: Optional[float],
    tol_m: float = 1.0e-6,
) -> str:
    if body_outer_y is None or body_outer_y <= tol_m:
        return "outer_wing"

    y0 = abs(float(section_a.y_m))
    y1 = abs(float(section_b.y_m))
    low = min(y0, y1)
    high = max(y0, y1)

    if high <= body_outer_y + tol_m:
        return "centerbody"
    if low < body_outer_y - tol_m and high > body_outer_y + tol_m:
        return "junction"
    if low <= body_outer_y + tol_m and high > body_outer_y + tol_m:
        return "junction"
    return "outer_wing"


def make_region_compounds(
    segment_records: List[Dict[str, object]],
) -> Dict[str, object]:
    grouped: Dict[str, List[object]] = {}
    for record in segment_records:
        region = str(record["region"])
        grouped.setdefault(region, []).append(record["shape"])

    compounds: Dict[str, object] = {}
    for region, shapes in grouped.items():
        compounds[region] = make_compound(shapes)
    return compounds


def maybe_transform_shape(shape, *, scale_factor: float, mirror_to_full_aircraft: bool):
    transformed = shape
    if scale_factor != 1.0:
        transformed = scale_shape(transformed, scale_factor)
    if mirror_to_full_aircraft:
        transformed = make_compound([transformed, mirror_y(transformed)])
    return transformed


def write_shape(shape, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(write_step(shape, str(path)))


def export_region_shapes(
    region_shapes: Dict[str, object],
    output_dir: Path,
    *,
    scale_factor: float,
    full_aircraft: bool,
) -> List[Dict[str, object]]:
    files: List[Dict[str, object]] = []
    for region, shape in sorted(region_shapes.items()):
        transformed = maybe_transform_shape(
            shape,
            scale_factor=scale_factor,
            mirror_to_full_aircraft=full_aircraft,
        )
        filename = f"{region}.step"
        path = output_dir / filename
        ok = write_shape(transformed, path)
        files.append(
            {
                "name": region,
                "path": str(path.resolve()),
                "status": "written" if ok else "failed",
            }
        )
    return files


def export_segment_shapes(
    segment_records: List[Dict[str, object]],
    output_dir: Path,
    *,
    scale_factor: float,
    full_aircraft: bool,
) -> List[Dict[str, object]]:
    files: List[Dict[str, object]] = []
    for record in segment_records:
        transformed = maybe_transform_shape(
            record["shape"],
            scale_factor=scale_factor,
            mirror_to_full_aircraft=full_aircraft,
        )
        name = str(record["name"])
        path = output_dir / f"{name}.step"
        ok = write_shape(transformed, path)
        files.append(
            {
                "name": name,
                "region": record["region"],
                "path": str(path.resolve()),
                "status": "written" if ok else "failed",
            }
        )
    return files


def export_control_surface_shapes(
    project: Project,
    output_dir: Path,
    *,
    scale_factor: float,
    full_aircraft: bool,
) -> List[Dict[str, object]]:
    service = AeroSandboxService(project)
    sections = service.spanwise_sections()
    control_surfaces = build_control_surface_geometry(sections, project.wing.planform, project)

    files: List[Dict[str, object]] = []
    for name, shape in sorted(control_surfaces.items()):
        transformed = maybe_transform_shape(
            shape,
            scale_factor=scale_factor,
            mirror_to_full_aircraft=full_aircraft,
        )
        slug = slugify(name)
        path = output_dir / f"{slug}.step"
        ok = write_shape(transformed, path)
        files.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "status": "written" if ok else "failed",
                "note": "Optional auxiliary export. Control surfaces may overlap the base OML split.",
            }
        )
    return files


def build_segment_records(project: Project) -> Tuple[List[SpanwiseSection], List[Dict[str, object]], Optional[float]]:
    service = AeroSandboxService(project)
    sections = service.spanwise_sections()
    if len(sections) < 2:
        raise ValueError(f"Need at least two spanwise sections, got {len(sections)}.")

    planform = project.wing.planform
    sorted_body_sections = sorted(planform.body_sections, key=lambda sec: sec.y_pos)
    body_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else None

    records: List[Dict[str, object]] = []
    for idx in range(len(sections) - 1):
        section_a = sections[idx]
        section_b = sections[idx + 1]
        wire_a = build_section_wire(section_a)
        wire_b = build_section_wire(section_b)
        if wire_a is None or wire_b is None:
            raise RuntimeError(f"Failed to build section wire for segment {idx}.")

        strip_solid = loft_solid_from_wires([wire_a, wire_b])
        if strip_solid is None or strip_solid.IsNull():
            raise RuntimeError(f"Failed to loft closed strip solid for segment {idx}.")

        region = classify_segment(section_a, section_b, body_outer_y)
        name = f"{region}_seg_{idx:02d}"
        records.append(
            {
                "name": name,
                "region": region,
                "shape": strip_solid,
                "section_a_index": int(section_a.index),
                "section_b_index": int(section_b.index),
                "span_start_m": float(section_a.y_m),
                "span_end_m": float(section_b.y_m),
                "span_fraction_start": float(section_a.span_fraction),
                "span_fraction_end": float(section_b.span_fraction),
            }
        )

    return sections, records, body_outer_y


def serializable_segments(segment_records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    serializable: List[Dict[str, object]] = []
    for record in segment_records:
        serializable.append(
            {
                key: value
                for key, value in record.items()
                if key != "shape"
            }
        )
    return serializable


def main() -> None:
    args = parse_args()
    project_path = Path(args.project_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project = Project.load(str(project_path))
    sections, segment_records, body_outer_y = build_segment_records(project)
    region_shapes = make_region_compounds(segment_records)

    case_name = args.case_name or project_path.stem
    region_dir = output_dir / "region_patches"
    strip_dir = output_dir / "strip_patches"
    control_dir = output_dir / "optional_control_surfaces"

    region_files = export_region_shapes(
        region_shapes,
        region_dir,
        scale_factor=args.scale_factor,
        full_aircraft=args.full_aircraft,
    )
    strip_files = export_segment_shapes(
        segment_records,
        strip_dir,
        scale_factor=args.scale_factor,
        full_aircraft=args.full_aircraft,
    )

    control_files: List[Dict[str, object]] = []
    if args.include_control_surfaces:
        control_files = export_control_surface_shapes(
            project,
            control_dir,
            scale_factor=args.scale_factor,
            full_aircraft=args.full_aircraft,
        )

    summary = {
        "case_name": case_name,
        "project_json": str(project_path),
        "output_dir": str(output_dir),
        "scale_factor": float(args.scale_factor),
        "full_aircraft": bool(args.full_aircraft),
        "body_outer_y_m": float(body_outer_y) if body_outer_y is not None else None,
        "section_count": len(sections),
        "sections": [
            {
                "index": int(section.index),
                "span_fraction": float(section.span_fraction),
                "y_m": float(section.y_m),
                "x_le_m": float(section.x_le_m),
                "z_m": float(section.z_m),
                "chord_m": float(section.chord_m),
                "twist_deg": float(section.twist_deg),
            }
            for section in sections
        ],
        "segments": serializable_segments(segment_records),
        "region_patch_files": region_files,
        "strip_patch_files": strip_files,
        "optional_control_surface_files": control_files,
        "notes": [
            "Region patches are compounds of closed spanwise strip solids built from the repo's existing geometry stack.",
            "Default grouping is centerbody / junction / outer_wing to support drag-attribution remeshes.",
            "The script exports a half-model by default because the current CFD workflow uses a symmetry plane.",
            "Control surfaces are optional auxiliaries because they can overlap the base OML split.",
        ],
    }

    manifest_path = output_dir / "cfd_patch_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    print(f"[baram_export_cfd_geometry] Wrote manifest: {manifest_path}")
    for item in region_files:
        print(f"[baram_export_cfd_geometry] region {item['name']}: {item['status']} -> {item['path']}")
    for item in strip_files:
        print(f"[baram_export_cfd_geometry] strip {item['name']}: {item['status']} -> {item['path']}")
    if control_files:
        for item in control_files:
            print(f"[baram_export_cfd_geometry] control {item['name']}: {item['status']} -> {item['path']}")


if __name__ == "__main__":
    main()
