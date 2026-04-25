"""Compatibility layer for STEP export helpers.

The canonical manufacturing STEP implementation lives in
``services.export.step_export``. This module keeps older imports working and
hosts the CFD half-wing helper that is still imported from ``services``.
"""

from __future__ import annotations

from OCC.Core.TopoDS import TopoDS_Shape

from core.occ_utils.shapes import (
    bool_fuse,
    loft_solid_from_wires,
    make_airfoil_wire_spline,
    make_compound,
    scale_shape as occ_scale_shape,
)
from core.state import Project
from services.export.geometry_builder import WingGeometryConfig
from services.export.step_export import (
    assemble_airframe,
    build_elevon_compound,
    build_full_step_from_processed,
    build_ribs_compound,
    build_spars_compound,
    build_step_from_project,
    loft_wing_from_ribs,
    write_step,
)

__all__ = [
    "WingGeometryConfig",
    "assemble_airframe",
    "build_cfd_wing_solid",
    "build_elevon_compound",
    "build_full_step_from_processed",
    "build_ribs_compound",
    "build_spars_compound",
    "build_step_from_project",
    "export_step_from_project",
    "loft_wing_from_ribs",
    "write_step",
]


def export_step_from_project(
    project: Project,
    output_path: str,
    config: WingGeometryConfig | None = None,
) -> bool:
    """Build and write a STEP file from native project state."""
    try:
        compound = build_step_from_project(project, config)
        if compound.IsNull():
            print("[StepExport] Error: No geometry generated")
            return False

        success = write_step(compound, output_path)
        if success:
            print(f"[StepExport] Successfully exported to {output_path}")
        else:
            print("[StepExport] Failed to write STEP file")
        return success
    except Exception as exc:
        print(f"[StepExport] Export failed: {exc}")
        return False


def build_cfd_wing_solid(
    project: Project,
    scale_to_mm: bool = True,
) -> TopoDS_Shape:
    """Build a solid half-wing outer mold line for CFD meshing."""
    import math

    from services.geometry import AeroSandboxService

    svc = AeroSandboxService(project)
    sections = svc.spanwise_sections()

    if len(sections) < 2:
        print("[CFD Wing] Error: Need at least 2 sections")
        return TopoDS_Shape()

    print(f"[CFD Wing] Building solid from {len(sections)} sections...")

    wires = []
    for section in sections:
        try:
            coords = section.airfoil.coordinates
            if coords is None or len(coords) < 3:
                raise ValueError("Invalid airfoil")
        except Exception as exc:
            print(f"[CFD Wing] Warning: Skipping section {section.index}: {exc}")
            continue

        twist_rad = math.radians(float(section.twist_deg))
        cos_twist = math.cos(twist_rad)
        sin_twist = math.sin(twist_rad)

        pts_3d = []
        for x_norm, z_norm in coords:
            x_local = float(x_norm) * section.chord_m
            z_local = float(z_norm) * section.chord_m
            x_twisted = x_local * cos_twist + z_local * sin_twist
            z_twisted = -x_local * sin_twist + z_local * cos_twist
            pts_3d.append(
                [
                    section.x_le_m + x_twisted,
                    section.y_m,
                    section.z_m + z_twisted,
                ]
            )

        wire = make_airfoil_wire_spline(pts_3d)
        if wire is not None:
            wires.append(wire)
            print(
                "[CFD Wing] Section "
                f"{section.index}: y={section.y_m:.3f}m, "
                f"chord={section.chord_m:.3f}m, twist={section.twist_deg:.2f} deg"
            )
        else:
            print(f"[CFD Wing] Warning: Could not create wire for section {section.index}")

    if len(wires) < 2:
        print("[CFD Wing] Error: Could not create enough section wires")
        return TopoDS_Shape()

    segments = []
    for i in range(len(wires) - 1):
        segment = loft_solid_from_wires(
            [wires[i], wires[i + 1]],
            continuity="C2",
            max_degree=8,
        )
        if segment and not segment.IsNull():
            segments.append(segment)
            print(f"[CFD Wing] Created loft segment {i} -> {i + 1}")
        else:
            print(f"[CFD Wing] Warning: Loft failed for segment {i} -> {i + 1}")

    if not segments:
        print("[CFD Wing] Error: No loft segments created")
        return TopoDS_Shape()

    result = segments[0]
    for idx, segment in enumerate(segments[1:], start=1):
        fused = bool_fuse(result, segment)
        if fused and not fused.IsNull():
            result = fused
        else:
            print(f"[CFD Wing] Warning: Fuse failed at segment {idx}, using compound fallback")
            result = make_compound([result, segment])

    if scale_to_mm:
        result = occ_scale_shape(result, 1000.0)
        print("[CFD Wing] Scaled to mm (1000x)")

    print("[CFD Wing] Successfully created half-wing solid")
    return result
