from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin, tan
from typing import Literal

from core.aircraft.references import SurfaceTransform
from core.aircraft.surfaces import LiftingSurface, SymmetryMode


@dataclass
class ExpandedSurfaceGeometry:
    area_m2: float
    span_m: float
    half_span_m: float
    root_chord_m: float
    tip_chord_m: float
    mean_aerodynamic_chord_m: float
    aerodynamic_center_m: tuple[float, float, float]
    leading_edge_points_m: list[tuple[float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "area_m2": self.area_m2,
            "span_m": self.span_m,
            "half_span_m": self.half_span_m,
            "root_chord_m": self.root_chord_m,
            "tip_chord_m": self.tip_chord_m,
            "mean_aerodynamic_chord_m": self.mean_aerodynamic_chord_m,
            "aerodynamic_center_m": list(self.aerodynamic_center_m),
            "leading_edge_points_m": [list(p) for p in self.leading_edge_points_m],
        }


@dataclass
class SurfaceInstance:
    source_surface_uid: str
    instance_uid: str
    side: Literal["left", "right", "center", "explicit"]
    transform: SurfaceTransform
    expanded_geometry: ExpandedSurfaceGeometry

    def as_dict(self) -> dict:
        return {
            "source_surface_uid": self.source_surface_uid,
            "instance_uid": self.instance_uid,
            "side": self.side,
            "transform": self.transform.as_dict(),
            "expanded_geometry": self.expanded_geometry.as_dict(),
        }


def assemble_surface_instances(surfaces: list[LiftingSurface]) -> list[SurfaceInstance]:
    instances: list[SurfaceInstance] = []
    for surface in surfaces:
        if not surface.active:
            continue
        symmetry = _enum_value(surface.symmetry)
        if symmetry == SymmetryMode.MIRRORED_ABOUT_XZ.value:
            instances.append(_make_instance(surface, "right", mirror_y=False))
            instances.append(_make_instance(surface, "left", mirror_y=True))
        elif symmetry == SymmetryMode.PAIRED_EXPLICIT.value:
            instances.append(_make_instance(surface, "explicit", mirror_y=False))
        else:
            instances.append(_make_instance(surface, "center", mirror_y=False))
    return instances


def surface_reference_totals(surfaces: list[LiftingSurface]) -> dict:
    instances = assemble_surface_instances(surfaces)
    area = sum(inst.expanded_geometry.area_m2 for inst in instances)
    span = max((inst.expanded_geometry.span_m for inst in instances), default=0.0)
    mac_weight = sum(inst.expanded_geometry.mean_aerodynamic_chord_m * inst.expanded_geometry.area_m2 for inst in instances)
    mac = mac_weight / area if area > 0.0 else 0.0
    return {"area_m2": area, "span_m": span, "mean_aerodynamic_chord_m": mac, "instances": instances}


def _make_instance(surface: LiftingSurface, side: str, mirror_y: bool) -> SurfaceInstance:
    transform = _mirrored_transform(surface.transform) if mirror_y else surface.transform
    geom = _expand_geometry(surface, transform, mirror_y)
    return SurfaceInstance(
        source_surface_uid=surface.uid,
        instance_uid=f"{surface.uid}_{side}",
        side=side,
        transform=transform,
        expanded_geometry=geom,
    )


def _expand_geometry(surface: LiftingSurface, transform: SurfaceTransform, mirror_y: bool) -> ExpandedSurfaceGeometry:
    pf = surface.planform
    full_area = max(0.0, float(pf.actual_area()))
    full_span = max(0.0, float(pf.actual_span()))
    half_span = max(0.0, full_span / 2.0)
    root = float(pf.root_chord())
    tip = float(pf.tip_chord())
    mac = float(pf.mean_aerodynamic_chord())
    single_side = _enum_value(surface.symmetry) == SymmetryMode.MIRRORED_ABOUT_XZ.value
    area = full_area / 2.0 if single_side else full_area
    span_for_instance = half_span if single_side else full_span

    sweep_dx = tan(radians(float(pf.sweep_le_deg))) * span_for_instance
    dihedral_dz = tan(radians(float(pf.dihedral_deg))) * span_for_instance
    span_axis = _enum_value(surface.local_span_axis)
    sign = -1.0 if mirror_y else 1.0
    root_le = transform.origin_m
    tip_local = _span_vector(span_axis, sign * span_for_instance, sweep_dx, dihedral_dz)
    tip_le = _add(root_le, _rotate_xyz(tip_local, transform.orientation_euler_deg))
    ac_local_span = _span_vector(span_axis, sign * 0.5 * span_for_instance, sweep_dx * 0.5, dihedral_dz * 0.5)
    ac = _add(root_le, _rotate_xyz((ac_local_span[0] + 0.25 * mac, ac_local_span[1], ac_local_span[2]), transform.orientation_euler_deg))
    return ExpandedSurfaceGeometry(
        area_m2=area,
        span_m=span_for_instance,
        half_span_m=span_for_instance,
        root_chord_m=root,
        tip_chord_m=tip,
        mean_aerodynamic_chord_m=mac,
        aerodynamic_center_m=ac,
        leading_edge_points_m=[root_le, tip_le],
    )


def _span_vector(axis: str, span: float, sweep_dx: float, dihedral_dz: float) -> tuple[float, float, float]:
    if axis in ("+Z", "-Z"):
        z = span if axis == "+Z" else -span
        return (sweep_dx, 0.0, z)
    if axis in ("+X", "-X"):
        x = span if axis == "+X" else -span
        return (x, 0.0, dihedral_dz)
    y = span if axis == "+Y" else -span
    return (sweep_dx, y, dihedral_dz)


def _mirrored_transform(transform: SurfaceTransform) -> SurfaceTransform:
    ox, oy, oz = transform.origin_m
    rx, ry, rz = transform.orientation_euler_deg
    return SurfaceTransform(origin_m=(ox, -oy, oz), orientation_euler_deg=(rx, -ry, -rz), parent_uid=transform.parent_uid)


def _rotate_xyz(point: tuple[float, float, float], euler_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    rx, ry, rz = (radians(v) for v in euler_deg)
    cy, sy = cos(rx), sin(rx)
    y, z = y * cy - z * sy, y * sy + z * cy
    cx, sx = cos(ry), sin(ry)
    x, z = x * cx + z * sx, -x * sx + z * cx
    cz, sz = cos(rz), sin(rz)
    x, y = x * cz - y * sz, x * sz + y * cz
    return (x, y, z)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)

