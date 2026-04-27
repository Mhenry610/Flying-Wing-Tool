from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from core.aircraft.bodies import BodyEnvelope, BodyObject
from core.aircraft.project import AircraftProject


@dataclass
class TrussGenerationSettings:
    body_uid: str | None = None
    truss_type: str = "Warren Truss"
    num_bays: int = 6
    inward_offset_m: float = 0.005
    profile_type: str = "Circular"
    tube_radius_m: float = 0.003
    member_width_m: float = 0.006
    member_height_m: float = 0.006

    def as_dict(self) -> dict:
        return {
            "body_uid": self.body_uid,
            "truss_type": self.truss_type,
            "num_bays": self.num_bays,
            "inward_offset_m": self.inward_offset_m,
            "profile_type": self.profile_type,
            "tube_radius_m": self.tube_radius_m,
            "member_width_m": self.member_width_m,
            "member_height_m": self.member_height_m,
        }


@dataclass
class TrussMember:
    start_index: int
    end_index: int
    member_type: str
    panel_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def as_dict(self) -> dict:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "member_type": self.member_type,
            "panel_normal": list(self.panel_normal),
        }


@dataclass
class TrussFrameworkResult:
    body_uid: str
    vertices_m: list[tuple[float, float, float]]
    members: list[TrussMember]
    settings: TrussGenerationSettings
    warnings: list[str] = field(default_factory=list)

    @property
    def total_member_length_m(self) -> float:
        total = 0.0
        for member in self.members:
            a = self.vertices_m[member.start_index]
            b = self.vertices_m[member.end_index]
            total += _distance(a, b)
        return total

    def as_dict(self) -> dict:
        return {
            "body_uid": self.body_uid,
            "vertices_m": [list(v) for v in self.vertices_m],
            "members": [m.as_dict() for m in self.members],
            "settings": self.settings.as_dict(),
            "total_member_length_m": self.total_member_length_m,
            "warnings": list(self.warnings),
        }


def generate_body_truss(
    project: AircraftProject,
    settings: TrussGenerationSettings,
) -> TrussFrameworkResult:
    body = _select_body(project, settings.body_uid)
    if body is None:
        raise ValueError("No active body is available for truss generation.")
    envelope = body.envelope or BodyEnvelope()
    if envelope.length_m <= 0.0:
        raise ValueError(f"Body '{body.uid}' has no positive length.")

    stations = _body_stations(body, settings)
    vertices: list[tuple[float, float, float]] = []
    for station in stations:
        vertices.extend(_rectangular_station_vertices(station, settings.inward_offset_m))

    members = _generate_members(vertices, settings.truss_type)
    warnings = []
    if settings.truss_type not in ("Warren Truss", "X Truss"):
        warnings.append(f"Unknown truss type '{settings.truss_type}' was treated as Warren Truss.")
    result = TrussFrameworkResult(body.uid, vertices, members, settings, warnings)
    project.analyses.results["truss_framework"] = result.as_dict()
    return result


def export_truss_step(result: TrussFrameworkResult | dict, output_path: str | Path) -> bool:
    """Export generated truss members as a STEP compound using pythonocc."""
    data = _result_dict(result)
    vertices = [tuple(float(v) for v in vertex[:3]) for vertex in data.get("vertices_m", [])]
    members = data.get("members", [])
    settings = data.get("settings", {})
    profile = str(settings.get("profile_type", "Circular"))

    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
        from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
        from OCC.Core.TopoDS import TopoDS_Compound
    except Exception as exc:
        raise ImportError(f"pythonocc-core is required for truss STEP export: {exc}") from exc

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for member in members:
        i = int(member.get("start_index", -1))
        j = int(member.get("end_index", -1))
        if i < 0 or j < 0 or i >= len(vertices) or j >= len(vertices):
            continue
        start = np.asarray(vertices[i], dtype=float)
        end = np.asarray(vertices[j], dtype=float)
        direction = end - start
        if np.linalg.norm(direction) < 1e-9:
            continue
        normal = member.get("panel_normal")
        if profile == "Rectangular":
            shape = _make_rectangular_occ_member(
                start,
                end,
                float(settings.get("member_width_m", 0.006)),
                float(settings.get("member_height_m", 0.006)),
                normal,
                BRepBuilderAPI_MakePolygon,
                BRepBuilderAPI_MakeFace,
                BRepPrimAPI_MakePrism,
                gp_Pnt,
                gp_Vec,
            )
        else:
            shape = _make_circular_occ_member(
                start,
                end,
                float(settings.get("tube_radius_m", 0.003)),
                normal,
                BRepPrimAPI_MakeCylinder,
                gp_Ax2,
                gp_Dir,
                gp_Pnt,
            )
        if shape is not None and not shape.IsNull():
            builder.Add(compound, shape)

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return writer.Write(str(output_path)) == IFSelect_RetDone


def truss_result_from_dict(data: dict) -> TrussFrameworkResult:
    settings = TrussGenerationSettings(**{k: v for k, v in data.get("settings", {}).items() if k in TrussGenerationSettings.__dataclass_fields__})
    members = [
        TrussMember(
            int(m.get("start_index", 0)),
            int(m.get("end_index", 0)),
            str(m.get("member_type", "member")),
            tuple(float(v) for v in (m.get("panel_normal") or (0.0, 0.0, 1.0))[:3]),
        )
        for m in data.get("members", [])
    ]
    vertices = [tuple(float(v) for v in vertex[:3]) for vertex in data.get("vertices_m", [])]
    return TrussFrameworkResult(str(data.get("body_uid", "")), vertices, members, settings, list(data.get("warnings", [])))


def _select_body(project: AircraftProject, uid: str | None) -> BodyObject | None:
    active_bodies = [body for body in project.bodies if body.active]
    if uid:
        for body in active_bodies:
            if body.uid == uid:
                return body
    return active_bodies[0] if active_bodies else None


def _body_stations(body: BodyObject, settings: TrussGenerationSettings) -> list[dict[str, float]]:
    envelope = body.envelope or BodyEnvelope()
    x0, y0, z0 = body.transform.origin_m
    raw_sections = sorted(envelope.cross_sections, key=lambda item: float(item.get("x_m", item.get("eta", 0.0))))
    if raw_sections:
        stations = []
        for section in raw_sections:
            eta = float(section.get("eta", 0.0))
            local_x = float(section.get("x_m", eta * envelope.length_m))
            stations.append(
                {
                    "x": x0 + local_x,
                    "y": y0 + float(section.get("centroid_y_m", 0.0)),
                    "z": z0 + float(section.get("centroid_z_m", 0.0)),
                    "width": float(section.get("width_m", envelope.max_width_m)),
                    "height": float(section.get("height_m", envelope.max_height_m)),
                }
            )
        if len(stations) >= 2:
            return stations

    num_bays = max(1, int(settings.num_bays))
    stations = []
    for idx in range(num_bays + 1):
        eta = idx / num_bays
        stations.append(
            {
                "x": x0 + eta * envelope.length_m,
                "y": y0,
                "z": z0,
                "width": envelope.max_width_m,
                "height": envelope.max_height_m,
            }
        )
    return stations


def _rectangular_station_vertices(station: dict[str, float], inward_offset_m: float) -> list[tuple[float, float, float]]:
    half_w = max(0.0, 0.5 * station["width"] - inward_offset_m)
    half_h = max(0.0, 0.5 * station["height"] - inward_offset_m)
    x, y, z = station["x"], station["y"], station["z"]
    return [
        (x, y + half_w, z + half_h),
        (x, y - half_w, z + half_h),
        (x, y - half_w, z - half_h),
        (x, y + half_w, z - half_h),
    ]


def _generate_members(vertices: list[tuple[float, float, float]], truss_type: str) -> list[TrussMember]:
    n_per = 4
    station_count = len(vertices) // n_per
    members_by_edge: dict[tuple[int, int], TrussMember] = {}
    truss = truss_type if truss_type in ("Warren Truss", "X Truss") else "Warren Truss"
    vertices_np = np.asarray(vertices, dtype=float)

    def add(i: int, j: int, member_type: str, normal=(0.0, 0.0, 1.0)) -> None:
        edge = tuple(sorted((i, j)))
        if edge[0] == edge[1] or edge in members_by_edge:
            return
        members_by_edge[edge] = TrussMember(edge[0], edge[1], member_type, tuple(float(v) for v in normal))

    for bay in range(max(0, station_count - 1)):
        current = bay * n_per
        nxt = (bay + 1) * n_per
        bay_indices = [current + k for k in range(n_per)] + [nxt + k for k in range(n_per)]
        bay_centroid = np.mean(vertices_np[bay_indices], axis=0)

        for face in range(n_per):
            p0 = current + face
            p1 = nxt + face
            p2 = nxt + ((face + 1) % n_per)
            p3 = current + ((face + 1) % n_per)
            normal = _panel_normal(vertices_np[p0], vertices_np[p1], vertices_np[p2], vertices_np[p3], bay_centroid)
            add(p0, p1, "longeron", normal)
            if truss == "X Truss":
                add(p0, p2, "diagonal", normal)
                add(p3, p1, "diagonal", normal)
            elif (bay + face) % 2 == 0:
                add(p0, p2, "diagonal", normal)
            else:
                add(p3, p1, "diagonal", normal)

        for station_base in (current, nxt):
            for edge in range(n_per):
                add(station_base + edge, station_base + ((edge + 1) % n_per), "frame")

    return list(members_by_edge.values())


def _panel_normal(p0, p1, p2, p3, bay_centroid) -> tuple[float, float, float]:
    normal = np.cross(p1 - p0, p3 - p0)
    length = np.linalg.norm(normal)
    if length < 1e-9:
        normal = np.cross(p2 - p1, p0 - p1)
        length = np.linalg.norm(normal)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    normal = normal / length
    panel_centroid = (p0 + p1 + p2 + p3) / 4.0
    if np.dot(normal, panel_centroid - bay_centroid) < 0.0:
        normal = -normal
    return tuple(float(v) for v in normal)


def _basis_from_direction(direction, alignment_vector=None):
    z_axis = _unit(direction)
    up = None
    align = None if alignment_vector is None else np.asarray(alignment_vector, dtype=float)
    if align is not None and align.size >= 3:
        align = align[:3] - z_axis * float(np.dot(align[:3], z_axis))
        if np.linalg.norm(align) > 1e-6:
            up = _unit(align)
    if up is None:
        for candidate in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
            candidate = candidate - z_axis * float(np.dot(candidate, z_axis))
            if np.linalg.norm(candidate) > 1e-6:
                up = _unit(candidate)
                break
    x_axis = _unit(np.cross(up, z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def _make_rectangular_occ_member(start, end, width, height, alignment, make_polygon, make_face, make_prism, gp_pnt, gp_vec):
    direction = end - start
    x_axis, y_axis, _ = _basis_from_direction(direction, alignment)
    corners = [
        start - 0.5 * width * x_axis - 0.5 * height * y_axis,
        start + 0.5 * width * x_axis - 0.5 * height * y_axis,
        start + 0.5 * width * x_axis + 0.5 * height * y_axis,
        start - 0.5 * width * x_axis + 0.5 * height * y_axis,
    ]
    polygon = make_polygon()
    for corner in corners:
        polygon.Add(gp_pnt(*corner))
    polygon.Close()
    if not polygon.IsDone():
        return None
    face = make_face(polygon.Wire())
    if not face.IsDone():
        return None
    return make_prism(face.Face(), gp_vec(*direction)).Shape()


def _make_circular_occ_member(start, end, radius, alignment, make_cylinder, gp_ax2, gp_dir, gp_pnt):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return None
    x_axis, _, z_axis = _basis_from_direction(direction, alignment)
    axis = gp_ax2(gp_pnt(*start), gp_dir(*z_axis), gp_dir(*x_axis))
    return make_cylinder(axis, radius, length).Shape()


def _result_dict(result: TrussFrameworkResult | dict) -> dict:
    return result.as_dict() if hasattr(result, "as_dict") else dict(result)


def _unit(vector) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(vector, dtype=float) / length


def _distance(a, b) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
