from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi
from typing import Literal


class StructuralElementType(str, Enum):
    WINGBOX = "wingbox"
    ROUND_SPAR = "round_spar"
    TUBE_SPAR = "tube_spar"
    SPAR_CAP = "spar_cap"
    SPAR_WEB = "spar_web"
    SHEAR_WEB = "shear_web"
    RIB = "rib"
    FORMER = "former"
    SKIN_PANEL = "skin_panel"
    STRINGER = "stringer"
    LEADING_EDGE_D_TUBE = "leading_edge_d_tube"
    TRAILING_EDGE_MEMBER = "trailing_edge_member"
    STRUT = "strut"
    WIRE = "wire"
    HARDPOINT_REINFORCEMENT = "hardpoint_reinforcement"
    CUSTOM_BEAM = "custom_beam"


@dataclass
class StructuralLocation:
    eta: float | None = None
    chord_fraction: float | None = None
    z_offset_m: float = 0.0
    aircraft_xyz_m: tuple[float, float, float] | None = None
    hardpoint_uid: str | None = None

    def as_dict(self) -> dict:
        return {
            "eta": self.eta,
            "chord_fraction": self.chord_fraction,
            "z_offset_m": self.z_offset_m,
            "aircraft_xyz_m": list(self.aircraft_xyz_m) if self.aircraft_xyz_m else None,
            "hardpoint_uid": self.hardpoint_uid,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "StructuralLocation":
        data = data or {}
        xyz = data.get("aircraft_xyz_m")
        return cls(
            eta=data.get("eta"),
            chord_fraction=data.get("chord_fraction"),
            z_offset_m=float(data.get("z_offset_m", 0.0)),
            aircraft_xyz_m=tuple(float(v) for v in xyz[:3]) if xyz else None,
            hardpoint_uid=data.get("hardpoint_uid"),
        )


@dataclass
class StructuralSection:
    shape: Literal["rectangular", "solid_round", "tube", "sheet", "custom"] = "rectangular"
    width_mm: float | None = None
    height_mm: float | None = None
    thickness_mm: float | None = None
    outer_diameter_mm: float | None = None
    wall_thickness_mm: float | None = None
    area_m2: float | None = None
    ixx_m4: float | None = None
    iyy_m4: float | None = None
    j_m4: float | None = None

    def cross_section_area_m2(self) -> float:
        if self.area_m2 is not None:
            return max(0.0, float(self.area_m2))
        if self.shape == "solid_round":
            d = _mm(self.outer_diameter_mm)
            return pi * d * d / 4.0
        if self.shape == "tube":
            od = _mm(self.outer_diameter_mm)
            wt = _mm(self.wall_thickness_mm)
            inner = max(0.0, od - 2.0 * wt)
            return pi * (od * od - inner * inner) / 4.0
        if self.shape == "sheet":
            panel_width = self.width_mm if self.width_mm is not None else self.height_mm
            return _mm(panel_width) * _mm(self.thickness_mm)
        return _mm(self.width_mm) * _mm(self.height_mm)

    def bending_ixx_m4(self) -> float:
        if self.ixx_m4 is not None:
            return max(0.0, float(self.ixx_m4))
        if self.shape in ("solid_round", "tube"):
            od = _mm(self.outer_diameter_mm)
            inner = max(0.0, od - 2.0 * _mm(self.wall_thickness_mm))
            return pi * (od**4 - inner**4) / 64.0
        b = _mm(self.width_mm)
        h = _mm(self.height_mm or self.thickness_mm)
        return b * h**3 / 12.0

    def torsion_j_m4(self) -> float:
        if self.j_m4 is not None:
            return max(0.0, float(self.j_m4))
        if self.shape in ("solid_round", "tube"):
            od = _mm(self.outer_diameter_mm)
            inner = max(0.0, od - 2.0 * _mm(self.wall_thickness_mm))
            return pi * (od**4 - inner**4) / 32.0
        return self.bending_ixx_m4()

    def as_dict(self) -> dict:
        return {
            "shape": self.shape,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "thickness_mm": self.thickness_mm,
            "outer_diameter_mm": self.outer_diameter_mm,
            "wall_thickness_mm": self.wall_thickness_mm,
            "area_m2": self.area_m2,
            "ixx_m4": self.ixx_m4,
            "iyy_m4": self.iyy_m4,
            "j_m4": self.j_m4,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "StructuralSection":
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Hardpoint:
    uid: str
    name: str = ""
    location: StructuralLocation = field(default_factory=StructuralLocation)
    allowable_load_n: float | None = None

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "location": self.location.as_dict(),
            "allowable_load_n": self.allowable_load_n,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hardpoint":
        return cls(
            uid=data["uid"],
            name=data.get("name", ""),
            location=StructuralLocation.from_dict(data.get("location")),
            allowable_load_n=data.get("allowable_load_n"),
        )


@dataclass
class StructuralJoint:
    uid: str
    connected_uids: list[str] = field(default_factory=list)
    joint_type: str = "fixed"

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "connected_uids": list(self.connected_uids),
            "joint_type": self.joint_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuralJoint":
        return cls(data["uid"], list(data.get("connected_uids", [])), data.get("joint_type", "fixed"))


@dataclass
class LoadCase:
    uid: str
    name: str = "Design load"
    load_factor_g: float = 1.0
    dynamic_pressure_pa: float | None = None

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "load_factor_g": self.load_factor_g,
            "dynamic_pressure_pa": self.dynamic_pressure_pa,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadCase":
        return cls(
            uid=data["uid"],
            name=data.get("name", "Design load"),
            load_factor_g=float(data.get("load_factor_g", 1.0)),
            dynamic_pressure_pa=data.get("dynamic_pressure_pa"),
        )


@dataclass
class StructuralElement:
    uid: str
    type: StructuralElementType | str
    material_uid: str = "default"
    start: StructuralLocation = field(default_factory=StructuralLocation)
    end: StructuralLocation = field(default_factory=StructuralLocation)
    section: StructuralSection = field(default_factory=StructuralSection)
    connection_uids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "type": _enum_value(self.type),
            "material_uid": self.material_uid,
            "start": self.start.as_dict(),
            "end": self.end.as_dict(),
            "section": self.section.as_dict(),
            "connection_uids": list(self.connection_uids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuralElement":
        return cls(
            uid=data["uid"],
            type=data.get("type", StructuralElementType.CUSTOM_BEAM.value),
            material_uid=data.get("material_uid", "default"),
            start=StructuralLocation.from_dict(data.get("start")),
            end=StructuralLocation.from_dict(data.get("end")),
            section=StructuralSection.from_dict(data.get("section")),
            connection_uids=list(data.get("connection_uids", [])),
        )


@dataclass
class StructuralAnalysisSettings:
    fidelity: Literal["conceptual", "legacy_wingbox", "external"] = "conceptual"
    include_buckling: bool = False
    warnings_enabled: bool = True

    def as_dict(self) -> dict:
        return {
            "fidelity": self.fidelity,
            "include_buckling": self.include_buckling,
            "warnings_enabled": self.warnings_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "StructuralAnalysisSettings":
        data = data or {}
        return cls(
            fidelity=data.get("fidelity", "conceptual"),
            include_buckling=bool(data.get("include_buckling", False)),
            warnings_enabled=bool(data.get("warnings_enabled", True)),
        )


@dataclass
class StructuralLayout:
    uid: str = "structure"
    coordinate_system: Literal["surface_local", "aircraft"] = "surface_local"
    elements: list[StructuralElement] = field(default_factory=list)
    joints: list[StructuralJoint] = field(default_factory=list)
    hardpoints: list[Hardpoint] = field(default_factory=list)
    load_cases: list[LoadCase] = field(default_factory=list)
    analysis_settings: StructuralAnalysisSettings = field(default_factory=StructuralAnalysisSettings)

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "coordinate_system": self.coordinate_system,
            "elements": [e.as_dict() for e in self.elements],
            "joints": [j.as_dict() for j in self.joints],
            "hardpoints": [h.as_dict() for h in self.hardpoints],
            "load_cases": [lc.as_dict() for lc in self.load_cases],
            "analysis_settings": self.analysis_settings.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "StructuralLayout":
        data = data or {}
        return cls(
            uid=data.get("uid", "structure"),
            coordinate_system=data.get("coordinate_system", "surface_local"),
            elements=[StructuralElement.from_dict(e) for e in data.get("elements", [])],
            joints=[StructuralJoint.from_dict(j) for j in data.get("joints", [])],
            hardpoints=[Hardpoint.from_dict(h) for h in data.get("hardpoints", [])],
            load_cases=[LoadCase.from_dict(lc) for lc in data.get("load_cases", [])],
            analysis_settings=StructuralAnalysisSettings.from_dict(data.get("analysis_settings")),
        )

    @classmethod
    def from_legacy_wingbox(cls, uid: str, planform) -> "StructuralLayout":
        return cls(
            uid=uid,
            coordinate_system="surface_local",
            elements=[
                StructuralElement(
                    uid=f"{uid}_front_spar",
                    type=StructuralElementType.SPAR_WEB,
                    material_uid=getattr(planform, "spar_material_name", "spar"),
                    start=StructuralLocation(eta=0.0, chord_fraction=getattr(planform, "front_spar_root_percent", 20.0) / 100.0),
                    end=StructuralLocation(eta=1.0, chord_fraction=getattr(planform, "front_spar_tip_percent", 20.0) / 100.0),
                    section=StructuralSection(shape="sheet", thickness_mm=getattr(planform, "spar_thickness_mm", 3.0), height_mm=20.0),
                ),
                StructuralElement(
                    uid=f"{uid}_rear_spar",
                    type=StructuralElementType.SPAR_WEB,
                    material_uid=getattr(planform, "spar_material_name", "spar"),
                    start=StructuralLocation(eta=0.0, chord_fraction=getattr(planform, "rear_spar_root_percent", 65.0) / 100.0),
                    end=StructuralLocation(eta=getattr(planform, "rear_spar_span_percent", 100.0) / 100.0, chord_fraction=getattr(planform, "rear_spar_tip_percent", 65.0) / 100.0),
                    section=StructuralSection(shape="sheet", thickness_mm=getattr(planform, "spar_thickness_mm", 3.0), height_mm=20.0),
                ),
                StructuralElement(
                    uid=f"{uid}_skin",
                    type=StructuralElementType.SKIN_PANEL,
                    material_uid=getattr(planform, "skin_material_name", "skin"),
                    section=StructuralSection(shape="sheet", thickness_mm=getattr(planform, "skin_thickness_mm", 1.5)),
                ),
            ],
            load_cases=[LoadCase(uid=f"{uid}_limit", name="Legacy limit load", load_factor_g=1.0)],
            analysis_settings=StructuralAnalysisSettings(fidelity="legacy_wingbox", include_buckling=True),
        )


def _mm(value: float | None) -> float:
    return float(value or 0.0) / 1000.0


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)
