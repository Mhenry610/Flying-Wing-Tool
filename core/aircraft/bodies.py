from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .mass import MassProperties
from .references import SurfaceTransform


@dataclass
class BodyTransform:
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "origin_m": list(self.origin_m),
            "orientation_euler_deg": list(self.orientation_euler_deg),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "BodyTransform":
        data = data or {}
        origin = data.get("origin_m", (0.0, 0.0, 0.0))
        orient = data.get("orientation_euler_deg", (0.0, 0.0, 0.0))
        return cls(
            origin_m=tuple(float(v) for v in (list(origin) + [0.0, 0.0, 0.0])[:3]),
            orientation_euler_deg=tuple(float(v) for v in (list(orient) + [0.0, 0.0, 0.0])[:3]),
        )


@dataclass
class BodyEnvelope:
    length_m: float = 0.0
    max_width_m: float = 0.0
    max_height_m: float = 0.0
    cross_sections: list[dict] = field(default_factory=list)

    def wetted_area_estimate_m2(self) -> float:
        # Elliptic-cylinder conceptual estimate; exact lofting is handled by future TiGL-backed work.
        if self.length_m <= 0.0:
            return 0.0
        perimeter = 3.141592653589793 * (1.5 * (self.max_width_m + self.max_height_m) - ((self.max_width_m * self.max_height_m) ** 0.5))
        return max(0.0, perimeter * self.length_m)

    def frontal_area_m2(self) -> float:
        return 3.141592653589793 * self.max_width_m * self.max_height_m / 4.0

    def as_dict(self) -> dict:
        return {
            "length_m": self.length_m,
            "max_width_m": self.max_width_m,
            "max_height_m": self.max_height_m,
            "cross_sections": list(self.cross_sections),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "BodyEnvelope | None":
        if data is None:
            return None
        return cls(
            length_m=float(data.get("length_m", 0.0)),
            max_width_m=float(data.get("max_width_m", 0.0)),
            max_height_m=float(data.get("max_height_m", 0.0)),
            cross_sections=list(data.get("cross_sections", [])),
        )


@dataclass
class AttachmentPoint:
    uid: str
    name: str = ""
    transform: SurfaceTransform = field(default_factory=SurfaceTransform)
    target_uid: str | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "transform": self.transform.as_dict(),
            "target_uid": self.target_uid,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttachmentPoint":
        return cls(
            uid=data["uid"],
            name=data.get("name", data["uid"]),
            transform=SurfaceTransform.from_dict(data.get("transform")),
            target_uid=data.get("target_uid"),
            notes=data.get("notes", ""),
        )


@dataclass
class BodyObject:
    uid: str
    name: str
    role: Literal["fuselage", "pod", "boom", "payload_bay", "placeholder"] = "placeholder"
    active: bool = True
    transform: BodyTransform = field(default_factory=BodyTransform)
    mass_properties: MassProperties | None = None
    drag_area_estimate_m2: float | None = None
    envelope: BodyEnvelope | None = None
    attachments: list[AttachmentPoint] = field(default_factory=list)
    external_refs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "role": self.role,
            "active": self.active,
            "transform": self.transform.as_dict(),
            "mass_properties": self.mass_properties.as_dict() if self.mass_properties else None,
            "drag_area_estimate_m2": self.drag_area_estimate_m2,
            "envelope": self.envelope.as_dict() if self.envelope else None,
            "attachments": [a.as_dict() for a in self.attachments],
            "external_refs": dict(self.external_refs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BodyObject":
        return cls(
            uid=data["uid"],
            name=data.get("name", data["uid"]),
            role=data.get("role", "placeholder"),
            active=bool(data.get("active", True)),
            transform=BodyTransform.from_dict(data.get("transform")),
            mass_properties=MassProperties.from_dict(data.get("mass_properties")),
            drag_area_estimate_m2=data.get("drag_area_estimate_m2"),
            envelope=BodyEnvelope.from_dict(data.get("envelope")),
            attachments=[AttachmentPoint.from_dict(a) for a in data.get("attachments", [])],
            external_refs=dict(data.get("external_refs", {})),
        )

