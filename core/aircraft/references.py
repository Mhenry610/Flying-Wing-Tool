from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


Vec3 = Tuple[float, float, float]


class Axis(str, Enum):
    X = "+X"
    NEG_X = "-X"
    Y = "+Y"
    NEG_Y = "-Y"
    Z = "+Z"
    NEG_Z = "-Z"


@dataclass
class AircraftReferenceFrame:
    """Aircraft-level reference quantities for force and moment aggregation."""

    reference_area_m2: float = 15.0
    reference_span_m: float = 10.0
    reference_chord_m: float = 1.5
    moment_reference_m: Vec3 = (0.0, 0.0, 0.0)
    cg_m: Vec3 = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "reference_area_m2": self.reference_area_m2,
            "reference_span_m": self.reference_span_m,
            "reference_chord_m": self.reference_chord_m,
            "moment_reference_m": list(self.moment_reference_m),
            "cg_m": list(self.cg_m),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AircraftReferenceFrame":
        data = data or {}
        return cls(
            reference_area_m2=float(data.get("reference_area_m2", 15.0)),
            reference_span_m=float(data.get("reference_span_m", 10.0)),
            reference_chord_m=float(data.get("reference_chord_m", 1.5)),
            moment_reference_m=_vec3(data.get("moment_reference_m", (0.0, 0.0, 0.0))),
            cg_m=_vec3(data.get("cg_m", data.get("moment_reference_m", (0.0, 0.0, 0.0)))),
        )


@dataclass
class SurfaceTransform:
    origin_m: Vec3 = (0.0, 0.0, 0.0)
    orientation_euler_deg: Vec3 = (0.0, 0.0, 0.0)
    parent_uid: str | None = None

    def as_dict(self) -> dict:
        return {
            "origin_m": list(self.origin_m),
            "orientation_euler_deg": list(self.orientation_euler_deg),
            "parent_uid": self.parent_uid,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "SurfaceTransform":
        data = data or {}
        return cls(
            origin_m=_vec3(data.get("origin_m", (0.0, 0.0, 0.0))),
            orientation_euler_deg=_vec3(data.get("orientation_euler_deg", (0.0, 0.0, 0.0))),
            parent_uid=data.get("parent_uid"),
        )


def _vec3(value: Iterable[float] | None) -> Vec3:
    values = list(value or (0.0, 0.0, 0.0))
    values = (values + [0.0, 0.0, 0.0])[:3]
    return (float(values[0]), float(values[1]), float(values[2]))

