from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MassProperties:
    mass_kg: float = 0.0
    cg_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kg_m2: tuple[float, float, float] | None = None

    def as_dict(self) -> dict:
        return {
            "mass_kg": self.mass_kg,
            "cg_m": list(self.cg_m),
            "inertia_kg_m2": list(self.inertia_kg_m2) if self.inertia_kg_m2 else None,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "MassProperties | None":
        if data is None:
            return None
        cg = data.get("cg_m", (0.0, 0.0, 0.0))
        inertia = data.get("inertia_kg_m2")
        return cls(
            mass_kg=float(data.get("mass_kg", 0.0)),
            cg_m=tuple(float(v) for v in (list(cg) + [0.0, 0.0, 0.0])[:3]),
            inertia_kg_m2=tuple(float(v) for v in inertia[:3]) if inertia else None,
        )


@dataclass
class MassItem:
    uid: str
    name: str
    mass_kg: float
    cg_m: tuple[float, float, float]
    category: Literal[
        "battery",
        "motor",
        "esc",
        "servo",
        "payload",
        "receiver",
        "autopilot",
        "structure",
        "fuselage",
        "landing_gear",
        "other",
    ] = "other"
    source_uid: str | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "mass_kg": self.mass_kg,
            "cg_m": list(self.cg_m),
            "category": self.category,
            "source_uid": self.source_uid,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MassItem":
        cg = data.get("cg_m", (0.0, 0.0, 0.0))
        return cls(
            uid=data["uid"],
            name=data.get("name", data["uid"]),
            mass_kg=float(data.get("mass_kg", 0.0)),
            cg_m=tuple(float(v) for v in (list(cg) + [0.0, 0.0, 0.0])[:3]),
            category=data.get("category", "other"),
            source_uid=data.get("source_uid"),
            notes=data.get("notes", ""),
        )


@dataclass
class MassBalance:
    total_mass_kg: float = 0.0
    cg_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    items: list[MassItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_mass_kg": self.total_mass_kg,
            "cg_m": list(self.cg_m),
            "items": [item.as_dict() for item in self.items],
            "warnings": list(self.warnings),
        }

