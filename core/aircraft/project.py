from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .bodies import BodyObject
from .mass import MassItem
from .references import AircraftReferenceFrame
from .surfaces import LiftingSurface


AIRCRAFT_SCHEMA_VERSION = 2


@dataclass
class ProjectMetadata:
    name: str = "AircraftProject"
    created_with: str = "FWT_V1-0"
    notes: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "created_with": self.created_with, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProjectMetadata":
        data = data or {}
        return cls(
            name=data.get("name", "AircraftProject"),
            created_with=data.get("created_with", "FWT_V1-0"),
            notes=data.get("notes", ""),
        )


@dataclass
class AircraftRequirements:
    payload_mass_kg: float = 0.0
    cruise_speed_mps: float | None = None
    stall_speed_mps: float | None = None
    endurance_min: float | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "payload_mass_kg": self.payload_mass_kg,
            "cruise_speed_mps": self.cruise_speed_mps,
            "stall_speed_mps": self.stall_speed_mps,
            "endurance_min": self.endurance_min,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AircraftRequirements":
        data = data or {}
        return cls(
            payload_mass_kg=float(data.get("payload_mass_kg", 0.0)),
            cruise_speed_mps=data.get("cruise_speed_mps"),
            stall_speed_mps=data.get("stall_speed_mps"),
            endurance_min=data.get("endurance_min"),
            notes=data.get("notes", ""),
        )


@dataclass
class AnalysisStore:
    results: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"results": dict(self.results), "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "AnalysisStore":
        data = data or {}
        return cls(results=dict(data.get("results", {})), warnings=list(data.get("warnings", [])))


@dataclass
class ExportSettings:
    cpacs_version_target: str = "3.5"
    last_export_path: str | None = None

    def as_dict(self) -> dict:
        return {"cpacs_version_target": self.cpacs_version_target, "last_export_path": self.last_export_path}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExportSettings":
        data = data or {}
        return cls(cpacs_version_target=data.get("cpacs_version_target", "3.5"), last_export_path=data.get("last_export_path"))


@dataclass
class ExternalReferenceStore:
    refs: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"refs": dict(self.refs)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExternalReferenceStore":
        data = data or {}
        return cls(refs=dict(data.get("refs", {})))


@dataclass
class AircraftProject:
    schema_version: int = AIRCRAFT_SCHEMA_VERSION
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    requirements: AircraftRequirements = field(default_factory=AircraftRequirements)
    reference: AircraftReferenceFrame = field(default_factory=AircraftReferenceFrame)
    surfaces: list[LiftingSurface] = field(default_factory=list)
    bodies: list[BodyObject] = field(default_factory=list)
    propulsion_systems: list[dict] = field(default_factory=list)
    mass_items: list[MassItem] = field(default_factory=list)
    mission: dict = field(default_factory=dict)
    analyses: AnalysisStore = field(default_factory=AnalysisStore)
    exports: ExportSettings = field(default_factory=ExportSettings)
    external_refs: ExternalReferenceStore = field(default_factory=ExternalReferenceStore)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.as_dict(),
            "requirements": self.requirements.as_dict(),
            "reference": self.reference.as_dict(),
            "surfaces": [surface.as_dict() for surface in self.surfaces],
            "bodies": [body.as_dict() for body in self.bodies],
            "propulsion_systems": list(self.propulsion_systems),
            "mass_items": [item.as_dict() for item in self.mass_items],
            "mission": dict(self.mission),
            "analyses": self.analyses.as_dict(),
            "exports": self.exports.as_dict(),
            "external_refs": self.external_refs.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AircraftProject":
        data = data or {}
        return cls(
            schema_version=int(data.get("schema_version", AIRCRAFT_SCHEMA_VERSION)),
            metadata=ProjectMetadata.from_dict(data.get("metadata")),
            requirements=AircraftRequirements.from_dict(data.get("requirements")),
            reference=AircraftReferenceFrame.from_dict(data.get("reference")),
            surfaces=[LiftingSurface.from_dict(s) for s in data.get("surfaces", [])],
            bodies=[BodyObject.from_dict(b) for b in data.get("bodies", [])],
            propulsion_systems=list(data.get("propulsion_systems", [])),
            mass_items=[MassItem.from_dict(item) for item in data.get("mass_items", [])],
            mission=dict(data.get("mission", {})),
            analyses=AnalysisStore.from_dict(data.get("analyses")),
            exports=ExportSettings.from_dict(data.get("exports")),
            external_refs=ExternalReferenceStore.from_dict(data.get("external_refs")),
        )

    @classmethod
    def from_legacy_project(cls, wing_project, mission=None, analysis=None) -> "AircraftProject":
        surface = LiftingSurface.from_legacy_wing(wing_project)
        reference = AircraftReferenceFrame(
            reference_area_m2=wing_project.planform.actual_area(),
            reference_span_m=wing_project.planform.actual_span(),
            reference_chord_m=wing_project.planform.mean_aerodynamic_chord(),
            moment_reference_m=(analysis.x_cg, 0.0, 0.0) if analysis and analysis.x_cg is not None else (0.25 * wing_project.planform.mean_aerodynamic_chord(), 0.0, 0.0),
            cg_m=(analysis.x_cg, 0.0, 0.0) if analysis and analysis.x_cg is not None else (0.25 * wing_project.planform.mean_aerodynamic_chord(), 0.0, 0.0),
        )
        mass_items = []
        if mission is not None and getattr(mission, "payload_mass_kg", 0.0) > 0.0:
            mass_items.append(MassItem("payload", "Payload", mission.payload_mass_kg, reference.cg_m, "payload"))
        return cls(
            metadata=ProjectMetadata(name=wing_project.name, notes=f"Migrated from legacy wing project on {datetime.utcnow().isoformat()}Z"),
            requirements=AircraftRequirements(payload_mass_kg=getattr(mission, "payload_mass_kg", 0.0) if mission else 0.0),
            reference=reference,
            surfaces=[surface],
            mass_items=mass_items,
            mission=mission.as_dict() if mission and hasattr(mission, "as_dict") else {},
            analyses=AnalysisStore(results=analysis.as_dict() if analysis and hasattr(analysis, "as_dict") else {}),
            external_refs=ExternalReferenceStore(refs={"legacy_wing": {"uid": "main_wing", "source": "Project.wing"}}),
        )

