from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import xml.etree.ElementTree as ET

from core.aircraft.bodies import BodyEnvelope, BodyObject
from core.aircraft.project import AircraftProject, ExternalReferenceStore
from core.aircraft.references import SurfaceTransform
from core.aircraft.surfaces import LiftingSurface, SurfaceRole, SymmetryMode
from core.models.airfoil import AirfoilInterpolation
from core.models.planform import PlanformGeometry


CPACS_REFERENCES = [
    {"label": "CPACS documentation", "url": "https://dlr-sl.github.io/cpacs-website/pages/documentation.html"},
    {"label": "TiGL documentation", "url": "https://dlr-sc.github.io/tigl/pages/documentation.html"},
]


@dataclass
class CPACSDiagnostic:
    level: str
    message: str
    object_uid: str | None = None

    def as_dict(self) -> dict:
        return {"level": self.level, "message": self.message, "object_uid": self.object_uid}


@dataclass
class CPACSUIDMapEntry:
    internal_uid: str
    cpacs_uid: str
    cpacs_xpath: str
    last_sync_direction: str
    last_sync_timestamp: str
    sync_status: str = "synced"
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "internal_uid": self.internal_uid,
            "cpacs_uid": self.cpacs_uid,
            "cpacs_xpath": self.cpacs_xpath,
            "last_sync_direction": self.last_sync_direction,
            "last_sync_timestamp": self.last_sync_timestamp,
            "sync_status": self.sync_status,
            "warnings": list(self.warnings),
        }


@dataclass
class CPACSExportResult:
    xml_text: str
    uid_map: list[CPACSUIDMapEntry]
    diagnostics: list[CPACSDiagnostic]
    references: list[dict] = field(default_factory=lambda: list(CPACS_REFERENCES))

    def as_dict(self) -> dict:
        return {
            "xml_text": self.xml_text,
            "uid_map": [m.as_dict() for m in self.uid_map],
            "diagnostics": [d.as_dict() for d in self.diagnostics],
            "references": list(self.references),
        }


@dataclass
class CPACSImportResult:
    aircraft: AircraftProject
    uid_map: list[CPACSUIDMapEntry]
    diagnostics: list[CPACSDiagnostic]
    references: list[dict] = field(default_factory=lambda: list(CPACS_REFERENCES))


class CPACSAdapter:
    """Supported-subset CPACS adapter.

    Internal AircraftProject data remains authoritative. Unsupported CPACS objects are
    reported through diagnostics instead of being silently discarded.
    """

    def export_project(self, project: AircraftProject) -> CPACSExportResult:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        diagnostics: list[CPACSDiagnostic] = []
        uid_map: list[CPACSUIDMapEntry] = []
        cpacs = ET.Element("cpacs")
        cpacs.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        header = ET.SubElement(cpacs, "header")
        ET.SubElement(header, "name").text = project.metadata.name
        ET.SubElement(header, "cpacsVersion").text = project.exports.cpacs_version_target
        ET.SubElement(header, "creator").text = "FWT_V1-0 AircraftProject CPACSAdapter"
        ET.SubElement(header, "timestamp").text = timestamp
        vehicles = ET.SubElement(cpacs, "vehicles")
        aircraft = ET.SubElement(vehicles, "aircraft")
        model = ET.SubElement(aircraft, "model", uID=_uid(project.metadata.name or "aircraft"))
        ET.SubElement(model, "name").text = project.metadata.name
        wings = ET.SubElement(model, "wings")
        fuselages = ET.SubElement(model, "fuselages")

        for idx, surface in enumerate(project.surfaces, start=1):
            wing_uid = _uid(surface.uid)
            wing = ET.SubElement(wings, "wing", uID=wing_uid)
            wing.set("symmetry", _cpacs_symmetry(surface.symmetry))
            ET.SubElement(wing, "name").text = surface.name
            _write_transformation(wing, surface.transform, surface.incidence_deg)
            _write_wing_sections(wing, surface)
            uid_map.append(CPACSUIDMapEntry(surface.uid, wing_uid, f"/cpacs/vehicles/aircraft/model/wings/wing[{idx}]", "export", timestamp))

        for idx, body in enumerate(project.bodies, start=1):
            if body.role not in ("fuselage", "pod", "boom", "payload_bay", "placeholder"):
                diagnostics.append(CPACSDiagnostic("warning", f"Unsupported body role exported as generic fuselage: {body.role}", body.uid))
            fus_uid = _uid(body.uid)
            fus = ET.SubElement(fuselages, "fuselage", uID=fus_uid)
            ET.SubElement(fus, "name").text = body.name
            if body.envelope:
                ET.SubElement(fus, "length").text = str(body.envelope.length_m)
                ET.SubElement(fus, "maxWidth").text = str(body.envelope.max_width_m)
                ET.SubElement(fus, "maxHeight").text = str(body.envelope.max_height_m)
            uid_map.append(CPACSUIDMapEntry(body.uid, fus_uid, f"/cpacs/vehicles/aircraft/model/fuselages/fuselage[{idx}]", "export", timestamp))

        _record_uid_map(project, uid_map)
        xml_text = ET.tostring(cpacs, encoding="unicode")
        return CPACSExportResult(xml_text=xml_text, uid_map=uid_map, diagnostics=diagnostics)

    def import_project(self, xml_text: str) -> CPACSImportResult:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        diagnostics: list[CPACSDiagnostic] = []
        uid_map: list[CPACSUIDMapEntry] = []
        root = ET.fromstring(xml_text)
        name = root.findtext("./header/name") or "CPACS Import"
        project = AircraftProject()
        project.metadata.name = name
        project.surfaces = []
        project.bodies = []
        for idx, wing in enumerate(root.findall(".//wings/wing"), start=1):
            uid = wing.attrib.get("uID", f"wing_{idx}")
            surface = _surface_from_cpacs_wing(wing)
            project.surfaces.append(surface)
            uid_map.append(CPACSUIDMapEntry(surface.uid, uid, f"/cpacs/vehicles/aircraft/model/wings/wing[{idx}]", "import", timestamp))
        for idx, fus in enumerate(root.findall(".//fuselages/fuselage"), start=1):
            uid = fus.attrib.get("uID", f"fuselage_{idx}")
            envelope = BodyEnvelope(
                length_m=float(fus.findtext("length") or 0.0),
                max_width_m=float(fus.findtext("maxWidth") or 0.0),
                max_height_m=float(fus.findtext("maxHeight") or 0.0),
            )
            project.bodies.append(BodyObject(uid=uid, name=fus.findtext("name") or uid, role="fuselage", envelope=envelope))
            uid_map.append(CPACSUIDMapEntry(uid, uid, f"/cpacs/vehicles/aircraft/model/fuselages/fuselage[{idx}]", "import", timestamp))
        if not project.surfaces:
            diagnostics.append(CPACSDiagnostic("warning", "No supported CPACS wings were found."))
        _record_uid_map(project, uid_map)
        return CPACSImportResult(aircraft=project, uid_map=uid_map, diagnostics=diagnostics)


def _write_transformation(parent: ET.Element, transform: SurfaceTransform, incidence_deg: float) -> None:
    tr = ET.SubElement(parent, "transformation")
    tx = ET.SubElement(tr, "translation")
    ET.SubElement(tx, "x").text = str(transform.origin_m[0])
    ET.SubElement(tx, "y").text = str(transform.origin_m[1])
    ET.SubElement(tx, "z").text = str(transform.origin_m[2])
    rot = ET.SubElement(tr, "rotation")
    ET.SubElement(rot, "x").text = str(transform.orientation_euler_deg[0])
    ET.SubElement(rot, "y").text = str(transform.orientation_euler_deg[1] + incidence_deg)
    ET.SubElement(rot, "z").text = str(transform.orientation_euler_deg[2])


def _write_wing_sections(wing: ET.Element, surface: LiftingSurface) -> None:
    sections = ET.SubElement(wing, "sections")
    pf = surface.planform
    for idx, eta in enumerate((0.0, 1.0), start=1):
        section = ET.SubElement(sections, "section", uID=f"{_uid(surface.uid)}_sec_{idx}")
        ET.SubElement(section, "name").text = f"{surface.name} section {idx}"
        trans = ET.SubElement(ET.SubElement(section, "transformation"), "translation")
        y = pf.half_span() * eta
        x = y * __import__("math").tan(__import__("math").radians(pf.sweep_le_deg))
        z = y * __import__("math").tan(__import__("math").radians(pf.dihedral_deg))
        ET.SubElement(trans, "x").text = str(x)
        ET.SubElement(trans, "y").text = str(y)
        ET.SubElement(trans, "z").text = str(z)
        elements = ET.SubElement(section, "elements")
        el = ET.SubElement(elements, "element", uID=f"{_uid(surface.uid)}_el_{idx}")
        ET.SubElement(el, "name").text = f"{surface.name} element {idx}"
        scaling = ET.SubElement(ET.SubElement(el, "transformation"), "scaling")
        chord = pf.root_chord() if eta == 0.0 else pf.tip_chord()
        ET.SubElement(scaling, "x").text = str(chord)
        ET.SubElement(scaling, "y").text = "1.0"
        ET.SubElement(scaling, "z").text = str(chord)


def _surface_from_cpacs_wing(wing: ET.Element) -> LiftingSurface:
    uid = wing.attrib.get("uID", "cpacs_wing")
    name = wing.findtext("name") or uid
    sections = wing.findall("./sections/section")
    root_chord = _section_chord(sections[0]) if sections else 1.0
    tip_chord = _section_chord(sections[-1]) if len(sections) > 1 else root_chord
    span = _section_y(sections[-1]) if len(sections) > 1 else 1.0
    area = (root_chord + tip_chord) * max(span, 1e-6)
    taper = tip_chord / root_chord if root_chord > 0 else 1.0
    planform = PlanformGeometry(wing_area_m2=area, aspect_ratio=(2.0 * span) ** 2 / max(2.0 * area, 1e-9), taper_ratio=taper)
    return LiftingSurface(
        uid=uid,
        name=name,
        role=SurfaceRole.MAIN_WING,
        symmetry=SymmetryMode.MIRRORED_ABOUT_XZ if wing.attrib.get("symmetry") == "x-z-plane" else SymmetryMode.NONE,
        planform=planform,
        airfoils=AirfoilInterpolation(),
        external_refs={"cpacs_uid": uid},
    )


def _section_chord(section: ET.Element) -> float:
    return float(section.findtext(".//scaling/x") or 1.0)


def _section_y(section: ET.Element) -> float:
    return float(section.findtext(".//translation/y") or 0.5)


def _cpacs_symmetry(symmetry) -> str:
    value = symmetry.value if hasattr(symmetry, "value") else str(symmetry)
    return "x-z-plane" if value == SymmetryMode.MIRRORED_ABOUT_XZ.value else "none"


def _uid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value).strip("_") or "uid"


def _record_uid_map(project: AircraftProject, uid_map: list[CPACSUIDMapEntry]) -> None:
    refs = dict(project.external_refs.refs)
    for entry in uid_map:
        refs[entry.internal_uid] = entry.as_dict()
    project.external_refs = ExternalReferenceStore(refs=refs)
