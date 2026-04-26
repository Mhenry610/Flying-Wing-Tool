from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from core.aircraft.project import AircraftProject
from core.geometry.assembly import assemble_surface_instances
from core.structures.elements import StructuralElement, StructuralElementType


@dataclass
class StructuralElementResult:
    uid: str
    element_type: str
    mass_kg: float
    bending_stiffness_n_m2: float
    torsional_stiffness_n_m2: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "element_type": self.element_type,
            "mass_kg": self.mass_kg,
            "bending_stiffness_n_m2": self.bending_stiffness_n_m2,
            "torsional_stiffness_n_m2": self.torsional_stiffness_n_m2,
            "warnings": list(self.warnings),
        }


@dataclass
class ConceptualStructureResult:
    total_structural_mass_kg: float
    mass_by_surface: dict[str, float]
    mass_by_element_type: dict[str, float]
    critical_load_case: str | None
    critical_element: str | None
    stress_margin: float | None
    buckling_margin: float | None
    tip_deflection_m: float | None
    warnings: list[str]
    element_results: list[StructuralElementResult]

    def as_dict(self) -> dict:
        return {
            "total_structural_mass_kg": self.total_structural_mass_kg,
            "mass_by_surface": dict(self.mass_by_surface),
            "mass_by_element_type": dict(self.mass_by_element_type),
            "critical_load_case": self.critical_load_case,
            "critical_element": self.critical_element,
            "stress_margin": self.stress_margin,
            "buckling_margin": self.buckling_margin,
            "tip_deflection_m": self.tip_deflection_m,
            "warnings": list(self.warnings),
            "element_results": [r.as_dict() for r in self.element_results],
        }


MATERIAL_DENSITY = {
    "carbon": 1600.0,
    "carbon_tube": 1600.0,
    "balsa": 160.0,
    "plywood": 500.0,
    "default": 450.0,
}

MATERIAL_E = {
    "carbon": 70e9,
    "carbon_tube": 70e9,
    "balsa": 3e9,
    "plywood": 6e9,
    "default": 5e9,
}

MATERIAL_G = {
    "carbon": 5e9,
    "carbon_tube": 5e9,
    "balsa": 0.3e9,
    "plywood": 0.6e9,
    "default": 0.5e9,
}


def analyze_conceptual_structure(project: AircraftProject) -> ConceptualStructureResult:
    instances = {inst.source_surface_uid: inst for inst in assemble_surface_instances(project.surfaces)}
    mass_by_surface: dict[str, float] = {}
    mass_by_type: dict[str, float] = {}
    element_results: list[StructuralElementResult] = []
    warnings: list[str] = []
    critical = None
    min_margin = None

    for surface in project.surfaces:
        surface_mass = 0.0
        geom = instances.get(surface.uid).expanded_geometry if surface.uid in instances else None
        for element in surface.structural_layout.elements:
            result = _analyze_element(element, geom)
            element_results.append(result)
            surface_mass += result.mass_kg
            mass_by_type[result.element_type] = mass_by_type.get(result.element_type, 0.0) + result.mass_kg
            warnings.extend(result.warnings)
            if result.bending_stiffness_n_m2 <= 0.0 and critical is None:
                critical = result.uid
                min_margin = 0.0
        mass_by_surface[surface.uid] = surface_mass

    if any(_enum_value(e.type) in (StructuralElementType.STRUT.value, StructuralElementType.WIRE.value) for s in project.surfaces for e in s.structural_layout.elements):
        warnings.append("Braced layout load sharing is represented conceptually; detailed joint reactions need a higher-fidelity solver.")

    tip_deflection = None
    if element_results:
        stiffness_sum = sum(max(0.0, r.bending_stiffness_n_m2) for r in element_results)
        if stiffness_sum > 0.0:
            span = max((inst.expanded_geometry.span_m for inst in assemble_surface_instances(project.surfaces)), default=0.0)
            load = max(1.0, sum(project_masses(project)) * 9.80665)
            tip_deflection = load * span**3 / (3.0 * stiffness_sum)

    return ConceptualStructureResult(
        total_structural_mass_kg=sum(mass_by_surface.values()),
        mass_by_surface=mass_by_surface,
        mass_by_element_type=mass_by_type,
        critical_load_case="conceptual_limit" if element_results else None,
        critical_element=critical,
        stress_margin=min_margin,
        buckling_margin=None,
        tip_deflection_m=tip_deflection,
        warnings=sorted(set(warnings)),
        element_results=element_results,
    )


def _analyze_element(element: StructuralElement, geom) -> StructuralElementResult:
    material_key = _material_key(element.material_uid)
    length = _element_length_m(element, geom)
    area = element.section.cross_section_area_m2()
    density = MATERIAL_DENSITY.get(material_key, MATERIAL_DENSITY["default"])
    E = MATERIAL_E.get(material_key, MATERIAL_E["default"])
    G = MATERIAL_G.get(material_key, MATERIAL_G["default"])
    mass = area * length * density
    EI = E * element.section.bending_ixx_m4()
    GJ = G * element.section.torsion_j_m4()
    warnings = []
    etype = _enum_value(element.type)
    if etype in (StructuralElementType.STRUT.value, StructuralElementType.WIRE.value):
        warnings.append(f"{element.uid} uses simplified axial/bracing assumptions.")
    if length <= 0.0 and etype != StructuralElementType.SKIN_PANEL.value:
        warnings.append(f"{element.uid} has zero or unresolved length.")
    return StructuralElementResult(element.uid, etype, mass, EI, GJ, warnings)


def _element_length_m(element: StructuralElement, geom) -> float:
    if element.start.aircraft_xyz_m and element.end.aircraft_xyz_m:
        return _distance(element.start.aircraft_xyz_m, element.end.aircraft_xyz_m)
    if geom and element.start.eta is not None and element.end.eta is not None:
        return abs(float(element.end.eta) - float(element.start.eta)) * geom.span_m
    if geom and _enum_value(element.type) == StructuralElementType.SKIN_PANEL.value:
        return geom.area_m2
    return 0.0


def project_masses(project: AircraftProject) -> list[float]:
    return [item.mass_kg for item in project.mass_items] or [1.0]


def _distance(a, b) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _material_key(uid: str) -> str:
    lower = uid.lower()
    if "carbon" in lower:
        return "carbon_tube" if "tube" in lower else "carbon"
    if "balsa" in lower:
        return "balsa"
    if "ply" in lower:
        return "plywood"
    return "default"


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)

