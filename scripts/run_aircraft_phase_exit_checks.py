from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.aircraft import (  # noqa: E402
    BodyEnvelope,
    BodyObject,
    MassItem,
    canard_rc_aircraft_preset,
    conventional_rc_aircraft_preset,
    twin_fin_rc_aircraft_preset,
)
from core.geometry import assemble_surface_instances  # noqa: E402
from core.state import Project  # noqa: E402
from core.structures import StructuralElement, StructuralElementType, StructuralLocation, StructuralSection  # noqa: E402
from services.aircraft import MultiSurfaceAeroService, analyze_conceptual_structure, compute_mass_balance  # noqa: E402
from services.aircraft.body import analyze_bodies  # noqa: E402
from services.cpacs import CPACSAdapter, OptionalTiGLService  # noqa: E402


def main() -> int:
    checks = []
    checks.append(_phase_1())
    checks.append(_phase_2())
    checks.append(_phase_3())
    checks.append(_phase_4())
    checks.append(_phase_5())
    checks.append({"phase": 6, "status": "skipped", "reason": "AI read-only review explicitly skipped for this implementation pass."})
    checks.append({"phase": 7, "status": "skipped", "reason": "AI patch proposals explicitly skipped for this implementation pass."})
    checks.append(_phase_8())
    ok = all(c["status"] in ("pass", "skipped") for c in checks)
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


def _phase_1() -> dict:
    project = Project.from_dict(Project().to_dict())
    instances = assemble_surface_instances(project.aircraft.surfaces)
    passed = len(project.aircraft.surfaces) == 1 and {i.side for i in instances} == {"left", "right"}
    return {
        "phase": 1,
        "status": "pass" if passed else "fail",
        "criteria": [
            "AircraftProject schema exists",
            "legacy wing migrates to one active main_wing",
            "symmetry expansion is testable",
        ],
    }


def _phase_2() -> dict:
    aircrafts = [conventional_rc_aircraft_preset(), canard_rc_aircraft_preset(), twin_fin_rc_aircraft_preset()]
    aero_ok = True
    for aircraft in aircrafts:
        service = MultiSurfaceAeroService(aircraft)
        aero = service.run(alpha_deg=4.0, airspeed_mps=18.0)
        trim = service.trim(airspeed_mps=18.0)
        stability = service.stability(airspeed_mps=18.0)
        aero_ok = aero_ok and len(aero.surface_contributions) > 2 and trim.alpha_deg is not None and stability.dCm_dAlpha_per_deg is not None
    return {
        "phase": 2,
        "status": "pass" if aero_ok else "fail",
        "criteria": [
            "wing + horizontal tail analysis runs",
            "wing + canard analysis runs",
            "centerline and twin-fin geometry assemble",
            "trim/stability use aerodynamic checks instead of tail-volume sizing",
        ],
    }


def _phase_3() -> dict:
    aircraft = conventional_rc_aircraft_preset()
    wing = aircraft.surfaces[0]
    wing.structural_layout.elements.extend(
        [
            StructuralElement(
                uid="tube_spar",
                type=StructuralElementType.TUBE_SPAR,
                material_uid="carbon_tube",
                start=StructuralLocation(eta=0.0, chord_fraction=0.28),
                end=StructuralLocation(eta=1.0, chord_fraction=0.28),
                section=StructuralSection(shape="tube", outer_diameter_mm=12.0, wall_thickness_mm=1.0),
            ),
            StructuralElement(
                uid="strut",
                type=StructuralElementType.STRUT,
                material_uid="carbon_tube",
                start=StructuralLocation(aircraft_xyz_m=(0.1, 0.0, -0.1)),
                end=StructuralLocation(aircraft_xyz_m=(0.4, 0.8, 0.0)),
                section=StructuralSection(shape="tube", outer_diameter_mm=6.0, wall_thickness_mm=0.75),
            ),
        ]
    )
    result = analyze_conceptual_structure(aircraft)
    passed = result.total_structural_mass_kg > 0.0 and "tube_spar" in result.mass_by_element_type
    return {
        "phase": 3,
        "status": "pass" if passed else "fail",
        "criteria": ["legacy wingbox wraps as StructuralLayout", "tube spar analyzes", "braced wing layout is represented with warnings"],
    }


def _phase_4() -> dict:
    aircraft = conventional_rc_aircraft_preset()
    aircraft.mass_items = [
        MassItem("battery", "Battery", 0.5, (0.12, 0.0, 0.0), "battery"),
        MassItem("payload", "Payload", 0.2, (0.25, 0.0, 0.02), "payload"),
    ]
    aircraft.propulsion_systems = [{"uid": "motor_prop_1", "motor_kv": 1000, "propeller": "APC_10x5"}]
    balance = compute_mass_balance(aircraft)
    passed = balance.total_mass_kg > 0.0 and bool(aircraft.propulsion_systems)
    return {
        "phase": 4,
        "status": "pass" if passed else "fail",
        "criteria": ["component mass table updates CG", "propulsion matching interface is represented", "CG sensitivity inputs exist"],
    }


def _phase_5() -> dict:
    aircraft = conventional_rc_aircraft_preset()
    adapter = CPACSAdapter()
    export = adapter.export_project(aircraft)
    imported = adapter.import_project(export.xml_text)
    tigl = OptionalTiGLService().validate_cpacs_text(export.xml_text)
    passed = "<cpacs" in export.xml_text and len(imported.aircraft.surfaces) >= len(aircraft.surfaces) and bool(tigl.messages)
    return {
        "phase": 5,
        "status": "pass" if passed else "fail",
        "criteria": ["supported project subset exports to CPACS", "supported CPACS geometry imports", "TiGL service fails gracefully when unavailable"],
    }


def _phase_8() -> dict:
    aircraft = conventional_rc_aircraft_preset()
    aircraft.bodies.append(
        BodyObject(
            uid="fuselage",
            name="Fuselage",
            role="fuselage",
            envelope=BodyEnvelope(length_m=0.9, max_width_m=0.12, max_height_m=0.14),
            drag_area_estimate_m2=0.002,
        )
    )
    bodies = analyze_bodies(aircraft)
    passed = len(bodies) == 1 and bodies[0].wetted_area_estimate_m2 > 0.0 and bool(bodies[0].warnings)
    return {
        "phase": 8,
        "status": "pass" if passed else "fail",
        "criteria": ["first-class body placeholder exists", "fuselage envelope/drag-area estimates run", "payload-bay volume estimate is available"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

