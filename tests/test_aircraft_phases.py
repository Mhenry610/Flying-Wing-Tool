from __future__ import annotations

import unittest

from core.aircraft import (
    BodyEnvelope,
    BodyObject,
    MassItem,
    canard_rc_aircraft_preset,
    conventional_rc_aircraft_preset,
    twin_fin_rc_aircraft_preset,
)
from core.geometry import assemble_surface_instances
from core.state import Project
from core.structures import StructuralElement, StructuralElementType, StructuralLocation, StructuralSection
from services.aircraft import MultiSurfaceAeroService, analyze_conceptual_structure, compute_mass_balance
from services.aircraft.body import analyze_bodies
from services.cpacs import CPACSAdapter, OptionalTiGLService


class AircraftPhaseExitCriteriaTests(unittest.TestCase):
    def test_phase_1_legacy_project_migrates_to_main_wing_surface(self):
        project = Project()
        data = project.to_dict()
        loaded = Project.from_dict(data)
        self.assertEqual(loaded.aircraft.schema_version, 2)
        self.assertEqual(len(loaded.aircraft.surfaces), 1)
        self.assertEqual(loaded.aircraft.surfaces[0].uid, "main_wing")
        instances = assemble_surface_instances(loaded.aircraft.surfaces)
        self.assertEqual({i.side for i in instances}, {"left", "right"})

    def test_phase_2_conventional_canard_and_twin_fin_aero_run(self):
        for aircraft in (conventional_rc_aircraft_preset(), canard_rc_aircraft_preset(), twin_fin_rc_aircraft_preset()):
            result = MultiSurfaceAeroService(aircraft).run(alpha_deg=4.0, airspeed_mps=18.0)
            self.assertGreater(len(result.surface_contributions), 2)
            self.assertTrue(abs(result.CL) > 0.01)
            trim = MultiSurfaceAeroService(aircraft).trim(airspeed_mps=18.0)
            self.assertIn(trim.exists, (True, False))
            stability = MultiSurfaceAeroService(aircraft).stability(airspeed_mps=18.0)
            self.assertIsNotNone(stability.dCm_dAlpha_per_deg)

    def test_phase_3_generic_structure_supports_tube_spar_and_bracing(self):
        aircraft = conventional_rc_aircraft_preset()
        wing = aircraft.surfaces[0]
        wing.structural_layout.elements.append(
            StructuralElement(
                uid="carbon_tube_spar",
                type=StructuralElementType.TUBE_SPAR,
                material_uid="carbon_tube",
                start=StructuralLocation(eta=0.0, chord_fraction=0.28),
                end=StructuralLocation(eta=1.0, chord_fraction=0.28),
                section=StructuralSection(shape="tube", outer_diameter_mm=12.0, wall_thickness_mm=1.0),
            )
        )
        wing.structural_layout.elements.append(
            StructuralElement(
                uid="right_strut",
                type=StructuralElementType.STRUT,
                material_uid="carbon_tube",
                start=StructuralLocation(aircraft_xyz_m=(0.1, 0.0, -0.1)),
                end=StructuralLocation(aircraft_xyz_m=(0.3, 0.8, 0.0)),
                section=StructuralSection(shape="tube", outer_diameter_mm=6.0, wall_thickness_mm=0.75),
            )
        )
        result = analyze_conceptual_structure(aircraft)
        self.assertGreater(result.total_structural_mass_kg, 0.0)
        self.assertIn("tube_spar", result.mass_by_element_type)
        self.assertTrue(any("Braced layout" in w for w in result.warnings))

    def test_phase_4_mass_cg_and_propulsion_hooks(self):
        aircraft = conventional_rc_aircraft_preset()
        aircraft.mass_items = [
            MassItem("battery", "Battery", 0.5, (0.12, 0.0, 0.0), "battery"),
            MassItem("payload", "Payload", 0.2, (0.25, 0.0, 0.02), "payload"),
        ]
        aircraft.propulsion_systems = [{"uid": "motor_prop_1", "motor_kv": 1000, "propeller": "APC_10x5"}]
        balance = compute_mass_balance(aircraft)
        self.assertAlmostEqual(balance.total_mass_kg, 0.7)
        self.assertGreater(balance.cg_m[0], 0.12)
        self.assertEqual(aircraft.propulsion_systems[0]["uid"], "motor_prop_1")

    def test_phase_5_cpacs_export_import_and_tigl_graceful_probe(self):
        aircraft = conventional_rc_aircraft_preset()
        exported = CPACSAdapter().export_project(aircraft)
        self.assertIn("<cpacs", exported.xml_text)
        self.assertGreaterEqual(len(exported.uid_map), len(aircraft.surfaces))
        imported = CPACSAdapter().import_project(exported.xml_text)
        self.assertGreaterEqual(len(imported.aircraft.surfaces), len(aircraft.surfaces))
        tigl_diag = OptionalTiGLService().validate_cpacs_text(exported.xml_text)
        self.assertIn(tigl_diag.available, (True, False))
        self.assertTrue(tigl_diag.messages)

    def test_phase_8_body_placeholder_analysis(self):
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
        results = analyze_bodies(aircraft)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].wetted_area_estimate_m2, 0.0)
        self.assertGreater(results[0].payload_bay_volume_m3 or 0.0, 0.0)
        self.assertTrue(any("Fuselage aero" in w for w in results[0].warnings))


if __name__ == "__main__":
    unittest.main()

