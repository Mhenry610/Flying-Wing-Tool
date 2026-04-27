from __future__ import annotations

import unittest

from core.aircraft import (
    BodyEnvelope,
    BodyObject,
    MassItem,
    LiftingSurface,
    SurfaceAnalysisSettings,
    SurfaceRole,
    SymmetryMode,
    canard_rc_aircraft_preset,
    conventional_rc_aircraft_preset,
    twin_fin_rc_aircraft_preset,
)
from core.geometry import assemble_surface_instances
from core.models.planform import PlanformGeometry
from core.state import Project
from core.structures import StructuralElement, StructuralElementType, StructuralLocation, StructuralSection
from services.aircraft import MultiSurfaceAeroService, analyze_conceptual_structure, compute_mass_balance
from services.aircraft.body import analyze_bodies
from services.cpacs import CPACSAdapter, OptionalTiGLService
from services.geometry import AeroSandboxService


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

    def test_trim_surface_adjustment_moves_tail_and_sets_incidence(self):
        aircraft = conventional_rc_aircraft_preset()
        tail = next(s for s in aircraft.surfaces if s.uid == "horizontal_tail")
        original_x = tail.transform.origin_m[0]
        result = MultiSurfaceAeroService(aircraft).optimize_trim_surface(
            target_static_margin_percent=8.0,
            target_cm=0.0,
            airspeed_mps=18.0,
        )
        self.assertEqual(result.surface_uid, "horizontal_tail")
        self.assertIsNotNone(result.static_margin_percent)
        self.assertIsNotNone(result.cm)
        self.assertLessEqual(result.trim_surface_cl, 0.0)
        self.assertEqual(tail.analysis_settings.trim_lift_direction, "negative")
        aero = MultiSurfaceAeroService(aircraft).run(alpha_deg=4.0, airspeed_mps=18.0)
        tail_cls = [c.CL for c in aero.surface_contributions if c.surface_uid == "horizontal_tail"]
        self.assertTrue(tail_cls)
        self.assertTrue(all(cl <= 0.0 for cl in tail_cls))
        self.assertTrue(abs(tail.transform.origin_m[0] - original_x) > 1e-9 or abs(tail.incidence_deg) > 1e-9)

    def test_twist_optimizer_returns_root_relative_bounded_twist(self):
        project = Project()
        project.wing.planform.max_tip_twist_deg = 3.0
        service = AeroSandboxService(project)
        twist = service._normalize_and_limit_twist([13.7, 4.0, -17.0])
        self.assertAlmostEqual(twist[0], 0.0)
        self.assertLessEqual(max(abs(v) for v in twist), 3.0 + 1e-9)

    def test_chordwise_lift_distribution_planform_modes_preserve_area(self):
        linear = PlanformGeometry(wing_area_m2=1.0, aspect_ratio=6.0, taper_ratio=0.5)
        elliptical = PlanformGeometry(
            wing_area_m2=1.0,
            aspect_ratio=6.0,
            chord_distribution_mode="elliptical",
            chord_distribution_tip_floor_percent=5.0,
        )
        bell = PlanformGeometry(
            wing_area_m2=1.0,
            aspect_ratio=6.0,
            chord_distribution_mode="bell",
            chord_distribution_tip_floor_percent=5.0,
        )
        self.assertGreater(linear.chord_at_span_fraction(1.0), elliptical.chord_at_span_fraction(1.0))
        self.assertGreater(elliptical.chord_at_span_fraction(0.0), elliptical.chord_at_span_fraction(1.0))
        self.assertGreater(bell.chord_at_span_fraction(0.5), bell.chord_at_span_fraction(1.0))
        self.assertAlmostEqual(elliptical.area_for_root_chord(elliptical.root_chord()), elliptical.wing_area_m2, places=3)

        eta = 0.5
        baseline = elliptical.linear_chord_at_span_fraction(eta)
        shifted = elliptical.chord_at_span_fraction(eta)
        self.assertEqual(elliptical.leading_edge_offset_at_span_fraction(eta), 0.0)
        elliptical.split_chord_distribution_offsets = True
        self.assertAlmostEqual(elliptical.leading_edge_offset_at_span_fraction(eta), -0.5 * (shifted - baseline))

    def test_multi_main_wing_and_per_surface_lift_distribution_are_preserved(self):
        project = Project()
        project.aircraft.surfaces.append(
            LiftingSurface(
                uid="main_wing_2",
                name="Second Main Wing",
                role=SurfaceRole.MAIN_WING,
                symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
                planform=PlanformGeometry(wing_area_m2=0.2, aspect_ratio=5.0),
            )
        )
        tail = LiftingSurface(
            uid="horizontal_tail",
            name="Horizontal Tail",
            role=SurfaceRole.HORIZONTAL_TAIL,
            symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
            planform=PlanformGeometry(wing_area_m2=0.1, aspect_ratio=4.0),
            analysis_settings=SurfaceAnalysisSettings(design_cl=0.25, lift_distribution="elliptical"),
        )
        project.aircraft.surfaces.append(tail)
        project.sync_legacy_wing_to_aircraft()

        main_wings = [s for s in project.aircraft.surfaces if s.role == SurfaceRole.MAIN_WING or s.role == SurfaceRole.MAIN_WING.value]
        self.assertGreaterEqual(len(main_wings), 2)
        self.assertEqual(tail.analysis_settings.lift_distribution, "elliptical")

        loaded = Project.from_dict(project.to_dict())
        loaded_tail = next(s for s in loaded.aircraft.surfaces if s.uid == "horizontal_tail")
        self.assertEqual(loaded_tail.analysis_settings.lift_distribution, "elliptical")
        self.assertAlmostEqual(loaded_tail.analysis_settings.design_cl, 0.25)

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

