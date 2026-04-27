from __future__ import annotations

from core.models.planform import ControlSurface, PlanformGeometry

from .project import AircraftProject, ProjectMetadata
from .references import AircraftReferenceFrame, Axis, SurfaceTransform
from .surfaces import LiftingSurface, SurfaceAnalysisSettings, SurfaceRole, SymmetryMode


def conventional_rc_aircraft_preset(name: str = "Conventional RC Aircraft") -> AircraftProject:
    wing = LiftingSurface.from_legacy_wing(__import__("core.models.project", fromlist=["WingProject"]).WingProject())
    wing.name = "Main Wing"
    wing.uid = "main_wing"
    wing.planform = PlanformGeometry(wing_area_m2=0.7, aspect_ratio=7.0, taper_ratio=0.55, sweep_le_deg=3.0, dihedral_deg=3.0)
    wing.control_surfaces = [ControlSurface("Aileron", "Aileron", 55.0, 95.0, 72.0, 72.0)]
    wing.planform.control_surfaces = list(wing.control_surfaces)

    htail = LiftingSurface(
        uid="horizontal_tail",
        name="Horizontal Tail",
        role=SurfaceRole.HORIZONTAL_TAIL,
        transform=SurfaceTransform(origin_m=(0.95, 0.0, 0.02)),
        symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
        local_span_axis=Axis.Y,
        planform=PlanformGeometry(wing_area_m2=0.16, aspect_ratio=4.0, taper_ratio=0.8, sweep_le_deg=0.0),
        control_surfaces=[ControlSurface("Elevator", "Elevator", 0.0, 100.0, 70.0, 70.0)],
        analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.075, zero_lift_aoa_deg=0.0, cm0=0.0, cd0=0.015, cl_max=1.1, trim_lift_direction="negative"),
    )
    htail.planform.control_surfaces = list(htail.control_surfaces)

    vtail = LiftingSurface(
        uid="vertical_tail",
        name="Vertical Tail",
        role=SurfaceRole.VERTICAL_TAIL,
        transform=SurfaceTransform(origin_m=(0.9, 0.0, 0.04)),
        symmetry=SymmetryMode.SINGLE_CENTERLINE,
        local_span_axis=Axis.Z,
        planform=PlanformGeometry(wing_area_m2=0.06, aspect_ratio=1.5, taper_ratio=0.65, sweep_le_deg=15.0),
        control_surfaces=[ControlSurface("Rudder", "Rudder", 20.0, 100.0, 70.0, 70.0)],
        analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.055, zero_lift_aoa_deg=0.0, cm0=0.0, cd0=0.015, cl_max=0.9),
    )
    vtail.planform.control_surfaces = list(vtail.control_surfaces)

    project = AircraftProject(
        metadata=ProjectMetadata(name=name),
        reference=AircraftReferenceFrame(reference_area_m2=wing.planform.actual_area(), reference_span_m=wing.planform.actual_span(), reference_chord_m=wing.planform.mean_aerodynamic_chord(), moment_reference_m=(0.18, 0.0, 0.0), cg_m=(0.18, 0.0, 0.0)),
        surfaces=[wing, htail, vtail],
    )
    return project


def canard_rc_aircraft_preset(name: str = "Canard RC Aircraft") -> AircraftProject:
    project = conventional_rc_aircraft_preset(name)
    project.surfaces = [s for s in project.surfaces if s.uid != "horizontal_tail"]
    canard = LiftingSurface(
        uid="canard",
        name="Canard",
        role=SurfaceRole.CANARD,
        transform=SurfaceTransform(origin_m=(-0.45, 0.0, 0.02)),
        symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
        local_span_axis=Axis.Y,
        planform=PlanformGeometry(wing_area_m2=0.12, aspect_ratio=4.2, taper_ratio=0.8),
        control_surfaces=[ControlSurface("Canard Elevator", "Elevator", 0.0, 100.0, 70.0, 70.0)],
        analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.075, zero_lift_aoa_deg=0.0, cm0=0.0, cd0=0.015, cl_max=1.05, trim_lift_direction="positive"),
    )
    canard.planform.control_surfaces = list(canard.control_surfaces)
    project.surfaces.append(canard)
    return project


def twin_fin_rc_aircraft_preset(name: str = "Twin Fin RC Aircraft") -> AircraftProject:
    project = conventional_rc_aircraft_preset(name)
    project.surfaces = [s for s in project.surfaces if s.uid != "vertical_tail"]
    fin = LiftingSurface(
        uid="twin_vertical_fin",
        name="Twin Vertical Fin",
        role=SurfaceRole.VERTICAL_TAIL,
        transform=SurfaceTransform(origin_m=(0.86, 0.28, 0.04)),
        symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
        local_span_axis=Axis.Z,
        planform=PlanformGeometry(wing_area_m2=0.045, aspect_ratio=1.4, taper_ratio=0.6, sweep_le_deg=18.0),
        control_surfaces=[ControlSurface("Rudder", "Rudder", 20.0, 100.0, 70.0, 70.0)],
        analysis_settings=SurfaceAnalysisSettings(cl_alpha_per_deg=0.05, zero_lift_aoa_deg=0.0, cm0=0.0, cd0=0.015, cl_max=0.85),
    )
    fin.planform.control_surfaces = list(fin.control_surfaces)
    project.surfaces.append(fin)
    return project

