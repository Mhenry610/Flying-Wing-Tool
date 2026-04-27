from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.models.airfoil import AirfoilInterpolation
from core.models.planform import ControlSurface, PlanformGeometry
from core.structures.elements import StructuralLayout

from .references import Axis, SurfaceTransform


class SurfaceRole(str, Enum):
    MAIN_WING = "main_wing"
    HORIZONTAL_TAIL = "horizontal_tail"
    VERTICAL_TAIL = "vertical_tail"
    CANARD = "canard"
    FIN = "fin"
    WINGLET = "winglet"
    STABILATOR = "stabilator"
    CUSTOM = "custom"


class SymmetryMode(str, Enum):
    NONE = "none"
    MIRRORED_ABOUT_XZ = "mirrored_about_xz"
    SINGLE_CENTERLINE = "single_centerline"
    PAIRED_EXPLICIT = "paired_explicit"


@dataclass
class TwistDistribution:
    twist_deg: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"twist_deg": list(self.twist_deg)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "TwistDistribution":
        data = data or {}
        return cls(twist_deg=[float(v) for v in data.get("twist_deg", [])])


@dataclass
class SurfaceAnalysisSettings:
    cl_alpha_per_deg: float = 0.085
    zero_lift_aoa_deg: float = -2.0
    cm0: float = -0.02
    cd0: float = 0.018
    oswald_efficiency: float = 0.82
    cl_max: float = 1.2
    design_cl: float = 0.45
    lift_distribution: str = "bell"
    trim_lift_direction: str = "auto"
    active_in_aero: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cl_alpha_per_deg": self.cl_alpha_per_deg,
            "zero_lift_aoa_deg": self.zero_lift_aoa_deg,
            "cm0": self.cm0,
            "cd0": self.cd0,
            "oswald_efficiency": self.oswald_efficiency,
            "cl_max": self.cl_max,
            "design_cl": self.design_cl,
            "lift_distribution": self.lift_distribution,
            "trim_lift_direction": self.trim_lift_direction,
            "active_in_aero": self.active_in_aero,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "SurfaceAnalysisSettings":
        data = data or {}
        return cls(
            cl_alpha_per_deg=float(data.get("cl_alpha_per_deg", 0.085)),
            zero_lift_aoa_deg=float(data.get("zero_lift_aoa_deg", -2.0)),
            cm0=float(data.get("cm0", -0.02)),
            cd0=float(data.get("cd0", 0.018)),
            oswald_efficiency=float(data.get("oswald_efficiency", 0.82)),
            cl_max=float(data.get("cl_max", 1.2)),
            design_cl=float(data.get("design_cl", 0.45)),
            lift_distribution=str(data.get("lift_distribution", "bell")),
            trim_lift_direction=str(data.get("trim_lift_direction", "auto")),
            active_in_aero=bool(data.get("active_in_aero", True)),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class LiftingSurface:
    uid: str
    name: str
    role: SurfaceRole | str = SurfaceRole.MAIN_WING
    active: bool = True
    transform: SurfaceTransform = field(default_factory=SurfaceTransform)
    symmetry: SymmetryMode | str = SymmetryMode.MIRRORED_ABOUT_XZ
    local_span_axis: Axis | str = Axis.Y
    planform: PlanformGeometry = field(default_factory=PlanformGeometry)
    airfoils: AirfoilInterpolation = field(default_factory=AirfoilInterpolation)
    twist: TwistDistribution = field(default_factory=TwistDistribution)
    incidence_deg: float = 0.0
    control_surfaces: list[ControlSurface] = field(default_factory=list)
    structural_layout: StructuralLayout = field(default_factory=StructuralLayout)
    analysis_settings: SurfaceAnalysisSettings = field(default_factory=SurfaceAnalysisSettings)
    external_refs: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "role": _value(self.role),
            "active": self.active,
            "transform": self.transform.as_dict(),
            "symmetry": _value(self.symmetry),
            "local_span_axis": _value(self.local_span_axis),
            "planform": self.planform.as_dict(),
            "airfoils": self.airfoils.as_dict(),
            "twist": self.twist.as_dict(),
            "incidence_deg": self.incidence_deg,
            "control_surfaces": [cs.as_dict() for cs in self.control_surfaces],
            "structural_layout": self.structural_layout.as_dict(),
            "analysis_settings": self.analysis_settings.as_dict(),
            "external_refs": dict(self.external_refs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LiftingSurface":
        planform = _planform_from_dict(data.get("planform", {}))
        controls_data = data.get("control_surfaces")
        controls = [ControlSurface.from_dict(cs) for cs in controls_data] if controls_data is not None else list(planform.control_surfaces)
        planform.control_surfaces = list(controls)
        return cls(
            uid=data["uid"],
            name=data.get("name", data["uid"]),
            role=data.get("role", SurfaceRole.MAIN_WING.value),
            active=bool(data.get("active", True)),
            transform=SurfaceTransform.from_dict(data.get("transform")),
            symmetry=data.get("symmetry", SymmetryMode.MIRRORED_ABOUT_XZ.value),
            local_span_axis=data.get("local_span_axis", Axis.Y.value),
            planform=planform,
            airfoils=AirfoilInterpolation(**data.get("airfoils", {})),
            twist=TwistDistribution.from_dict(data.get("twist")),
            incidence_deg=float(data.get("incidence_deg", 0.0)),
            control_surfaces=controls,
            structural_layout=StructuralLayout.from_dict(data.get("structural_layout")),
            analysis_settings=SurfaceAnalysisSettings.from_dict(data.get("analysis_settings")),
            external_refs=dict(data.get("external_refs", {})),
        )

    @classmethod
    def from_legacy_wing(cls, wing_project) -> "LiftingSurface":
        planform = wing_project.planform
        controls = list(planform.control_surfaces)
        if not controls:
            controls = [
                ControlSurface(
                    name="Elevon",
                    surface_type="Elevon",
                    span_start_percent=planform.elevon_root_span_percent,
                    span_end_percent=100.0,
                    chord_start_percent=100.0 - planform.elevon_root_chord_percent,
                    chord_end_percent=100.0 - planform.elevon_tip_chord_percent,
                )
            ]
            planform.control_surfaces = list(controls)
        return cls(
            uid="main_wing",
            name="Main Wing",
            role=SurfaceRole.MAIN_WING,
            transform=SurfaceTransform(origin_m=(0.0, 0.0, 0.0)),
            symmetry=SymmetryMode.MIRRORED_ABOUT_XZ,
            local_span_axis=Axis.Y,
            planform=planform,
            airfoils=wing_project.airfoil,
            twist=TwistDistribution(twist_deg=list(wing_project.optimized_twist_deg or [])),
            incidence_deg=0.0,
            control_surfaces=controls,
            structural_layout=StructuralLayout.from_legacy_wingbox("main_wing_structure", planform),
            analysis_settings=SurfaceAnalysisSettings(
                cl_alpha_per_deg=(wing_project.twist_trim.cl_alpha_root_per_deg + wing_project.twist_trim.cl_alpha_tip_per_deg) / 2.0,
                zero_lift_aoa_deg=(wing_project.twist_trim.zero_lift_aoa_root_deg + wing_project.twist_trim.zero_lift_aoa_tip_deg) / 2.0,
                cm0=(wing_project.twist_trim.cm0_root + wing_project.twist_trim.cm0_tip) / 2.0,
                design_cl=wing_project.twist_trim.design_cl,
                lift_distribution=wing_project.twist_trim.lift_distribution,
                cl_max=wing_project.twist_trim.estimated_cl_max,
            ),
            external_refs={"legacy_wing_project": wing_project.name},
        )


def _planform_from_dict(data: dict) -> PlanformGeometry:
    from core.models.planform import BodySection
    import dataclasses

    valid = {f.name for f in dataclasses.fields(PlanformGeometry) if f.init} - {"body_sections", "control_surfaces"}
    kwargs = {k: v for k, v in data.items() if k in valid}
    body_sections = [BodySection.from_dict(bs) for bs in data.get("body_sections", [])]
    controls = [ControlSurface.from_dict(cs) for cs in data.get("control_surfaces", [])]
    return PlanformGeometry(**kwargs, body_sections=body_sections, control_surfaces=controls)


def _value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)

