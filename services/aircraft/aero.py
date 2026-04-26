from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from core.aircraft.project import AircraftProject
from core.geometry.assembly import SurfaceInstance, assemble_surface_instances


MODEL_REFERENCES = [
    {
        "label": "NASA Glenn, lift/drag coefficient reference-area convention",
        "url": "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/size-effects-on-drag/",
    },
    {
        "label": "NASA Glenn, induced drag coefficient relation",
        "url": "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/",
    },
]


@dataclass
class SurfaceAeroContribution:
    surface_uid: str
    instance_uid: str
    CL: float
    CD: float
    CM: float
    force_n: tuple[float, float, float]
    moment_nm: tuple[float, float, float]
    alpha_deg: float
    beta_deg: float
    reynolds: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "surface_uid": self.surface_uid,
            "instance_uid": self.instance_uid,
            "CL": self.CL,
            "CD": self.CD,
            "CM": self.CM,
            "force_n": list(self.force_n),
            "moment_nm": list(self.moment_nm),
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "reynolds": self.reynolds,
            "warnings": list(self.warnings),
        }


@dataclass
class AircraftAeroResult:
    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    force_n: tuple[float, float, float]
    moment_nm: tuple[float, float, float]
    surface_contributions: list[SurfaceAeroContribution]
    warnings: list[str] = field(default_factory=list)
    references: list[dict] = field(default_factory=lambda: list(MODEL_REFERENCES))

    def as_dict(self) -> dict:
        return {
            "CL": self.CL,
            "CD": self.CD,
            "CY": self.CY,
            "Cl": self.Cl,
            "Cm": self.Cm,
            "Cn": self.Cn,
            "force_n": list(self.force_n),
            "moment_nm": list(self.moment_nm),
            "surface_contributions": [c.as_dict() for c in self.surface_contributions],
            "warnings": list(self.warnings),
            "references": list(self.references),
        }


@dataclass
class TrimResult:
    exists: bool
    alpha_deg: float | None
    required_control_deflection_deg: float | None
    remaining_control_margin_deg: float | None
    surface_cls: dict[str, float]
    stall_margins: dict[str, float]
    cg_m: tuple[float, float, float]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "exists": self.exists,
            "alpha_deg": self.alpha_deg,
            "required_control_deflection_deg": self.required_control_deflection_deg,
            "remaining_control_margin_deg": self.remaining_control_margin_deg,
            "surface_cls": dict(self.surface_cls),
            "stall_margins": dict(self.stall_margins),
            "cg_m": list(self.cg_m),
            "warnings": list(self.warnings),
        }


@dataclass
class StabilityResult:
    dCm_dAlpha_per_deg: float
    neutral_point_x_m: float | None
    static_margin_percent: float | None
    control_authority_margin: float | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dCm_dAlpha_per_deg": self.dCm_dAlpha_per_deg,
            "neutral_point_x_m": self.neutral_point_x_m,
            "static_margin_percent": self.static_margin_percent,
            "control_authority_margin": self.control_authority_margin,
            "warnings": list(self.warnings),
        }


class MultiSurfaceAeroService:
    """Conceptual multi-surface aero aggregation.

    This service intentionally uses transparent finite-wing coefficient equations and
    moment summation. Higher-fidelity AeroSandbox/VLM paths can replace the per-surface
    model later without changing the aircraft-level result schema.
    """

    def __init__(self, aircraft: AircraftProject) -> None:
        self.aircraft = aircraft

    def run(
        self,
        alpha_deg: float,
        beta_deg: float = 0.0,
        airspeed_mps: float = 20.0,
        rho_kg_m3: float = 1.225,
        control_deflections: dict[str, float] | None = None,
    ) -> AircraftAeroResult:
        q = 0.5 * rho_kg_m3 * max(0.1, airspeed_mps) ** 2
        ref = self.aircraft.reference
        ref_area = max(ref.reference_area_m2, 1e-9)
        ref_span = max(ref.reference_span_m, 1e-9)
        ref_chord = max(ref.reference_chord_m, 1e-9)
        contributions: list[SurfaceAeroContribution] = []
        total_f = (0.0, 0.0, 0.0)
        total_m = (0.0, 0.0, 0.0)
        warnings: list[str] = []
        by_uid = {s.uid: s for s in self.aircraft.surfaces}

        for instance in assemble_surface_instances(self.aircraft.surfaces):
            surface = by_uid[instance.source_surface_uid]
            if not surface.analysis_settings.active_in_aero:
                continue
            contribution = self._surface_contribution(
                surface=surface,
                instance=instance,
                alpha_deg=alpha_deg,
                beta_deg=beta_deg,
                q=q,
                control_deflections=control_deflections or {},
            )
            contributions.append(contribution)
            total_f = _add(total_f, contribution.force_n)
            total_m = _add(total_m, contribution.moment_nm)
            warnings.extend(contribution.warnings)

        return AircraftAeroResult(
            CL=total_f[2] / (q * ref_area),
            CD=-total_f[0] / (q * ref_area),
            CY=total_f[1] / (q * ref_area),
            Cl=total_m[0] / (q * ref_area * ref_span),
            Cm=total_m[1] / (q * ref_area * ref_chord),
            Cn=total_m[2] / (q * ref_area * ref_span),
            force_n=total_f,
            moment_nm=total_m,
            surface_contributions=contributions,
            warnings=sorted(set(warnings)),
        )

    def trim(
        self,
        target_CL: float | None = None,
        airspeed_mps: float = 20.0,
        rho_kg_m3: float = 1.225,
        alpha_bounds_deg: tuple[float, float] = (-8.0, 18.0),
        control_bounds_deg: tuple[float, float] = (-25.0, 25.0),
    ) -> TrimResult:
        warnings: list[str] = []
        if target_CL is None:
            total_mass = _project_mass_kg(self.aircraft)
            target_CL = total_mass * 9.80665 / (0.5 * rho_kg_m3 * max(0.1, airspeed_mps) ** 2 * max(self.aircraft.reference.reference_area_m2, 1e-9))

        best = None
        best_err = float("inf")
        for i in range(81):
            alpha = alpha_bounds_deg[0] + (alpha_bounds_deg[1] - alpha_bounds_deg[0]) * i / 80.0
            result = self.run(alpha, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
            err = abs(result.CL - target_CL) + 8.0 * abs(result.Cm)
            if err < best_err:
                best_err = err
                best = (alpha, result)

        if best is None:
            return TrimResult(False, None, None, None, {}, {}, self.aircraft.reference.cg_m, ["No active aerodynamic surfaces."])

        alpha, result = best
        control = self._estimate_pitch_control_for_trim(alpha, control_bounds_deg, airspeed_mps, rho_kg_m3)
        surface_cls = {c.instance_uid: c.CL for c in result.surface_contributions}
        stall_margins = {}
        by_uid = {s.uid: s for s in self.aircraft.surfaces}
        for c in result.surface_contributions:
            cl_max = by_uid[c.surface_uid].analysis_settings.cl_max
            stall_margins[c.instance_uid] = cl_max - abs(c.CL)
        if abs(result.Cm) > 0.03 and control is None:
            warnings.append("Trim moment residual exceeds conceptual tolerance and no pitch control authority was found.")
        return TrimResult(
            exists=abs(result.Cm) <= 0.03 or control is not None,
            alpha_deg=alpha,
            required_control_deflection_deg=control,
            remaining_control_margin_deg=min(abs(control_bounds_deg[0] - control), abs(control_bounds_deg[1] - control)) if control is not None else None,
            surface_cls=surface_cls,
            stall_margins=stall_margins,
            cg_m=self.aircraft.reference.cg_m,
            warnings=warnings + result.warnings,
        )

    def stability(self, alpha_deg: float = 4.0, airspeed_mps: float = 20.0) -> StabilityResult:
        da = 0.5
        minus = self.run(alpha_deg - da, airspeed_mps=airspeed_mps)
        plus = self.run(alpha_deg + da, airspeed_mps=airspeed_mps)
        dcm_da = (plus.Cm - minus.Cm) / (2.0 * da)
        warnings: list[str] = []
        mac = self.aircraft.reference.reference_chord_m
        cg_x = self.aircraft.reference.cg_m[0]
        neutral = None
        static_margin = None
        if abs(plus.CL - minus.CL) > 1e-9:
            dcl_da = (plus.CL - minus.CL) / (2.0 * da)
            neutral = cg_x - dcm_da / dcl_da * mac
            static_margin = (neutral - cg_x) / mac * 100.0 if mac > 0 else None
        if dcm_da >= 0.0:
            warnings.append("Static longitudinal stability derivative is non-negative at the evaluated condition.")
        return StabilityResult(
            dCm_dAlpha_per_deg=dcm_da,
            neutral_point_x_m=neutral,
            static_margin_percent=static_margin,
            control_authority_margin=None,
            warnings=warnings + plus.warnings,
        )

    def _surface_contribution(
        self,
        surface,
        instance: SurfaceInstance,
        alpha_deg: float,
        beta_deg: float,
        q: float,
        control_deflections: dict[str, float],
    ) -> SurfaceAeroContribution:
        geom = instance.expanded_geometry
        settings = surface.analysis_settings
        effective_alpha = alpha_deg + surface.incidence_deg - settings.zero_lift_aoa_deg
        control_effect = _pitch_control_effect(surface, control_deflections)
        cl = settings.cl_alpha_per_deg * (effective_alpha + control_effect)
        ar = geom.span_m**2 / max(geom.area_m2, 1e-9)
        e = max(0.1, settings.oswald_efficiency)
        cd = settings.cd0 + cl * cl / (pi * max(ar, 0.1) * e)
        cm = settings.cm0 - 0.01 * control_effect
        force = (-cd * q * geom.area_m2, 0.0, cl * q * geom.area_m2)
        r = _sub(geom.aerodynamic_center_m, self.aircraft.reference.moment_reference_m)
        aero_moment = _cross(r, force)
        pitch_moment = (0.0, cm * q * geom.area_m2 * geom.mean_aerodynamic_chord_m, 0.0)
        moment = _add(aero_moment, pitch_moment)
        warnings = list(settings.warnings)
        if abs(cl) > 0.9 * max(0.1, settings.cl_max):
            warnings.append(f"{surface.name} CL is near or above the configured CLmax.")
        return SurfaceAeroContribution(
            surface_uid=surface.uid,
            instance_uid=instance.instance_uid,
            CL=cl,
            CD=cd,
            CM=cm,
            force_n=force,
            moment_nm=moment,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            warnings=warnings,
        )

    def _estimate_pitch_control_for_trim(self, alpha: float, bounds: tuple[float, float], airspeed: float, rho: float) -> float | None:
        pitch_names = []
        for surface in self.aircraft.surfaces:
            for cs in surface.control_surfaces:
                if cs.surface_type.lower() in ("elevon", "elevator", "stabilator"):
                    pitch_names.append(cs.name)
        if not pitch_names:
            return None
        lo, hi = bounds
        best = None
        best_abs = float("inf")
        for i in range(61):
            delta = lo + (hi - lo) * i / 60.0
            controls = {name: delta for name in pitch_names}
            result = self.run(alpha, airspeed_mps=airspeed, rho_kg_m3=rho, control_deflections=controls)
            if abs(result.Cm) < best_abs:
                best_abs = abs(result.Cm)
                best = delta
        return best if best_abs <= 0.03 else None


def _pitch_control_effect(surface, control_deflections: dict[str, float]) -> float:
    effect = 0.0
    for cs in surface.control_surfaces:
        if cs.surface_type.lower() in ("elevon", "elevator", "stabilator"):
            effect += 0.35 * float(control_deflections.get(cs.name, control_deflections.get(f"{cs.name}_pitch", 0.0)))
    return effect


def _project_mass_kg(project: AircraftProject) -> float:
    mass = sum(item.mass_kg for item in project.mass_items)
    if mass > 0.0:
        return mass
    legacy = project.analyses.results.get("performance_metrics", {})
    return float(legacy.get("gross_takeoff_weight_kg", 1.0))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

