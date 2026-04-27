from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from core.aircraft.project import AircraftProject
from core.aircraft.references import SurfaceTransform
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


@dataclass
class TrimSurfaceAdjustmentResult:
    surface_uid: str | None
    moved_x_m: float | None
    incidence_deg: float | None
    alpha_deg: float | None
    cm: float | None
    trim_surface_cl: float | None
    static_margin_percent: float | None
    target_static_margin_percent: float | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "surface_uid": self.surface_uid,
            "moved_x_m": self.moved_x_m,
            "incidence_deg": self.incidence_deg,
            "alpha_deg": self.alpha_deg,
            "cm": self.cm,
            "trim_surface_cl": self.trim_surface_cl,
            "static_margin_percent": self.static_margin_percent,
            "target_static_margin_percent": self.target_static_margin_percent,
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

    def stability(self, alpha_deg: float = 4.0, airspeed_mps: float = 20.0, rho_kg_m3: float = 1.225) -> StabilityResult:
        da = 0.5
        minus = self.run(alpha_deg - da, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
        plus = self.run(alpha_deg + da, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
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

    def optimize_trim_surface(
        self,
        target_static_margin_percent: float,
        target_cm: float = 0.0,
        airspeed_mps: float = 20.0,
        rho_kg_m3: float = 1.225,
        alpha_bounds_deg: tuple[float, float] = (-8.0, 18.0),
        incidence_bounds_deg: tuple[float, float] = (-15.0, 15.0),
        x_search_span_m: float | None = None,
    ) -> TrimSurfaceAdjustmentResult:
        """Move the active trim surface for static margin, then set incidence for Cm.

        This is a conceptual sizing helper for conventional/canard topologies. The
        trim surface x-location affects dCm/dAlpha and therefore neutral point;
        incidence primarily changes the moment intercept at the design condition.
        """
        trim_surface = self._primary_trim_surface()
        if trim_surface is None:
            return TrimSurfaceAdjustmentResult(None, None, None, None, None, None, None, target_static_margin_percent, ["No active trim surface was found."])

        original_transform = trim_surface.transform
        original_incidence = trim_surface.incidence_deg
        warnings: list[str] = []
        ref_chord = max(self.aircraft.reference.reference_chord_m, 0.05)
        search_span = x_search_span_m if x_search_span_m is not None else max(0.5, 5.0 * ref_chord)
        x0, y0, z0 = trim_surface.transform.origin_m

        best_x = x0
        best_sm = None
        best_score = float("inf")
        alpha_for_stability = self._alpha_for_target_cl(airspeed_mps, rho_kg_m3, alpha_bounds_deg)
        for i in range(81):
            x = x0 - 0.5 * search_span + search_span * i / 80.0
            self._set_surface_x(trim_surface, x)
            stability = self.stability(alpha_deg=alpha_for_stability, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
            if stability.static_margin_percent is None:
                continue
            score = abs(stability.static_margin_percent - target_static_margin_percent)
            if score < best_score:
                best_score = score
                best_x = x
                best_sm = stability.static_margin_percent

        self._set_surface_x(trim_surface, best_x)
        if best_sm is None:
            warnings.append("Static margin could not be evaluated while moving the trim surface.")

        best_incidence = original_incidence
        best_alpha = None
        best_cm = None
        best_trim_cl = None
        best_inc_score = float("inf")
        best_any = None
        best_any_score = float("inf")
        direction = str(getattr(trim_surface.analysis_settings, "trim_lift_direction", "auto") or "auto").lower()
        inc_lo, inc_hi = incidence_bounds_deg
        for i in range(81):
            incidence = inc_lo + (inc_hi - inc_lo) * i / 80.0
            trim_surface.incidence_deg = incidence
            alpha = self._alpha_for_target_cl(airspeed_mps, rho_kg_m3, alpha_bounds_deg)
            result = self.run(alpha, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
            trim_cl = _surface_average_cl(result, trim_surface.uid)
            cm_error = abs(result.Cm - target_cm)
            any_score = cm_error + 0.02 * abs(trim_cl)
            if any_score < best_any_score:
                best_any_score = any_score
                best_any = (incidence, alpha, result.Cm, trim_cl)
            if not _trim_lift_sign_allowed(trim_cl, direction):
                continue
            score = cm_error
            if score < best_inc_score:
                best_inc_score = score
                best_incidence = incidence
                best_alpha = alpha
                best_cm = result.Cm
                best_trim_cl = trim_cl

        if best_alpha is None and best_any is not None:
            best_incidence, best_alpha, best_cm, best_trim_cl = best_any
            warnings.append(f"Trim lift convention '{direction}' could not be satisfied while searching incidence.")

        trim_surface.incidence_deg = best_incidence
        final_stability = self.stability(alpha_deg=best_alpha if best_alpha is not None else alpha_for_stability, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
        if best_sm is not None and abs((final_stability.static_margin_percent or best_sm) - target_static_margin_percent) > 2.0:
            warnings.append("Trim surface x search did not reach the requested static margin within 2 percentage points.")
        if best_cm is not None and abs(best_cm - target_cm) > 0.03:
            warnings.append("Trim surface incidence search did not reach the requested Cm within 0.03.")
        if not _trim_lift_sign_allowed(best_trim_cl, direction):
            warnings.append(f"Trim surface lift sign does not match the configured '{direction}' convention.")

        # Keep the intentional optimized transform/incidence; restore only if no solution at all.
        if best_sm is None and best_cm is None:
            trim_surface.transform = original_transform
            trim_surface.incidence_deg = original_incidence

        return TrimSurfaceAdjustmentResult(
            surface_uid=trim_surface.uid,
            moved_x_m=trim_surface.transform.origin_m[0],
            incidence_deg=trim_surface.incidence_deg,
            alpha_deg=best_alpha,
            cm=best_cm,
            trim_surface_cl=best_trim_cl,
            static_margin_percent=final_stability.static_margin_percent,
            target_static_margin_percent=target_static_margin_percent,
            warnings=warnings + final_stability.warnings,
        )

    def _primary_trim_surface(self):
        trim_roles = {"horizontal_tail", "canard", "stabilator"}
        for surface in self.aircraft.surfaces:
            role = surface.role.value if hasattr(surface.role, "value") else str(surface.role)
            if surface.active and role in trim_roles and surface.analysis_settings.active_in_aero:
                return surface
        return None

    def _set_surface_x(self, surface, x_m: float) -> None:
        _x, y, z = surface.transform.origin_m
        surface.transform = SurfaceTransform(
            origin_m=(float(x_m), y, z),
            orientation_euler_deg=surface.transform.orientation_euler_deg,
            parent_uid=surface.transform.parent_uid,
        )

    def _alpha_for_target_cl(
        self,
        airspeed_mps: float,
        rho_kg_m3: float,
        alpha_bounds_deg: tuple[float, float],
    ) -> float:
        target_cl = _project_mass_kg(self.aircraft) * 9.80665 / (
            0.5 * rho_kg_m3 * max(0.1, airspeed_mps) ** 2 * max(self.aircraft.reference.reference_area_m2, 1e-9)
        )
        best_alpha = alpha_bounds_deg[0]
        best_err = float("inf")
        for i in range(81):
            alpha = alpha_bounds_deg[0] + (alpha_bounds_deg[1] - alpha_bounds_deg[0]) * i / 80.0
            result = self.run(alpha, airspeed_mps=airspeed_mps, rho_kg_m3=rho_kg_m3)
            err = abs(result.CL - target_cl)
            if err < best_err:
                best_err = err
                best_alpha = alpha
        return best_alpha

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
        cl, convention_warning = _apply_trim_lift_convention(cl, settings.trim_lift_direction)
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
        if convention_warning:
            warnings.append(f"{surface.name} lift was limited by its '{settings.trim_lift_direction}' trim-lift convention.")
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


def _surface_average_cl(result: AircraftAeroResult, surface_uid: str) -> float:
    cls = [c.CL for c in result.surface_contributions if c.surface_uid == surface_uid]
    if not cls:
        return 0.0
    return sum(cls) / len(cls)


def _trim_lift_sign_allowed(trim_cl: float | None, direction: str) -> bool:
    if trim_cl is None:
        return False
    if direction == "positive":
        return trim_cl >= -1e-6
    if direction == "negative":
        return trim_cl <= 1e-6
    return True


def _apply_trim_lift_convention(cl: float, direction: str) -> tuple[float, bool]:
    convention = str(direction or "auto").lower()
    if convention == "negative" and cl > 0.0:
        return 0.0, True
    if convention == "positive" and cl < 0.0:
        return 0.0, True
    return cl, False


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

