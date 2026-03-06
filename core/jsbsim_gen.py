# pyright: ignore
"""
JSBSim aircraft export for flying wing projects.

Generates JSBSim-compatible aircraft XML plus optional engine and propeller
files using geometry and aerodynamic data derived from AeroSandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple, List, Any, Dict, TYPE_CHECKING, cast
import xml.etree.ElementTree as ET

import numpy as np
import aerosandbox as asb

from core.state import Project
from services.geometry import AeroSandboxService


@dataclass
class JSBSimPropellerTable:
    advance_ratio: np.ndarray
    ct: np.ndarray
    cp: np.ndarray


@dataclass
class JSBSimPropulsionConfig:
    engine_power_w: float
    engine_count: int = 1
    engine_file: Optional[str] = None
    propeller_file: Optional[str] = None
    propeller_diameter_in: float = 10.0
    propeller_blades: int = 2
    propeller_gearratio: float = 1.0
    propeller_table: Optional[JSBSimPropellerTable] = None
    throttle_mapping: Tuple[float, float, float] = (0.0, 0.25, 1.0)
    max_thrust_scale: Optional[float] = None
    target_thrust_ratio: float = 0.6
    max_thrust_N: Optional[float] = None
    thruster_locations_m: Optional[List[Tuple[float, float, float]]] = None
    thruster_orient_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class JSBSimExportConfig:
    model_name: Optional[str] = None
    alpha_min_deg: float = -10.0
    alpha_max_deg: float = 20.0
    alpha_points: int = 16
    beta_min_deg: float = -10.0
    beta_max_deg: float = 10.0
    beta_points: int = 9
    delta_e_deg: float = 5.0
    delta_a_deg: float = 5.0
    delta_r_deg: float = 5.0
    max_elevator_deflection_deg: float = 20.0
    max_aileron_deflection_deg: float = 20.0
    max_rudder_deflection_deg: float = 20.0
    reference_altitude_m: float = 0.0
    reference_velocity_m_s: Optional[float] = None
    roll_inertia_scale: float = 1.0
    pitch_inertia_scale: float = 1.0
    yaw_inertia_scale: float = 1.0
    include_zero_axes: bool = True
    include_flight_control: bool = True
    include_lateral_derivatives: bool = True
    include_ground_reactions: bool = True
    propulsion: Optional[JSBSimPropulsionConfig] = None


@dataclass
class JSBSimExportResult:
    aircraft_path: str
    engine_path: Optional[str] = None
    propeller_path: Optional[str] = None


@dataclass
class ControlSurfaceInfo:
    name: str
    surface_type: str


@dataclass
class PostStallParams:
    alpha_stall_deg: float = 16.0
    blend_width_deg: float = 4.0
    eta_min: float = 0.3
    eta_power: float = 2.0
    CN_max: float = 1.8
    CA_0: float = 0.02
    CA_k: float = 0.8
    Cm_post_stall_slope: float = -0.02
    Cm_alpha_width_deg: float = 20.0
    Cl_sep_factor: float = 0.3
    Cn_sep_factor: float = 0.3
    CY_sep_factor: float = 0.3


if TYPE_CHECKING:
    class _ADStabilityAnalyzerWithPostStall:
        def __init__(
            self,
            service: AeroSandboxService,
            x_cg: float,
            airspeed: float,
            post_stall_params: Optional[PostStallParams] = None,
        ) -> None:
            ...

        def evaluate(
            self,
            alpha_deg: float,
            beta_deg: float,
            p: float = 0.0,
            q: float = 0.0,
            r: float = 0.0,
            control_deflections_deg: Optional[Dict[str, float]] = None,
        ) -> Dict[str, float]:
            ...
else:
    class _ADStabilityAnalyzerWithPostStall:
        def __init__(
            self,
            service: AeroSandboxService,
            x_cg: float,
            airspeed: float,
            post_stall_params: Optional[PostStallParams] = None,
        ) -> None:
            try:
                import casadi as cas
            except ImportError as exc:
                raise RuntimeError("CasADi is required for AD derivatives. Install with: pip install casadi") from exc

            self._cas: Any = cas
            self.service = service
            self.x_cg = x_cg
            self.airspeed = airspeed
            self.ps = post_stall_params or PostStallParams()

            wing = service.build_wing()
            self.c_ref = wing.mean_aerodynamic_chord()
            self.b_ref = wing.span()

            self.control_surface_names: List[str] = self._detect_control_surfaces()
            self._build_symbolic_functions()

        def _detect_control_surfaces(self) -> List[str]:
            planform = self.service.wing_project.planform
            names: List[str] = []
            for cs in planform.control_surfaces:
                surface_type = cs.surface_type.lower()
                if surface_type == "elevon":
                    names.append(f"{cs.name}_pitch")
                    names.append(f"{cs.name}_roll")
                else:
                    names.append(cs.name)
            return names

        def _build_symbolic_functions(self) -> None:
            cas: Any = self._cas
            ps = self.ps

            sym = cast(Any, cas.SX.sym)
            self.sym_alpha = sym("alpha")  # pyright: ignore[reportArgumentType]
            self.sym_beta = sym("beta")  # pyright: ignore[reportArgumentType]
            self.sym_p = sym("p")  # pyright: ignore[reportArgumentType]
            self.sym_q = sym("q")  # pyright: ignore[reportArgumentType]
            self.sym_r = sym("r")  # pyright: ignore[reportArgumentType]

            self.sym_controls: Dict[str, Any] = {}
            for name in self.control_surface_names:
                self.sym_controls[name] = sym(name)  # pyright: ignore[reportArgumentType]

            input_list = [self.sym_alpha, self.sym_beta, self.sym_p, self.sym_q, self.sym_r]
            input_list.extend([self.sym_controls[name] for name in self.control_surface_names])
            self.sym_inputs = cas.vertcat(*input_list)

            self.input_names = ["alpha", "beta", "p", "q", "r"] + self.control_surface_names
            self.n_inputs = len(self.input_names)

            alpha_deg = self.sym_alpha * 180 / cas.pi
            beta_deg = self.sym_beta * 180 / cas.pi

            alpha_stall_rad = np.radians(ps.alpha_stall_deg)
            blend_width_rad = np.radians(ps.blend_width_deg)

            alpha_abs = cas.fabs(self.sym_alpha)
            w_blend = 0.5 * (1 + cas.tanh((alpha_abs - alpha_stall_rad) / blend_width_rad))
            eta_ctrl = ps.eta_min + (1 - ps.eta_min) * cas.power(1 - w_blend, ps.eta_power)

            stall_sign = cas.if_else(self.sym_alpha >= 0, 1.0, -1.0)
            alpha_stall_signed_rad = alpha_stall_rad * stall_sign
            alpha_stall_signed_deg = alpha_stall_signed_rad * 180 / cas.pi

            control_deflections_zero: Dict[str, float] = {name: 0.0 for name in self.control_surface_names}
            wing_baseline = self.service.build_wing(control_deflections=control_deflections_zero)
            airplane_baseline = asb.Airplane(
                name=self.service.wing_project.name + "_baseline",
                wings=[wing_baseline],
                xyz_ref=[self.x_cg, 0.0, 0.0],
            )
            op_point_baseline = asb.OperatingPoint(
                atmosphere=self.service.atmosphere,
                velocity=self.airspeed,
                alpha=alpha_deg,
                beta=beta_deg,
                p=self.sym_p,
                q=self.sym_q,
                r=self.sym_r,
            )
            aero_baseline = asb.AeroBuildup(airplane=airplane_baseline, op_point=op_point_baseline)
            result_baseline: Dict[str, Any] = aero_baseline.run()

            op_point_stall = asb.OperatingPoint(
                atmosphere=self.service.atmosphere,
                velocity=self.airspeed,
                alpha=alpha_stall_signed_deg,
                beta=beta_deg,
                p=self.sym_p,
                q=self.sym_q,
                r=self.sym_r,
            )
            aero_stall = asb.AeroBuildup(airplane=airplane_baseline, op_point=op_point_stall)
            result_stall: Dict[str, Any] = aero_stall.run()

            CL_att_0: Any = cast(Any, result_baseline["CL"])
            CD_att_0: Any = cast(Any, result_baseline["CD"])
            CY_att_0: Any = cast(Any, result_baseline["CY"])
            Cl_att_0: Any = cast(Any, result_baseline["Cl"])
            Cm_att_0: Any = cast(Any, result_baseline["Cm"])
            Cn_att_0: Any = cast(Any, result_baseline["Cn"])
            Cm_att_stall: Any = cast(Any, result_stall["Cm"])

            control_deflections_cmd: Dict[str, Any] = {}
            for name in self.control_surface_names:
                control_deflections_cmd[name] = self.sym_controls[name] * 180 / cas.pi
            wing_cmd = self.service.build_wing(control_deflections=control_deflections_cmd)
            airplane_cmd = asb.Airplane(
                name=self.service.wing_project.name + "_commanded",
                wings=[wing_cmd],
                xyz_ref=[self.x_cg, 0.0, 0.0],
            )
            op_point_cmd = asb.OperatingPoint(
                atmosphere=self.service.atmosphere,
                velocity=self.airspeed,
                alpha=alpha_deg,
                beta=beta_deg,
                p=self.sym_p,
                q=self.sym_q,
                r=self.sym_r,
            )
            aero_cmd = asb.AeroBuildup(airplane=airplane_cmd, op_point=op_point_cmd)
            result_cmd: Dict[str, Any] = aero_cmd.run()

            CL_att_cmd: Any = cast(Any, result_cmd["CL"])
            CD_att_cmd: Any = cast(Any, result_cmd["CD"])
            CY_att_cmd: Any = cast(Any, result_cmd["CY"])
            Cl_att_cmd: Any = cast(Any, result_cmd["Cl"])
            Cm_att_cmd: Any = cast(Any, result_cmd["Cm"])
            Cn_att_cmd: Any = cast(Any, result_cmd["Cn"])

            dCL_ctrl = cast(Any, CL_att_cmd) - cast(Any, CL_att_0)  # pyright: ignore[reportOperatorIssue]
            dCD_ctrl = cast(Any, CD_att_cmd) - cast(Any, CD_att_0)  # pyright: ignore[reportOperatorIssue]
            dCY_ctrl = cast(Any, CY_att_cmd) - cast(Any, CY_att_0)  # pyright: ignore[reportOperatorIssue]
            dCl_ctrl = cast(Any, Cl_att_cmd) - cast(Any, Cl_att_0)  # pyright: ignore[reportOperatorIssue]
            dCm_ctrl = cast(Any, Cm_att_cmd) - cast(Any, Cm_att_0)  # pyright: ignore[reportOperatorIssue]
            dCn_ctrl = cast(Any, Cn_att_cmd) - cast(Any, Cn_att_0)  # pyright: ignore[reportOperatorIssue]

            CL_att_faded = cast(Any, CL_att_0) + cast(Any, eta_ctrl) * cast(Any, dCL_ctrl)  # pyright: ignore[reportOperatorIssue]
            CD_att_faded = cast(Any, CD_att_0) + cast(Any, eta_ctrl) * cast(Any, dCD_ctrl)  # pyright: ignore[reportOperatorIssue]
            CY_att_faded = cast(Any, CY_att_0) + cast(Any, eta_ctrl) * cast(Any, dCY_ctrl)  # pyright: ignore[reportOperatorIssue]
            Cl_att_faded = cast(Any, Cl_att_0) + cast(Any, eta_ctrl) * cast(Any, dCl_ctrl)  # pyright: ignore[reportOperatorIssue]
            Cm_att_faded = cast(Any, Cm_att_0) + cast(Any, eta_ctrl) * cast(Any, dCm_ctrl)  # pyright: ignore[reportOperatorIssue]
            Cn_att_faded = cast(Any, Cn_att_0) + cast(Any, eta_ctrl) * cast(Any, dCn_ctrl)  # pyright: ignore[reportOperatorIssue]

            sin_alpha = cas.sin(self.sym_alpha)
            cos_alpha = cas.cos(self.sym_alpha)
            sin_2alpha = cas.sin(2 * self.sym_alpha)

            CN_sep = ps.CN_max * sin_2alpha  # pyright: ignore[reportOperatorIssue]
            CA_sep = ps.CA_0 + ps.CA_k * cas.power(sin_alpha, 2)  # pyright: ignore[reportOperatorIssue]

            CL_sep = CN_sep * cos_alpha - CA_sep * sin_alpha  # pyright: ignore[reportOperatorIssue]
            CD_sep_raw = CN_sep * sin_alpha + CA_sep * cos_alpha  # pyright: ignore[reportOperatorIssue]
            CD_sep = cas.fmax(CD_sep_raw, 0.02)

            CY_sep = ps.CY_sep_factor * CY_att_0

            alpha_excess = self.sym_alpha - alpha_stall_rad * cas.sign(self.sym_alpha)
            Cm_post_stall_delta = ps.Cm_post_stall_slope * (alpha_abs - alpha_stall_rad) * 180 / cas.pi
            Cm_sep = Cm_att_stall + Cm_post_stall_delta * cas.tanh(alpha_excess / np.radians(ps.Cm_alpha_width_deg))

            Cl_sep = ps.Cl_sep_factor * Cl_att_0
            Cn_sep = ps.Cn_sep_factor * Cn_att_0

            CL_final = (1 - w_blend) * CL_att_faded + w_blend * CL_sep
            CD_final = (1 - w_blend) * CD_att_faded + w_blend * CD_sep
            CY_final = (1 - w_blend) * CY_att_faded + w_blend * CY_sep
            Cl_final = (1 - w_blend) * Cl_att_faded + w_blend * Cl_sep
            Cm_final = (1 - w_blend) * Cm_att_faded + w_blend * Cm_sep
            Cn_final = (1 - w_blend) * Cn_att_faded + w_blend * Cn_sep

            self.sym_outputs = cas.vertcat(CL_final, CD_final, CY_final, Cl_final, Cm_final, Cn_final)
            self.output_names = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]

            self.sym_jacobian = cas.jacobian(self.sym_outputs, self.sym_inputs)

            self.combined_func = cas.Function(
                "aero_poststall",
                [self.sym_inputs],
                [self.sym_outputs, self.sym_jacobian, w_blend, eta_ctrl],
                ["inputs"],
                ["outputs", "jacobian", "w_blend", "eta_ctrl"],
            )

        def evaluate(
            self,
            alpha_deg: float,
            beta_deg: float,
            p: float = 0.0,
            q: float = 0.0,
            r: float = 0.0,
            control_deflections_deg: Optional[Dict[str, float]] = None,
        ) -> Dict[str, float]:
            if control_deflections_deg is None:
                control_deflections_deg = {}  # pyright: ignore[reportMissingTypeArgument]
            control_deflections: Dict[str, float] = control_deflections_deg

            alpha_rad = np.radians(alpha_deg)
            beta_rad = np.radians(beta_deg)

            inputs = [alpha_rad, beta_rad, p, q, r]
            for name in self.control_surface_names:  # pyright: ignore[reportGeneralTypeIssues]
                deflection_rad = np.radians(control_deflections.get(name, 0.0))
                inputs.append(deflection_rad)
            inputs = np.array(inputs)

            outputs, jacobian, w_blend, eta_ctrl = self.combined_func(inputs)

            outputs = np.array(outputs).flatten()
            jacobian = np.array(jacobian)
            w_blend = float(w_blend)
            eta_ctrl = float(eta_ctrl)

            result: Dict[str, float] = {
                "alpha": alpha_deg,
                "beta": beta_deg,
                "airspeed": self.airspeed,
                "w_blend": w_blend,
                "eta_ctrl": eta_ctrl,
            }

            for i, name in enumerate(self.output_names):
                result[name] = float(outputs[i])

            for name in self.control_surface_names:
                result[f"d_{name}"] = control_deflections.get(name, 0.0)

            V = float(self.airspeed)
            b = float(self.b_ref)
            c = float(self.c_ref)
            p_norm = b / (2 * V)  # pyright: ignore[reportOperatorIssue]
            q_norm = c / (2 * V)  # pyright: ignore[reportOperatorIssue]
            r_norm = b / (2 * V)  # pyright: ignore[reportOperatorIssue]

            for i, coef in enumerate(self.output_names):
                for j, inp in enumerate(self.input_names):
                    deriv_value = float(jacobian[i, j])

                    if inp == "p":
                        deriv_value /= p_norm
                        deriv_name = f"{coef}p"
                    elif inp == "q":
                        deriv_value /= q_norm
                        deriv_name = f"{coef}q"
                    elif inp == "r":
                        deriv_value /= r_norm
                        deriv_name = f"{coef}r"
                    elif inp == "alpha":
                        deriv_name = f"{coef}a"
                    elif inp == "beta":
                        deriv_name = f"{coef}b"
                    else:
                        short_inp = inp.replace("_pitch", "e").replace("_roll", "a")
                        short_inp = short_inp.replace("Surface1", "elv")
                        deriv_name = f"{coef}d{short_inp}"

                    result[deriv_name] = deriv_value

            return result

def _sanitize_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._- ").strip() or "FlyingWingProject"




def _format_float(value: float, decimals: int = 6) -> str:
    return f"{value:.{decimals}f}"


def _format_table_data(xs: np.ndarray, ys: np.ndarray, decimals: int = 6) -> str:
    rows = [f"{x:.{decimals}f} {y:.{decimals}f}" for x, y in zip(xs, ys)]
    return "\n" + "\n".join(rows) + "\n"


def _estimate_inertia(
    mass_kg: float,
    span_m: float,
    mac_m: float,
    roll_scale: float = 1.0,
    pitch_scale: float = 1.0,
    yaw_scale: float = 1.0,
) -> Tuple[float, float, float]:
    ixx = roll_scale * mass_kg * span_m**2 / 12.0
    iyy = pitch_scale * mass_kg * mac_m**2 / 12.0
    izz = yaw_scale * mass_kg * (span_m**2 + mac_m**2) / 12.0
    return ixx, iyy, izz


def _default_thruster_locations(
    engine_count: int,
    span_m: float,
    x_cg: float,
) -> List[Tuple[float, float, float]]:
    if engine_count <= 1:
        return [(x_cg, 0.0, 0.0)]
    half_span = 0.5 * span_m
    y_extent = 0.3 * half_span
    ys = np.linspace(-y_extent, y_extent, engine_count)
    return [(x_cg, float(y), 0.0) for y in ys]


def _array_to_scalar(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(arr.reshape(-1)[0])
    except Exception:
        return 0.0


def _find_pitch_control(project: Project) -> Optional[ControlSurfaceInfo]:
    for surface in project.wing.planform.control_surfaces:
        surface_type = surface.surface_type.lower()
        if surface_type in ("elevon", "elevator"):
            return ControlSurfaceInfo(surface.name, surface_type)
    return None


def _find_roll_control(project: Project) -> Optional[ControlSurfaceInfo]:
    roll_candidate = None
    for surface in project.wing.planform.control_surfaces:
        surface_type = surface.surface_type.lower()
        if surface_type == "aileron":
            return ControlSurfaceInfo(surface.name, surface_type)
        if surface_type == "elevon":
            roll_candidate = ControlSurfaceInfo(surface.name, surface_type)
    return roll_candidate


def _find_yaw_control(project: Project) -> Optional[ControlSurfaceInfo]:
    for surface in project.wing.planform.control_surfaces:
        surface_type = surface.surface_type.lower()
        if surface_type == "rudder":
            return ControlSurfaceInfo(surface.name, surface_type)
    return None


def _roll_control_key(surface: ControlSurfaceInfo) -> str:
    if surface.surface_type == "elevon":
        return f"{surface.name}_roll"
    return surface.name


def _control_derivative_key(surface: ControlSurfaceInfo, axis: str) -> str:
    if surface.surface_type == "elevon":
        input_name = f"{surface.name}_{axis}"
    else:
        input_name = surface.name
    short_inp = input_name.replace("_pitch", "e").replace("_roll", "a")
    short_inp = short_inp.replace("Surface1", "elv")
    return short_inp


def _build_aero_tables(
    project: Project,
    config: JSBSimExportConfig,
    x_cg: float,
    pitch_control: Optional[ControlSurfaceInfo],
    service: Optional[AeroSandboxService] = None,
    analyzer: Optional[_ADStabilityAnalyzerWithPostStall] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    service = service or AeroSandboxService(project)
    velocity = config.reference_velocity_m_s or service.design_velocity()
    analyzer = analyzer or _ADStabilityAnalyzerWithPostStall(service, x_cg, velocity)
    alpha_deg = np.linspace(config.alpha_min_deg, config.alpha_max_deg, max(2, config.alpha_points))
    cl = []
    cd = []
    cm = []
    for alpha in alpha_deg:
        result = analyzer.evaluate(alpha_deg=float(alpha), beta_deg=0.0)
        cl.append(float(result.get("CL", 0.0)))
        cd.append(float(result.get("CD", 0.0)))
        cm.append(float(result.get("Cm", 0.0)))

    cl = np.asarray(cl, dtype=float)
    cd = np.asarray(cd, dtype=float)
    cm = np.asarray(cm, dtype=float)

    cl_de = None
    cm_de = None
    if pitch_control and config.delta_e_deg > 0:
        alpha_ref = service.estimate_alpha_for_cl()
        result = analyzer.evaluate(alpha_deg=float(alpha_ref), beta_deg=0.0)
        key = _control_derivative_key(pitch_control, "pitch")
        cl_de = result.get(f"CLd{key}")
        cm_de = result.get(f"Cmd{key}")

    return alpha_deg, cl, cd, cm, cl_de, cm_de


def _build_lateral_tables(
    project: Project,
    config: JSBSimExportConfig,
    x_cg: float,
    roll_control: Optional[ControlSurfaceInfo],
    yaw_control: Optional[ControlSurfaceInfo],
    service: Optional[AeroSandboxService] = None,
    analyzer: Optional[_ADStabilityAnalyzerWithPostStall] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float], Optional[float], Optional[float]]:
    service = service or AeroSandboxService(project)
    velocity = config.reference_velocity_m_s or service.design_velocity()
    analyzer = analyzer or _ADStabilityAnalyzerWithPostStall(service, x_cg, velocity)
    beta_deg = np.linspace(config.beta_min_deg, config.beta_max_deg, max(2, config.beta_points))
    alpha_ref = service.estimate_alpha_for_cl()

    cy_beta = []
    cl_beta = []
    cn_beta = []
    for beta in beta_deg:
        result = analyzer.evaluate(alpha_deg=float(alpha_ref), beta_deg=float(beta))
        cy_beta.append(float(result.get("CY", 0.0)))
        cl_beta.append(float(result.get("Cl", 0.0)))
        cn_beta.append(float(result.get("Cn", 0.0)))

    cl_da = None
    if roll_control and config.delta_a_deg > 0:
        result = analyzer.evaluate(alpha_deg=float(alpha_ref), beta_deg=0.0)
        key = _control_derivative_key(roll_control, "roll")
        cl_da = result.get(f"Cld{key}")

    cn_dr = None
    cy_dr = None
    if yaw_control and config.delta_r_deg > 0:
        result = analyzer.evaluate(alpha_deg=float(alpha_ref), beta_deg=0.0)
        key = _control_derivative_key(yaw_control, "yaw")
        cn_dr = result.get(f"Cnd{key}")
        cy_dr = result.get(f"CYd{key}")

    return (
        beta_deg,
        np.asarray(cy_beta, dtype=float),
        np.asarray(cl_beta, dtype=float),
        np.asarray(cn_beta, dtype=float),
        cl_da,
        cn_dr,
        cy_dr,
    )


def _default_propeller_table() -> JSBSimPropellerTable:
    advance_ratio = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
    ct = np.array([0.09, 0.07, 0.04, 0.01, 0.0], dtype=float)
    cp = np.array([0.06, 0.05, 0.03, 0.01, 0.0], dtype=float)
    return JSBSimPropellerTable(advance_ratio=advance_ratio, ct=ct, cp=cp)


def _normalized_propeller_table(table: JSBSimPropellerTable) -> JSBSimPropellerTable:
    """
    Normalize propeller CT/CP tables for JSBSim stability.
    
    Ensures:
    1. Sorted by advance ratio
    2. Monotonic decay after peak - once CT/CP goes below threshold, stays there
    3. Minimum CP floor to prevent torque balance instability
    """
    advance_ratio = np.asarray(table.advance_ratio, dtype=float)
    ct = np.asarray(table.ct, dtype=float)
    cp = np.asarray(table.cp, dtype=float)
    if advance_ratio.size < 2 or ct.size != advance_ratio.size or cp.size != advance_ratio.size:
        return _default_propeller_table()
    
    # Sort by advance ratio
    order = np.argsort(advance_ratio)
    j_sorted = advance_ratio[order]
    ct_sorted = ct[order]
    cp_sorted = cp[order]
    
    # Enforce monotonic decay after peak for CT
    # Once CT drops below a threshold (e.g., 10% of max), it stays at zero
    ct_max = np.max(ct_sorted)
    ct_threshold = 0.1 * ct_max if ct_max > 0 else 0.01
    ct_fixed = np.copy(ct_sorted)
    hit_floor = False
    for i in range(len(ct_fixed)):
        if hit_floor:
            ct_fixed[i] = 0.0
        elif ct_fixed[i] < ct_threshold and i > 0:
            # Check if this is a genuine decay (not just the start)
            if ct_sorted[i-1] > ct_threshold:
                hit_floor = True
                ct_fixed[i] = 0.0
    
    # Enforce minimum CP floor - CP can't drop to zero while CT is positive
    # This prevents torque balance instability
    cp_min = 0.001  # Minimum power coefficient
    cp_fixed = np.copy(cp_sorted)
    for i in range(len(cp_fixed)):
        if ct_fixed[i] > 0.0:
            cp_fixed[i] = max(cp_fixed[i], cp_min)
        elif ct_fixed[i] <= 0.0:
            # If CT is zero, CP should also be minimal (windmilling)
            cp_fixed[i] = max(cp_fixed[i], cp_min * 0.1)
    
    return JSBSimPropellerTable(
        advance_ratio=j_sorted,
        ct=ct_fixed,
        cp=cp_fixed,
    )


def _estimate_propeller_ixx(diameter_in: float, num_blades: int = 2) -> float:
    """
    Estimate propeller rotational inertia (Ixx) in slug*ft^2.
    
    Uses empirical relationship for small electric propellers.
    A typical small prop has mass roughly proportional to D^2.5
    and inertia roughly proportional to D^4.5 (mass * r^2).
    
    Args:
        diameter_in: Propeller diameter in inches
        num_blades: Number of blades
        
    Returns:
        Ixx in slug*ft^2
    """
    # Empirical: a 12" 2-blade prop weighs ~40g, 10" ~25g, 14" ~60g
    # Mass scales roughly as D^2.5
    d_ref = 12.0  # reference diameter inches
    m_ref_kg = 0.040  # reference mass kg for 2 blades
    
    mass_kg = m_ref_kg * (diameter_in / d_ref) ** 2.5 * (num_blades / 2.0)
    
    # Convert to slug (1 slug = 14.5939 kg)
    mass_slug = mass_kg / 14.5939
    
    # Radius in feet
    radius_ft = (diameter_in / 2.0) / 12.0
    
    # For a prop, Ixx ≈ (1/3) * m * r^2 per blade, but blades are thin
    # Use 0.4 as factor for realistic prop blade distribution
    ixx = 0.4 * mass_slug * radius_ft ** 2
    
    # Minimum reasonable value to prevent numerical instability
    return max(ixx, 0.0001)


def _estimate_ct_factor(
    table: JSBSimPropellerTable,
    prop_diam_in: float,
    engine_power_w: float,
    engine_count: int,
    mass_kg: float,
    target_thrust_ratio: float,
    max_thrust_N: Optional[float],
) -> float:
    rho = 1.225
    diameter_m = prop_diam_in * 0.0254
    if diameter_m <= 1e-3 or engine_power_w <= 0.0 or mass_kg <= 0.0:
        return 1.0
    cp0 = float(table.cp[0]) if table.cp.size > 0 else 0.0
    ct0 = float(table.ct[0]) if table.ct.size > 0 else 0.0
    if cp0 <= 1e-6 or ct0 <= 1e-6:
        return 1.0
    n_rev_s = (engine_power_w / (cp0 * rho * diameter_m**5)) ** (1.0 / 3.0)
    thrust_per_engine = ct0 * rho * n_rev_s**2 * diameter_m**4
    total_thrust = thrust_per_engine * max(1, engine_count)
    if max_thrust_N is not None and np.isfinite(max_thrust_N) and max_thrust_N > 0:
        target_thrust = float(max_thrust_N)
    else:
        target_thrust = mass_kg * 9.81 * max(0.1, target_thrust_ratio)
    factor = target_thrust / max(total_thrust, 1e-6)
    return float(np.clip(factor, 0.05, 1.0))


def _build_ground_reactions(
    root: ET.Element,
    span_m: float,
    mac_m: float,
    x_cg: float,
    mass_kg: float,
) -> None:
    """
    Add ground_reactions section with flying wing skid configuration.
    
    Creates a belly skid plus two wingtip skids for ground handling.
    Uses JSBSim BOGEY contact type with spring-damper properties.
    
    Args:
        root: XML root element to append to
        span_m: Wing span in meters
        mac_m: Mean aerodynamic chord in meters  
        x_cg: CG x-position in meters
        mass_kg: Aircraft mass in kg
    """
    ground_reactions = ET.SubElement(root, "ground_reactions")
    
    # Conservative skid stiffness/damping for stable ground contact
    center_spring = max(1200.0, mass_kg * 30.0)  # N/m
    tip_spring = center_spring * 0.5
    center_damp = max(120.0, center_spring * 0.15)  # N*s/m
    tip_damp = center_damp * 0.5
    
    # Position calculations
    half_span = span_m / 2.0
    # Center skid: under CG, at bottom of fuselage (Z negative per JSBSim ground_reactions examples)
    center_x = x_cg
    center_z = -max(0.03, mac_m * 0.04)
    
    # Wingtip skids: near wingtips, under CG
    tip_x = x_cg
    tip_y = half_span * 0.85  # 85% of half-span
    tip_z = -max(0.02, mac_m * 0.03)
    
    def add_contact(name: str, x: float, y: float, z: float, 
                    spring: float, damp: float, is_center: bool = False) -> None:
        """Add a BOGEY contact point."""
        contact = ET.SubElement(ground_reactions, "contact", type="BOGEY", name=name)
        
        loc = ET.SubElement(contact, "location", unit="M")
        ET.SubElement(loc, "x").text = _format_float(x, 6)
        ET.SubElement(loc, "y").text = _format_float(y, 6)
        ET.SubElement(loc, "z").text = _format_float(z, 6)
        
        # Static friction (sliding on belly)
        ET.SubElement(contact, "static_friction").text = "0.50"
        # Dynamic friction
        ET.SubElement(contact, "dynamic_friction").text = "0.40"
        # Rolling friction
        ET.SubElement(contact, "rolling_friction").text = "0.40"
        
        # Spring coefficient (N/M)
        ET.SubElement(contact, "spring_coeff", unit="N/M").text = _format_float(spring, 2)
        # Damping coefficient (N/M/SEC)
        ET.SubElement(contact, "damping_coeff", unit="N/M/SEC").text = _format_float(damp, 2)
        # Max steering angle (skids don't steer)
        ET.SubElement(contact, "max_steer", unit="DEG").text = "0.0"
        # Brake group (no brakes on skids)
        ET.SubElement(contact, "brake_group").text = "NONE"
        # Retractable (fixed gear)
        ET.SubElement(contact, "retractable").text = "0"
    
    # Center belly skid
    add_contact("CENTER_SKID", center_x, 0.0, center_z, 
                center_spring, center_damp, is_center=True)
    
    # Left wingtip skid  
    add_contact("LEFT_SKID", tip_x, -tip_y, tip_z,
                tip_spring, tip_damp)
    
    # Right wingtip skid
    add_contact("RIGHT_SKID", tip_x, tip_y, tip_z,
                tip_spring, tip_damp)


def generate_jsbsim_xml(project: Project, config: Optional[JSBSimExportConfig] = None) -> ET.Element:
    if config is None:
        config = JSBSimExportConfig()

    model_name = config.model_name or project.wing.name or "FlyingWingProject"
    model_name = _sanitize_name(model_name)

    service = AeroSandboxService(project)
    wing = service.build_wing()
    span_m = project.wing.planform.actual_span()
    area_m2 = project.wing.planform.actual_area()
    mac_m = wing.mean_aerodynamic_chord()
    x_np = wing.aerodynamic_center()[0]

    static_margin = project.wing.twist_trim.static_margin_percent
    x_cg = x_np - (static_margin / 100.0) * mac_m

    mass_kg = project.wing.twist_trim.gross_takeoff_weight_kg
    ixx, iyy, izz = _estimate_inertia(
        mass_kg,
        span_m,
        mac_m,
        roll_scale=config.roll_inertia_scale,
        pitch_scale=config.pitch_inertia_scale,
        yaw_scale=config.yaw_inertia_scale,
    )

    pitch_control = _find_pitch_control(project)
    roll_control = _find_roll_control(project)
    yaw_control = _find_yaw_control(project)

    velocity = config.reference_velocity_m_s or service.design_velocity()
    ad_analyzer = _ADStabilityAnalyzerWithPostStall(service, x_cg, velocity)

    alpha_deg, cl, cd, cm, cl_de, cm_de = _build_aero_tables(
        project,
        config,
        x_cg,
        pitch_control,
        service=service,
        analyzer=ad_analyzer,
    )
    alpha_rad = np.radians(alpha_deg)

    cl_q = []
    cm_q = []
    cl_p = []
    cn_r = []
    for alpha in alpha_deg:
        result = ad_analyzer.evaluate(alpha_deg=float(alpha), beta_deg=0.0)
        cl_q.append(float(result.get("CLq", 0.0)))
        cm_q.append(float(result.get("Cmq", 0.0)))
        cl_p.append(float(result.get("Clp", 0.0)))
        cn_r.append(float(result.get("Cnr", 0.0)))

    cl_q = np.asarray(cl_q, dtype=float)
    cm_q = np.asarray(cm_q, dtype=float)
    cl_p = np.asarray(cl_p, dtype=float)
    cn_r = np.asarray(cn_r, dtype=float)

    lateral = None
    if config.include_lateral_derivatives:
        lateral = _build_lateral_tables(
            project,
            config,
            x_cg,
            roll_control,
            yaw_control,
            service=service,
            analyzer=ad_analyzer,
        )

    root = ET.Element(
        "fdm_config",
        {
            "name": model_name,
            "version": "2.0",
            "release": "BETA",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://jsbsim.sourceforge.net/JSBSim.xsd",
        },
    )

    header = ET.SubElement(root, "fileheader")
    ET.SubElement(header, "author").text = "Flying Wing Tool"
    ET.SubElement(header, "description").text = "JSBSim export (flight dynamics focus)"

    metrics = ET.SubElement(root, "metrics")
    ET.SubElement(metrics, "wingarea", unit="M2").text = _format_float(area_m2, 6)
    ET.SubElement(metrics, "wingspan", unit="M").text = _format_float(span_m, 6)
    ET.SubElement(metrics, "chord", unit="M").text = _format_float(mac_m, 6)

    aerorp = ET.SubElement(metrics, "location", name="AERORP", unit="M")
    ET.SubElement(aerorp, "x").text = _format_float(x_cg, 6)
    ET.SubElement(aerorp, "y").text = "0.000000"
    ET.SubElement(aerorp, "z").text = "0.000000"

    mass_balance = ET.SubElement(root, "mass_balance", negated_crossproduct_inertia="true")
    ET.SubElement(mass_balance, "ixx", unit="KG*M2").text = _format_float(ixx, 6)
    ET.SubElement(mass_balance, "iyy", unit="KG*M2").text = _format_float(iyy, 6)
    ET.SubElement(mass_balance, "izz", unit="KG*M2").text = _format_float(izz, 6)
    ET.SubElement(mass_balance, "ixy", unit="KG*M2").text = "0.000000"
    ET.SubElement(mass_balance, "ixz", unit="KG*M2").text = "0.000000"
    ET.SubElement(mass_balance, "iyz", unit="KG*M2").text = "0.000000"
    empty_weight_lbs = mass_kg * 2.2046226218
    ET.SubElement(mass_balance, "emptywt", unit="LBS").text = _format_float(empty_weight_lbs, 6)

    cg_loc = ET.SubElement(mass_balance, "location", name="CG", unit="M")
    ET.SubElement(cg_loc, "x").text = _format_float(x_cg, 6)
    ET.SubElement(cg_loc, "y").text = "0.000000"
    ET.SubElement(cg_loc, "z").text = "0.000000"

    if config.propulsion is not None:
        prop = config.propulsion
        engine_file: str = prop.engine_file or f"{model_name}_engine"
        prop_file: str = prop.propeller_file or f"{model_name}_prop"
        propulsion = ET.SubElement(root, "propulsion")
        locations = prop.thruster_locations_m or _default_thruster_locations(
            max(1, prop.engine_count),
            span_m,
            x_cg,
        )
        roll_deg, pitch_deg, yaw_deg = prop.thruster_orient_deg
        for idx in range(max(1, prop.engine_count)):
            engine_elem = ET.SubElement(propulsion, "engine", file=engine_file)
            thruster = ET.SubElement(engine_elem, "thruster", file=prop_file)
            location = locations[idx] if idx < len(locations) else locations[-1]
            loc_elem = ET.SubElement(thruster, "location", unit="M")
            ET.SubElement(loc_elem, "x").text = _format_float(location[0], 6)
            ET.SubElement(loc_elem, "y").text = _format_float(location[1], 6)
            ET.SubElement(loc_elem, "z").text = _format_float(location[2], 6)
            orient = ET.SubElement(thruster, "orient", unit="DEG")
            ET.SubElement(orient, "roll").text = _format_float(roll_deg, 6)
            ET.SubElement(orient, "pitch").text = _format_float(pitch_deg, 6)
            ET.SubElement(orient, "yaw").text = _format_float(yaw_deg, 6)

    # Ground reactions (landing gear / skids)
    if config.include_ground_reactions:
        _build_ground_reactions(root, span_m, mac_m, x_cg, mass_kg)

    has_propulsion = config.propulsion is not None
    if config.include_flight_control and (pitch_control or roll_control or yaw_control or has_propulsion):
        fcs = ET.SubElement(root, "flight_control", name="FCS: Flying Wing")

        if pitch_control:
            pitch_channel = ET.SubElement(fcs, "channel", name="Pitch")
            pitch_input = ET.SubElement(pitch_channel, "pure_gain", name="fcs/elevator-cmd-norm")
            ET.SubElement(pitch_input, "input").text = "/controls/flight/elevator"
            ET.SubElement(pitch_input, "gain").text = "1.0"
            ET.SubElement(pitch_input, "output").text = "fcs/elevator-cmd-norm"
            elevator = ET.SubElement(pitch_channel, "aerosurface_scale", name="Elevator")
            ET.SubElement(elevator, "input").text = "fcs/elevator-cmd-norm"
            elevator_range = ET.SubElement(elevator, "range")
            max_deflection_rad = np.radians(config.max_elevator_deflection_deg)
            ET.SubElement(elevator_range, "min").text = _format_float(-max_deflection_rad, 6)
            ET.SubElement(elevator_range, "max").text = _format_float(max_deflection_rad, 6)
            ET.SubElement(elevator, "output").text = "fcs/elevator-pos-rad"

        if roll_control:
            roll_channel = ET.SubElement(fcs, "channel", name="Roll")
            roll_input = ET.SubElement(roll_channel, "pure_gain", name="fcs/aileron-cmd-norm")
            ET.SubElement(roll_input, "input").text = "/controls/flight/aileron"
            ET.SubElement(roll_input, "gain").text = "-1.0"
            ET.SubElement(roll_input, "output").text = "fcs/aileron-cmd-norm"
            aileron = ET.SubElement(roll_channel, "aerosurface_scale", name="Aileron")
            ET.SubElement(aileron, "input").text = "fcs/aileron-cmd-norm"
            aileron_range = ET.SubElement(aileron, "range")
            max_deflection_rad = np.radians(config.max_aileron_deflection_deg)
            ET.SubElement(aileron_range, "min").text = _format_float(-max_deflection_rad, 6)
            ET.SubElement(aileron_range, "max").text = _format_float(max_deflection_rad, 6)
            ET.SubElement(aileron, "output").text = "fcs/aileron-pos-rad"

        if yaw_control:
            yaw_channel = ET.SubElement(fcs, "channel", name="Yaw")
            yaw_input = ET.SubElement(yaw_channel, "pure_gain", name="fcs/rudder-cmd-norm")
            ET.SubElement(yaw_input, "input").text = "/controls/flight/rudder"
            ET.SubElement(yaw_input, "gain").text = "1.0"
            ET.SubElement(yaw_input, "output").text = "fcs/rudder-cmd-norm"
            rudder = ET.SubElement(yaw_channel, "aerosurface_scale", name="Rudder")
            ET.SubElement(rudder, "input").text = "fcs/rudder-cmd-norm"
            rudder_range = ET.SubElement(rudder, "range")
            max_deflection_rad = np.radians(config.max_rudder_deflection_deg)
            ET.SubElement(rudder_range, "min").text = _format_float(-max_deflection_rad, 6)
            ET.SubElement(rudder_range, "max").text = _format_float(max_deflection_rad, 6)
            ET.SubElement(rudder, "output").text = "fcs/rudder-pos-rad"

        if has_propulsion:
            engine_count = 1
            if config.propulsion is not None:
                engine_count = max(1, int(config.propulsion.engine_count))
            throttle_channel = ET.SubElement(fcs, "channel", name="Throttle")
            throttle_cmd = ET.SubElement(throttle_channel, "scheduled_gain", name="fcs/throttle-cmd-norm")
            ET.SubElement(throttle_cmd, "input").text = "/controls/engines/engine[0]/throttle"
            throttle_table = ET.SubElement(throttle_cmd, "table")
            ET.SubElement(throttle_table, "independentVar").text = "/controls/engines/engine[0]/throttle"
            t_min, t_mid, t_max = (0.0, 0.25, 1.0)
            if config.propulsion is not None:
                t_min, t_mid, t_max = config.propulsion.throttle_mapping
            # More conservative throttle curve - quadratic response
            # First 20% of input gives almost nothing, then gradual ramp
            throttle_data = ET.SubElement(throttle_table, "tableData")
            throttle_data.text = _format_table_data(
                np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                np.array([t_min, 0.02, 0.10, 0.30, 0.60, t_max]),
                6,
            )

            for idx in range(engine_count):
                engine_throttle = ET.SubElement(
                    throttle_channel,
                    "pure_gain",
                    name=f"propulsion/engine[{idx}]/throttle",
                )
                ET.SubElement(engine_throttle, "input").text = "fcs/throttle-cmd-norm"
                ET.SubElement(engine_throttle, "gain").text = "1.0"
                ET.SubElement(engine_throttle, "output").text = f"propulsion/engine[{idx}]/throttle"

    aerodynamics = ET.SubElement(root, "aerodynamics")

    lift_axis = ET.SubElement(aerodynamics, "axis", name="LIFT")
    lift_fn = ET.SubElement(lift_axis, "function", name="aero/coefficient/CLalpha")
    ET.SubElement(lift_fn, "description").text = "Lift due to alpha"
    lift_prod = ET.SubElement(lift_fn, "product")
    ET.SubElement(lift_prod, "property").text = "aero/qbar-psf"
    ET.SubElement(lift_prod, "property").text = "metrics/Sw-sqft"
    lift_table = ET.SubElement(lift_prod, "table")
    ET.SubElement(lift_table, "independentVar").text = "aero/alpha-rad"
    lift_table_data = ET.SubElement(lift_table, "tableData")
    lift_table_data.text = _format_table_data(alpha_rad, cl, 6)

    if cl_de is not None:
        lift_ctrl = ET.SubElement(lift_axis, "function", name="aero/coefficient/CLde")
        ET.SubElement(lift_ctrl, "description").text = "Lift due to elevator"
        lift_ctrl_prod = ET.SubElement(lift_ctrl, "product")
        ET.SubElement(lift_ctrl_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(lift_ctrl_prod, "property").text = "metrics/Sw-sqft"
        lift_ctrl_table = ET.SubElement(lift_ctrl_prod, "table")
        ET.SubElement(lift_ctrl_table, "independentVar").text = "fcs/elevator-pos-rad"
        lift_ctrl_data = ET.SubElement(lift_ctrl_table, "tableData")
        max_deflection_rad = np.radians(config.max_elevator_deflection_deg)
        lift_ctrl_data.text = _format_table_data(
            np.array([-max_deflection_rad, 0.0, max_deflection_rad]),
            np.array([
                cl_de * -max_deflection_rad,
                0.0,
                cl_de * max_deflection_rad,
            ]),
            6,
        )

    lift_q = ET.SubElement(lift_axis, "function", name="aero/coefficient/CLq")
    ET.SubElement(lift_q, "description").text = "Lift due to pitch rate"
    lift_q_prod = ET.SubElement(lift_q, "product")
    ET.SubElement(lift_q_prod, "property").text = "aero/qbar-psf"
    ET.SubElement(lift_q_prod, "property").text = "metrics/Sw-sqft"
    ET.SubElement(lift_q_prod, "property").text = "velocities/q-aero-rad_sec"
    ET.SubElement(lift_q_prod, "property").text = "aero/ci2vel"
    lift_q_table = ET.SubElement(lift_q_prod, "table")
    ET.SubElement(lift_q_table, "independentVar").text = "aero/alpha-rad"
    lift_q_data = ET.SubElement(lift_q_table, "tableData")
    lift_q_data.text = _format_table_data(alpha_rad, cl_q, 6)

    drag_axis = ET.SubElement(aerodynamics, "axis", name="DRAG")
    drag_fn = ET.SubElement(drag_axis, "function", name="aero/coefficient/CDalpha")
    ET.SubElement(drag_fn, "description").text = "Drag due to alpha"
    drag_prod = ET.SubElement(drag_fn, "product")
    ET.SubElement(drag_prod, "property").text = "aero/qbar-psf"
    ET.SubElement(drag_prod, "property").text = "metrics/Sw-sqft"
    drag_table = ET.SubElement(drag_prod, "table")
    ET.SubElement(drag_table, "independentVar").text = "aero/alpha-rad"
    drag_table_data = ET.SubElement(drag_table, "tableData")
    drag_table_data.text = _format_table_data(alpha_rad, cd, 6)

    pitch_axis = ET.SubElement(aerodynamics, "axis", name="PITCH")
    pitch_fn = ET.SubElement(pitch_axis, "function", name="aero/coefficient/CMalpha")
    ET.SubElement(pitch_fn, "description").text = "Pitch moment due to alpha"
    pitch_prod = ET.SubElement(pitch_fn, "product")
    ET.SubElement(pitch_prod, "property").text = "aero/qbar-psf"
    ET.SubElement(pitch_prod, "property").text = "metrics/Sw-sqft"
    ET.SubElement(pitch_prod, "property").text = "metrics/cbarw-ft"
    pitch_table = ET.SubElement(pitch_prod, "table")
    ET.SubElement(pitch_table, "independentVar").text = "aero/alpha-rad"
    pitch_table_data = ET.SubElement(pitch_table, "tableData")
    pitch_table_data.text = _format_table_data(alpha_rad, cm, 6)

    if cm_de is not None:
        pitch_ctrl = ET.SubElement(pitch_axis, "function", name="aero/coefficient/CMde")
        ET.SubElement(pitch_ctrl, "description").text = "Pitch moment due to elevator"
        pitch_ctrl_prod = ET.SubElement(pitch_ctrl, "product")
        ET.SubElement(pitch_ctrl_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(pitch_ctrl_prod, "property").text = "metrics/Sw-sqft"
        ET.SubElement(pitch_ctrl_prod, "property").text = "metrics/cbarw-ft"
        pitch_ctrl_table = ET.SubElement(pitch_ctrl_prod, "table")
        ET.SubElement(pitch_ctrl_table, "independentVar").text = "fcs/elevator-pos-rad"
        pitch_ctrl_data = ET.SubElement(pitch_ctrl_table, "tableData")
        max_deflection_rad = np.radians(config.max_elevator_deflection_deg)
        pitch_ctrl_data.text = _format_table_data(
            np.array([-max_deflection_rad, 0.0, max_deflection_rad]),
            np.array([
                cm_de * -max_deflection_rad,
                0.0,
                cm_de * max_deflection_rad,
            ]),
            6,
        )

    pitch_q = ET.SubElement(pitch_axis, "function", name="aero/coefficient/Cmq")
    ET.SubElement(pitch_q, "description").text = "Pitch moment due to pitch rate"
    pitch_q_prod = ET.SubElement(pitch_q, "product")
    ET.SubElement(pitch_q_prod, "property").text = "aero/qbar-psf"
    ET.SubElement(pitch_q_prod, "property").text = "metrics/Sw-sqft"
    ET.SubElement(pitch_q_prod, "property").text = "metrics/cbarw-ft"
    ET.SubElement(pitch_q_prod, "property").text = "velocities/q-aero-rad_sec"
    ET.SubElement(pitch_q_prod, "property").text = "aero/ci2vel"
    pitch_q_table = ET.SubElement(pitch_q_prod, "table")
    ET.SubElement(pitch_q_table, "independentVar").text = "aero/alpha-rad"
    pitch_q_data = ET.SubElement(pitch_q_table, "tableData")
    pitch_q_data.text = _format_table_data(alpha_rad, cm_q, 6)

    has_lateral = lateral is not None
    if has_lateral:
        beta_deg, cy_beta, cl_beta, cn_beta, cl_da, cn_dr, cy_dr = lateral
        beta_rad = np.radians(beta_deg)

        side_axis = ET.SubElement(aerodynamics, "axis", name="SIDE")
        side_fn = ET.SubElement(side_axis, "function", name="aero/coefficient/CYbeta")
        ET.SubElement(side_fn, "description").text = "Side force due to beta"
        side_prod = ET.SubElement(side_fn, "product")
        ET.SubElement(side_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(side_prod, "property").text = "metrics/Sw-sqft"
        side_table = ET.SubElement(side_prod, "table")
        ET.SubElement(side_table, "independentVar").text = "aero/beta-rad"
        side_table_data = ET.SubElement(side_table, "tableData")
        side_table_data.text = _format_table_data(beta_rad, cy_beta, 6)

        if cy_dr is not None:
            side_ctrl = ET.SubElement(side_axis, "function", name="aero/coefficient/CYdr")
            ET.SubElement(side_ctrl, "description").text = "Side force due to rudder"
            side_ctrl_prod = ET.SubElement(side_ctrl, "product")
            ET.SubElement(side_ctrl_prod, "property").text = "aero/qbar-psf"
            ET.SubElement(side_ctrl_prod, "property").text = "metrics/Sw-sqft"
            side_ctrl_table = ET.SubElement(side_ctrl_prod, "table")
            ET.SubElement(side_ctrl_table, "independentVar").text = "fcs/rudder-pos-rad"
            side_ctrl_data = ET.SubElement(side_ctrl_table, "tableData")
            max_deflection_rad = np.radians(config.max_rudder_deflection_deg)
            side_ctrl_data.text = _format_table_data(
                np.array([-max_deflection_rad, 0.0, max_deflection_rad]),
                np.array([
                    cy_dr * -max_deflection_rad,
                    0.0,
                    cy_dr * max_deflection_rad,
                ]),
                6,
            )

        roll_axis = ET.SubElement(aerodynamics, "axis", name="ROLL")
        roll_fn = ET.SubElement(roll_axis, "function", name="aero/coefficient/Clbeta")
        ET.SubElement(roll_fn, "description").text = "Roll moment due to beta"
        roll_prod = ET.SubElement(roll_fn, "product")
        ET.SubElement(roll_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(roll_prod, "property").text = "metrics/Sw-sqft"
        ET.SubElement(roll_prod, "property").text = "metrics/bw-ft"
        roll_table = ET.SubElement(roll_prod, "table")
        ET.SubElement(roll_table, "independentVar").text = "aero/beta-rad"
        roll_table_data = ET.SubElement(roll_table, "tableData")
        roll_table_data.text = _format_table_data(beta_rad, cl_beta, 6)

        roll_p = ET.SubElement(roll_axis, "function", name="aero/coefficient/Clp")
        ET.SubElement(roll_p, "description").text = "Roll moment due to roll rate"
        roll_p_prod = ET.SubElement(roll_p, "product")
        ET.SubElement(roll_p_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(roll_p_prod, "property").text = "metrics/Sw-sqft"
        ET.SubElement(roll_p_prod, "property").text = "metrics/bw-ft"
        ET.SubElement(roll_p_prod, "property").text = "velocities/p-aero-rad_sec"
        ET.SubElement(roll_p_prod, "property").text = "aero/bi2vel"
        roll_p_table = ET.SubElement(roll_p_prod, "table")
        ET.SubElement(roll_p_table, "independentVar").text = "aero/alpha-rad"
        roll_p_data = ET.SubElement(roll_p_table, "tableData")
        roll_p_data.text = _format_table_data(alpha_rad, cl_p, 6)

        if cl_da is not None:
            roll_ctrl = ET.SubElement(roll_axis, "function", name="aero/coefficient/Clda")
            ET.SubElement(roll_ctrl, "description").text = "Roll moment due to aileron"
            roll_ctrl_prod = ET.SubElement(roll_ctrl, "product")
            ET.SubElement(roll_ctrl_prod, "property").text = "aero/qbar-psf"
            ET.SubElement(roll_ctrl_prod, "property").text = "metrics/Sw-sqft"
            ET.SubElement(roll_ctrl_prod, "property").text = "metrics/bw-ft"
            roll_ctrl_table = ET.SubElement(roll_ctrl_prod, "table")
            ET.SubElement(roll_ctrl_table, "independentVar").text = "fcs/aileron-pos-rad"
            roll_ctrl_data = ET.SubElement(roll_ctrl_table, "tableData")
            max_deflection_rad = np.radians(config.max_aileron_deflection_deg)
            roll_ctrl_data.text = _format_table_data(
                np.array([-max_deflection_rad, 0.0, max_deflection_rad]),
                np.array([
                    cl_da * -max_deflection_rad,
                    0.0,
                    cl_da * max_deflection_rad,
                ]),
                6,
            )

        yaw_axis = ET.SubElement(aerodynamics, "axis", name="YAW")
        yaw_fn = ET.SubElement(yaw_axis, "function", name="aero/coefficient/Cnbeta")
        ET.SubElement(yaw_fn, "description").text = "Yaw moment due to beta"
        yaw_prod = ET.SubElement(yaw_fn, "product")
        ET.SubElement(yaw_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(yaw_prod, "property").text = "metrics/Sw-sqft"
        ET.SubElement(yaw_prod, "property").text = "metrics/bw-ft"
        yaw_table = ET.SubElement(yaw_prod, "table")
        ET.SubElement(yaw_table, "independentVar").text = "aero/beta-rad"
        yaw_table_data = ET.SubElement(yaw_table, "tableData")
        yaw_table_data.text = _format_table_data(beta_rad, cn_beta, 6)

        yaw_r = ET.SubElement(yaw_axis, "function", name="aero/coefficient/Cnr")
        ET.SubElement(yaw_r, "description").text = "Yaw moment due to yaw rate"
        yaw_r_prod = ET.SubElement(yaw_r, "product")
        ET.SubElement(yaw_r_prod, "property").text = "aero/qbar-psf"
        ET.SubElement(yaw_r_prod, "property").text = "metrics/Sw-sqft"
        ET.SubElement(yaw_r_prod, "property").text = "metrics/bw-ft"
        ET.SubElement(yaw_r_prod, "property").text = "velocities/r-aero-rad_sec"
        ET.SubElement(yaw_r_prod, "property").text = "aero/bi2vel"
        yaw_r_table = ET.SubElement(yaw_r_prod, "table")
        ET.SubElement(yaw_r_table, "independentVar").text = "aero/alpha-rad"
        yaw_r_data = ET.SubElement(yaw_r_table, "tableData")
        yaw_r_data.text = _format_table_data(alpha_rad, cn_r, 6)

        if cn_dr is not None:
            yaw_ctrl = ET.SubElement(yaw_axis, "function", name="aero/coefficient/Cndr")
            ET.SubElement(yaw_ctrl, "description").text = "Yaw moment due to rudder"
            yaw_ctrl_prod = ET.SubElement(yaw_ctrl, "product")
            ET.SubElement(yaw_ctrl_prod, "property").text = "aero/qbar-psf"
            ET.SubElement(yaw_ctrl_prod, "property").text = "metrics/Sw-sqft"
            ET.SubElement(yaw_ctrl_prod, "property").text = "metrics/bw-ft"
            yaw_ctrl_table = ET.SubElement(yaw_ctrl_prod, "table")
            ET.SubElement(yaw_ctrl_table, "independentVar").text = "fcs/rudder-pos-rad"
            yaw_ctrl_data = ET.SubElement(yaw_ctrl_table, "tableData")
            max_deflection_rad = np.radians(config.max_rudder_deflection_deg)
            yaw_ctrl_data.text = _format_table_data(
                np.array([-max_deflection_rad, 0.0, max_deflection_rad]),
                np.array([
                    cn_dr * -max_deflection_rad,
                    0.0,
                    cn_dr * max_deflection_rad,
                ]),
                6,
            )
    elif config.include_zero_axes:
        for axis_name in ("SIDE", "ROLL", "YAW"):
            axis = ET.SubElement(aerodynamics, "axis", name=axis_name)
            zero_fn = ET.SubElement(axis, "function", name=f"aero/coefficient/{axis_name.lower()}0")
            ET.SubElement(zero_fn, "value").text = "0.0"

    return root


def export_jsbsim_project(
    project: Project,
    output_dir: str,
    config: Optional[JSBSimExportConfig] = None,
) -> JSBSimExportResult:
    if config is None:
        config = JSBSimExportConfig()

    model_name = config.model_name or project.wing.name or "FlyingWingProject"
    model_name = _sanitize_name(model_name)

    if config.propulsion is not None:
        if not config.propulsion.engine_file:
            config.propulsion.engine_file = f"{model_name}_engine"
        if not config.propulsion.propeller_file:
            config.propulsion.propeller_file = f"{model_name}_prop"

    os.makedirs(output_dir, exist_ok=True)
    xml_root = generate_jsbsim_xml(project, config)
    xml_str = ET.tostring(xml_root, encoding="unicode")

    import xml.dom.minidom

    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    xml_path = os.path.join(output_dir, f"{model_name}.xml")

    with open(xml_path, "w", encoding="utf-8") as handle:
        handle.write(pretty_xml)

    result = JSBSimExportResult(aircraft_path=os.path.abspath(xml_path))

    if config.propulsion is not None:
        prop = config.propulsion
        engine_file: str = prop.engine_file or f"{model_name}_engine"
        prop_file: str = prop.propeller_file or f"{model_name}_prop"
        prop.engine_file = engine_file
        prop.propeller_file = prop_file
        engine_path = os.path.join(output_dir, f"{engine_file}.xml")
        prop_path = os.path.join(output_dir, f"{prop_file}.xml")
        engine_dir = os.path.dirname(engine_path)
        prop_dir = os.path.dirname(prop_path)
        if engine_dir:
            os.makedirs(engine_dir, exist_ok=True)
        if prop_dir:
            os.makedirs(prop_dir, exist_ok=True)

        engine_root = ET.Element("electric_engine", name=engine_file)
        ET.SubElement(engine_root, "power", unit="WATTS").text = _format_float(
            float(prop.engine_power_w), 2
        )
        ET.SubElement(engine_root, "min_throttle").text = "0.0"
        ET.SubElement(engine_root, "max_throttle").text = "1.0"

        engine_dom = xml.dom.minidom.parseString(ET.tostring(engine_root, encoding="unicode"))
        with open(engine_path, "w", encoding="utf-8") as handle:
            handle.write(engine_dom.toprettyxml(indent="  "))
        engine_path_no_ext = os.path.join(output_dir, engine_file)
        with open(engine_path_no_ext, "w", encoding="utf-8") as handle:
            handle.write(engine_dom.toprettyxml(indent="  "))

        table = _normalized_propeller_table(prop.propeller_table or _default_propeller_table())
        prop_root = ET.Element("propeller", version="1.01", name=prop_file)
        # Compute propeller rotational inertia - critical for realistic RPM dynamics
        prop_ixx = _estimate_propeller_ixx(prop.propeller_diameter_in, prop.propeller_blades)
        ET.SubElement(prop_root, "ixx", unit="SLUG*FT2").text = _format_float(prop_ixx, 6)
        ET.SubElement(prop_root, "diameter", unit="IN").text = _format_float(
            float(prop.propeller_diameter_in), 6
        )
        ET.SubElement(prop_root, "numblades").text = str(int(prop.propeller_blades))
        ET.SubElement(prop_root, "gearratio").text = _format_float(
            float(prop.propeller_gearratio), 6
        )
        ET.SubElement(prop_root, "cp_factor").text = "1.00"
        engine_count = max(1, int(prop.engine_count))
        ct_factor = prop.max_thrust_scale
        if ct_factor is None:
            ct_factor = _estimate_ct_factor(
                table=table,
                prop_diam_in=prop.propeller_diameter_in,
                engine_power_w=prop.engine_power_w,
                engine_count=engine_count,
                mass_kg=project.wing.twist_trim.gross_takeoff_weight_kg,
                target_thrust_ratio=prop.target_thrust_ratio,
                max_thrust_N=prop.max_thrust_N,
            )
        ET.SubElement(prop_root, "ct_factor").text = _format_float(float(ct_factor), 3)

        ct_table = ET.SubElement(prop_root, "table", name="C_THRUST", type="internal")
        ct_data = ET.SubElement(ct_table, "tableData")
        ct_data.text = _format_table_data(table.advance_ratio, table.ct, 6)

        cp_table = ET.SubElement(prop_root, "table", name="C_POWER", type="internal")
        cp_data = ET.SubElement(cp_table, "tableData")
        cp_data.text = _format_table_data(table.advance_ratio, table.cp, 6)

        prop_dom = xml.dom.minidom.parseString(ET.tostring(prop_root, encoding="unicode"))
        with open(prop_path, "w", encoding="utf-8") as handle:
            handle.write(prop_dom.toprettyxml(indent="  "))
        prop_path_no_ext = os.path.join(output_dir, prop_file)
        with open(prop_path_no_ext, "w", encoding="utf-8") as handle:
            handle.write(prop_dom.toprettyxml(indent="  "))

        result.engine_path = os.path.abspath(engine_path)
        result.propeller_path = os.path.abspath(prop_path)

    return result
