from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_drag_models_vs_cfd import (
    ConditionSpec,
    as_float,
    build_alpha_grid,
    build_conditions_from_manifest,
    build_single_condition,
    compute_pressure_proxy,
    import_model_stack,
    load_cfd_result,
    run_aerobuildup_reference,
    run_low_order_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current low-order methods against a prototype hybrid BWB body-pressure "
            "surrogate that augments the existing lifting-line + strip model with geometry-derived "
            "attached-body lift and pressure-drag terms."
        )
    )
    parser.add_argument("--project-json", required=True, help="Path to the project JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and CSV output.")
    parser.add_argument(
        "--grid-study-manifest",
        help="Optional BARAM grid-study manifest. If supplied, all listed conditions are compared.",
    )
    parser.add_argument("--conditions", nargs="*", help="Optional condition-name filter for the manifest.")
    parser.add_argument("--condition-name", help="Single explicit condition name.")
    parser.add_argument("--alpha-deg", type=float, help="Single explicit condition alpha in degrees.")
    parser.add_argument("--velocity-m-s", type=float, help="Single explicit condition velocity in m/s.")
    parser.add_argument("--altitude-m", type=float, help="Single explicit condition altitude in meters.")
    parser.add_argument("--cfd-case-root", help="Optional single explicit CFD case root.")
    parser.add_argument("--cfd-summary-json", help="Optional single explicit CFD summary JSON.")
    parser.add_argument(
        "--reference-density",
        type=float,
        default=1.225,
        help="Reference density for CFD normalization in kg/m^3.",
    )
    parser.add_argument(
        "--average-count",
        type=int,
        default=20,
        help="Number of trailing force-monitor rows to average for CFD.",
    )
    parser.add_argument("--n-strips", type=int, default=81, help="Spanwise strip count for the low-order model.")
    parser.add_argument("--spanwise-resolution", type=int, help="Optional explicit lifting-line spanwise resolution.")
    parser.add_argument("--polar-alpha-min", type=float, default=-8.0, help="Minimum alpha for local section polars.")
    parser.add_argument("--polar-alpha-max", type=float, default=20.0, help="Maximum alpha for local section polars.")
    parser.add_argument(
        "--polar-alpha-steps",
        type=int,
        default=113,
        help="Number of alpha samples for each local section polar.",
    )
    parser.add_argument("--sep-k", type=float, default=0.002, help="Quadratic local separation-drag rise coefficient.")
    parser.add_argument(
        "--sep-slope-fraction",
        type=float,
        default=0.7,
        help="Fraction of linear dCL/dalpha used to infer local separation onset.",
    )
    parser.add_argument(
        "--sep-thickness-start",
        type=float,
        default=0.08,
        help="t/c below which no extra separation-drag rise is applied.",
    )
    parser.add_argument(
        "--sep-thickness-full",
        type=float,
        default=0.14,
        help="t/c at which the separation-drag weighting saturates to 1.",
    )
    parser.add_argument(
        "--drag-rise-alpha-min",
        type=float,
        default=-2.0,
        help="Minimum alpha included in the low-alpha section drag baseline fit.",
    )
    parser.add_argument(
        "--drag-rise-alpha-max",
        type=float,
        default=4.0,
        help="Maximum alpha included in the low-alpha section drag baseline fit.",
    )
    parser.add_argument(
        "--drag-rise-alpha-window",
        type=float,
        default=4.0,
        help="Alpha-margin window over which pressure-drag rise ramps in before local onset.",
    )
    parser.add_argument(
        "--mode",
        choices=("blind", "calibrated"),
        default="blind",
        help="Use a blind physics-derived body correction or a CFD-informed calibrated body correction.",
    )
    parser.add_argument(
        "--calibration-condition",
        default="cruise",
        help="Condition name used to fit the hybrid body-lift and body-pressure gains.",
    )
    parser.add_argument(
        "--body-alpha-break-sweep-factor",
        type=float,
        default=0.096,
        help="Attached-body lift sign-change alpha, expressed as factor * leading-edge sweep in degrees.",
    )
    parser.add_argument(
        "--body-drag-pressure-decay",
        type=float,
        default=0.06,
        help="Pressure-proxy scale controlling how fast attached-body pressure drag decays as separation grows.",
    )
    parser.add_argument(
        "--body-x-samples",
        type=int,
        default=600,
        help="Number of x samples used to build the centerbody cross-section-area proxy.",
    )
    return parser.parse_args()


def naca_thickness_distribution(x_over_c: np.ndarray, thickness_ratio: float) -> np.ndarray:
    x = np.clip(np.asarray(x_over_c, dtype=float), 0.0, 1.0)
    y_t = 5.0 * float(thickness_ratio) * (
        0.2969 * np.sqrt(np.clip(x, 1e-12, 1.0))
        - 0.1260 * x
        - 0.3516 * x * x
        + 0.2843 * x * x * x
        - 0.1015 * x * x * x * x
    )
    return 2.0 * y_t


def build_body_geometry_proxies(service, s_ref_m2: float, x_samples: int) -> dict[str, float]:
    planform = service.wing_project.planform
    sections = sorted(planform.body_sections, key=lambda section: section.y_pos)
    if len(sections) < 2 or s_ref_m2 <= 0.0:
        return {
            "body_area_m2": 0.0,
            "body_area_ratio": 0.0,
            "body_camber_area_ratio": 0.0,
            "body_mean_thickness_ratio": 0.0,
            "body_length_m": 0.0,
            "body_form_drag_proxy": 0.0,
            "body_peak_cross_section_area_m2": 0.0,
        }

    body_area_one_side = 0.0
    body_camber_area_one_side = 0.0
    body_thickness_area_one_side = 0.0
    x_min = min(section.x_offset for section in sections)
    x_max = max(section.x_offset + section.chord for section in sections)
    body_length_m = max(1e-9, x_max - x_min)

    x_grid = np.linspace(x_min, x_max, max(80, int(x_samples)))
    cross_section_area = np.zeros_like(x_grid)

    for inboard, outboard in zip(sections[:-1], sections[1:]):
        dy = float(outboard.y_pos - inboard.y_pos)
        if dy <= 0.0:
            continue

        chord_m = 0.5 * float(inboard.chord + outboard.chord)
        x_le_m = 0.5 * float(inboard.x_offset + outboard.x_offset)
        airfoil_inboard = service._get_airfoil(inboard.airfoil)
        airfoil_outboard = service._get_airfoil(outboard.airfoil)
        max_camber = 0.5 * (
            as_float(airfoil_inboard.max_camber(), default=0.0) + as_float(airfoil_outboard.max_camber(), default=0.0)
        )
        thickness_ratio = 0.5 * (
            as_float(airfoil_inboard.max_thickness(), default=0.0)
            + as_float(airfoil_outboard.max_thickness(), default=0.0)
        )

        body_area_one_side += chord_m * dy
        body_camber_area_one_side += chord_m * max_camber * dy
        body_thickness_area_one_side += chord_m * thickness_ratio * dy

        local_mask = (x_grid >= x_le_m) & (x_grid <= x_le_m + chord_m)
        if not np.any(local_mask):
            continue
        x_local = (x_grid[local_mask] - x_le_m) / max(chord_m, 1e-9)
        local_thickness = naca_thickness_distribution(x_local, thickness_ratio) * chord_m
        cross_section_area[local_mask] += 2.0 * dy * local_thickness

    body_area_m2 = 2.0 * body_area_one_side
    body_camber_area_ratio = 2.0 * body_camber_area_one_side / s_ref_m2
    body_mean_thickness_ratio = (2.0 * body_thickness_area_one_side / body_area_m2) if body_area_m2 > 0.0 else 0.0
    dA_dx = np.gradient(cross_section_area, x_grid)
    body_form_drag_proxy = float(np.trapezoid(dA_dx * dA_dx, x_grid)) / max(s_ref_m2 * body_length_m, 1e-9)

    return {
        "body_area_m2": body_area_m2,
        "body_area_ratio": body_area_m2 / s_ref_m2,
        "body_camber_area_ratio": body_camber_area_ratio,
        "body_mean_thickness_ratio": body_mean_thickness_ratio,
        "body_length_m": body_length_m,
        "body_form_drag_proxy": body_form_drag_proxy,
        "body_peak_cross_section_area_m2": float(np.max(cross_section_area)) if len(cross_section_area) else 0.0,
    }


def calibrate_hybrid_body_gains(
    reference_name: str,
    per_condition: list[dict[str, Any]],
    body_proxies: dict[str, float],
) -> dict[str, Any]:
    reference = next((item for item in per_condition if item["name"] == reference_name), None)
    if reference is None:
        raise KeyError(f"Calibration condition '{reference_name}' was not found.")
    low_order = reference.get("lifting_line_plus_sep")
    cfd = reference.get("cfd")
    if low_order is None or cfd is None:
        raise ValueError(f"Calibration condition '{reference_name}' must have both low-order and CFD results.")

    camber_proxy = max(as_float(body_proxies.get("body_camber_area_ratio")), 1e-9)
    drag_proxy = max(as_float(body_proxies.get("body_form_drag_proxy")), 1e-9)
    lift_gain = (as_float(cfd.get("CL")) - as_float(low_order.get("CL"))) / camber_proxy
    drag_gain = max(0.0, (as_float(cfd.get("CD")) - as_float(low_order.get("CD"))) / drag_proxy)

    return {
        "reference_condition": reference_name,
        "lift_gain_per_camber_area_ratio": float(lift_gain),
        "drag_gain_per_form_proxy": float(drag_gain),
    }


def compute_body_cl0_proxy(
    service,
    condition: ConditionSpec,
    s_ref_m2: float,
) -> float:
    if s_ref_m2 <= 0.0:
        return 0.0

    asb, _, _ = import_model_stack()
    atmosphere = asb.Atmosphere(altitude=condition.altitude_m)
    rho = as_float(atmosphere.density())
    mu = as_float(atmosphere.dynamic_viscosity())
    speed_of_sound = as_float(atmosphere.speed_of_sound())
    mach = float(condition.velocity_m_s) / max(speed_of_sound, 1e-9)

    planform = service.wing_project.planform
    sections = sorted(planform.body_sections, key=lambda section: section.y_pos)
    if len(sections) < 2:
        return 0.0

    body_cl0_area = 0.0
    for inboard, outboard in zip(sections[:-1], sections[1:]):
        dy = float(outboard.y_pos - inboard.y_pos)
        if dy <= 0.0:
            continue
        chord_m = 0.5 * float(inboard.chord + outboard.chord)
        reynolds = rho * float(condition.velocity_m_s) * chord_m / max(mu, 1e-12)
        airfoil_inboard = service._get_airfoil(inboard.airfoil)
        airfoil_outboard = service._get_airfoil(outboard.airfoil)
        aero_inboard = airfoil_inboard.get_aero_from_neuralfoil(alpha=0.0, Re=reynolds, mach=mach)
        aero_outboard = airfoil_outboard.get_aero_from_neuralfoil(alpha=0.0, Re=reynolds, mach=mach)
        cl0 = 0.5 * (
            as_float(aero_inboard.get("CL"), default=0.0) + as_float(aero_outboard.get("CL"), default=0.0)
        )
        body_cl0_area += 2.0 * cl0 * chord_m * dy

    return float(body_cl0_area) / s_ref_m2


def compute_body_symmetry_factor(body_cl0_proxy: float, body_proxies: dict[str, float]) -> float:
    cl0_scale = 0.04
    camber_area_scale = 0.005
    camber_area_ratio = abs(as_float(body_proxies.get("body_camber_area_ratio"), default=0.0))
    symmetry_factor = math.exp(-abs(float(body_cl0_proxy)) / cl0_scale) * math.exp(-camber_area_ratio / camber_area_scale)
    return float(np.clip(symmetry_factor, 0.0, 1.0))


def build_blind_body_model_parameters(service, body_proxies: dict[str, float]) -> dict[str, float]:
    sweep_deg = as_float(service.wing_project.planform.sweep_le_deg)
    sweep_cos = max(0.05, math.cos(math.radians(sweep_deg)))
    mean_thickness = max(0.0, as_float(body_proxies.get("body_mean_thickness_ratio")))
    body_length = max(1e-9, as_float(body_proxies.get("body_length_m")))
    peak_cross_section_area = max(0.0, as_float(body_proxies.get("body_peak_cross_section_area_m2")))
    bluffness_ratio = math.sqrt(peak_cross_section_area) / body_length
    form_factor_extra = max(0.0, 2.0 * mean_thickness + 60.0 * mean_thickness**4)
    pressure_decay = max(1e-6, 0.75 * mean_thickness)
    return {
        "sweep_cosine": sweep_cos,
        "drag_form_factor_extra": form_factor_extra,
        "pressure_decay_scale": pressure_decay,
        "body_bluffness_ratio": bluffness_ratio,
        "cambered_lift_scale": 0.96,
        "alpha_break_sweep_factor": 0.096,
        "alpha_relief_gain": 4.1,
        "symmetric_drag_floor": 0.205,
        "symmetric_form_drag_gain": 2.2,
        "cambered_drag_alpha_gain": 1.8,
        "relief_region_mode": "chord_0.55",
        "relief_onset_floor": 0.15,
        "relief_onset_power": 2.0,
        "relief_root_power": 0.5,
        "relief_cap_fraction": 0.95,
        "relief_target_scale": 1.0,
    }


def apply_hybrid_bwb_body_method(
    service,
    condition: ConditionSpec,
    low_order: dict[str, Any],
    body_proxies: dict[str, float],
    gains: dict[str, Any],
    s_ref_m2: float,
    body_alpha_break_sweep_factor: float,
    body_drag_pressure_decay: float,
) -> dict[str, Any]:
    low_cl = as_float(low_order.get("CL"))
    low_cd = as_float(low_order.get("CD"))
    low_cm = as_float(low_order.get("CM"))
    q = as_float(low_order.get("dynamic_pressure_Pa"))
    pressure_proxy = compute_pressure_proxy(low_order.get("strip_rows", []), s_ref_m2=s_ref_m2)

    alpha_break_factor = float(body_alpha_break_sweep_factor)
    alpha_break_deg = max(0.5, alpha_break_factor * as_float(service.wing_project.planform.sweep_le_deg))
    lift_retention = 1.0 - (float(condition.alpha_deg) / alpha_break_deg) ** 2
    lift_retention = float(np.clip(lift_retention, -0.75, 1.25))
    drag_retention = math.exp(-pressure_proxy / max(body_drag_pressure_decay, 1e-9))

    delta_cl_body = (
        as_float(gains.get("lift_gain_per_camber_area_ratio"))
        * as_float(body_proxies.get("body_camber_area_ratio"))
        * lift_retention
    )
    delta_cd_body = max(
        0.0,
        as_float(gains.get("drag_gain_per_form_proxy"))
        * as_float(body_proxies.get("body_form_drag_proxy"))
        * drag_retention,
    )

    total_cl = low_cl + delta_cl_body
    total_cd = low_cd + delta_cd_body
    total_cm = low_cm * (total_cl / low_cl) if abs(low_cl) > 1e-9 else low_cm
    lift_n = total_cl * q * s_ref_m2
    drag_n = total_cd * q * s_ref_m2

    return {
        "CL": total_cl,
        "CD": total_cd,
        "CM": total_cm,
        "L_over_D": total_cl / max(total_cd, 1e-9),
        "lift_N": lift_n,
        "drag_N": drag_n,
        "dynamic_pressure_Pa": q,
        "reference_area_m2": s_ref_m2,
        "geometry_reference": low_order.get("geometry_reference"),
        "body_pressure_proxy": pressure_proxy,
        "body_alpha_break_deg": alpha_break_deg,
        "body_lift_retention": lift_retention,
        "body_drag_retention": drag_retention,
        "body_delta_CL": delta_cl_body,
        "body_delta_CD": delta_cd_body,
        "base_low_order_CL": low_cl,
        "base_low_order_CD": low_cd,
    }


def apply_blind_hybrid_bwb_body_method(
    service,
    condition: ConditionSpec,
    low_order: dict[str, Any],
    body_proxies: dict[str, float],
    blind_parameters: dict[str, float],
    s_ref_m2: float,
    body_alpha_break_sweep_factor: float,
    reference_attached_cd: float,
) -> dict[str, Any]:
    low_cl = as_float(low_order.get("CL"))
    low_cd = as_float(low_order.get("CD"))
    low_cm = as_float(low_order.get("CM"))
    q = as_float(low_order.get("dynamic_pressure_Pa"))
    strip_rows = low_order.get("strip_rows", [])
    pressure_proxy = compute_pressure_proxy(strip_rows, s_ref_m2=s_ref_m2)
    body_cl0_proxy = compute_body_cl0_proxy(service=service, condition=condition, s_ref_m2=s_ref_m2)
    symmetry_factor = compute_body_symmetry_factor(body_cl0_proxy=body_cl0_proxy, body_proxies=body_proxies)

    alpha_break_factor = as_float(blind_parameters.get("alpha_break_sweep_factor"), default=body_alpha_break_sweep_factor)
    alpha_break_deg = max(0.5, alpha_break_factor * as_float(service.wing_project.planform.sweep_le_deg))
    lift_retention = 1.0 - (float(condition.alpha_deg) / alpha_break_deg) ** 2
    lift_retention = float(np.clip(lift_retention, -0.75, 1.25))

    distributed_delta = {
        "delta_cl_total": 0.0,
        "cambered_delta_cl": 0.0,
        "relief_delta_cl": 0.0,
        "pressure_proxy": pressure_proxy,
        "symmetry_factor": symmetry_factor,
        "body_cl0_proxy": body_cl0_proxy,
        "lift_retention": lift_retention,
        "body_strip_rows": [],
    }
    if strip_rows:
        y_half = np.asarray([as_float(row.get("y_m")) for row in strip_rows], dtype=float)
        lift_per_span_half = np.asarray([as_float(row.get("lift_per_span_N_m")) for row in strip_rows], dtype=float)
        distributed_delta = service._build_distributed_blind_body_lift_delta(
            y_half=y_half,
            lift_per_span_half=lift_per_span_half,
            velocity=condition.velocity_m_s,
            alpha=condition.alpha_deg,
            load_factor=1.0,
            altitude_m=condition.altitude_m,
            q=q,
            s_ref_m2=s_ref_m2,
            blind_parameters=blind_parameters,
            strip_rows=strip_rows,
        )
        pressure_proxy = as_float(distributed_delta.get("pressure_proxy"), default=pressure_proxy)
        symmetry_factor = as_float(distributed_delta.get("symmetry_factor"), default=symmetry_factor)
        body_cl0_proxy = as_float(distributed_delta.get("body_cl0_proxy"), default=body_cl0_proxy)
        lift_retention = as_float(distributed_delta.get("lift_retention"), default=lift_retention)

    sweep_cos = as_float(blind_parameters.get("sweep_cosine"), default=1.0)
    drag_form_factor_extra = as_float(blind_parameters.get("drag_form_factor_extra"), default=0.0)
    pressure_decay_scale = as_float(blind_parameters.get("pressure_decay_scale"), default=1e-6)
    body_bluffness_ratio = as_float(blind_parameters.get("body_bluffness_ratio"), default=0.0)
    alpha_relief_gain = as_float(blind_parameters.get("alpha_relief_gain"), default=0.0)
    symmetric_drag_floor = as_float(blind_parameters.get("symmetric_drag_floor"), default=0.0)
    symmetric_form_drag_gain = as_float(blind_parameters.get("symmetric_form_drag_gain"), default=0.0)
    cambered_drag_alpha_gain = as_float(blind_parameters.get("cambered_drag_alpha_gain"), default=0.0)
    body_area_ratio = max(0.0, as_float(body_proxies.get("body_area_ratio")))
    drag_retention = math.exp(-pressure_proxy / max(pressure_decay_scale, 1e-9))
    drag_retention_floor = symmetry_factor * symmetric_drag_floor * min(1.0, pressure_proxy / max(pressure_decay_scale, 1e-9))
    effective_drag_retention = max(drag_retention, drag_retention_floor)
    alpha_ratio = abs(float(condition.alpha_deg)) / max(alpha_break_deg, 1e-9)
    symmetric_drag_multiplier = 1.0 + symmetric_form_drag_gain * symmetry_factor * body_bluffness_ratio
    cambered_drag_multiplier = 1.0 + cambered_drag_alpha_gain * (1.0 - symmetry_factor) * body_area_ratio * max(
        0.0, alpha_ratio - 0.15
    )
    drag_shape_multiplier = symmetric_drag_multiplier * cambered_drag_multiplier

    cambered_body_delta_cl = as_float(distributed_delta.get("cambered_delta_cl"))
    symmetric_alpha_relief_cl = as_float(distributed_delta.get("relief_delta_cl"))
    delta_cl_body = as_float(distributed_delta.get("delta_cl_total"))
    delta_cd_body = (
        as_float(body_proxies.get("body_form_drag_proxy"))
        * drag_form_factor_extra
        * math.sqrt(sweep_cos)
        * effective_drag_retention
        * drag_shape_multiplier
    )

    total_cl = low_cl + delta_cl_body
    cl_scale = total_cl / low_cl if abs(low_cl) > 1e-9 else 1.0
    cl_scale = float(np.clip(cl_scale, 0.0, 2.0))
    attached_cd = max(0.0, float(reference_attached_cd))
    excess_cd = max(0.0, low_cd - attached_cd)
    relieved_low_cd = attached_cd + excess_cd * (1.0 - symmetry_factor + symmetry_factor * cl_scale * cl_scale)
    total_cd = relieved_low_cd + delta_cd_body
    total_cm = low_cm * (total_cl / low_cl) if abs(low_cl) > 1e-9 else low_cm
    lift_n = total_cl * q * s_ref_m2
    drag_n = total_cd * q * s_ref_m2

    return {
        "CL": total_cl,
        "CD": total_cd,
        "CM": total_cm,
        "L_over_D": total_cl / max(total_cd, 1e-9),
        "lift_N": lift_n,
        "drag_N": drag_n,
        "dynamic_pressure_Pa": q,
        "reference_area_m2": s_ref_m2,
        "geometry_reference": low_order.get("geometry_reference"),
        "body_pressure_proxy": pressure_proxy,
        "body_section_cl0_proxy": body_cl0_proxy,
        "body_symmetry_factor": symmetry_factor,
        "body_alpha_break_deg": alpha_break_deg,
        "body_lift_retention": lift_retention,
        "body_drag_retention": effective_drag_retention,
        "body_drag_retention_floor": drag_retention_floor,
        "body_alpha_ratio": alpha_ratio,
        "body_delta_CL": delta_cl_body,
        "body_delta_CD": delta_cd_body,
        "body_camber_delta_CL": cambered_body_delta_cl,
        "body_symmetric_relief_CL": symmetric_alpha_relief_cl,
        "body_drag_form_factor_extra": drag_form_factor_extra,
        "body_bluffness_ratio": body_bluffness_ratio,
        "body_symmetric_drag_multiplier": symmetric_drag_multiplier,
        "body_cambered_drag_multiplier": cambered_drag_multiplier,
        "body_drag_shape_multiplier": drag_shape_multiplier,
        "body_distributed_strip_count": len(distributed_delta.get("body_strip_rows", [])),
        "base_low_order_relieved_CD": relieved_low_cd,
        "reference_attached_CD": attached_cd,
        "base_low_order_CL": low_cl,
        "base_low_order_CD": low_cd,
    }


def delta_vs_cfd(model: dict[str, Any], cfd: dict[str, Any] | None) -> dict[str, float]:
    if cfd is None:
        return {}
    return {
        "delta_CL": as_float(model.get("CL")) - as_float(cfd.get("CL")),
        "delta_CD": as_float(model.get("CD")) - as_float(cfd.get("CD")),
        "delta_L_over_D": as_float(model.get("L_over_D")) - as_float(cfd.get("L_over_D")),
    }


def build_csv_rows(
    condition_name: str,
    methods: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, values in methods.items():
        if values is None:
            continue
        row = {
            "condition": condition_name,
            "method": method_name,
            "CL": as_float(values.get("CL")),
            "CD": as_float(values.get("CD")),
            "L_over_D": as_float(values.get("L_over_D")),
            "lift_N": as_float(values.get("lift_N")),
            "drag_N": as_float(values.get("drag_N")),
        }
        if method_name == "hybrid_bwb_body":
            row["body_delta_CL"] = as_float(values.get("body_delta_CL"))
            row["body_delta_CD"] = as_float(values.get("body_delta_CD"))
            row["body_pressure_proxy"] = as_float(values.get("body_pressure_proxy"))
            row["body_lift_retention"] = as_float(values.get("body_lift_retention"))
            row["body_drag_retention"] = as_float(values.get("body_drag_retention"))
            row["body_symmetry_factor"] = as_float(values.get("body_symmetry_factor"))
            row["body_symmetric_relief_CL"] = as_float(values.get("body_symmetric_relief_CL"))
            row["base_low_order_relieved_CD"] = as_float(values.get("base_low_order_relieved_CD"))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, Project, AeroSandboxService = import_model_stack()
    project_path = Path(args.project_json).resolve()
    project = Project.load(str(project_path))
    service = AeroSandboxService(project.wing)
    s_ref_m2 = float(project.wing.planform.actual_area())
    cruise_altitude_m = float(project.wing.twist_trim.cruise_altitude_m)

    if args.grid_study_manifest:
        condition_filter = set(args.conditions) if args.conditions else None
        conditions = build_conditions_from_manifest(
            manifest_path=Path(args.grid_study_manifest).resolve(),
            cruise_altitude_m=cruise_altitude_m,
            only_conditions=condition_filter,
        )
    else:
        conditions = build_single_condition(args, cruise_altitude_m=cruise_altitude_m)

    if not conditions:
        raise ValueError("No conditions were selected.")

    alpha_grid = build_alpha_grid(
        alpha_min=float(args.polar_alpha_min),
        alpha_max=float(args.polar_alpha_max),
        alpha_steps=int(args.polar_alpha_steps),
    )

    body_proxies = build_body_geometry_proxies(
        service=service,
        s_ref_m2=s_ref_m2,
        x_samples=int(args.body_x_samples),
    )

    per_condition: list[dict[str, Any]] = []
    for condition in conditions:
        established = run_aerobuildup_reference(
            service=service,
            condition=condition,
            s_ref_m2=s_ref_m2,
        )
        low_order = run_low_order_model(
            service=service,
            condition=condition,
            s_ref_m2=s_ref_m2,
            n_strips=int(args.n_strips),
            spanwise_resolution=args.spanwise_resolution,
            alpha_grid=alpha_grid,
            sep_k=float(args.sep_k),
            sep_slope_fraction=float(args.sep_slope_fraction),
            sep_thickness_start=float(args.sep_thickness_start),
            sep_thickness_full=float(args.sep_thickness_full),
            drag_rise_alpha_min=float(args.drag_rise_alpha_min),
            drag_rise_alpha_max=float(args.drag_rise_alpha_max),
            drag_rise_alpha_window=float(args.drag_rise_alpha_window),
        )
        cfd = load_cfd_result(
            condition=condition,
            s_ref_m2=s_ref_m2,
            density_kg_m3=float(args.reference_density),
            average_count=int(args.average_count),
        )
        per_condition.append(
            {
                "name": condition.name,
                "flight_condition": {
                    "alpha_deg": condition.alpha_deg,
                    "velocity_m_s": condition.velocity_m_s,
                    "altitude_m": condition.altitude_m,
                },
                "aerobuildup": established,
                "lifting_line_plus_sep": low_order,
                "cfd": cfd,
            }
        )

    gains = None
    if str(args.mode) == "calibrated":
        gains = calibrate_hybrid_body_gains(
            reference_name=str(args.calibration_condition),
            per_condition=per_condition,
            body_proxies=body_proxies,
        )

    blind_parameters = None
    if str(args.mode) == "blind":
        blind_parameters = build_blind_body_model_parameters(service=service, body_proxies=body_proxies)
        attached_reference = min(per_condition, key=lambda item: abs(as_float(item["flight_condition"]["alpha_deg"])))
        blind_parameters["reference_condition"] = str(attached_reference["name"])
        blind_parameters["reference_low_order_cd"] = as_float(attached_reference["lifting_line_plus_sep"].get("CD"))

    summary: dict[str, Any] = {
        "project_json": str(project_path),
        "reference_area_m2": s_ref_m2,
        "body_proxies": body_proxies,
        "hybrid_settings": {
            "mode": str(args.mode),
            "calibration_condition": str(args.calibration_condition),
            "body_alpha_break_sweep_factor": float(args.body_alpha_break_sweep_factor),
            "body_drag_pressure_decay": float(args.body_drag_pressure_decay),
            "body_x_samples": int(args.body_x_samples),
        },
        "conditions": {},
    }
    if str(args.mode) == "calibrated":
        summary["hybrid_calibration"] = gains
    else:
        summary["blind_body_parameters"] = blind_parameters
    csv_rows: list[dict[str, Any]] = []

    for item in per_condition:
        condition_name = item["name"]
        low_order = item["lifting_line_plus_sep"]
        cfd = item["cfd"]
        condition_spec = ConditionSpec(
            name=condition_name,
            alpha_deg=float(item["flight_condition"]["alpha_deg"]),
            velocity_m_s=float(item["flight_condition"]["velocity_m_s"]),
            altitude_m=float(item["flight_condition"]["altitude_m"]),
        )
        if str(args.mode) == "calibrated":
            hybrid = apply_hybrid_bwb_body_method(
                service=service,
                condition=condition_spec,
                low_order=low_order,
                body_proxies=body_proxies,
                gains=gains,
                s_ref_m2=s_ref_m2,
                body_alpha_break_sweep_factor=float(args.body_alpha_break_sweep_factor),
                body_drag_pressure_decay=float(args.body_drag_pressure_decay),
            )
        else:
            hybrid = apply_blind_hybrid_bwb_body_method(
                service=service,
                condition=condition_spec,
                low_order=low_order,
                body_proxies=body_proxies,
                blind_parameters=blind_parameters,
                s_ref_m2=s_ref_m2,
                body_alpha_break_sweep_factor=float(args.body_alpha_break_sweep_factor),
                reference_attached_cd=as_float(blind_parameters.get("reference_low_order_cd")),
            )

        summary["conditions"][condition_name] = {
            "flight_condition": item["flight_condition"],
            "aerobuildup": item["aerobuildup"],
            "lifting_line_plus_sep": {key: value for key, value in low_order.items() if key != "strip_rows"},
            "hybrid_bwb_body": hybrid,
            "cfd": cfd,
            "delta_vs_cfd": {
                "aerobuildup_minus_cfd": delta_vs_cfd(item["aerobuildup"], cfd),
                "low_order_minus_cfd": delta_vs_cfd(low_order, cfd),
                "hybrid_bwb_body_minus_cfd": delta_vs_cfd(hybrid, cfd),
            },
        }

        csv_rows.extend(
            build_csv_rows(
                condition_name=condition_name,
                methods={
                    "aerobuildup": item["aerobuildup"],
                    "lifting_line_plus_sep": low_order,
                    "hybrid_bwb_body": hybrid,
                    "cfd": cfd,
                },
            )
        )

    json_path = output_dir / f"hybrid_bwb_body_{args.mode}_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path = output_dir / f"hybrid_bwb_body_{args.mode}_comparison_table.csv"
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote hybrid comparison summary: {json_path}")
    print(f"Wrote hybrid comparison table: {csv_path}")


if __name__ == "__main__":
    main()
