from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baram_case_tools import (
    average_last,
    compute_dynamic_pressure,
    find_monitor_file,
    infer_geometry_length_scale_to_m,
    normalize_case_dir,
    parse_coefficient_dat,
    parse_force_dat,
    project_force_components,
    read_force_axis_dirs,
)


PRESET_RANK = {
    "extra_fine": 0,
    "fine": 1,
    "medium": 2,
    "coarse": 3,
}


@dataclass
class ConditionSpec:
    name: str
    alpha_deg: float
    velocity_m_s: float
    altitude_m: float
    cfd_case_root: Path | None = None
    cfd_summary_json: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current AeroBuildup-based drag estimate, a low-order "
            "lifting-line + local separation-rise model, and CFD."
        )
    )
    parser.add_argument(
        "--project-json",
        required=True,
        help="Path to the project JSON used by the program.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for comparison JSON, CSV, and plots.",
    )
    parser.add_argument(
        "--grid-study-manifest",
        help=(
            "Optional BARAM grid-study manifest. If supplied, the script compares "
            "all listed conditions and auto-picks the best solved CFD variant."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        help="Optional condition-name filter when using --grid-study-manifest.",
    )
    parser.add_argument("--condition-name", help="Single explicit condition name.")
    parser.add_argument("--alpha-deg", type=float, help="Single explicit condition alpha in degrees.")
    parser.add_argument("--velocity-m-s", type=float, help="Single explicit condition velocity in m/s.")
    parser.add_argument(
        "--altitude-m",
        type=float,
        help="Single explicit condition altitude in meters. Defaults to cruise altitude unless the name implies takeoff.",
    )
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
    parser.add_argument(
        "--n-strips",
        type=int,
        default=81,
        help="Spanwise strip count for the low-order model.",
    )
    parser.add_argument(
        "--spanwise-resolution",
        type=int,
        help="Optional explicit lifting-line spanwise resolution.",
    )
    parser.add_argument(
        "--polar-alpha-min",
        type=float,
        default=-8.0,
        help="Minimum alpha for local NeuralFoil section polars.",
    )
    parser.add_argument(
        "--polar-alpha-max",
        type=float,
        default=20.0,
        help="Maximum alpha for local NeuralFoil section polars.",
    )
    parser.add_argument(
        "--polar-alpha-steps",
        type=int,
        default=113,
        help="Number of alpha samples for each local NeuralFoil section polar.",
    )
    parser.add_argument(
        "--sep-k",
        type=float,
        default=0.002,
        help="Quadratic local separation-drag rise coefficient in cd/deg^2.",
    )
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
        help="t/c at which the local separation-drag weighting saturates to 1.",
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
        help="Maximum alpha included in the low-alpha section drag baseline fit before drag-rise extraction.",
    )
    parser.add_argument(
        "--drag-rise-alpha-window",
        type=float,
        default=4.0,
        help="Alpha-margin window over which pressure-drag rise ramps in before local onset.",
    )
    parser.add_argument(
        "--calibration-json",
        help=(
            "Optional JSON file containing a saved calibration block for the "
            "calibrated_bwb_surrogate. Accepts either a raw calibration dict or "
            "a full comparison summary with a top-level 'calibration' block."
        ),
    )
    return parser.parse_args()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            return float(default)
        return as_float(value[0], default=default)
    try:
        return float(value)
    except Exception:
        return float(default)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_saved_calibration(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    calibration = payload.get("calibration", payload) if isinstance(payload, dict) else payload
    if not isinstance(calibration, dict):
        raise ValueError(f"Calibration payload must be a JSON object: {path}")

    required_keys = {
        "lift_scale_intercept",
        "lift_scale_slope",
        "drag_scale_exponent",
        "pressure_base_CD",
    }
    missing = sorted(key for key in required_keys if key not in calibration)
    if missing:
        raise KeyError(f"Calibration JSON is missing required keys {missing}: {path}")
    return calibration


def import_model_stack():
    import aerosandbox as asb

    from core.state import Project
    from services.geometry import AeroSandboxService

    return asb, Project, AeroSandboxService


def guess_altitude_from_name(name: str, cruise_altitude_m: float) -> float:
    lowered = name.lower()
    if "takeoff" in lowered or "landing" in lowered or "ground" in lowered:
        return 0.0
    return float(cruise_altitude_m)


def case_has_force_monitor(case_root: Path) -> bool:
    case_dir = normalize_case_dir(case_root)
    monitor_dir = case_dir / "postProcessing" / "force-mon-1_forces"
    return monitor_dir.is_dir()


def select_best_variant(manifest: dict[str, Any], condition_name: str) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for variant in manifest.get("variants", []):
        if not isinstance(variant, dict):
            continue
        if variant.get("condition") != condition_name:
            continue
        bundle_root = variant.get("bundle_root")
        case_dir = variant.get("case_dir")
        candidate_root = Path(bundle_root) if bundle_root else Path(case_dir) if case_dir else None
        has_case = candidate_root is not None and candidate_root.exists() and case_has_force_monitor(candidate_root)
        solve_ok = 0 if variant.get("solve_status") == "solve_ok" else 1
        preset_rank = PRESET_RANK.get(str(variant.get("mesh_preset")), 99)
        if has_case:
            candidates.append((solve_ok, preset_rank, variant))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("name"))))
    return candidates[0][2]


def build_conditions_from_manifest(
    manifest_path: Path,
    cruise_altitude_m: float,
    only_conditions: set[str] | None,
) -> list[ConditionSpec]:
    manifest = load_json(manifest_path)
    conditions: list[ConditionSpec] = []
    for condition in manifest.get("conditions", []):
        if not isinstance(condition, dict):
            continue
        name = str(condition["name"])
        if only_conditions and name not in only_conditions:
            continue
        variant = select_best_variant(manifest, name)
        cfd_case_root = None
        cfd_summary_json = None
        if variant is not None:
            if variant.get("bundle_root"):
                cfd_case_root = Path(str(variant["bundle_root"])).resolve()
            elif variant.get("case_dir"):
                cfd_case_root = Path(str(variant["case_dir"])).resolve()
            report_output_dir = manifest.get("report_output_dir")
            if report_output_dir:
                candidate_summary = Path(str(report_output_dir)).resolve() / f"{variant['name']}_summary.json"
                if candidate_summary.exists():
                    cfd_summary_json = candidate_summary
        conditions.append(
            ConditionSpec(
                name=name,
                alpha_deg=float(condition["alpha_deg"]),
                velocity_m_s=float(condition["velocity_m_s"]),
                altitude_m=guess_altitude_from_name(name, cruise_altitude_m),
                cfd_case_root=cfd_case_root,
                cfd_summary_json=cfd_summary_json,
            )
        )
    return conditions


def build_single_condition(args: argparse.Namespace, cruise_altitude_m: float) -> list[ConditionSpec]:
    if args.condition_name is None or args.alpha_deg is None or args.velocity_m_s is None:
        raise ValueError(
            "Without --grid-study-manifest you must supply --condition-name, --alpha-deg, and --velocity-m-s."
        )
    altitude_m = (
        float(args.altitude_m)
        if args.altitude_m is not None
        else guess_altitude_from_name(str(args.condition_name), cruise_altitude_m)
    )
    return [
        ConditionSpec(
            name=str(args.condition_name),
            alpha_deg=float(args.alpha_deg),
            velocity_m_s=float(args.velocity_m_s),
            altitude_m=altitude_m,
            cfd_case_root=Path(args.cfd_case_root).resolve() if args.cfd_case_root else None,
            cfd_summary_json=Path(args.cfd_summary_json).resolve() if args.cfd_summary_json else None,
        )
    ]


def build_airplane(service, asb):
    wing = service.build_wing()
    x_np = as_float(wing.aerodynamic_center()[0])
    mac = as_float(wing.mean_aerodynamic_chord())
    static_margin = as_float(service.wing_project.twist_trim.static_margin_percent)
    x_cg = x_np - (static_margin / 100.0) * mac
    airplane = asb.Airplane(
        name=service.wing_project.name,
        wings=[wing],
        xyz_ref=[x_cg, 0.0, 0.0],
    )
    return wing, airplane, {"x_np": x_np, "mac": mac, "x_cg": x_cg}


def make_operating_point(asb, condition: ConditionSpec):
    atmosphere = asb.Atmosphere(altitude=condition.altitude_m)
    return asb.OperatingPoint(
        atmosphere=atmosphere,
        velocity=condition.velocity_m_s,
        alpha=condition.alpha_deg,
    )


def result_to_coefficients(result: dict[str, Any], q: float, s_ref_m2: float) -> dict[str, float]:
    lift = as_float(result.get("L"))
    drag = as_float(result.get("D"))
    if not math.isfinite(lift) or abs(lift) < 1e-12:
        lift = as_float(result.get("CL")) * q * s_ref_m2
    if not math.isfinite(drag) or abs(drag) < 1e-12:
        drag = as_float(result.get("CD")) * q * s_ref_m2
    cl = lift / (q * s_ref_m2) if q > 0 and s_ref_m2 > 0 else as_float(result.get("CL"))
    cd = drag / (q * s_ref_m2) if q > 0 and s_ref_m2 > 0 else as_float(result.get("CD"))
    cm = as_float(result.get("CM", result.get("Cm")))
    ld = cl / max(cd, 1e-9)
    return {
        "CL": cl,
        "CD": cd,
        "CM": cm,
        "L_over_D": ld,
        "lift_N": lift,
        "drag_N": drag,
    }


def run_aerobuildup_reference(service, condition: ConditionSpec, s_ref_m2: float) -> dict[str, Any]:
    asb, _, _ = import_model_stack()
    _, airplane, geometry_ref = build_airplane(service, asb)
    op_point = make_operating_point(asb, condition)
    q = as_float(op_point.dynamic_pressure())
    result = asb.AeroBuildup(airplane=airplane, op_point=op_point).run()
    summary = result_to_coefficients(result, q=q, s_ref_m2=s_ref_m2)
    summary["reference_area_m2"] = s_ref_m2
    summary["dynamic_pressure_Pa"] = q
    summary["geometry_reference"] = geometry_ref
    return summary


def build_alpha_grid(alpha_min: float, alpha_max: float, alpha_steps: int) -> np.ndarray:
    if alpha_steps < 3:
        raise ValueError("alpha_steps must be at least 3.")
    return np.linspace(alpha_min, alpha_max, alpha_steps)


def evaluate_local_section_drag(
    airfoil,
    cl_target: float,
    reynolds: float,
    mach: float,
    alpha_grid: np.ndarray,
    sep_k: float,
    sep_slope_fraction: float,
    sep_thickness_start: float,
    sep_thickness_full: float,
    drag_rise_alpha_min: float,
    drag_rise_alpha_max: float,
    drag_rise_alpha_window: float,
    polar_cache: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, float]:
    cache_key = (
        str(getattr(airfoil, "name", "airfoil")),
        int(round(reynolds / 5000.0)),
        int(round(mach * 1000.0)),
    )
    if cache_key not in polar_cache:
        reynolds_array = np.full_like(alpha_grid, fill_value=float(reynolds), dtype=float)
        mach_array = np.full_like(alpha_grid, fill_value=float(mach), dtype=float)
        aero = airfoil.get_aero_from_neuralfoil(alpha=alpha_grid, Re=reynolds_array, mach=mach_array)
        polar_cache[cache_key] = {
            "alpha_deg": alpha_grid.copy(),
            "CL": np.asarray(aero["CL"], dtype=float),
            "CD": np.asarray(aero["CD"], dtype=float),
        }
    polar = polar_cache[cache_key]
    cl_values = polar["CL"]
    cd_values = polar["CD"]
    alpha_values = polar["alpha_deg"]

    nearest_idx = int(np.argmin(np.abs(cl_values - cl_target)))
    alpha_eff = float(alpha_values[nearest_idx])
    cl_attached = float(cl_values[nearest_idx])
    cd_attached = max(float(cd_values[nearest_idx]), 0.0)

    dcl_dalpha = np.gradient(cl_values, alpha_values)
    linear_mask = (alpha_values >= -1.0) & (alpha_values <= 4.0)
    if np.any(linear_mask):
        reference_slope = float(np.median(dcl_dalpha[linear_mask]))
    else:
        reference_slope = float(np.max(dcl_dalpha))
    if not math.isfinite(reference_slope) or abs(reference_slope) < 1e-9:
        reference_slope = float(np.max(dcl_dalpha))
    slope_limit = sep_slope_fraction * reference_slope
    sep_candidates = np.where((alpha_values > 0.0) & (dcl_dalpha < slope_limit))[0]
    if len(sep_candidates) > 0:
        alpha_sep = float(alpha_values[int(sep_candidates[0])])
    else:
        alpha_sep = float(alpha_values[-1])
    alpha_margin = alpha_sep - alpha_eff

    thickness_ratio = as_float(airfoil.max_thickness(), default=0.0)
    if sep_thickness_full <= sep_thickness_start:
        thickness_factor = 1.0 if thickness_ratio > sep_thickness_start else 0.0
    else:
        thickness_factor = np.clip(
            (thickness_ratio - sep_thickness_start) / (sep_thickness_full - sep_thickness_start),
            0.0,
            1.0,
        )
    baseline_alpha_max = min(float(drag_rise_alpha_max), alpha_sep)
    baseline_mask = (alpha_values >= float(drag_rise_alpha_min)) & (alpha_values <= baseline_alpha_max)
    if np.count_nonzero(baseline_mask) < 5:
        baseline_mask = alpha_values <= alpha_sep
    fit_cl = cl_values[baseline_mask]
    fit_cd = cd_values[baseline_mask]
    if len(fit_cl) >= 3:
        baseline_coeffs = np.polyfit(fit_cl, fit_cd, deg=2)
        cd_baseline = float(np.polyval(baseline_coeffs, cl_target))
    else:
        cd_baseline = float(np.min(cd_values))
    cd_baseline = max(cd_baseline, 0.0)
    cd_drag_rise_2d = max(0.0, cd_attached - cd_baseline)
    if drag_rise_alpha_window <= 1e-9:
        onset_proximity = 1.0 if alpha_margin <= 0.0 else 0.0
    else:
        onset_proximity = float(np.clip(1.0 - alpha_margin / drag_rise_alpha_window, 0.0, 1.0))
    cd_pressure_rise = float(thickness_factor) * onset_proximity * cd_drag_rise_2d
    alpha_excess = max(0.0, alpha_eff - alpha_sep)
    cd_separation = float(sep_k) * float(thickness_factor) * alpha_excess * alpha_excess

    return {
        "alpha_eff_deg": alpha_eff,
        "alpha_sep_deg": alpha_sep,
        "alpha_margin_deg": alpha_margin,
        "alpha_excess_deg": alpha_excess,
        "cl_attached": cl_attached,
        "cd_baseline": cd_baseline,
        "cd_drag_rise_2d": cd_drag_rise_2d,
        "onset_proximity": onset_proximity,
        "cd_pressure_rise": cd_pressure_rise,
        "cd_attached": cd_attached,
        "cd_separation": cd_separation,
        "thickness_ratio": thickness_ratio,
        "thickness_factor": float(thickness_factor),
        "reference_slope_per_deg": float(reference_slope),
        "slope_limit_per_deg": float(slope_limit),
    }


def summarize_strip_rows(strip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not strip_rows:
        return {
            "active_sep_strip_count": 0,
            "active_pressure_rise_strip_count": 0,
            "min_alpha_margin_deg": None,
            "max_alpha_excess_deg": 0.0,
            "max_cd_pressure_rise": 0.0,
            "max_cd_separation": 0.0,
        }

    margins = [row["alpha_sep_deg"] - row["alpha_eff_deg"] for row in strip_rows]
    alpha_excess = [row["alpha_excess_deg"] for row in strip_rows]
    cd_pressure = [row["cd_pressure_rise"] for row in strip_rows]
    cd_sep = [row["cd_separation"] for row in strip_rows]
    thickness = [row["thickness_ratio"] for row in strip_rows]
    cl_local = [row["cl_local"] for row in strip_rows]

    min_margin_idx = int(np.argmin(margins))
    max_excess_idx = int(np.argmax(alpha_excess))
    max_pressure_idx = int(np.argmax(cd_pressure))
    max_sep_idx = int(np.argmax(cd_sep))
    max_thickness_idx = int(np.argmax(thickness))
    max_cl_idx = int(np.argmax(cl_local))

    return {
        "active_sep_strip_count": int(sum(1 for value in alpha_excess if value > 0.0)),
        "active_pressure_rise_strip_count": int(sum(1 for value in cd_pressure if value > 0.0)),
        "min_alpha_margin_deg": float(margins[min_margin_idx]),
        "min_alpha_margin_y_m": float(strip_rows[min_margin_idx]["y_m"]),
        "max_alpha_excess_deg": float(alpha_excess[max_excess_idx]),
        "max_alpha_excess_y_m": float(strip_rows[max_excess_idx]["y_m"]),
        "max_cd_pressure_rise": float(cd_pressure[max_pressure_idx]),
        "max_cd_pressure_rise_y_m": float(strip_rows[max_pressure_idx]["y_m"]),
        "max_cd_separation": float(cd_sep[max_sep_idx]),
        "max_cd_separation_y_m": float(strip_rows[max_sep_idx]["y_m"]),
        "max_thickness_ratio": float(thickness[max_thickness_idx]),
        "max_thickness_y_m": float(strip_rows[max_thickness_idx]["y_m"]),
        "max_cl_local": float(cl_local[max_cl_idx]),
        "max_cl_local_y_m": float(strip_rows[max_cl_idx]["y_m"]),
    }


def compute_pressure_proxy(strip_rows: list[dict[str, Any]], s_ref_m2: float) -> float:
    if len(strip_rows) < 2 or s_ref_m2 <= 0.0:
        return 0.0
    y = np.asarray([as_float(row.get("y_m")) for row in strip_rows], dtype=float)
    chord = np.asarray([as_float(row.get("chord_m")) for row in strip_rows], dtype=float)
    proximity = np.asarray(
        [
            as_float(row.get("onset_proximity")) * as_float(row.get("thickness_factor"), default=1.0)
            for row in strip_rows
        ],
        dtype=float,
    )
    return 2.0 * float(np.trapezoid(chord * proximity, y)) / max(s_ref_m2, 1e-9)


def fit_drag_scaling_exponent(
    rows: list[dict[str, float]],
    lift_scale_intercept: float,
    lift_scale_slope: float,
) -> tuple[float, float]:
    def objective(exponent: float) -> tuple[float, float]:
        residual_base = []
        for row in rows:
            cl_scale = max(0.05, lift_scale_intercept + lift_scale_slope * row["pressure_proxy"])
            residual_base.append(row["cfd_cd"] - row["reference_cd"] * (cl_scale**exponent))
        pressure_base_cd = float(np.mean(residual_base))
        mse = float(
            np.mean(
                [
                    (row["cfd_cd"] - (row["reference_cd"] * (max(0.05, lift_scale_intercept + lift_scale_slope * row["pressure_proxy"]) ** exponent) + pressure_base_cd))
                    ** 2
                    for row in rows
                ]
            )
        )
        return mse, pressure_base_cd

    lower = -1.0
    upper = 6.0
    best_exponent = 2.0
    best_pressure_base = 0.0
    for _ in range(6):
        grid = np.linspace(lower, upper, 1201)
        values = [objective(float(exponent)) for exponent in grid]
        best_index = int(np.argmin([item[0] for item in values]))
        best_exponent = float(grid[best_index])
        best_pressure_base = float(values[best_index][1])
        if best_index == 0:
            lower, upper = grid[0], grid[1]
        elif best_index == len(grid) - 1:
            lower, upper = grid[-2], grid[-1]
        else:
            lower, upper = grid[best_index - 1], grid[best_index + 1]
    return best_exponent, best_pressure_base


def calibrate_bwb_surrogate(
    per_condition: list[dict[str, Any]],
    s_ref_m2: float,
) -> dict[str, Any] | None:
    calibration_rows = []
    for item in per_condition:
        low_order = item.get("lifting_line_plus_sep")
        cfd = item.get("cfd")
        if low_order is None or cfd is None:
            continue
        calibration_rows.append(
            {
                "condition": item["name"],
                "low_cl": as_float(low_order.get("CL")),
                "cfd_cl": as_float(cfd.get("CL")),
                "reference_cd": as_float(low_order.get("base_lifting_line_CD", low_order.get("CD"))),
                "cfd_cd": as_float(cfd.get("CD")),
                "pressure_proxy": compute_pressure_proxy(low_order.get("strip_rows", []), s_ref_m2=s_ref_m2),
            }
        )
    if len(calibration_rows) < 2:
        return None

    a_matrix = np.asarray(
        [[row["low_cl"], row["low_cl"] * row["pressure_proxy"]] for row in calibration_rows],
        dtype=float,
    )
    b_vector = np.asarray([row["cfd_cl"] for row in calibration_rows], dtype=float)
    lift_params, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    lift_scale_intercept = float(lift_params[0])
    lift_scale_slope = float(lift_params[1])

    drag_scale_exponent, pressure_base_cd = fit_drag_scaling_exponent(
        calibration_rows,
        lift_scale_intercept=lift_scale_intercept,
        lift_scale_slope=lift_scale_slope,
    )

    fit_rows = []
    cl_errors = []
    cd_errors = []
    for row in calibration_rows:
        cl_scale = max(0.05, lift_scale_intercept + lift_scale_slope * row["pressure_proxy"])
        cl_pred = row["low_cl"] * cl_scale
        cd_pred = row["reference_cd"] * (cl_scale**drag_scale_exponent) + pressure_base_cd
        cl_error = cl_pred - row["cfd_cl"]
        cd_error = cd_pred - row["cfd_cd"]
        cl_errors.append(cl_error)
        cd_errors.append(cd_error)
        fit_rows.append(
            {
                "condition": row["condition"],
                "pressure_proxy": row["pressure_proxy"],
                "cl_scale": cl_scale,
                "cl_pred": cl_pred,
                "cl_target": row["cfd_cl"],
                "cd_pred": cd_pred,
                "cd_target": row["cfd_cd"],
                "delta_cl": cl_error,
                "delta_cd": cd_error,
            }
        )

    return {
        "method": "identified_bwb_pressure_base",
        "conditions_used": [row["condition"] for row in calibration_rows],
        "lift_scale_intercept": lift_scale_intercept,
        "lift_scale_slope": lift_scale_slope,
        "drag_scale_exponent": drag_scale_exponent,
        "pressure_base_CD": pressure_base_cd,
        "fit_rows": fit_rows,
        "rmse_CL": float(np.sqrt(np.mean(np.square(cl_errors)))),
        "rmse_CD": float(np.sqrt(np.mean(np.square(cd_errors)))),
    }


def apply_bwb_surrogate(
    low_order: dict[str, Any],
    calibration: dict[str, Any],
    s_ref_m2: float,
) -> dict[str, Any]:
    pressure_proxy = compute_pressure_proxy(low_order.get("strip_rows", []), s_ref_m2=s_ref_m2)
    cl_scale = max(
        0.05,
        as_float(calibration.get("lift_scale_intercept"), default=1.0)
        + as_float(calibration.get("lift_scale_slope")) * pressure_proxy,
    )
    reference_cd = as_float(low_order.get("base_lifting_line_CD", low_order.get("CD")))
    drag_scale_exponent = as_float(calibration.get("drag_scale_exponent"), default=2.0)
    cd_nonpressure = reference_cd * (cl_scale**drag_scale_exponent)
    pressure_base_cd = as_float(calibration.get("pressure_base_CD"))
    total_cd = cd_nonpressure + pressure_base_cd + as_float(low_order.get("separation_CD"))

    q = as_float(low_order.get("dynamic_pressure_Pa"))
    cl = as_float(low_order.get("CL")) * cl_scale
    cm = as_float(low_order.get("CM")) * cl_scale
    lift_n = cl * q * s_ref_m2
    drag_n = total_cd * q * s_ref_m2
    return {
        "CL": cl,
        "CD": total_cd,
        "CM": cm,
        "L_over_D": cl / max(total_cd, 1e-9),
        "lift_N": lift_n,
        "drag_N": drag_n,
        "pressure_proxy": pressure_proxy,
        "cl_scale": cl_scale,
        "reference_CD": reference_cd,
        "scaled_reference_CD": cd_nonpressure,
        "pressure_base_CD": pressure_base_cd,
        "separation_CD": as_float(low_order.get("separation_CD")),
        "dynamic_pressure_Pa": q,
        "reference_area_m2": s_ref_m2,
        "geometry_reference": low_order.get("geometry_reference"),
    }


def run_low_order_model(
    service,
    condition: ConditionSpec,
    s_ref_m2: float,
    n_strips: int,
    spanwise_resolution: int | None,
    alpha_grid: np.ndarray,
    sep_k: float,
    sep_slope_fraction: float,
    sep_thickness_start: float,
    sep_thickness_full: float,
    drag_rise_alpha_min: float,
    drag_rise_alpha_max: float,
    drag_rise_alpha_window: float,
) -> dict[str, Any]:
    asb, _, _ = import_model_stack()
    _, airplane, geometry_ref = build_airplane(service, asb)
    op_point = make_operating_point(asb, condition)
    q = as_float(op_point.dynamic_pressure())
    rho = as_float(op_point.atmosphere.density())
    mu = as_float(op_point.atmosphere.dynamic_viscosity())
    a = as_float(op_point.atmosphere.speed_of_sound())
    mach = condition.velocity_m_s / max(a, 1e-9)

    section_count = len(service.spanwise_sections())
    ll_resolution = spanwise_resolution or max(10, section_count)
    ll_result = asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=ll_resolution,
    ).run()
    ll_summary = result_to_coefficients(ll_result, q=q, s_ref_m2=s_ref_m2)

    y_half, lift_per_span_half = service.get_spanwise_lift_distribution(
        velocity=condition.velocity_m_s,
        alpha=condition.alpha_deg,
        n_spanwise_points=n_strips,
        use_vlm=True,
        altitude_m=condition.altitude_m,
    )
    y_half = np.asarray(y_half, dtype=float)
    lift_per_span_half = np.asarray(lift_per_span_half, dtype=float)

    sections = service.spanwise_sections()
    sec_y = np.asarray([abs(section.y_m) for section in sections], dtype=float)
    sec_chord = np.asarray([section.chord_m for section in sections], dtype=float)

    polar_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    profile_drag_per_span_half = np.zeros_like(y_half)
    pressure_rise_drag_per_span_half = np.zeros_like(y_half)
    separation_drag_per_span_half = np.zeros_like(y_half)
    strip_rows: list[dict[str, Any]] = []

    for idx, y_value in enumerate(y_half):
        chord_m = float(np.interp(y_value, sec_y, sec_chord))
        chord_m = max(chord_m, 1e-6)
        nearest_section_index = int(np.argmin(np.abs(sec_y - y_value)))
        airfoil = sections[nearest_section_index].airfoil
        cl_local = float(lift_per_span_half[idx] / max(q * chord_m, 1e-9))
        reynolds = float(rho * condition.velocity_m_s * chord_m / max(mu, 1e-12))

        local = evaluate_local_section_drag(
            airfoil=airfoil,
            cl_target=cl_local,
            reynolds=reynolds,
            mach=mach,
            alpha_grid=alpha_grid,
            sep_k=sep_k,
            sep_slope_fraction=sep_slope_fraction,
            sep_thickness_start=sep_thickness_start,
            sep_thickness_full=sep_thickness_full,
            drag_rise_alpha_min=drag_rise_alpha_min,
            drag_rise_alpha_max=drag_rise_alpha_max,
            drag_rise_alpha_window=drag_rise_alpha_window,
            polar_cache=polar_cache,
        )
        profile_drag_per_span_half[idx] = q * chord_m * local["cd_attached"]
        pressure_rise_drag_per_span_half[idx] = q * chord_m * local["cd_pressure_rise"]
        separation_drag_per_span_half[idx] = q * chord_m * local["cd_separation"]
        strip_rows.append(
            {
                "y_m": float(y_value),
                "airfoil_name": str(getattr(airfoil, "name", "airfoil")),
                "section_index": float(nearest_section_index),
                "chord_m": chord_m,
                "cl_local": cl_local,
                "reynolds": reynolds,
                "alpha_eff_deg": local["alpha_eff_deg"],
                "alpha_sep_deg": local["alpha_sep_deg"],
                "alpha_margin_deg": local["alpha_margin_deg"],
                "alpha_excess_deg": local["alpha_excess_deg"],
                "thickness_ratio": local["thickness_ratio"],
                "thickness_factor": local["thickness_factor"],
                "cl_attached": local["cl_attached"],
                "cd_baseline": local["cd_baseline"],
                "cd_drag_rise_2d": local["cd_drag_rise_2d"],
                "onset_proximity": local["onset_proximity"],
                "cd_pressure_rise": local["cd_pressure_rise"],
                "cd_attached": local["cd_attached"],
                "cd_separation": local["cd_separation"],
                "reference_slope_per_deg": local["reference_slope_per_deg"],
                "slope_limit_per_deg": local["slope_limit_per_deg"],
                "lift_per_span_N_m": float(lift_per_span_half[idx]),
            }
        )

    profile_drag_N = 2.0 * float(np.trapezoid(profile_drag_per_span_half, y_half))
    pressure_rise_drag_N = 2.0 * float(np.trapezoid(pressure_rise_drag_per_span_half, y_half))
    separation_drag_N = 2.0 * float(np.trapezoid(separation_drag_per_span_half, y_half))
    profile_cd = profile_drag_N / max(q * s_ref_m2, 1e-9)
    pressure_rise_cd = pressure_rise_drag_N / max(q * s_ref_m2, 1e-9)
    separation_cd = separation_drag_N / max(q * s_ref_m2, 1e-9)
    total_cd = ll_summary["CD"] + pressure_rise_cd + separation_cd
    total_drag_N = ll_summary["drag_N"] + pressure_rise_drag_N + separation_drag_N
    strip_diagnostics = summarize_strip_rows(strip_rows)

    return {
        "CL": ll_summary["CL"],
        "CD": total_cd,
        "CM": ll_summary["CM"],
        "L_over_D": ll_summary["CL"] / max(total_cd, 1e-9),
        "lift_N": ll_summary["lift_N"],
        "drag_N": total_drag_N,
        "base_lifting_line_CD": ll_summary["CD"],
        "base_lifting_line_drag_N": ll_summary["drag_N"],
        "strip_profile_CD": profile_cd,
        "strip_profile_drag_N": profile_drag_N,
        "pressure_rise_CD": pressure_rise_cd,
        "pressure_rise_drag_N": pressure_rise_drag_N,
        "separation_CD": separation_cd,
        "separation_drag_N": separation_drag_N,
        "dynamic_pressure_Pa": q,
        "reference_area_m2": s_ref_m2,
        "geometry_reference": geometry_ref,
        "strip_diagnostics": strip_diagnostics,
        "strip_rows": strip_rows,
    }


def parse_cfd_summary_json(summary_path: Path, q: float, s_ref_m2: float) -> dict[str, Any]:
    data = load_json(summary_path)
    if "scaled_last_average" in data and "drag_breakdown_last_average_full_aircraft_N" in data:
        drag_N = as_float(data["scaled_last_average"].get("drag_full_N"))
        lift_N = as_float(data["scaled_last_average"].get("lift_full_N"))
        return {
            "CL": lift_N / max(q * s_ref_m2, 1e-9),
            "CD": drag_N / max(q * s_ref_m2, 1e-9),
            "L_over_D": lift_N / max(drag_N, 1e-9),
            "lift_N": lift_N,
            "drag_N": drag_N,
            "pressure_drag_N": as_float(data["drag_breakdown_last_average_full_aircraft_N"].get("pressure_drag")),
            "viscous_drag_N": as_float(data["drag_breakdown_last_average_full_aircraft_N"].get("viscous_drag")),
            "source": str(summary_path),
            "source_type": "summary_json",
        }
    if "baram_cruise_scaled_last20_average" in data and "drag_breakdown_last20_average_full_aircraft_N" in data:
        drag_N = as_float(data["baram_cruise_scaled_last20_average"].get("drag_full_N"))
        lift_N = as_float(data["baram_cruise_scaled_last20_average"].get("lift_full_N"))
        return {
            "CL": lift_N / max(q * s_ref_m2, 1e-9),
            "CD": drag_N / max(q * s_ref_m2, 1e-9),
            "L_over_D": lift_N / max(drag_N, 1e-9),
            "lift_N": lift_N,
            "drag_N": drag_N,
            "pressure_drag_N": as_float(data["drag_breakdown_last20_average_full_aircraft_N"].get("pressure_drag")),
            "viscous_drag_N": as_float(data["drag_breakdown_last20_average_full_aircraft_N"].get("viscous_drag")),
            "source": str(summary_path),
            "source_type": "summary_json",
        }
    raise ValueError(f"Unsupported CFD summary format: {summary_path}")


def parse_cfd_case(
    case_root: Path,
    s_ref_m2: float,
    velocity_m_s: float,
    density_kg_m3: float,
    average_count: int,
) -> dict[str, Any]:
    case_dir = normalize_case_dir(case_root)
    coeff_path = find_monitor_file(case_dir, "force-mon-1", "coefficient.dat")
    force_path = find_monitor_file(case_dir, "force-mon-1_forces", "force.dat")

    coeff_rows = parse_coefficient_dat(coeff_path)
    force_rows = parse_force_dat(force_path)
    if len(coeff_rows) != len(force_rows):
        raise ValueError(
            f"Monitor length mismatch: {len(coeff_rows)} coefficient rows vs {len(force_rows)} force rows."
        )

    q = compute_dynamic_pressure(density_kg_m3, velocity_m_s)
    geometry_scale = infer_geometry_length_scale_to_m(case_dir)
    force_scale = float(geometry_scale["area_scale_to_m2"])
    axis_dirs = read_force_axis_dirs(case_dir)

    combined = []
    for coeff_row, force_row in zip(coeff_rows, force_rows):
        projected = project_force_components(force_row, axis_dirs["drag_dir"], axis_dirs["lift_dir"])
        drag_full = 2.0 * projected["total_drag_half"] * force_scale
        lift_full = 2.0 * projected["total_lift_half"] * force_scale
        pressure_drag_full = 2.0 * projected["pressure_drag_half"] * force_scale
        viscous_drag_full = 2.0 * projected["viscous_drag_half"] * force_scale
        combined.append(
            {
                "time": coeff_row["time"],
                "Cd_raw": coeff_row["Cd_raw"],
                "Cl_raw": coeff_row["Cl_raw"],
                "drag_full_N": drag_full,
                "lift_full_N": lift_full,
                "pressure_drag_N": pressure_drag_full,
                "viscous_drag_N": viscous_drag_full,
            }
        )

    avg_count = max(1, min(int(average_count), len(combined)))
    drag_N = average_last(combined, "drag_full_N", avg_count)
    lift_N = average_last(combined, "lift_full_N", avg_count)
    pressure_drag_N = average_last(combined, "pressure_drag_N", avg_count)
    viscous_drag_N = average_last(combined, "viscous_drag_N", avg_count)

    return {
        "CL": lift_N / max(q * s_ref_m2, 1e-9),
        "CD": drag_N / max(q * s_ref_m2, 1e-9),
        "L_over_D": lift_N / max(drag_N, 1e-9),
        "lift_N": lift_N,
        "drag_N": drag_N,
        "pressure_drag_N": pressure_drag_N,
        "viscous_drag_N": viscous_drag_N,
        "raw_last": combined[-1],
        "average_count": avg_count,
        "geometry_scale": geometry_scale,
        "drag_dir": axis_dirs["drag_dir"],
        "lift_dir": axis_dirs["lift_dir"],
        "source": str(case_root),
        "source_type": "case_root",
    }


def load_cfd_result(
    condition: ConditionSpec,
    s_ref_m2: float,
    density_kg_m3: float,
    average_count: int,
) -> dict[str, Any] | None:
    q = compute_dynamic_pressure(density_kg_m3, condition.velocity_m_s)
    if condition.cfd_summary_json and condition.cfd_summary_json.exists():
        return parse_cfd_summary_json(condition.cfd_summary_json, q=q, s_ref_m2=s_ref_m2)
    if condition.cfd_case_root and condition.cfd_case_root.exists():
        return parse_cfd_case(
            case_root=condition.cfd_case_root,
            s_ref_m2=s_ref_m2,
            velocity_m_s=condition.velocity_m_s,
            density_kg_m3=density_kg_m3,
            average_count=average_count,
        )
    return None


def build_comparison_rows(condition_name: str, method_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method_name, values in method_rows.items():
        row = {
            "condition": condition_name,
            "method": method_name,
            "CL": as_float(values.get("CL")),
            "CD": as_float(values.get("CD")),
            "L_over_D": as_float(values.get("L_over_D")),
            "lift_N": as_float(values.get("lift_N")),
            "drag_N": as_float(values.get("drag_N")),
        }
        if "pressure_drag_N" in values:
            row["pressure_drag_N"] = as_float(values.get("pressure_drag_N"))
            row["viscous_drag_N"] = as_float(values.get("viscous_drag_N"))
        if "base_lifting_line_CD" in values:
            row["base_lifting_line_CD"] = as_float(values.get("base_lifting_line_CD"))
            row["strip_profile_CD"] = as_float(values.get("strip_profile_CD"))
            row["pressure_rise_CD"] = as_float(values.get("pressure_rise_CD"))
            row["separation_CD"] = as_float(values.get("separation_CD"))
        if "pressure_proxy" in values:
            row["pressure_proxy"] = as_float(values.get("pressure_proxy"))
            row["cl_scale"] = as_float(values.get("cl_scale"), default=1.0)
        if "pressure_base_CD" in values:
            row["pressure_base_CD"] = as_float(values.get("pressure_base_CD"))
            row["scaled_reference_CD"] = as_float(values.get("scaled_reference_CD"))
        rows.append(row)
    return rows


def plot_condition(output_dir: Path, condition_name: str, comparison: dict[str, Any]) -> None:
    labels = []
    cd_values = []
    cl_values = []
    ld_values = []
    for label, block in comparison.items():
        if block is None:
            continue
        labels.append(label)
        cd_values.append(as_float(block.get("CD")))
        cl_values.append(as_float(block.get("CL")))
        ld_values.append(as_float(block.get("L_over_D")))

    if labels:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        x = np.arange(len(labels))
        axes[0].bar(x, cl_values, color="#4c78a8")
        axes[0].set_title("CL")
        axes[1].bar(x, cd_values, color="#f58518")
        axes[1].set_title("CD")
        axes[2].bar(x, ld_values, color="#54a24b")
        axes[2].set_title("L/D")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{condition_name}: model comparison")
        fig.tight_layout()
        fig.savefig(output_dir / f"{condition_name}_model_comparison.png", dpi=200)
        plt.close(fig)

    low_order = comparison.get("lifting_line_plus_sep")
    cfd = comparison.get("cfd")
    if low_order is not None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        parts = [
            ("LL base", as_float(low_order.get("base_lifting_line_CD"))),
            ("Pressure rise", as_float(low_order.get("pressure_rise_CD"))),
            ("Separation", as_float(low_order.get("separation_CD"))),
        ]
        if "strip_profile_CD" in low_order:
            parts.append(("Strip profile", as_float(low_order.get("strip_profile_CD"))))
        part_labels = [item[0] for item in parts]
        part_values = [item[1] for item in parts]
        ax.bar(part_labels, part_values, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"][: len(parts)])
        total_cd = as_float(low_order.get("CD"))
        ax.axhline(total_cd, color="black", linestyle="--", linewidth=1.2, label="LL + separation total")
        if cfd is not None:
            ax.axhline(as_float(cfd.get("CD")), color="#f58518", linestyle=":", linewidth=1.5, label="CFD total")
        ax.set_ylabel("CD")
        ax.set_title(f"{condition_name}: low-order drag breakdown")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{condition_name}_low_order_breakdown.png", dpi=200)
        plt.close(fig)


def write_strip_debug_csv(output_dir: Path, condition_name: str, strip_rows: list[dict[str, Any]]) -> None:
    if not strip_rows:
        return
    csv_path = output_dir / f"{condition_name}_strip_debug.csv"
    fieldnames = list(strip_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strip_rows)


def plot_spanwise_debug(output_dir: Path, condition_name: str, strip_rows: list[dict[str, Any]]) -> None:
    if not strip_rows:
        return
    y = [row["y_m"] for row in strip_rows]
    cl_local = [row["cl_local"] for row in strip_rows]
    alpha_margin = [row["alpha_margin_deg"] for row in strip_rows]
    cd_pressure = [row["cd_pressure_rise"] for row in strip_rows]
    cd_sep = [row["cd_separation"] for row in strip_rows]
    thickness = [row["thickness_ratio"] for row in strip_rows]

    fig, axes = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(y, cl_local, color="#4c78a8")
    axes[0].set_ylabel("cl(y)")
    axes[1].plot(y, thickness, color="#72b7b2")
    axes[1].set_ylabel("t/c")
    axes[2].plot(y, alpha_margin, color="#b279a2")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("alpha_sep - alpha_eff [deg]")
    axes[3].plot(y, cd_pressure, color="#f58518")
    axes[3].set_ylabel("cd_pr(y)")
    axes[4].plot(y, cd_sep, color="#e45756")
    axes[4].set_ylabel("cd_sep(y)")
    axes[4].set_xlabel("Half-span y [m]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{condition_name}: spanwise low-order diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / f"{condition_name}_spanwise_debug.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, Project, AeroSandboxService = import_model_stack()
    project_path = Path(args.project_json).resolve()
    project = Project.load(str(project_path))
    service = AeroSandboxService(project.wing)
    reference_area_m2 = float(project.wing.planform.actual_area())
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
        raise ValueError("No conditions were selected. Check the manifest path or --conditions filter.")

    alpha_grid = build_alpha_grid(
        alpha_min=float(args.polar_alpha_min),
        alpha_max=float(args.polar_alpha_max),
        alpha_steps=int(args.polar_alpha_steps),
    )

    summary: dict[str, Any] = {
        "project_json": str(project_path),
        "reference_area_m2": reference_area_m2,
        "conditions": {},
        "settings": {
            "reference_density_kg_m3": float(args.reference_density),
            "average_count": int(args.average_count),
            "n_strips": int(args.n_strips),
            "spanwise_resolution": args.spanwise_resolution,
            "polar_alpha_min": float(args.polar_alpha_min),
            "polar_alpha_max": float(args.polar_alpha_max),
            "polar_alpha_steps": int(args.polar_alpha_steps),
            "sep_k": float(args.sep_k),
            "sep_slope_fraction": float(args.sep_slope_fraction),
            "sep_thickness_start": float(args.sep_thickness_start),
            "sep_thickness_full": float(args.sep_thickness_full),
            "drag_rise_alpha_min": float(args.drag_rise_alpha_min),
            "drag_rise_alpha_max": float(args.drag_rise_alpha_max),
            "drag_rise_alpha_window": float(args.drag_rise_alpha_window),
            "calibration_json": str(Path(args.calibration_json).resolve()) if args.calibration_json else None,
        },
    }
    csv_rows: list[dict[str, Any]] = []
    per_condition_results: list[dict[str, Any]] = []

    for condition in conditions:
        established = run_aerobuildup_reference(
            service=service,
            condition=condition,
            s_ref_m2=reference_area_m2,
        )
        low_order = run_low_order_model(
            service=service,
            condition=condition,
            s_ref_m2=reference_area_m2,
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
            s_ref_m2=reference_area_m2,
            density_kg_m3=float(args.reference_density),
            average_count=int(args.average_count),
        )
        per_condition_results.append(
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

    calibration = calibrate_bwb_surrogate(per_condition_results, s_ref_m2=reference_area_m2)
    if calibration is None and args.calibration_json:
        calibration = load_saved_calibration(Path(args.calibration_json).resolve())
        summary["calibration_source"] = {
            "type": "loaded_json",
            "path": str(Path(args.calibration_json).resolve()),
        }
    elif calibration is not None:
        summary["calibration_source"] = {"type": "derived_from_attached_cfd"}
    if calibration is not None:
        summary["calibration"] = calibration

    for item in per_condition_results:
        condition_name = item["name"]
        established = item["aerobuildup"]
        low_order = item["lifting_line_plus_sep"]
        cfd = item["cfd"]
        calibrated = apply_bwb_surrogate(low_order, calibration, s_ref_m2=reference_area_m2) if calibration else None

        deltas: dict[str, Any] = {}
        if cfd is not None:
            deltas = {
                "aerobuildup_minus_cfd": {
                    "delta_CL": established["CL"] - cfd["CL"],
                    "delta_CD": established["CD"] - cfd["CD"],
                    "delta_L_over_D": established["L_over_D"] - cfd["L_over_D"],
                },
                "low_order_minus_cfd": {
                    "delta_CL": low_order["CL"] - cfd["CL"],
                    "delta_CD": low_order["CD"] - cfd["CD"],
                    "delta_L_over_D": low_order["L_over_D"] - cfd["L_over_D"],
                },
            }
            if calibrated is not None:
                deltas["calibrated_bwb_surrogate_minus_cfd"] = {
                    "delta_CL": calibrated["CL"] - cfd["CL"],
                    "delta_CD": calibrated["CD"] - cfd["CD"],
                    "delta_L_over_D": calibrated["L_over_D"] - cfd["L_over_D"],
                }

        condition_block = {
            "flight_condition": item["flight_condition"],
            "aerobuildup": established,
            "lifting_line_plus_sep": {key: value for key, value in low_order.items() if key != "strip_rows"},
            "calibrated_bwb_surrogate": calibrated,
            "cfd": cfd,
            "delta_vs_cfd": deltas,
        }
        summary["conditions"][condition_name] = condition_block

        method_rows = {
            "aerobuildup": established,
            "lifting_line_plus_sep": low_order,
        }
        if calibrated is not None:
            method_rows["calibrated_bwb_surrogate"] = calibrated
        if cfd is not None:
            method_rows["cfd"] = cfd
        csv_rows.extend(build_comparison_rows(condition_name, method_rows))
        plot_condition(output_dir=output_dir, condition_name=condition_name, comparison=method_rows)
        plot_spanwise_debug(
            output_dir=output_dir,
            condition_name=condition_name,
            strip_rows=low_order["strip_rows"],
        )
        write_strip_debug_csv(
            output_dir=output_dir,
            condition_name=condition_name,
            strip_rows=low_order["strip_rows"],
        )

    json_path = output_dir / "drag_model_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path = output_dir / "drag_model_comparison_table.csv"
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote comparison summary: {json_path}")
    print(f"Wrote comparison table: {csv_path}")


if __name__ == "__main__":
    main()
