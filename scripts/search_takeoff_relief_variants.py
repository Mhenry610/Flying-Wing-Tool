from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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
    compute_pressure_proxy,
    import_model_stack,
    load_cfd_result,
    run_low_order_model,
)
from compare_hybrid_bwb_body_method import (
    apply_blind_hybrid_bwb_body_method,
    build_blind_body_model_parameters,
    build_body_geometry_proxies,
    compute_body_cl0_proxy,
    compute_body_symmetry_factor,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    project_json: Path
    manifest_json: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype and rank takeoff-relief variants for the blind BWB hybrid model "
            "without modifying the live solver path."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "Reports" / "Report" / "drag_model_vs_cfd" / "prototype_takeoff_relief_search"),
        help="Directory for the prototype search summary artifacts.",
    )
    parser.add_argument("--n-strips", type=int, default=81, help="Spanwise strip count for the low-order model.")
    parser.add_argument("--spanwise-resolution", type=int, help="Optional explicit lifting-line spanwise resolution.")
    parser.add_argument("--polar-alpha-min", type=float, default=-8.0)
    parser.add_argument("--polar-alpha-max", type=float, default=20.0)
    parser.add_argument("--polar-alpha-steps", type=int, default=113)
    parser.add_argument("--sep-k", type=float, default=0.002)
    parser.add_argument("--sep-slope-fraction", type=float, default=0.7)
    parser.add_argument("--sep-thickness-start", type=float, default=0.08)
    parser.add_argument("--sep-thickness-full", type=float, default=0.14)
    parser.add_argument("--drag-rise-alpha-min", type=float, default=-2.0)
    parser.add_argument("--drag-rise-alpha-max", type=float, default=4.0)
    parser.add_argument("--drag-rise-alpha-window", type=float, default=4.0)
    parser.add_argument("--reference-density", type=float, default=1.225)
    parser.add_argument("--average-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12, help="Number of top-ranked variants to keep.")
    return parser.parse_args()


def default_datasets() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="x48_10ft",
            project_json=ROOT / "X48_10ft_Approx_FWT.json",
            manifest_json=ROOT / "BARAM" / "studies" / "x48_10ft_fine_validation" / "grid_study_manifest.json",
            output_dir=ROOT / "Reports" / "Report" / "drag_model_vs_cfd" / "x48_10ft",
        ),
        DatasetSpec(
            name="intendedvalidation2",
            project_json=ROOT / "IntendedValidation2.json",
            manifest_json=ROOT / "BARAM" / "studies" / "intendedvalidation2_grid_independence" / "grid_study_manifest.json",
            output_dir=ROOT / "Reports" / "Report" / "drag_model_vs_cfd" / "intendedvalidation2",
        ),
    ]


def compute_region_end(
    region_mode: str,
    y_half: np.ndarray,
    chord_half: np.ndarray,
    body_y_max: float,
) -> float:
    half_span = float(y_half[-1]) if len(y_half) else max(body_y_max, 1e-9)
    if region_mode == "body":
        return max(body_y_max, 1e-9)
    if region_mode.startswith("chord_"):
        chord_fraction = float(region_mode.split("_", 1)[1])
        root_chord = max(float(chord_half[0]), 1e-9)
        mask = chord_half >= chord_fraction * root_chord
        if np.any(mask):
            return max(body_y_max, float(y_half[np.where(mask)[0][-1]]))
        return max(body_y_max, 1e-9)
    if region_mode.startswith("span_"):
        span_fraction = float(region_mode.split("_", 1)[1])
        return max(body_y_max, span_fraction * half_span)
    raise ValueError(f"Unsupported region mode '{region_mode}'.")


def apply_cl_cd_from_delta(
    components: dict[str, Any],
    blind_parameters: dict[str, float],
    body_proxies: dict[str, float],
    delta_cl_body: float,
) -> dict[str, Any]:
    low_cl = float(components["low_cl"])
    low_cd = float(components["low_cd"])
    low_cm = float(components["low_cm"])
    q = float(components["q"])
    s_ref_m2 = float(components["s_ref_m2"])
    symmetry_factor = float(components["symmetry_factor"])
    pressure_proxy = float(components["pressure_proxy"])
    alpha_ratio = float(components["alpha_ratio"])
    alpha_break_deg = float(components["alpha_break_deg"])
    lift_retention = float(components["lift_retention"])
    body_cl0_proxy = float(components["body_cl0_proxy"])
    attached_cd = float(components["reference_attached_cd"])

    sweep_cos = as_float(blind_parameters.get("sweep_cosine"), default=1.0)
    drag_form_factor_extra = as_float(blind_parameters.get("drag_form_factor_extra"), default=0.0)
    pressure_decay_scale = as_float(blind_parameters.get("pressure_decay_scale"), default=1e-6)
    body_bluffness_ratio = as_float(blind_parameters.get("body_bluffness_ratio"), default=0.0)
    symmetric_drag_floor = as_float(blind_parameters.get("symmetric_drag_floor"), default=0.0)
    symmetric_form_drag_gain = as_float(blind_parameters.get("symmetric_form_drag_gain"), default=0.0)
    cambered_drag_alpha_gain = as_float(blind_parameters.get("cambered_drag_alpha_gain"), default=0.0)
    body_area_ratio = max(0.0, as_float(body_proxies.get("body_area_ratio")))

    drag_retention = math.exp(-pressure_proxy / max(pressure_decay_scale, 1e-9))
    drag_retention_floor = symmetry_factor * symmetric_drag_floor * min(1.0, pressure_proxy / max(pressure_decay_scale, 1e-9))
    effective_drag_retention = max(drag_retention, drag_retention_floor)
    symmetric_drag_multiplier = 1.0 + symmetric_form_drag_gain * symmetry_factor * body_bluffness_ratio
    cambered_drag_multiplier = 1.0 + cambered_drag_alpha_gain * (1.0 - symmetry_factor) * body_area_ratio * max(
        0.0, alpha_ratio - 0.15
    )
    drag_shape_multiplier = symmetric_drag_multiplier * cambered_drag_multiplier

    delta_cd_body = (
        as_float(body_proxies.get("body_form_drag_proxy"))
        * drag_form_factor_extra
        * math.sqrt(sweep_cos)
        * effective_drag_retention
        * drag_shape_multiplier
    )

    total_cl = low_cl + float(delta_cl_body)
    cl_scale = total_cl / low_cl if abs(low_cl) > 1e-9 else 1.0
    cl_scale = float(np.clip(cl_scale, 0.0, 2.0))
    excess_cd = max(0.0, low_cd - attached_cd)
    relieved_low_cd = attached_cd + excess_cd * (1.0 - symmetry_factor + symmetry_factor * cl_scale * cl_scale)
    total_cd = relieved_low_cd + delta_cd_body
    total_cm = low_cm * (total_cl / low_cl) if abs(low_cl) > 1e-9 else low_cm

    return {
        "CL": total_cl,
        "CD": total_cd,
        "CM": total_cm,
        "L_over_D": total_cl / max(total_cd, 1e-9),
        "lift_N": total_cl * q * s_ref_m2,
        "drag_N": total_cd * q * s_ref_m2,
        "body_pressure_proxy": pressure_proxy,
        "body_section_cl0_proxy": body_cl0_proxy,
        "body_symmetry_factor": symmetry_factor,
        "body_alpha_break_deg": alpha_break_deg,
        "body_lift_retention": lift_retention,
        "body_drag_retention": effective_drag_retention,
        "body_drag_retention_floor": drag_retention_floor,
        "body_alpha_ratio": alpha_ratio,
        "body_delta_CD": delta_cd_body,
        "base_low_order_relieved_CD": relieved_low_cd,
        "reference_attached_CD": attached_cd,
        "base_low_order_CL": low_cl,
        "base_low_order_CD": low_cd,
        "dynamic_pressure_Pa": q,
        "reference_area_m2": s_ref_m2,
    }


def compute_blind_components(
    service,
    condition: ConditionSpec,
    low_order: dict[str, Any],
    body_proxies: dict[str, float],
    blind_parameters: dict[str, float],
    s_ref_m2: float,
) -> dict[str, Any]:
    low_cl = as_float(low_order.get("CL"))
    low_cd = as_float(low_order.get("CD"))
    low_cm = as_float(low_order.get("CM"))
    q = as_float(low_order.get("dynamic_pressure_Pa"))
    strip_rows = low_order.get("strip_rows", [])
    pressure_proxy = compute_pressure_proxy(strip_rows, s_ref_m2=s_ref_m2)
    body_cl0_proxy = compute_body_cl0_proxy(service=service, condition=condition, s_ref_m2=s_ref_m2)
    symmetry_factor = compute_body_symmetry_factor(body_cl0_proxy=body_cl0_proxy, body_proxies=body_proxies)
    alpha_break_factor = as_float(blind_parameters.get("alpha_break_sweep_factor"), default=0.096)
    alpha_break_deg = max(0.5, alpha_break_factor * as_float(service.wing_project.planform.sweep_le_deg))
    lift_retention = 1.0 - (float(condition.alpha_deg) / alpha_break_deg) ** 2
    lift_retention = float(np.clip(lift_retention, -0.75, 1.25))
    sweep_cos = as_float(blind_parameters.get("sweep_cosine"), default=1.0)
    raw_relief_target_cl = (
        -symmetry_factor
        * as_float(blind_parameters.get("alpha_relief_gain"), default=0.0)
        * abs(float(condition.alpha_deg))
        * pressure_proxy
        * as_float(body_proxies.get("body_form_drag_proxy"))
        / max(sweep_cos, 0.05)
    )
    cambered_body_delta_cl = body_cl0_proxy * sweep_cos * lift_retention
    y_half = np.asarray([as_float(row.get("y_m")) for row in strip_rows], dtype=float)
    lift_per_span_half = np.asarray([as_float(row.get("lift_per_span_N_m")) for row in strip_rows], dtype=float)
    chord_half = np.asarray([as_float(row.get("chord_m")) for row in strip_rows], dtype=float)
    onset_half = np.asarray([as_float(row.get("onset_proximity")) for row in strip_rows], dtype=float)
    thickness_half = np.asarray([as_float(row.get("thickness_factor")) for row in strip_rows], dtype=float)
    body_sections = sorted(service.wing_project.planform.body_sections, key=lambda section: section.y_pos)
    body_y_max = float(body_sections[-1].y_pos) if body_sections else 0.0
    return {
        "low_cl": low_cl,
        "low_cd": low_cd,
        "low_cm": low_cm,
        "q": q,
        "s_ref_m2": s_ref_m2,
        "strip_rows": strip_rows,
        "y_half": y_half,
        "lift_per_span_half": lift_per_span_half,
        "chord_half": chord_half,
        "onset_half": onset_half,
        "thickness_half": thickness_half,
        "pressure_proxy": pressure_proxy,
        "body_cl0_proxy": body_cl0_proxy,
        "symmetry_factor": symmetry_factor,
        "alpha_break_deg": alpha_break_deg,
        "alpha_ratio": abs(float(condition.alpha_deg)) / max(alpha_break_deg, 1e-9),
        "lift_retention": lift_retention,
        "cambered_body_delta_cl": cambered_body_delta_cl,
        "raw_relief_target_cl": raw_relief_target_cl,
        "reference_attached_cd": as_float(blind_parameters.get("reference_low_order_cd")),
        "body_y_max": body_y_max,
        "geometry_reference": low_order.get("geometry_reference"),
    }


def evaluate_fractional_unload_variant(
    components: dict[str, Any],
    blind_parameters: dict[str, float],
    body_proxies: dict[str, float],
    region_mode: str,
    onset_floor: float,
    onset_power: float,
    root_power: float,
    cap_fraction: float,
    target_scale: float,
) -> dict[str, Any]:
    delta_cl_body = float(components["cambered_body_delta_cl"])
    raw_relief_target_cl = float(components["raw_relief_target_cl"])
    y_half = np.asarray(components["y_half"], dtype=float)
    lift_per_span_half = np.asarray(components["lift_per_span_half"], dtype=float)
    chord_half = np.asarray(components["chord_half"], dtype=float)
    onset_half = np.asarray(components["onset_half"], dtype=float)
    thickness_half = np.asarray(components["thickness_half"], dtype=float)
    q = float(components["q"])
    s_ref_m2 = float(components["s_ref_m2"])
    body_y_max = float(components["body_y_max"])

    if len(y_half) < 2 or q <= 0.0 or s_ref_m2 <= 0.0 or raw_relief_target_cl >= 0.0:
        result = apply_cl_cd_from_delta(components, blind_parameters, body_proxies, delta_cl_body)
        result.update(
            {
                "body_delta_CL": delta_cl_body,
                "body_camber_delta_CL": float(components["cambered_body_delta_cl"]),
                "body_symmetric_relief_CL": 0.0,
                "variant_relief_region_end_m": body_y_max,
                "variant_relief_cap_fraction": cap_fraction,
                "variant_target_scale": target_scale,
            }
        )
        return result

    y_end = compute_region_end(region_mode, y_half=y_half, chord_half=chord_half, body_y_max=body_y_max)
    region_mask = y_half <= y_end + 1e-9
    if not np.any(region_mask):
        result = apply_cl_cd_from_delta(components, blind_parameters, body_proxies, delta_cl_body)
        result.update(
            {
                "body_delta_CL": delta_cl_body,
                "body_camber_delta_CL": float(components["cambered_body_delta_cl"]),
                "body_symmetric_relief_CL": 0.0,
                "variant_relief_region_end_m": y_end,
                "variant_relief_cap_fraction": cap_fraction,
                "variant_target_scale": target_scale,
            }
        )
        return result

    root_eta = np.zeros_like(y_half, dtype=float)
    if y_end > 1e-9:
        root_eta = np.clip(y_half / y_end, 0.0, 1.0)
    root_mask = np.clip(1.0 - root_eta, 0.0, 1.0) ** root_power
    onset_shape = np.maximum(onset_half, onset_floor) ** onset_power
    thickness_shape = np.maximum(thickness_half, 0.15)
    shape = onset_shape * thickness_shape * root_mask
    shape = np.where(region_mask, shape, 0.0)
    base_lift = np.maximum(lift_per_span_half, 0.0)

    target_relief_half = abs(raw_relief_target_cl) * q * s_ref_m2 * 0.5 * target_scale
    if target_relief_half <= 1e-9 or not np.any(shape > 0.0):
        result = apply_cl_cd_from_delta(components, blind_parameters, body_proxies, delta_cl_body)
        result.update(
            {
                "body_delta_CL": delta_cl_body,
                "body_camber_delta_CL": float(components["cambered_body_delta_cl"]),
                "body_symmetric_relief_CL": 0.0,
                "variant_relief_region_end_m": y_end,
                "variant_relief_cap_fraction": cap_fraction,
                "variant_target_scale": target_scale,
            }
        )
        return result

    def relieved_half_lift(scale: float) -> float:
        relief_fraction = np.minimum(cap_fraction, scale * shape)
        return float(np.trapezoid(relief_fraction * base_lift, y_half))

    max_possible_half = relieved_half_lift(1.0e6)
    applied_relief_half = min(target_relief_half, max_possible_half)
    scale_lo = 0.0
    scale_hi = 1.0
    while relieved_half_lift(scale_hi) < applied_relief_half and scale_hi < 1.0e6:
        scale_hi *= 2.0
    for _ in range(60):
        scale_mid = 0.5 * (scale_lo + scale_hi)
        if relieved_half_lift(scale_mid) >= applied_relief_half:
            scale_hi = scale_mid
        else:
            scale_lo = scale_mid

    applied_relief_half = relieved_half_lift(scale_hi)
    applied_relief_cl = -2.0 * applied_relief_half / max(q * s_ref_m2, 1e-9)
    delta_cl_body = float(components["cambered_body_delta_cl"]) + applied_relief_cl
    result = apply_cl_cd_from_delta(components, blind_parameters, body_proxies, delta_cl_body)
    result.update(
        {
            "body_delta_CL": delta_cl_body,
            "body_camber_delta_CL": float(components["cambered_body_delta_cl"]),
            "body_symmetric_relief_CL": applied_relief_cl,
            "variant_relief_region_end_m": y_end,
            "variant_relief_cap_fraction": cap_fraction,
            "variant_target_scale": target_scale,
            "variant_shape_integral": float(np.trapezoid(shape, y_half)),
        }
    )
    return result


def rel_error(value: float, target: float) -> float:
    return abs(value - target) / max(abs(target), 1e-9)


def build_case_context(
    dataset: DatasetSpec,
    args: argparse.Namespace,
    alpha_grid: np.ndarray,
) -> dict[str, Any]:
    _, Project, AeroSandboxService = import_model_stack()
    project = Project.load(str(dataset.project_json))
    service = AeroSandboxService(project.wing)
    s_ref_m2 = float(project.wing.planform.actual_area())
    cruise_altitude_m = float(project.wing.twist_trim.cruise_altitude_m)
    conditions = build_conditions_from_manifest(
        manifest_path=dataset.manifest_json.resolve(),
        cruise_altitude_m=cruise_altitude_m,
        only_conditions={"cruise", "takeoff"},
    )
    if not conditions:
        raise ValueError(f"No conditions found for dataset '{dataset.name}'.")

    body_proxies = build_body_geometry_proxies(service=service, s_ref_m2=s_ref_m2, x_samples=600)
    blind_parameters = build_blind_body_model_parameters(service=service, body_proxies=body_proxies)

    per_condition = {}
    for condition in conditions:
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
        per_condition[condition.name] = {
            "condition": condition,
            "low_order": low_order,
            "cfd": cfd,
            "baseline": apply_blind_hybrid_bwb_body_method(
                service=service,
                condition=condition,
                low_order=low_order,
                body_proxies=body_proxies,
                blind_parameters={**blind_parameters},
                s_ref_m2=s_ref_m2,
                body_alpha_break_sweep_factor=0.095,
                reference_attached_cd=0.0,
            ),
        }

    reference_condition = min(conditions, key=lambda item: abs(float(item.alpha_deg)))
    blind_parameters["reference_condition"] = reference_condition.name
    blind_parameters["reference_low_order_cd"] = as_float(
        per_condition[reference_condition.name]["low_order"].get("CD")
    )

    for block in per_condition.values():
        block["baseline"] = apply_blind_hybrid_bwb_body_method(
            service=service,
            condition=block["condition"],
            low_order=block["low_order"],
            body_proxies=body_proxies,
            blind_parameters=blind_parameters,
            s_ref_m2=s_ref_m2,
            body_alpha_break_sweep_factor=0.095,
            reference_attached_cd=as_float(blind_parameters.get("reference_low_order_cd")),
        )
        block["components"] = compute_blind_components(
            service=service,
            condition=block["condition"],
            low_order=block["low_order"],
            body_proxies=body_proxies,
            blind_parameters=blind_parameters,
            s_ref_m2=s_ref_m2,
        )

    return {
        "dataset": dataset,
        "service": service,
        "s_ref_m2": s_ref_m2,
        "body_proxies": body_proxies,
        "blind_parameters": blind_parameters,
        "conditions": per_condition,
    }


def score_variant(results: list[dict[str, Any]]) -> dict[str, float]:
    rel_errors = []
    for item in results:
        cfd = item["cfd"]
        model = item["model"]
        rel_errors.append(rel_error(as_float(model.get("CL")), as_float(cfd.get("CL"))))
        rel_errors.append(rel_error(as_float(model.get("CD")), as_float(cfd.get("CD"))))
    return {
        "max_rel_error": float(max(rel_errors)) if rel_errors else 0.0,
        "mean_rel_error": float(sum(rel_errors) / max(len(rel_errors), 1)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = build_alpha_grid(
        alpha_min=float(args.polar_alpha_min),
        alpha_max=float(args.polar_alpha_max),
        alpha_steps=int(args.polar_alpha_steps),
    )

    contexts = [build_case_context(dataset, args=args, alpha_grid=alpha_grid) for dataset in default_datasets()]

    variants = [
        {
            "name": "baseline_current",
            "kind": "baseline",
        },
        {
            "name": "raw_target_full",
            "kind": "raw_target",
            "target_scale": 1.0,
        },
    ]

    for region_mode in ("body", "chord_0.70", "chord_0.55", "span_0.35"):
        for onset_floor in (0.0, 0.15, 0.30):
            for onset_power in (1.0, 1.5, 2.0):
                for root_power in (0.5, 1.0, 2.0):
                    for cap_fraction in (0.60, 0.80, 0.95):
                        for target_scale in (0.90, 1.00, 1.10, 1.20, 1.30):
                            variants.append(
                                {
                                    "name": (
                                        f"frac_{region_mode}_floor{onset_floor:.2f}_"
                                        f"op{onset_power:.1f}_rp{root_power:.1f}_"
                                        f"cap{cap_fraction:.2f}_ts{target_scale:.2f}"
                                    ),
                                    "kind": "fractional_unload",
                                    "region_mode": region_mode,
                                    "onset_floor": onset_floor,
                                    "onset_power": onset_power,
                                    "root_power": root_power,
                                    "cap_fraction": cap_fraction,
                                    "target_scale": target_scale,
                                }
                            )

    ranked_results = []
    for variant in variants:
        per_case_rows = []
        for context in contexts:
            dataset = context["dataset"]
            service = context["service"]
            body_proxies = context["body_proxies"]
            blind_parameters = context["blind_parameters"]
            for condition_name, block in context["conditions"].items():
                condition = block["condition"]
                cfd = block["cfd"]
                if variant["kind"] == "baseline":
                    model = block["baseline"]
                elif variant["kind"] == "raw_target":
                    delta_cl_body = (
                        float(block["components"]["cambered_body_delta_cl"])
                        + float(block["components"]["raw_relief_target_cl"]) * float(variant["target_scale"])
                    )
                    model = apply_cl_cd_from_delta(
                        block["components"],
                        blind_parameters=blind_parameters,
                        body_proxies=body_proxies,
                        delta_cl_body=delta_cl_body,
                    )
                    model["body_delta_CL"] = delta_cl_body
                    model["body_camber_delta_CL"] = float(block["components"]["cambered_body_delta_cl"])
                    model["body_symmetric_relief_CL"] = float(block["components"]["raw_relief_target_cl"]) * float(
                        variant["target_scale"]
                    )
                else:
                    model = evaluate_fractional_unload_variant(
                        block["components"],
                        blind_parameters=blind_parameters,
                        body_proxies=body_proxies,
                        region_mode=str(variant["region_mode"]),
                        onset_floor=float(variant["onset_floor"]),
                        onset_power=float(variant["onset_power"]),
                        root_power=float(variant["root_power"]),
                        cap_fraction=float(variant["cap_fraction"]),
                        target_scale=float(variant["target_scale"]),
                    )

                per_case_rows.append(
                    {
                        "dataset": dataset.name,
                        "condition": condition_name,
                        "model": model,
                        "cfd": cfd,
                        "delta_cl": as_float(model.get("CL")) - as_float(cfd.get("CL")),
                        "delta_cd": as_float(model.get("CD")) - as_float(cfd.get("CD")),
                        "rel_cl_error": rel_error(as_float(model.get("CL")), as_float(cfd.get("CL"))),
                        "rel_cd_error": rel_error(as_float(model.get("CD")), as_float(cfd.get("CD"))),
                    }
                )

        scores = score_variant(per_case_rows)
        ranked_results.append(
            {
                "variant": variant,
                "scores": scores,
                "results": per_case_rows,
            }
        )

    ranked_results.sort(
        key=lambda item: (
            item["scores"]["max_rel_error"],
            item["scores"]["mean_rel_error"],
        )
    )

    top_k = max(1, int(args.top_k))
    top_results = ranked_results[:top_k]
    summary = {
        "top_ranked_variants": top_results,
        "baseline_current": next(item for item in ranked_results if item["variant"]["name"] == "baseline_current"),
        "raw_target_full": next(item for item in ranked_results if item["variant"]["name"] == "raw_target_full"),
        "searched_variant_count": len(ranked_results),
    }

    json_path = output_dir / "takeoff_relief_variant_search_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_rows = []
    for rank, item in enumerate(top_results, start=1):
        variant = item["variant"]
        for case_row in item["results"]:
            model = case_row["model"]
            cfd = case_row["cfd"]
            csv_rows.append(
                {
                    "rank": rank,
                    "variant_name": variant["name"],
                    "kind": variant["kind"],
                    "dataset": case_row["dataset"],
                    "condition": case_row["condition"],
                    "max_rel_error": item["scores"]["max_rel_error"],
                    "mean_rel_error": item["scores"]["mean_rel_error"],
                    "model_CL": as_float(model.get("CL")),
                    "model_CD": as_float(model.get("CD")),
                    "cfd_CL": as_float(cfd.get("CL")),
                    "cfd_CD": as_float(cfd.get("CD")),
                    "delta_CL": case_row["delta_cl"],
                    "delta_CD": case_row["delta_cd"],
                    "rel_CL_error": case_row["rel_cl_error"],
                    "rel_CD_error": case_row["rel_cd_error"],
                    "body_delta_CL": as_float(model.get("body_delta_CL")),
                    "body_delta_CD": as_float(model.get("body_delta_CD")),
                    "body_symmetric_relief_CL": as_float(model.get("body_symmetric_relief_CL")),
                }
            )

    csv_path = output_dir / "takeoff_relief_variant_search_top.csv"
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote relief-variant search summary: {json_path}")
    print(f"Wrote relief-variant search table: {csv_path}")


if __name__ == "__main__":
    main()
