from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

from .apc_map import APCMap
from .motor import MotorProp
from .ops import solve_for_thrust_total
from .util import isa_density


@dataclass
class MissionSegment:
    name: str
    duration_s: float
    speed_mps: float
    alt_m: float
    # If provided (>=0), overrides mode-based thrust calculation
    thrust_total_N: Optional[float] = None
    # Optional mode: 'cruise' | 'takeoff' | 'thrust'
    mode: Optional[str] = None
    # For takeoff mode
    distance_m: Optional[float] = None


@dataclass
class AeroConfig:
    # Lift coefficient at cruise (required)
    cl_cruise: float
    # Drag coefficients per component (dimensionless)
    cd_wing: float = 0.0
    cd_fuse: float = 0.0
    cd_tail: float = 0.0
    # Reference areas per component [m^2]
    s_wing_m2: Optional[float] = None  # if None/<=0, infer from W, V, rho, Cl at cruise
    s_fuse_m2: float = 0.0
    s_tail_m2: float = 0.0
    # Other aero/ground params
    cl_max: float = 1.6
    mu_roll: float = 0.03


def compute_ref_area(weight_N: float, cl_cruise: float, v_mps: float, alt_m: float) -> float:
    """Compute wing reference area S from W = q S Cl at the given cruise point."""
    rho = isa_density(alt_m)
    q = 0.5 * rho * v_mps * v_mps
    if q <= 0 or cl_cruise <= 0:
        raise ValueError("Invalid cruise point for area computation.")
    return weight_N / (q * cl_cruise)


def thrust_required_cruise_split(q: float, aero: AeroConfig, s_wing_m2: float) -> float:
    """Total drag at cruise using separate CD and reference areas.

    T_req ≈ q * (CD_wing*S_w + CD_fuse*S_f + CD_tail*S_t)
    """
    s_f = max(0.0, aero.s_fuse_m2 or 0.0)
    s_t = max(0.0, aero.s_tail_m2 or 0.0)
    s_w = max(0.0, s_wing_m2)
    return q * (aero.cd_wing * s_w + aero.cd_fuse * s_f + aero.cd_tail * s_t)


def thrust_required_cruise(weight_N: float, aero: AeroConfig) -> float:
    """Level flight thrust ≈ W * (Cd/Cl)."""
    if aero.cl_cruise <= 0:
        raise ValueError("cl_cruise must be > 0")
    return weight_N * (aero.cd / aero.cl_cruise)


def estimate_takeoff_requirements(
    distance_m: float,
    weight_N: float,
    S_m2: float,
    alt_m: float,
    aero: AeroConfig,
):
    """
    Rough takeoff model.
    - Liftoff speed V_to ≈ 1.2 * V_stall(TO), where V_stall(TO) uses CL_max.
    - Average acceleration a_avg = V_to^2 / (2 * s).
    - Required average thrust T_avg = m a_avg + D_avg + mu_roll * (W - L_avg).
      where averages use V_avg ≈ 0.7 * V_to.
    - Estimated roll time t ≈ 2*s / V_to (constant-accel approximation).
    Returns: dict(T_avg_N, V_to, t_roll_s)
    """
    g0 = 9.80665
    rho = isa_density(alt_m)
    cl_max = max(1e-6, aero.cl_max)
    V_stall = (2.0 * weight_N / (rho * S_m2 * cl_max)) ** 0.5
    V_to = 1.2 * V_stall
    if distance_m <= 0 or not np.isfinite(distance_m):
        raise ValueError("distance_m must be > 0 for takeoff segment")
    a_avg = (V_to * V_to) / (2.0 * distance_m)
    m = weight_N / g0
    V_avg = 0.7 * V_to
    q_avg = 0.5 * rho * V_avg * V_avg
    L_avg = q_avg * S_m2 * (0.8 * cl_max)  # approximate average Cl during roll (wing)
    D_avg = q_avg * (
        aero.cd_wing * S_m2
        + aero.cd_fuse * (aero.s_fuse_m2 or 0.0)
        + aero.cd_tail * (aero.s_tail_m2 or 0.0)
    )
    T_avg = m * a_avg + D_avg + aero.mu_roll * max(0.0, weight_N - L_avg)
    t_roll = 2.0 * distance_m / max(1e-6, V_to)
    return dict(T_avg_N=T_avg, V_to=V_to, t_roll_s=t_roll)


def evaluate_segment(
    seg: MissionSegment,
    apc: APCMap,
    motor: MotorProp,
    Vb: float,
    nprops: int,
    D_m: float,
    weight_N: Optional[float] = None,
    aero: Optional[AeroConfig] = None,
    S_ref_m2: Optional[float] = None,
) -> Dict:
    rho = isa_density(seg.alt_m)

    # Determine required thrust
    T_req = None
    if seg.thrust_total_N is not None:
        T_req = float(seg.thrust_total_N)
    elif seg.mode == "cruise":
        if weight_N is None or aero is None:
            raise ValueError("Cruise mode requires weight_N and aero config")
        # Determine wing area from provided S_ref or aero.s_wing_m2; infer if needed
        rho_c = isa_density(seg.alt_m)
        q_c = 0.5 * rho_c * seg.speed_mps * seg.speed_mps
        s_w = S_ref_m2 if (S_ref_m2 is not None and S_ref_m2 > 0) else (aero.s_wing_m2 or 0.0)
        if s_w <= 0.0:
            s_w = compute_ref_area(weight_N, aero.cl_cruise, seg.speed_mps, seg.alt_m)
        T_req = thrust_required_cruise_split(q_c, aero, s_w)
    elif seg.mode == "takeoff":
        if weight_N is None or aero is None or S_ref_m2 is None or seg.distance_m is None:
            raise ValueError("Takeoff mode requires weight_N, aero, S_ref_m2, and distance_m")
        est = estimate_takeoff_requirements(seg.distance_m, weight_N, S_ref_m2, seg.alt_m, aero)
        T_req = est["T_avg_N"]
        # If duration not specified (>0), approximate from t_roll
        if seg.duration_s <= 0:
            seg = MissionSegment(
                name=seg.name,
                duration_s=est["t_roll_s"],
                speed_mps=0.7 * est["V_to"],
                alt_m=seg.alt_m,
                thrust_total_N=T_req,
                mode=seg.mode,
                distance_m=seg.distance_m,
            )
    else:
        raise ValueError("Segment must specify thrust_total_N or a supported mode")

    sol = solve_for_thrust_total(seg.speed_mps, T_req, motor, Vb, nprops, rho, D_m, apc)
    if sol is None:
        # Unreachable: report s=1 operating point and shortfall
        out_hi = motor.solve_rpm(1.0 * Vb, seg.speed_mps, rho, D_m, apc)
        T_total = nprops * out_hi["T"]
        shortfall = T_req - T_total
        Pelec_total = nprops * (1.0 * Vb) * out_hi["I"]
        return dict(
            name=seg.name,
            reachable=False,
            duty=1.0,
            rpm=out_hi["rpm"],
            I=out_hi["I"],
            T_total=T_total,
            thrust_required=T_req,
            shortfall_N=shortfall,
            Pelec_total=Pelec_total,
            duration_s=float(seg.duration_s),
            energy_J=Pelec_total * float(seg.duration_s),
        )

    out = sol["out"]
    return dict(
        name=seg.name,
        reachable=True,
        duty=sol["s"],
        rpm=out["rpm"],
        I=out["I"],
        T_total=sol["T_total"],
        thrust_required=T_req,
        shortfall_N=0.0,
        Pelec_total=sol["P_elec_total"],
        duration_s=float(seg.duration_s),
        energy_J=sol["P_elec_total"] * float(seg.duration_s),
    )


def evaluate_mission(
    segments: List[MissionSegment],
    apc: APCMap,
    motor: MotorProp,
    Vb: float,
    nprops: int,
    D_m: float,
    weight_N: Optional[float] = None,
    aero: Optional[AeroConfig] = None,
    S_ref_m2: Optional[float] = None,
) -> Dict:
    # Compute reference wing area from the first cruise-like segment if needed
    S_ref = None
    if S_ref_m2 is not None and S_ref_m2 > 0:
        S_ref = float(S_ref_m2)
    elif weight_N is not None and aero is not None:
        for seg in segments:
            if (seg.mode == "cruise") or (seg.mode is None and seg.thrust_total_N is None):
                try:
                    S_ref = compute_ref_area(weight_N, aero.cl_cruise, seg.speed_mps, seg.alt_m)
                    break
                except Exception:
                    continue
    results = [
        evaluate_segment(seg, apc, motor, Vb, nprops, D_m, weight_N=weight_N, aero=aero, S_ref_m2=S_ref)
        for seg in segments
    ]
    total_energy_J = float(sum(r.get("energy_J", 0.0) for r in results))
    total_time_s = float(sum(r.get("duration_s", 0.0) for r in results))
    return dict(results=results, total_energy_J=total_energy_J, total_time_s=total_time_s)
