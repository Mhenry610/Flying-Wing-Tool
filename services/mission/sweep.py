import math
from typing import Dict, List

import numpy as np

from .apc_map import APCMap
from .motor import MotorProp


def compute_sweep(
    V: np.ndarray,
    marks_W: List[float],
    motor: MotorProp,
    Vb: float,
    nprops: int,
    rho: float,
    D_m: float,
    apc: APCMap,
) -> Dict:
    """
    Compute sweep over velocities and constant-total-power marks.

    Returns dict with keys: series (list), p_full (ndarray), p_full_max (float)
    Each series[i] contains: P, V, thrust, rpm, eff_overall, torque, reachable
    """
    V = np.asarray(V, dtype=float)

    # Precompute full-throttle total electrical power envelope across V
    p_full = np.full_like(V, np.nan, dtype=float)
    for j, v0 in enumerate(V):
        out_full = motor.solve_rpm(Vb, v0, rho, D_m, apc)
        p_full[j] = nprops * out_full.get("Pelec", (Vb * out_full["I"]))
    p_full_max = float(np.nanmax(p_full)) if p_full.size else 0.0

    def solve_at_power(v0: float, p_total_target: float):
        s_lo, s_hi = 0.0, 1.0
        out_lo = motor.solve_rpm(s_lo * Vb, v0, rho, D_m, apc)
        p_lo = nprops * (s_lo * Vb) * out_lo["I"]
        out_hi = motor.solve_rpm(s_hi * Vb, v0, rho, D_m, apc)
        p_hi = nprops * (s_hi * Vb) * out_hi["I"]
        if p_total_target > p_hi * 1.0001:
            return None
        if p_total_target < p_lo * 0.9999:
            s = 0.0
            return dict(out=out_lo, s=s)
        for _ in range(40):
            s_mid = 0.5 * (s_lo + s_hi)
            out_mid = motor.solve_rpm(s_mid * Vb, v0, rho, D_m, apc)
            p_mid = nprops * (s_mid * Vb) * out_mid["I"]
            if abs(p_mid - p_total_target) / max(1.0, p_total_target) < 1e-3:
                return dict(out=out_mid, s=s_mid)
            if p_mid < p_total_target:
                s_lo = s_mid
            else:
                s_hi = s_mid
        return dict(out=out_mid, s=s_mid)

    series = []
    for pmark in marks_W:
        thrust_i = np.full_like(V, np.nan, dtype=float)
        rpm_i = np.full_like(V, np.nan, dtype=float)
        eff_overall_i = np.full_like(V, np.nan, dtype=float)
        torque_i = np.full_like(V, np.nan, dtype=float)
        for j, v0 in enumerate(V):
            # skip if mark exceeds full-throttle power at this speed
            if np.isfinite(p_full[j]) and (pmark > p_full[j] * 1.0001):
                continue
            sol = solve_at_power(v0, pmark)
            if sol is None:
                continue
            out = sol["out"]
            thrust_i[j] = out["T"] * nprops
            rpm_i[j] = out["rpm"]
            w = out["rpm"] * 2 * math.pi / 60.0
            torque_i[j] = (out["Pshaft"] / w) if w > 1e-9 else np.nan  # per motor
            eff_overall_i[j] = (thrust_i[j] * v0) / pmark if pmark > 1e-9 else np.nan
        ok = np.isfinite(rpm_i).any()
        series.append(
            dict(
                P=pmark,
                V=V,
                thrust=thrust_i,
                rpm=rpm_i,
                eff_overall=eff_overall_i,
                torque=torque_i,
                reachable=bool(ok),
            )
        )

    return dict(series=series, p_full=p_full, p_full_max=p_full_max)

