import math
from typing import Dict, Optional

import numpy as np

from .apc_map import APCMap
from .motor import MotorProp


def solve_for_thrust_total(
    v0_mps: float,
    thrust_total_N: float,
    motor: MotorProp,
    Vb: float,
    nprops: int,
    rho: float,
    D_m: float,
    apc: APCMap,
    tol_rel: float = 1e-3,
    max_iter: int = 40,
) -> Optional[Dict[str, float]]:
    """
    Solve for duty s in [0,1] to achieve a required total thrust at a given speed/altitude.
    Returns a dict with keys: s, out (per-motor dict from solve_rpm), T_total, P_elec_total.
    Returns None if target is unattainable (even at s=1).
    """
    target = float(thrust_total_N)
    if target <= 0:
        # No thrust required
        out = motor.solve_rpm(0.0, v0_mps, rho, D_m, apc)
        return dict(s=0.0, out=out, T_total=0.0, P_elec_total=0.0)

    # Bounds
    s_lo, s_hi = 0.0, 1.0
    out_lo = motor.solve_rpm(s_lo * Vb, v0_mps, rho, D_m, apc)
    T_lo = nprops * out_lo["T"]
    out_hi = motor.solve_rpm(s_hi * Vb, v0_mps, rho, D_m, apc)
    T_hi = nprops * out_hi["T"]

    if target > T_hi * (1.0 + 1e-6):
        # Unreachable
        return None
    if target < T_lo * (1.0 - 1e-6):
        return dict(s=0.0, out=out_lo, T_total=T_lo, P_elec_total=nprops * (s_lo * Vb) * out_lo["I"])  # zero thrust

    # Bisection on duty to meet thrust
    for _ in range(max_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        out_mid = motor.solve_rpm(s_mid * Vb, v0_mps, rho, D_m, apc)
        T_mid = nprops * out_mid["T"]
        if abs(T_mid - target) / max(1.0, target) < tol_rel:
            return dict(s=s_mid, out=out_mid, T_total=T_mid, P_elec_total=nprops * (s_mid * Vb) * out_mid["I"])
        if T_mid < target:
            s_lo = s_mid
        else:
            s_hi = s_mid
    # Return last mid
    return dict(s=s_mid, out=out_mid, T_total=T_mid, P_elec_total=nprops * (s_mid * Vb) * out_mid["I"])

