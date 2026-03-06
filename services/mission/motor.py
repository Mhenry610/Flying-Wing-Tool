import math
from typing import Dict

from .apc_map import APCMap


class MotorProp:
    """
    First-order motor model.
    RPM solved such that motor voltage equals applied battery voltage.
    """

    def __init__(self, KV_rpm_per_V: float, Ri_mOhm: float, Io_A: float, V_at_Io: float | None = None):
        self.KV = KV_rpm_per_V
        self.R20 = Ri_mOhm / 1000.0  # Ohm
        self.Io = Io_A
        self.V_at_Io = V_at_Io

        self.kv_rad_per_V = self.KV * (2 * math.pi / 60)
        self.Ke = 1.0 / max(1e-9, self.kv_rad_per_V)  # V/(rad/s)
        self.Kt = self.Ke  # N·m/A (SI motor)

    def I0_eff(self, omega: float):
        if self.V_at_Io is None or self.V_at_Io <= 0:
            # scale proportional to speed relative to 12V reference
            return self.Io * (omega / max(1.0, self.kv_rad_per_V * 12.0))
        # scale Io linearly with speed relative to measured no-load
        omega_nl = self.kv_rad_per_V * self.V_at_Io
        return self.Io * (omega / max(1.0, omega_nl))

    def solve_rpm(self, V_batt: float, V0_mps: float, rho: float, D_m: float, APC: APCMap) -> Dict[str, float]:
        # bracket: [0, KV*V_batt]
        rpm_lo = 0.0
        rpm_hi = max(10.0, self.KV * max(0.0, V_batt))

        def residual(rpm: float):
            # motor-per-prop residual V_req - V_batt
            J, Ct, T, Pshaft = APC.thrust_power(rpm, V0_mps, rho, D_m)
            w = rpm * 2 * math.pi / 60.0
            if w < 1e-9:
                return -V_batt, 0.0, 0.0, 0.0, 0.0, J, Ct
            tau = Pshaft / w
            I = self.I0_eff(w) + tau / self.Kt
            V_req = self.Ke * w + I * self.R20
            return V_req - V_batt, T, Pshaft, I, w, J, Ct

        # Bisection on V_req - V_batt
        f_lo = residual(rpm_lo)[0]
        f_hi = residual(rpm_hi)[0]
        tries = 0
        while f_lo * f_hi > 0 and tries < 25:
            rpm_hi *= 0.8
            f_hi = residual(rpm_hi)[0]
            tries += 1
        if f_lo * f_hi > 0:
            # fallback: use hi as solution
            r = rpm_hi
        else:
            for _ in range(80):
                r = 0.5 * (rpm_lo + rpm_hi)
                f_mid = residual(r)[0]
                if abs(f_mid) < 1e-3 or abs(rpm_hi - rpm_lo) < 1e-3:
                    break
                if f_lo * f_mid <= 0:
                    rpm_hi = r
                    f_hi = f_mid
                else:
                    rpm_lo = r
                    f_lo = f_mid

        # Final evaluation
        f, T, Pshaft, I, w, J, Ct = residual(r)
        V_req = self.Ke * w + I * self.R20
        Pelec = max(0.0, V_req * I)
        eta = (Pshaft / Pelec) if Pelec > 1e-9 else 0.0
        return dict(rpm=r, T=T, Pshaft=Pshaft, I=I, Pelec=Pelec, eta=eta, J=J, Ct=Ct)

