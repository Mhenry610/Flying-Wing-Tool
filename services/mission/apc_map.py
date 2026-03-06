import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline

from .util import floats_in_line


class APCMap:
    """
    Parse APC .dat and build a smooth CT,CP surface f(RPM,J) using:
      - PCHIP along J within each RPM band, resampled to a common J-grid
      - RectBivariateSpline across (RPM,J) on that grid
    """

    def __init__(self):
        self.loaded = False
        self.d_in = None
        self.rpms = None
        self.Jg = None
        self.Ct_surf = None
        self.Cp_surf = None

    def load(self, path: str, j_nodes: int = 80, smooth: float = 0.0):
        txt = Path(path).read_text(errors="replace")
        # Diameter from filename if present (e.g., "12x8.dat")
        m = re.search(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', Path(path).stem)
        self.d_in = float(m.group(1)) if m else None

        # Pull blocks keyed by "PROP RPM = ####"
        blocks = {}
        lines = txt.splitlines()
        N = len(lines)
        idxs = [i for i, ln in enumerate(lines) if re.search(r'PROP\s+RPM\s*=\s*([0-9.]+)', ln)]
        for k, i0 in enumerate(idxs):
            m2 = re.search(r'PROP\s+RPM\s*=\s*([0-9.]+)', lines[i0])
            rpm = float(m2.group(1))
            i1 = idxs[k + 1] if k + 1 < len(idxs) else N
            rows = []
            # parse numeric rows after header line(s)
            for ln in lines[i0 + 1:i1]:
                if not ln.strip():
                    continue
                nums = floats_in_line(ln)
                if len(nums) < 5:
                    continue
                # Expect columns: [mph, J, Pe, Ct, Cp, ... Watts, HP, Torque, Thrust(N), ...]
                rows.append(nums)
            if rows:
                arr = np.array(rows, dtype=float)
                J = arr[:, 1]
                Ct = arr[:, 3]
                Cp = arr[:, 4]
                # Sort and unique in J
                ksort = np.argsort(J)
                J = J[ksort]
                Ct = Ct[ksort]
                Cp = Cp[ksort]
                Juniq, idx = np.unique(J, return_index=True)
                blocks[rpm] = dict(J=Juniq, Ct=Ct[idx], Cp=Cp[idx])

        if not blocks:
            raise ValueError("No APC data blocks found.")

        # RPM list
        rpms = np.array(sorted(blocks.keys()))
        # Common J-range across all RPMs
        Jmin = max(b["J"][0] for b in blocks.values())
        Jmax = min(b["J"][-1] for b in blocks.values())
        if Jmax <= Jmin:
            raise ValueError("APC J ranges do not overlap across RPM bands.")
        Jg = np.linspace(Jmin, Jmax, j_nodes)

        # Resample with PCHIP per RPM along J
        Ct_grid = np.zeros((len(rpms), len(Jg)))
        Cp_grid = np.zeros_like(Ct_grid)
        for i, r in enumerate(rpms):
            b = blocks[r]
            fCt = PchipInterpolator(b["J"], b["Ct"], extrapolate=False)
            fCp = PchipInterpolator(b["J"], b["Cp"], extrapolate=False)
            Ct_grid[i, :] = np.nan_to_num(fCt(Jg), nan=0.0)
            Cp_grid[i, :] = np.nan_to_num(fCp(Jg), nan=0.0)

        # Smooth bicubic surface f(RPM,J)
        self.rpms = rpms
        self.Jg = Jg
        self.Ct_surf = RectBivariateSpline(rpms, Jg, Ct_grid, s=smooth)
        self.Cp_surf = RectBivariateSpline(rpms, Jg, Cp_grid, s=smooth)
        self.loaded = True

    def ct_cp(self, rpm: float, J: float):
        if not self.loaded:
            raise RuntimeError("APC map not loaded.")
        # clamp to grid
        r = float(np.clip(rpm, self.rpms[0], self.rpms[-1]))
        j = float(np.clip(J, self.Jg[0], self.Jg[-1]))
        Ct = float(self.Ct_surf(r, j, grid=False))
        Cp = float(self.Cp_surf(r, j, grid=False))
        # physical guards
        return max(0.0, Ct), max(0.0, Cp)

    def thrust_power(self, rpm: float, V0_mps: float, rho: float, D_m: float):
        n = rpm / 60.0
        if n <= 1e-9 or D_m <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        J = V0_mps / (n * D_m)
        Ct, Cp = self.ct_cp(rpm, J)
        T = rho * (n ** 2) * (D_m ** 4) * Ct
        P = rho * (n ** 3) * (D_m ** 5) * Cp
        return J, Ct, T, P

