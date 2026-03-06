from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LiftDistributionType = Literal["bell", "elliptical"]


@dataclass
class TwistTrimParameters:
    gross_takeoff_weight_kg: float = 12.0
    cruise_altitude_m: float = 2000.0
    cm0_root: float = -0.02
    cm0_tip: float = -0.02
    zero_lift_aoa_root_deg: float = -2.0
    zero_lift_aoa_tip_deg: float = -1.0
    cl_alpha_root_per_deg: float = 0.08
    cl_alpha_tip_per_deg: float = 0.075
    design_cl: float = 0.45
    lift_distribution: LiftDistributionType = "bell"
    static_margin_percent: float = 8.0
    estimated_cl_max: float = 1.2
    estimated_cl_max_speed: float = 0.35

    def as_dict(self) -> dict:
        return {
            "gross_takeoff_weight_kg": self.gross_takeoff_weight_kg,
            "cruise_altitude_m": self.cruise_altitude_m,
            "cm0_root": self.cm0_root,
            "cm0_tip": self.cm0_tip,
            "zero_lift_aoa_root_deg": self.zero_lift_aoa_root_deg,
            "zero_lift_aoa_tip_deg": self.zero_lift_aoa_tip_deg,
            "cl_alpha_root_per_deg": self.cl_alpha_root_per_deg,
            "cl_alpha_tip_per_deg": self.cl_alpha_tip_per_deg,
            "design_cl": self.design_cl,
            "lift_distribution": self.lift_distribution,
            "static_margin_percent": self.static_margin_percent,
            "estimated_cl_max": self.estimated_cl_max,
            "estimated_cl_max_speed": self.estimated_cl_max_speed,
        }
