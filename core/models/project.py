from __future__ import annotations

from dataclasses import dataclass, field

from .airfoil import AirfoilInterpolation
from .planform import PlanformGeometry
from .twist_trim import TwistTrimParameters


@dataclass
class WingProject:
    name: str = "FlyingWingProject"
    planform: PlanformGeometry = field(default_factory=PlanformGeometry)
    twist_trim: TwistTrimParameters = field(default_factory=TwistTrimParameters)
    airfoil: AirfoilInterpolation = field(default_factory=AirfoilInterpolation)
    optimized_twist_deg: Optional[List[float]] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "planform": self.planform.as_dict(),
            "twist_trim": self.twist_trim.as_dict(),
            "airfoil": self.airfoil.as_dict(),
            "optimized_twist_deg": self.optimized_twist_deg,
        }
