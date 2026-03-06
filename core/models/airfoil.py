from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AirfoilInterpolation:
    bwb_airfoil: str = "naca2412"     # Airfoil for BWB body sections
    root_airfoil: str = "naca2412"    # Wing root airfoil (at BWB junction)
    tip_airfoil: str = "naca0009"     # Wing tip airfoil
    num_sections: int = 13

    def as_dict(self) -> dict:
        return {
            "bwb_airfoil": self.bwb_airfoil,
            "root_airfoil": self.root_airfoil,
            "tip_airfoil": self.tip_airfoil,
            "num_sections": self.num_sections,
        }
