from __future__ import annotations

from dataclasses import dataclass, field

from core.aircraft.bodies import BodyEnvelope, BodyObject
from core.aircraft.project import AircraftProject


BODY_MODEL_REFERENCES = [
    {
        "label": "NASA Glenn, drag reference-area choices",
        "url": "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/size-effects-on-drag/",
    },
]


@dataclass
class BodyAnalysisResult:
    body_uid: str
    wetted_area_estimate_m2: float
    frontal_area_m2: float
    drag_area_estimate_m2: float
    payload_bay_volume_m3: float | None
    warnings: list[str] = field(default_factory=list)
    references: list[dict] = field(default_factory=lambda: list(BODY_MODEL_REFERENCES))

    def as_dict(self) -> dict:
        return {
            "body_uid": self.body_uid,
            "wetted_area_estimate_m2": self.wetted_area_estimate_m2,
            "frontal_area_m2": self.frontal_area_m2,
            "drag_area_estimate_m2": self.drag_area_estimate_m2,
            "payload_bay_volume_m3": self.payload_bay_volume_m3,
            "warnings": list(self.warnings),
            "references": list(self.references),
        }


def analyze_body_placeholder(body: BodyObject, reference_area_m2: float) -> BodyAnalysisResult:
    warnings: list[str] = []
    envelope = body.envelope or BodyEnvelope()
    wetted = envelope.wetted_area_estimate_m2()
    frontal = envelope.frontal_area_m2()
    drag_area = body.drag_area_estimate_m2 if body.drag_area_estimate_m2 is not None else 0.08 * frontal
    volume = None
    if body.role in ("payload_bay", "fuselage") and envelope.length_m > 0.0:
        volume = envelope.length_m * frontal
    if body.role == "fuselage":
        warnings.append("Fuselage aero is a drag-area estimate only; body lift and interference are not modeled.")
    if reference_area_m2 <= 0.0:
        warnings.append("Reference area is not positive, so body drag cannot be non-dimensionalized.")
    return BodyAnalysisResult(body.uid, wetted, frontal, drag_area, volume, warnings)


def analyze_bodies(project: AircraftProject) -> list[BodyAnalysisResult]:
    return [
        analyze_body_placeholder(body, project.reference.reference_area_m2)
        for body in project.bodies
        if body.active
    ]

