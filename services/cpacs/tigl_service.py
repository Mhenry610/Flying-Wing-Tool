from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TiGLDiagnostic:
    available: bool
    version: str | None = None
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"available": self.available, "version": self.version, "messages": list(self.messages)}


class OptionalTiGLService:
    """Optional TiGL facade that fails gracefully when TiGL/TiXI are not installed."""

    def probe(self) -> TiGLDiagnostic:
        try:
            import tigl3  # type: ignore
        except Exception as exc:
            return TiGLDiagnostic(False, None, [f"TiGL unavailable: {exc}"])
        version = getattr(tigl3, "__version__", None)
        return TiGLDiagnostic(True, str(version) if version else None, ["TiGL Python bindings are importable."])

    def validate_cpacs_text(self, xml_text: str) -> TiGLDiagnostic:
        probe = self.probe()
        if not probe.available:
            return probe
        if "<cpacs" not in xml_text:
            return TiGLDiagnostic(True, probe.version, ["Input does not look like CPACS XML."])
        return TiGLDiagnostic(True, probe.version, ["CPACS text is well-formed enough for adapter-level handoff."])

