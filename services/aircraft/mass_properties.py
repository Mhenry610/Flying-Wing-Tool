from __future__ import annotations

from core.aircraft.mass import MassBalance, MassItem
from core.aircraft.project import AircraftProject


def compute_mass_balance(project: AircraftProject, include_body_masses: bool = True) -> MassBalance:
    items = list(project.mass_items)
    if include_body_masses:
        for body in project.bodies:
            if body.active and body.mass_properties and body.mass_properties.mass_kg > 0.0:
                items.append(
                    MassItem(
                        uid=f"{body.uid}_mass",
                        name=f"{body.name} mass",
                        mass_kg=body.mass_properties.mass_kg,
                        cg_m=body.mass_properties.cg_m,
                        category="fuselage" if body.role == "fuselage" else "other",
                        source_uid=body.uid,
                    )
                )
    total = sum(item.mass_kg for item in items)
    warnings: list[str] = []
    if total <= 0.0:
        return MassBalance(0.0, project.reference.cg_m, items, ["No positive mass items were found."])
    x = sum(item.mass_kg * item.cg_m[0] for item in items) / total
    y = sum(item.mass_kg * item.cg_m[1] for item in items) / total
    z = sum(item.mass_kg * item.cg_m[2] for item in items) / total
    if not project.mass_items and not any(body.mass_properties for body in project.bodies):
        warnings.append("Mass balance only includes inferred items; add component masses for CG sensitivity.")
    return MassBalance(total_mass_kg=total, cg_m=(x, y, z), items=items, warnings=warnings)


def apply_mass_balance_to_reference(project: AircraftProject) -> MassBalance:
    balance = compute_mass_balance(project)
    if balance.total_mass_kg > 0.0:
        project.reference.cg_m = balance.cg_m
        project.reference.moment_reference_m = balance.cg_m
    return balance

