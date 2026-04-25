from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import aerosandbox.numpy as np
import aerosandbox as asb
import numpy as _np_std

from core.models.project import WingProject
from core.naca_generator.naca456 import generate_naca_airfoil


g = 9.80665


# ==============================================================================
# Geometry Correction Functions (for DXF accuracy)
# ==============================================================================

def compute_local_spar_sweep_angles(
    sections: List['SpanwiseSection'],
    spar_type: str,  # "front" or "rear"
    planform,
) -> List[float]:
    """
    Compute local spar sweep angle at each rib station.
    
    The sweep angle is computed from the spar position delta between
    adjacent sections, giving the true geometric sweep at each station.
    
    On a swept wing, spars pass through ribs at an oblique angle. The effective
    horizontal thickness of the spar as seen by the rib is wider than the material
    thickness: t_effective = t_material / cos(Λ_local)
    
    The spar sweep angle varies along the span due to:
    - Leading edge sweep (sweep_le_deg)
    - Taper (chord shrinks toward tip)
    - Varying spar chord-% positions (front spar may have different root/tip positions)
    
    Args:
        sections: List of SpanwiseSection from spanwise_sections()
        spar_type: "front" or "rear"
        planform: PlanformGeometry with spar positions
    
    Returns:
        List of sweep angles in radians, one per section.
        First section uses forward difference, others use backward difference.
    """
    sweep_angles = []
    
    for i, section in enumerate(sections):
        # Get corrected spar chord-% at this section
        xsi = get_spar_xsi_at_section(section, planform, spar_type)
        
        # Spar X position at this section
        spar_x = section.x_le_m + xsi * section.chord_m
        spar_y = section.y_m
        
        # Get adjacent section for delta calculation
        if i == 0 and len(sections) > 1:
            # First section: use forward difference
            next_sec = sections[i + 1]
            next_xsi = get_spar_xsi_at_section(next_sec, planform, spar_type)
            
            next_spar_x = next_sec.x_le_m + next_xsi * next_sec.chord_m
            next_spar_y = next_sec.y_m
            
            dx = next_spar_x - spar_x
            dy = next_spar_y - spar_y
        else:
            # Other sections: use backward difference
            prev_sec = sections[i - 1]
            prev_xsi = get_spar_xsi_at_section(prev_sec, planform, spar_type)
            
            prev_spar_x = prev_sec.x_le_m + prev_xsi * prev_sec.chord_m
            prev_spar_y = prev_sec.y_m
            
            dx = spar_x - prev_spar_x
            dy = spar_y - prev_spar_y
            
        sweep_angles.append(math.atan2(dx, dy))
        
    return sweep_angles


def get_spar_xsi_at_section(
    section: 'SpanwiseSection',
    planform,
    spar_type: str,  # "front" or "rear"
) -> float:
    """
    Get the chordwise percentage (xsi) of a spar at a given section,
    accounting for piece-wise linear interpolation in BWB configurations.
    
    This ensures that spars follow straight lines in 3D space even when
    the planform (chord/sweep) is non-linear in blended regions.
    """
    eta = section.span_fraction
    
    def get_raw_xsi(eta_val: float) -> float:
        if spar_type == "front":
            return (planform.front_spar_root_percent * (1 - eta_val) + 
                    planform.front_spar_tip_percent * eta_val) / 100.0
        else:
            root_xsi = planform.rear_spar_root_percent / 100.0
            tip_xsi = planform.rear_spar_tip_percent / 100.0
            return root_xsi * (1 - eta_val) + tip_xsi * eta_val

    is_bwb = len(planform.body_sections) > 0
    if not is_bwb:
        return get_raw_xsi(eta)
        
    # BWB mode: piece-wise linear in absolute X
    wing_half_span = planform.half_span()
    
    # Boundary points
    last_body_section = max(planform.body_sections, key=lambda bs: bs.y_pos)
    y_junc = last_body_section.y_pos
    total_half_span = y_junc + wing_half_span
    
    if total_half_span <= 0:
        return get_raw_xsi(eta)
        
    y_abs = abs(section.y_m)
    
    # Boundary 1: Root
    first_body_section = min(planform.body_sections, key=lambda bs: bs.y_pos)
    y_root = first_body_section.y_pos
    chord_root = first_body_section.chord
    x_le_root = first_body_section.x_offset
    xsi_root = get_raw_xsi(y_root / total_half_span)
    x_root = x_le_root + xsi_root * chord_root
    
    # Boundary 2: Junction
    chord_junc = last_body_section.chord
    x_le_junc = last_body_section.x_offset
    xsi_junc = get_raw_xsi(y_junc / total_half_span)
    x_junc = x_le_junc + xsi_junc * chord_junc
    
    # Boundary 3: Tip
    y_tip = total_half_span
    chord_tip = planform.tip_chord()
    x_le_tip = x_le_junc + wing_half_span * math.tan(math.radians(planform.sweep_le_deg))
    xsi_tip = get_raw_xsi(1.0)
    x_tip = x_le_tip + xsi_tip * chord_tip
    
    # Interpolate absolute X
    if y_abs <= y_junc + 0.001:
        # BWB section: straight spanwise line (constant X)
        # The spar ignores intermediate body section chords/offsets and stays at junction X
        x_abs = x_junc
    else:
        t = (y_abs - y_junc) / max(0.001, y_tip - y_junc)
        x_abs = x_junc * (1 - t) + x_tip * t
        
    # Convert back to local xsi
    if section.chord_m > 1e-6:
        return float((x_abs - section.x_le_m) / section.chord_m)
    else:
        return get_raw_xsi(eta)


@dataclass
class SpanwiseSection:
    index: int
    span_fraction: float
    y_m: float
    z_m: float
    chord_m: float
    twist_deg: float
    x_le_m: float
    airfoil_uid: str
    airfoil: Any


@dataclass
class SpanwiseDistribution:
    section: SpanwiseSection
    cl_design: float
    cl_min_speed: float
    cl_max_speed: float
    reynolds_min: float
    reynolds_trim: float
    reynolds_max: float
    gamma_trim: float
    alpha_zero_lift_deg: float
    cl_alpha_per_deg: float
    total_twist_deg: float
    geometric_twist_deg: float


class AeroSandboxService:
    """Convenience wrapper for building aerosandbox geometries and running analyses."""

    def __init__(self, project: Union[WingProject, 'Project']) -> None:
        # Support both WingProject (legacy) and Project (unified)
        if hasattr(project, 'wing'):
            # It's a unified Project
            self.wing_project = project.wing
        else:
            # It's a WingProject directly
            self.wing_project = project
        self._bwb_airfoil: Optional[Any] = None
        self._root_airfoil: Optional[Any] = None
        self._tip_airfoil: Optional[Any] = None

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            if len(value) == 1:
                try:
                    return float(value[0])
                except (ValueError, TypeError):
                    return value[0]
            return value
        if isinstance(value, (complex,)):
            return float(value.real)
        if isinstance(value, (int, float)):
            return float(value)
        return value

    @property
    def atmosphere(self) -> asb.Atmosphere:
        return asb.Atmosphere(altitude=self.wing_project.twist_trim.cruise_altitude_m)

    def _get_airfoil(self, name: str) -> Any:
        """
        Robustly load an airfoil.
        1. Try new NACA generator (supports 4, 5, 6, 7-series).
        2. Try standard AeroSandbox load.
        """
        # Try custom generation first
        try:
            # Strip 'naca' prefix if present
            clean_name = name.lower().replace("naca", "").strip()
            x, y = generate_naca_airfoil(clean_name, n_points=200)
            coords = np.column_stack((x, y))
            return asb.Airfoil(name, coordinates=coords)
        except Exception:
            # Not a supported NACA designation or generation failed
            pass

        try:
            af = asb.Airfoil(name)
        except Exception:
            af = asb.Airfoil(name) # Retry or just proceed to check coordinates
            
        if af.coordinates is None:
            # Fallback if generation failed but it was supposed to be handled,
            # or if it's another type that failed loading.
            pass
            
        return af

    def _get_bwb_airfoil(self) -> Any:
        if self._bwb_airfoil is None:
            self._bwb_airfoil = self._get_airfoil(self.wing_project.airfoil.bwb_airfoil)
        return self._bwb_airfoil

    def _get_root_airfoil(self) -> Any:
        if self._root_airfoil is None:
            self._root_airfoil = self._get_airfoil(self.wing_project.airfoil.root_airfoil)
        return self._root_airfoil

    def _get_tip_airfoil(self) -> Any:
        if self._tip_airfoil is None:
            self._tip_airfoil = self._get_airfoil(self.wing_project.airfoil.tip_airfoil)
        return self._tip_airfoil

    def _deduplicate_sections(
        self,
        sections: List[SpanwiseSection], 
        tol_m: float = 1e-4
    ) -> List[SpanwiseSection]:
        """
        Remove sections with near-identical y positions to prevent VLM singularities.
        
        When two sections are coincident (within tolerance), preference order:
        1. BWB sections (lower index, built first)
        2. Section with larger chord (if within wing portion)
        
        Args:
            sections: List of SpanwiseSection
            tol_m: Tolerance in meters (default 0.1mm)
        
        Returns:
            Deduplicated list of sections with re-indexed values
        """
        if len(sections) < 2:
            return sections
        
        # Sort by y position
        sorted_secs = sorted(sections, key=lambda s: s.y_m)
        result = [sorted_secs[0]]
        
        for sec in sorted_secs[1:]:
            if abs(sec.y_m - result[-1].y_m) < tol_m:
                # Coincident sections detected
                existing = result[-1]
                
                # Determine which to keep:
                # - Lower index = BWB section (built first) - prefer these
                # - Otherwise prefer larger chord
                if existing.index <= sec.index:
                    # Keep existing (BWB or earlier)
                    print(f"[Geometry] Removed coincident section at y={sec.y_m:.4f}m "
                          f"(kept section {existing.index}, removed {sec.index})")
                else:
                    # New section is BWB (lower index), replace
                    print(f"[Geometry] Removed coincident section at y={existing.y_m:.4f}m "
                          f"(kept section {sec.index}, removed {existing.index})")
                    result[-1] = sec
            else:
                result.append(sec)
        
        # Re-index if any were removed
        if len(result) < len(sections):
            for i, sec in enumerate(result):
                sec.index = i
        
        return result

    def spanwise_sections(self, twist_override: Optional[List[float]] = None) -> List[SpanwiseSection]:
        plan = self.wing_project.planform
        twist = self.wing_project.twist_trim
        n = max(2, self.wing_project.airfoil.num_sections)
        root_airfoil = self._get_root_airfoil()
        tip_airfoil = self._get_tip_airfoil()

        sections: List[SpanwiseSection] = []
        
        wing_half_span = plan.half_span()
        taper = plan.taper_ratio
        sweep_rad = np.radians(plan.sweep_le_deg)
        dihedral_rad = np.radians(plan.dihedral_deg)
        bwb_dihedral_rad = np.radians(plan.bwb_dihedral_deg)  # Separate dihedral for BWB
        root_twist = -twist.zero_lift_aoa_root_deg
        tip_twist = -twist.zero_lift_aoa_tip_deg
        
        # Check if BWB mode is enabled (body_sections defined)
        bwb_mode = len(plan.body_sections) > 0
        
        if bwb_mode:
            # Sort body sections by y_pos
            sorted_body_sections = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else 0.0
            bwb_outer_chord = sorted_body_sections[-1].chord if sorted_body_sections else plan.root_chord()
            bwb_outer_x_offset = sorted_body_sections[-1].x_offset if sorted_body_sections else 0.0
            bwb_outer_z_offset = sorted_body_sections[-1].z_offset if sorted_body_sections else 0.0
            
            # Calculate actual z-height at BWB outer edge (including BWB dihedral effect)
            bwb_outer_z = bwb_outer_z_offset + np.tan(bwb_dihedral_rad) * bwb_outer_y
            
            # TOTAL half-span = BWB body + Wing (additive)
            total_half_span = bwb_outer_y + wing_half_span
            
            # BWB sections: use exact number of body sections defined by user
            n_bwb = len(sorted_body_sections)
            
            # Wing sections: use num_sections directly (independent of BWB)
            n_wing = n
        else:
            n_bwb = 0
            n_wing = n
            bwb_outer_y = 0.0
            bwb_outer_chord = plan.root_chord()
            bwb_outer_x_offset = 0.0
            bwb_outer_z_offset = 0.0
            bwb_outer_z = 0.0  # No BWB z contribution
            total_half_span = wing_half_span

        if bwb_mode:
            # We skip the first wing section as it duplicates BWB end
            n_wing_effective = n_wing - 1
        else:
            n_wing_effective = n_wing
            
        # Total section count for validating optimized_twist array
        total_sections = n_bwb + n_wing_effective
        opt_twist = self.wing_project.optimized_twist_deg
        
        use_opt_twist = opt_twist is not None and len(opt_twist) == total_sections
        
        idx = 0
        
        # --- BWB Body Sections (center insert, additive to span) ---
        if bwb_mode and n_bwb >= 1:
            for i, body_sec in enumerate(sorted_body_sections):
                y = body_sec.y_pos
                
                # Use exact section properties from body section definition
                bwb_airfoil = self._get_bwb_airfoil()
                chord = body_sec.chord
                x_le = body_sec.x_offset
                z_offset = body_sec.z_offset
                z = z_offset + np.tan(bwb_dihedral_rad) * y  # Use BWB-specific dihedral
                
                span_fraction = y / total_half_span if total_half_span > 0 else 0.0
                
                if twist_override is not None:
                    twist_deg = twist_override[idx] if idx < len(twist_override) else 0.0
                elif use_opt_twist and idx < len(opt_twist):
                    twist_deg = opt_twist[idx]
                else:
                    twist_deg = root_twist + (tip_twist - root_twist) * span_fraction
                
                sections.append(SpanwiseSection(
                    index=idx,
                    span_fraction=span_fraction,
                    y_m=y,
                    z_m=z,
                    chord_m=chord,
                    twist_deg=twist_deg,
                    x_le_m=x_le,
                    airfoil_uid=f"airfoil_sec{idx + 1}",
                    airfoil=bwb_airfoil,
                ))
                idx += 1
        
        # --- Wing Sections (attached at BWB outer edge, full wing span) ---
        # No epsilon needed; we will skip the first wing section if it duplicates the BWB end
        
        for i in range(n_wing):
            # Wing local fraction (0=wing root at BWB junction, 1=wing tip)
            wing_local_frac = i / (n_wing - 1) if n_wing > 1 else 0.0
            
            # Y position: BWB outer + wing span fraction
            y = bwb_outer_y + wing_half_span * wing_local_frac

            # If BWB mode is active, the first wing section (i=0) is at bwb_outer_y.
            # This duplicates the last BWB section which creates a zero-length segment or overlapping panels.
            # We skip it to ensure smooth transition from BWB end to the next wing station.
            if bwb_mode and i == 0:
                continue
            
            # Span fraction relative to total span
            span_fraction = y / total_half_span if total_half_span > 0 else 0.0
            
            # Chord calculation
            if bwb_mode:
                # Independent wing geometry with blend region
                wing_root = plan.root_chord()  # Wing's own root chord
                wing_tip = plan.tip_chord()
                
                # Blend region: interpolate from BWB outer chord to wing root chord
                blend_span = wing_half_span * (plan.bwb_blend_span_percent / 100.0)
                wing_y_from_junction = y - bwb_outer_y  # Position along wing from junction
                
                if blend_span > 0 and wing_y_from_junction < blend_span:
                    # Within blend region
                    blend_frac = wing_y_from_junction / blend_span
                    # Cosine blend for smooth transition
                    blend_weight = 0.5 * (1 - np.cos(np.pi * blend_frac))
                    effective_root = bwb_outer_chord + (wing_root - bwb_outer_chord) * blend_weight
                else:
                    # Beyond blend region - use pure wing geometry
                    effective_root = wing_root
                
                # Calculate wing local fraction for taper (0=junction, 1=tip)
                chord = effective_root + (wing_tip - effective_root) * wing_local_frac
                
                # X leading edge: continue sweep from BWB junction
                x_le = bwb_outer_x_offset + np.tan(sweep_rad) * (y - bwb_outer_y)
            else:
                chord = plan.root_chord() * (1 - (1 - taper) * span_fraction)
                x_le = np.tan(sweep_rad) * y
            
            # Apply M-shaped trailing edge extension (only for non-BWB)
            if not bwb_mode:
                center_span_pct = plan.center_section_span_percent
                center_ext_pct = plan.center_chord_extension_percent
                
                if center_span_pct > 0 and center_ext_pct > 0:
                    limit_frac = center_span_pct / 100.0
                    if span_fraction < limit_frac:
                        if plan.center_extension_linear:
                            blend = 1.0 - (span_fraction / limit_frac)
                        else:
                            blend = 0.5 * (1 + np.cos(np.pi * span_fraction / limit_frac))
                        
                        extension = plan.root_chord() * (center_ext_pct / 100.0) * blend
                        chord += extension
            
            # Z height: start from BWB junction z and apply wing dihedral only to wing portion
            if bwb_mode:
                z = bwb_outer_z + np.tan(dihedral_rad) * (y - bwb_outer_y)
            else:
                z = np.tan(dihedral_rad) * y
            
            # Airfoil blending across full span
            if idx == 0:
                section_airfoil = root_airfoil
            elif span_fraction >= 0.99:
                section_airfoil = tip_airfoil
            else:
                section_airfoil = root_airfoil.blend_with_another_airfoil(
                    airfoil=tip_airfoil,
                    blend_fraction=span_fraction,
                )

            if twist_override is not None:
                twist_deg = twist_override[idx] if idx < len(twist_override) else 0.0
            elif use_opt_twist and idx < len(opt_twist):
                twist_deg = opt_twist[idx]
            else:
                twist_deg = root_twist + (tip_twist - root_twist) * span_fraction
                
            sections.append(SpanwiseSection(
                index=idx,
                span_fraction=span_fraction,
                y_m=y,
                z_m=z,
                chord_m=chord,
                twist_deg=twist_deg,
                x_le_m=x_le,
                airfoil_uid=f"airfoil_sec{idx + 1}",
                airfoil=section_airfoil,
            ))
            idx += 1

        # Deduplicate sections to prevent VLM singularities from coincident panels
        sections = self._deduplicate_sections(sections, tol_m=1e-4)
        
        return sections
    
    def _interpolate_bwb_section(
        self, 
        sorted_body_sections: List, 
        y: float, 
        bwb_airfoil
    ) -> Tuple[float, float, float, Any]:
        """
        Interpolates BWB body section geometry at spanwise position y.
        Returns: (chord, x_le, z_offset, bwb_airfoil)
        """
        # Find bounding sections
        lower_sec = sorted_body_sections[0]
        upper_sec = sorted_body_sections[-1]
        
        for i in range(len(sorted_body_sections) - 1):
            if sorted_body_sections[i].y_pos <= y <= sorted_body_sections[i + 1].y_pos:
                lower_sec = sorted_body_sections[i]
                upper_sec = sorted_body_sections[i + 1]
                break
        
        # Calculate interpolation factor
        if upper_sec.y_pos == lower_sec.y_pos:
            t = 0.0
        else:
            t = (y - lower_sec.y_pos) / (upper_sec.y_pos - lower_sec.y_pos)
        
        # Linear interpolation of geometry
        chord = lower_sec.chord + t * (upper_sec.chord - lower_sec.chord)
        x_le = lower_sec.x_offset + t * (upper_sec.x_offset - lower_sec.x_offset)
        z_offset = lower_sec.z_offset + t * (upper_sec.z_offset - lower_sec.z_offset)
        
        # Use dedicated BWB airfoil (same across entire BWB body)
        return chord, x_le, z_offset, bwb_airfoil

    def velocity_for_cl(self, cl: float) -> float:
        cl = max(float(cl), 1e-4)
        atmo = self.atmosphere
        air_density = atmo.density()
        weight_n = self.wing_project.twist_trim.gross_takeoff_weight_kg * g
        # Use actual_area() to include BWB sections in area calculation
        wing_area = self.wing_project.planform.actual_area()
        velocity_sq = 2 * weight_n / max(air_density * wing_area * cl, 1e-6)
        return math.sqrt(velocity_sq)

    def design_velocity(self, cl_target: Optional[float] = None) -> float:
        cl = cl_target if cl_target is not None else self.wing_project.twist_trim.design_cl
        return self.velocity_for_cl(cl)

    def estimate_alpha_for_cl(self, cl_target: Optional[float] = None) -> float:
        params = self.wing_project.twist_trim
        cl = cl_target if cl_target is not None else params.design_cl
        cl_alpha = 0.5 * (params.cl_alpha_root_per_deg + params.cl_alpha_tip_per_deg)
        alpha_l0 = 0.5 * (params.zero_lift_aoa_root_deg + params.zero_lift_aoa_tip_deg)
        return alpha_l0 + cl / max(cl_alpha, 1e-6)

    def spanwise_distribution(self) -> List[SpanwiseDistribution]:
        sections = self.spanwise_sections()
        params = self.wing_project.twist_trim
        plan = self.wing_project.planform
        atmo = self.atmosphere
        rho = atmo.density()
        mu = atmo.dynamic_viscosity()

        v_trim = self.design_velocity()
        v_min = self.velocity_for_cl(params.estimated_cl_max)
        v_max = self.velocity_for_cl(params.estimated_cl_max_speed)

        if params.lift_distribution == "elliptical":
            shape = [math.sqrt(max(0.0, 1 - s.span_fraction ** 2)) for s in sections]
        else:
            # Bell distribution (Prandtl-D) uses power 1.5
            shape = [max(0.0, 1 - s.span_fraction ** 2) ** 1.5 for s in sections]

        integral_shape = 0.0
        for i in range(len(sections) - 1):
            dy = sections[i + 1].y_m - sections[i].y_m
            integral_shape += 0.5 * dy * (
                shape[i] * sections[i].chord_m + shape[i + 1] * sections[i + 1].chord_m
            )
        if integral_shape <= 0:
            scale = 0.0
        else:
            # Use actual_area() to include BWB sections
            scale = params.design_cl * plan.actual_area() / (2.0 * integral_shape)

        cl_factor_min = params.estimated_cl_max / max(params.design_cl, 1e-6)
        cl_factor_max = params.estimated_cl_max_speed / max(params.design_cl, 1e-6)
        alpha_design = self.estimate_alpha_for_cl()

        distributions: List[SpanwiseDistribution] = []
        geom_root = sections[0].twist_deg
        for idx, section in enumerate(sections):
            base_shape = shape[idx]
            cl_design = scale * base_shape
            cl_min = cl_design * cl_factor_min
            cl_max = cl_design * cl_factor_max

            reynolds_trim = rho * v_trim * section.chord_m / max(mu, 1e-9)
            reynolds_min = rho * v_min * section.chord_m / max(mu, 1e-9)
            reynolds_max = rho * v_max * section.chord_m / max(mu, 1e-9)

            cl_alpha_section = params.cl_alpha_root_per_deg + (
                params.cl_alpha_tip_per_deg - params.cl_alpha_root_per_deg
            ) * section.span_fraction
            alpha_l0_section = params.zero_lift_aoa_root_deg + (
                params.zero_lift_aoa_tip_deg - params.zero_lift_aoa_root_deg
            ) * section.span_fraction
            
            # Calculate required twist for this lift distribution
            # Cl = Cl_alpha * (alpha_geom + twist - alpha_l0)
            # twist = Cl / Cl_alpha - alpha_geom + alpha_l0
            # Here alpha_design is the root angle of attack relative to flight path?
            # Actually: Cl = Cl_alpha * (alpha_local - alpha_l0)
            # alpha_local = alpha_root + twist
            # So: twist = Cl / Cl_alpha + alpha_l0 - alpha_root
            
            required_twist_absolute = cl_design / max(cl_alpha_section, 1e-6) + alpha_l0_section
            # We want twist relative to root, but alpha_design is the trim alpha.
            # Let's just calculate the total twist required to achieve cl_design at alpha_design
            
            # Total twist (Effective Incidence) = (alpha_design + geometric_twist) - alpha_l0
            # This represents the angle of attack relative to the zero-lift line.
            total_twist = (alpha_design + section.twist_deg) - alpha_l0_section
            
            gamma_trim = 0.5 * v_trim * section.chord_m * cl_design

            distributions.append(
                SpanwiseDistribution(
                    section=section,
                    cl_design=cl_design,
                    cl_min_speed=cl_min,
                    cl_max_speed=cl_max,
                    reynolds_min=reynolds_min,
                    reynolds_trim=reynolds_trim,
                    reynolds_max=reynolds_max,
                    gamma_trim=gamma_trim,
                    alpha_zero_lift_deg=alpha_l0_section,
                    cl_alpha_per_deg=cl_alpha_section,
                    total_twist_deg=total_twist, # Kept as is for now, might need review
                    geometric_twist_deg=section.twist_deg - geom_root,
                )
            )

        return distributions

    def calculate_optimized_twist(self) -> List[float]:
        """
        Calculates the geometric twist distribution required to achieve the target lift distribution.
        Ensures both CL = design_cl AND Cm = 0 at alpha = 0.
        Returns a list of twist angles in degrees (relative to root).
        """
        import scipy.optimize as opt
        
        # 1. Get target Cl distribution
        # Temporarily disable optimized twist to avoid circular dependency if we were using it
        original_opt_twist = self.wing_project.optimized_twist_deg
        self.wing_project.optimized_twist_deg = None 
        
        try:
            # Get spanwise distribution for target Cl shape
            dists = self.spanwise_distribution()
            
            # 2. Calculate initial twist shape (geometric approach)
            # This gives us the SHAPE of the twist distribution
            root_dist = dists[0]
            alpha_root_req = (root_dist.cl_design / root_dist.cl_alpha_per_deg) + root_dist.alpha_zero_lift_deg
            
            base_twist = []
            for d in dists:
                req_twist = (d.cl_design / d.cl_alpha_per_deg) + d.alpha_zero_lift_deg - alpha_root_req
                base_twist.append(req_twist)
            
            # 3. Optimization: Find scale and offset that satisfy CL=design_cl AND Cm=0
            # Parameters: [scale, offset]
            # scale: Multiplies the twist shape (affects CL distribution)
            # offset: Added to all twists (affects overall incidence)
            
            cl_target = self.wing_project.twist_trim.design_cl
            
            def get_cl_cm(params):
                """Returns (CL, Cm) for given scale and offset."""
                scale, offset = params
                trial_twist = [t * scale + offset for t in base_twist]
                
                wing = self.build_wing(twist_override=trial_twist)
                
                x_np = wing.aerodynamic_center()[0]
                mac = wing.mean_aerodynamic_chord()
                static_margin = self.wing_project.twist_trim.static_margin_percent
                x_cg = x_np - (static_margin / 100.0) * mac
                xyz_ref = [x_cg, 0.0, 0.0]
                
                airplane = asb.Airplane(
                    name=self.wing_project.name,
                    wings=[wing],
                    xyz_ref=xyz_ref,
                )
                
                op_point = asb.OperatingPoint(
                    atmosphere=self.atmosphere,
                    velocity=self.design_velocity(),
                    alpha=0.0,
                )
                
                analysis = asb.AeroBuildup(
                    airplane=airplane,
                    op_point=op_point,
                )
                res = analysis.run()
                
                cl = res.get("CL", res.get("Cl", 0.0))
                cm = res.get("CM", res.get("Cm", 0.0))
                if hasattr(cl, "item"): cl = cl.item()
                if hasattr(cm, "item"): cm = cm.item()
                
                return cl, cm
            
            def residuals(params):
                """Returns [CL - target, Cm] for least squares optimization."""
                cl, cm = get_cl_cm(params)
                return [cl - cl_target, cm]
            
            print("Optimizing twist for CL and Cm constraints...")
            print(f"  Target CL: {cl_target:.4f}, Target Cm: 0.0")
            
            # Initial guess: scale=1, offset=0
            x0 = [1.0, 0.0]
            
            try:
                # Use least squares to find scale and offset
                result = opt.least_squares(
                    residuals, 
                    x0, 
                    bounds=([-5.0, -20.0], [5.0, 20.0]),
                    ftol=1e-6,
                    xtol=1e-6,
                    verbose=1
                )
                
                scale_sol, offset_sol = result.x
                print(f"  Converged! Scale: {scale_sol:.4f}, Offset: {offset_sol:.4f} deg")
                
                # Verify solution
                cl_final, cm_final = get_cl_cm(result.x)
                print(f"  Final CL: {cl_final:.4f} (target: {cl_target:.4f})")
                print(f"  Final Cm: {cm_final:.6f} (target: 0.0)")
                
                optimized_twist = [t * scale_sol + offset_sol for t in base_twist]
                
            except Exception as e:
                print(f"Warning: Optimization failed: {e}")
                print("Falling back to Cm=0 only constraint...")
                
                # Fallback: Just optimize for Cm=0 (original behavior)
                def get_cm_only(delta):
                    _, cm = get_cl_cm([1.0, delta])
                    return cm
                
                try:
                    delta_sol = opt.newton(get_cm_only, 0.0, tol=1e-4)
                    optimized_twist = [t + delta_sol for t in base_twist]
                    print(f"  Cm=0 fallback converged with offset: {delta_sol:.4f} deg")
                except:
                    optimized_twist = base_twist
                    print("  Warning: Fallback also failed, using base twist")
            
            return optimized_twist
            
        finally:
            self.wing_project.optimized_twist_deg = original_opt_twist

    def optimize_twist_for_trim(self) -> Tuple[List[float], float]:
        """
        Optimizes twist distribution to achieve Cm=0 and CL=design at alpha=0 (level flight).
        Uses AeroSandbox Opti stack.
        
        Returns:
            Tuple of (optimized_twist_deg, alpha) where alpha is always 0.0
        """
        opti = asb.Opti()
        
        n = max(2, self.wing_project.airfoil.num_sections)
        
        # Variables: Twist at each section
        # Initialize with current twist or linear guess
        current_twist = self.wing_project.optimized_twist_deg
        if current_twist is None or len(current_twist) != n:
            root = -self.wing_project.twist_trim.zero_lift_aoa_root_deg
            tip = -self.wing_project.twist_trim.zero_lift_aoa_tip_deg
            current_twist = np.linspace(root, tip, n)
            
        twist = opti.variable(init_guess=current_twist)
        
        # Build Wing with variable twist
        # We need to convert the CasADi variable to a list/array that spanwise_sections can handle
        # spanwise_sections iterates and indexes, which CasADi supports.
        wing = self.build_wing(twist_override=twist)
        
        # Build Airplane
        # Calculate CG (fixed during optimization)
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        
        # Define Operating Point
        # User wants trim at alpha=0 (level flight)
        # So we fix alpha=0 and optimize twist to achieve CL and Cm targets
        alpha_cruise = 0.0  # Level flight
        
        op_point = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=self.design_velocity(),
            alpha=alpha_cruise,
        )
        
        # Analysis
        # Use AeroBuildup for speed and differentiability
        analysis = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point,
        )
        res = analysis.run()
        
        # Constraints
        # 1. CL = CL_design
        cl_res = res.get("CL", res.get("Cl"))
        cl_target = self.wing_project.twist_trim.design_cl
        opti.subject_to(cl_res == cl_target)
        
        # 2. Cm = 0 (Trim)
        cm_res = res.get("CM", res.get("Cm"))
        opti.subject_to(cm_res == 0.0)
        
        # Objective
        # Minimize induced drag (CDi) or just CD
        cd_res = res.get("CD", res.get("Cd"))
        # Also add regularization to keep twist smooth
        
        # Smoothness: Minimize 2nd derivative of twist (curvature)
        # Finite difference approximation
        # Smoothness: Minimize 2nd derivative of twist (curvature)
        # Finite difference approximation
        if n > 2:
            d2_twist = twist[:-2] - 2*twist[1:-1] + twist[2:]
            opti.minimize(cd_res + 0.1 * np.sum(d2_twist**2))
        else:
            opti.minimize(cd_res)
            
        # Solve
        try:
            sol = opti.solve(verbose=True)
            optimized_twist = sol.value(twist)
            
            # Return as list of floats
            if hasattr(optimized_twist, "tolist"):
                twist_list = optimized_twist.tolist()
            else:
                twist_list = list(optimized_twist)
                
            # Alpha is fixed at 0, so we return 0
            return twist_list, 0.0
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Return current guess if failed, or raise
            # For now, return the debug value if available, else raise
            try:
                twist_list = opti.debug.value(twist).tolist()
                return twist_list, 0.0
            except:
                raise e

    def build_wing(
        self, 
        twist_override: Optional[List[float]] = None,
        control_deflections: Optional[Dict[str, float]] = None,
    ) -> asb.Wing:
        """
        Build AeroSandbox Wing with optional control surfaces.
        
        Args:
            twist_override: Override twist distribution
            control_deflections: Dict mapping surface names to deflections [deg]
                                e.g., {'Elevon': 5.0, 'Aileron': -3.0}
                                Positive deflection = trailing edge down
        
        Returns:
            asb.Wing with control surfaces attached to appropriate sections
        """
        planform = self.wing_project.planform
        control_surfaces = planform.control_surfaces
        total_half_span = planform.half_span()
        snap = planform.snap_to_sections
        
        xsecs = []
        sections = self.spanwise_sections(twist_override=twist_override)
        
        for section in sections:
            # Compute eta (normalized span position, 0=root, 1=tip)
            eta = section.y_m / total_half_span if total_half_span > 0 else 0.0
            eta_percent = eta * 100.0  # Convert to percent for comparison
            
            # Find control surfaces that apply to this spanwise station
            asb_control_surfaces = []
            for cs in control_surfaces:
                # Determine if this section is within the control surface span
                if snap:
                    # Snapping logic: find sections closest to the requested start/end %
                    # and include all sections between those snapped indices.
                    cs_start_y = total_half_span * (cs.span_start_percent / 100.0)
                    cs_end_y = total_half_span * (cs.span_end_percent / 100.0)
                    
                    # Find closest section y to requested start/end
                    y_positions = [s.y_m for s in sections]
                    idx_start = np.argmin(np.abs(np.array(y_positions) - cs_start_y))
                    idx_end = np.argmin(np.abs(np.array(y_positions) - cs_end_y))
                    
                    # Ensure indices are ordered
                    idx_min = min(idx_start, idx_end)
                    idx_max = max(idx_start, idx_end)
                    
                    # Section is in surface if its index is within snapped range
                    in_surface = (idx_min <= section.index <= idx_max)
                    
                    # Adjust effective cs_start/end for interpolation of hinge line
                    # Using the actual snapped section Y positions
                    eff_start_y = sections[idx_min].y_m
                    eff_end_y = sections[idx_max].y_m
                    eff_eta_percent = section.y_m / total_half_span * 100.0 if total_half_span > 0 else 0.0
                    
                    # Local interpolation parameters for hinge line
                    span_range = eff_end_y - eff_start_y
                    if span_range > 1e-6:
                        t = (section.y_m - eff_start_y) / span_range
                    else:
                        t = 0.0
                else:
                    # Standard logic: literal inclusion based on %
                    in_surface = (cs.span_start_percent <= eta_percent <= cs.span_end_percent)
                    span_range_pct = cs.span_end_percent - cs.span_start_percent
                    if span_range_pct > 0:
                        t = (eta_percent - cs.span_start_percent) / span_range_pct
                    else:
                        t = 0.0

                if in_surface:
                    hinge_chord_percent = (
                        cs.chord_start_percent + t * (cs.chord_end_percent - cs.chord_start_percent)
                    )
                    hinge_frac = hinge_chord_percent / 100.0  # Convert to fraction
                    
                    # Get deflection if provided
                    deflection = 0.0
                    if control_deflections and cs.name in control_deflections:
                        deflection = control_deflections[cs.name]
                    
                    # Determine surface behavior based on type
                    surface_type_lower = cs.surface_type.lower()
                    
                    if surface_type_lower == 'elevon':
                        # Elevons need BOTH symmetric (pitch) and anti-symmetric (roll) behavior
                        # Create two control surface objects at the same hinge point:
                        # - *_pitch: symmetric=True, responds to elevator input
                        # - *_roll:  symmetric=False, responds to aileron input
                        # AeroSandbox will sum the deflections from both
                        
                        pitch_deflection = 0.0
                        roll_deflection = 0.0
                        if control_deflections:
                            # Look for both combined and split naming conventions
                            pitch_deflection = control_deflections.get(f'{cs.name}_pitch', 
                                               control_deflections.get(cs.name, 0.0))
                            roll_deflection = control_deflections.get(f'{cs.name}_roll', 0.0)
                        
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=f'{cs.name}_pitch',
                                symmetric=True,  # Both wings deflect same direction
                                hinge_point=hinge_frac,
                                deflection=pitch_deflection,
                                trailing_edge=True,
                            )
                        )
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=f'{cs.name}_roll',
                                symmetric=False,  # Right = +deflection, Left = -deflection
                                hinge_point=hinge_frac,
                                deflection=roll_deflection,
                                trailing_edge=True,
                            )
                        )
                    elif surface_type_lower in ['flap', 'elevator']:
                        # Pure symmetric surfaces (both sides move same direction)
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=cs.name,
                                symmetric=True,
                                hinge_point=hinge_frac,
                                deflection=deflection,
                                trailing_edge=True,
                            )
                        )
                    elif surface_type_lower == 'aileron':
                        # Pure anti-symmetric surfaces (opposite directions for roll)
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=cs.name,
                                symmetric=False,
                                hinge_point=hinge_frac,
                                deflection=deflection,
                                trailing_edge=True,
                            )
                        )
                    elif surface_type_lower == 'rudder':
                        # Rudder on vertical surface (not typically on flying wing main wing)
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=cs.name,
                                symmetric=True,  # Single surface
                                hinge_point=hinge_frac,
                                deflection=deflection,
                                trailing_edge=True,
                            )
                        )
                    else:
                        # Unknown type - default to symmetric
                        asb_control_surfaces.append(
                            asb.ControlSurface(
                                name=cs.name,
                                symmetric=True,
                                hinge_point=hinge_frac,
                                deflection=deflection,
                                trailing_edge=True,
                            )
                        )
            
            xsecs.append(
                asb.WingXSec(
                    xyz_le=[section.x_le_m, section.y_m, section.z_m],
                    chord=section.chord_m,
                    twist=section.twist_deg,
                    airfoil=section.airfoil,
                    control_surfaces=asb_control_surfaces if asb_control_surfaces else None,
                )
            )
        return asb.Wing(
            name="Flying Wing",
            xsecs=xsecs,
            symmetric=True,
        )

    def build_airplane(
        self, 
        xyz_ref: Optional[List[float]] = None,
        control_deflections: Optional[Dict[str, float]] = None,
    ) -> asb.Airplane:
        """
        Build AeroSandbox Airplane with optional control surface deflections.
        
        Args:
            xyz_ref: Reference point for moments [x, y, z]
            control_deflections: Dict mapping surface names to deflections [deg]
        """
        return asb.Airplane(
            name=self.wing_project.name,
            wings=[self.build_wing(control_deflections=control_deflections)],
            xyz_ref=xyz_ref,
        )

    def run_aero_buildup(self, cl_target: Optional[float] = None) -> Dict[str, Any]:
        # 1. Build wing to get geometric properties
        wing = self.build_wing()
        
        # 2. Calculate CG based on Static Margin
        # X_cg = X_np - (StaticMargin/100) * MAC
        # For a flying wing, we can approximate X_np as the wing's aerodynamic center.
        # Note: ASB's aerodynamic_center() returns [x, y, z]
        
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        # 3. Build airplane with this reference point
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        
        velocity = self.design_velocity(cl_target)
        alpha = self.estimate_alpha_for_cl(cl_target)
        atmo = self.atmosphere
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=velocity,
            alpha=alpha,
        )
        analysis = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point,
        )
        results = analysis.run()
        
        # Normalize keys for consistency (ASB might return Cm, we want CM)
        if "Cm" in results and "CM" not in results:
            results["CM"] = results["Cm"]
        if "Cl" in results and "CL" not in results:
            results["CL"] = results["Cl"]
        if "Cd" in results and "CD" not in results:
            results["CD"] = results["Cd"]
            
        # Calculate Cma (Longitudinal Stability Derivative) via finite difference
        # Run a second point at alpha + 0.1 deg
        delta_alpha = 0.1
        op_point_2 = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=velocity,
            alpha=alpha + delta_alpha,
        )
        analysis_2 = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_2,
        )
        results_2 = analysis_2.run()
        
        cm1 = results.get("CM", results.get("Cm", 0.0))
        cm2 = results_2.get("CM", results_2.get("Cm", 0.0))
        
        # Handle potential list wrapping
        if isinstance(cm1, list): cm1 = cm1[0]
        if isinstance(cm2, list): cm2 = cm2[0]
            
        cma = (cm2 - cm1) / delta_alpha
        results["Cma"] = cma
        
        # Add calculated properties to results
        results["x_cg"] = x_cg
        results["x_np"] = x_np
        results["mac"] = mac
        
        return {key: self._coerce_value(value) for key, value in results.items()}

    def run_lifting_line(self, cl_target: Optional[float] = None, spanwise_resolution: Optional[int] = None) -> Dict[str, Any]:
        # 1. Build wing to get geometric properties
        wing = self.build_wing()
        
        # 2. Calculate CG based on Static Margin
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        # 3. Build airplane with this reference point
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        
        velocity = self.design_velocity(cl_target)
        alpha = self.estimate_alpha_for_cl(cl_target)
        atmo = self.atmosphere
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=velocity,
            alpha=alpha,
        )
        analysis = asb.LiftingLine(
            airplane=airplane,
            op_point=op_point,
            spanwise_resolution=spanwise_resolution or self.wing_project.airfoil.num_sections,
        )
        
        results = analysis.run()
        
        # Calculate Cma (Longitudinal Stability Derivative) via finite difference
        delta_alpha = 0.1
        op_point_2 = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=velocity,
            alpha=alpha + delta_alpha,
        )
        analysis_2 = asb.LiftingLine(
            airplane=airplane,
            op_point=op_point_2,
            spanwise_resolution=spanwise_resolution or self.wing_project.airfoil.num_sections,
        )
        results_2 = analysis_2.run()
        
        cm1 = results.get("CM", results.get("Cm", 0.0))
        cm2 = results_2.get("CM", results_2.get("Cm", 0.0))
        
        # Handle potential list wrapping
        if isinstance(cm1, list): cm1 = cm1[0]
        if isinstance(cm2, list): cm2 = cm2[0]
            
        cma = (cm2 - cm1) / delta_alpha
        results["Cma"] = cma
        
        # Add calculated properties to results
        results["x_cg"] = x_cg
        results["x_np"] = x_np
        results["mac"] = mac
        
        return {key: self._coerce_value(value) for key, value in results.items()}

    def neuralfoil_polar(
        self,
        airfoil_name: str,
        alpha_deg: float,
        reynolds: float,
        mach: float = 0.1,
    ) -> Dict[str, float]:
        # Use robust loader to handle NACA 5-digit airfoils
        airfoil = self._get_airfoil(airfoil_name)
        
        aero = airfoil.get_aero_from_neuralfoil(
            alpha=alpha_deg,
            Re=reynolds,
            mach=mach,
        )
        return {key: float(value) for key, value in aero.items()}

    def analyze_airfoil_parameters(self, airfoil_name: str) -> Dict[str, float]:
        """
        Estimates Cm0, Zero-lift AoA, and Cl_alpha for a given airfoil using NeuralFoil.
        """
        # Run a small sweep
        alphas = np.linspace(-5, 10, 10)
        reynolds = 1e6 # Typical
        
        cl_list = []
        cm_list = []
        
        for alpha in alphas:
            res = self.neuralfoil_polar(airfoil_name, alpha, reynolds)
            cl_list.append(res["CL"])
            cm_list.append(res["CM"])
            
        # Fit linear curve for Cl
        # Cl = Cl_alpha * (alpha - alpha_l0)
        # Cl = Cl_alpha * alpha - Cl_alpha * alpha_l0
        # y = mx + c
        
        coeffs = np.polyfit(alphas, cl_list, 1)
        cl_alpha = coeffs[0]
        intercept = coeffs[1]
        
        # alpha_l0 = -intercept / cl_alpha
        alpha_l0 = -intercept / cl_alpha
        
        # Cm0 is roughly constant, take average or value at alpha=0
        cm0 = np.mean(cm_list)
        
        return {
            "cm0": float(cm0),
            "zero_lift_aoa_deg": float(alpha_l0),
            "cl_alpha_per_deg": float(cl_alpha)
        }

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculates predicted performance metrics:
        - Cruise Speed, AoA, L/D (at actual CL for cruise alpha, ensuring L=W)
        - Takeoff Speed, AoA, L/D (at estimated Cl max)
        """
        metrics = {}
        
        # Build wing with current twist (optimized or not)
        wing = self.build_wing()
        
        # Calculate CG based on static margin
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        
        # 1. Cruise Condition
        # Determine cruise alpha
        if self.wing_project.optimized_twist_deg is not None:
            alpha_cruise = 0.0
        else:
            alpha_cruise = self.estimate_alpha_for_cl()
        
        # Run initial analysis at a reference velocity to get actual CL at cruise alpha
        # CL is independent of velocity for incompressible flow, so we use design velocity as reference
        v_ref = self.design_velocity()
        
        op_point_ref = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=v_ref,
            alpha=alpha_cruise,
        )
        
        analysis_ref = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_ref,
        )
        res_ref = analysis_ref.run()
        
        # Get actual CL at cruise alpha from AeroBuildup
        actual_cruise_cl = self._coerce_value(res_ref.get("CL", res_ref.get("Cl", 0.0)))
        
        # Calculate TRUE cruise velocity using actual CL (ensures L = W)
        if actual_cruise_cl > 1e-4:
            v_cruise = self.velocity_for_cl(actual_cruise_cl)
        else:
            # Fallback if CL is near zero (shouldn't happen for valid configs)
            v_cruise = v_ref
        
        # Re-run analysis at correct cruise velocity for accurate CD, CM
        # (Reynolds number effects on CD)
        op_point_cruise = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=v_cruise,
            alpha=alpha_cruise,
        )
        
        analysis_cruise = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_cruise,
        )
        res_cruise = analysis_cruise.run()
        
        # Normalize keys
        if "Cm" in res_cruise and "CM" not in res_cruise: res_cruise["CM"] = res_cruise["Cm"]
        if "Cl" in res_cruise and "CL" not in res_cruise: res_cruise["CL"] = res_cruise["Cl"]
        if "Cd" in res_cruise and "CD" not in res_cruise: res_cruise["CD"] = res_cruise["Cd"]
        cruise_drag_correction = self._estimate_blind_body_pressure_drag_delta_cd(
            cl=float(self._coerce_value(res_cruise.get("CL", 0.0))),
            velocity=float(v_cruise),
            alpha=float(alpha_cruise),
            altitude_m=float(self.wing_project.twist_trim.cruise_altitude_m),
        )
        
        metrics["cruise_velocity"] = v_cruise
        metrics["cruise_alpha"] = alpha_cruise
        metrics["cruise_cl"] = self._coerce_value(res_cruise.get("CL", 0.0))
        metrics["cruise_cd_uncorrected"] = self._coerce_value(res_cruise.get("CD", 0.0))
        metrics["cruise_pressure_drag_delta_cd"] = cruise_drag_correction
        metrics["cruise_cd"] = metrics["cruise_cd_uncorrected"] + cruise_drag_correction
        metrics["cruise_l_d_uncorrected"] = metrics["cruise_cl"] / max(metrics["cruise_cd_uncorrected"], 1e-6)
        metrics["cruise_cm"] = self._coerce_value(res_cruise.get("CM", 0.0))
        metrics["cruise_l_d"] = metrics["cruise_cl"] / max(metrics["cruise_cd"], 1e-6)
        
        # 2. Takeoff Condition
        # Takeoff speed is 1.2 * V_stall (standard safety margin)
        # At V_takeoff, the aircraft has excess lift (L > W) for rotation/climb
        # V_stall is the speed where L = W at CL_max
        
        cl_max = self.wing_project.twist_trim.estimated_cl_max
        
        # First, get actual CL_max at high alpha from AeroBuildup
        # Estimate alpha for CL_max
        alpha_stall = self.estimate_alpha_for_cl(cl_max)
        
        # Run analysis at stall alpha to get actual CL_max
        v_ref_stall = self.velocity_for_cl(cl_max)
        op_point_stall = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=v_ref_stall,
            alpha=alpha_stall,
        )
        analysis_stall = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_stall,
        )
        res_stall = analysis_stall.run()
        actual_cl_max = self._coerce_value(res_stall.get("CL", res_stall.get("Cl", 0.0)))
        
        # Calculate V_stall using actual CL_max (speed where L = W at CL_max)
        if actual_cl_max > 1e-4:
            v_stall = self.velocity_for_cl(actual_cl_max)
        else:
            v_stall = v_ref_stall
        
        # V_takeoff = 1.2 * V_stall (provides safety margin and excess lift)
        v_takeoff = 1.2 * v_stall
        
        # At V_takeoff, the expected CL is lower (CL_max / 1.44) due to higher speed
        # Estimate alpha for this CL
        cl_takeoff_expected = actual_cl_max / 1.44
        alpha_takeoff = self.estimate_alpha_for_cl(cl_takeoff_expected)
        
        # Run analysis at takeoff condition
        op_point_to = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),  # Takeoff at sea level
            velocity=v_takeoff,
            alpha=alpha_takeoff,
        )
        analysis_to = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_to,
        )
        res_to = analysis_to.run()
        
        # Normalize keys
        if "Cm" in res_to and "CM" not in res_to: res_to["CM"] = res_to["Cm"]
        if "Cl" in res_to and "CL" not in res_to: res_to["CL"] = res_to["Cl"]
        if "Cd" in res_to and "CD" not in res_to: res_to["CD"] = res_to["Cd"]
        takeoff_drag_correction = self._estimate_blind_body_pressure_drag_delta_cd(
            cl=float(self._coerce_value(res_to.get("CL", 0.0))),
            velocity=float(v_takeoff),
            alpha=float(alpha_takeoff),
            altitude_m=0.0,
        )
        
        metrics["takeoff_velocity"] = v_takeoff
        metrics["takeoff_alpha"] = alpha_takeoff
        metrics["takeoff_cl"] = self._coerce_value(res_to.get("CL", 0.0))
        metrics["takeoff_cd_uncorrected"] = self._coerce_value(res_to.get("CD", 0.0))
        metrics["takeoff_pressure_drag_delta_cd"] = takeoff_drag_correction
        metrics["takeoff_cd"] = metrics["takeoff_cd_uncorrected"] + takeoff_drag_correction
        metrics["takeoff_l_d_uncorrected"] = metrics["takeoff_cl"] / max(metrics["takeoff_cd_uncorrected"], 1e-6)
        metrics["takeoff_cm"] = self._coerce_value(res_to.get("CM", 0.0))
        metrics["takeoff_l_d"] = metrics["takeoff_cl"] / max(metrics["takeoff_cd"], 1e-6)
        
        return metrics

    def get_named_flight_condition(self, condition_name: str) -> Dict[str, float]:
        """
        Resolve a named flight condition using the same definitions as the Performance tab.

        Supported names:
        - "cruise": Trimmed cruise point at cruise altitude
        - "takeoff": 1.2 * V_stall point at sea level
        """
        name = str(condition_name).strip().lower()
        metrics = self.calculate_performance_metrics()

        if name == "cruise":
            return {
                "condition_name": "cruise",
                "velocity_mps": float(metrics["cruise_velocity"]),
                "alpha_deg": float(metrics["cruise_alpha"]),
                "altitude_m": float(self.wing_project.twist_trim.cruise_altitude_m),
            }
        if name == "takeoff":
            return {
                "condition_name": "takeoff",
                "velocity_mps": float(metrics["takeoff_velocity"]),
                "alpha_deg": float(metrics["takeoff_alpha"]),
                "altitude_m": 0.0,
            }
        raise ValueError(f"Unsupported flight condition '{condition_name}'.")

    def resolve_structural_flight_condition(
        self,
        flight_condition: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Merge a structural-analysis flight-condition request with named cruise/takeoff points.
        """
        resolved: Dict[str, Any] = dict(flight_condition or {})
        name = str(resolved.get("condition_name", "")).strip().lower()
        if name in {"cruise", "takeoff"}:
            resolved.update(self.get_named_flight_condition(name))
        elif not name:
            resolved["condition_name"] = "custom"

        resolved["load_factor"] = float(resolved.get("load_factor", 2.5))
        return resolved

    def calculate_performance_polars(
        self, alpha_range: Tuple[float, float] = (-10, 15), num_points: int = 20
    ) -> Dict[str, Any]:
        """
        Calculates performance polars (CL, CD, Cm vs Alpha) for Cruise and Takeoff conditions.
        Uses actual cruise velocity (based on AeroBuildup CL) to ensure consistency with metrics.
        """
        # 1. Build Airplane
        wing = self.build_wing()
        
        # Calculate CG for reference
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        
        # 2. Define Conditions
        # Calculate actual cruise velocity (same logic as calculate_performance_metrics)
        # First get actual CL at cruise alpha from AeroBuildup
        if self.wing_project.optimized_twist_deg is not None:
            alpha_cruise = 0.0
        else:
            alpha_cruise = self.estimate_alpha_for_cl()
        
        # Run a reference analysis to get actual CL at cruise alpha
        v_ref = self.design_velocity()
        op_point_ref = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=v_ref,
            alpha=alpha_cruise,
        )
        analysis_ref = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_ref,
        )
        res_ref = analysis_ref.run()
        actual_cruise_cl = self._coerce_value(res_ref.get("CL", res_ref.get("Cl", 0.0)))
        
        # Calculate TRUE cruise velocity using actual CL (ensures L = W)
        if actual_cruise_cl > 1e-4:
            v_cruise = self.velocity_for_cl(actual_cruise_cl)
        else:
            v_cruise = v_ref
        
        # Calculate V_takeoff: 1.2 * V_stall (based on actual CL_max from AeroBuildup)
        cl_max = self.wing_project.twist_trim.estimated_cl_max
        alpha_stall = self.estimate_alpha_for_cl(cl_max)
        
        # Get actual CL_max at stall alpha
        v_ref_stall = self.velocity_for_cl(cl_max)
        op_point_stall = asb.OperatingPoint(
            atmosphere=asb.Atmosphere(altitude=0),
            velocity=v_ref_stall,
            alpha=alpha_stall,
        )
        analysis_stall = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point_stall,
        )
        res_stall = analysis_stall.run()
        actual_cl_max = self._coerce_value(res_stall.get("CL", res_stall.get("Cl", 0.0)))
        
        # V_stall where L = W at actual CL_max
        if actual_cl_max > 1e-4:
            v_stall = self.velocity_for_cl(actual_cl_max)
        else:
            v_stall = v_ref_stall
        
        # V_takeoff = 1.2 * V_stall (provides excess lift for rotation)
        v_takeoff = 1.2 * v_stall

        conditions = {
            "cruise": {
                "velocity": v_cruise,  # Use actual cruise velocity, not design_velocity()
                "altitude": self.wing_project.twist_trim.cruise_altitude_m
            },
            "takeoff": {
                "velocity": v_takeoff,  # 1.2 * V_stall for excess lift
                "altitude": 0.0  # Takeoff at Sea Level
            }
        }
        
        # 3. Run Sweeps
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        results = {"alpha": alphas.tolist()}
        
        for name, cond in conditions.items():
            atmo = asb.Atmosphere(altitude=cond["altitude"])
            op_point = asb.OperatingPoint(
                atmosphere=atmo,
                velocity=cond["velocity"],
                alpha=alphas,
            )
            
            analysis = asb.AeroBuildup(
                airplane=airplane,
                op_point=op_point,
            )
            res = analysis.run()
            
            # Extract and compute forces
            q = op_point.dynamic_pressure()
            s = airplane.s_ref
            c = airplane.c_ref
            
            cl = res.get("CL", res.get("Cl", np.zeros_like(alphas)))
            cd = res.get("CD", res.get("Cd", np.zeros_like(alphas)))
            cm = res.get("CM", res.get("Cm", np.zeros_like(alphas)))
            cd_uncorrected = cd
            cd_pressure_delta = _np_std.asarray(
                [
                    self._estimate_blind_body_pressure_drag_delta_cd(
                        cl=float(cl_i),
                        velocity=float(cond["velocity"]),
                        alpha=float(alpha_i),
                        altitude_m=float(cond["altitude"]),
                    )
                    for cl_i, alpha_i in zip(_np_std.asarray(cl, dtype=float), _np_std.asarray(alphas, dtype=float))
                ],
                dtype=float,
            )
            cd = cd_uncorrected + cd_pressure_delta
            
            lift = cl * q * s
            drag = cd * q * s
            moment = cm * q * s * c
            l_d = cl / np.maximum(cd, 1e-6)
            
            results[name] = {
                "CL": self._coerce_value(cl),
                "CD": self._coerce_value(cd),
                "CD_uncorrected": self._coerce_value(cd_uncorrected),
                "CD_pressure_drag_delta": self._coerce_value(cd_pressure_delta),
                "CM": self._coerce_value(cm),
                "L": self._coerce_value(lift),
                "D": self._coerce_value(drag),
                "M": self._coerce_value(moment),
                "L_D": self._coerce_value(l_d),
            }
            
        return results

    def _filter_sections_for_vlm(
        self,
        twist_override: Optional[List[float]] = None,
        min_section_spacing_m: Optional[float] = None,
        max_sections: int = 24,
    ) -> List[SpanwiseSection]:
        sections = self.spanwise_sections(twist_override=twist_override)
        if len(sections) <= 2:
            return sections

        total_half_span = max(abs(sections[-1].y_m), 1e-9)
        min_spacing = (
            float(min_section_spacing_m)
            if min_section_spacing_m is not None
            else max(total_half_span / 40.0, 0.02)
        )

        filtered = [sections[0]]
        for section in sections[1:-1]:
            if section.y_m - filtered[-1].y_m >= min_spacing:
                filtered.append(section)

        if filtered[-1].y_m < sections[-1].y_m - 1e-9:
            filtered.append(sections[-1])

        if len(filtered) > max_sections:
            sample_idx = [int(round(idx)) for idx in np.linspace(0, len(filtered) - 1, max_sections)]
            dedup_idx: List[int] = []
            for idx in sample_idx:
                if not dedup_idx or idx != dedup_idx[-1]:
                    dedup_idx.append(idx)
            filtered = [filtered[idx] for idx in dedup_idx]
            filtered[0] = sections[0]
            filtered[-1] = sections[-1]

        return filtered

    def _build_vlm_wing(
        self,
        twist_override: Optional[List[float]] = None,
        min_section_spacing_m: Optional[float] = None,
    ) -> Tuple[asb.Wing, List[SpanwiseSection]]:
        vlm_sections = self._filter_sections_for_vlm(
            twist_override=twist_override,
            min_section_spacing_m=min_section_spacing_m,
        )
        xsecs = [
            asb.WingXSec(
                xyz_le=[section.x_le_m, section.y_m, section.z_m],
                chord=section.chord_m,
                twist=section.twist_deg,
                airfoil=section.airfoil,
            )
            for section in vlm_sections
        ]
        return (
            asb.Wing(
                name="Flying Wing VLM",
                xsecs=xsecs,
                symmetric=True,
            ),
            vlm_sections,
        )

    def _run_stable_vlm(
        self,
        velocity: float,
        alpha: float,
        xyz_ref: List[float],
        altitude_m: Optional[float] = None,
        twist_override: Optional[List[float]] = None,
        spanwise_resolution_hint: Optional[int] = None,
        chordwise_resolution_hint: Optional[int] = None,
    ) -> Tuple[asb.VortexLatticeMethod, Dict[str, Any], Dict[str, Any]]:
        import numpy as _np_std

        base_sections = self.spanwise_sections(twist_override=twist_override)
        total_half_span = max(abs(base_sections[-1].y_m), 1e-9) if base_sections else max(self.wing_project.planform.half_span(), 1e-9)
        atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        op_point = asb.OperatingPoint(
            atmosphere=atmosphere,
            velocity=velocity,
            alpha=alpha,
        )

        last_error: Optional[Exception] = None
        attempt_errors: List[str] = []

        for spacing_scale in (40.0, 24.0):
            min_spacing = max(total_half_span / spacing_scale, 0.02)
            wing, vlm_sections = self._build_vlm_wing(
                twist_override=twist_override,
                min_section_spacing_m=min_spacing,
            )
            airplane = asb.Airplane(
                name=self.wing_project.name,
                wings=[wing],
                xyz_ref=xyz_ref,
            )

            xsec_count = len(vlm_sections)
            sr_primary = max(4, min(6, xsec_count // 4))
            sr_secondary = max(4, min(8, xsec_count // 3))

            candidate_settings: List[Tuple[int, int]] = []
            if spanwise_resolution_hint is not None or chordwise_resolution_hint is not None:
                hinted_sr = sr_primary if spanwise_resolution_hint is None else min(max(4, int(spanwise_resolution_hint)), 8)
                hinted_cr = 3 if chordwise_resolution_hint is None else min(max(3, int(chordwise_resolution_hint)), 4)
                candidate_settings.append((hinted_sr, hinted_cr))
            candidate_settings.extend(
                [
                    (sr_primary, 3),
                    (sr_secondary, 3),
                    (sr_primary, 4),
                    (sr_secondary, 4),
                ]
            )

            dedup_settings: List[Tuple[int, int]] = []
            for setting in candidate_settings:
                if setting not in dedup_settings:
                    dedup_settings.append(setting)

            for spanwise_resolution, chordwise_resolution in dedup_settings:
                try:
                    vlm = asb.VortexLatticeMethod(
                        airplane=airplane,
                        op_point=op_point,
                        spanwise_resolution=spanwise_resolution,
                        chordwise_resolution=chordwise_resolution,
                    )
                    aero_result = vlm.run()
                    total_lift_vlm = float(aero_result.get("L", 0.0))
                    cl_vlm = float(aero_result.get("CL", 0.0))
                    forces_g = _np_std.asarray(vlm.forces_geometry)
                    vortex_centers = _np_std.asarray(vlm.vortex_centers)

                    if (
                        not math.isfinite(cl_vlm)
                        or not math.isfinite(total_lift_vlm)
                        or abs(cl_vlm) > 5.0
                        or total_lift_vlm <= 0.0
                        or forces_g.ndim != 2
                        or vortex_centers.ndim != 2
                        or len(forces_g) == 0
                        or not _np_std.all(_np_std.isfinite(forces_g))
                        or not _np_std.all(_np_std.isfinite(vortex_centers))
                    ):
                        raise ValueError(f"unstable result (CL={cl_vlm:.3g}, L={total_lift_vlm:.3g})")

                    return vlm, aero_result, {
                        "spanwise_resolution": spanwise_resolution,
                        "chordwise_resolution": chordwise_resolution,
                        "filtered_section_count": len(vlm_sections),
                        "original_section_count": len(base_sections),
                        "min_section_spacing_m": min_spacing,
                    }
                except Exception as exc:
                    last_error = exc
                    attempt_errors.append(
                        f"spacing={min_spacing:.4f} sr={spanwise_resolution} cr={chordwise_resolution}: {exc}"
                    )

        detail = "; ".join(attempt_errors[-6:]) if attempt_errors else "no attempts recorded"
        raise ValueError(f"Could not find a stable VLM discretization. Last error: {last_error}. Attempts: {detail}")

    @staticmethod
    def _naca_thickness_distribution(x_over_c: np.ndarray, thickness_ratio: float) -> np.ndarray:
        x = _np_std.clip(_np_std.asarray(x_over_c, dtype=float), 0.0, 1.0)
        y_t = 5.0 * float(thickness_ratio) * (
            0.2969 * _np_std.sqrt(_np_std.clip(x, 1e-12, 1.0))
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x * x * x
            - 0.1015 * x * x * x * x
        )
        return 2.0 * y_t

    def _build_body_geometry_proxies(
        self,
        s_ref_m2: float,
        x_samples: int = 320,
    ) -> Dict[str, float]:
        planform = self.wing_project.planform
        sections = sorted(planform.body_sections, key=lambda section: section.y_pos)
        if len(sections) < 2 or s_ref_m2 <= 0.0:
            return {
                "body_area_m2": 0.0,
                "body_area_ratio": 0.0,
                "body_camber_area_ratio": 0.0,
                "body_mean_thickness_ratio": 0.0,
                "body_length_m": 0.0,
                "body_form_drag_proxy": 0.0,
                "body_peak_cross_section_area_m2": 0.0,
            }

        body_area_one_side = 0.0
        body_camber_area_one_side = 0.0
        body_thickness_area_one_side = 0.0
        x_min = min(section.x_offset for section in sections)
        x_max = max(section.x_offset + section.chord for section in sections)
        body_length_m = max(1e-9, x_max - x_min)

        x_grid = _np_std.linspace(x_min, x_max, max(80, int(x_samples)))
        cross_section_area = _np_std.zeros_like(x_grid)

        for inboard, outboard in zip(sections[:-1], sections[1:]):
            dy = float(outboard.y_pos - inboard.y_pos)
            if dy <= 0.0:
                continue

            chord_m = 0.5 * float(inboard.chord + outboard.chord)
            x_le_m = 0.5 * float(inboard.x_offset + outboard.x_offset)
            airfoil_inboard = self._get_airfoil(inboard.airfoil)
            airfoil_outboard = self._get_airfoil(outboard.airfoil)

            try:
                max_camber = 0.5 * (float(airfoil_inboard.max_camber()) + float(airfoil_outboard.max_camber()))
            except Exception:
                max_camber = 0.0
            try:
                thickness_ratio = 0.5 * (
                    float(airfoil_inboard.max_thickness()) + float(airfoil_outboard.max_thickness())
                )
            except Exception:
                thickness_ratio = 0.0

            body_area_one_side += chord_m * dy
            body_camber_area_one_side += chord_m * max_camber * dy
            body_thickness_area_one_side += chord_m * thickness_ratio * dy

            local_mask = (x_grid >= x_le_m) & (x_grid <= x_le_m + chord_m)
            if not _np_std.any(local_mask):
                continue

            x_local = (x_grid[local_mask] - x_le_m) / max(chord_m, 1e-9)
            local_thickness = self._naca_thickness_distribution(x_local, thickness_ratio) * chord_m
            cross_section_area[local_mask] += 2.0 * dy * local_thickness

        body_area_m2 = 2.0 * body_area_one_side
        body_camber_area_ratio = 2.0 * body_camber_area_one_side / s_ref_m2
        body_mean_thickness_ratio = (2.0 * body_thickness_area_one_side / body_area_m2) if body_area_m2 > 0.0 else 0.0
        dA_dx = _np_std.gradient(cross_section_area, x_grid)
        body_form_drag_proxy = float(_np_std.trapezoid(dA_dx * dA_dx, x_grid)) / max(s_ref_m2 * body_length_m, 1e-9)

        return {
            "body_area_m2": float(body_area_m2),
            "body_area_ratio": float(body_area_m2 / s_ref_m2),
            "body_camber_area_ratio": float(body_camber_area_ratio),
            "body_mean_thickness_ratio": float(body_mean_thickness_ratio),
            "body_length_m": float(body_length_m),
            "body_form_drag_proxy": float(body_form_drag_proxy),
            "body_peak_cross_section_area_m2": float(_np_std.max(cross_section_area)) if len(cross_section_area) else 0.0,
        }

    @staticmethod
    def _compute_pressure_proxy_from_strip_rows(
        strip_rows: List[Dict[str, float]],
        s_ref_m2: float,
    ) -> float:
        if len(strip_rows) < 2 or s_ref_m2 <= 0.0:
            return 0.0
        y = _np_std.asarray([float(row["y_m"]) for row in strip_rows], dtype=float)
        chord = _np_std.asarray([float(row["chord_m"]) for row in strip_rows], dtype=float)
        proximity = _np_std.asarray(
            [float(row["onset_proximity"]) * float(row["thickness_factor"]) for row in strip_rows],
            dtype=float,
        )
        return 2.0 * float(_np_std.trapezoid(chord * proximity, y)) / max(s_ref_m2, 1e-9)

    def _evaluate_local_section_onset(
        self,
        airfoil: Any,
        cl_target: float,
        reynolds: float,
        mach: float,
        alpha_grid: np.ndarray,
        polar_cache: Dict[Tuple[str, int, int], Dict[str, _np_std.ndarray]],
        sep_slope_fraction: float = 0.7,
        sep_thickness_start: float = 0.08,
        sep_thickness_full: float = 0.14,
        drag_rise_alpha_window: float = 4.0,
    ) -> Dict[str, float]:
        cache_key = (
            str(getattr(airfoil, "name", "airfoil")),
            int(round(reynolds / 5000.0)),
            int(round(mach * 1000.0)),
        )
        if cache_key not in polar_cache:
            reynolds_array = _np_std.full_like(alpha_grid, fill_value=float(reynolds), dtype=float)
            mach_array = _np_std.full_like(alpha_grid, fill_value=float(mach), dtype=float)
            aero = airfoil.get_aero_from_neuralfoil(alpha=alpha_grid, Re=reynolds_array, mach=mach_array)
            polar_cache[cache_key] = {
                "alpha_deg": _np_std.asarray(alpha_grid, dtype=float).copy(),
                "CL": _np_std.asarray(aero["CL"], dtype=float),
            }

        polar = polar_cache[cache_key]
        cl_values = polar["CL"]
        alpha_values = polar["alpha_deg"]

        nearest_idx = int(_np_std.argmin(_np_std.abs(cl_values - cl_target)))
        alpha_eff = float(alpha_values[nearest_idx])
        dcl_dalpha = _np_std.gradient(cl_values, alpha_values)
        linear_mask = (alpha_values >= -1.0) & (alpha_values <= 4.0)
        if _np_std.any(linear_mask):
            reference_slope = float(_np_std.median(dcl_dalpha[linear_mask]))
        else:
            reference_slope = float(_np_std.max(dcl_dalpha))
        if not math.isfinite(reference_slope) or abs(reference_slope) < 1e-9:
            reference_slope = float(_np_std.max(dcl_dalpha))

        slope_limit = sep_slope_fraction * reference_slope
        sep_candidates = _np_std.where((alpha_values > 0.0) & (dcl_dalpha < slope_limit))[0]
        alpha_sep = float(alpha_values[int(sep_candidates[0])]) if len(sep_candidates) > 0 else float(alpha_values[-1])
        alpha_margin = alpha_sep - alpha_eff

        try:
            thickness_ratio = float(airfoil.max_thickness())
        except Exception:
            thickness_ratio = 0.0

        if sep_thickness_full <= sep_thickness_start:
            thickness_factor = 1.0 if thickness_ratio > sep_thickness_start else 0.0
        else:
            thickness_factor = float(
                _np_std.clip(
                    (thickness_ratio - sep_thickness_start) / (sep_thickness_full - sep_thickness_start),
                    0.0,
                    1.0,
                )
            )

        if drag_rise_alpha_window <= 1e-9:
            onset_proximity = 1.0 if alpha_margin <= 0.0 else 0.0
        else:
            onset_proximity = float(_np_std.clip(1.0 - alpha_margin / drag_rise_alpha_window, 0.0, 1.0))

        return {
            "onset_proximity": onset_proximity,
            "thickness_factor": thickness_factor,
        }

    def _compute_spanload_pressure_proxy(
        self,
        y_half: np.ndarray,
        lift_per_span_half: np.ndarray,
        velocity: float,
        q: float,
        altitude_m: Optional[float],
        s_ref_m2: float,
    ) -> float:
        if len(y_half) < 2 or q <= 0.0 or s_ref_m2 <= 0.0:
            return 0.0

        atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        rho = float(atmosphere.density())
        mu = float(atmosphere.dynamic_viscosity())
        speed_of_sound = float(atmosphere.speed_of_sound())
        mach = float(velocity) / max(speed_of_sound, 1e-9)
        alpha_grid = _np_std.linspace(-8.0, 20.0, 113)

        sections = self.spanwise_sections()
        sec_y = _np_std.asarray([abs(section.y_m) for section in sections], dtype=float)
        sec_chord = _np_std.asarray([section.chord_m for section in sections], dtype=float)
        polar_cache: Dict[Tuple[str, int, int], Dict[str, _np_std.ndarray]] = {}
        strip_rows: List[Dict[str, float]] = []

        for idx, y_value in enumerate(y_half):
            chord_m = float(_np_std.interp(y_value, sec_y, sec_chord))
            chord_m = max(chord_m, 1e-6)
            nearest_section_index = int(_np_std.argmin(_np_std.abs(sec_y - y_value)))
            airfoil = sections[nearest_section_index].airfoil
            cl_local = float(lift_per_span_half[idx] / max(q * chord_m, 1e-9))
            reynolds = float(rho * velocity * chord_m / max(mu, 1e-12))
            local = self._evaluate_local_section_onset(
                airfoil=airfoil,
                cl_target=cl_local,
                reynolds=reynolds,
                mach=mach,
                alpha_grid=alpha_grid,
                polar_cache=polar_cache,
            )
            strip_rows.append(
                {
                    "y_m": float(y_value),
                    "chord_m": chord_m,
                    "onset_proximity": float(local["onset_proximity"]),
                    "thickness_factor": float(local["thickness_factor"]),
                }
            )

        return self._compute_pressure_proxy_from_strip_rows(strip_rows, s_ref_m2=s_ref_m2)

    def _compute_body_cl0_proxy(
        self,
        velocity: float,
        altitude_m: Optional[float],
        s_ref_m2: float,
    ) -> float:
        if s_ref_m2 <= 0.0:
            return 0.0

        sections = sorted(self.wing_project.planform.body_sections, key=lambda section: section.y_pos)
        if len(sections) < 2:
            return 0.0

        atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        rho = float(atmosphere.density())
        mu = float(atmosphere.dynamic_viscosity())
        speed_of_sound = float(atmosphere.speed_of_sound())
        mach = float(velocity) / max(speed_of_sound, 1e-9)

        body_cl0_area = 0.0
        for inboard, outboard in zip(sections[:-1], sections[1:]):
            dy = float(outboard.y_pos - inboard.y_pos)
            if dy <= 0.0:
                continue
            chord_m = 0.5 * float(inboard.chord + outboard.chord)
            reynolds = rho * float(velocity) * chord_m / max(mu, 1e-12)
            airfoil_inboard = self._get_airfoil(inboard.airfoil)
            airfoil_outboard = self._get_airfoil(outboard.airfoil)
            aero_inboard = airfoil_inboard.get_aero_from_neuralfoil(alpha=0.0, Re=reynolds, mach=mach)
            aero_outboard = airfoil_outboard.get_aero_from_neuralfoil(alpha=0.0, Re=reynolds, mach=mach)
            cl0 = 0.5 * (float(aero_inboard.get("CL", 0.0)) + float(aero_outboard.get("CL", 0.0)))
            body_cl0_area += 2.0 * cl0 * chord_m * dy

        return float(body_cl0_area) / s_ref_m2

    @staticmethod
    def _compute_body_symmetry_factor(body_cl0_proxy: float, body_proxies: Dict[str, float]) -> float:
        cl0_scale = 0.04
        camber_area_scale = 0.005
        camber_area_ratio = abs(float(body_proxies.get("body_camber_area_ratio", 0.0)))
        symmetry_factor = math.exp(-abs(float(body_cl0_proxy)) / cl0_scale) * math.exp(-camber_area_ratio / camber_area_scale)
        return float(_np_std.clip(symmetry_factor, 0.0, 1.0))

    def _build_blind_body_model_parameters(self, body_proxies: Dict[str, float]) -> Dict[str, float]:
        sweep_deg = float(self.wing_project.planform.sweep_le_deg)
        sweep_cos = max(0.05, math.cos(math.radians(sweep_deg)))
        mean_thickness = max(0.0, float(body_proxies.get("body_mean_thickness_ratio", 0.0)))
        body_length = max(1e-9, float(body_proxies.get("body_length_m", 0.0)))
        peak_cross_section_area = max(0.0, float(body_proxies.get("body_peak_cross_section_area_m2", 0.0)))
        bluffness_ratio = math.sqrt(peak_cross_section_area) / body_length
        form_factor_extra = max(0.0, 2.0 * mean_thickness + 60.0 * mean_thickness**4)
        pressure_decay = max(1e-6, 0.75 * mean_thickness)
        return {
            "sweep_cosine": sweep_cos,
            "drag_form_factor_extra": form_factor_extra,
            "pressure_decay_scale": pressure_decay,
            "body_bluffness_ratio": bluffness_ratio,
            "cambered_lift_scale": 0.96,
            "alpha_break_sweep_factor": 0.096,
            "alpha_relief_gain": 4.1,
            "symmetric_drag_floor": 0.205,
            "symmetric_form_drag_gain": 2.2,
            "cambered_drag_alpha_gain": 1.8,
            "performance_pressure_drag_scale": 5.35,
            "relief_region_mode": "chord_0.55",
            "relief_onset_floor": 0.15,
            "relief_onset_power": 2.0,
            "relief_root_power": 0.5,
            "relief_cap_fraction": 0.95,
            "relief_target_scale": 1.0,
        }

    def _estimate_blind_body_pressure_drag_delta_cd(
        self,
        cl: float,
        velocity: float,
        alpha: float,
        altitude_m: Optional[float],
    ) -> float:
        """Estimate the BWB pressure-drag increment missing from AeroBuildup."""
        try:
            plan = self.wing_project.planform
            if len(plan.body_sections) < 2:
                return 0.0

            s_ref_m2 = float(plan.actual_area())
            if s_ref_m2 <= 0.0 or velocity <= 0.0:
                return 0.0

            body_proxies = self._build_body_geometry_proxies(s_ref_m2=s_ref_m2)
            if body_proxies["body_area_ratio"] <= 0.0:
                return 0.0

            atmosphere = asb.Atmosphere(
                altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
            )
            q = 0.5 * float(atmosphere.density()) * float(velocity) ** 2
            if q <= 0.0:
                return 0.0

            sections = self.spanwise_sections()
            if len(sections) < 2:
                return 0.0

            y_half = _np_std.asarray([max(0.0, float(section.y_m)) for section in sections], dtype=float)
            chord = _np_std.asarray([max(1e-6, float(section.chord_m)) for section in sections], dtype=float)
            order = _np_std.argsort(y_half)
            y_half = y_half[order]
            chord = chord[order]
            half_lift = 0.5 * float(cl) * q * s_ref_m2
            chord_integral = float(_np_std.trapezoid(chord, y_half))
            if abs(chord_integral) <= 1e-9:
                return 0.0

            lift_per_span = half_lift * chord / chord_integral
            pressure_proxy = self._estimate_geometry_pressure_proxy(
                y_half=y_half,
                chord_half=chord,
                lift_per_span_half=lift_per_span,
                q=q,
                s_ref_m2=s_ref_m2,
            )
            body_cl0_proxy = self._estimate_body_camber_cl0_proxy(body_proxies)
            params = self._build_blind_body_model_parameters(body_proxies)
            symmetry_factor = self._compute_body_symmetry_factor(body_cl0_proxy, body_proxies)
            sweep_cos = max(float(params["sweep_cosine"]), 0.05)
            symmetric_term = (
                float(params["symmetric_drag_floor"])
                + float(params["symmetric_form_drag_gain"]) * float(body_proxies["body_form_drag_proxy"])
            )
            cambered_term = (
                (1.0 - symmetry_factor)
                * float(params["cambered_drag_alpha_gain"])
                * abs(float(alpha))
                * abs(float(body_cl0_proxy))
            )
            delta_cd = (
                float(body_proxies["body_area_ratio"])
                * float(params["drag_form_factor_extra"])
                * (symmetric_term + cambered_term)
                * max(float(pressure_proxy), 0.0)
                * max(0.0, float(params.get("performance_pressure_drag_scale", 1.0)))
                / sweep_cos
            )
            return float(_np_std.clip(delta_cd, 0.0, 0.25))
        except Exception as exc:
            print(f"[Performance] BWB pressure drag correction skipped: {exc}")
            return 0.0

    @staticmethod
    def _estimate_geometry_pressure_proxy(
        y_half: np.ndarray,
        chord_half: np.ndarray,
        lift_per_span_half: np.ndarray,
        q: float,
        s_ref_m2: float,
    ) -> float:
        if len(y_half) < 2 or q <= 0.0 or s_ref_m2 <= 0.0:
            return 0.0
        y_half = _np_std.asarray(y_half, dtype=float)
        chord_half = _np_std.maximum(_np_std.asarray(chord_half, dtype=float), 1e-6)
        lift_per_span_half = _np_std.asarray(lift_per_span_half, dtype=float)
        cl_local = _np_std.abs(lift_per_span_half) / _np_std.maximum(q * chord_half, 1e-9)
        load_pressure = _np_std.clip((cl_local - 0.10) / 0.75, 0.0, 1.0)
        return 2.0 * float(_np_std.trapezoid(chord_half * load_pressure, y_half)) / max(s_ref_m2, 1e-9)

    @staticmethod
    def _estimate_body_camber_cl0_proxy(body_proxies: Dict[str, float]) -> float:
        body_area_ratio = max(0.0, float(body_proxies.get("body_area_ratio", 0.0)))
        if body_area_ratio <= 0.0:
            return 0.0
        camber_area_ratio = float(body_proxies.get("body_camber_area_ratio", 0.0))
        mean_camber = camber_area_ratio / max(body_area_ratio, 1e-9)
        return float(_np_std.clip(4.0 * mean_camber * body_area_ratio, -0.25, 0.25))

    @staticmethod
    def _compute_relief_region_end(
        y_half: np.ndarray,
        chord_half: np.ndarray,
        body_y_max: float,
        region_mode: str,
    ) -> float:
        y_half = _np_std.asarray(y_half, dtype=float)
        chord_half = _np_std.asarray(chord_half, dtype=float)
        if len(y_half) == 0:
            return max(body_y_max, 1e-9)
        if region_mode == "body":
            return max(body_y_max, 1e-9)
        if region_mode.startswith("chord_"):
            chord_fraction = float(region_mode.split("_", 1)[1])
            root_chord = max(float(chord_half[0]), 1e-9)
            mask = chord_half >= chord_fraction * root_chord
            if _np_std.any(mask):
                last_idx = int(_np_std.where(mask)[0][-1])
                return max(body_y_max, float(y_half[last_idx]))
            return max(body_y_max, 1e-9)
        if region_mode.startswith("span_"):
            span_fraction = float(region_mode.split("_", 1)[1])
            return max(body_y_max, span_fraction * float(y_half[-1]))
        return max(body_y_max, 1e-9)

    def _build_body_section_zero_alpha_data(
        self,
        velocity: float,
        altitude_m: Optional[float],
    ) -> Dict[str, Any]:
        sections = sorted(self.wing_project.planform.body_sections, key=lambda section: section.y_pos)
        if len(sections) < 2:
            return {
                "y_m": _np_std.asarray([], dtype=float),
                "chord_m": _np_std.asarray([], dtype=float),
                "x_le_m": _np_std.asarray([], dtype=float),
                "cl0": _np_std.asarray([], dtype=float),
                "camber_ratio": _np_std.asarray([], dtype=float),
                "thickness_ratio": _np_std.asarray([], dtype=float),
                "airfoils": [],
                "mean_body_chord_m": 0.0,
            }

        atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        rho = float(atmosphere.density())
        mu = float(atmosphere.dynamic_viscosity())
        speed_of_sound = float(atmosphere.speed_of_sound())
        mach = float(velocity) / max(speed_of_sound, 1e-9)

        y_values = []
        chord_values = []
        x_le_values = []
        cl0_values = []
        camber_values = []
        thickness_values = []
        airfoils = []

        for section in sections:
            chord_m = max(float(section.chord), 1e-6)
            airfoil = self._get_airfoil(section.airfoil)
            reynolds = rho * float(velocity) * chord_m / max(mu, 1e-12)
            aero = airfoil.get_aero_from_neuralfoil(alpha=0.0, Re=reynolds, mach=mach)
            try:
                camber_ratio = float(airfoil.max_camber())
            except Exception:
                camber_ratio = 0.0
            try:
                thickness_ratio = float(airfoil.max_thickness())
            except Exception:
                thickness_ratio = 0.0

            y_values.append(float(section.y_pos))
            chord_values.append(chord_m)
            x_le_values.append(float(section.x_offset))
            cl0_values.append(float(aero.get("CL", 0.0)))
            camber_values.append(camber_ratio)
            thickness_values.append(thickness_ratio)
            airfoils.append(airfoil)

        chord_array = _np_std.asarray(chord_values, dtype=float)
        return {
            "y_m": _np_std.asarray(y_values, dtype=float),
            "chord_m": chord_array,
            "x_le_m": _np_std.asarray(x_le_values, dtype=float),
            "cl0": _np_std.asarray(cl0_values, dtype=float),
            "camber_ratio": _np_std.asarray(camber_values, dtype=float),
            "thickness_ratio": _np_std.asarray(thickness_values, dtype=float),
            "airfoils": airfoils,
            "mean_body_chord_m": float(_np_std.mean(chord_array)) if len(chord_array) else 0.0,
        }

    def _build_distributed_blind_body_lift_delta(
        self,
        y_half: np.ndarray,
        lift_per_span_half: np.ndarray,
        velocity: float,
        alpha: float,
        load_factor: float,
        altitude_m: Optional[float],
        q: float,
        s_ref_m2: float,
        blind_parameters: Optional[Dict[str, float]] = None,
        strip_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        y_half = _np_std.asarray(y_half, dtype=float)
        lift_per_span_half = _np_std.asarray(lift_per_span_half, dtype=float)
        zero = _np_std.zeros_like(lift_per_span_half, dtype=float)
        if len(self.wing_project.planform.body_sections) < 2:
            return {
                "delta_lift_per_span_half": zero,
                "body_strip_rows": [],
                "pressure_proxy": 0.0,
                "symmetry_factor": 0.0,
                "body_cl0_proxy": 0.0,
                "lift_retention": 0.0,
                "cambered_delta_cl": 0.0,
                "relief_delta_cl": 0.0,
                "delta_cl_total": 0.0,
            }
        if len(y_half) < 2 or q <= 0.0 or s_ref_m2 <= 0.0:
            return {
                "delta_lift_per_span_half": zero,
                "body_strip_rows": [],
                "pressure_proxy": 0.0,
                "symmetry_factor": 0.0,
                "body_cl0_proxy": 0.0,
                "lift_retention": 0.0,
                "cambered_delta_cl": 0.0,
                "relief_delta_cl": 0.0,
                "delta_cl_total": 0.0,
            }

        body_proxies = self._build_body_geometry_proxies(s_ref_m2=s_ref_m2)
        if body_proxies["body_area_ratio"] <= 0.0:
            return {
                "delta_lift_per_span_half": zero,
                "body_strip_rows": [],
                "pressure_proxy": 0.0,
                "symmetry_factor": 0.0,
                "body_cl0_proxy": 0.0,
                "lift_retention": 0.0,
                "cambered_delta_cl": 0.0,
                "relief_delta_cl": 0.0,
                "delta_cl_total": 0.0,
            }

        if blind_parameters is None:
            blind_parameters = self._build_blind_body_model_parameters(body_proxies)
        body_section_data = self._build_body_section_zero_alpha_data(
            velocity=velocity,
            altitude_m=altitude_m,
        )
        body_y = body_section_data["y_m"]
        if len(body_y) < 2:
            return {
                "delta_lift_per_span_half": zero,
                "body_strip_rows": [],
                "pressure_proxy": 0.0,
                "symmetry_factor": 0.0,
                "body_cl0_proxy": 0.0,
                "lift_retention": 0.0,
                "cambered_delta_cl": 0.0,
                "relief_delta_cl": 0.0,
                "delta_cl_total": 0.0,
            }

        atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        rho = float(atmosphere.density())
        mu = float(atmosphere.dynamic_viscosity())
        speed_of_sound = float(atmosphere.speed_of_sound())
        mach = float(velocity) / max(speed_of_sound, 1e-9)
        alpha_grid = _np_std.linspace(-8.0, 20.0, 113)
        polar_cache: Dict[Tuple[str, int, int], Dict[str, _np_std.ndarray]] = {}

        pressure_proxy = self._compute_spanload_pressure_proxy(
            y_half=y_half,
            lift_per_span_half=lift_per_span_half,
            velocity=velocity,
            q=q,
            altitude_m=altitude_m,
            s_ref_m2=s_ref_m2,
        )
        body_cl0_proxy = self._compute_body_cl0_proxy(
            velocity=velocity,
            altitude_m=altitude_m,
            s_ref_m2=s_ref_m2,
        )
        symmetry_factor = self._compute_body_symmetry_factor(body_cl0_proxy, body_proxies)
        sweep_cos = float(blind_parameters["sweep_cosine"])
        cambered_lift_scale = max(0.0, float(blind_parameters.get("cambered_lift_scale", 1.0)))
        alpha_break_factor = max(1e-6, float(blind_parameters.get("alpha_break_sweep_factor", 0.096)))
        alpha_break_deg = max(0.5, alpha_break_factor * float(self.wing_project.planform.sweep_le_deg))
        lift_retention = float(_np_std.clip(1.0 - (float(alpha) / alpha_break_deg) ** 2, -0.75, 1.25))
        cambered_body_delta_cl = cambered_lift_scale * body_cl0_proxy * sweep_cos * lift_retention
        raw_relief_target_cl = (
            -symmetry_factor
            * float(blind_parameters["alpha_relief_gain"])
            * abs(float(alpha))
            * pressure_proxy
            * float(body_proxies["body_form_drag_proxy"])
            / max(sweep_cos, 0.05)
        )
        relief_target_scale = max(0.0, float(blind_parameters.get("relief_target_scale", 1.0)))
        cambered_half_lift_target = cambered_body_delta_cl * q * s_ref_m2 * 0.5 * load_factor
        relief_half_lift_target = raw_relief_target_cl * q * s_ref_m2 * 0.5 * load_factor * relief_target_scale

        mean_body_chord = max(float(body_section_data["mean_body_chord_m"]), 1e-6)
        body_y_max = float(body_y[-1])

        cambered_basis = _np_std.zeros_like(lift_per_span_half, dtype=float)
        relief_shape = _np_std.zeros_like(lift_per_span_half, dtype=float)
        body_strip_rows: List[Dict[str, float]] = []

        sections = [section for section in self.spanwise_sections() if float(section.y_m) >= -1e-9]
        sections = sorted(sections, key=lambda section: float(section.y_m))
        if len(sections) < 2:
            return {
                "delta_lift_per_span_half": zero,
                "body_strip_rows": [],
                "pressure_proxy": float(pressure_proxy),
                "symmetry_factor": float(symmetry_factor),
                "body_cl0_proxy": float(body_cl0_proxy),
                "lift_retention": float(lift_retention),
                "cambered_delta_cl": 0.0,
                "relief_delta_cl": 0.0,
                "delta_cl_total": 0.0,
            }

        section_y = _np_std.asarray([float(section.y_m) for section in sections], dtype=float)
        section_chord = _np_std.asarray([float(section.chord_m) for section in sections], dtype=float)
        section_x_le = _np_std.asarray([float(section.x_le_m) for section in sections], dtype=float)
        section_thickness = []
        section_airfoils = []
        for section in sections:
            section_airfoils.append(section.airfoil)
            try:
                section_thickness.append(float(section.airfoil.max_thickness()))
            except Exception:
                section_thickness.append(0.0)
        section_thickness = _np_std.asarray(section_thickness, dtype=float)

        chord_half = _np_std.zeros_like(y_half, dtype=float)
        onset_half = _np_std.zeros_like(y_half, dtype=float)
        thickness_half = _np_std.zeros_like(y_half, dtype=float)
        supplied_y = None
        supplied_chord = None
        supplied_onset = None
        supplied_thickness = None
        if strip_rows:
            supplied_y = _np_std.asarray([float(row.get("y_m", 0.0) or 0.0) for row in strip_rows], dtype=float)
            supplied_chord = _np_std.asarray([float(row.get("chord_m", 0.0) or 0.0) for row in strip_rows], dtype=float)
            supplied_onset = _np_std.asarray([float(row.get("onset_proximity", 0.0) or 0.0) for row in strip_rows], dtype=float)
            supplied_thickness = _np_std.asarray(
                [float(row.get("thickness_factor", 0.0) or 0.0) for row in strip_rows],
                dtype=float,
            )
            if (
                len(supplied_y) != len(strip_rows)
                or len(supplied_y) == 0
                or len(supplied_y) != len(supplied_chord)
                or len(supplied_y) != len(supplied_onset)
                or len(supplied_y) != len(supplied_thickness)
            ):
                supplied_y = None
                supplied_chord = None
                supplied_onset = None
                supplied_thickness = None
        region_mode = str(blind_parameters.get("relief_region_mode", "chord_0.55"))
        onset_floor = max(0.0, float(blind_parameters.get("relief_onset_floor", 0.15)))
        onset_power = max(0.0, float(blind_parameters.get("relief_onset_power", 2.0)))
        root_power = max(0.0, float(blind_parameters.get("relief_root_power", 0.5)))
        cap_fraction = float(_np_std.clip(float(blind_parameters.get("relief_cap_fraction", 0.95)), 0.0, 0.999))

        for idx, y_value in enumerate(y_half):
            wing_hi = int(_np_std.searchsorted(section_y, y_value, side="right"))
            wing_idx = min(max(wing_hi - 1, 0), len(section_y) - 2)
            wy0 = float(section_y[wing_idx])
            wy1 = float(section_y[wing_idx + 1])
            wing_dy = max(wy1 - wy0, 1e-9)
            wing_blend = float(_np_std.clip((y_value - wy0) / wing_dy, 0.0, 1.0))

            wing_chord = (1.0 - wing_blend) * float(section_chord[wing_idx]) + wing_blend * float(section_chord[wing_idx + 1])
            wing_chord = max(wing_chord, 1e-6)
            wing_x0 = float(section_x_le[wing_idx])
            wing_x1 = float(section_x_le[wing_idx + 1])
            local_sweep_deg = abs(math.degrees(math.atan2(wing_x1 - wing_x0, wing_dy)))
            local_airfoil_index = wing_idx if wing_blend < 0.5 else wing_idx + 1
            local_airfoil = section_airfoils[local_airfoil_index]
            local_thickness = (
                (1.0 - wing_blend) * float(section_thickness[wing_idx])
                + wing_blend * float(section_thickness[wing_idx + 1])
            )
            local_thickness = max(local_thickness, 0.0)

            if supplied_y is not None:
                wing_chord = max(
                    1e-6,
                    float(_np_std.interp(y_value, supplied_y, supplied_chord, left=supplied_chord[0], right=supplied_chord[-1])),
                )
                onset_value = float(_np_std.interp(y_value, supplied_y, supplied_onset, left=supplied_onset[0], right=supplied_onset[-1]))
                thickness_value = float(
                    _np_std.interp(y_value, supplied_y, supplied_thickness, left=supplied_thickness[0], right=supplied_thickness[-1])
                )
                local_onset = {
                    "onset_proximity": onset_value,
                    "thickness_factor": thickness_value,
                }
                local_thickness = thickness_value
            else:
                cl_local = float(lift_per_span_half[idx] / max(q * wing_chord, 1e-9))
                reynolds = float(rho * velocity * wing_chord / max(mu, 1e-12))
                local_onset = self._evaluate_local_section_onset(
                    airfoil=local_airfoil,
                    cl_target=cl_local,
                    reynolds=reynolds,
                    mach=mach,
                    alpha_grid=alpha_grid,
                    polar_cache=polar_cache,
                )

            chord_half[idx] = wing_chord
            thickness_half[idx] = max(local_thickness, 0.0)
            onset_half[idx] = float(local_onset["onset_proximity"])

            if y_value < float(body_y[0]) - 1e-9 or y_value > body_y_max + 1e-9:
                body_strip_rows.append(
                    {
                        "y_m": float(y_value),
                        "chord_m": wing_chord,
                        "body_chord_m": 0.0,
                        "body_cl0_local": 0.0,
                        "body_sweep_deg": float(local_sweep_deg),
                        "body_thickness_ratio": float(local_thickness),
                        "onset_proximity": float(local_onset["onset_proximity"]),
                        "thickness_factor": float(local_onset["thickness_factor"]),
                        "cambered_basis": 0.0,
                        "relief_shape": 0.0,
                    }
                )
                continue

            seg_hi = int(_np_std.searchsorted(body_y, y_value, side="right"))
            seg_idx = min(max(seg_hi - 1, 0), len(body_y) - 2)
            y0 = float(body_y[seg_idx])
            y1 = float(body_y[seg_idx + 1])
            dy = max(y1 - y0, 1e-9)
            blend = float(_np_std.clip((y_value - y0) / dy, 0.0, 1.0))

            chord0 = float(body_section_data["chord_m"][seg_idx])
            chord1 = float(body_section_data["chord_m"][seg_idx + 1])
            x0 = float(body_section_data["x_le_m"][seg_idx])
            x1 = float(body_section_data["x_le_m"][seg_idx + 1])
            cl0_0 = float(body_section_data["cl0"][seg_idx])
            cl0_1 = float(body_section_data["cl0"][seg_idx + 1])

            chord_m = (1.0 - blend) * chord0 + blend * chord1
            chord_m = max(chord_m, 1e-6)
            local_cl0 = (1.0 - blend) * cl0_0 + blend * cl0_1
            cambered_basis[idx] = chord_m * max(abs(local_cl0), 1e-6)
            body_strip_rows.append(
                {
                    "y_m": float(y_value),
                    "chord_m": wing_chord,
                    "body_chord_m": chord_m,
                    "body_cl0_local": float(local_cl0),
                    "body_sweep_deg": float(local_sweep_deg),
                    "body_thickness_ratio": float(local_thickness),
                    "onset_proximity": float(local_onset["onset_proximity"]),
                    "thickness_factor": float(local_onset["thickness_factor"]),
                    "cambered_basis": float(cambered_basis[idx]),
                    "relief_shape": 0.0,
                }
            )

        y_end = self._compute_relief_region_end(
            y_half=y_half,
            chord_half=chord_half,
            body_y_max=body_y_max,
            region_mode=region_mode,
        )
        region_mask = y_half <= y_end + 1e-9
        root_eta = _np_std.zeros_like(y_half, dtype=float)
        if y_end > 1e-9:
            root_eta = _np_std.clip(y_half / y_end, 0.0, 1.0)
        root_mask = _np_std.clip(1.0 - root_eta, 0.0, 1.0) ** root_power
        onset_shape = _np_std.maximum(onset_half, onset_floor) ** onset_power
        thickness_shape = _np_std.maximum(thickness_half, 0.15)
        relief_shape = _np_std.where(region_mask, onset_shape * thickness_shape * root_mask, 0.0)
        for idx, row in enumerate(body_strip_rows):
            row["relief_shape"] = float(relief_shape[idx])

        cambered_lift_per_span = _np_std.zeros_like(lift_per_span_half, dtype=float)
        relief_lift_per_span = _np_std.zeros_like(lift_per_span_half, dtype=float)
        cambered_basis_integral = float(_np_std.trapezoid(cambered_basis, y_half))
        if abs(cambered_half_lift_target) > 1e-9 and cambered_basis_integral > 1e-9:
            cambered_lift_per_span = cambered_half_lift_target * cambered_basis / cambered_basis_integral

        base_lift = _np_std.maximum(lift_per_span_half, 0.0)
        target_relief_half = max(0.0, abs(relief_half_lift_target))
        applied_relief_half = 0.0
        relief_fraction = _np_std.zeros_like(y_half, dtype=float)
        if target_relief_half > 1e-9 and _np_std.any(relief_shape > 0.0):
            def relieved_half_lift(scale: float) -> float:
                trial_fraction = _np_std.minimum(cap_fraction, scale * relief_shape)
                return float(_np_std.trapezoid(trial_fraction * base_lift, y_half))

            max_possible_half = relieved_half_lift(1.0e6)
            applied_relief_half = min(target_relief_half, max_possible_half)
            scale_lo = 0.0
            scale_hi = 1.0
            while relieved_half_lift(scale_hi) < applied_relief_half and scale_hi < 1.0e6:
                scale_hi *= 2.0
            for _ in range(60):
                scale_mid = 0.5 * (scale_lo + scale_hi)
                if relieved_half_lift(scale_mid) >= applied_relief_half:
                    scale_hi = scale_mid
                else:
                    scale_lo = scale_mid
            relief_fraction = _np_std.minimum(cap_fraction, scale_hi * relief_shape)
            applied_relief_half = float(_np_std.trapezoid(relief_fraction * base_lift, y_half))
            relief_lift_per_span = -relief_fraction * base_lift

        applied_delta = cambered_lift_per_span + relief_lift_per_span

        cambered_half_lift = float(_np_std.trapezoid(cambered_lift_per_span, y_half))
        relief_half_lift = -applied_relief_half
        applied_half_lift = float(_np_std.trapezoid(applied_delta, y_half))
        q_s = max(q * s_ref_m2, 1e-9)

        return {
            "delta_lift_per_span_half": applied_delta,
            "body_strip_rows": body_strip_rows,
            "pressure_proxy": float(pressure_proxy),
            "symmetry_factor": float(symmetry_factor),
            "body_cl0_proxy": float(body_cl0_proxy),
            "lift_retention": float(lift_retention),
            "cambered_delta_cl": 2.0 * cambered_half_lift / q_s,
            "relief_delta_cl": 2.0 * relief_half_lift / q_s,
            "delta_cl_total": 2.0 * applied_half_lift / q_s,
            "variant_relief_region_end_m": float(y_end),
            "variant_relief_cap_fraction": float(cap_fraction),
            "variant_target_scale": float(relief_target_scale),
            "variant_shape_integral": float(_np_std.trapezoid(relief_shape, y_half)),
        }

    def _build_body_spanload_weight(
        self,
        y_half: np.ndarray,
        bias_power: float = 0.0,
    ) -> np.ndarray:
        sections = sorted(self.wing_project.planform.body_sections, key=lambda section: section.y_pos)
        if len(sections) < 2:
            return np.zeros_like(y_half)

        body_y = _np_std.asarray([float(section.y_pos) for section in sections], dtype=float)
        body_chord = _np_std.asarray([float(section.chord) for section in sections], dtype=float)
        max_body_y = max(float(body_y[-1]), 1e-9)
        weights = _np_std.zeros_like(y_half, dtype=float)
        in_body_mask = y_half <= max_body_y
        if _np_std.any(in_body_mask):
            base_weight = _np_std.interp(y_half[in_body_mask], body_y, body_chord)
            if bias_power > 0.0:
                root_bias = _np_std.clip(1.0 - y_half[in_body_mask] / max_body_y, 0.0, 1.0) ** bias_power
                base_weight = base_weight * (0.35 + 0.65 * root_bias)
            weights[in_body_mask] = base_weight

        integral = float(_np_std.trapezoid(weights, y_half))
        if integral <= 1e-9:
            return _np_std.zeros_like(y_half)
        return weights / integral

    @staticmethod
    def _preserve_root_high_spanload_shape(
        y_half: np.ndarray,
        lift_per_span_half: np.ndarray,
        target_half_lift: float,
    ) -> np.ndarray:
        """
        Remove artificial root/center dips from a half-span structural load.

        Bell and elliptical structural loads should be highest at the centerline
        and decay outboard. Pressure-relief diagnostics can otherwise subtract
        from the root bin and create a nonphysical local valley that drives the
        beam with a shape the user did not request.
        """
        y_half = _np_std.asarray(y_half, dtype=float)
        adjusted = _np_std.asarray(lift_per_span_half, dtype=float).copy()
        if len(adjusted) < 2:
            return adjusted

        for idx in range(len(adjusted) - 2, -1, -1):
            if adjusted[idx] < adjusted[idx + 1]:
                adjusted[idx] = adjusted[idx + 1]

        current_half_lift = float(_np_std.trapezoid(adjusted, y_half))
        if abs(current_half_lift) > 1e-9 and abs(target_half_lift) > 1e-9:
            adjusted *= float(target_half_lift) / current_half_lift
        return adjusted

    def _guard_structural_bwb_spanload_shape(
        self,
        y_half: np.ndarray,
        lift_per_span_half: np.ndarray,
    ) -> np.ndarray:
        plan = self.wing_project.planform
        if len(plan.body_sections) < 2:
            return lift_per_span_half
        lift_dist_type = str(getattr(self.wing_project.twist_trim, "lift_distribution", "") or "").lower()
        if lift_dist_type not in {"bell", "elliptical"}:
            return lift_per_span_half
        target_half_lift = float(_np_std.trapezoid(lift_per_span_half, y_half))
        return self._preserve_root_high_spanload_shape(
            y_half=y_half,
            lift_per_span_half=lift_per_span_half,
            target_half_lift=target_half_lift,
        )

    def _apply_blind_hybrid_body_spanload(
        self,
        y_half: np.ndarray,
        lift_per_span_half: np.ndarray,
        velocity: float,
        alpha: float,
        load_factor: float,
        altitude_m: Optional[float],
        q: float,
        s_ref_m2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.wing_project.planform.body_sections) < 2:
            return y_half, lift_per_span_half
        if len(y_half) < 2 or q <= 0.0 or s_ref_m2 <= 0.0:
            return y_half, lift_per_span_half

        body_proxies = self._build_body_geometry_proxies(s_ref_m2=s_ref_m2)
        if body_proxies["body_area_ratio"] <= 0.0:
            return y_half, lift_per_span_half

        pressure_proxy = self._compute_spanload_pressure_proxy(
            y_half=y_half,
            lift_per_span_half=lift_per_span_half,
            velocity=velocity,
            q=q,
            altitude_m=altitude_m,
            s_ref_m2=s_ref_m2,
        )
        body_cl0_proxy = self._compute_body_cl0_proxy(
            velocity=velocity,
            altitude_m=altitude_m,
            s_ref_m2=s_ref_m2,
        )
        symmetry_factor = self._compute_body_symmetry_factor(body_cl0_proxy, body_proxies)
        blind_parameters = self._build_blind_body_model_parameters(body_proxies)
        distributed_delta = self._build_distributed_blind_body_lift_delta(
            y_half=y_half,
            lift_per_span_half=lift_per_span_half,
            velocity=velocity,
            alpha=alpha,
            load_factor=load_factor,
            altitude_m=altitude_m,
            q=q,
            s_ref_m2=s_ref_m2,
            blind_parameters=blind_parameters,
        )
        zero = _np_std.zeros_like(lift_per_span_half)
        delta_lift_per_span = _np_std.asarray(distributed_delta.get("delta_lift_per_span_half", zero), dtype=float)
        if not _np_std.any(_np_std.abs(delta_lift_per_span) > 1e-12):
            return y_half, self._guard_structural_bwb_spanload_shape(y_half, lift_per_span_half)
        adjusted = _np_std.asarray(lift_per_span_half, dtype=float) + delta_lift_per_span
        adjusted = self._guard_structural_bwb_spanload_shape(y_half, adjusted)

        print(
            "[AeroStructural] Blind hybrid BWB spanload applied "
            f"(dCL={float(distributed_delta.get('delta_cl_total', 0.0) or 0.0):+.4f}, "
            f"pressure_proxy={float(distributed_delta.get('pressure_proxy', 0.0) or 0.0):.4f}, "
            f"symmetry={float(distributed_delta.get('symmetry_factor', 0.0) or 0.0):.3f})"
        )
        return y_half, adjusted

    def visualize_flow(self, spanwise_resolution: int = 10, chordwise_resolution: int = 10) -> None:
        """
        Runs a VLM analysis and opens a 3D visualization window showing pressure distribution and streamlines.
        """
        # 1. Build Airplane
        wing = self.build_wing()
        
        # Calculate CG
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        # 2. Define Operating Point (Cruise)
        # Determine alpha
        if self.wing_project.optimized_twist_deg is not None:
            alpha_cruise = 0.0
        else:
            alpha_cruise = self.estimate_alpha_for_cl()
        
        # Get actual CL at cruise alpha to calculate true cruise velocity
        v_ref = self.design_velocity()
        op_point_ref = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=v_ref,
            alpha=alpha_cruise,
        )
        analysis_ref = asb.AeroBuildup(
            airplane=asb.Airplane(
                name=self.wing_project.name,
                wings=[wing],
                xyz_ref=xyz_ref,
            ),
            op_point=op_point_ref,
        )
        res_ref = analysis_ref.run()
        actual_cruise_cl = self._coerce_value(res_ref.get("CL", res_ref.get("Cl", 0.0)))
        
        # Calculate TRUE cruise velocity using actual CL
        if actual_cruise_cl > 1e-4:
            v_cruise = self.velocity_for_cl(actual_cruise_cl)
        else:
            v_cruise = v_ref
            
        # 3. Run VLM Analysis
        # AeroBuildup doesn't support draw() with streamlines in the same way VLM does
        vlm, _, vlm_info = self._run_stable_vlm(
            velocity=v_cruise,
            alpha=alpha_cruise,
            xyz_ref=xyz_ref,
            spanwise_resolution_hint=spanwise_resolution,
            chordwise_resolution_hint=chordwise_resolution,
        )
        if vlm_info["filtered_section_count"] < vlm_info["original_section_count"]:
            print(
                "[AeroStructural] Visualizer using VLM-safe section filter "
                f"({vlm_info['filtered_section_count']}/{vlm_info['original_section_count']} sections, "
                f"min dy={vlm_info['min_section_spacing_m']:.4f} m)"
            )
        
        # 4. Visualize
        # show_kwargs argument allows customizing the PyVista plotter if needed
        vlm.draw(
            show=True,
            show_kwargs={
                "jupyter_backend": "static" # Prevents issues in some environments, though 'interactive' is default
            }
        )

    def get_spanwise_lift_distribution(
        self,
        velocity: Optional[float] = None,
        alpha: Optional[float] = None,
        load_factor: float = 1.0,
        n_spanwise_points: int = 50,
        use_vlm: bool = True,
        altitude_m: Optional[float] = None,
        analysis_method: str = "vlm",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the spanwise lift distribution using aerodynamic analysis.
        
        Tries VLM first (provides actual spanwise resolution), falls back to 
        AeroBuildup with elliptical assumption if VLM fails.
        
        Args:
            velocity: Flight velocity [m/s]. If None, uses design velocity.
            alpha: Angle of attack [deg]. If None, estimates from design CL.
            load_factor: Load factor multiplier for the lift.
            n_spanwise_points: Number of points for output distribution.
            use_vlm: If True, try VLM first. If False, skip to AeroBuildup.
            altitude_m: Optional analysis altitude [m]. Defaults to cruise altitude.
            analysis_method: Spanload method. "vlm" uses the raw aerodynamic distribution.
                "blind_hybrid_bwb_body" applies the blind BWB body-lift correction used by
                the alternate low-order structural path.
        
        Returns:
            Tuple of (y_positions [m], lift_per_span [N/m]) for half-wing.
            y_positions are from 0 (root) to half_span (tip).
        """
        import numpy as _np_std
        
        # Get flight conditions
        if velocity is None:
            velocity = self.design_velocity()
        if alpha is None:
            # If twist is optimized for Cm=0 at alpha=0, use alpha=0
            if self.wing_project.optimized_twist_deg is not None:
                alpha = 0.0
            else:
                alpha = self.estimate_alpha_for_cl()
        
        # Build wing and airplane
        wing = self.build_wing()
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = self.wing_project.twist_trim.static_margin_percent
        x_cg = x_np - (static_margin / 100.0) * mac
        xyz_ref = [x_cg, 0.0, 0.0]
        
        airplane = asb.Airplane(
            name=self.wing_project.name,
            wings=[wing],
            xyz_ref=xyz_ref,
        )
        analysis_atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        op_point = asb.OperatingPoint(
            atmosphere=analysis_atmosphere,
            velocity=velocity,
            alpha=alpha,
        )
        q = float(op_point.dynamic_pressure())
        
        # Calculate total half-span (including BWB body if present)
        plan = self.wing_project.planform
        wing_half_span = plan.half_span()
        bwb_mode = len(plan.body_sections) > 0
        if bwb_mode:
            sorted_body_sections = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else 0.0
            half_span = bwb_outer_y + wing_half_span
        else:
            half_span = wing_half_span
        
        y_out = _np_std.linspace(0, half_span, n_spanwise_points)
        
        # Try VLM first - it gives us actual panel-level lift distribution
        if use_vlm:
            try:
                vlm, aero_result, vlm_info = self._run_stable_vlm(
                    velocity=velocity,
                    alpha=alpha,
                    xyz_ref=xyz_ref,
                    altitude_m=altitude_m,
                    spanwise_resolution_hint=min(8, max(4, n_spanwise_points // 16)),
                    chordwise_resolution_hint=3,
                )
                total_lift_vlm = float(aero_result.get('L', 0))
                half_lift_vlm = total_lift_vlm / 2.0  # For symmetric wing
                if vlm_info["filtered_section_count"] < vlm_info["original_section_count"]:
                    print(
                        "[AeroStructural] VLM section filter applied "
                        f"({vlm_info['filtered_section_count']}/{vlm_info['original_section_count']} sections, "
                        f"min dy={vlm_info['min_section_spacing_m']:.4f} m)"
                    )
                print(
                    "[AeroStructural] VLM panel settings "
                    f"sr={vlm_info['spanwise_resolution']}, cr={vlm_info['chordwise_resolution']}"
                )
                
                # Extract panel forces to determine the SHAPE of the distribution
                forces_g = _np_std.array(vlm.forces_geometry)
                vortex_centers = _np_std.array(vlm.vortex_centers)
                left_vertices = _np_std.array(vlm.left_vortex_vertices)
                right_vertices = _np_std.array(vlm.right_vortex_vertices)
                
                # Get data for positive y side only
                positive_y_mask = vortex_centers[:, 1] >= 0
                panel_y = vortex_centers[positive_y_mask, 1]
                panel_fz = _np_std.abs(forces_g[positive_y_mask, 2])
                panel_widths = _np_std.abs(right_vertices[positive_y_mask, 1] - left_vertices[positive_y_mask, 1])
                
                # Use histogram-based binning to aggregate chordwise panels
                # This is more robust than tolerance-based grouping
                n_bins = int(vlm_info["spanwise_resolution"]) + 1
                bin_edges = _np_std.linspace(0, half_span * 1.001, n_bins + 1)  # Slightly beyond tip
                
                # For each bin, sum forces and get representative width
                bin_y = []
                bin_lift_per_span = []
                
                for i in range(n_bins):
                    mask = (panel_y >= bin_edges[i]) & (panel_y < bin_edges[i + 1])
                    if _np_std.any(mask):
                        # Sum all forces in this bin
                        total_force = _np_std.sum(panel_fz[mask])
                        # Use mean y position
                        mean_y = _np_std.mean(panel_y[mask])
                        # Use mean width (should be same for all chordwise panels)
                        mean_width = _np_std.mean(panel_widths[mask])
                        
                        if mean_width > 1e-10:
                            lift_per_span = total_force / mean_width
                            bin_y.append(mean_y)
                            bin_lift_per_span.append(lift_per_span)
                
                if len(bin_y) >= 2:
                    bin_y = _np_std.array(bin_y)
                    bin_lift_per_span = _np_std.array(bin_lift_per_span)
                    
                    # Sort by y (should already be sorted, but be safe)
                    sort_idx = _np_std.argsort(bin_y)
                    bin_y = bin_y[sort_idx]
                    bin_lift_per_span = bin_lift_per_span[sort_idx]
                    
                    # Interpolate to output points
                    shape_interp = _np_std.interp(y_out, bin_y, bin_lift_per_span)
                    
                    # Normalize so integral matches half_lift_vlm
                    shape_integral = _np_std.trapz(shape_interp, y_out)
                    if abs(shape_integral) > 1e-10:
                        scale_factor = half_lift_vlm / shape_integral
                    else:
                        scale_factor = 1.0
                    
                    lift_per_span = shape_interp * scale_factor * load_factor
                    
                    total_half_lift = _np_std.trapz(lift_per_span, y_out)
                    
                    print(f"[AeroStructural] VLM lift distribution computed successfully")
                    print(f"  Total half-wing lift: {total_half_lift:.1f} N (expected: {half_lift_vlm * load_factor:.1f} N)")

                    if analysis_method == "blind_hybrid_bwb_body":
                        return self._apply_blind_hybrid_body_spanload(
                            y_half=y_out,
                            lift_per_span_half=lift_per_span,
                            velocity=velocity,
                            alpha=alpha,
                            load_factor=load_factor,
                            altitude_m=altitude_m,
                            q=q,
                            s_ref_m2=plan.actual_area(),
                        )
                    return y_out, self._guard_structural_bwb_spanload_shape(y_out, lift_per_span)
                    
            except Exception as e:
                print(f"[AeroStructural] VLM failed: {e}, falling back to AeroBuildup")
        
        # Fallback: Use AeroBuildup to get total lift, then apply configured distribution
        try:
            analysis = asb.AeroBuildup(
                airplane=airplane,
                op_point=op_point,
            )
            results = analysis.run()
            
            # Get total lift
            total_lift = float(results.get("L", 0))
            
            # Sanity check: lift should be positive for a flying aircraft
            # If AeroBuildup returns negative/zero lift, fall through to weight-based fallback
            if total_lift <= 0:
                print(f"[AeroStructural] AeroBuildup returned non-positive lift ({total_lift:.1f} N)")
                print(f"  This may indicate the wing is producing downforce at alpha={alpha:.1f}°")
                print(f"  Falling back to weight-based lift estimate")
                raise ValueError(f"Non-positive lift from AeroBuildup: {total_lift:.1f} N")
            
            half_lift = total_lift / 2.0  # For half-wing (symmetric)
            
            # Apply load factor
            half_lift = half_lift * load_factor
            
            # Get lift distribution type from project settings
            lift_dist_type = getattr(self.wing_project.twist_trim, 'lift_distribution', 'elliptical')
            
            # Create distribution shape based on setting
            eta = y_out / half_span
            eta = _np_std.clip(eta, 0, 0.999)
            
            if lift_dist_type == "bell":
                # Bell distribution (Prandtl-D): (1 - eta²)^1.5
                shape = (1 - eta ** 2) ** 1.5
                # Integral of (1-x²)^1.5 from 0 to 1 = 3π/16
                integral_factor = 3 * _np_std.pi / 16
            else:
                # Elliptical distribution: sqrt(1 - eta²)
                shape = _np_std.sqrt(1 - eta ** 2)
                # Integral of sqrt(1-x²) from 0 to 1 = π/4
                integral_factor = _np_std.pi / 4
            
            # Scale to match total lift
            q0 = half_lift / (half_span * integral_factor)
            lift_per_span = q0 * shape
            
            print(f"[AeroStructural] AeroBuildup {lift_dist_type} distribution computed")
            print(f"  Total half-wing lift: {_np_std.trapz(lift_per_span, y_out):.1f} N")

            if analysis_method == "blind_hybrid_bwb_body":
                return self._apply_blind_hybrid_body_spanload(
                    y_half=y_out,
                    lift_per_span_half=lift_per_span,
                    velocity=velocity,
                    alpha=alpha,
                    load_factor=load_factor,
                    altitude_m=altitude_m,
                    q=q,
                    s_ref_m2=plan.actual_area(),
                )
            return y_out, self._guard_structural_bwb_spanload_shape(y_out, lift_per_span)
            
        except Exception as e:
            print(f"[AeroStructural] AeroBuildup also failed: {e}")
            # Ultimate fallback: weight-based distribution using configured type
            # This assumes lift = weight for level flight
            total_weight = self.wing_project.twist_trim.gross_takeoff_weight_kg * g
            half_lift = total_weight * load_factor / 2.0
            
            print(f"[AeroStructural] Using weight-based fallback:")
            print(f"  GTW: {self.wing_project.twist_trim.gross_takeoff_weight_kg:.2f} kg")
            print(f"  Load factor: {load_factor:.1f}")
            print(f"  Half-wing lift: {half_lift:.1f} N")
            
            # Get lift distribution type from project settings
            lift_dist_type = getattr(self.wing_project.twist_trim, 'lift_distribution', 'elliptical')
            
            eta = y_out / half_span
            eta = _np_std.clip(eta, 0, 0.999)
            
            if lift_dist_type == "bell":
                shape = (1 - eta ** 2) ** 1.5
                integral_factor = 3 * _np_std.pi / 16
            else:
                shape = _np_std.sqrt(1 - eta ** 2)
                integral_factor = _np_std.pi / 4
            
            q0 = half_lift / (half_span * integral_factor)
            lift_per_span = q0 * shape
            
            print(f"[AeroStructural] Using weight-based {lift_dist_type} fallback")
            if analysis_method == "blind_hybrid_bwb_body":
                return self._apply_blind_hybrid_body_spanload(
                    y_half=y_out,
                    lift_per_span_half=lift_per_span,
                    velocity=velocity,
                    alpha=alpha,
                    load_factor=load_factor,
                    altitude_m=altitude_m,
                    q=q,
                    s_ref_m2=plan.actual_area(),
                )
            return y_out, self._guard_structural_bwb_spanload_shape(y_out, lift_per_span)

    def get_spanwise_moment_distribution(
        self,
        velocity: Optional[float] = None,
        alpha: Optional[float] = None,
        load_factor: float = 1.0,
        n_spanwise_points: int = 50,
        altitude_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the spanwise pitching moment distribution for torsion analysis.
        
        For a flying wing, the pitching moment comes from:
        1. Airfoil Cm0 (from reflex airfoils)
        2. Lift acting at offset from shear center (sweep effect)
        
        This is a simplified estimation based on sectional properties.
        
        Args:
            velocity: Flight velocity [m/s]. If None, uses design velocity.
            alpha: Angle of attack [deg]. If None, estimates from design CL.
            load_factor: Load factor multiplier.
            n_spanwise_points: Number of points for output distribution.
        
        Returns:
            Tuple of (y_positions [m], moment_per_span [N*m/m]) for half-wing.
            Moment is the sectional pitching moment per unit span about the
            quarter-chord (local aerodynamic center).
        """
        import numpy as _np_std
        
        # Get flight conditions
        if velocity is None:
            velocity = self.design_velocity()
        if alpha is None:
            if self.wing_project.optimized_twist_deg is not None:
                alpha = 0.0
            else:
                alpha = self.estimate_alpha_for_cl()
        
        planform = self.wing_project.planform
        
        # Calculate half-span
        wing_half_span = planform.half_span()
        bwb_mode = len(planform.body_sections) > 0
        if bwb_mode:
            sorted_body_sections = sorted(planform.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else 0.0
            half_span = bwb_outer_y + wing_half_span
        else:
            half_span = wing_half_span
        
        y_out = _np_std.linspace(0, half_span, n_spanwise_points)
        
        # Get sections for chord distribution
        sections = self.spanwise_sections()
        sec_y = _np_std.array([abs(sec.y_m) for sec in sections])
        sec_chord = _np_std.array([sec.chord_m for sec in sections])
        
        # Sort by y position
        sort_idx = _np_std.argsort(sec_y)
        sec_y = sec_y[sort_idx]
        sec_chord = sec_chord[sort_idx]
        
        # Interpolate chord to output points
        chord_interp = _np_std.interp(y_out, sec_y, sec_chord)
        
        analysis_atmosphere = asb.Atmosphere(
            altitude=self.wing_project.twist_trim.cruise_altitude_m if altitude_m is None else altitude_m
        )
        # Dynamic pressure
        rho = analysis_atmosphere.density()
        q = 0.5 * rho * velocity**2
        
        # Pitching moment coefficient (Cm0 for flying wing airfoils is typically -0.05 to -0.10)
        # Negative Cm0 = nose-down moment
        # For reflex airfoils used in flying wings, Cm0 is near zero or slightly positive
        cm0 = getattr(planform, 'airfoil_cm0', -0.02)  # Default: slightly nose-down
        
        # Sectional pitching moment per unit span: m = q * c^2 * Cm0
        # Note: This is moment per span length, in [N*m/m]
        moment_per_span = q * chord_interp**2 * cm0 * load_factor
        
        return y_out, moment_per_span

    def run_aerostructural_analysis(
        self,
        flight_condition: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run coupled aerostructural analysis.
        
        Steps:
        1. Get lift distribution from aerodynamic analysis
        2. Build structural model from geometry
        3. Solve structural response
        4. Return combined aero + structural results
        
        Args:
            flight_condition: Optional dict with 'velocity_mps', 'altitude_m', 'load_factor'
        
        Returns:
            Dictionary with keys:
            - 'structure': Structural results from analysis
            - 'feasible': Boolean indicating if all constraints are met
            - 'error': Error message if analysis failed (optional)
        """
        try:
            from services.structure import (
                WingBoxSection, 
                analyze_wingbox_beam,
                StringerProperties,
                RibProperties,
                ControlSurfaceProperties,
                PostBucklingConfig,
            )
            from core.models.materials import get_material_by_name
        except ImportError as e:
            return {
                "structure": {},
                "feasible": False,
                "error": f"Missing dependencies: {e}",
            }
        
        planform = self.wing_project.planform
        twist = self.wing_project.twist_trim
        
        # Default flight condition
        if flight_condition is None:
            flight_condition = {
                "velocity_mps": getattr(twist, 'cruise_speed_mps', 25.0) if hasattr(twist, 'cruise_speed_mps') else 25.0,
                "altitude_m": 0.0,
                "load_factor": 2.5,
            }
        flight_condition = self.resolve_structural_flight_condition(flight_condition)
        
        load_factor = flight_condition.get("load_factor", 2.5)
        
        try:
            # Step 1: Get spanwise sections for geometry
            sections = self.spanwise_sections()
            
            if len(sections) < 2:
                return {
                    "structure": {},
                    "feasible": False,
                    "error": "Need at least 2 spanwise sections for structural analysis",
                }
            
            # Step 2: Build WingBoxSection list
            # Calculate total half-span (including BWB body if present)
            wing_half_span = planform.half_span()
            bwb_mode = len(planform.body_sections) > 0
            if bwb_mode:
                sorted_body_sections = sorted(planform.body_sections, key=lambda bs: bs.y_pos)
                bwb_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else 0.0
                half_span = bwb_outer_y + wing_half_span
            else:
                half_span = wing_half_span
            
            wing_box_sections = []
            
            for sec in sections:
                y_pos = sec.y_m  # Spanwise position
                chord = sec.chord_m
                
                # Get thickness ratio from airfoil (approximate)
                t_over_c = 0.12  # Default assumption
                if hasattr(sec, 'airfoil') and sec.airfoil is not None:
                    try:
                        t_over_c = sec.airfoil.max_thickness()
                    except Exception:
                        pass
                
                # Interpolate spar positions (linear from root to tip)
                eta = abs(y_pos) / half_span if half_span > 0 else 0
                front_spar = (
                    planform.front_spar_root_percent / 100 * (1 - eta) +
                    planform.front_spar_tip_percent / 100 * eta
                )
                rear_spar = planform.rear_spar_root_percent / 100
                
                wing_box_sections.append(WingBoxSection(
                    y=abs(y_pos),
                    chord=chord,
                    thickness_ratio=t_over_c,
                    front_spar_xsi=front_spar,
                    rear_spar_xsi=rear_spar,
                ))
            
            # Sort by spanwise position
            wing_box_sections.sort(key=lambda s: s.y)
            
            # Step 3: Get aerodynamic lift distribution from actual analysis
            # This uses VLM or AeroBuildup instead of assumed elliptical
            import numpy as _np_std
            from scipy.interpolate import interp1d
            
            try:
                y_aero, lift_per_span_aero = self.get_spanwise_lift_distribution(
                    velocity=flight_condition.get("velocity_mps"),
                    alpha=flight_condition.get("alpha_deg"),
                    altitude_m=flight_condition.get("altitude_m"),
                    load_factor=load_factor,
                    n_spanwise_points=100,
                    use_vlm=True,
                    analysis_method=getattr(twist, "structural_spanload_model", "vlm"),
                )
                
                # Create interpolation function for the structural solver
                # Ensure we handle edge cases (extrapolation at boundaries)
                lift_interp = interp1d(
                    y_aero, 
                    lift_per_span_aero, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(float(lift_per_span_aero[0]), 0.0)  # Root value at y<0, 0 at y>tip
                )
                
                def lift_distribution(y):
                    """Lift per unit span [N/m] as function of spanwise position."""
                    y_arr = _np_std.atleast_1d(y)
                    return lift_interp(y_arr)
                    
            except Exception as e:
                print(f"[AeroStructural] Could not compute aero lift distribution: {e}")
                
                # Fallback to configured distribution type
                lift_dist_type = getattr(twist, 'lift_distribution', 'elliptical')
                print(f"[AeroStructural] Falling back to {lift_dist_type} assumption")
                
                total_weight = twist.gross_takeoff_weight_kg * g
                total_lift = total_weight * load_factor
                half_lift = total_lift / 2.0
                
                from services.structure import create_lift_distribution
                lift_distribution = create_lift_distribution(
                    half_span=half_span,
                    total_lift_N=half_lift,
                    distribution_type=lift_dist_type,
                )
            
            # Step 4: Get materials
            spar_material = get_material_by_name(planform.spar_material_name)
            skin_material = get_material_by_name(planform.skin_material_name)
            
            # Step 5: Get rib positions from section spanwise positions
            # In a flying wing, ribs are typically placed at each section station
            rib_positions = sorted([abs(sec.y_m) for sec in sections])
            
            # Step 5b: Prepare stringer properties (if configured)
            stringer_props = None
            if getattr(planform, 'stringer_count', 0) > 0:
                stringer_material = get_material_by_name(
                    getattr(planform, 'stringer_material_name', planform.skin_material_name)
                )
                stringer_props = StringerProperties(
                    count=planform.stringer_count,
                    height_m=getattr(planform, 'stringer_height_mm', 10.0) / 1000,
                    thickness_m=getattr(planform, 'stringer_thickness_mm', 1.5) / 1000,
                    material=stringer_material,
                )
            
            # Step 5c: Get boundary condition and curvature effect settings
            boundary_condition = getattr(planform, 'skin_boundary_condition', 'semi_restrained')
            include_curvature = getattr(planform, 'include_curvature_effect', True)
            
            # Step 5d: Collect airfoil objects for curvature calculation
            airfoils = [sec.airfoil for sec in sections if hasattr(sec, 'airfoil')]
            
            # Step 5e: Prepare rib properties
            rib_material = get_material_by_name(
                getattr(planform, 'rib_material_name', planform.skin_material_name)
            )
            rib_props = RibProperties(
                thickness_m=getattr(planform, 'rib_thickness_mm', 3.0) / 1000,
                material=rib_material,
                lightening_hole_fraction=getattr(planform, 'rib_lightening_fraction', 0.4),
                spar_cap_width_m=getattr(planform, 'spar_cap_width_mm', 10.0) / 1000,
            )
            
            # Step 5f: Build control surface properties from planform
            control_surface_props_list = []
            if hasattr(planform, 'control_surfaces') and len(planform.control_surfaces) > 0:
                for cs in planform.control_surfaces:
                    # Convert from planform ControlSurface to structure ControlSurfaceProperties
                    cs_props = ControlSurfaceProperties(
                        span_start=cs.span_start_percent / 100.0,
                        span_end=cs.span_end_percent / 100.0,
                        chord_fraction_start=(100.0 - cs.chord_start_percent) / 100.0,
                        chord_fraction_end=(100.0 - cs.chord_end_percent) / 100.0,
                        # Use thinner skin for control surfaces (typically 60-70% of main skin)
                        skin_thickness_m=planform.skin_thickness_mm / 1000 * 0.65,
                        rib_thickness_m=planform.rib_thickness_mm / 1000 * 0.75,
                        rib_spacing_m=0.08,  # ~80mm rib spacing for control surfaces
                        hinge_spar_thickness_m=planform.spar_thickness_mm / 1000 * 0.5,
                        skin_material=skin_material,
                        rib_material=rib_material,
                        rib_lightening_fraction=0.3,
                    )
                    control_surface_props_list.append(cs_props)
            
            # Step 5g: Get fastener/adhesive fraction (default 10%)
            fastener_fraction = getattr(planform, 'fastener_adhesive_fraction', 0.10)
            
            # Step 5h: Build post-buckling config from planform
            post_buckling_config = None
            if getattr(planform, 'post_buckling_enabled', False):
                post_buckling_config = PostBucklingConfig(
                    enabled=True,
                    require_stringers=True,  # Stringers are required for post-buckling
                )
            
            # Step 5i: Get pitching moment distribution for torsion analysis
            moment_distribution = None
            try:
                y_moment, moment_per_span = self.get_spanwise_moment_distribution(
                    velocity=flight_condition.get("velocity_mps"),
                    alpha=flight_condition.get("alpha_deg"),
                    load_factor=load_factor,
                    n_spanwise_points=100,
                    altitude_m=flight_condition.get("altitude_m"),
                )
                
                # Create interpolation function
                moment_interp = interp1d(
                    y_moment,
                    moment_per_span,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(float(moment_per_span[0]), 0.0)
                )
                
                def moment_distribution(y):
                    """Pitching moment per unit span [N*m/m] as function of spanwise position."""
                    y_arr = _np_std.atleast_1d(y)
                    return moment_interp(y_arr)
                
                print(f"[AeroStructural] Moment distribution computed for torsion analysis")
                
            except Exception as e:
                print(f"[AeroStructural] Could not compute moment distribution: {e}")
                print(f"[AeroStructural] Torsion will not be included in analysis")
                moment_distribution = None
            
            # Step 6: Run structural analysis
            result = analyze_wingbox_beam(
                sections=wing_box_sections,
                spar_thickness=planform.spar_thickness_mm / 1000,  # Convert mm to m
                skin_thickness=planform.skin_thickness_mm / 1000,  # Convert mm to m
                spar_material=spar_material,
                skin_material=skin_material,
                lift_distribution=lift_distribution,
                moment_distribution=moment_distribution,
                rib_positions=rib_positions,
                factor_of_safety=planform.factor_of_safety,
                max_deflection_fraction=planform.max_tip_deflection_percent / 100,
                # Enhanced buckling analysis parameters
                stringer_props=stringer_props,
                rib_props=rib_props,
                boundary_condition=boundary_condition,
                include_curvature=include_curvature,
                airfoils=airfoils,
                # Control surface and fastener parameters
                control_surface_props=control_surface_props_list if control_surface_props_list else None,
                fastener_adhesive_fraction=fastener_fraction,
                # Post-buckling analysis
                post_buckling_config=post_buckling_config,
                # Twist constraint
                max_twist_deg=getattr(planform, 'max_tip_twist_deg', 3.0),
            )
            
            return {
                "structure": result.as_dict(),
                "feasible": result.is_feasible,
                "flight_condition": flight_condition,
            }
            
        except Exception as e:
            import traceback
            return {
                "structure": {},
                "feasible": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def build_deformed_wing(
        self,
        heave_displacements: Optional[np.ndarray] = None,
        twist_displacements: Optional[np.ndarray] = None,
        structural_result: Optional[Dict[str, Any]] = None,
    ) -> asb.Wing:
        """
        Build a wing with applied heave (vertical) and twist displacements.
        
        Uses the AeroSandbox make_wing pattern for proper handling of 
        deformed wing geometry including section rotations.
        
        Args:
            heave_displacements: Vertical displacement at each section [m].
                                 If None and structural_result provided, extracted from it.
            twist_displacements: Twist displacement at each section [deg].
                                 If None, defaults to zeros.
            structural_result: Optional dict from run_aerostructural_analysis().
                              If provided, extracts heave from 'displacement' field.
        
        Returns:
            AeroSandbox Wing object with deformed geometry.
        """
        import numpy as _np_std
        
        sections = self.spanwise_sections()
        n_sections = len(sections)
        
        if n_sections < 2:
            raise ValueError("Need at least 2 sections to build wing")
        
        # Calculate total half-span (including BWB if present)
        plan = self.wing_project.planform
        wing_half_span = plan.half_span()
        bwb_mode = len(plan.body_sections) > 0
        if bwb_mode:
            sorted_body_sections = sorted(plan.body_sections, key=lambda bs: bs.y_pos)
            bwb_outer_y = sorted_body_sections[-1].y_pos if sorted_body_sections else 0.0
            total_half_span = bwb_outer_y + wing_half_span
        else:
            total_half_span = wing_half_span
        
        # Extract section data
        ys_over_half_span = _np_std.array([s.span_fraction for s in sections])
        chords = _np_std.array([s.chord_m for s in sections])
        twists = _np_std.array([s.twist_deg for s in sections])
        x_les = _np_std.array([s.x_le_m for s in sections])
        z_positions = _np_std.array([s.z_m for s in sections])
        
        # Get or create displacement arrays
        if heave_displacements is None:
            if structural_result is not None:
                # Extract from structural result
                struct = structural_result.get("structure", {})
                y_struct = _np_std.array(struct.get("y", []))
                disp_struct = _np_std.array(struct.get("displacement", []))
                
                if len(y_struct) > 0 and len(disp_struct) > 0:
                    # Interpolate to section positions
                    section_y = _np_std.array([s.y_m for s in sections])
                    heave_displacements = _np_std.interp(section_y, y_struct, disp_struct)
                else:
                    heave_displacements = _np_std.zeros(n_sections)
            else:
                heave_displacements = _np_std.zeros(n_sections)
        
        if twist_displacements is None:
            # Could extract from structural result if slope data is available
            # For now, default to zeros (bending-only deformation)
            twist_displacements = _np_std.zeros(n_sections)
        
        # Shear center location (fraction of chord from LE)
        # Typical value for thin-walled closed sections
        x_ref_over_chord = 0.33
        
        # Build deformed wing sections
        xsecs = []
        for i in range(n_sections):
            section = sections[i]
            chord = chords[i]
            base_twist = twists[i]
            heave = float(heave_displacements[i])
            twist_disp = float(twist_displacements[i])
            
            # Compute deformed leading edge position
            # Start with base position
            xyz_le = _np_std.array([
                x_les[i],
                section.y_m,
                z_positions[i]
            ])
            
            # Apply heave displacement (vertical offset)
            xyz_le[2] += heave
            
            # Total twist = jig twist + elastic twist
            total_twist = base_twist + twist_disp
            
            xsecs.append(
                asb.WingXSec(
                    xyz_le=xyz_le.tolist(),
                    chord=chord,
                    twist=total_twist,
                    airfoil=section.airfoil,
                )
            )
        
        return asb.Wing(
            name="Deformed Flying Wing",
            xsecs=xsecs,
            symmetric=True,
        )

    def get_wing_pair_for_visualization(
        self,
        structural_result: Optional[Dict[str, Any]] = None,
        exaggeration_factor: float = 1.0,
    ) -> Tuple[asb.Wing, asb.Wing]:
        """
        Get a pair of wings for visualization: undeformed and deformed.
        
        Includes both heave (vertical displacement) and twist (from beam slope).
        The slope from beam bending creates an effective twist change at each section.
        
        Args:
            structural_result: Result from run_aerostructural_analysis()
            exaggeration_factor: Multiplier for displacements (for visualization clarity)
        
        Returns:
            Tuple of (undeformed_wing, deformed_wing)
        """
        import numpy as _np_std
        
        # Undeformed wing
        wing_undeformed = self.build_wing()
        
        # Get heave and twist displacements with optional exaggeration
        heave = None
        twist = None
        
        if structural_result is not None:
            struct = structural_result.get("structure", {})
            y_struct = _np_std.array(struct.get("y", []))
            disp_struct = _np_std.array(struct.get("displacement", []))
            slope_struct = _np_std.array(struct.get("slope", []))
            
            sections = self.spanwise_sections()
            section_y = _np_std.array([s.y_m for s in sections])
            
            # Heave displacement
            if len(y_struct) > 0 and len(disp_struct) > 0:
                # Apply exaggeration
                disp_exaggerated = disp_struct * exaggeration_factor
                heave = _np_std.interp(section_y, y_struct, disp_exaggerated)
            
            # Twist from beam slope
            # The slope (du/dy) from Euler-Bernoulli beam bending creates an 
            # effective change in angle of attack at each section.
            # For a swept wing, this is more complex, but for visualization
            # we treat it as a twist change in degrees.
            if len(y_struct) > 0 and len(slope_struct) > 0:
                # Convert slope [rad] to twist [deg] with exaggeration
                slope_exaggerated = slope_struct * exaggeration_factor
                slope_interp = _np_std.interp(section_y, y_struct, slope_exaggerated)
                # Slope du/dy gives rotation about the spanwise axis
                # This effectively changes the local angle of attack (twist)
                twist = _np_std.degrees(slope_interp)
        
        # Deformed wing
        wing_deformed = self.build_deformed_wing(
            heave_displacements=heave,
            twist_displacements=twist,
        )
        
        return wing_undeformed, wing_deformed


