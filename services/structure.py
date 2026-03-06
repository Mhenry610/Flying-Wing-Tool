# services/structure.py
"""
Wingbox beam structural analysis using AeroSandbox.
Based on Euler-Bernoulli beam theory with box cross-section.

Implements coupled aerostructural analysis where:
- Aerodynamic loads are computed from the current wing shape
- Structural deflections update the wing shape
- Iteration continues until convergence

Reference implementations:
- AeroSandbox TubeSparBendingStructure
- OpenAeroStruct wingbox model (oas_wb_mdolab.pdf)
"""
from __future__ import annotations
import aerosandbox as asb
import aerosandbox.numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Union, Tuple

from core.models.materials import StructuralMaterial, MaterialType

# Import buckling functions if available
try:
    from aerosandbox.structures.buckling import plate_buckling_critical_load
    BUCKLING_AVAILABLE = True
except ImportError:
    BUCKLING_AVAILABLE = False
    plate_buckling_critical_load = None


@dataclass
class WingBoxSection:
    """Cross-section geometry at a spanwise station."""
    y: float                      # Spanwise position [m]
    chord: float                  # Local chord [m]
    thickness_ratio: float        # t/c at this station
    front_spar_xsi: float         # Front spar position (fraction of chord)
    rear_spar_xsi: float          # Rear spar position (fraction of chord)
    
    @property
    def box_width(self) -> float:
        """Width of the wingbox (distance between spars)."""
        return self.chord * (self.rear_spar_xsi - self.front_spar_xsi)
    
    @property
    def box_height(self) -> float:
        """Height of the wingbox (local airfoil thickness at spar region)."""
        # Approximate: use t/c ratio. More accurate would sample airfoil at spar locations.
        return self.chord * self.thickness_ratio * 0.9  # 90% of max thickness at spar region


@dataclass
class StringerProperties:
    """Stringer (longitudinal stiffener) properties for smeared stiffener analysis.
    
    Supports multiple stringer section types:
    - "rectangular": Simple rectangular strip (balsa strip glued on edge)
    - "L_section": L-shaped stringer (flange + web)
    - "T_section": T-shaped stringer (centered flange + web)
    - "hat": Hat section stringer (for composite construction)
    
    For crippling analysis, the critical element is typically the outstanding
    flange (for L/T sections) or the web (for rectangular sections).
    """
    count: int = 0                    # Number of stringers per skin panel
    height_m: float = 0.010           # Stringer height [m]
    thickness_m: float = 0.0015       # Stringer web thickness [m]
    material: Optional[StructuralMaterial] = None
    
    # Stringer section type for crippling analysis
    section_type: str = "rectangular"  # "rectangular", "L_section", "T_section", "hat"
    
    # Flange properties (for L, T, hat sections)
    flange_width_m: float = 0.0       # Flange width [m] (0 = no flange, use height as web)
    flange_thickness_m: float = 0.0   # Flange thickness [m] (0 = same as web thickness)
    
    @property
    def area(self) -> float:
        """Cross-sectional area of one stringer [m²]."""
        if self.section_type == "rectangular":
            # Simple rectangular strip
            return self.height_m * self.thickness_m
        elif self.section_type in ("L_section", "T_section"):
            # Web + flange
            web_area = self.height_m * self.thickness_m
            flange_t = self.flange_thickness_m if self.flange_thickness_m > 0 else self.thickness_m
            flange_area = self.flange_width_m * flange_t
            return web_area + flange_area
        else:
            # Default to rectangular
            return self.height_m * self.thickness_m
    
    @property
    def I_self(self) -> float:
        """Moment of inertia of one stringer about its own centroid [m⁴]."""
        # For rectangular section: I = t*h³/12
        h, t = self.height_m, self.thickness_m
        return t * h**3 / 12
    
    @property
    def crippling_b_over_t(self) -> float:
        """Critical b/t ratio for crippling analysis.
        
        Returns the width-to-thickness ratio of the most critical element
        (outstanding flange for L/T, web for rectangular).
        """
        if self.section_type == "rectangular":
            # Web is the critical element (edge-supported on one side)
            return self.height_m / (self.thickness_m + 1e-10)
        elif self.section_type in ("L_section", "T_section"):
            # Outstanding flange is typically critical
            if self.flange_width_m > 0:
                flange_t = self.flange_thickness_m if self.flange_thickness_m > 0 else self.thickness_m
                return self.flange_width_m / (flange_t + 1e-10)
            else:
                # No flange defined, use web
                return self.height_m / (self.thickness_m + 1e-10)
        else:
            return self.height_m / (self.thickness_m + 1e-10)


@dataclass
class RibProperties:
    """Rib properties for mass calculation and failure analysis."""
    thickness_m: float = 0.003        # Rib thickness [m]
    material: Optional[StructuralMaterial] = None
    lightening_hole_fraction: float = 0.4  # Fraction of material removed (0-0.6 typical)
    spar_cap_width_m: float = 0.010   # Width of spar cap bearing on rib [m] (for crushing check)
    boundary_condition: str = "simply_supported"  # Rib edge support for buckling


@dataclass
class ControlSurfaceProperties:
    """Control surface (elevon/flap) properties for mass calculation.
    
    Control surfaces are modeled as:
    - Top and bottom skins (specified thickness)
    - Hinge rib at inboard edge
    - Tip rib at outboard edge
    - Internal ribs at regular spacing
    - Leading edge spar along hinge line
    
    Mass is computed based on planform area and construction details.
    """
    # Span extent (as fraction of half-span, 0=root, 1=tip)
    span_start: float = 0.4           # eta start (e.g., 0.4 = 40% span)
    span_end: float = 1.0             # eta end (e.g., 1.0 = tip)
    
    # Chord extent (as fraction of local chord, from hinge to trailing edge)
    chord_fraction_start: float = 0.25  # At span_start (25% = hinge at 75% chord)
    chord_fraction_end: float = 0.25    # At span_end
    
    # Structural properties
    skin_thickness_m: float = 0.001   # Skin thickness [m] (typically thinner than main wing)
    rib_thickness_m: float = 0.002    # Internal rib thickness [m]
    rib_spacing_m: float = 0.1        # Spacing between internal ribs [m]
    hinge_spar_thickness_m: float = 0.002  # Hinge line spar thickness [m]
    
    # Material (if None, uses main wing skin material)
    skin_material: Optional[StructuralMaterial] = None
    rib_material: Optional[StructuralMaterial] = None
    
    # Lightening
    rib_lightening_fraction: float = 0.3  # Fraction of rib material removed
    
    @property
    def span_fraction(self) -> float:
        """Span extent as fraction of half-span."""
        return self.span_end - self.span_start
    
    def chord_fraction_at_eta(self, eta: float) -> float:
        """Interpolate chord fraction at given spanwise position (0-1 within control surface span)."""
        return self.chord_fraction_start + (self.chord_fraction_end - self.chord_fraction_start) * eta


# =============================================================================
# TIMOSHENKO BEAM THEORY IMPLEMENTATION
# =============================================================================
# Timoshenko beam theory accounts for shear deformation, which is significant
# for thick, short beams (L/h < 15) common in flying wing structures.
# 
# Key differences from Euler-Bernoulli:
# - Bending rotation (ψ) and deflection (w) are independent variables
# - Shear strain γ = V / (κGA) contributes to total slope
# - Total slope: dw/dy = ψ + γ (bending rotation + shear strain)
#
# For slender beams, Timoshenko reduces to Euler-Bernoulli automatically.
# =============================================================================

@dataclass
class TimoshenkoBeamState:
    """State variables for Timoshenko beam solution at each station.
    
    Timoshenko beam theory treats bending rotation (ψ) and deflection (w) as
    independent variables, allowing proper capture of shear deformation effects.
    
    The key difference from Euler-Bernoulli is the shear strain term:
        Total slope = ψ + γ = ψ + V/(κGA)
    
    For Euler-Bernoulli (infinite shear stiffness): γ → 0, slope = ψ
    """
    y: np.ndarray           # Spanwise positions [m]
    w: np.ndarray           # Transverse deflection [m]
    psi: np.ndarray         # Bending rotation [rad] (NOT total slope)
    V: np.ndarray           # Shear force [N]
    M: np.ndarray           # Bending moment [N·m]
    gamma: np.ndarray       # Shear strain [rad]
    
    @property
    def total_slope(self) -> np.ndarray:
        """Total slope = bending rotation + shear strain."""
        return self.psi + self.gamma
    
    @property
    def curvature(self) -> np.ndarray:
        """Curvature (derivative of bending rotation) [1/m]."""
        import numpy as _np
        dy = _np.diff(self.y)
        # Central difference for curvature
        dpsi = _np.diff(self.psi)
        kappa = _np.zeros_like(self.y)
        kappa[:-1] = dpsi / dy
        kappa[-1] = kappa[-2]  # Extrapolate last point
        return kappa


def compute_timoshenko_shear_stiffness(
    box_height: float,
    box_width: float,
    t_spar: float,
    t_skin: float,
    spar_material: StructuralMaterial,
    skin_material: StructuralMaterial,
) -> tuple:
    """
    Compute effective shear stiffness (κGA) for Timoshenko beam.
    
    For a closed wingbox section, vertical shear is primarily carried by
    the spar webs. The effective shear stiffness depends on:
    - Shear modulus of the web material (G_xz for orthotropic)
    - Web area (2 * height * thickness for both spars)
    - Shear correction factor κ (accounts for non-uniform shear distribution)
    
    Args:
        box_height: Height of wingbox [m]
        box_width: Width of wingbox [m]  
        t_spar: Spar web thickness [m]
        t_skin: Skin thickness [m]
        spar_material: Material for spar webs
        skin_material: Material for skins
    
    Returns:
        (kappa_GA, kappa): Effective shear stiffness [N] and correction factor
    
    Notes:
        For composites, G_xz (transverse shear modulus) is often much lower than
        the in-plane shear modulus G_12, making shear deformation MORE significant.
    """
    # Get transverse shear modulus for spar webs
    # This governs beam shear deformation (vertical shear through webs)
    G_spar = spar_material.get_transverse_shear_modulus()
    
    # Cross-sectional area that resists vertical shear (spar webs)
    A_shear = 2 * box_height * t_spar  # Both front and rear spar webs
    
    # Shear correction factor for closed box section
    # Accounts for non-uniform shear stress distribution across section
    # κ ≈ 0.5-0.6 for thin-walled box (lower than solid rectangular ~0.833)
    # For orthotropic materials, shear lag effects reduce κ further
    if hasattr(spar_material, 'G_xz') and spar_material.G_xz is not None:
        # Orthotropic: lower correction factor due to shear lag
        kappa = 0.5
    elif spar_material.is_isotropic:
        # Isotropic thin-walled box
        kappa = 0.6
    else:
        # Default for orthotropic without explicit G_xz
        kappa = 0.5
    
    # Effective shear stiffness
    kappa_GA = kappa * G_spar * A_shear
    
    return kappa_GA, kappa


def compute_shear_stiffness_distribution(
    y: np.ndarray,
    box_height: np.ndarray,
    box_width: np.ndarray,
    t_spar: np.ndarray,
    t_skin: np.ndarray,
    spar_material: StructuralMaterial,
    skin_material: StructuralMaterial,
) -> np.ndarray:
    """
    Compute κGA distribution along span for tapered wingbox.
    
    Args:
        y: Spanwise positions [m]
        box_height: Box height at each station [m]
        box_width: Box width at each station [m]
        t_spar: Spar thickness at each station [m]
        t_skin: Skin thickness at each station [m]
        spar_material: Spar material
        skin_material: Skin material
    
    Returns:
        Array of κGA values at each spanwise station [N]
    """
    import numpy as _np
    n = len(y)
    kappa_GA = _np.zeros(n)
    
    for i in range(n):
        h_i = float(box_height[i]) if hasattr(box_height, '__getitem__') else float(box_height)
        w_i = float(box_width[i]) if hasattr(box_width, '__getitem__') else float(box_width)
        ts_i = float(t_spar[i]) if hasattr(t_spar, '__getitem__') else float(t_spar)
        tk_i = float(t_skin[i]) if hasattr(t_skin, '__getitem__') else float(t_skin)
        
        kappa_GA[i], _ = compute_timoshenko_shear_stiffness(
            box_height=h_i,
            box_width=w_i,
            t_spar=ts_i,
            t_skin=tk_i,
            spar_material=spar_material,
            skin_material=skin_material,
        )
    
    return kappa_GA


def solve_timoshenko_beam(
    y_stations: np.ndarray,
    q: np.ndarray,              # Distributed load [N/m]
    EI: np.ndarray,             # Bending stiffness at each station [N·m²]
    kappa_GA: np.ndarray,       # Shear stiffness at each station [N]
    bc_root: str = "clamped",   # "clamped" or "pinned"
    bc_tip: str = "free",       # "free" or "pinned"
) -> TimoshenkoBeamState:
    """
    Solve Timoshenko beam equations using finite difference method.
    
    This REPLACES the Euler-Bernoulli solver for improved accuracy,
    especially for thick beams and orthotropic materials.
    
    Governing equations:
        EI * d²ψ/dy² = V
        κGA * (dw/dy - ψ) = V
        dV/dy = -q
        dM/dy = V
    
    For cantilever (clamped-free):
        Root: w(0) = 0, ψ(0) = 0
        Tip: M(L) = 0, V(L) = 0
    
    Integration approach (stable for non-uniform stiffness):
    1. Integrate load to get shear force (tip to root)
    2. Integrate shear to get moment (tip to root)
    3. Compute shear strain: γ = V / (κGA)
    4. Integrate curvature to get bending rotation (root to tip)
    5. Integrate total slope to get deflection (root to tip)
    
    Args:
        y_stations: Spanwise positions [m]
        q: Distributed load at each station [N/m] (positive = upward)
        EI: Bending stiffness distribution [N·m²]
        kappa_GA: Shear stiffness distribution [N]
        bc_root: Root boundary condition ("clamped" or "pinned")
        bc_tip: Tip boundary condition ("free" or "pinned")
    
    Returns:
        TimoshenkoBeamState with all solution variables
    
    Notes:
        For very stiff beams (high κGA), this automatically reduces to
        Euler-Bernoulli behavior since γ → 0.
    """
    import numpy as _np
    
    n = len(y_stations)
    dy = _np.diff(y_stations)
    
    # Initialize arrays
    V = _np.zeros(n)      # Shear force
    M = _np.zeros(n)      # Bending moment
    psi = _np.zeros(n)    # Bending rotation
    w = _np.zeros(n)      # Deflection
    gamma = _np.zeros(n)  # Shear strain
    
    # Ensure EI and kappa_GA are numpy arrays
    EI = _np.asarray(EI)
    kappa_GA = _np.asarray(kappa_GA)
    q = _np.asarray(q)
    
    # Step 1: Integrate load to get shear force (tip to root)
    # V(tip) = 0 for free end
    # dV/dy = -q  =>  V(i) = V(i+1) + q(i) * Δy
    V[-1] = 0.0  # Free tip: no shear
    for i in range(n - 2, -1, -1):
        # Trapezoidal integration for better accuracy
        q_avg = 0.5 * (q[i] + q[i + 1])
        V[i] = V[i + 1] + q_avg * dy[i]
    
    # Step 2: Integrate shear to get moment (tip to root)
    # M(tip) = 0 for free end
    # dM/dy = V  =>  M(i) = M(i+1) + V(i) * Δy
    M[-1] = 0.0  # Free tip: no moment
    for i in range(n - 2, -1, -1):
        # Trapezoidal integration
        V_avg = 0.5 * (V[i] + V[i + 1])
        M[i] = M[i + 1] + V_avg * dy[i]
    
    # Step 3: Compute shear strain at each station
    # γ = V / (κGA)
    # Add small value to avoid division by zero
    gamma = V / (kappa_GA + 1e-10)
    
    # Step 4: Integrate curvature to get bending rotation (root to tip)
    # Curvature κ = M / EI
    # dψ/dy = κ  =>  ψ(i+1) = ψ(i) + κ(i) * Δy
    curvature = M / (EI + 1e-10)
    
    if bc_root == "clamped":
        psi[0] = 0.0  # Clamped: zero bending rotation at root
    else:
        psi[0] = 0.0  # Pinned: also zero rotation for simple case
    
    for i in range(n - 1):
        # Trapezoidal integration
        kappa_avg = 0.5 * (curvature[i] + curvature[i + 1])
        psi[i + 1] = psi[i] + kappa_avg * dy[i]
    
    # Step 5: Integrate total slope to get deflection (root to tip)
    # Total slope = ψ + γ (bending rotation + shear strain)
    # dw/dy = ψ + γ  =>  w(i+1) = w(i) + (ψ(i) + γ(i)) * Δy
    if bc_root == "clamped":
        w[0] = 0.0  # Clamped: zero deflection at root
    else:
        w[0] = 0.0  # Pinned: also zero deflection
    
    for i in range(n - 1):
        # Trapezoidal integration using total slope
        total_slope_avg = 0.5 * ((psi[i] + gamma[i]) + (psi[i + 1] + gamma[i + 1]))
        w[i + 1] = w[i] + total_slope_avg * dy[i]
    
    return TimoshenkoBeamState(
        y=y_stations,
        w=w,
        psi=psi,
        V=V,
        M=M,
        gamma=gamma,
    )


def estimate_shear_deformation_ratio(
    half_span: float,
    avg_box_height: float,
    E: float,
    G_xz: float,
) -> float:
    """
    Estimate the ratio of shear deformation to bending deformation.
    
    This helps determine if Timoshenko beam theory is necessary or if
    Euler-Bernoulli is sufficient.
    
    Args:
        half_span: Half-span length [m]
        avg_box_height: Average box height [m]
        E: Young's modulus [Pa]
        G_xz: Transverse shear modulus [Pa]
    
    Returns:
        Approximate ratio δ_shear / δ_bending
        - < 0.05: Euler-Bernoulli is adequate
        - 0.05 - 0.15: Timoshenko gives 5-15% correction
        - > 0.15: Timoshenko is necessary
    
    Reference:
        For a uniform cantilever under tip load:
        δ_shear/δ_total = 3*E*I / (κ*G*A*L²) = 3/κ * (E/G) * (h/L)² * (I/A*h²)
        
        For thin-walled box, I/(A*h²) ≈ 0.5, κ ≈ 0.5:
        δ_shear/δ_total ≈ 3 * (E/G_xz) * (h/L)²
    """
    import numpy as _np
    
    L = half_span
    h = avg_box_height
    
    if L <= 0 or h <= 0 or G_xz <= 0:
        return 0.0
    
    # Aspect ratio (span / depth)
    L_over_h = L / h
    
    # Modulus ratio
    E_over_G = E / G_xz
    
    # Approximate shear deformation ratio for thin-walled box
    # This is a simplified estimate based on beam theory
    shear_ratio = 3.0 * E_over_G * (1.0 / L_over_h) ** 2
    
    return float(_np.clip(shear_ratio, 0, 1.0))


# =============================================================================
# BUCKLING CONSTANTS AND FUNCTIONS
# =============================================================================

# Boundary condition coefficients for plate buckling (uniaxial compression)
# k_c values for simply-supported loaded edges
SKIN_BUCKLING_COEFFICIENTS = {
    "simply_supported": 4.0,    # All edges simply supported
    "semi_restrained": 5.2,     # Partial rotation restraint (typical construction)
    "clamped": 6.35,            # All edges clamped
}

# Shear buckling coefficients for spar webs
SHEAR_BUCKLING_COEFFICIENTS = {
    "simply_supported": 5.35,
    "semi_restrained": 7.0,
    "clamped": 8.98,
}


def calculate_skin_curvature_radius(
    airfoil: Any,
    front_spar_xsi: float,
    rear_spar_xsi: float,
    chord: float,
    n_samples: int = 10,
) -> float:
    """
    Calculate approximate radius of curvature for skin panel between spars.
    
    Samples the airfoil camber line between spars and fits a circular arc.
    
    Args:
        airfoil: AeroSandbox Airfoil object with local_camber() method
        front_spar_xsi: Front spar position as fraction of chord
        rear_spar_xsi: Rear spar position as fraction of chord
        chord: Local chord length [m]
        n_samples: Number of sample points
    
    Returns:
        Radius of curvature [m]. Returns float('inf') for flat sections.
    """
    import numpy as _np
    
    if airfoil is None:
        return float('inf')
    
    try:
        # Sample camber between spars
        x_over_c = _np.linspace(front_spar_xsi, rear_spar_xsi, n_samples)
        
        # Get camber values (AeroSandbox Airfoil.local_camber method)
        if hasattr(airfoil, 'local_camber'):
            camber = airfoil.local_camber(x_over_c=x_over_c)
        else:
            # Fallback: assume flat
            return float('inf')
        
        # Convert to physical coordinates
        x = x_over_c * chord
        z = _np.array(camber) * chord  # Camber is normalized by chord
        
        # Calculate curvature using second derivative
        # κ = |d²z/dx²| / (1 + (dz/dx)²)^(3/2)
        dx = _np.diff(x)
        dz = _np.diff(z)
        dz_dx = dz / (dx + 1e-10)
        
        # Second derivative (central difference)
        if len(dz_dx) < 2:
            return float('inf')
        
        d2z_dx2 = _np.diff(dz_dx) / ((dx[:-1] + dx[1:]) / 2 + 1e-10)
        dz_dx_mid = (dz_dx[:-1] + dz_dx[1:]) / 2
        
        # Curvature at each point
        curvature = _np.abs(d2z_dx2) / (1 + dz_dx_mid**2)**1.5
        
        # Use maximum curvature (minimum radius)
        max_curvature = _np.max(curvature)
        
        if max_curvature < 1e-6:
            return float('inf')  # Essentially flat
        
        radius = 1.0 / max_curvature
        return float(radius)
        
    except Exception:
        return float('inf')  # Fallback to flat plate assumption


def curved_panel_buckling_factor(
    panel_width: float,
    radius_of_curvature: float,
    thickness: float,
    poissons_ratio: float = 0.3,
) -> float:
    """
    Calculate buckling enhancement factor for curved (cylindrical) panels.
    
    Uses Batdorf's curvature parameter Z to determine correction.
    
    Args:
        panel_width: Panel width [m]
        radius_of_curvature: Radius [m]
        thickness: Skin thickness [m]
        poissons_ratio: Material Poisson's ratio
    
    Returns:
        Multiplier for flat plate buckling stress (≥1.0)
    
    Reference:
        NACA TN-2661, Batdorf (1947)
    """
    import numpy as _np
    
    if radius_of_curvature <= 0 or radius_of_curvature == float('inf'):
        return 1.0
    
    if panel_width <= 0 or thickness <= 0:
        return 1.0
    
    # Batdorf curvature parameter
    Z = (panel_width**2) / (radius_of_curvature * thickness) * _np.sqrt(1 - poissons_ratio**2)
    
    if Z < 2.5:
        return 1.0  # Flat plate behavior
    
    # Empirical enhancement (from NACA data)
    if Z < 100:
        enhancement = 1.0 + 0.1 * _np.sqrt(Z)
    else:
        enhancement = 1.0 + 1.0 * _np.log10(Z)
    
    return min(float(enhancement), 3.0)  # Cap at 3x


def ks_aggregate(
    values: np.ndarray,
    rho: float = 50.0,
    minimize: bool = True,
) -> float:
    """
    Kreisselmeier-Steinhauser (KS) aggregation function for smooth min/max.
    
    Used to convert a set of constraint values into a single differentiable
    constraint for gradient-based optimization. Provides a smooth approximation
    to max(values) or min(values).
    
    Args:
        values: Array of constraint values to aggregate
        rho: Aggregation parameter (higher = closer to true max/min, but less smooth)
             Typical range: 10-100. Default 50 works well for structural constraints.
        minimize: If True, approximates min(values). If False, approximates max(values).
    
    Returns:
        Scalar approximation of the extreme value.
    
    Notes:
        For constraint g_i <= 0, use KS = (1/rho) * ln(sum(exp(rho * g_i)))
        The KS function slightly overestimates max (conservative for constraints).
        
    Reference:
        Kreisselmeier, G., & Steinhauser, R. (1979). Systematic control design
        by optimizing a vector performance index.
    """
    if minimize:
        # For min: negate, compute max, negate result
        values = -values
    
    # Shift values for numerical stability (subtract max before exp)
    max_val = np.max(values)
    shifted = values - max_val
    
    # KS aggregation
    ks_value = max_val + (1.0 / rho) * np.log(np.sum(np.exp(rho * shifted)))
    
    if minimize:
        ks_value = -ks_value
    
    return ks_value


def calculate_orthotropic_buckling_stress(
    panel_length: float,
    panel_width: float,
    thickness: float,
    material: StructuralMaterial,
    grain_spanwise: bool = True,
    boundary_condition: str = "clamped",
) -> float:
    """
    Calculate critical buckling stress for orthotropic plate under uniaxial compression.
    
    Uses closed-form solution for simply-supported edges with correction
    for other boundary conditions.
    
    Args:
        panel_length: Length in load direction (rib spacing) [m]
        panel_width: Width perpendicular to load (stringer spacing) [m]
        thickness: Plate thickness [m]
        material: StructuralMaterial with orthotropic properties
        grain_spanwise: True if grain runs in load direction
        boundary_condition: "simply_supported", "semi_restrained", or "clamped"
    
    Returns:
        Critical buckling stress σ_cr [Pa]
    
    Reference:
        Lekhnitskii (1968), Whitney (1987)
    """
    import numpy as _np
    
    a = panel_length
    b = panel_width
    t = thickness
    
    if a <= 0 or b <= 0 or t <= 0:
        return float('inf')  # No buckling possible
    
    # Boundary condition factor (relative to simply-supported)
    bc_factor = SKIN_BUCKLING_COEFFICIENTS.get(boundary_condition, 6.35) / 4.0
    
    # Get plate bending stiffnesses
    if hasattr(material, 'get_plate_bending_stiffnesses'):
        D = material.get_plate_bending_stiffnesses(t, grain_spanwise)
        D_11, D_22, D_12, D_66 = D["D_11"], D["D_22"], D["D_12"], D["D_66"]
    else:
        # Fallback to isotropic calculation
        E = material.E_1
        nu = material.nu_12
        G = material.G_12
        D_base = E * t**3 / (12 * (1 - nu**2))
        D_11 = D_22 = D_base
        D_12 = nu * D_base
        D_66 = G * t**3 / 12
    
    if material.is_isotropic or getattr(material, 'material_type', None) == 'quasi_isotropic':
        # Isotropic: use standard formula
        E = material.E_1
        nu = material.nu_12
        k_c = 4.0 * bc_factor  # Base k=4 for simply-supported
        sigma_cr = k_c * (_np.pi**2 * E) / (12 * (1 - nu**2)) * (t / b)**2
        return float(sigma_cr)
    
    # Orthotropic: find minimum N_cr over half-wave numbers m
    aspect_ratio = a / b if b > 0 else 1.0
    
    # Optimal m depends on orthotropy and aspect ratio
    # For highly orthotropic (wood), m=1 often optimal even for long plates
    gamma = (D_11 / (D_22 + 1e-20))**0.25
    m_max = max(1, int(_np.ceil(aspect_ratio * gamma)))
    
    N_cr_min = float('inf')
    for m in range(1, m_max + 3):
        # N_cr = π²/b² * [D_11*(mb/a)² + 2*(D_12 + 2*D_66) + D_22*(a/(mb))²]
        term1 = D_11 * (m * b / a)**2
        term2 = 2 * (D_12 + 2 * D_66)
        term3 = D_22 * (a / (m * b))**2
        N_cr = _np.pi**2 / b**2 * (term1 + term2 + term3)
        N_cr_min = min(N_cr_min, N_cr)
    
    # Apply boundary condition factor
    N_cr_min *= bc_factor
    
    # Convert to stress: σ = N / t
    sigma_cr = N_cr_min / t
    
    return float(sigma_cr)


def calculate_orthotropic_buckling_stress_symbolic(
    panel_length: float,
    panel_width: float,
    thickness,  # Can be CasADi symbolic or float
    material: StructuralMaterial,
    grain_spanwise: bool = True,
    boundary_condition: str = "clamped",
):
    """
    Calculate critical buckling stress for orthotropic plate - CasADi compatible.
    
    This version supports symbolic thickness for gradient-based optimization.
    Uses the same physics as calculate_orthotropic_buckling_stress() but with
    smooth operations that work with CasADi.
    
    Args:
        panel_length: Length in load direction (rib spacing) [m]
        panel_width: Width perpendicular to load (stringer spacing) [m]
        thickness: Plate thickness [m] - can be CasADi MX/SX
        material: StructuralMaterial with orthotropic properties
        grain_spanwise: True if grain runs in load direction
        boundary_condition: "simply_supported", "semi_restrained", or "clamped"
    
    Returns:
        Critical buckling stress σ_cr [Pa] - same type as thickness input
    """
    import numpy as _np_std
    
    a = panel_length
    b = panel_width
    t = thickness
    
    # Boundary condition factor (relative to simply-supported)
    bc_factor = SKIN_BUCKLING_COEFFICIENTS.get(boundary_condition, 6.35) / 4.0
    
    # Get material properties
    E_1 = material.E_1
    E_2 = material.E_2 if hasattr(material, 'E_2') else material.E_1
    nu_12 = material.nu_12
    nu_21 = getattr(material, 'nu_21', nu_12 * E_2 / E_1)
    G_12 = material.G_12
    
    # Compute plate bending stiffnesses (D matrix) - symbolic compatible
    # D_ij = Q_ij * t³ / 12 for thin plates
    denom = 1 - nu_12 * nu_21
    
    if grain_spanwise:
        # Grain along span (load direction)
        Q_11 = E_1 / denom
        Q_22 = E_2 / denom
        Q_12 = nu_12 * E_2 / denom
    else:
        # Grain perpendicular to span
        Q_11 = E_2 / denom
        Q_22 = E_1 / denom
        Q_12 = nu_21 * E_1 / denom
    Q_66 = G_12
    
    # Bending stiffnesses
    t_cubed_12 = t ** 3 / 12
    D_11 = Q_11 * t_cubed_12
    D_22 = Q_22 * t_cubed_12
    D_12 = Q_12 * t_cubed_12
    D_66 = Q_66 * t_cubed_12
    
    # Check if material is effectively isotropic
    is_isotropic = material.is_isotropic or getattr(material, 'material_type', None) == 'quasi_isotropic'
    
    if is_isotropic:
        # Isotropic: use standard formula
        E = material.E_1
        nu = material.nu_12
        k_c = 4.0 * bc_factor
        sigma_cr = k_c * (_np_std.pi ** 2 * E) / (12 * (1 - nu ** 2)) * (t / b) ** 2
        return sigma_cr
    
    # Orthotropic: evaluate N_cr for multiple half-wave numbers and take minimum
    # N_cr(m) = π²/b² * [D_11*(mb/a)² + 2*(D_12 + 2*D_66) + D_22*(a/(mb))²]
    
    # Evaluate for m = 1, 2, 3, 4, 5 (covers most practical cases)
    pi_sq_over_b_sq = _np_std.pi ** 2 / (b ** 2)
    term2 = 2 * (D_12 + 2 * D_66)  # This term is constant across m
    
    N_cr_candidates = []
    for m in range(1, 6):
        term1 = D_11 * (m * b / a) ** 2
        term3 = D_22 * (a / (m * b)) ** 2
        N_cr_m = pi_sq_over_b_sq * (term1 + term2 + term3)
        N_cr_candidates.append(N_cr_m)
    
    # Use smooth minimum (softmin with large rho approximates true min)
    # For CasADi compatibility, use np.minimum chain
    N_cr_min = N_cr_candidates[0]
    for N_cr_m in N_cr_candidates[1:]:
        N_cr_min = np.minimum(N_cr_min, N_cr_m)
    
    # Apply boundary condition factor
    N_cr_min = N_cr_min * bc_factor
    
    # Convert to stress: σ = N / t
    sigma_cr = N_cr_min / t
    
    return sigma_cr


def calculate_spar_shear_buckling_stress_symbolic(
    spar_height: float,
    thickness,  # Can be CasADi symbolic or float
    material: StructuralMaterial,
    boundary_condition: str = "semi_restrained",
):
    """
    Calculate critical shear buckling stress for spar web - CasADi compatible.
    
    Uses the same formula as _compute_buckling_data() for consistency.
    
    Args:
        spar_height: Height of spar web [m]
        thickness: Spar web thickness [m] - can be CasADi MX/SX
        material: StructuralMaterial for spar
        boundary_condition: Edge support condition
    
    Returns:
        Critical shear buckling stress τ_cr [Pa]
    """
    import numpy as _np_std
    
    h = spar_height
    t = thickness
    
    E = material.E_1
    nu = material.nu_12
    k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(boundary_condition, 7.0)
    
    # τ_cr = k_s * π² * E * (t/h)² / (12 * (1 - ν²))
    tau_cr = k_shear * (_np_std.pi ** 2 * E) / (12 * (1 - nu ** 2)) * (t / h) ** 2
    
    return tau_cr


def calculate_rib_shear_buckling_stress(
    rib_height: float,
    rib_thickness: float,
    material: StructuralMaterial,
    lightening_fraction: float = 0.0,
    boundary_condition: str = "simply_supported",
) -> float:
    """
    Calculate critical shear buckling stress for rib web.
    
    Ribs transfer shear loads from skin panels and can buckle in shear,
    especially when thin or heavily lightened.
    
    Args:
        rib_height: Height of rib (wingbox height) [m]
        rib_thickness: Rib web thickness [m]
        material: StructuralMaterial for rib
        lightening_fraction: Fraction of material removed (reduces effective properties)
        boundary_condition: Edge support condition
    
    Returns:
        Critical shear buckling stress τ_cr [Pa]
    
    Notes:
        - Lightening holes reduce effective stiffness and introduce stress concentrations
        - For heavily lightened ribs (>40%), this is a rough approximation
    """
    import numpy as _np
    
    h = rib_height
    t = rib_thickness
    
    if h <= 0 or t <= 0:
        return float('inf')
    
    E = material.E_1  # Use primary modulus
    nu = material.nu_12
    
    # Shear buckling coefficient
    # k_s ≈ 5.35 for simply-supported edges (typical for ribs)
    # k_s ≈ 8.98 for clamped edges
    k_shear = {
        "simply_supported": 5.35,
        "semi_restrained": 7.0,
        "clamped": 8.98,
    }.get(boundary_condition, 5.35)
    
    # Critical shear stress for solid rib
    # τ_cr = k_s * π² * E * (t/h)² / (12 * (1 - ν²))
    tau_cr = k_shear * (_np.pi ** 2 * E) / (12 * (1 - nu ** 2)) * (t / h) ** 2
    
    # Apply knockdown for lightening holes
    # Holes reduce effective stiffness and create stress concentrations
    if lightening_fraction > 0:
        # Stiffness knockdown: approximately (1 - f)^1.5 for random holes
        stiffness_knockdown = (1.0 - lightening_fraction) ** 1.5
        
        # Stress concentration knockdown (holes create local stress risers)
        # Kt ≈ 1.0 + 2.0 * f for moderate lightening
        if lightening_fraction < 0.3:
            stress_knockdown = 1.0 / (1.0 + 1.5 * lightening_fraction)
        elif lightening_fraction < 0.5:
            stress_knockdown = 1.0 / (1.5 + 2.0 * (lightening_fraction - 0.3))
        else:
            stress_knockdown = 1.0 / (1.9 + 3.0 * (lightening_fraction - 0.5))
        
        tau_cr *= stiffness_knockdown * stress_knockdown
    
    return float(tau_cr)


def calculate_rib_crushing_stress(
    bending_moment: float,
    box_height: float,
    rib_thickness: float,
    spar_cap_width: float,
) -> float:
    """
    Calculate bearing/crushing stress where spar cap bears on rib.
    
    The spar cap carries axial load from bending. This load transfers
    into the rib at a concentrated bearing area, potentially crushing
    soft rib materials like balsa.
    
    Args:
        bending_moment: Local bending moment [N*m]
        box_height: Height of wingbox [m]
        rib_thickness: Rib thickness [m]
        spar_cap_width: Width of spar cap bearing on rib [m]
    
    Returns:
        Bearing stress σ_bearing [Pa]
    
    Notes:
        Compare against material.sigma_2_compression (cross-grain crushing strength)
        for wood materials, as ribs typically have grain perpendicular to spar caps.
    """
    if box_height <= 0 or rib_thickness <= 0 or spar_cap_width <= 0:
        return 0.0
    
    # Spar cap force from bending: P = M / h (approximate, assumes caps at h/2 from NA)
    # More accurately: P = M / (h - t_flange), but h is close enough for thin flanges
    P_cap = abs(bending_moment) / (box_height + 1e-10)
    
    # Bearing area = rib_thickness * spar_cap_width
    A_bearing = rib_thickness * spar_cap_width
    
    # Bearing stress
    sigma_bearing = P_cap / (A_bearing + 1e-10)
    
    return float(sigma_bearing)


def calculate_rib_crushing_margin(
    bending_moment: float,
    box_height: float,
    rib_thickness: float,
    spar_cap_width: float,
    material: StructuralMaterial,
) -> float:
    """
    Calculate rib crushing margin (allowable / applied).
    
    Args:
        bending_moment: Local bending moment [N*m]
        box_height: Height of wingbox [m]
        rib_thickness: Rib thickness [m]
        spar_cap_width: Width of spar cap bearing on rib [m]
        material: Rib material
    
    Returns:
        Crushing margin = σ_allowable / σ_bearing (>1.0 is safe)
    """
    sigma_bearing = calculate_rib_crushing_stress(
        bending_moment, box_height, rib_thickness, spar_cap_width
    )
    
    if sigma_bearing <= 0:
        return float('inf')
    
    # Use cross-grain compression strength for wood (grain is typically vertical in ribs)
    # This is typically sigma_2_compression for orthotropic materials
    sigma_allowable = getattr(material, 'sigma_2_compression', material.sigma_1_compression)
    
    margin = sigma_allowable / sigma_bearing
    return float(margin)


def calculate_skin_shear_buckling_stress(
    panel_length: float,
    panel_width: float,
    thickness: float,
    material: StructuralMaterial,
    grain_spanwise: bool = True,
    boundary_condition: str = "clamped",
) -> float:
    """
    Calculate critical shear buckling stress for skin panel under torsion.
    
    Skin panels experience torsional shear stress from wing twisting moments.
    This can cause shear buckling (diagonal wrinkling) before compressive
    buckling occurs, especially for high-torque cases (cambered airfoils).
    
    For orthotropic plates under shear, the critical stress depends on
    the D_66 (shear stiffness) and the geometric mean of D_11 and D_22.
    
    Args:
        panel_length: Length of panel (rib spacing) [m]
        panel_width: Width of panel (stringer spacing or box width) [m]
        thickness: Skin thickness [m]
        material: StructuralMaterial with orthotropic properties
        grain_spanwise: True if grain runs spanwise
        boundary_condition: Edge support condition
    
    Returns:
        Critical shear buckling stress τ_cr [Pa]
    
    Reference:
        NASA SP-8007 "Buckling of Thin-Walled Circular Cylinders"
        Whitney, J.M. (1987) "Structural Analysis of Laminated Composites"
    """
    import numpy as _np
    
    a = panel_length
    b = panel_width
    t = thickness
    
    if a <= 0 or b <= 0 or t <= 0:
        return float('inf')  # No buckling possible
    
    # Shear buckling coefficient
    k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(boundary_condition, 7.0)
    
    # Get plate bending stiffnesses
    if hasattr(material, 'get_plate_bending_stiffnesses'):
        D = material.get_plate_bending_stiffnesses(t, grain_spanwise)
        D_11, D_22, D_12, D_66 = D["D_11"], D["D_22"], D["D_12"], D["D_66"]
    else:
        # Fallback to isotropic calculation
        E = material.E_1
        nu = material.nu_12
        G = material.G_12
        D_base = E * t**3 / (12 * (1 - nu**2))
        D_11 = D_22 = D_base
        D_12 = nu * D_base
        D_66 = G * t**3 / 12
    
    if material.is_isotropic or getattr(material, 'material_type', None) == 'quasi_isotropic':
        # Isotropic: use standard shear buckling formula
        # τ_cr = k_s * π² * E / (12 * (1 - ν²)) * (t/b)²
        E = material.E_1
        nu = material.nu_12
        tau_cr = k_shear * (_np.pi**2 * E) / (12 * (1 - nu**2)) * (t / b)**2
        return float(tau_cr)
    
    # Orthotropic shear buckling
    # For long plates (a >> b), the formula simplifies to:
    # τ_cr = (4/b²) * (D_11 * D_22³)^0.25 * [k_s_ortho]
    # where k_s_ortho depends on D_12 + 2*D_66
    
    # Use the general orthotropic formula from Whitney (1987):
    # N_xy_cr = π²/b² * 4 * (D_11 * D_22)^0.5 * k_s(β)
    # where β = (D_12 + 2*D_66) / (D_11 * D_22)^0.5
    
    # Calculate orthotropic parameter
    sqrt_D11_D22 = _np.sqrt(D_11 * D_22)
    beta = (D_12 + 2 * D_66) / (sqrt_D11_D22 + 1e-20)
    
    # Aspect ratio parameter
    aspect_ratio = a / b
    gamma = (D_11 / D_22) ** 0.25
    
    # Effective aspect ratio for orthotropic plate
    phi = aspect_ratio / gamma
    
    # Shear buckling coefficient for orthotropic plate
    # From NASA SP-8007, for simply-supported edges:
    # k_s ≈ 5.35 + 4.0 / φ² for φ > 1 (long plates)
    # k_s ≈ 8.98 * (φ/aspect_ratio)² for φ < 1 (short plates)
    if phi >= 1.0:
        k_s_ortho = 5.35 + 4.0 / (phi**2)
    else:
        k_s_ortho = 5.35 * (1.0 / phi)**2 + 4.0
    
    # Apply boundary condition factor
    bc_factor = k_shear / 7.0  # Normalize to semi_restrained
    k_s_ortho *= bc_factor
    
    # Critical shear buckling load per unit width
    # N_xy_cr = π²/b² * 4 * sqrt(D_11 * D_22) * k_s * [1 + 0.3*β] (approximate)
    N_xy_cr = (_np.pi**2 / b**2) * 4 * sqrt_D11_D22 * k_s_ortho
    
    # Correction for D_12 + 2*D_66 contribution (increases buckling resistance)
    # This is a simplified correction; full solution requires solving eigenvalue problem
    correction = 1.0 + 0.1 * min(beta, 2.0)  # Cap correction for numerical stability
    N_xy_cr *= correction
    
    # Convert to stress: τ_cr = N_xy_cr / t
    tau_cr = N_xy_cr / t
    
    return float(tau_cr)


def calculate_skin_shear_buckling_stress_symbolic(
    panel_length: float,
    panel_width: float,
    thickness,  # Can be CasADi symbolic or float
    material: StructuralMaterial,
    grain_spanwise: bool = True,
    boundary_condition: str = "clamped",
):
    """
    Calculate critical shear buckling stress for skin panel - CasADi compatible.
    
    This version supports symbolic thickness for gradient-based optimization.
    Uses the same physics as calculate_skin_shear_buckling_stress() but with
    smooth operations that work with CasADi.
    
    Args:
        panel_length: Length of panel (rib spacing) [m]
        panel_width: Width of panel (stringer spacing or box width) [m]
        thickness: Skin thickness [m] - can be CasADi MX/SX
        material: StructuralMaterial with orthotropic properties
        grain_spanwise: True if grain runs spanwise
        boundary_condition: Edge support condition
    
    Returns:
        Critical shear buckling stress τ_cr [Pa] - same type as thickness input
    """
    import numpy as _np_std
    
    a = panel_length
    b = panel_width
    t = thickness
    
    # Shear buckling coefficient
    k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(boundary_condition, 7.0)
    
    # Get material properties
    E_1 = material.E_1
    E_2 = material.E_2 if hasattr(material, 'E_2') else material.E_1
    nu_12 = material.nu_12
    nu_21 = getattr(material, 'nu_21', nu_12 * E_2 / E_1)
    G_12 = material.G_12
    
    # Compute plate bending stiffnesses (D matrix) - symbolic compatible
    denom = 1 - nu_12 * nu_21
    
    if grain_spanwise:
        Q_11 = E_1 / denom
        Q_22 = E_2 / denom
        Q_12 = nu_12 * E_2 / denom
    else:
        Q_11 = E_2 / denom
        Q_22 = E_1 / denom
        Q_12 = nu_21 * E_1 / denom
    Q_66 = G_12
    
    # Bending stiffnesses
    t_cubed_12 = t ** 3 / 12
    D_11 = Q_11 * t_cubed_12
    D_22 = Q_22 * t_cubed_12
    D_66 = Q_66 * t_cubed_12
    
    # Check if material is effectively isotropic
    is_isotropic = material.is_isotropic or getattr(material, 'material_type', None) == 'quasi_isotropic'
    
    if is_isotropic:
        # Isotropic: use standard shear buckling formula
        E = material.E_1
        nu = material.nu_12
        tau_cr = k_shear * (_np_std.pi ** 2 * E) / (12 * (1 - nu ** 2)) * (t / b) ** 2
        return tau_cr
    
    # Orthotropic: simplified formula that works with CasADi
    # τ_cr ≈ k_s * (π²/b²) * (D_11 * D_22)^0.5 / t * factor
    # Using geometric mean of stiffnesses
    sqrt_D11_D22 = (D_11 * D_22) ** 0.5
    
    # Aspect ratio correction (simplified for smooth gradient)
    aspect_ratio = a / b
    gamma = (D_11 / (D_22 + 1e-20)) ** 0.25
    phi = aspect_ratio / (gamma + 1e-10)
    
    # Smooth k_s that works for all aspect ratios
    # k_s ≈ 5.35 + 4.0 / max(phi², 0.1)
    k_s_ortho = 5.35 + 4.0 / (phi ** 2 + 0.1)
    
    # Apply boundary condition factor
    bc_factor = k_shear / 7.0
    k_s_ortho = k_s_ortho * bc_factor
    
    # Critical shear stress
    tau_cr = k_s_ortho * (_np_std.pi ** 2 / b ** 2) * 4 * sqrt_D11_D22 / t
    
    return tau_cr


def calculate_combined_buckling_margin(
    sigma_applied: float,
    sigma_cr: float,
    tau_applied: float,
    tau_cr: float,
    interaction_exponent: float = 2.0,
) -> float:
    """
    Calculate combined buckling margin for biaxial stress state (σ + τ).
    
    Skin panels under combined compression and shear have reduced buckling
    strength compared to either load acting alone. This interaction is
    captured by an interaction equation.
    
    The interaction equation (NASA SP-8007):
        (σ/σ_cr)^α + (τ/τ_cr)^β ≤ 1.0
    
    For most cases, α = β = 2 (parabolic interaction) is appropriate.
    For conservative design, α = 1, β = 2 (linear-parabolic) may be used.
    
    Args:
        sigma_applied: Applied compressive stress [Pa] (positive = compression)
        sigma_cr: Critical compressive buckling stress [Pa]
        tau_applied: Applied shear stress [Pa]
        tau_cr: Critical shear buckling stress [Pa]
        interaction_exponent: Exponent for both terms (default 2.0)
    
    Returns:
        Combined buckling margin = 1.0 / sqrt(R_σ^α + R_τ^β)
        Where R_σ = σ/σ_cr and R_τ = τ/τ_cr
        Margin > 1.0 means safe, < 1.0 means buckling expected
    
    Reference:
        NASA SP-8007 "Buckling of Thin-Walled Circular Cylinders"
        Bruhn, E.F. "Analysis and Design of Flight Vehicle Structures"
    """
    import numpy as _np
    
    # Stress ratios
    R_sigma = abs(sigma_applied) / (sigma_cr + 1e-10)
    R_tau = abs(tau_applied) / (tau_cr + 1e-10)
    
    # Interaction sum
    alpha = interaction_exponent
    beta = interaction_exponent
    interaction_sum = R_sigma**alpha + R_tau**beta
    
    if interaction_sum <= 0:
        return float('inf')  # No load, no buckling
    
    # Combined margin = 1 / (interaction_sum)^(1/max(α,β))
    # This gives margin = 1.0 when exactly at buckling boundary
    margin = 1.0 / (interaction_sum ** (1.0 / max(alpha, beta)))
    
    return float(margin)


# =============================================================================
# STRINGER CRIPPLING ANALYSIS (Priority 6)
# =============================================================================
# Stringer crippling is a local instability failure where thin elements of the
# stringer (flanges, webs) buckle locally before the stringer fails as a column.
# This is critical for thin-walled stringer sections under compressive loads.
#
# The Gerard method provides empirical crippling stress formulas based on
# extensive testing of aircraft structural elements (primarily aluminum).
# For orthotropic materials (wood, composites), use plate buckling formulas.
# =============================================================================


def _calculate_gerard_crippling(
    t_over_b: float,
    E: float,
    sigma_cy: float,
    section_type: str,
    edge_condition: str,
) -> float:
    """
    Original Gerard method for isotropic metals (aluminum).
    
    Args:
        t_over_b: Thickness-to-width ratio of critical element
        E: Elastic modulus [Pa]
        sigma_cy: Compressive yield strength [Pa]
        section_type: Stringer section type
        edge_condition: Edge support condition
    
    Returns:
        Crippling stress σ_cc [Pa]
    """
    if edge_condition == "one_edge_free":
        if section_type == "rectangular":
            beta, m, n = 0.316, 0.25, 1.5
        elif section_type in ("L_section", "T_section"):
            beta, m, n = 0.342, 0.25, 1.5
        else:
            beta, m, n = 0.316, 0.25, 1.5
    else:  # both_edges_supported
        beta, m, n = 0.56, 0.33, 1.33
    
    sigma_ratio = sigma_cy / E
    sigma_cc = beta * E * (t_over_b ** n) * (sigma_ratio ** (1 + m))
    
    return sigma_cc


def _calculate_orthotropic_crippling(
    b_over_t: float,
    t_over_b: float,
    material: 'StructuralMaterial',
    section_type: str,
    edge_condition: str,
) -> float:
    """
    Calculate crippling stress for orthotropic materials (wood, composites).
    
    Uses classical plate buckling formula with orthotropic stiffness:
        σ_cr = k * π² * E_eff / (12 * (1 - ν²)) * (t/b)²
    
    For wood stringers, the formula accounts for:
    1. Grain running along stringer length (spanwise)
    2. Buckling across the grain (transverse direction)
    3. Material compression limits
    
    Reference:
        - Timoshenko & Gere, "Theory of Elastic Stability" Ch. 9
        - Forest Products Laboratory, "Wood Handbook" Ch. 9
    
    Args:
        b_over_t: Width-to-thickness ratio of critical element
        t_over_b: Thickness-to-width ratio (1/b_over_t)
        material: Orthotropic material properties
        section_type: Stringer section type
        edge_condition: Edge support condition
    
    Returns:
        Crippling stress σ_cc [Pa]
    """
    import numpy as _np
    
    E_1 = material.E_1  # Along grain (spanwise)
    E_2 = material.E_2  # Across grain (buckling direction)
    G_12 = material.G_12
    nu_12 = material.nu_12
    sigma_cy = material.sigma_1_compression
    
    # Buckling coefficient based on edge condition
    if edge_condition == "one_edge_free":
        # Outstanding element (one edge free)
        # k = 0.425 for simply supported loaded edges
        k = 0.425
    else:
        # Both edges supported
        # k = 4.0 for simply supported, 6.97 for clamped
        k = 4.0
    
    # For orthotropic plates, use geometric mean of moduli
    # This accounts for plate resistance in both directions
    E_eff = _np.sqrt(E_1 * E_2)
    
    # Poisson effect (use 0.5 floor to prevent invalid combinations)
    denom = max(1 - nu_12**2, 0.5)
    
    # Classical plate buckling stress
    # σ_cr = k * π² * E / (12 * (1 - ν²)) * (t/b)²
    sigma_cr_plate = k * _np.pi**2 * E_eff / (12 * denom) * t_over_b**2
    
    # For wood, also check local crushing/compression limit
    # Crippling cannot exceed local crushing strength
    # Use 80% of compression strength as practical limit
    sigma_crushing = 0.80 * sigma_cy
    
    # Transition between plate buckling and crushing based on slenderness
    # For slender elements (b/t > 20), plate buckling dominates
    # For stocky elements (b/t < 8), crushing dominates
    slenderness = b_over_t
    if slenderness > 20:
        sigma_cc = sigma_cr_plate
    elif slenderness < 8:
        sigma_cc = sigma_crushing
    else:
        # Johnson-Euler type transition: smooth interpolation
        transition_factor = (slenderness - 8) / 12  # 0 at b/t=8, 1 at b/t=20
        sigma_cc = sigma_crushing * (1 - transition_factor) + sigma_cr_plate * transition_factor
    
    # Apply knockdown for imperfections (wood has natural variability)
    knockdown = 0.80  # 20% knockdown for wood imperfections
    sigma_cc *= knockdown
    
    return sigma_cc


def calculate_stringer_crippling_stress(
    b_over_t: float,
    material: StructuralMaterial,
    section_type: str = "rectangular",
    edge_condition: str = "one_edge_free",
) -> float:
    """
    Calculate crippling stress for stringer element.
    
    Uses material-appropriate methods:
    - Isotropic metals: Gerard method (empirical, calibrated for aluminum)
    - Orthotropic (wood/composites): Classical plate buckling with material limits
    
    Crippling is local buckling of thin stringer elements (flanges, webs)
    that occurs before global column buckling. Critical for thin-walled sections.
    
    Args:
        b_over_t: Width-to-thickness ratio of the critical element
        material: Structural material properties
        section_type: "rectangular", "L_section", "T_section", "hat"
        edge_condition: "one_edge_free" (outstanding) or "both_edges_supported"
    
    Returns:
        Crippling stress σ_cc [Pa]
    
    Reference:
        - Gerard, G. (1957) "Handbook of Structural Stability - Part V" NACA TN-3785
        - Bruhn, E.F. "Analysis and Design of Flight Vehicle Structures" Ch. C7
        - Timoshenko & Gere, "Theory of Elastic Stability" Ch. 9
    """
    import numpy as _np
    
    if b_over_t <= 0:
        return float('inf')  # No crippling possible
    
    t_over_b = 1.0 / b_over_t
    
    # Get material properties
    E = material.E_1  # Use longitudinal modulus (stringer runs spanwise)
    sigma_cy = material.sigma_1_compression  # Compressive yield strength
    
    # === Material-specific crippling calculation ===
    if material.material_type == MaterialType.ISOTROPIC:
        # Metals: Use Gerard method (original empirical approach)
        sigma_cc = _calculate_gerard_crippling(
            t_over_b, E, sigma_cy, section_type, edge_condition
        )
    
    elif material.material_type in (MaterialType.ORTHOTROPIC, MaterialType.QUASI_ISOTROPIC):
        # Wood and composites: Use plate buckling with material limits
        sigma_cc = _calculate_orthotropic_crippling(
            b_over_t, t_over_b, material, section_type, edge_condition
        )
    
    else:
        # Fallback to Gerard (conservative)
        sigma_cc = _calculate_gerard_crippling(
            t_over_b, E, sigma_cy, section_type, edge_condition
        )
    
    # Cap at compressive yield (crippling can't exceed material strength)
    sigma_cc = min(sigma_cc, sigma_cy)
    
    return float(sigma_cc)


def calculate_stringer_crippling_stress_symbolic(
    b_over_t,  # Can be CasADi symbolic or float
    material: StructuralMaterial,
    section_type: str = "rectangular",
    edge_condition: str = "one_edge_free",
):
    """
    Calculate stringer crippling stress - CasADi compatible version.
    
    Uses material-appropriate methods (same logic as non-symbolic version):
    - Isotropic metals: Gerard method
    - Orthotropic (wood/composites): Plate buckling with material limits
    
    Args:
        b_over_t: Width-to-thickness ratio (can be symbolic)
        material: Structural material properties
        section_type: Stringer section type
        edge_condition: Edge support condition
    
    Returns:
        Crippling stress σ_cc [Pa] - same type as b_over_t
    """
    t_over_b = 1.0 / (b_over_t + 1e-10)
    
    E = material.E_1
    E_2 = material.E_2
    nu_12 = material.nu_12
    sigma_cy = material.sigma_1_compression
    
    # === Material-specific calculation ===
    if material.material_type == MaterialType.ISOTROPIC:
        # Gerard method for metals
        if edge_condition == "one_edge_free":
            if section_type == "rectangular":
                beta, m, n = 0.316, 0.25, 1.5
            elif section_type in ("L_section", "T_section"):
                beta, m, n = 0.342, 0.25, 1.5
            else:
                beta, m, n = 0.316, 0.25, 1.5
        else:
            beta, m, n = 0.56, 0.33, 1.33
        
        sigma_ratio = sigma_cy / E
        sigma_cc = beta * E * (t_over_b ** n) * (sigma_ratio ** (1 + m))
    
    else:
        # Orthotropic: plate buckling with material limits
        # Buckling coefficient
        k = 0.425 if edge_condition == "one_edge_free" else 4.0
        
        # Effective modulus (geometric mean)
        E_eff = np.sqrt(E * E_2)
        
        # Poisson effect
        denom = np.maximum(1 - nu_12**2, 0.5)
        
        # Plate buckling stress
        sigma_cr_plate = k * np.pi**2 * E_eff / (12 * denom) * t_over_b**2
        
        # Crushing limit (80% of yield)
        sigma_crushing = 0.80 * sigma_cy
        
        # Smooth transition based on slenderness
        # Use smooth approximation for CasADi compatibility
        # sigmoid-like transition centered at b/t = 14
        transition = 1.0 / (1.0 + np.exp(-(b_over_t - 14) / 3))
        sigma_cc = sigma_crushing * (1 - transition) + sigma_cr_plate * transition
        
        # Apply knockdown for imperfections
        sigma_cc = sigma_cc * 0.80
    
    # Use smooth min to cap at yield (CasADi compatible)
    sigma_cc = np.minimum(sigma_cc, sigma_cy)
    
    return sigma_cc


def calculate_stringer_column_buckling_stress(
    stringer_length: float,
    stringer_props: 'StringerProperties',
    end_fixity: float = 1.0,
) -> float:
    """
    Calculate Euler column buckling stress for stringer.
    
    For slender stringers, global column buckling may occur before local crippling.
    The critical stress is the minimum of crippling and column buckling.
    
    Args:
        stringer_length: Unsupported length (rib spacing) [m]
        stringer_props: Stringer geometry and material
        end_fixity: End fixity coefficient (1.0 = pinned-pinned, 4.0 = fixed-fixed)
    
    Returns:
        Euler column buckling stress σ_euler [Pa]
    
    Reference:
        σ_euler = π² * E * I / (A * (K*L)²)
        where K = 1/sqrt(end_fixity)
    """
    import numpy as _np
    
    L = stringer_length
    if L <= 0:
        return float('inf')
    
    # Effective length factor
    K = 1.0 / _np.sqrt(end_fixity)
    L_eff = K * L
    
    # Get stringer properties
    I = stringer_props.I_self
    A = stringer_props.area
    
    if A <= 0 or I <= 0:
        return float('inf')
    
    # Material modulus
    if stringer_props.material is not None:
        E = stringer_props.material.E_1
    else:
        E = 3.5e9  # Default to typical balsa
    
    # Euler buckling stress
    sigma_euler = _np.pi**2 * E * I / (A * L_eff**2)
    
    return float(sigma_euler)


def calculate_stringer_allowable_stress(
    stringer_props: 'StringerProperties',
    rib_spacing: float,
    end_fixity: float = 2.0,  # Assume semi-fixed ends at ribs
) -> tuple:
    """
    Calculate allowable stress for stringer considering all failure modes.
    
    Checks:
    1. Local crippling of critical element
    2. Global column buckling (Euler)
    3. Material yield
    
    Returns the minimum (most critical) allowable stress.
    
    Args:
        stringer_props: Stringer geometry and material
        rib_spacing: Distance between ribs [m]
        end_fixity: Column end fixity coefficient
    
    Returns:
        tuple: (allowable_stress, failure_mode)
        where failure_mode is "crippling", "column_buckling", or "yield"
    """
    import numpy as _np
    
    if stringer_props.material is None:
        return (float('inf'), "no_material")
    
    material = stringer_props.material
    
    # 1. Local crippling stress
    b_over_t = stringer_props.crippling_b_over_t
    
    # Determine edge condition based on section type
    if stringer_props.section_type == "rectangular":
        # Rectangular stringer attached at one edge
        edge_condition = "one_edge_free"
    elif stringer_props.section_type in ("L_section", "T_section"):
        # Outstanding flange is one_edge_free, web is both_edges_supported
        # Use one_edge_free (more critical) for conservative estimate
        edge_condition = "one_edge_free"
    else:
        edge_condition = "one_edge_free"
    
    sigma_crippling = calculate_stringer_crippling_stress(
        b_over_t=b_over_t,
        material=material,
        section_type=stringer_props.section_type,
        edge_condition=edge_condition,
    )
    
    # 2. Column buckling stress
    sigma_euler = calculate_stringer_column_buckling_stress(
        stringer_length=rib_spacing,
        stringer_props=stringer_props,
        end_fixity=end_fixity,
    )
    
    # 3. Material yield
    sigma_yield = material.sigma_1_compression
    
    # Find minimum (most critical)
    stresses = [
        (sigma_crippling, "crippling"),
        (sigma_euler, "column_buckling"),
        (sigma_yield, "yield"),
    ]
    
    min_stress, failure_mode = min(stresses, key=lambda x: x[0])
    
    return (float(min_stress), failure_mode)


def calculate_stringer_stress(
    bending_moment: float,
    box_height: float,
    stringer_props: 'StringerProperties',
    E_skin: float,
    t_skin: float,
    box_width: float,
) -> float:
    """
    Calculate compressive stress in stringer due to wing bending.
    
    Stringers are located on the top and bottom skins. They experience
    compressive stress when the wing bends (compression on top, tension on bottom).
    
    The stress in the stringer depends on its distance from the neutral axis
    and the strain compatibility with the skin.
    
    Args:
        bending_moment: Local bending moment [N*m]
        box_height: Height of wingbox [m]
        stringer_props: Stringer properties
        E_skin: Skin elastic modulus [Pa]
        t_skin: Skin thickness [m]
        box_width: Width of wingbox [m]
    
    Returns:
        Stringer compressive stress [Pa] (positive = compression)
    """
    import numpy as _np
    
    if stringer_props.material is None:
        return 0.0
    
    # Distance from neutral axis to stringer (approximately at skin surface)
    y_stringer = box_height / 2
    
    # Compute section properties including stringers
    # Simplified: assume stringers don't significantly shift neutral axis
    
    # Skin contribution to I
    A_skin = box_width * t_skin
    I_skin = 2 * A_skin * (box_height / 2)**2
    
    # Stringer contribution to I (use parallel axis theorem)
    n_stringers = stringer_props.count * 2  # Both top and bottom skins
    A_stringer = stringer_props.area
    I_stringer_self = stringer_props.I_self
    # Stringers at y = ±h/2 from neutral axis
    I_stringer_total = n_stringers * (I_stringer_self + A_stringer * y_stringer**2)
    
    # Total I (simplified - ignores spar contribution for stringer stress calc)
    I_total = I_skin + I_stringer_total
    
    # Curvature
    E_stringer = stringer_props.material.E_1
    
    # Weighted EI
    EI = E_skin * I_skin + E_stringer * I_stringer_total
    
    # Curvature from bending
    kappa = abs(bending_moment) / (EI + 1e-10)
    
    # Stress in stringer: σ = E * κ * y
    sigma_stringer = E_stringer * kappa * y_stringer
    
    return float(sigma_stringer)


# =============================================================================
# TORSION-BENDING SHEAR INTERACTION (Priority 2)
# =============================================================================
# The current implementation uses conservative direct addition of shear stresses:
#   τ_total = τ_bending + τ_torsion
#
# This is overly conservative because:
# 1. Bending shear varies around the perimeter (max at neutral axis, zero at skins)
# 2. Torsional shear is constant around the perimeter (Bredt's formula)
# 3. Front and rear spars have OPPOSITE sign torsional shear
#
# Proper combination:
# - At spar webs (near NA): τ_combined = τ_bending ± τ_torsion (sign depends on spar)
# - At skins: τ_combined ≈ τ_torsion (bending shear is small)
#
# References:
#   - Megson, T.H.G. "Aircraft Structures for Engineering Students" Ch. 17
#   - Bruhn, E.F. "Analysis and Design of Flight Vehicle Structures" Ch. A15
# =============================================================================

@dataclass
class WingboxShearDistribution:
    """Shear stress distribution around wingbox cross-section.
    
    Tracks shear stresses at key locations around the closed wingbox:
    - Front spar web (at neutral axis)
    - Rear spar web (at neutral axis)
    - Top skin (at mid-width)
    - Bottom skin (at mid-width)
    
    All stresses are in [Pa]. Sign convention:
    - Positive shear = clockwise flow around the section
    - For vertical shear (V > 0 causing nose-down rotation):
      * Front spar: positive shear flows downward
      * Rear spar: positive shear flows upward
    - For positive torque (nose-up):
      * Clockwise shear flow when viewed from root
    """
    # Bending shear components
    tau_bending_front_spar: float    # At front spar web, near neutral axis
    tau_bending_rear_spar: float     # At rear spar web, near neutral axis
    tau_bending_top_skin: float      # At top skin (typically small)
    tau_bending_bottom_skin: float   # At bottom skin (typically small)
    
    # Torsional shear components (constant magnitude, varies in sign)
    tau_torsion_front_spar: float    # Front spar (same sign as shear flow)
    tau_torsion_rear_spar: float     # Rear spar (opposite to front)
    tau_torsion_top_skin: float      # Top skin
    tau_torsion_bottom_skin: float   # Bottom skin
    
    # Combined shear stresses
    tau_combined_front_spar: float   # Maximum combined at front spar
    tau_combined_rear_spar: float    # Maximum combined at rear spar
    tau_combined_top_skin: float     # Combined at top skin
    tau_combined_bottom_skin: float  # Combined at bottom skin
    
    # Critical values for buckling checks
    tau_max_spar: float              # Maximum shear in either spar web
    tau_max_skin: float              # Maximum shear in either skin
    critical_location: str           # "front_spar", "rear_spar", "top_skin", "bottom_skin"


def calculate_wingbox_shear_distribution(
    shear_force: float,
    torque: float,
    box_width: float,
    box_height: float,
    t_spar: float,
    t_skin: float,
    front_spar_thickness: Optional[float] = None,
    rear_spar_thickness: Optional[float] = None,
) -> WingboxShearDistribution:
    """
    Calculate shear stress distribution around a closed wingbox section.
    
    Computes both bending shear and torsional shear at key locations,
    then combines them properly (not just adding magnitudes).
    
    Physics:
    
    Bending Shear (from vertical shear force V):
        For a closed thin-walled section, shear flow q = V * Q / I
        where Q = first moment of area above the cut.
        
        At spar webs (cutting at neutral axis): maximum shear
        At skins (cutting near top/bottom): minimum shear (Q ≈ 0)
        
        For symmetric box: q_web = V * (h/2 * w * t_skin) / I
        τ_web = q / t_spar
    
    Torsional Shear (from torque T):
        Bredt's formula: q_torsion = T / (2 * A_enclosed)
        τ_torsion = q_torsion / t
        
        Constant magnitude around perimeter, but direction follows
        the closed loop. For positive T (nose-up moment):
        - Front spar: upward flow
        - Rear spar: downward flow
    
    Combined:
        At each location: τ_combined = τ_bending + τ_torsion (with signs)
        Critical location is where |τ_combined| is maximum.
    
    Args:
        shear_force: Vertical shear force [N] (positive = upward on wing)
        torque: Torsional moment [N*m] (positive = nose-up)
        box_width: Width of wingbox [m]
        box_height: Height of wingbox [m]
        t_spar: Default spar thickness [m] (used if front/rear not specified)
        t_skin: Skin thickness [m]
        front_spar_thickness: Front spar thickness [m] (optional)
        rear_spar_thickness: Rear spar thickness [m] (optional)
    
    Returns:
        WingboxShearDistribution with all shear stress components
    """
    import numpy as _np
    
    # Use specific spar thicknesses if provided
    t_front = front_spar_thickness if front_spar_thickness is not None else t_spar
    t_rear = rear_spar_thickness if rear_spar_thickness is not None else t_spar
    
    w = box_width
    h = box_height
    
    # === Geometric properties ===
    # Second moment of area (approximation for thin-walled box)
    # I ≈ 2 * A_skin * (h/2)² + 2 * (1/12) * t_spar * h³
    I_skin = 2 * (w * t_skin) * (h / 2)**2
    I_spar = 2 * (1/12) * t_spar * h**3
    I_total = I_skin + I_spar
    
    # Enclosed area for Bredt's formula
    A_enclosed = w * h
    
    # === Bending Shear Distribution ===
    # At neutral axis (y=0), cutting through a spar web:
    # Q = first moment of area above cut = A_skin * (h/2)
    # For front spar, we cut through front spar, Q includes top skin contribution
    Q_at_NA = (w * t_skin) * (h / 2)
    
    # Shear flow at neutral axis
    q_bending_NA = abs(shear_force) * Q_at_NA / (I_total + 1e-10)
    
    # Shear stress in spar webs (at NA)
    # Total shear is split between front and rear spars based on thickness
    total_spar_thickness = t_front + t_rear
    
    # For symmetric loading, shear splits based on relative stiffness
    # Simplified: assume equal distribution if same thickness
    fraction_front = t_front / (total_spar_thickness + 1e-10)
    fraction_rear = t_rear / (total_spar_thickness + 1e-10)
    
    # Average shear stress formula: τ = 1.5 * V / A_web for parabolic distribution
    # At neutral axis (peak): τ_max = 1.5 * τ_avg
    A_web_total = (t_front + t_rear) * h
    tau_avg = abs(shear_force) / (A_web_total + 1e-10)
    
    # Peak bending shear at neutral axis of each spar
    # Note: in reality, the shear distributes between spars, but for a symmetric
    # box section, we approximate each spar carrying half the shear
    tau_bending_front = 1.5 * abs(shear_force) / (2 * t_front * h + 1e-10)
    tau_bending_rear = 1.5 * abs(shear_force) / (2 * t_rear * h + 1e-10)
    
    # Bending shear in skins is much smaller (Q approaches zero near top/bottom)
    # For thin skins, τ_skin_bending ≈ 0 at mid-chord
    tau_bending_top_skin = 0.0
    tau_bending_bottom_skin = 0.0
    
    # === Torsional Shear Distribution ===
    # Bredt's formula: q = T / (2 * A_enclosed)
    # τ = q / t
    q_torsion = torque / (2 * A_enclosed + 1e-10)
    
    # Torsional shear stress in each element
    tau_torsion_front = q_torsion / (t_front + 1e-10)
    tau_torsion_rear = q_torsion / (t_rear + 1e-10)
    tau_torsion_top = q_torsion / (t_skin + 1e-10)
    tau_torsion_bottom = q_torsion / (t_skin + 1e-10)
    
    # Sign convention for torsion:
    # Positive torque (nose-up) creates clockwise shear flow (viewed from root)
    # - Front spar: shear flows UP (positive)
    # - Top skin: shear flows BACKWARD (positive)
    # - Rear spar: shear flows DOWN (negative relative to front)
    # - Bottom skin: shear flows FORWARD (positive)
    
    # === Combined Shear ===
    # At spar webs, bending and torsion can add or subtract
    # Convention: positive bending shear is downward in both spars
    # Positive torsion shear is upward in front spar, downward in rear spar
    
    # For shear force V > 0 (upward on wing section):
    # - Front spar bending shear: positive (downward)
    # - Rear spar bending shear: positive (downward)
    
    # For torque T > 0 (nose-up):
    # - Front spar torsion: upward (opposes bending shear if V > 0)
    # - Rear spar torsion: downward (adds to bending shear if V > 0)
    
    # Combined (using absolute values for stress magnitude):
    # Front spar: bending and torsion oppose → subtract
    # Rear spar: bending and torsion add → add
    
    if torque >= 0:
        # Positive torque: front spar torsion opposes bending, rear adds
        tau_combined_front = abs(tau_bending_front - abs(tau_torsion_front))
        tau_combined_rear = tau_bending_rear + abs(tau_torsion_rear)
    else:
        # Negative torque: front spar torsion adds, rear opposes
        tau_combined_front = tau_bending_front + abs(tau_torsion_front)
        tau_combined_rear = abs(tau_bending_rear - abs(tau_torsion_rear))
    
    # For skins, bending shear is negligible, only torsion matters
    tau_combined_top = abs(tau_torsion_top)
    tau_combined_bottom = abs(tau_torsion_bottom)
    
    # === Find Critical Location ===
    shear_values = {
        'front_spar': tau_combined_front,
        'rear_spar': tau_combined_rear,
        'top_skin': tau_combined_top,
        'bottom_skin': tau_combined_bottom,
    }
    
    tau_max_spar = max(tau_combined_front, tau_combined_rear)
    tau_max_skin = max(tau_combined_top, tau_combined_bottom)
    critical_location = max(shear_values, key=shear_values.get)
    
    return WingboxShearDistribution(
        tau_bending_front_spar=tau_bending_front,
        tau_bending_rear_spar=tau_bending_rear,
        tau_bending_top_skin=tau_bending_top_skin,
        tau_bending_bottom_skin=tau_bending_bottom_skin,
        tau_torsion_front_spar=abs(tau_torsion_front),
        tau_torsion_rear_spar=abs(tau_torsion_rear),
        tau_torsion_top_skin=abs(tau_torsion_top),
        tau_torsion_bottom_skin=abs(tau_torsion_bottom),
        tau_combined_front_spar=tau_combined_front,
        tau_combined_rear_spar=tau_combined_rear,
        tau_combined_top_skin=tau_combined_top,
        tau_combined_bottom_skin=tau_combined_bottom,
        tau_max_spar=tau_max_spar,
        tau_max_skin=tau_max_skin,
        critical_location=critical_location,
    )


def calculate_improved_spar_shear(
    shear_force: float,
    torque: float,
    box_width: float,
    box_height: float,
    t_spar: float,
    t_skin: float,
) -> tuple:
    """
    Calculate improved combined spar shear stress using proper interaction.
    
    This replaces the conservative τ_total = τ_bending + τ_torsion with
    proper signed addition that accounts for front/rear spar differences.
    
    Args:
        shear_force: Vertical shear force [N]
        torque: Torsional moment [N*m]
        box_width: Width of wingbox [m]
        box_height: Height of wingbox [m]
        t_spar: Spar web thickness [m]
        t_skin: Skin thickness [m]
    
    Returns:
        tuple: (tau_max_spar, tau_max_skin, reduction_factor)
        - tau_max_spar: Maximum combined shear in spar webs [Pa]
        - tau_max_skin: Maximum combined shear in skins [Pa]
        - reduction_factor: Ratio of improved/conservative (typically 0.8-0.95)
    """
    dist = calculate_wingbox_shear_distribution(
        shear_force=shear_force,
        torque=torque,
        box_width=box_width,
        box_height=box_height,
        t_spar=t_spar,
        t_skin=t_skin,
    )
    
    # Conservative estimate (old method)
    A_web = 2 * box_height * t_spar
    A_enclosed = box_width * box_height
    tau_bending_old = 1.5 * abs(shear_force) / (A_web + 1e-10)
    tau_torsion_old = abs(torque) / (2 * A_enclosed * t_spar + 1e-10)
    tau_conservative = tau_bending_old + tau_torsion_old
    
    # Improved estimate
    tau_improved = dist.tau_max_spar
    
    # Reduction factor (how much less conservative the new method is)
    if tau_conservative > 0:
        reduction_factor = tau_improved / tau_conservative
    else:
        reduction_factor = 1.0
    
    return (dist.tau_max_spar, dist.tau_max_skin, reduction_factor)


# =============================================================================
# POST-BUCKLING ANALYSIS (Priority 4)
# =============================================================================
# Many aircraft structures allow post-buckled skins. After initial buckling,
# load redistributes to stringers and spar caps via "tension field" action.
# This allows significant weight savings but requires checking additional
# failure modes:
#   - Stringer column buckling under redistributed load
#   - Stringer crippling (already implemented in Priority 6)
#   - Skin-stringer separation (peel failure)
#   - Tension field in shear (diagonal tension)
#
# References:
#   - Bruhn, E.F. "Analysis and Design of Flight Vehicle Structures" Ch. C7, C9
#   - NASA SP-8007 "Buckling of Thin-Walled Circular Cylinders"
#   - Niu, M.C.Y. "Airframe Structural Design" Ch. 14
# =============================================================================

@dataclass
class PostBucklingConfig:
    """Configuration for post-buckling analysis.
    
    When post-buckling is enabled, skin panels are allowed to buckle
    (initial buckling), and the structure relies on:
    1. Effective width of buckled skin at panel edges
    2. Stringers to carry redistributed load
    3. Tension field action in shear-buckled webs
    
    Attributes:
        enabled: Enable post-buckling analysis (default: False = initial buckling is failure)
        max_skin_buckle_ratio: Maximum σ_applied / σ_cr allowed in post-buckling (typically 2-4)
        tension_field_factor: Factor to increase shear allowable in tension field (1.5-3x)
        separation_margin: Required margin against skin-stringer separation (typically 2.0)
        require_stringers: If True, post-buckling only enabled if stringers present
    """
    enabled: bool = False
    max_skin_buckle_ratio: float = 3.0       # Max ratio of applied/critical stress
    tension_field_factor: float = 2.0         # Factor for post-buckled shear allowable
    separation_margin: float = 2.0            # Required margin for skin-stringer separation
    require_stringers: bool = True            # Require stringers for post-buckling


def calculate_effective_width(
    panel_width: float,
    sigma_cr: float,
    sigma_applied: float,
    method: str = "von_karman",
) -> float:
    """
    Calculate effective width of buckled skin panel.
    
    After initial buckling, a buckled skin panel can still carry load
    at its edges where it is supported by stringers or spar caps.
    The "effective width" is the equivalent width of unbuckled panel
    that would carry the same edge load.
    
    Args:
        panel_width: Original panel width [m]
        sigma_cr: Critical buckling stress [Pa]
        sigma_applied: Applied stress [Pa] (must be > sigma_cr for buckling)
        method: "von_karman" (classic) or "winter" (modified for imperfections)
    
    Returns:
        Effective width [m]
    
    Physics:
        von Kármán formula:
            b_eff = b * sqrt(sigma_cr / sigma_applied)
        
        This gives the width of panel at σ_cr that would carry same
        edge load as the full buckled panel.
    
    Reference:
        von Kármán, T., Sechler, E.E., and Donnell, L.H. (1932)
        "The Strength of Thin Plates in Compression"
        ASME Transactions, Vol. 54
    """
    import numpy as _np
    
    b = panel_width
    
    if sigma_applied <= 0 or sigma_cr <= 0:
        return b  # No load or invalid input
    
    if sigma_applied <= sigma_cr:
        # Not buckled yet - full width effective
        return b
    
    # Panel is buckled (σ_applied > σ_cr)
    if method == "von_karman":
        # Classic von Kármán formula
        b_eff = b * _np.sqrt(sigma_cr / sigma_applied)
    elif method == "winter":
        # Winter formula (accounts for imperfections, more conservative)
        # b_eff = b * sqrt(σ_cr/σ) * (1 - 0.22 * sqrt(σ_cr/σ))
        ratio = _np.sqrt(sigma_cr / sigma_applied)
        b_eff = b * ratio * (1.0 - 0.22 * ratio)
    else:
        b_eff = b * _np.sqrt(sigma_cr / sigma_applied)
    
    # Ensure b_eff is positive and not greater than b
    b_eff = max(0.0, min(b_eff, b))
    
    return float(b_eff)


def calculate_effective_width_symbolic(
    panel_width: float,
    sigma_cr,  # Can be CasADi symbolic
    sigma_applied,  # Can be CasADi symbolic
    method: str = "von_karman",
):
    """
    Calculate effective width - CasADi compatible version.
    
    Args:
        panel_width: Original panel width [m]
        sigma_cr: Critical buckling stress [Pa] (can be symbolic)
        sigma_applied: Applied stress [Pa] (can be symbolic)
        method: Calculation method
    
    Returns:
        Effective width [m] - same type as inputs
    """
    b = panel_width
    
    # Smooth transition: use softplus-like function for max
    # b_eff = b * sqrt(sigma_cr / sigma_applied) when buckled
    # b_eff = b when not buckled
    
    if method == "von_karman":
        ratio = sigma_cr / (sigma_applied + 1e-10)
        # Clamp ratio to [0, 1] using smooth min
        ratio_clamped = np.minimum(ratio, 1.0)
        ratio_clamped = np.maximum(ratio_clamped, 0.01)  # Avoid sqrt(0)
        b_eff = b * np.sqrt(ratio_clamped)
    else:
        # Default to von Kármán
        ratio = sigma_cr / (sigma_applied + 1e-10)
        ratio_clamped = np.minimum(ratio, 1.0)
        ratio_clamped = np.maximum(ratio_clamped, 0.01)
        b_eff = b * np.sqrt(ratio_clamped)
    
    return b_eff


def calculate_post_buckled_skin_load(
    panel_width: float,
    thickness: float,
    sigma_cr: float,
    sigma_applied: float,
    stringer_props: Optional['StringerProperties'],
    E_skin: float,
) -> dict:
    """
    Calculate load distribution in post-buckled skin panel.
    
    After buckling, load redistributes from the buckled skin to:
    1. Effective width of skin at edges (carries load at σ_cr)
    2. Stringers (carry additional load above σ_cr capacity)
    
    Args:
        panel_width: Width between stringers [m]
        thickness: Skin thickness [m]
        sigma_cr: Critical buckling stress [Pa]
        sigma_applied: Applied stress [Pa]
        stringer_props: Stringer properties (can be None)
        E_skin: Skin elastic modulus [Pa]
    
    Returns:
        dict with:
            'b_eff': Effective width [m]
            'skin_load': Load carried by effective skin [N]
            'stringer_load': Load per stringer [N]
            'stringer_stress': Stress in each stringer [Pa]
            'is_post_buckled': True if panel is buckled
    """
    import numpy as _np
    
    b = panel_width
    t = thickness
    
    # Total load per unit span = σ * A = σ * (b * t)
    P_total = sigma_applied * b * t
    
    if sigma_applied <= sigma_cr:
        # Not buckled - skin carries all load
        return {
            'b_eff': b,
            'skin_load': P_total,
            'stringer_load': 0.0,
            'stringer_stress': 0.0,
            'is_post_buckled': False,
        }
    
    # Post-buckled: calculate effective width
    b_eff = calculate_effective_width(b, sigma_cr, sigma_applied)
    
    # Load carried by effective skin width at σ_cr
    P_skin = sigma_cr * b_eff * t
    
    # Remaining load must be carried by stringers
    P_remaining = P_total - P_skin
    
    # Calculate stringer load and stress
    if stringer_props is not None and stringer_props.count > 0:
        # Assume load distributes equally to stringers on each side of panel
        # For a panel between two stringers, each stringer takes half
        # But each stringer is shared with adjacent panel, so net is P_remaining/2 per stringer
        n_stringers_per_panel_edge = 2  # One on each side
        stringer_area = stringer_props.area
        
        # Each stringer gets half the excess load from this panel
        # (and half from the adjacent panel on the other side)
        P_per_stringer = P_remaining / n_stringers_per_panel_edge
        
        # Stringer stress
        sigma_stringer = P_per_stringer / (stringer_area + 1e-10)
    else:
        # No stringers - all load must go somewhere (edge support)
        P_per_stringer = P_remaining / 2  # Conceptually to spar caps
        sigma_stringer = P_remaining / (b_eff * t + 1e-10)  # Edge stress
    
    return {
        'b_eff': float(b_eff),
        'skin_load': float(P_skin),
        'stringer_load': float(P_per_stringer),
        'stringer_stress': float(sigma_stringer),
        'is_post_buckled': True,
    }


def calculate_tension_field_allowable(
    tau_cr: float,
    tension_field_factor: float = 2.0,
    web_height: float = 0.0,
    stiffener_spacing: float = 0.0,
) -> float:
    """
    Calculate post-buckled shear allowable using tension field theory.
    
    After a spar web or skin panel buckles in shear, it develops a
    "tension field" - diagonal tension bands that continue to carry load.
    The post-buckled allowable is higher than initial buckling stress.
    
    Args:
        tau_cr: Initial shear buckling stress [Pa]
        tension_field_factor: Multiplier for post-buckled allowable (1.5-3x typical)
        web_height: Height of web [m] (for detailed calculation)
        stiffener_spacing: Distance between stiffeners [m] (for detailed calculation)
    
    Returns:
        Post-buckled shear allowable [Pa]
    
    Notes:
        The tension field factor depends on:
        - Stiffener spacing (closer = higher factor)
        - Web thickness (thinner = more tension field development)
        - Flange/cap rigidity (stiffer = better load transfer)
        
        Typical values:
        - 1.5x for widely spaced stiffeners
        - 2.0x for moderate spacing
        - 3.0x for closely spaced stiffeners
    
    Reference:
        Wagner, H. (1931) "Flat Sheet Metal Girders with Very Thin Webs"
        NACA TM 604-606
    """
    if tau_cr <= 0:
        return 0.0
    
    # Simple approach: apply factor to initial buckling stress
    tau_allowable_post = tau_cr * tension_field_factor
    
    # Could add more detailed calculation based on web geometry
    # but for practical purposes the factor approach is sufficient
    
    return float(tau_allowable_post)


def calculate_skin_stringer_separation_margin(
    sigma_skin: float,
    sigma_cr_skin: float,
    stringer_props: Optional['StringerProperties'],
    bond_strength: float = 5.0e6,  # Typical adhesive bond strength [Pa]
    peel_factor: float = 3.0,       # Peel stress concentration factor
) -> float:
    """
    Calculate margin against skin-stringer separation in post-buckled panel.
    
    When skin buckles between stringers, it develops out-of-plane deformation
    that creates peel stress at the skin-stringer bond. If peel stress exceeds
    bond strength, separation occurs.
    
    Args:
        sigma_skin: Applied stress in skin [Pa]
        sigma_cr_skin: Critical buckling stress of skin [Pa]
        stringer_props: Stringer properties
        bond_strength: Adhesive/fastener bond strength [Pa]
        peel_factor: Stress concentration at bond line edges
    
    Returns:
        Separation margin = bond_allowable / (peel_stress * factor)
        Margin > 1.0 means safe against separation
    
    Notes:
        Peel stress is approximated as:
            σ_peel ≈ σ_applied * (buckle_amplitude / panel_width)
        
        Buckle amplitude is related to post-buckling ratio:
            amplitude ≈ t * sqrt((σ/σ_cr) - 1)
        
        This is a simplified model - real separation analysis is complex.
    
    Reference:
        Niu, M.C.Y. "Airframe Structural Design" Ch. 14
    """
    import numpy as _np
    
    if sigma_cr_skin <= 0 or sigma_skin <= 0:
        return float('inf')  # No load, no separation risk
    
    if sigma_skin <= sigma_cr_skin:
        return float('inf')  # Not buckled, no peel stress
    
    # Post-buckling ratio
    R = sigma_skin / sigma_cr_skin
    
    # Estimate peel stress (simplified model)
    # As buckling develops, out-of-plane deformation creates peel
    # Peel stress scales with sqrt(R - 1) and applied stress
    peel_stress = sigma_skin * 0.1 * _np.sqrt(R - 1)  # Empirical factor
    
    # Apply stress concentration at bond edges
    peel_stress_peak = peel_stress * peel_factor
    
    # Margin against separation
    margin = bond_strength / (peel_stress_peak + 1e-10)
    
    return float(margin)


def calculate_post_buckling_margin(
    sigma_applied: float,
    sigma_cr: float,
    tau_applied: float,
    tau_cr: float,
    stringer_props: Optional['StringerProperties'],
    rib_spacing: float,
    config: Optional['PostBucklingConfig'] = None,
    skin_material: Optional['StructuralMaterial'] = None,
) -> dict:
    """
    Calculate comprehensive post-buckling margins for a skin panel.
    
    Checks all post-buckling failure modes:
    1. Maximum post-buckling ratio (σ/σ_cr not too high)
    2. Stringer can carry redistributed load without failure
    3. Tension field shear capacity (if shear-buckled)
    4. Skin-stringer separation
    
    Args:
        sigma_applied: Applied compressive stress in skin [Pa]
        sigma_cr: Critical buckling stress [Pa]
        tau_applied: Applied shear stress [Pa]
        tau_cr: Critical shear buckling stress [Pa]
        stringer_props: Stringer properties
        rib_spacing: Distance between ribs [m]
        config: Post-buckling configuration
        skin_material: Skin material properties
    
    Returns:
        dict with all post-buckling margins and status
    """
    import numpy as _np
    
    if config is None:
        config = PostBucklingConfig()
    
    # Initialize results
    result = {
        'is_buckled_compression': False,
        'is_buckled_shear': False,
        'post_buckling_ratio': 0.0,
        'effective_width_ratio': 1.0,
        'stringer_margin': float('inf'),
        'stringer_failure_mode': None,
        'tension_field_margin': float('inf'),
        'separation_margin': float('inf'),
        'overall_margin': float('inf'),
        'is_post_buckling_feasible': True,
    }
    
    # Check compression post-buckling
    if sigma_cr > 0 and sigma_applied > sigma_cr:
        result['is_buckled_compression'] = True
        result['post_buckling_ratio'] = sigma_applied / sigma_cr
        
        # Check max allowable post-buckling ratio
        if result['post_buckling_ratio'] > config.max_skin_buckle_ratio:
            result['is_post_buckling_feasible'] = False
            result['overall_margin'] = config.max_skin_buckle_ratio / result['post_buckling_ratio']
        
        # Calculate effective width
        b_eff_ratio = _np.sqrt(sigma_cr / sigma_applied)
        result['effective_width_ratio'] = b_eff_ratio
        
        # Check stringer capacity for redistributed load
        if stringer_props is not None and stringer_props.count > 0 and stringer_props.material is not None:
            # Get stringer allowable
            sigma_stringer_allow, mode = calculate_stringer_allowable_stress(
                stringer_props=stringer_props,
                rib_spacing=rib_spacing,
                end_fixity=2.0,
            )
            result['stringer_failure_mode'] = mode
            
            # Estimate stringer stress from post-buckling redistribution
            # Stringer stress ≈ skin stress at same location (strain compatibility)
            # Plus additional stress from load redistribution
            redistribution_factor = 1.0 + 0.5 * (result['post_buckling_ratio'] - 1.0)
            sigma_stringer_applied = sigma_applied * redistribution_factor
            
            result['stringer_margin'] = sigma_stringer_allow / (sigma_stringer_applied + 1e-10)
            
            if result['stringer_margin'] < 1.0:
                result['is_post_buckling_feasible'] = False
        elif config.require_stringers:
            # No stringers but they're required for post-buckling
            result['is_post_buckling_feasible'] = False
            result['stringer_margin'] = 0.0
        
        # Check skin-stringer separation
        result['separation_margin'] = calculate_skin_stringer_separation_margin(
            sigma_skin=sigma_applied,
            sigma_cr_skin=sigma_cr,
            stringer_props=stringer_props,
        )
        if result['separation_margin'] < config.separation_margin:
            result['is_post_buckling_feasible'] = False
    
    # Check shear post-buckling (tension field)
    if tau_cr > 0 and tau_applied > tau_cr:
        result['is_buckled_shear'] = True
        
        # Calculate tension field allowable
        tau_allowable_post = calculate_tension_field_allowable(
            tau_cr=tau_cr,
            tension_field_factor=config.tension_field_factor,
        )
        
        result['tension_field_margin'] = tau_allowable_post / (tau_applied + 1e-10)
        
        if result['tension_field_margin'] < 1.0:
            result['is_post_buckling_feasible'] = False
    
    # Calculate overall margin (minimum of all margins)
    margins = [
        result['stringer_margin'],
        result['tension_field_margin'],
        result['separation_margin'] / config.separation_margin,  # Normalize to required margin
    ]
    
    # Add post-buckling ratio margin
    if result['is_buckled_compression']:
        pb_ratio_margin = config.max_skin_buckle_ratio / (result['post_buckling_ratio'] + 1e-10)
        margins.append(pb_ratio_margin)
    
    result['overall_margin'] = min(margins)
    
    return result


@dataclass
class PostBucklingResult:
    """Results from post-buckling analysis at each spanwise station.
    
    Contains detailed post-buckling state and margins for each station.
    """
    y: np.ndarray                          # Spanwise positions [m]
    
    # Buckling state
    is_buckled_compression: np.ndarray     # True if skin is compression-buckled
    is_buckled_shear: np.ndarray           # True if skin is shear-buckled
    post_buckling_ratio: np.ndarray        # σ_applied / σ_cr (1.0 = at buckling)
    
    # Effective properties
    effective_width_ratio: np.ndarray      # b_eff / b_original
    
    # Margins
    stringer_margin: np.ndarray            # Stringer capacity margin
    tension_field_margin: np.ndarray       # Shear post-buckling margin
    separation_margin: np.ndarray          # Skin-stringer separation margin
    overall_margin: np.ndarray             # Minimum of all margins
    
    # Summary
    min_stringer_margin: float
    min_tension_field_margin: float
    min_separation_margin: float
    min_overall_margin: float
    is_post_buckling_feasible: bool
    
    # Configuration used
    config: PostBucklingConfig = None


@dataclass
class StructuralAnalysisResult:
    """Results from structural analysis."""
    # Spanwise coordinates
    y: np.ndarray                 # Spanwise positions [m]
    
    # Deflection and derivatives
    displacement: np.ndarray      # Vertical displacement [m]
    slope: np.ndarray             # Rotation [rad]
    curvature: np.ndarray         # Curvature [1/m]
    
    # Internal forces
    bending_moment: np.ndarray    # [N*m]
    shear_force: np.ndarray       # [N]
    
    # Load distributions (for visualization)
    aero_load: np.ndarray         # Aerodynamic lift per unit span [N/m]
    inertial_load: np.ndarray     # Inertial (weight) load per unit span [N/m] (negative = downward)
    net_load: np.ndarray          # Net load = aero + inertial [N/m]
    
    # Stresses
    sigma_spar: np.ndarray        # Spar bending stress [Pa]
    sigma_skin: np.ndarray        # Skin bending stress [Pa]
    max_stress_ratio: np.ndarray  # Stress / allowable
    
    # Buckling - skin panels
    buckling_margin: np.ndarray   # sigma_cr / sigma_applied (skin)
    
    # Section properties
    EI: np.ndarray                # Bending stiffness [N*m^2]
    GJ: np.ndarray                # Torsional stiffness [N*m^2]
    
    # Summary metrics (required fields)
    mass_kg: float
    tip_deflection_m: float
    max_stress_MPa: float
    min_buckling_margin: float
    stress_margin: float
    is_feasible: bool
    
    # Optional fields with defaults (must come after required fields)
    spar_buckling_margin: Optional[np.ndarray] = None   # tau_cr / tau_applied
    tau_spar: Optional[np.ndarray] = None               # Bending shear stress in spar webs [Pa]
    
    # Timoshenko beam theory fields
    shear_strain: Optional[np.ndarray] = None           # Shear strain γ = V/(κGA) [rad]
    shear_displacement: Optional[np.ndarray] = None     # Shear contribution to deflection [m]
    shear_stiffness: Optional[np.ndarray] = None        # κGA distribution [N]
    shear_deformation_ratio: Optional[float] = None     # Estimated δ_shear / δ_total
    
    # Torsion results
    torque: Optional[np.ndarray] = None                 # Torque distribution [N*m]
    tau_torsion_skin: Optional[np.ndarray] = None       # Torsional shear stress in skins [Pa]
    tau_torsion_spar: Optional[np.ndarray] = None       # Torsional shear stress in spar webs [Pa]
    tau_total_spar: Optional[np.ndarray] = None         # Combined shear in spar webs (bending + torsion) [Pa]
    min_torsion_margin: Optional[float] = None          # Min margin for combined shear buckling
    
    # Skin shear buckling (from torsion) - Priority 3 enhancement
    skin_shear_buckling_margin: Optional[np.ndarray] = None  # tau_cr_skin / tau_torsion_skin
    min_skin_shear_buckling_margin: Optional[float] = None
    
    # Biaxial stress interaction (σ + τ combined) - Priority 5 enhancement
    combined_buckling_margin: Optional[np.ndarray] = None  # Combined σ + τ interaction margin
    min_combined_buckling_margin: Optional[float] = None
    
    curvature_factors: Optional[np.ndarray] = None      # Curvature enhancement factors
    panel_widths: Optional[np.ndarray] = None           # Effective panel widths [m]
    min_spar_buckling_margin: Optional[float] = None
    factor_of_safety: float = 1.5  # Store FOS for plotting reference
    
    # Rib failure margins (at each rib location)
    rib_shear_buckling_margin: Optional[np.ndarray] = None  # tau_cr / tau_applied for ribs
    rib_crushing_margin: Optional[np.ndarray] = None        # sigma_allowable / sigma_bearing
    min_rib_buckling_margin: Optional[float] = None
    min_rib_crushing_margin: Optional[float] = None
    
    # Stringer crippling (Priority 6 enhancement)
    stringer_crippling_margin: Optional[np.ndarray] = None  # sigma_allowable / sigma_applied for stringers
    min_stringer_crippling_margin: Optional[float] = None
    stringer_failure_mode: Optional[str] = None              # "crippling", "column_buckling", or "yield"
    
    # Post-buckling analysis (Priority 4 enhancement)
    post_buckling_result: Optional['PostBucklingResult'] = None  # Full post-buckling analysis results
    post_buckling_enabled: bool = False                           # Whether post-buckling was analyzed
    is_post_buckling_feasible: Optional[bool] = None              # True if post-buckling design is valid
    min_post_buckling_margin: Optional[float] = None              # Minimum overall post-buckling margin
    
    # Timoshenko beam identification (Priority 1)
    is_timoshenko: bool = False                                   # True if solved with full Timoshenko
    shear_deflection_contribution: Optional[float] = None         # Fraction of tip deflection from shear
    
    # Twist constraint fields
    tip_twist_deg: Optional[float] = None                          # Total tip twist (bending + torsion) [deg]
    max_tip_twist_deg: Optional[float] = None                      # Max allowable tip twist [deg]
    twist_margin: Optional[float] = None                           # max_tip_twist / tip_twist (>1 = OK)
    
    # Mass breakdown and assumptions
    mass_breakdown: Optional[Dict[str, float]] = None   # Component masses in kg
    assumptions: Optional[List[str]] = None             # List of modeling assumptions
    
    def as_dict(self) -> Dict[str, Any]:
        """Export results as dictionary for UI/serialization."""
        def to_list(arr):
            if arr is None:
                return None
            if hasattr(arr, 'tolist'):
                return arr.tolist()
            return list(arr) if hasattr(arr, '__iter__') else arr
        
        return {
            "y": to_list(self.y),
            "displacement": to_list(self.displacement),
            "slope": to_list(self.slope),  # Rotation [rad] for twist visualization
            "bending_moment": to_list(self.bending_moment),
            "shear_force": to_list(self.shear_force),
            "aero_load": to_list(self.aero_load),
            "inertial_load": to_list(self.inertial_load),
            "net_load": to_list(self.net_load),
            "sigma_spar": to_list(self.sigma_spar),
            "sigma_skin": to_list(self.sigma_skin),
            "max_stress_ratio": to_list(self.max_stress_ratio),
            "buckling_margin": to_list(self.buckling_margin),
            "spar_buckling_margin": to_list(self.spar_buckling_margin),
            "tau_spar": to_list(self.tau_spar),
            # Timoshenko beam theory fields
            "shear_strain": to_list(self.shear_strain),
            "shear_displacement": to_list(self.shear_displacement),
            "shear_stiffness": to_list(self.shear_stiffness),
            "shear_deformation_ratio": float(self.shear_deformation_ratio) if self.shear_deformation_ratio else None,
            "torque": to_list(self.torque),
            "tau_torsion_skin": to_list(self.tau_torsion_skin),
            "tau_torsion_spar": to_list(self.tau_torsion_spar),
            "tau_total_spar": to_list(self.tau_total_spar),
            "min_torsion_margin": float(self.min_torsion_margin) if self.min_torsion_margin else None,
            # Skin shear buckling (Priority 3)
            "skin_shear_buckling_margin": to_list(self.skin_shear_buckling_margin),
            "min_skin_shear_buckling_margin": float(self.min_skin_shear_buckling_margin) if self.min_skin_shear_buckling_margin else None,
            # Biaxial interaction (Priority 5)
            "combined_buckling_margin": to_list(self.combined_buckling_margin),
            "min_combined_buckling_margin": float(self.min_combined_buckling_margin) if self.min_combined_buckling_margin else None,
            "curvature_factors": to_list(self.curvature_factors),
            "panel_widths": to_list(self.panel_widths),
            "rib_shear_buckling_margin": to_list(self.rib_shear_buckling_margin),
            "rib_crushing_margin": to_list(self.rib_crushing_margin),
            "EI": to_list(self.EI),
            "GJ": to_list(self.GJ),
            "mass_kg": float(self.mass_kg),
            "tip_deflection_m": float(self.tip_deflection_m),
            "max_stress_MPa": float(self.max_stress_MPa),
            "min_buckling_margin": float(self.min_buckling_margin),
            "min_spar_buckling_margin": float(self.min_spar_buckling_margin) if self.min_spar_buckling_margin else None,
            "min_rib_buckling_margin": float(self.min_rib_buckling_margin) if self.min_rib_buckling_margin else None,
            "min_rib_crushing_margin": float(self.min_rib_crushing_margin) if self.min_rib_crushing_margin else None,
            # Stringer crippling (Priority 6)
            "stringer_crippling_margin": to_list(self.stringer_crippling_margin),
            "min_stringer_crippling_margin": float(self.min_stringer_crippling_margin) if self.min_stringer_crippling_margin else None,
            "stringer_failure_mode": self.stringer_failure_mode,
            # Post-buckling (Priority 4)
            "post_buckling_enabled": bool(self.post_buckling_enabled),
            "is_post_buckling_feasible": bool(self.is_post_buckling_feasible) if self.is_post_buckling_feasible is not None else None,
            "min_post_buckling_margin": float(self.min_post_buckling_margin) if self.min_post_buckling_margin else None,
            # Timoshenko (Priority 1)
            "is_timoshenko": bool(self.is_timoshenko),
            "shear_deflection_contribution": float(self.shear_deflection_contribution) if self.shear_deflection_contribution else None,
            # Twist constraint
            "tip_twist_deg": float(self.tip_twist_deg) if self.tip_twist_deg is not None else None,
            "max_tip_twist_deg": float(self.max_tip_twist_deg) if self.max_tip_twist_deg is not None else None,
            "twist_margin": float(self.twist_margin) if self.twist_margin is not None else None,
            "stress_margin": float(self.stress_margin),
            "is_feasible": bool(self.is_feasible),
            "factor_of_safety": float(self.factor_of_safety),
            "mass_breakdown": self.mass_breakdown,
            "assumptions": self.assumptions,
        }


class WingBoxBeam(asb.ImplicitAnalysis):
    """
    Structural model for a tapered wingbox beam.
    
    Models the wing structure as a cantilever beam with box cross-section
    (front spar, rear spar, top skin, bottom skin). 
    
    Uses Euler-Bernoulli beam theory for the implicit analysis (optimization),
    with Timoshenko shear deformation correction applied in post-processing
    via get_result(). This provides accurate deflections for both slender
    and thick beams while maintaining gradient-based optimization capability.
    
    For slender beams (L/h > 15), shear deformation is negligible and the
    solution approaches pure Euler-Bernoulli. For thick beams typical of
    flying wings, shear deformation can add 5-15% to tip deflection.
    
    TODO: Implement full Timoshenko beam formulation as a separate class
    (TimoshenkoWingBoxBeam) that solves the coupled ψ-w system directly
    within CasADi. This would replace the current E-B + post-correction
    approach with a unified Timoshenko solver. The current implementation
    is a pragmatic intermediate step - the full implementation should treat
    bending rotation (ψ) and deflection (w) as independent optimization
    variables with proper coupled constraints.
    
    Reference implementations:
    - AeroSandbox TubeSparBendingStructure
    - OpenAeroStruct wingbox model (oas_wb_mdolab.pdf)
    - Timoshenko, S.P. (1921) "On the correction for shear..."
    """
    
    @asb.ImplicitAnalysis.initialize
    def __init__(
        self,
        sections: List[WingBoxSection],
        spar_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
        skin_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
        spar_material: StructuralMaterial,
        skin_material: StructuralMaterial,
        lift_distribution: Callable[[np.ndarray], np.ndarray],
        moment_distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rib_positions: Optional[List[float]] = None,
        n_points: int = 50,
        EI_guess: Optional[float] = None,
        GJ_guess: Optional[float] = None,
        # New parameters for enhanced buckling analysis
        stringer_props: Optional[StringerProperties] = None,
        rib_props: Optional['RibProperties'] = None,
        boundary_condition: str = "semi_restrained",
        include_curvature: bool = True,
        airfoils: Optional[List[Any]] = None,  # For curvature calculation
        # Control surface and fastener parameters
        control_surface_props: Optional[List['ControlSurfaceProperties']] = None,
        fastener_adhesive_fraction: float = 0.10,  # 10% mass adder for fasteners/adhesive
        # Post-buckling analysis (Priority 4)
        post_buckling_config: Optional['PostBucklingConfig'] = None,
    ):
        """
        Initialize wingbox beam analysis.
        
        Args:
            sections: List of WingBoxSection defining geometry at span stations
            spar_thickness: Spar web thickness [m] - constant or function of y
            skin_thickness: Skin panel thickness [m] - constant or function of y
            spar_material: Material properties for spars
            skin_material: Material properties for skins
            lift_distribution: Function q(y) giving lift per unit span [N/m]
            moment_distribution: Function m(y) giving pitching moment per span [N*m/m]
            rib_positions: List of spanwise positions of ribs [m] for buckling
            n_points: Number of discretization points
            EI_guess: Initial guess for bending stiffness (for optimizer scaling)
            GJ_guess: Initial guess for torsional stiffness (for optimizer scaling)
            stringer_props: StringerProperties for smeared stiffener analysis
            rib_props: RibProperties for rib thickness and material
            boundary_condition: "simply_supported", "semi_restrained", or "clamped"
            include_curvature: Whether to include curvature effects in buckling
            airfoils: List of airfoil objects (AeroSandbox) for curvature calculation
            control_surface_props: List of ControlSurfaceProperties for elevon/flap mass
            fastener_adhesive_fraction: Mass adder fraction for fasteners/adhesive (default 10%)
            post_buckling_config: PostBucklingConfig for post-buckling analysis (Priority 4)
        
        Note: The 'opti' argument is handled by the @ImplicitAnalysis.initialize decorator.
              Pass opti=your_opti as a keyword argument if coupling with other analyses.
        """
        self.sections = sections
        self.spar_material = spar_material
        self.skin_material = skin_material
        self.n_points = n_points
        
        # Store enhanced buckling parameters
        self._stringer_props = stringer_props
        self._rib_props = rib_props
        self._boundary_condition = boundary_condition
        self._include_curvature = include_curvature
        self._airfoils = airfoils
        
        # Store control surface and fastener parameters
        self._control_surface_props = control_surface_props or []
        self._fastener_adhesive_fraction = fastener_adhesive_fraction
        
        # Store post-buckling config (Priority 4)
        self._post_buckling_config = post_buckling_config or PostBucklingConfig()
        
        # Extract spanwise coordinates from sections
        section_y = np.array([s.y for s in sections])
        self.length = float(np.max(section_y))
        
        if self.length <= 0:
            raise ValueError("Sections must have positive span extent")
        
        # Discretize along span
        y = np.linspace(0, self.length, n_points)
        self.y = y
        N = n_points
        
        # Interpolate section properties to discretization points
        box_width = np.interp(y, section_y, np.array([s.box_width for s in sections]))
        box_height = np.interp(y, section_y, np.array([s.box_height for s in sections]))
        
        self.box_width = box_width
        self.box_height = box_height
        
        # Store chord and spar positions for LE/TE mass calculation
        self._chord = np.interp(y, section_y, np.array([s.chord for s in sections]))
        self._front_spar_xsi = np.interp(y, section_y, np.array([s.front_spar_xsi for s in sections]))
        self._rear_spar_xsi = np.interp(y, section_y, np.array([s.rear_spar_xsi for s in sections]))
        
        # Evaluate thickness functions
        if callable(spar_thickness):
            t_spar = spar_thickness(y)
        else:
            t_spar = spar_thickness * np.ones_like(y)
        
        if callable(skin_thickness):
            t_skin = skin_thickness(y)
        else:
            t_skin = skin_thickness * np.ones_like(y)
        
        self.t_spar = t_spar
        self.t_skin = t_skin
        
        # === Cross-sectional Properties ===
        h = box_height  # Height
        w = box_width   # Width
        
        # Material properties (orthotropic handling)
        # Spar bending modulus: grain runs spanwise -> use E_1
        E_spar = spar_material.get_bending_modulus(grain_spanwise=True)
        E_skin = skin_material.get_bending_modulus(grain_spanwise=True)
        G_spar = spar_material.get_shear_modulus()
        G_skin = skin_material.get_shear_modulus()
        
        # Bending inertia I_xx (about horizontal axis)
        # Skin contribution: I = A * d^2 where d = h/2
        A_skin = w * t_skin
        I_skin = 2 * A_skin * (h / 2) ** 2  # Top + bottom skins
        
        # Spar contribution: I = (1/12) * t * h^3 for each spar web
        I_spar = 2 * (1 / 12) * t_spar * h ** 3
        
        # === Stringer contribution to bending stiffness ===
        # Stringers are longitudinal stiffeners attached to top/bottom skins
        # They contribute to bending stiffness via parallel axis theorem:
        #   I_stringer = n * (I_self + A * y^2)
        # where y is the distance from neutral axis to stringer centroid
        I_stringer = np.zeros_like(h)
        E_stringer = E_skin  # Default to skin modulus if no stringers
        
        if self._stringer_props is not None and self._stringer_props.count > 0:
            stringer_props = self._stringer_props
            
            # Get stringer material modulus
            if stringer_props.material is not None:
                E_stringer = stringer_props.material.get_bending_modulus(grain_spanwise=True)
            
            # Stringer centroid position: attached to skin surface
            # For rectangular stringer glued on edge, centroid at y = h/2 + height/2
            # Conservative approach: use skin surface y = h/2
            y_stringer = h / 2
            
            # Number of stringers: count is per skin, both top and bottom skins
            n_stringers_total = stringer_props.count * 2
            
            # Stringer cross-section properties
            A_one_stringer = stringer_props.area
            I_self_one = stringer_props.I_self
            
            # Total stringer contribution via parallel axis theorem
            # I_stringer = n * (I_self + A * y^2)
            I_stringer = n_stringers_total * (I_self_one + A_one_stringer * y_stringer**2)
        
        # Total bending inertia
        I_xx = I_skin + I_spar + I_stringer
        self.I_xx = I_xx
        self.I_stringer = I_stringer  # Store for reference
        
        # Weighted EI based on material and area contribution
        EI = E_skin * I_skin + E_spar * I_spar + E_stringer * I_stringer
        self.EI = EI
        
        # Store individual moduli for stress calculations
        self.E_spar = E_spar
        self.E_skin = E_skin
        self.E_stringer = E_stringer
        
        # Torsion constant J (Bredt's formula for thin-walled closed section)
        # J = 4 * A_enclosed^2 / perimeter_integral
        A_enclosed = w * h
        perimeter_integral = 2 * (w / (t_skin + 1e-10) + h / (t_spar + 1e-10))
        J = 4 * A_enclosed ** 2 / (perimeter_integral + 1e-10)
        self.J = J
        
        # Effective shear modulus (thickness-weighted average)
        skin_perimeter = 2 * w
        spar_perimeter = 2 * h
        total_perimeter = skin_perimeter + spar_perimeter + 1e-10
        G_eff = (G_skin * skin_perimeter + G_spar * spar_perimeter) / total_perimeter
        
        GJ = G_eff * J
        self.GJ = GJ
        
        # === Shear Stiffness for Timoshenko Beam (κGA) ===
        # This accounts for shear deformation in the beam analysis
        # For orthotropic materials, transverse shear modulus (G_xz) is used
        G_xz_spar = spar_material.get_transverse_shear_modulus()
        
        # Shear area (spar webs carry vertical shear)
        A_shear = 2 * h * t_spar
        
        # Shear correction factor for thin-walled box section
        # Lower for orthotropic due to shear lag effects
        if hasattr(spar_material, 'G_xz') and spar_material.G_xz is not None:
            kappa = 0.5  # Orthotropic: accounts for shear lag
        elif spar_material.is_isotropic:
            kappa = 0.6  # Isotropic thin-walled box
        else:
            kappa = 0.5  # Default for orthotropic
        
        self.kappa_GA = kappa * G_xz_spar * A_shear
        self._shear_correction_factor = kappa
        self._G_xz_spar = G_xz_spar
        
        # Estimate shear deformation significance for reporting
        import numpy as _np_std_shear
        try:
            avg_h = float(_np_std_shear.mean([s.box_height for s in sections]))
            E_avg = (E_spar + E_skin) / 2
            self._shear_ratio_estimate = estimate_shear_deformation_ratio(
                half_span=self.length,
                avg_box_height=avg_h,
                E=E_avg,
                G_xz=G_xz_spar,
            )
        except (ValueError, TypeError, RuntimeError):
            self._shear_ratio_estimate = 0.0
        
        # Initial guesses for stiffness (for optimizer scaling)
        # When thicknesses are symbolic (CasADi), we can't compute mean directly
        # Use geometry-based estimates instead
        if EI_guess is None:
            try:
                # Try to evaluate - works for numeric values
                EI_guess = float(np.mean(EI))
            except (RuntimeError, TypeError, ValueError):
                # Fallback for symbolic values: estimate from geometry
                import numpy as _np_std
                avg_h = float(_np_std.mean([s.box_height for s in sections]))
                avg_w = float(_np_std.mean([s.box_width for s in sections]))
                # Rough estimate: EI ~ E * (w * t * h^2) for thin-walled box
                # Use typical thickness ~2mm for estimate
                t_est = 0.002
                EI_guess = E_skin * avg_w * t_est * (avg_h / 2) ** 2 + E_spar * t_est * avg_h ** 3 / 6
                EI_guess = max(EI_guess, 1.0)  # Ensure positive
        
        if GJ_guess is None:
            try:
                GJ_guess = float(np.mean(GJ))
            except (RuntimeError, TypeError, ValueError):
                # Fallback for symbolic values
                import numpy as _np_std
                avg_h = float(_np_std.mean([s.box_height for s in sections]))
                avg_w = float(_np_std.mean([s.box_width for s in sections]))
                t_est = 0.002
                GJ_guess = G_skin * 4 * (avg_w * avg_h) ** 2 / (2 * (avg_w + avg_h) / t_est)
                GJ_guess = max(GJ_guess, 1.0)
        
        # === Load Distribution ===
        # Aerodynamic load (upward, positive)
        q_aero = lift_distribution(y)
        self.q_aero = q_aero
        
        # Inertial load (weight of structure per unit span, downward = negative)
        # Mass per unit span = (skin area + spar area) * density
        # Skin: 2 skins * width * thickness
        # Spar: 2 spars * height * thickness
        skin_area_per_length = 2 * w * t_skin  # [m^2/m]
        spar_area_per_length = 2 * h * t_spar  # [m^2/m]
        
        mass_per_length = (skin_area_per_length * skin_material.density + 
                          spar_area_per_length * spar_material.density)  # [kg/m]
        
        # Inertial load = mass * g (negative because weight acts downward)
        g = 9.80665  # m/s^2
        q_inertial = -mass_per_length * g  # [N/m] (negative = downward)
        self.q_inertial = q_inertial
        
        # Net load = aerodynamic (up) + inertial (down)
        q_net = q_aero + q_inertial
        self.q_net = q_net
        
        # For the beam analysis, we use the NET load
        q_lift = q_net
        self.q_lift = q_lift
        
        if moment_distribution is not None:
            m_twist = moment_distribution(y)
        else:
            m_twist = np.zeros_like(y)
        self.m_twist = m_twist
        
        # === Torque Distribution (from pitching moment) ===
        # Torque at station y = integral of m_twist from y to tip
        # T(y) = ∫[y to L] m_twist(η) dη
        # This is computed using cumulative integration from tip to root
        import numpy as _np_std_torque
        try:
            m_twist_np = _np_std_torque.array(m_twist)
            y_np_torque = _np_std_torque.array(y)
            # Integrate from tip to root (reverse, then flip back)
            # Using trapezoidal rule for cumulative integral
            torque_np = _np_std_torque.zeros_like(y_np_torque)
            for i in range(len(y_np_torque) - 2, -1, -1):
                dy = y_np_torque[i + 1] - y_np_torque[i]
                torque_np[i] = torque_np[i + 1] + 0.5 * (m_twist_np[i] + m_twist_np[i + 1]) * dy
            self.torque = torque_np
        except (ValueError, TypeError, RuntimeError):
            # Fallback for symbolic - use cumsum approximation
            self.torque = np.zeros_like(y)
        
        # === Torsional Shear Stress ===
        # For thin-walled closed section (Bredt's formula):
        # Shear flow: q = T / (2 * A_enclosed)
        # Shear stress in wall: τ = q / t = T / (2 * A * t)
        A_enclosed = w * h
        self.A_enclosed = A_enclosed
        
        # Torsional shear in skins: τ_torsion_skin = T / (2 * A * t_skin)
        # Torsional shear in spars: τ_torsion_spar = T / (2 * A * t_spar)
        # Note: We use absolute torque since shear stress direction doesn't affect buckling
        T_abs = np.abs(self.torque) if hasattr(self.torque, '__abs__') else _np_std_torque.abs(self.torque)
        self.tau_torsion_skin = T_abs / (2 * A_enclosed * t_skin + 1e-10)
        self.tau_torsion_spar = T_abs / (2 * A_enclosed * t_spar + 1e-10)
        
        # === Bending Analysis (Euler-Bernoulli using Opti) ===
        # Governing equation: (EI * u'')'' = q(y)
        # Boundary conditions (cantilever): u(0)=0, u'(0)=0, M(L)=0, V(L)=0
        
        # Scale factor for displacement guess - use simple estimate to avoid symbolic issues
        # For a uniform load on a cantilever: delta_max ~ q*L^4 / (8*EI)
        # Estimate q as total_lift / span
        import numpy as _np_std  # Use standard numpy for scaling calculations
        q_lift_np = _np_std.array(q_aero) if hasattr(q_aero, '__iter__') else q_aero
        y_np = _np_std.array(y)
        try:
            total_load = float(_np_std.trapz(q_lift_np, y_np)) if len(y_np) > 1 else 0.0
        except (ValueError, TypeError):
            # Fallback if q_lift contains symbolic values
            total_load = 50.0  # Default estimate
        disp_scale = max(1e-6, abs(total_load) * self.length ** 3 / (EI_guess * 3) + 1e-10)
        
        u = self.opti.variable(
            init_guess=np.zeros_like(y),
            scale=disp_scale,
        )
        
        du = self.opti.derivative_of(
            u, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=disp_scale / self.length,
        )
        
        ddu = self.opti.derivative_of(
            du, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=disp_scale / self.length ** 2,
        )
        
        dEIddu = self.opti.derivative_of(
            EI * ddu, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=max(1.0, abs(total_load) * self.length / EI_guess),
        )
        
        # Constrain derivative to match distributed load
        self.opti.constrain_derivative(
            variable=dEIddu,
            with_respect_to=y,
            derivative=q_lift,
        )
        
        # Cantilever boundary conditions
        self.opti.subject_to([
            u[0] == 0,       # Zero displacement at root
            du[0] == 0,      # Zero slope at root
            ddu[-1] == 0,    # Zero moment at tip (M = -EI*u'')
            dEIddu[-1] == 0  # Zero shear at tip (V = -(EI*u'')')
        ])
        
        self.u = u              # Displacement [m]
        self.du = du            # Slope [rad]
        self.ddu = ddu          # Curvature [1/m]
        
        # Derived quantities
        self.bending_moment = -EI * ddu       # [N*m]
        self.shear_force = -dEIddu            # [N]
        
        # === Stress Calculations ===
        # Bending stress at extreme fiber: sigma = E * kappa * y_max = E * kappa * (h/2)
        self.sigma_spar = E_spar * ddu * (h / 2)
        self.sigma_skin = E_skin * ddu * (h / 2)
        
        # Stress ratios (orthotropic failure criterion)
        # Note: np.where from aerosandbox.numpy handles CasADi properly
        spar_tension_ratio = np.where(
            self.sigma_spar >= 0,
            self.sigma_spar / (spar_material.sigma_1_tension + 1e-10),
            np.abs(self.sigma_spar) / (spar_material.sigma_1_compression + 1e-10)
        )
        
        skin_tension_ratio = np.where(
            self.sigma_skin >= 0,
            self.sigma_skin / (skin_material.sigma_1_tension + 1e-10),
            np.abs(self.sigma_skin) / (skin_material.sigma_1_compression + 1e-10)
        )
        
        self._max_stress_ratio = np.maximum(spar_tension_ratio, skin_tension_ratio)
        
        # === Enhanced Buckling Analysis ===
        # NOTE: We store parameters here and compute buckling margins in get_result()
        # because w, h, t_skin, t_spar, sigma_skin are CasADi symbolic during __init__
        import numpy as _np_std
        
        # Store rib positions for mass calculation
        self._rib_positions = rib_positions
        
        # Determine rib spacing from actual rib positions
        if rib_positions is not None and len(rib_positions) >= 2:
            sorted_ribs = sorted(rib_positions)
            spacings = [sorted_ribs[i+1] - sorted_ribs[i] for i in range(len(sorted_ribs)-1)]
            rib_spacing = sum(spacings) / len(spacings) if spacings else self.length
        else:
            rib_spacing = self.length / max(1, len(sections) - 1)
        self.rib_spacing = rib_spacing
        
        # Store stringer count for panel width calculation in get_result()
        self._stringer_count = stringer_props.count if stringer_props is not None else 0
        
        # Store parameters for deferred buckling calculation
        self._boundary_condition = boundary_condition
        self._include_curvature = include_curvature
        self._airfoils = airfoils
        self._sections = sections  # For curvature calculation
        
        # === Curvature Radius Calculation (can be done now - uses only geometry) ===
        curvature_radii = _np_std.full(n_points, float('inf'))
        if include_curvature and airfoils is not None and len(airfoils) > 0:
            section_y_np = _np_std.array([s.y for s in sections])
            section_front_spar = _np_std.array([s.front_spar_xsi for s in sections])
            section_rear_spar = _np_std.array([s.rear_spar_xsi for s in sections])
            section_chord = _np_std.array([s.chord for s in sections])
            y_np = _np_std.array([float(yi) for yi in y]) if hasattr(y[0], '__float__') else _np_std.array(y)
            
            for i in range(len(y_np)):
                # Find nearest section for airfoil
                idx = _np_std.argmin(_np_std.abs(section_y_np - y_np[i]))
                if idx < len(airfoils):
                    airfoil = airfoils[idx]
                    front_spar = float(section_front_spar[idx])
                    rear_spar = float(section_rear_spar[idx])
                    chord = float(section_chord[idx])
                    
                    R = calculate_skin_curvature_radius(
                        airfoil=airfoil,
                        front_spar_xsi=front_spar,
                        rear_spar_xsi=rear_spar,
                        chord=chord,
                    )
                    curvature_radii[i] = R
        
        self._curvature_radii = curvature_radii
        
        # Store material references for deferred calculation
        self._skin_material = skin_material
        self._spar_material = spar_material
        
        # Spar web buckling coefficient
        k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(boundary_condition, 7.0)
        self._k_shear = k_shear
    
    def mass_with_breakdown(self) -> tuple:
        """
        Calculate total structural mass [kg] for FULL wing (both halves).
        
        Returns:
            tuple: (total_mass_kg, mass_breakdown_dict, assumptions_list)
        
        Includes:
        - Skin panels (top + bottom)
        - Spar webs (front + rear)
        - Ribs (if rib_positions provided)
        - Stringers (if stringer_props provided)
        - Leading edge structure (skin forward of front spar)
        - Trailing edge structure (skin aft of rear spar)
        - Control surfaces (if control_surface_props provided)
        - Fasteners/adhesive allowance (default 10%)
        """
        import numpy as _np_std  # Use standard numpy for post-processing
        
        # Convert to numpy arrays
        box_width = _np_std.array(self.box_width)
        box_height = _np_std.array(self.box_height)
        t_skin = _np_std.array(self.t_skin)
        t_spar = _np_std.array(self.t_spar)
        y = _np_std.array(self.y)
        chord = _np_std.array(self._chord)
        front_spar_xsi = _np_std.array(self._front_spar_xsi)
        rear_spar_xsi = _np_std.array(self._rear_spar_xsi)
        half_span = self.length
        
        assumptions = []
        
        # === Wingbox Skin and Spar Mass ===
        # Skin volume: 2 skins (top + bottom) x width x thickness x length
        V_skin = 2 * _np_std.trapz(box_width * t_skin, y)
        # Spar volume: 2 spars (front + rear) x height x thickness x length
        V_spar = 2 * _np_std.trapz(box_height * t_spar, y)
        
        m_skin = V_skin * self.skin_material.density
        m_spar = V_spar * self.spar_material.density
        
        # === Leading Edge Structure ===
        # LE skin from nose to front spar (top + bottom)
        le_thickness_factor = 0.7
        le_width = chord * front_spar_xsi  # Width from LE to front spar
        le_skin_thickness = t_skin * le_thickness_factor
        V_le = 2 * _np_std.trapz(le_width * le_skin_thickness, y)
        m_le = V_le * self.skin_material.density
        assumptions.append(f"LE skin thickness = {le_thickness_factor:.0%} of wingbox skin (D-tube construction)")
        
        # === Trailing Edge Structure ===
        # TE skin from rear spar to trailing edge (top + bottom)
        te_thickness_factor = 0.5
        te_width = chord * (1.0 - rear_spar_xsi)  # Width from rear spar to TE
        te_skin_thickness = t_skin * te_thickness_factor
        V_te = 2 * _np_std.trapz(te_width * te_skin_thickness, y)
        m_te = V_te * self.skin_material.density
        assumptions.append(f"TE skin thickness = {te_thickness_factor:.0%} of wingbox skin (lighter construction)")
        
        # === Rib Mass ===
        m_ribs = 0.0
        n_ribs = 0
        rib_positions = self._rib_positions
        rib_props = self._rib_props
        
        if rib_positions is not None and len(rib_positions) > 0:
            n_ribs = len(rib_positions)
            
            # Get rib properties (use defaults if not specified)
            if rib_props is not None:
                rib_thickness = rib_props.thickness_m
                rib_material = rib_props.material if rib_props.material is not None else self.skin_material
                lightening_fraction = rib_props.lightening_hole_fraction
            else:
                # Default: 3mm plywood-like thickness, skin material, 40% lightening
                rib_thickness = 0.003
                rib_material = self.skin_material
                lightening_fraction = 0.4
                assumptions.append(f"Rib thickness defaulted to {rib_thickness*1000:.1f}mm (no rib_props specified)")
            
            solid_fraction = 1.0 - lightening_fraction
            assumptions.append(f"Ribs have {lightening_fraction:.0%} material removed (lightening holes)")
            
            for rib_y in rib_positions:
                # Interpolate wingbox dimensions at rib location
                w_rib = float(_np_std.interp(rib_y, y, box_width))
                h_rib = float(_np_std.interp(rib_y, y, box_height))
                
                # Rib area with lightening holes
                rib_area = w_rib * h_rib * solid_fraction
                rib_volume = rib_area * rib_thickness
                
                m_ribs += rib_volume * rib_material.density
        
        # === Stringer Mass ===
        m_stringers = 0.0
        stringer_props = self._stringer_props
        if stringer_props is not None and stringer_props.count > 0:
            # Stringers run the full span on top and bottom skins
            stringer_area = stringer_props.area
            n_stringers = stringer_props.count * 2  # Top + bottom skin
            
            # Total stringer length = span length
            stringer_length = self.length
            V_stringers = n_stringers * stringer_area * stringer_length
            
            # Use stringer material if specified, else skin material
            stringer_density = (stringer_props.material.density 
                               if stringer_props.material is not None 
                               else self.skin_material.density)
            m_stringers = V_stringers * stringer_density
            assumptions.append(f"Stringers modeled as rectangular strips (area = height × thickness)")
        
        # === Control Surface Mass ===
        m_control_surfaces = 0.0
        n_control_surfaces = 0
        control_surface_props = self._control_surface_props
        
        if control_surface_props and len(control_surface_props) > 0:
            n_control_surfaces = len(control_surface_props)
            
            for cs_props in control_surface_props:
                # Get materials (default to main wing materials)
                cs_skin_material = cs_props.skin_material if cs_props.skin_material is not None else self.skin_material
                cs_rib_material = cs_props.rib_material if cs_props.rib_material is not None else self.skin_material
                
                # Calculate control surface span extent
                y_start = cs_props.span_start * half_span
                y_end = cs_props.span_end * half_span
                cs_span = y_end - y_start
                
                if cs_span <= 0:
                    continue
                
                # Sample points along control surface span
                n_sample = 20
                y_cs = _np_std.linspace(y_start, y_end, n_sample)
                
                # Interpolate chord at each point
                chord_at_y = _np_std.interp(y_cs, y, chord)
                
                # Control surface chord at each station (linear interpolation of fraction)
                eta_cs = (y_cs - y_start) / (cs_span + 1e-10)
                cs_chord_fraction = cs_props.chord_fraction_start + (
                    cs_props.chord_fraction_end - cs_props.chord_fraction_start
                ) * eta_cs
                cs_chord = chord_at_y * cs_chord_fraction
                
                # Control surface area (planform)
                cs_area = _np_std.trapz(cs_chord, y_cs)
                
                # Skin mass: top + bottom skins
                V_cs_skin = 2 * cs_area * cs_props.skin_thickness_m
                m_cs_skin = V_cs_skin * cs_skin_material.density
                
                # Hinge spar: runs along the hinge line (full control surface span)
                # Approximate height as proportional to local wingbox height
                h_at_y = _np_std.interp(y_cs, y, box_height)
                avg_height = _np_std.mean(h_at_y) * 0.6  # Control surface is ~60% of main spar height
                V_hinge_spar = cs_span * avg_height * cs_props.hinge_spar_thickness_m
                m_cs_hinge_spar = V_hinge_spar * cs_skin_material.density
                
                # Internal ribs: spaced along span
                n_cs_ribs = max(2, int(cs_span / cs_props.rib_spacing_m) + 1)
                avg_cs_chord = float(_np_std.mean(cs_chord))
                cs_rib_area = avg_cs_chord * avg_height * (1.0 - cs_props.rib_lightening_fraction)
                V_cs_ribs = n_cs_ribs * cs_rib_area * cs_props.rib_thickness_m
                m_cs_ribs = V_cs_ribs * cs_rib_material.density
                
                # Total control surface mass (one side)
                m_cs_one = m_cs_skin + m_cs_hinge_spar + m_cs_ribs
                m_control_surfaces += m_cs_one
            
            assumptions.append(f"Control surfaces include skin, hinge spar, and internal ribs")
            assumptions.append(f"Control surface skin thickness: {cs_props.skin_thickness_m*1000:.1f}mm")
        
        # === Primary Structure Total (half-wing) ===
        m_half_wing_primary = m_skin + m_spar + m_le + m_te + m_ribs + m_stringers + m_control_surfaces
        
        # === Fastener/Adhesive Allowance ===
        fastener_fraction = self._fastener_adhesive_fraction
        m_fasteners_half = m_half_wing_primary * fastener_fraction
        
        # Total half-wing mass
        m_half_wing = m_half_wing_primary + m_fasteners_half
        
        # Full wing mass (both halves)
        total_mass = 2.0 * m_half_wing
        
        # Build mass breakdown (full wing values)
        mass_breakdown = {
            "wingbox_skins": float(2.0 * m_skin),
            "spar_webs": float(2.0 * m_spar),
            "leading_edge": float(2.0 * m_le),
            "trailing_edge": float(2.0 * m_te),
            "ribs": float(2.0 * m_ribs),
            "stringers": float(2.0 * m_stringers),
            "control_surfaces": float(2.0 * m_control_surfaces),
            "fasteners_adhesive": float(2.0 * m_fasteners_half),
            "total": float(total_mass),
            "n_ribs": n_ribs * 2,  # Both wing halves
            "n_stringers": (stringer_props.count * 2 * 2) if stringer_props and stringer_props.count > 0 else 0,
            "n_control_surfaces": n_control_surfaces * 2,  # Both wing halves
        }
        
        # Add general assumptions
        assumptions.insert(0, "Mass is for FULL wing (both halves)")
        assumptions.append("Spar caps/flanges not modeled separately (included in spar web)")
        if fastener_fraction > 0:
            assumptions.append(f"Fasteners/adhesive allowance: {fastener_fraction:.0%} of structural mass")
        
        return float(total_mass), mass_breakdown, assumptions
    
    def mass(self) -> float:
        """Calculate total structural mass [kg] for FULL wing (both halves)."""
        total_mass, _, _ = self.mass_with_breakdown()
        return total_mass
    
    def tip_deflection(self) -> float:
        """Get tip deflection [m]."""
        val = self.u[-1]
        return float(val) if hasattr(val, '__float__') else val
    
    def max_stress(self) -> float:
        """Get maximum bending stress [Pa]."""
        import numpy as _np_std
        sigma_spar = _np_std.array(self.sigma_spar)
        sigma_skin = _np_std.array(self.sigma_skin)
        return float(_np_std.max(_np_std.maximum(_np_std.abs(sigma_spar), _np_std.abs(sigma_skin))))
    
    def max_stress_ratio_value(self) -> float:
        """Get maximum stress ratio (orthotropic failure criterion)."""
        import numpy as _np_std
        return float(_np_std.max(_np_std.array(self._max_stress_ratio)))
    
    def _compute_buckling_data(self) -> dict:
        """
        Compute all buckling-related data from solved values.
        
        This is called AFTER the implicit solve, when all values are concrete numbers.
        Returns dict with 'skin_buckling_margin', 'spar_buckling_margin', 'tau_spar',
        'panel_widths', 'curvature_factors'.
        """
        import numpy as _np_std
        
        # Get solved (concrete) values
        w = _np_std.array(self.box_width)
        h = _np_std.array(self.box_height)
        t_skin = _np_std.array(self.t_skin)
        t_spar = _np_std.array(self.t_spar)
        sigma_skin = _np_std.array(self.sigma_skin)
        shear_force = _np_std.array(self.shear_force)
        n_points = len(w)
        
        # === Skin Buckling Analysis ===
        stringer_count = self._stringer_count
        rib_spacing = self.rib_spacing
        boundary_condition = self._boundary_condition
        include_curvature = self._include_curvature
        skin_material = self._skin_material
        curvature_radii = self._curvature_radii
        
        # Calculate panel widths (affected by stringers)
        if stringer_count > 0:
            # Stringers divide the skin into smaller panels
            panel_widths = w / (stringer_count + 1)
        else:
            panel_widths = w.copy()
        
        # Calculate skin buckling stress at each station
        skin_buckling_stress = _np_std.zeros(n_points)
        curvature_factors = _np_std.ones(n_points)
        
        for i in range(n_points):
            # Get local geometry
            b_panel = float(panel_widths[i])
            a_panel = float(rib_spacing)  # Panel length between ribs
            t_local = float(t_skin[i])
            
            if t_local <= 0 or b_panel <= 0:
                skin_buckling_stress[i] = float('inf')
                continue
            
            # Calculate critical buckling stress (orthotropic or isotropic)
            sigma_cr = calculate_orthotropic_buckling_stress(
                panel_length=a_panel,
                panel_width=b_panel,
                thickness=t_local,
                material=skin_material,
                grain_spanwise=True,
                boundary_condition=boundary_condition,
            )
            
            # Apply curvature enhancement if enabled
            if include_curvature and curvature_radii[i] < float('inf'):
                cf = curved_panel_buckling_factor(
                    panel_width=b_panel,
                    radius_of_curvature=curvature_radii[i],
                    thickness=t_local,
                    poissons_ratio=skin_material.nu_12,
                )
                curvature_factors[i] = cf
                sigma_cr *= cf
            
            skin_buckling_stress[i] = sigma_cr
        
        # Calculate skin buckling margin
        sigma_skin_abs = _np_std.abs(sigma_skin)
        skin_buckling_margin = skin_buckling_stress / (sigma_skin_abs + 1e-10)
        
        # === Spar Web Buckling Analysis (Priority 2: Improved Torsion-Bending Interaction) ===
        spar_material = self._spar_material
        k_shear = self._k_shear
        
        # Get torque distribution (if available)
        torque = _np_std.array(self.torque) if hasattr(self, 'torque') else _np_std.zeros(n_points)
        
        # Torsional shear stress in spar webs (from Bredt's formula)
        # τ_torsion = T / (2 * A_enclosed * t_spar)
        tau_torsion_spar = _np_std.array(self.tau_torsion_spar) if hasattr(self, 'tau_torsion_spar') else _np_std.zeros(n_points)
        
        # Calculate improved combined shear using proper torsion-bending interaction
        # This properly accounts for the fact that torsion adds to one spar and subtracts from the other
        tau_spar_bending = _np_std.zeros(n_points)
        tau_spar_total = _np_std.zeros(n_points)
        tau_spar_total_conservative = _np_std.zeros(n_points)  # For comparison
        tau_skin_combined = _np_std.zeros(n_points)
        
        for i in range(n_points):
            V_local = float(shear_force[i])
            T_local = float(torque[i])
            w_local = float(w[i])
            h_local = float(h[i])
            t_spar_local = float(t_spar[i])
            t_skin_local = float(t_skin[i])
            
            if h_local > 0 and t_spar_local > 0:
                # Use improved calculation with proper shear interaction
                shear_dist = calculate_wingbox_shear_distribution(
                    shear_force=V_local,
                    torque=T_local,
                    box_width=w_local,
                    box_height=h_local,
                    t_spar=t_spar_local,
                    t_skin=t_skin_local,
                )
                
                # Maximum combined shear in spar webs (proper interaction)
                tau_spar_total[i] = shear_dist.tau_max_spar
                tau_skin_combined[i] = shear_dist.tau_max_skin
                
                # Also store bending component for diagnostics
                tau_spar_bending[i] = max(shear_dist.tau_bending_front_spar, 
                                          shear_dist.tau_bending_rear_spar)
                
                # Conservative estimate for comparison
                A_web = 2 * h_local * t_spar_local
                tau_bending_cons = 1.5 * abs(V_local) / (A_web + 1e-10)
                tau_spar_total_conservative[i] = tau_bending_cons + abs(tau_torsion_spar[i])
            else:
                tau_spar_bending[i] = 0.0
                tau_spar_total[i] = 0.0
                tau_spar_total_conservative[i] = 0.0
                tau_skin_combined[i] = 0.0
        
        # Critical shear buckling stress for spar webs
        # tau_cr = k_s * pi^2 * E * (t/h)^2 / (12 * (1 - nu^2))
        E_spar = spar_material.E_1
        nu_spar = spar_material.nu_12
        
        spar_buckling_stress = _np_std.zeros(n_points)
        for i in range(n_points):
            h_local = float(h[i])
            t_local = float(t_spar[i])
            if h_local > 0 and t_local > 0:
                tau_cr = k_shear * (_np_std.pi ** 2 * E_spar) / (12 * (1 - nu_spar ** 2)) * (t_local / h_local) ** 2
                spar_buckling_stress[i] = tau_cr
            else:
                spar_buckling_stress[i] = float('inf')
        
        # Use IMPROVED combined shear stress for buckling margin (proper interaction)
        spar_buckling_margin = spar_buckling_stress / (tau_spar_total + 1e-10)
        
        # === Rib Failure Analysis ===
        rib_positions = self._rib_positions
        rib_props = self._rib_props
        bending_moment = _np_std.array(self.bending_moment)
        y = _np_std.array(self.y)
        
        # Initialize rib failure arrays (at rib locations, not all spanwise points)
        rib_shear_buckling_margin = None
        rib_crushing_margin = None
        
        if rib_positions is not None and len(rib_positions) > 0:
            n_ribs = len(rib_positions)
            rib_shear_buckling_margin = _np_std.zeros(n_ribs)
            rib_crushing_margin = _np_std.zeros(n_ribs)
            
            # Get rib properties
            if rib_props is not None:
                rib_thickness = rib_props.thickness_m
                rib_material = rib_props.material if rib_props.material is not None else skin_material
                lightening_fraction = rib_props.lightening_hole_fraction
                spar_cap_width = rib_props.spar_cap_width_m
                rib_boundary = rib_props.boundary_condition
            else:
                # Defaults
                rib_thickness = 0.003
                rib_material = skin_material
                lightening_fraction = 0.4
                spar_cap_width = 0.010
                rib_boundary = "simply_supported"
            
            for i, rib_y in enumerate(rib_positions):
                # Interpolate local geometry at rib location
                h_rib = float(_np_std.interp(rib_y, y, h))
                M_rib = float(_np_std.interp(rib_y, y, _np_std.abs(bending_moment)))
                V_rib = float(_np_std.interp(rib_y, y, _np_std.abs(shear_force)))
                
                # === Rib Shear Buckling ===
                # The rib carries shear from the skin panels
                # Approximate shear flow into rib from adjacent skin
                # τ_rib ≈ V / (h * t_rib) for simplicity
                if h_rib > 0 and rib_thickness > 0:
                    tau_rib = V_rib / (h_rib * rib_thickness + 1e-10)
                    tau_cr_rib = calculate_rib_shear_buckling_stress(
                        rib_height=h_rib,
                        rib_thickness=rib_thickness,
                        material=rib_material,
                        lightening_fraction=lightening_fraction,
                        boundary_condition=rib_boundary,
                    )
                    rib_shear_buckling_margin[i] = tau_cr_rib / (tau_rib + 1e-10)
                else:
                    rib_shear_buckling_margin[i] = float('inf')
                
                # === Rib Crushing ===
                rib_crushing_margin[i] = calculate_rib_crushing_margin(
                    bending_moment=M_rib,
                    box_height=h_rib,
                    rib_thickness=rib_thickness,
                    spar_cap_width=spar_cap_width,
                    material=rib_material,
                )
        
        # === Skin Shear Buckling Analysis (Priority 3) ===
        # Skin panels experience torsional shear stress from wing twisting.
        # This can cause shear buckling (diagonal wrinkling) in addition to compressive buckling.
        tau_torsion_skin = _np_std.array(self.tau_torsion_skin) if hasattr(self, 'tau_torsion_skin') else _np_std.zeros(n_points)
        
        skin_shear_buckling_stress = _np_std.zeros(n_points)
        for i in range(n_points):
            b_panel = float(panel_widths[i])
            a_panel = float(rib_spacing)
            t_local = float(t_skin[i])
            
            if t_local <= 0 or b_panel <= 0:
                skin_shear_buckling_stress[i] = float('inf')
                continue
            
            # Calculate critical shear buckling stress
            tau_cr_skin = calculate_skin_shear_buckling_stress(
                panel_length=a_panel,
                panel_width=b_panel,
                thickness=t_local,
                material=skin_material,
                grain_spanwise=True,
                boundary_condition=boundary_condition,
            )
            skin_shear_buckling_stress[i] = tau_cr_skin
        
        # Skin shear buckling margin: τ_cr / τ_applied
        skin_shear_buckling_margin = skin_shear_buckling_stress / (_np_std.abs(tau_torsion_skin) + 1e-10)
        
        # === Biaxial Stress Interaction (Priority 5) ===
        # Skin panels under combined compression (σ) and shear (τ) have reduced buckling
        # strength compared to either load acting alone. Use interaction equation.
        combined_buckling_margin = _np_std.zeros(n_points)
        for i in range(n_points):
            sigma_applied = float(_np_std.abs(sigma_skin[i]))
            sigma_cr = float(skin_buckling_stress[i])
            tau_applied = float(_np_std.abs(tau_torsion_skin[i]))
            tau_cr = float(skin_shear_buckling_stress[i])
            
            combined_buckling_margin[i] = calculate_combined_buckling_margin(
                sigma_applied=sigma_applied,
                sigma_cr=sigma_cr,
                tau_applied=tau_applied,
                tau_cr=tau_cr,
                interaction_exponent=2.0,  # Parabolic interaction (NASA SP-8007)
            )
        
        # === Stringer Crippling Analysis (Priority 6) ===
        # Stringers can fail by local crippling before global buckling occurs.
        # This is critical for thin-walled stringer sections under compression.
        stringer_crippling_margin = None
        stringer_failure_mode = None
        
        stringer_props = self._stringer_props
        if stringer_props is not None and stringer_props.count > 0 and stringer_props.material is not None:
            stringer_crippling_margin = _np_std.zeros(n_points)
            
            # Calculate stringer allowable stress (considering crippling, column buckling, yield)
            sigma_allowable, stringer_failure_mode = calculate_stringer_allowable_stress(
                stringer_props=stringer_props,
                rib_spacing=rib_spacing,
                end_fixity=2.0,  # Semi-fixed ends at ribs
            )
            
            for i in range(n_points):
                # Calculate stress in stringer due to bending
                M_local = float(_np_std.abs(bending_moment[i]))
                h_local = float(h[i])
                w_local = float(w[i])
                t_skin_local = float(t_skin[i])
                
                sigma_stringer = calculate_stringer_stress(
                    bending_moment=M_local,
                    box_height=h_local,
                    stringer_props=stringer_props,
                    E_skin=skin_material.E_1,
                    t_skin=t_skin_local,
                    box_width=w_local,
                )
                
                # Crippling margin = allowable / applied
                if sigma_stringer > 0:
                    stringer_crippling_margin[i] = sigma_allowable / (sigma_stringer + 1e-10)
                else:
                    stringer_crippling_margin[i] = float('inf')
        
        # === Post-Buckling Analysis (Priority 4) ===
        # When post-buckling is enabled, skin panels are allowed to buckle
        # and load redistributes to stringers. Check post-buckling failure modes.
        post_buckling_result = None
        post_buckling_config = self._post_buckling_config
        
        if post_buckling_config.enabled:
            # Verify stringers are present if required
            has_stringers = stringer_props is not None and stringer_props.count > 0
            if post_buckling_config.require_stringers and not has_stringers:
                # Cannot use post-buckling without stringers
                post_buckling_result = PostBucklingResult(
                    y=y,
                    is_buckled_compression=_np_std.zeros(n_points, dtype=bool),
                    is_buckled_shear=_np_std.zeros(n_points, dtype=bool),
                    post_buckling_ratio=_np_std.ones(n_points),
                    effective_width_ratio=_np_std.ones(n_points),
                    stringer_margin=_np_std.zeros(n_points),
                    tension_field_margin=_np_std.full(n_points, float('inf')),
                    separation_margin=_np_std.full(n_points, float('inf')),
                    overall_margin=_np_std.zeros(n_points),
                    min_stringer_margin=0.0,
                    min_tension_field_margin=float('inf'),
                    min_separation_margin=float('inf'),
                    min_overall_margin=0.0,
                    is_post_buckling_feasible=False,  # Infeasible: no stringers
                    config=post_buckling_config,
                )
            else:
                # Perform post-buckling analysis at each station
                is_buckled_compression = _np_std.zeros(n_points, dtype=bool)
                is_buckled_shear = _np_std.zeros(n_points, dtype=bool)
                post_buckling_ratio = _np_std.ones(n_points)
                effective_width_ratio = _np_std.ones(n_points)
                stringer_margin = _np_std.full(n_points, float('inf'))
                tension_field_margin = _np_std.full(n_points, float('inf'))
                separation_margin = _np_std.full(n_points, float('inf'))
                overall_margin = _np_std.full(n_points, float('inf'))
                
                for i in range(n_points):
                    sigma_applied_i = float(_np_std.abs(sigma_skin[i]))
                    sigma_cr_i = float(skin_buckling_stress[i])
                    tau_applied_i = float(_np_std.abs(tau_torsion_skin[i]))
                    tau_cr_i = float(skin_shear_buckling_stress[i])
                    
                    # Call comprehensive post-buckling margin function
                    pb_result = calculate_post_buckling_margin(
                        sigma_applied=sigma_applied_i,
                        sigma_cr=sigma_cr_i,
                        tau_applied=tau_applied_i,
                        tau_cr=tau_cr_i,
                        stringer_props=stringer_props,
                        rib_spacing=rib_spacing,
                        config=post_buckling_config,
                        skin_material=skin_material,
                    )
                    
                    is_buckled_compression[i] = pb_result['is_buckled_compression']
                    is_buckled_shear[i] = pb_result['is_buckled_shear']
                    post_buckling_ratio[i] = pb_result['post_buckling_ratio']
                    effective_width_ratio[i] = pb_result['effective_width_ratio']
                    stringer_margin[i] = pb_result['stringer_margin']
                    tension_field_margin[i] = pb_result['tension_field_margin']
                    separation_margin[i] = pb_result['separation_margin']
                    overall_margin[i] = pb_result['overall_margin']
                
                # Compute summary statistics
                min_stringer_margin = float(_np_std.min(stringer_margin))
                min_tension_field_margin = float(_np_std.min(tension_field_margin))
                min_separation_margin = float(_np_std.min(separation_margin))
                min_overall_margin = float(_np_std.min(overall_margin))
                
                # Design is feasible if overall margin >= 1.0 at all stations
                is_post_buckling_feasible = min_overall_margin >= 1.0
                
                post_buckling_result = PostBucklingResult(
                    y=y,
                    is_buckled_compression=is_buckled_compression,
                    is_buckled_shear=is_buckled_shear,
                    post_buckling_ratio=post_buckling_ratio,
                    effective_width_ratio=effective_width_ratio,
                    stringer_margin=stringer_margin,
                    tension_field_margin=tension_field_margin,
                    separation_margin=separation_margin,
                    overall_margin=overall_margin,
                    min_stringer_margin=min_stringer_margin,
                    min_tension_field_margin=min_tension_field_margin,
                    min_separation_margin=min_separation_margin,
                    min_overall_margin=min_overall_margin,
                    is_post_buckling_feasible=is_post_buckling_feasible,
                    config=post_buckling_config,
                )
        
        return {
            'skin_buckling_margin': skin_buckling_margin,
            'spar_buckling_margin': spar_buckling_margin,
            'tau_spar_bending': tau_spar_bending,
            'tau_torsion_spar': tau_torsion_spar,
            'tau_spar_total': tau_spar_total,
            # Priority 2: Improved torsion-bending interaction data
            'tau_spar_total_conservative': tau_spar_total_conservative,
            'tau_skin_combined': tau_skin_combined,
            'panel_widths': panel_widths,
            'curvature_factors': curvature_factors,
            'rib_shear_buckling_margin': rib_shear_buckling_margin,
            'rib_crushing_margin': rib_crushing_margin,
            # Priority 3: Skin shear buckling
            'skin_shear_buckling_margin': skin_shear_buckling_margin,
            'tau_torsion_skin': tau_torsion_skin,
            # Priority 5: Combined biaxial buckling
            'combined_buckling_margin': combined_buckling_margin,
            # Priority 6: Stringer crippling
            'stringer_crippling_margin': stringer_crippling_margin,
            'stringer_failure_mode': stringer_failure_mode,
            # Priority 4: Post-buckling analysis
            'post_buckling_result': post_buckling_result,
            'skin_buckling_stress': skin_buckling_stress,  # Needed for post-buckling checks
            'skin_shear_buckling_stress': skin_shear_buckling_stress,  # Needed for post-buckling checks
        }
    
    def min_buckling_margin(self) -> float:
        """Get minimum skin buckling margin (must be > 1 for safety)."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        return float(_np_std.min(buckling_data['skin_buckling_margin']))
    
    def min_spar_buckling_margin(self) -> float:
        """Get minimum spar web shear buckling margin."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        return float(_np_std.min(buckling_data['spar_buckling_margin']))
    
    def min_rib_buckling_margin(self) -> float:
        """Get minimum rib shear buckling margin."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        rib_margin = buckling_data.get('rib_shear_buckling_margin')
        if rib_margin is None or len(rib_margin) == 0:
            return float('inf')  # No ribs defined
        return float(_np_std.min(rib_margin))
    
    def min_rib_crushing_margin(self) -> float:
        """Get minimum rib crushing margin."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        crush_margin = buckling_data.get('rib_crushing_margin')
        if crush_margin is None or len(crush_margin) == 0:
            return float('inf')  # No ribs defined
        return float(_np_std.min(crush_margin))
    
    def stress_margin(self) -> float:
        """Get stress margin for orthotropic materials."""
        return 1.0 / (self.max_stress_ratio_value() + 1e-10)
    
    def min_skin_shear_buckling_margin(self) -> float:
        """Get minimum skin shear buckling margin (from torsion)."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        return float(_np_std.min(buckling_data['skin_shear_buckling_margin']))
    
    def min_combined_buckling_margin(self) -> float:
        """Get minimum combined (biaxial) buckling margin (σ + τ interaction)."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        return float(_np_std.min(buckling_data['combined_buckling_margin']))
    
    def min_stringer_crippling_margin(self) -> float:
        """Get minimum stringer crippling margin (local buckling of stringer elements)."""
        import numpy as _np_std
        buckling_data = self._compute_buckling_data()
        crippling_margin = buckling_data.get('stringer_crippling_margin')
        if crippling_margin is None or len(crippling_margin) == 0:
            return float('inf')  # No stringers defined
        return float(_np_std.min(crippling_margin))
    
    def min_post_buckling_margin(self) -> float:
        """Get minimum post-buckling margin (Priority 4).
        
        Returns:
            Minimum overall post-buckling margin across all stations.
            Returns inf if post-buckling is not enabled.
        """
        buckling_data = self._compute_buckling_data()
        post_buckling_result = buckling_data.get('post_buckling_result')
        if post_buckling_result is None:
            return float('inf')  # Post-buckling not enabled
        return post_buckling_result.min_overall_margin
    
    def is_post_buckling_feasible(self) -> bool:
        """Check if post-buckling design is feasible (Priority 4).
        
        Returns:
            True if post-buckling is not enabled, or if all post-buckling
            constraints are satisfied. False otherwise.
        """
        buckling_data = self._compute_buckling_data()
        post_buckling_result = buckling_data.get('post_buckling_result')
        if post_buckling_result is None:
            return True  # Post-buckling not enabled, so not a constraint
        return post_buckling_result.is_post_buckling_feasible
    
    def is_feasible(self, factor_of_safety: float = 1.5, max_deflection_fraction: float = 0.15, max_twist_deg: float = 3.0) -> bool:
        """Check if design meets all constraints including stringer crippling, post-buckling, and twist."""
        stress_ok = self.stress_margin() >= factor_of_safety
        spar_buckling_ok = self.min_spar_buckling_margin() >= factor_of_safety
        rib_buckling_ok = self.min_rib_buckling_margin() >= factor_of_safety
        rib_crushing_ok = self.min_rib_crushing_margin() >= factor_of_safety
        deflection_ok = abs(float(self.tip_deflection())) / self.length <= max_deflection_fraction
        
        # Twist constraint: calculate tip twist
        import math
        import numpy as _np_std
        bending_twist_rad = float(self.du[-1]) if hasattr(self, 'du') and len(self.du) > 0 else 0.0
        torsion_twist_rad = 0.0
        if hasattr(self, 'torque') and hasattr(self, 'GJ'):
            y_np = _np_std.array(self.y)
            torque_arr = _np_std.array(self.torque)
            GJ_arr = _np_std.array(self.GJ) if hasattr(self.GJ, '__iter__') else _np_std.full_like(y_np, float(self.GJ))
            twist_rate = torque_arr / (GJ_arr + 1e-10)
            torsion_twist_rad = float(_np_std.trapz(twist_rate, y_np))
        tip_twist_deg = math.degrees(abs(bending_twist_rad) + abs(torsion_twist_rad))
        twist_ok = tip_twist_deg <= max_twist_deg
        
        # Priority 4: Post-buckling - affects how we evaluate skin buckling
        post_buckling_config = self._post_buckling_config
        if post_buckling_config.enabled:
            # When post-buckling is enabled, skin buckling is ALLOWED (initial buckling)
            # We check post-buckling feasibility instead of skin buckling margin
            skin_buckling_ok = True  # Allowed to buckle
            post_buckling_ok = self.is_post_buckling_feasible()
        else:
            # Normal mode: skin must not buckle (margin >= FOS)
            skin_buckling_ok = self.min_buckling_margin() >= factor_of_safety
            post_buckling_ok = True  # Not applicable
        
        # Priority 3: Skin shear buckling from torsion
        skin_shear_ok = self.min_skin_shear_buckling_margin() >= factor_of_safety
        # Priority 5: Combined biaxial buckling (σ + τ interaction)
        combined_buckling_ok = self.min_combined_buckling_margin() >= factor_of_safety
        # Priority 6: Stringer crippling
        stringer_crippling_ok = self.min_stringer_crippling_margin() >= factor_of_safety
        
        return (stress_ok and skin_buckling_ok and spar_buckling_ok and rib_buckling_ok 
                and rib_crushing_ok and skin_shear_ok and combined_buckling_ok 
                and stringer_crippling_ok and post_buckling_ok and deflection_ok and twist_ok)
    
    def get_result(self, factor_of_safety: float = 1.5, max_deflection_fraction: float = 0.15, max_twist_deg: float = 3.0) -> StructuralAnalysisResult:
        """Get results as a StructuralAnalysisResult dataclass.
        
        Includes Timoshenko shear deformation correction to the E-B displacement.
        """
        import numpy as _np_std
        
        # Compute all buckling data from solved values
        buckling_data = self._compute_buckling_data()
        
        # Get mass with breakdown and assumptions
        total_mass, mass_breakdown, assumptions = self.mass_with_breakdown()
        
        # === Timoshenko Shear Deformation Correction ===
        # The E-B solver gives bending-only displacement (u)
        # We add shear displacement: w_total = w_bending + w_shear
        # where w_shear = ∫γ dy and γ = V / (κGA)
        y_np = _np_std.array(self.y)
        u_bending = _np_std.array(self.u)
        V = _np_std.array(self.shear_force)
        
        # Get shear stiffness (computed in __init__)
        if hasattr(self, 'kappa_GA'):
            kappa_GA = _np_std.array(self.kappa_GA)
        else:
            # Fallback: very high stiffness (no shear deformation)
            kappa_GA = _np_std.ones_like(y_np) * 1e15
        
        # Shear strain distribution
        gamma = V / (kappa_GA + 1e-10)
        
        # Integrate shear strain to get shear deflection (root to tip)
        # w_shear(y) = ∫₀ʸ γ(η) dη
        w_shear = _np_std.zeros_like(y_np)
        dy = _np_std.diff(y_np)
        for i in range(len(dy)):
            gamma_avg = 0.5 * (gamma[i] + gamma[i + 1])
            w_shear[i + 1] = w_shear[i] + gamma_avg * dy[i]
        
        # Total displacement = bending + shear (Timoshenko)
        u_total = u_bending + w_shear
        
        # Store shear strain for reporting
        self._gamma_shear = gamma
        self._w_shear = w_shear
        
        # Use Timoshenko total displacement for tip deflection
        tip_deflection_m = float(u_total[-1])
        
        # === Twist Calculation ===
        # Total twist = bending twist (slope) + torsional twist (∫T/GJ dy)
        import math
        
        # Bending twist at tip (from beam slope)
        bending_twist_rad = float(self.du[-1]) if hasattr(self, 'du') and len(self.du) > 0 else 0.0
        
        # Torsional twist at tip (integrate T/GJ from root to tip)
        torsion_twist_rad = 0.0
        if hasattr(self, 'torque') and hasattr(self, 'GJ'):
            torque_arr = _np_std.array(self.torque)
            GJ_arr = _np_std.array(self.GJ) if hasattr(self.GJ, '__iter__') else _np_std.full_like(y_np, float(self.GJ))
            # Twist rate = T / GJ [rad/m]
            twist_rate = torque_arr / (GJ_arr + 1e-10)
            # Integrate from root to tip: θ(y) = ∫₀ʸ (T/GJ) dη
            torsion_twist_rad = float(_np_std.trapz(twist_rate, y_np))
        
        # Total tip twist (absolute value - we care about magnitude)
        tip_twist_rad = abs(bending_twist_rad) + abs(torsion_twist_rad)
        tip_twist_deg = math.degrees(tip_twist_rad)
        
        # Twist margin: max_allowed / actual (>1 is OK)
        twist_margin = max_twist_deg / (tip_twist_deg + 1e-10) if tip_twist_deg > 0 else float('inf')
        
        # Add assumption about beam theory
        shear_ratio = getattr(self, '_shear_ratio_estimate', 0.0)
        if shear_ratio > 0.01:
            shear_pct = shear_ratio * 100
            assumptions.append(f"Timoshenko beam theory used (shear deformation ~{shear_pct:.1f}% of total)")
        else:
            assumptions.append("Euler-Bernoulli beam theory (shear deformation negligible)")
        
        max_stress_MPa = float(self.max_stress() / 1e6)
        min_buckling_margin_val = float(_np_std.min(buckling_data['skin_buckling_margin']))
        min_spar_buckling = float(_np_std.min(buckling_data['spar_buckling_margin']))
        stress_margin_val = float(self.stress_margin())
        
        # Rib failure margins
        rib_buckling_arr = buckling_data.get('rib_shear_buckling_margin')
        rib_crushing_arr = buckling_data.get('rib_crushing_margin')
        min_rib_buckling = float(_np_std.min(rib_buckling_arr)) if rib_buckling_arr is not None and len(rib_buckling_arr) > 0 else float('inf')
        min_rib_crushing = float(_np_std.min(rib_crushing_arr)) if rib_crushing_arr is not None and len(rib_crushing_arr) > 0 else float('inf')
        
        # Priority 3: Skin shear buckling margin (from torsion)
        skin_shear_arr = buckling_data.get('skin_shear_buckling_margin')
        min_skin_shear = float(_np_std.min(skin_shear_arr)) if skin_shear_arr is not None else float('inf')
        
        # Priority 5: Combined biaxial buckling margin (σ + τ interaction)
        combined_arr = buckling_data.get('combined_buckling_margin')
        min_combined = float(_np_std.min(combined_arr)) if combined_arr is not None else float('inf')
        
        # Priority 6: Stringer crippling margin
        stringer_crippling_arr = buckling_data.get('stringer_crippling_margin')
        stringer_failure_mode = buckling_data.get('stringer_failure_mode')
        min_stringer_crippling = float(_np_std.min(stringer_crippling_arr)) if stringer_crippling_arr is not None and len(stringer_crippling_arr) > 0 else float('inf')
        
        # Priority 4: Post-buckling analysis
        post_buckling_result = buckling_data.get('post_buckling_result')
        post_buckling_config = self._post_buckling_config
        if post_buckling_result is not None:
            min_post_buckling_margin = post_buckling_result.min_overall_margin
            is_post_buckling_feasible = post_buckling_result.is_post_buckling_feasible
        else:
            min_post_buckling_margin = None
            is_post_buckling_feasible = None
        
        # Feasibility check using scalar values (includes all failure modes)
        stress_ok = stress_margin_val >= factor_of_safety
        spar_buckling_ok = min_spar_buckling >= factor_of_safety
        rib_buckling_ok = min_rib_buckling >= factor_of_safety
        rib_crushing_ok = min_rib_crushing >= factor_of_safety
        skin_shear_ok = min_skin_shear >= factor_of_safety  # Priority 3
        combined_ok = min_combined >= factor_of_safety      # Priority 5
        stringer_ok = min_stringer_crippling >= factor_of_safety  # Priority 6
        deflection_ok = abs(tip_deflection_m) / self.length <= max_deflection_fraction
        twist_ok = twist_margin >= 1.0  # Twist margin is already ratio of allowable/actual
        
        # Priority 4: Post-buckling affects skin buckling feasibility
        if post_buckling_config.enabled:
            # Skin buckling is ALLOWED when post-buckling is enabled
            skin_buckling_ok = True
            post_buckling_ok = is_post_buckling_feasible if is_post_buckling_feasible is not None else True
        else:
            # Normal mode: skin must not buckle
            skin_buckling_ok = min_buckling_margin_val >= factor_of_safety
            post_buckling_ok = True
        
        is_feasible = (stress_ok and skin_buckling_ok and spar_buckling_ok and rib_buckling_ok 
                       and rib_crushing_ok and skin_shear_ok and combined_ok 
                       and stringer_ok and post_buckling_ok and deflection_ok and twist_ok)
        
        # Convert load distributions to numpy arrays
        q_aero = _np_std.array(self.q_aero)
        q_inertial = _np_std.array(self.q_inertial)
        q_net = _np_std.array(self.q_net)
        
        return StructuralAnalysisResult(
            y=self.y,
            displacement=u_total,  # Timoshenko: bending + shear deformation
            slope=self.du,
            curvature=self.ddu,
            bending_moment=self.bending_moment,
            shear_force=self.shear_force,
            aero_load=q_aero,
            inertial_load=q_inertial,
            net_load=q_net,
            sigma_spar=self.sigma_spar,
            sigma_skin=self.sigma_skin,
            max_stress_ratio=self._max_stress_ratio,
            buckling_margin=buckling_data['skin_buckling_margin'],
            spar_buckling_margin=buckling_data['spar_buckling_margin'],
            tau_spar=buckling_data['tau_spar_bending'],
            # Timoshenko beam theory fields
            shear_strain=gamma,
            shear_displacement=w_shear,
            shear_stiffness=kappa_GA,
            shear_deformation_ratio=shear_ratio,
            # Torsion fields
            torque=_np_std.array(self.torque) if hasattr(self, 'torque') else None,
            tau_torsion_skin=buckling_data.get('tau_torsion_skin'),
            tau_torsion_spar=buckling_data['tau_torsion_spar'],
            tau_total_spar=buckling_data['tau_spar_total'],
            min_torsion_margin=min_spar_buckling,  # Spar buckling margin now includes torsion
            # Priority 3: Skin shear buckling from torsion
            skin_shear_buckling_margin=skin_shear_arr,
            min_skin_shear_buckling_margin=min_skin_shear,
            # Priority 5: Combined biaxial buckling (σ + τ interaction)
            combined_buckling_margin=combined_arr,
            min_combined_buckling_margin=min_combined,
            # Priority 6: Stringer crippling
            stringer_crippling_margin=stringer_crippling_arr,
            min_stringer_crippling_margin=min_stringer_crippling,
            stringer_failure_mode=stringer_failure_mode,
            # Priority 4: Post-buckling analysis
            post_buckling_result=post_buckling_result,
            post_buckling_enabled=post_buckling_config.enabled,
            is_post_buckling_feasible=is_post_buckling_feasible,
            min_post_buckling_margin=min_post_buckling_margin,
            curvature_factors=buckling_data['curvature_factors'],
            panel_widths=buckling_data['panel_widths'],
            rib_shear_buckling_margin=rib_buckling_arr,
            rib_crushing_margin=rib_crushing_arr,
            EI=self.EI,
            GJ=self.GJ,
            mass_kg=total_mass,
            tip_deflection_m=tip_deflection_m,
            max_stress_MPa=max_stress_MPa,
            min_buckling_margin=min_buckling_margin_val,
            min_spar_buckling_margin=min_spar_buckling,
            min_rib_buckling_margin=min_rib_buckling,
            min_rib_crushing_margin=min_rib_crushing,
            stress_margin=stress_margin_val,
            is_feasible=is_feasible,
            factor_of_safety=factor_of_safety,
            # Twist constraint
            tip_twist_deg=tip_twist_deg,
            max_tip_twist_deg=max_twist_deg,
            twist_margin=twist_margin,
            mass_breakdown=mass_breakdown,
            assumptions=assumptions,
        )


def create_elliptical_lift_distribution(
    half_span: float,
    total_lift_N: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create an elliptical lift distribution function.
    
    Args:
        half_span: Half-span length [m]
        total_lift_N: Total lift for half-wing [N]
    
    Returns:
        Function q(y) returning lift per unit span [N/m]
    """
    def lift_distribution(y: np.ndarray) -> np.ndarray:
        eta = np.abs(y) / half_span
        eta = np.clip(eta, 0, 0.999)  # Avoid division by zero at tip
        
        # Elliptical shape: sqrt(1 - eta^2)
        shape = np.sqrt(1 - eta ** 2)
        
        # Scale so integral over half-span = total_lift_N
        # Integral of sqrt(1-eta^2) from 0 to 1 = pi/4
        scale = total_lift_N / (np.pi / 4 * half_span)
        
        return scale * shape
    
    return lift_distribution


def create_bell_lift_distribution(
    half_span: float,
    total_lift_N: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a bell-shaped (Prandtl-D) lift distribution function.
    
    The bell distribution produces proverse yaw and is used in
    flying wing designs for improved handling characteristics.
    
    Args:
        half_span: Half-span length [m]
        total_lift_N: Total lift for half-wing [N]
    
    Returns:
        Function q(y) returning lift per unit span [N/m]
    """
    def lift_distribution(y: np.ndarray) -> np.ndarray:
        eta = np.abs(y) / half_span
        eta = np.clip(eta, 0, 0.999)  # Avoid issues at tip
        
        # Bell shape: (1 - eta²)^1.5
        shape = (1 - eta ** 2) ** 1.5
        
        # Scale so integral over half-span = total_lift_N
        # Integral of (1-x²)^1.5 from 0 to 1 = 3π/16
        scale = total_lift_N / (3 * np.pi / 16 * half_span)
        
        return scale * shape
    
    return lift_distribution


def create_lift_distribution(
    half_span: float,
    total_lift_N: float,
    distribution_type: str = "elliptical",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a lift distribution function of the specified type.
    
    Args:
        half_span: Half-span length [m]
        total_lift_N: Total lift for half-wing [N]
        distribution_type: "elliptical" or "bell"
    
    Returns:
        Function q(y) returning lift per unit span [N/m]
    """
    if distribution_type == "bell":
        return create_bell_lift_distribution(half_span, total_lift_N)
    else:
        return create_elliptical_lift_distribution(half_span, total_lift_N)


def analyze_wingbox_beam(
    sections: List[WingBoxSection],
    spar_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
    skin_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
    spar_material: StructuralMaterial,
    skin_material: StructuralMaterial,
    lift_distribution: Callable[[np.ndarray], np.ndarray],
    moment_distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rib_positions: Optional[List[float]] = None,
    n_points: int = 50,
    factor_of_safety: float = 1.5,
    max_deflection_fraction: float = 0.15,
    # New parameters for enhanced buckling analysis
    stringer_props: Optional[StringerProperties] = None,
    rib_props: Optional[RibProperties] = None,
    boundary_condition: str = "semi_restrained",
    include_curvature: bool = True,
    airfoils: Optional[List[Any]] = None,
    # Control surface and fastener parameters
    control_surface_props: Optional[List[ControlSurfaceProperties]] = None,
    fastener_adhesive_fraction: float = 0.10,
    # Post-buckling analysis (Priority 4)
    post_buckling_config: Optional[PostBucklingConfig] = None,
    # Twist constraint
    max_twist_deg: float = 3.0,
) -> StructuralAnalysisResult:
    """
    Analyze a tapered wingbox beam using AeroSandbox's ImplicitAnalysis.
    
    This uses AeroSandbox's Opti stack for proper aerostructural coupling.
    The ImplicitAnalysis decorator automatically creates and solves the
    optimization problem.
    
    Args:
        sections: List of WingBoxSection defining geometry at span stations
        spar_thickness: Spar web thickness [m] - constant or function of y
        skin_thickness: Skin panel thickness [m] - constant or function of y
        spar_material: Material properties for spars
        skin_material: Material properties for skins
        lift_distribution: Function q(y) giving lift per unit span [N/m]
        moment_distribution: Function m(y) giving pitching moment per span [N*m/m]
        rib_positions: List of spanwise positions of ribs [m] for buckling
        n_points: Number of discretization points
        factor_of_safety: Design factor of safety
        max_deflection_fraction: Max tip deflection as fraction of half-span
        stringer_props: StringerProperties for smeared stiffener analysis
        rib_props: RibProperties for rib thickness and material
        boundary_condition: "simply_supported", "semi_restrained", or "clamped"
        include_curvature: Whether to include curvature effects in buckling
        airfoils: List of airfoil objects (AeroSandbox) for curvature calculation
        control_surface_props: List of ControlSurfaceProperties for elevon/flap mass
        fastener_adhesive_fraction: Mass adder fraction for fasteners/adhesive (default 10%)
        post_buckling_config: PostBucklingConfig for post-buckling analysis (Priority 4)
        max_twist_deg: Maximum allowable tip twist in degrees (default 3.0)
    
    Returns:
        StructuralAnalysisResult with all analysis outputs
    """
    # Create beam analysis - the @ImplicitAnalysis.initialize decorator
    # automatically creates an Opti instance and solves if opti is not provided
    beam = WingBoxBeam(
        sections=sections,
        spar_thickness=spar_thickness,
        skin_thickness=skin_thickness,
        spar_material=spar_material,
        skin_material=skin_material,
        lift_distribution=lift_distribution,
        moment_distribution=moment_distribution,
        rib_positions=rib_positions,
        n_points=n_points,
        stringer_props=stringer_props,
        rib_props=rib_props,
        boundary_condition=boundary_condition,
        include_curvature=include_curvature,
        airfoils=airfoils,
        control_surface_props=control_surface_props,
        fastener_adhesive_fraction=fastener_adhesive_fraction,
        post_buckling_config=post_buckling_config,
    )
    
    # The beam object now contains solved values (ImplicitAnalysis solves automatically)
    
    return beam.get_result(
        factor_of_safety=factor_of_safety,
        max_deflection_fraction=max_deflection_fraction,
        max_twist_deg=max_twist_deg,
    )


# =============================================================================
# THICKNESS OPTIMIZATION
# =============================================================================

@dataclass
class OptimizationResult:
    """Results from thickness optimization with optional stringer count."""
    success: bool
    message: str
    
    # Optimized thicknesses
    spar_thickness_mm: float
    skin_thickness_mm: float
    rib_thickness_mm: float  # Rib thickness [mm]
    
    # Optimized stringer count (per skin panel, total = count * 2 sides * 2 wings)
    stringer_count: int
    
    # Before/after comparison
    initial_mass_kg: float
    optimized_mass_kg: float
    mass_reduction_percent: float
    
    # Constraint margins at optimum
    stress_margin: float
    skin_buckling_margin: float
    spar_buckling_margin: float
    tip_deflection_percent: float
    
    # Rib failure margins (optional, only if ribs are defined)
    rib_buckling_margin: float = float('inf')
    rib_crushing_margin: float = float('inf')
    
    # Priority 3 & 5: New buckling margins
    skin_shear_buckling_margin: float = float('inf')  # Skin shear buckling from torsion
    combined_buckling_margin: float = float('inf')    # Biaxial stress interaction
    
    # Priority 6: Stringer crippling margin
    stringer_crippling_margin: float = float('inf')   # Stringer local buckling
    stringer_failure_mode: Optional[str] = None       # "crippling", "column_buckling", or "yield"
    
    # Twist constraint
    tip_twist_deg: float = 0.0                        # Actual tip twist [deg]
    twist_margin: float = float('inf')                # max_allowable / actual
    
    # Iteration info (for legacy compatibility; now primarily single-shot)
    iterations: int = 0
    
    # Full structural result at optimum
    structural_result: Optional[StructuralAnalysisResult] = None


# =============================================================================
# TIMOSHENKO WING BOX BEAM (Priority 1 - Full Implementation)
# =============================================================================
# This class implements full Timoshenko beam theory within CasADi, treating
# bending rotation (psi) and deflection (w) as independent optimization variables.
# 
# Key differences from Euler-Bernoulli (WingBoxBeam):
# - E-B: slope = dw/dy (no shear deformation)
# - Timoshenko: slope = psi + gamma, where gamma = V/(kappa*GA) is shear strain
#
# For slender beams (L/h > 15), shear is negligible and this reduces to E-B.
# For thick beams (flying wing proportions), shear adds 5-15% to deflection.
# =============================================================================

class TimoshenkoWingBoxBeam(asb.ImplicitAnalysis):
    """
    Full Timoshenko beam model for wing structural analysis.
    
    Treats bending rotation (psi) and deflection (w) as independent CasADi
    optimization variables with proper shear-bending coupling.
    
    Governing equations:
        dV/dy = -q(y)                      # Load equilibrium
        dM/dy = V                          # Moment equilibrium  
        M = EI * dpsi/dy                   # Bending constitutive
        V = kappa*GA * (dw/dy - psi)       # Shear constitutive (KEY)
    
    For thick beams (L/h < 15) typical of flying wings, shear deformation
    contributes significantly (5-15%) to total deflection. This implementation
    captures that effect exactly within the optimization loop.
    
    When kappa*GA is large (slender beams), the solution automatically
    converges to Euler-Bernoulli behavior.
    """
    
    @asb.ImplicitAnalysis.initialize
    def __init__(
        self,
        sections: List[WingBoxSection],
        spar_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
        skin_thickness: Union[float, Callable[[np.ndarray], np.ndarray]],
        spar_material: StructuralMaterial,
        skin_material: StructuralMaterial,
        lift_distribution: Callable[[np.ndarray], np.ndarray],
        moment_distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        rib_positions: Optional[List[float]] = None,
        n_points: int = 50,
        EI_guess: Optional[float] = None,
        GJ_guess: Optional[float] = None,
        stringer_props: Optional[StringerProperties] = None,
        rib_props: Optional['RibProperties'] = None,
        boundary_condition: str = "semi_restrained",
        include_curvature: bool = True,
        airfoils: Optional[List[Any]] = None,
        control_surface_props: Optional[List['ControlSurfaceProperties']] = None,
        fastener_adhesive_fraction: float = 0.10,
        post_buckling_config: Optional['PostBucklingConfig'] = None,
    ):
        """
        Initialize Timoshenko wingbox beam analysis.
        
        Args:
            sections: List of WingBoxSection defining geometry at span stations
            spar_thickness: Spar web thickness [m] - constant or function of y
            skin_thickness: Skin panel thickness [m] - constant or function of y
            spar_material: Material properties for spars
            skin_material: Material properties for skins
            lift_distribution: Function q(y) giving lift per unit span [N/m]
            moment_distribution: Function m(y) giving pitching moment per span [N*m/m]
            rib_positions: List of spanwise positions of ribs [m]
            n_points: Number of discretization points
            EI_guess: Initial guess for bending stiffness (optimizer scaling)
            GJ_guess: Initial guess for torsional stiffness (optimizer scaling)
            stringer_props: StringerProperties for smeared stiffener analysis
            rib_props: RibProperties for rib thickness and material
            boundary_condition: "simply_supported", "semi_restrained", or "clamped"
            include_curvature: Whether to include curvature effects in buckling
            airfoils: List of airfoil objects for curvature calculation
            control_surface_props: List of ControlSurfaceProperties
            fastener_adhesive_fraction: Mass adder fraction for fasteners
            post_buckling_config: PostBucklingConfig for post-buckling analysis
        """
        self.sections = sections
        self.spar_material = spar_material
        self.skin_material = skin_material
        self.n_points = n_points
        
        # Store enhanced buckling parameters
        self._stringer_props = stringer_props
        self._rib_props = rib_props
        self._boundary_condition = boundary_condition
        self._include_curvature = include_curvature
        self._airfoils = airfoils
        self._control_surface_props = control_surface_props or []
        self._fastener_adhesive_fraction = fastener_adhesive_fraction
        self._post_buckling_config = post_buckling_config or PostBucklingConfig()
        
        # Extract spanwise coordinates from sections
        section_y = np.array([s.y for s in sections])
        self.length = float(np.max(section_y))
        
        if self.length <= 0:
            raise ValueError("Sections must have positive span extent")
        
        # Discretize along span
        y = np.linspace(0, self.length, n_points)
        self.y = y
        N = n_points
        
        # Interpolate section properties to discretization points
        box_width = np.interp(y, section_y, np.array([s.box_width for s in sections]))
        box_height = np.interp(y, section_y, np.array([s.box_height for s in sections]))
        
        self.box_width = box_width
        self.box_height = box_height
        
        # Store chord and spar positions
        self._chord = np.interp(y, section_y, np.array([s.chord for s in sections]))
        self._front_spar_xsi = np.interp(y, section_y, np.array([s.front_spar_xsi for s in sections]))
        self._rear_spar_xsi = np.interp(y, section_y, np.array([s.rear_spar_xsi for s in sections]))
        
        # Evaluate thickness functions
        if callable(spar_thickness):
            t_spar = spar_thickness(y)
        else:
            t_spar = spar_thickness * np.ones_like(y)
        
        if callable(skin_thickness):
            t_skin = skin_thickness(y)
        else:
            t_skin = skin_thickness * np.ones_like(y)
        
        self.t_spar = t_spar
        self.t_skin = t_skin
        
        # === Cross-sectional Properties ===
        h = box_height
        w = box_width
        
        # Material properties
        E_spar = spar_material.get_bending_modulus(grain_spanwise=True)
        E_skin = skin_material.get_bending_modulus(grain_spanwise=True)
        G_spar = spar_material.get_shear_modulus()
        G_skin = skin_material.get_shear_modulus()
        
        # Bending inertia I_xx
        A_skin = w * t_skin
        I_skin = 2 * A_skin * (h / 2) ** 2  # Top + bottom skins
        I_spar = 2 * (1 / 12) * t_spar * h ** 3
        I_xx = I_skin + I_spar
        self.I_xx = I_xx
        
        # Weighted EI
        EI = E_skin * I_skin + E_spar * I_spar
        self.EI = EI
        
        self.E_spar = E_spar
        self.E_skin = E_skin
        
        # Torsion constant J (Bredt's formula)
        A_enclosed = w * h
        perimeter_integral = 2 * (w / (t_skin + 1e-10) + h / (t_spar + 1e-10))
        J = 4 * A_enclosed ** 2 / (perimeter_integral + 1e-10)
        self.J = J
        
        # Effective shear modulus
        skin_perimeter = 2 * w
        spar_perimeter = 2 * h
        total_perimeter = skin_perimeter + spar_perimeter + 1e-10
        G_eff = (G_skin * skin_perimeter + G_spar * spar_perimeter) / total_perimeter
        GJ = G_eff * J
        self.GJ = GJ
        
        # === Shear Stiffness for Timoshenko Beam (kappa*GA) ===
        # This is the KEY addition for Timoshenko theory
        G_xz_spar = spar_material.get_transverse_shear_modulus()
        A_shear = 2 * h * t_spar  # Spar webs carry vertical shear
        
        # Shear correction factor for thin-walled box
        if hasattr(spar_material, 'G_xz') and spar_material.G_xz is not None:
            kappa = 0.5  # Orthotropic: lower due to shear lag
        elif spar_material.is_isotropic:
            kappa = 0.6  # Isotropic thin-walled box
        else:
            kappa = 0.5
        
        kappa_GA = kappa * G_xz_spar * A_shear
        self.kappa_GA = kappa_GA
        self._shear_correction_factor = kappa
        self._G_xz_spar = G_xz_spar
        
        # Estimate shear deformation significance
        import numpy as _np_std_shear
        try:
            avg_h = float(_np_std_shear.mean([s.box_height for s in sections]))
            E_avg = (E_spar + E_skin) / 2
            self._shear_ratio_estimate = estimate_shear_deformation_ratio(
                half_span=self.length,
                avg_box_height=avg_h,
                E=E_avg,
                G_xz=G_xz_spar,
            )
        except (ValueError, TypeError, RuntimeError):
            self._shear_ratio_estimate = 0.0
        
        # Initial guesses for stiffness
        if EI_guess is None:
            try:
                EI_guess = float(np.mean(EI))
            except (RuntimeError, TypeError, ValueError):
                import numpy as _np_std
                avg_h = float(_np_std.mean([s.box_height for s in sections]))
                avg_w = float(_np_std.mean([s.box_width for s in sections]))
                t_est = 0.002
                EI_guess = E_skin * avg_w * t_est * (avg_h / 2) ** 2 + E_spar * t_est * avg_h ** 3 / 6
                EI_guess = max(EI_guess, 1.0)
        
        if GJ_guess is None:
            try:
                GJ_guess = float(np.mean(GJ))
            except (RuntimeError, TypeError, ValueError):
                import numpy as _np_std
                avg_h = float(_np_std.mean([s.box_height for s in sections]))
                avg_w = float(_np_std.mean([s.box_width for s in sections]))
                t_est = 0.002
                GJ_guess = G_skin * 4 * (avg_w * avg_h) ** 2 / (2 * (avg_w + avg_h) / t_est)
                GJ_guess = max(GJ_guess, 1.0)
        
        # === Load Distribution ===
        q_aero = lift_distribution(y)
        self.q_aero = q_aero
        
        # Inertial load
        skin_area_per_length = 2 * w * t_skin
        spar_area_per_length = 2 * h * t_spar
        mass_per_length = (skin_area_per_length * skin_material.density + 
                          spar_area_per_length * spar_material.density)
        
        g = 9.80665
        q_inertial = -mass_per_length * g
        self.q_inertial = q_inertial
        
        q_net = q_aero + q_inertial
        self.q_net = q_net
        q_lift = q_net
        self.q_lift = q_lift
        
        if moment_distribution is not None:
            m_twist = moment_distribution(y)
        else:
            m_twist = np.zeros_like(y)
        self.m_twist = m_twist
        
        # Torque distribution
        import numpy as _np_std_torque
        try:
            m_twist_np = _np_std_torque.array(m_twist)
            y_np_torque = _np_std_torque.array(y)
            torque_np = _np_std_torque.zeros_like(y_np_torque)
            for i in range(len(y_np_torque) - 2, -1, -1):
                dy = y_np_torque[i + 1] - y_np_torque[i]
                torque_np[i] = torque_np[i + 1] + 0.5 * (m_twist_np[i] + m_twist_np[i + 1]) * dy
            self.torque = torque_np
        except (ValueError, TypeError, RuntimeError):
            self.torque = np.zeros_like(y)
        
        # Torsional shear stress
        A_enclosed = w * h
        self.A_enclosed = A_enclosed
        T_abs = np.abs(self.torque) if hasattr(self.torque, '__abs__') else _np_std_torque.abs(self.torque)
        self.tau_torsion_skin = T_abs / (2 * A_enclosed * t_skin + 1e-10)
        self.tau_torsion_spar = T_abs / (2 * A_enclosed * t_spar + 1e-10)
        
        # === TIMOSHENKO BEAM ANALYSIS (CasADi Formulation) ===
        # 
        # We have TWO independent variables:
        #   w(y) = transverse deflection [m]
        #   psi(y) = bending rotation [rad] (NOT total slope)
        #
        # The key Timoshenko relationship:
        #   dw/dy = psi + gamma, where gamma = V / (kappa*GA)
        #   => V = kappa*GA * (dw/dy - psi)
        #
        # The bending equation:
        #   M = EI * dpsi/dy
        #   dM/dy = V
        #   => d(EI * dpsi/dy)/dy = -q (load equilibrium through shear)
        #
        # Strategy:
        # 1. Create w and psi as independent CasADi variables
        # 2. Compute derivatives: dw/dy, dpsi/dy
        # 3. Shear equation: V = kappa*GA * (dw/dy - psi)
        # 4. Bending equation constraint: d(EI * dpsi/dy)/dy = q
        # 5. Shear equation constraint: dV/dy = -q (or equivalently, couple w and psi)
        
        import numpy as _np_std
        q_lift_np = _np_std.array(q_aero) if hasattr(q_aero, '__iter__') else q_aero
        y_np = _np_std.array(y)
        try:
            total_load = float(_np_std.trapz(q_lift_np, y_np)) if len(y_np) > 1 else 0.0
        except (ValueError, TypeError):
            total_load = 50.0
        
        disp_scale = max(1e-6, abs(total_load) * self.length ** 3 / (EI_guess * 3) + 1e-10)
        rotation_scale = disp_scale / self.length  # Rotation scales as disp/length
        
        # === Variable 1: Deflection w ===
        w_var = self.opti.variable(
            init_guess=np.zeros_like(y),
            scale=disp_scale,
        )
        
        dw = self.opti.derivative_of(
            w_var, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=rotation_scale,
        )
        
        # === Variable 2: Bending Rotation psi (INDEPENDENT) ===
        psi = self.opti.variable(
            init_guess=np.zeros_like(y),
            scale=rotation_scale,
        )
        
        dpsi = self.opti.derivative_of(
            psi, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=rotation_scale / self.length,
        )
        
        # === Shear Strain ===
        # gamma = dw/dy - psi (shear strain)
        # For visualization and stress calculations
        gamma = dw - psi
        self.gamma = gamma
        
        # === Shear Force from Timoshenko ===
        # V = kappa*GA * gamma = kappa*GA * (dw/dy - psi)
        V_timoshenko = kappa_GA * gamma
        
        # === Bending Moment ===
        # M = EI * dpsi/dy (curvature from bending rotation only)
        M_bending = EI * dpsi
        
        # === Bending Equation Constraint ===
        # d(EI * dpsi/dy)/dy = q  (moment equilibrium -> load)
        # This is the same as E-B but uses psi instead of dw/dy
        dEIdpsi = self.opti.derivative_of(
            EI * dpsi, with_respect_to=y,
            derivative_init_guess=np.zeros_like(y),
            derivative_scale=max(1.0, abs(total_load) * self.length / EI_guess),
        )
        
        self.opti.constrain_derivative(
            variable=dEIdpsi,
            with_respect_to=y,
            derivative=q_lift,
        )
        
        # === Timoshenko Coupling Constraint ===
        # The shear force from bending equilibrium must equal shear force from shear strain
        # dM/dy = V => d(EI * dpsi/dy)/dy = V
        # So: dEIdpsi = V_timoshenko = kappa*GA * (dw/dy - psi)
        #
        # Constraint: dEIdpsi - kappa_GA * (dw - psi) = 0 at each point
        # This ensures w and psi are coupled through shear stiffness
        
        # For very stiff beams (high kappa_GA), gamma -> 0 and dw/dy -> psi (E-B limit)
        # For flexible beams (low kappa_GA), significant shear deformation
        
        # Shear compatibility: V from equilibrium = V from constitutive
        # dEIdpsi = -V (convention: positive V from positive load)
        # kappa_GA * (dw - psi) = V
        # => dEIdpsi + kappa_GA * (dw - psi) = 0
        # But load equilibrium gives: dV/dy = -q, and we already constrained d(dEIdpsi)/dy = q
        # So we need: dEIdpsi = -kappa_GA * (dw - psi) at each point... 
        # Actually, let's be careful about signs.
        #
        # Standard Timoshenko:
        #   M = EI * psi'
        #   V = kappa*GA * (w' - psi)
        #   M' = V  =>  (EI * psi')' = kappa*GA * (w' - psi)
        #   V' = -q
        #
        # Combining: (EI * psi')' = V, and we want (EI * psi')'' = -q
        # But we also have V = kappa*GA * (w' - psi), so V' = kappa*GA * (w'' - psi')
        # Setting V' = -q: kappa*GA * (w'' - psi') = -q
        #
        # Approach: We already have the bending equation via dEIdpsi.
        # Now we need to ensure the shear constitutive relation holds.
        # 
        # Let's use a soft coupling via constraint:
        # shear_force_bending = dEIdpsi (from bending equation)
        # shear_force_shear = kappa_GA * (dw - psi) (from shear constitutive)
        # Constraint: shear_force_bending + shear_force_shear ≈ 0 (they should be negatives due to sign convention)
        
        # Actually, the cleanest approach:
        # We have (EI * psi')' = q (already constrained)
        # We need w' = psi + V/(kappa*GA)
        # Where V = integral of q from y to tip
        #
        # Since V = integral of q from y to tip, and we've computed q above,
        # we can constrain: dw = psi + dEIdpsi / kappa_GA
        # (because dEIdpsi = V at each point from the bending equation)
        
        # Constraint: dw/dy - psi = V / (kappa*GA) = dEIdpsi / kappa_GA
        # => kappa_GA * (dw - psi) - dEIdpsi = 0
        shear_residual = kappa_GA * (dw - psi) - (-dEIdpsi)  # Note: dEIdpsi = -V by convention
        
        # We add this as a soft constraint (equality at each point)
        # However, this might over-constrain. Let's use a different approach:
        # 
        # Alternative: since dEIdpsi integrates to give shear, and dw-psi = V/(kappa*GA),
        # we can compute V directly and constrain the relationship.
        
        # Simpler approach for CasADi: 
        # Just constrain that the shear strain gamma satisfies the equilibrium.
        # gamma = V / (kappa*GA), where V comes from integrating q.
        #
        # Actually, the cleanest implementation:
        # We note that dEIdpsi = d(EI*dpsi)/dy represents the shear force V.
        # The Timoshenko shear constitutive says V = kappa*GA*(dw/dy - psi)
        # So: dw/dy = psi + V/(kappa*GA) = psi + dEIdpsi/(kappa*GA)
        # Wait, sign issue. Let's be very careful:
        #
        # Standard beam sign convention:
        #   M = EI * kappa = EI * dpsi/dy
        #   V = dM/dy = d(EI*dpsi/dy)/dy
        #   q = -dV/dy = -d²(EI*dpsi/dy)/dy²
        #
        # We constrained: d(dEIdpsi)/dy = q, which means dEIdpsi = integral of q = V
        # So dEIdpsi IS the shear force V.
        #
        # Timoshenko: V = kappa*GA*(dw/dy - psi)
        # => dw/dy - psi = V/(kappa*GA) = dEIdpsi/(kappa*GA)
        # => dw = psi + V/kappa_GA = psi - dEIdpsi/kappa_GA (since V = -dEIdpsi)
        
        # This gives us the constraint we need:
        # Instead of computing dw from opti.derivative_of, we constrain it.
        # But we already have dw as a derivative. So we add:
        # Note: shear_force = -dEIdpsi, so V = -dEIdpsi
        # Therefore: dw = psi + V/(kappa_GA) = psi + (-dEIdpsi)/(kappa_GA) = psi - dEIdpsi/kappa_GA
        self.opti.subject_to(dw == psi - dEIdpsi / (kappa_GA + 1e-10))
        
        # Wait, but dEIdpsi is already a variable. The above says dw should equal
        # psi + dEIdpsi/kappa_GA at every point. This IS the Timoshenko coupling.
        #
        # Note: For large kappa_GA (stiff shear), dEIdpsi/kappa_GA -> 0, so dw -> psi
        # which is exactly Euler-Bernoulli.
        
        # === Boundary Conditions (Cantilever: root clamped, tip free) ===
        self.opti.subject_to([
            w_var[0] == 0,    # Zero deflection at root
            psi[0] == 0,      # Zero bending rotation at root
            dpsi[-1] == 0,    # Zero curvature at tip (M = EI*dpsi/dy = 0)
            dEIdpsi[-1] == 0  # Zero shear at tip (V = 0)
        ])
        
        # Store solution variables
        self.w = w_var          # Deflection [m]
        self.u = w_var          # Alias for compatibility with WingBoxBeam
        self.psi = psi          # Bending rotation [rad]
        self.dw = dw            # Total slope [rad]
        self.dpsi = dpsi        # Curvature (bending only) [1/m]
        self.du = dw            # Alias for compatibility (total slope)
        self.ddu = dpsi         # Alias for compatibility (curvature from bending)
        
        # Derived quantities
        self.bending_moment = -EI * dpsi  # [N*m] (sign convention)
        self.shear_force = -dEIdpsi       # [N]
        
        # === Stress Calculations ===
        # Bending stress uses curvature from psi, not total slope
        self.sigma_spar = E_spar * dpsi * (h / 2)
        self.sigma_skin = E_skin * dpsi * (h / 2)
        
        # Stress ratios
        spar_tension_ratio = np.where(
            self.sigma_spar >= 0,
            self.sigma_spar / (spar_material.sigma_1_tension + 1e-10),
            np.abs(self.sigma_spar) / (spar_material.sigma_1_compression + 1e-10)
        )
        
        skin_tension_ratio = np.where(
            self.sigma_skin >= 0,
            self.sigma_skin / (skin_material.sigma_1_tension + 1e-10),
            np.abs(self.sigma_skin) / (skin_material.sigma_1_compression + 1e-10)
        )
        
        self._max_stress_ratio = np.maximum(spar_tension_ratio, skin_tension_ratio)
        
        # === Store for buckling analysis ===
        import numpy as _np_std
        self._rib_positions = rib_positions
        
        if rib_positions is not None and len(rib_positions) >= 2:
            sorted_ribs = sorted(rib_positions)
            spacings = [sorted_ribs[i+1] - sorted_ribs[i] for i in range(len(sorted_ribs)-1)]
            rib_spacing = sum(spacings) / len(spacings) if spacings else self.length
        else:
            rib_spacing = self.length / max(1, len(sections) - 1)
        self.rib_spacing = rib_spacing
        
        self._stringer_count = stringer_props.count if stringer_props is not None else 0
        self._sections = sections
        
        # Curvature radius calculation
        curvature_radii = _np_std.full(n_points, float('inf'))
        if include_curvature and airfoils is not None and len(airfoils) > 0:
            section_y_np = _np_std.array([s.y for s in sections])
            section_front_spar = _np_std.array([s.front_spar_xsi for s in sections])
            section_rear_spar = _np_std.array([s.rear_spar_xsi for s in sections])
            section_chord = _np_std.array([s.chord for s in sections])
            y_np = _np_std.array([float(yi) for yi in y]) if hasattr(y[0], '__float__') else _np_std.array(y)
            
            for i in range(len(y_np)):
                idx = _np_std.argmin(_np_std.abs(section_y_np - y_np[i]))
                if idx < len(airfoils):
                    airfoil = airfoils[idx]
                    front_spar = float(section_front_spar[idx])
                    rear_spar = float(section_rear_spar[idx])
                    chord = float(section_chord[idx])
                    
                    R = calculate_skin_curvature_radius(
                        airfoil=airfoil,
                        front_spar_xsi=front_spar,
                        rear_spar_xsi=rear_spar,
                        chord=chord,
                    )
                    curvature_radii[i] = R
        
        self._curvature_radii = curvature_radii
        self._skin_material = skin_material
        self._spar_material = spar_material
        
        k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(boundary_condition, 7.0)
        self._k_shear = k_shear
    
    def max_stress_ratio(self) -> float:
        """Maximum stress ratio (applied/allowable) across the span."""
        return float(np.max(self._max_stress_ratio))
    
    def min_buckling_margin(self) -> float:
        """Minimum buckling margin (allowable/applied - 1) across span."""
        # Delegate to get_result() for full buckling analysis
        result = self.get_result()
        return result.min_buckling_margin or 0.0
    
    def tip_deflection(self) -> float:
        """Tip deflection in meters."""
        return float(self.w[-1])
    
    def is_feasible(
        self,
        stress_margin: float = 0.0,
        buckling_margin: float = 0.0,
        max_deflection_frac: float = 0.15,
    ) -> bool:
        """
        Check if the design meets all structural constraints.
        
        Args:
            stress_margin: Required margin on stress (0.0 = exactly at allowable)
            buckling_margin: Required margin on buckling
            max_deflection_frac: Maximum tip deflection as fraction of span
        
        Returns:
            True if all constraints are satisfied
        """
        # Stress check
        if self.max_stress_ratio() > (1.0 - stress_margin):
            return False
        
        # Deflection check
        if abs(self.tip_deflection()) / self.length > max_deflection_frac:
            return False
        
        # Buckling check (via get_result)
        result = self.get_result()
        if result.min_buckling_margin is not None:
            if result.min_buckling_margin < buckling_margin:
                return False
        
        return True
    
    def mass(self) -> float:
        """Total structural mass [kg] for half-span."""
        # Same calculation as WingBoxBeam
        import numpy as _np_std
        
        y_np = _np_std.array(self.y)
        t_spar_np = _np_std.array(self.t_spar)
        t_skin_np = _np_std.array(self.t_skin)
        box_w_np = _np_std.array(self.box_width)
        box_h_np = _np_std.array(self.box_height)
        
        skin_area = 2 * box_w_np * t_skin_np
        spar_area = 2 * box_h_np * t_spar_np
        
        skin_mass_per_length = skin_area * self.skin_material.density
        spar_mass_per_length = spar_area * self.spar_material.density
        
        total_mass_per_length = skin_mass_per_length + spar_mass_per_length
        
        # Integrate along span
        mass = float(_np_std.trapz(total_mass_per_length, y_np))
        
        return mass
    
    def get_result(self, factor_of_safety: float = 1.5) -> 'StructuralAnalysisResult':
        """
        Get comprehensive structural analysis results.
        
        Returns StructuralAnalysisResult with deflections, stresses, margins, etc.
        This is compatible with WingBoxBeam.get_result() for interchangeability.
        """
        import numpy as _np_std
        
        # Extract numpy arrays from solution
        y_np = _np_std.array(self.y)
        
        try:
            w_np = _np_std.array([float(wi) for wi in self.w])
        except (RuntimeError, TypeError):
            w_np = _np_std.zeros_like(y_np)
        
        try:
            psi_np = _np_std.array([float(pi) for pi in self.psi])
        except (RuntimeError, TypeError):
            psi_np = _np_std.zeros_like(y_np)
        
        try:
            gamma_np = _np_std.array([float(gi) for gi in self.gamma])
        except (RuntimeError, TypeError):
            gamma_np = _np_std.zeros_like(y_np)
        
        try:
            M_np = _np_std.array([float(mi) for mi in self.bending_moment])
        except (RuntimeError, TypeError):
            M_np = _np_std.zeros_like(y_np)
        
        try:
            V_np = _np_std.array([float(vi) for vi in self.shear_force])
        except (RuntimeError, TypeError):
            V_np = _np_std.zeros_like(y_np)
        
        try:
            sigma_spar_np = _np_std.array([float(si) for si in self.sigma_spar])
        except (RuntimeError, TypeError):
            sigma_spar_np = _np_std.zeros_like(y_np)
        
        try:
            sigma_skin_np = _np_std.array([float(si) for si in self.sigma_skin])
        except (RuntimeError, TypeError):
            sigma_skin_np = _np_std.zeros_like(y_np)
        
        # For buckling analysis, we'll compute basic margins here
        # (Full buckling analysis would require the _compute_buckling_data method from WingBoxBeam)
        t_spar_np = _np_std.array(self.t_spar) if hasattr(self.t_spar, '__iter__') else _np_std.full_like(y_np, float(self.t_spar))
        t_skin_np = _np_std.array(self.t_skin) if hasattr(self.t_skin, '__iter__') else _np_std.full_like(y_np, float(self.t_skin))
        box_w_np = _np_std.array(self.box_width)
        box_h_np = _np_std.array(self.box_height)
        
        # Simple skin buckling calculation (same as used in WingBoxBeam)
        # Critical stress for plate buckling: σ_cr = k * π² * E * (t/b)² / (12 * (1 - ν²))
        E_skin = self._skin_material.E_1
        nu_skin = self._skin_material.nu_12
        k_compression = SKIN_BUCKLING_COEFFICIENTS.get(self._boundary_condition, 4.0)
        
        # Panel width (distance between stringers or ribs)
        n_stringers = self._stringer_count
        panel_width = box_w_np / (n_stringers + 1) if n_stringers > 0 else box_w_np
        
        # Critical buckling stress
        sigma_cr_skin = k_compression * _np_std.pi**2 * E_skin * (t_skin_np / panel_width)**2 / (12 * (1 - nu_skin**2) + 1e-10)
        
        # Buckling margin = σ_cr / |σ_applied| - 1
        sigma_applied_abs = _np_std.abs(sigma_skin_np) + 1e-10
        skin_buckling_margin = (sigma_cr_skin / sigma_applied_abs) - 1.0
        min_skin_buckling_margin = float(_np_std.min(skin_buckling_margin))
        
        # Spar shear buckling (simplified)
        G_spar = self._spar_material.G_12
        k_shear = SHEAR_BUCKLING_COEFFICIENTS.get(self._boundary_condition, 7.0)
        tau_cr_spar = k_shear * _np_std.pi**2 * G_spar * (t_spar_np / box_h_np)**2 / (12 * (1 - nu_skin**2) + 1e-10)
        tau_spar_applied = _np_std.abs(V_np) / (2 * box_h_np * t_spar_np + 1e-10)
        spar_buckling_margin = (tau_cr_spar / (tau_spar_applied + 1e-10)) - 1.0
        min_spar_buckling_margin = float(_np_std.min(spar_buckling_margin))
        
        # Mass calculation
        mass = self.mass()
        
        # Stress margin
        max_stress_ratio = self.max_stress_ratio()
        stress_margin = (1.0 / (max_stress_ratio + 1e-10)) - 1.0 if max_stress_ratio > 0 else float('inf')
        
        # Tip deflection
        tip_deflection = float(w_np[-1]) if len(w_np) > 0 else 0.0
        
        # Shear deformation contribution
        # Total deflection = bending + shear
        # Shear contribution = integral of gamma
        shear_deflection = float(_np_std.trapz(gamma_np, y_np)) if len(y_np) > 1 else 0.0
        bending_deflection = tip_deflection - shear_deflection
        shear_fraction = abs(shear_deflection) / (abs(tip_deflection) + 1e-10) if tip_deflection != 0 else 0.0
        
        return StructuralAnalysisResult(
            y=y_np.tolist(),
            displacement=w_np.tolist(),
            slope=psi_np.tolist(),  # Bending rotation (Timoshenko psi)
            curvature=(_np_std.gradient(psi_np, y_np) if len(y_np) > 1 else _np_std.zeros_like(y_np)).tolist(),
            bending_moment=M_np.tolist(),
            shear_force=V_np.tolist(),
            aero_load=(_np_std.array(self.q_aero) if hasattr(self.q_aero, '__iter__') else _np_std.zeros_like(y_np)).tolist(),
            inertial_load=(_np_std.array(self.q_inertial) if hasattr(self.q_inertial, '__iter__') else _np_std.zeros_like(y_np)).tolist(),
            net_load=(_np_std.array(self.q_net) if hasattr(self.q_net, '__iter__') else _np_std.zeros_like(y_np)).tolist(),
            sigma_spar=sigma_spar_np.tolist(),
            sigma_skin=sigma_skin_np.tolist(),
            max_stress_ratio=(_np_std.abs(sigma_skin_np) / (self._skin_material.sigma_1_compression + 1e-10)).tolist(),
            buckling_margin=skin_buckling_margin.tolist(),
            EI=(_np_std.array(self.EI) if hasattr(self.EI, '__iter__') else _np_std.full_like(y_np, float(self.EI))).tolist(),
            GJ=(_np_std.array(self.GJ) if hasattr(self.GJ, '__iter__') else _np_std.full_like(y_np, float(self.GJ))).tolist(),
            mass_kg=mass,
            tip_deflection_m=tip_deflection,
            max_stress_MPa=float(_np_std.max(_np_std.abs(sigma_skin_np))) / 1e6,
            min_buckling_margin=min_skin_buckling_margin,
            stress_margin=stress_margin,
            is_feasible=stress_margin > 0 and min_skin_buckling_margin > 0,
            spar_buckling_margin=spar_buckling_margin.tolist(),
            min_spar_buckling_margin=min_spar_buckling_margin,
            # Timoshenko-specific fields
            shear_strain=gamma_np.tolist(),
            shear_deflection_contribution=shear_fraction,
            is_timoshenko=True,
            post_buckling_enabled=self._post_buckling_config.enabled if self._post_buckling_config else False,
        )



def optimize_wingbox_thickness(
    sections: List[WingBoxSection],
    spar_material: StructuralMaterial,
    skin_material: StructuralMaterial,
    lift_distribution: Callable[[np.ndarray], np.ndarray],
    moment_distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rib_positions: Optional[List[float]] = None,
    # Thickness bounds [mm]
    spar_thickness_bounds: Tuple[float, float] = (1.0, 15.0),
    skin_thickness_bounds: Tuple[float, float] = (0.5, 10.0),
    rib_thickness_bounds: Tuple[float, float] = (1.0, 10.0),
    # Initial guess [mm]
    spar_thickness_init: float = 3.0,
    skin_thickness_init: float = 1.5,
    rib_thickness_init: float = 3.0,
    # Stringer optimization
    stringer_count_range: Tuple[int, int] = (0, 8),  # (min, max) stringers per skin panel
    stringer_height_m: float = 0.010,
    stringer_thickness_m: float = 0.0015,
    # Constraints
    factor_of_safety: float = 1.5,
    max_deflection_fraction: float = 0.15,
    max_twist_deg: float = 3.0,
    # Optional properties
    rib_props: Optional[RibProperties] = None,
    boundary_condition: str = "semi_restrained",
    include_curvature: bool = True,
    airfoils: Optional[List[Any]] = None,
    control_surface_props: Optional[List[ControlSurfaceProperties]] = None,
    fastener_adhesive_fraction: float = 0.10,
    # Post-buckling analysis (Priority 4)
    post_buckling_config: Optional[PostBucklingConfig] = None,
    # Optimization settings
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize spar thickness, skin thickness, and stringer count to minimize structural mass.
    
    Uses AeroSandbox's Opti stack for gradient-based optimization of continuous variables
    (thicknesses), with a multi-start approach for the discrete stringer count.
    
    All buckling constraints (skin and spar web) are enforced directly within the
    optimization using KS (Kreisselmeier-Steinhauser) aggregation for smooth gradients.
    
    Args:
        sections: List of WingBoxSection defining geometry
        spar_material: Material for spars
        skin_material: Material for skins
        lift_distribution: Lift per unit span function
        moment_distribution: Pitching moment per unit span function [N*m/m] (for torsion)
        rib_positions: Rib locations for buckling panel length
        spar_thickness_bounds: (min, max) spar thickness in mm
        skin_thickness_bounds: (min, max) skin thickness in mm
        spar_thickness_init: Initial spar thickness guess [mm]
        skin_thickness_init: Initial skin thickness guess [mm]
        stringer_count_range: (min, max) stringers per skin panel to evaluate
        stringer_height_m: Stringer height [m]
        stringer_thickness_m: Stringer web thickness [m]
        factor_of_safety: Required safety margin for all constraints
        max_deflection_fraction: Max tip deflection as fraction of span
        max_twist_deg: Max tip twist (bending + torsion) in degrees
        rib_props: Rib properties for mass calculation
        boundary_condition: Panel edge support condition
        include_curvature: Include curvature effects in buckling
        airfoils: Airfoil objects for curvature calculation
        control_surface_props: Control surface definitions
        fastener_adhesive_fraction: Fastener mass allowance fraction
        verbose: Print optimization progress
    
    Returns:
        OptimizationResult with optimized thicknesses, stringer count, and comparison
    """
    import numpy as _np_std
    
    # Get geometry parameters
    half_span = max(s.y for s in sections)
    section_y = _np_std.array([s.y for s in sections])
    box_widths = _np_std.array([s.box_width for s in sections])
    box_heights = _np_std.array([s.box_height for s in sections])
    avg_w = float(_np_std.mean(box_widths))
    avg_h = float(_np_std.mean(box_heights))
    
    # Calculate rib spacing for buckling panel length
    if rib_positions is not None and len(rib_positions) >= 2:
        sorted_ribs = sorted(rib_positions)
        rib_spacing = float(_np_std.mean(_np_std.diff(sorted_ribs)))
    else:
        rib_spacing = half_span / max(1, len(sections) - 1)
    
    # Get initial analysis for comparison (with initial stringer props if provided)
    initial_stringer_props = StringerProperties(
        count=stringer_count_range[0],
        height_m=stringer_height_m,
        thickness_m=stringer_thickness_m,
        material=skin_material,
    ) if stringer_count_range[0] > 0 else None
    
    # Create initial rib props with the initial thickness
    initial_rib_props = None
    if rib_props is not None:
        initial_rib_props = RibProperties(
            thickness_m=rib_thickness_init / 1000,
            material=rib_props.material,
            lightening_hole_fraction=rib_props.lightening_hole_fraction,
            spar_cap_width_m=rib_props.spar_cap_width_m,
            boundary_condition=rib_props.boundary_condition,
        )
    
    initial_result = analyze_wingbox_beam(
        sections=sections,
        spar_thickness=spar_thickness_init / 1000,
        skin_thickness=skin_thickness_init / 1000,
        spar_material=spar_material,
        skin_material=skin_material,
        lift_distribution=lift_distribution,
        rib_positions=rib_positions,
        factor_of_safety=factor_of_safety,
        max_deflection_fraction=max_deflection_fraction,
        stringer_props=initial_stringer_props,
        rib_props=initial_rib_props,
        boundary_condition=boundary_condition,
        include_curvature=include_curvature,
        airfoils=airfoils,
        control_surface_props=control_surface_props,
        fastener_adhesive_fraction=fastener_adhesive_fraction,
    )
    
    initial_mass = initial_result.mass_kg
    
    # Note: Rib thickness is now an optimization variable, so we don't pre-check rib constraints.
    # The optimizer will find the minimum rib thickness that satisfies buckling and crushing.
    
    if verbose:
        print(f"[Optimization] Initial mass: {initial_mass:.3f} kg")
        print(f"[Optimization] Initial thicknesses: spar={spar_thickness_init:.1f}mm, skin={skin_thickness_init:.1f}mm, rib={rib_thickness_init:.1f}mm")
        print(f"[Optimization] Stringer count range: {stringer_count_range[0]} to {stringer_count_range[1]}")
    
    # Multi-start optimization over stringer counts
    # Evaluate even stringer counts (0, 2, 4, 6, ...) for symmetric panels
    stringer_counts = list(range(stringer_count_range[0], stringer_count_range[1] + 1, 2))
    if stringer_count_range[0] % 2 != 0:
        stringer_counts = list(range(stringer_count_range[0], stringer_count_range[1] + 1, 1))
    
    best_result: Optional[OptimizationResult] = None
    best_mass = float('inf')
    
    for n_stringers in stringer_counts:
        if verbose:
            print(f"\n[Optimization] Trying n_stringers = {n_stringers}")
        
        try:
            result = _optimize_for_stringer_count(
                sections=sections,
                spar_material=spar_material,
                skin_material=skin_material,
                lift_distribution=lift_distribution,
                moment_distribution=moment_distribution,
                rib_positions=rib_positions,
                rib_spacing=rib_spacing,
                spar_thickness_bounds=spar_thickness_bounds,
                skin_thickness_bounds=skin_thickness_bounds,
                rib_thickness_bounds=rib_thickness_bounds,
                spar_thickness_init=spar_thickness_init,
                skin_thickness_init=skin_thickness_init,
                rib_thickness_init=rib_thickness_init,
                n_stringers=n_stringers,
                stringer_height_m=stringer_height_m,
                stringer_thickness_m=stringer_thickness_m,
                factor_of_safety=factor_of_safety,
                max_deflection_fraction=max_deflection_fraction,
                max_twist_deg=max_twist_deg,
                rib_props=rib_props,
                boundary_condition=boundary_condition,
                include_curvature=include_curvature,
                airfoils=airfoils,
                control_surface_props=control_surface_props,
                fastener_adhesive_fraction=fastener_adhesive_fraction,
                post_buckling_config=post_buckling_config,
                avg_w=avg_w,
                avg_h=avg_h,
                half_span=half_span,
                verbose=verbose,
            )
            
            if result.success and result.structural_result is not None:
                if result.structural_result.is_feasible and result.optimized_mass_kg < best_mass:
                    best_mass = result.optimized_mass_kg
                    best_result = result
                    if verbose:
                        print(f"  -> New best: mass={best_mass:.3f}kg, feasible=True")
                elif not result.structural_result.is_feasible and verbose:
                    sr = result.structural_result
                    rib_buck = sr.min_rib_buckling_margin if sr.min_rib_buckling_margin else float('inf')
                    rib_crush = sr.min_rib_crushing_margin if sr.min_rib_crushing_margin else float('inf')
                    print(f"  -> Not feasible (stress={result.stress_margin:.2f}, skin_buck={result.skin_buckling_margin:.2f}, "
                          f"spar_buck={result.spar_buckling_margin:.2f}, rib_buck={rib_buck:.2f}, rib_crush={rib_crush:.2f})")
            elif verbose:
                print(f"  -> Optimization failed: {result.message}")
                
        except Exception as e:
            if verbose:
                print(f"  -> Exception: {e}")
    
    # If no feasible solution found, return best infeasible or initial
    if best_result is None:
        if verbose:
            print("\n[Optimization] No feasible solution found. Returning initial configuration.")
        
        return OptimizationResult(
            success=False,
            message="No feasible solution found across all stringer counts",
            spar_thickness_mm=spar_thickness_init,
            skin_thickness_mm=skin_thickness_init,
            rib_thickness_mm=rib_thickness_init,
            stringer_count=stringer_count_range[0],
            initial_mass_kg=initial_mass,
            optimized_mass_kg=initial_mass,
            mass_reduction_percent=0.0,
            stress_margin=initial_result.stress_margin,
            skin_buckling_margin=initial_result.min_buckling_margin,
            spar_buckling_margin=initial_result.min_spar_buckling_margin or float('inf'),
            rib_buckling_margin=initial_result.min_rib_buckling_margin or float('inf'),
            rib_crushing_margin=initial_result.min_rib_crushing_margin or float('inf'),
            tip_deflection_percent=abs(initial_result.tip_deflection_m) / half_span * 100,
            tip_twist_deg=initial_result.tip_twist_deg or 0.0,
            twist_margin=initial_result.twist_margin or float('inf'),
            iterations=len(stringer_counts),
            structural_result=initial_result,
        )
    
    # Compute mass reduction
    mass_reduction = (initial_mass - best_result.optimized_mass_kg) / initial_mass * 100
    best_result.initial_mass_kg = initial_mass
    best_result.mass_reduction_percent = mass_reduction
    
    if verbose:
        print(f"\n[Optimization] === BEST SOLUTION ===")
        print(f"  Spar thickness: {best_result.spar_thickness_mm:.2f} mm")
        print(f"  Skin thickness: {best_result.skin_thickness_mm:.2f} mm")
        print(f"  Rib thickness: {best_result.rib_thickness_mm:.2f} mm")
        print(f"  Stringer count: {best_result.stringer_count} per panel")
        print(f"  Mass: {best_result.optimized_mass_kg:.3f} kg ({mass_reduction:.1f}% reduction)")
        print(f"  Feasible: {best_result.structural_result.is_feasible if best_result.structural_result else False}")
    
    return best_result


def _optimize_for_stringer_count(
    sections: List[WingBoxSection],
    spar_material: StructuralMaterial,
    skin_material: StructuralMaterial,
    lift_distribution: Callable[[np.ndarray], np.ndarray],
    moment_distribution: Optional[Callable[[np.ndarray], np.ndarray]],
    rib_positions: Optional[List[float]],
    rib_spacing: float,
    spar_thickness_bounds: Tuple[float, float],
    skin_thickness_bounds: Tuple[float, float],
    rib_thickness_bounds: Tuple[float, float],
    spar_thickness_init: float,
    skin_thickness_init: float,
    rib_thickness_init: float,
    n_stringers: int,
    stringer_height_m: float,
    stringer_thickness_m: float,
    factor_of_safety: float,
    max_deflection_fraction: float,
    max_twist_deg: float,
    rib_props: Optional[RibProperties],
    boundary_condition: str,
    include_curvature: bool,
    airfoils: Optional[List[Any]],
    control_surface_props: Optional[List[ControlSurfaceProperties]],
    fastener_adhesive_fraction: float,
    post_buckling_config: Optional[PostBucklingConfig],
    avg_w: float,
    avg_h: float,
    half_span: float,
    verbose: bool,
) -> OptimizationResult:
    """
    Internal function: optimize thicknesses for a fixed stringer count.
    
    Uses direct buckling constraints via KS aggregation.
    """
    import numpy as _np_std
    
    # Create stringer properties for this iteration
    stringer_props = StringerProperties(
        count=n_stringers,
        height_m=stringer_height_m,
        thickness_m=stringer_thickness_m,
        material=skin_material,
    ) if n_stringers > 0 else None
    
    # Panel width depends on stringer count
    if n_stringers > 0:
        panel_width = avg_w / (n_stringers + 1)
    else:
        panel_width = avg_w
    
    # Create optimization problem
    opti = asb.Opti()
    
    # Design variables: thicknesses in meters
    t_spar = opti.variable(
        init_guess=spar_thickness_init / 1000,
        scale=0.001,
        lower_bound=spar_thickness_bounds[0] / 1000,
        upper_bound=spar_thickness_bounds[1] / 1000,
    )
    t_skin = opti.variable(
        init_guess=skin_thickness_init / 1000,
        scale=0.001,
        lower_bound=skin_thickness_bounds[0] / 1000,
        upper_bound=skin_thickness_bounds[1] / 1000,
    )
    
    # Rib thickness variable (only if rib_props provided)
    t_rib = None
    if rib_props is not None:
        t_rib = opti.variable(
            init_guess=rib_thickness_init / 1000,
            scale=0.001,
            lower_bound=rib_thickness_bounds[0] / 1000,
            upper_bound=rib_thickness_bounds[1] / 1000,
        )
    
    # Build structural model with variable thicknesses
    # Note: WingBoxBeam uses rib_props for mass calculation, but we'll handle
    # rib constraints separately since t_rib is symbolic
    beam = WingBoxBeam(
        opti=opti,
        sections=sections,
        spar_thickness=t_spar,
        skin_thickness=t_skin,
        spar_material=spar_material,
        skin_material=skin_material,
        lift_distribution=lift_distribution,
        moment_distribution=moment_distribution,
        rib_positions=rib_positions,
        stringer_props=stringer_props,
        rib_props=rib_props,
        boundary_condition=boundary_condition,
        include_curvature=include_curvature,
        airfoils=airfoils,
        control_surface_props=control_surface_props,
        fastener_adhesive_fraction=fastener_adhesive_fraction,
        post_buckling_config=post_buckling_config,
    )
    
    # === OBJECTIVE: Minimize mass ===
    section_y = np.array([s.y for s in sections])
    box_width = np.interp(beam.y, section_y, np.array([s.box_width for s in sections]))
    box_height = np.interp(beam.y, section_y, np.array([s.box_height for s in sections]))
    
    # Primary structure volume
    skin_volume = 2 * np.sum(box_width[:-1] * t_skin * np.diff(beam.y))
    spar_volume = 2 * np.sum(box_height[:-1] * t_spar * np.diff(beam.y))
    
    # Stringer volume (if present)
    stringer_volume = 0.0
    if n_stringers > 0:
        stringer_area = stringer_height_m * stringer_thickness_m
        stringer_volume = n_stringers * 2 * stringer_area * half_span  # Both skins
    
    # Rib volume (if ribs present and t_rib is a variable)
    rib_volume = 0.0
    n_ribs = len(rib_positions) if rib_positions else 0
    if rib_props is not None and t_rib is not None and n_ribs > 0:
        # Average rib area = box_width * box_height (approximate)
        avg_rib_area = avg_w * avg_h
        # Apply lightening factor
        lightening = 1.0 - rib_props.lightening_hole_fraction
        # Total rib volume (all ribs, both wing halves)
        rib_volume = n_ribs * avg_rib_area * t_rib * lightening * 2
    
    approx_mass = (
        skin_volume * skin_material.density +
        spar_volume * spar_material.density +
        stringer_volume * skin_material.density
    ) * 2  # Full wing (both halves)
    
    # Add rib mass if applicable
    if rib_props is not None and t_rib is not None:
        approx_mass = approx_mass + rib_volume * rib_props.material.density
    
    opti.minimize(approx_mass)
    
    # === CONSTRAINT 1: Stress margin >= FOS ===
    stress_ratio = beam._max_stress_ratio
    max_stress_ratio = np.max(stress_ratio)
    opti.subject_to(max_stress_ratio <= 1.0 / factor_of_safety)
    
    # === CONSTRAINT 2: Tip deflection <= max allowed ===
    max_tip_deflection = half_span * max_deflection_fraction
    opti.subject_to(np.abs(beam.u[-1]) <= max_tip_deflection)
    
    # === CONSTRAINT 2b: Tip twist <= max allowed ===
    # Total twist = bending twist (slope) + torsional twist (∫T/GJ dy)
    # Bending twist at tip
    bending_twist_rad = beam.du[-1]  # Slope at tip [rad]
    
    # Torsional twist: integrate T/GJ from root to tip
    # Use trapezoidal integration (compatible with CasADi)
    if hasattr(beam, 'torque') and hasattr(beam, 'GJ'):
        torque_arr = np.array(beam.torque)
        GJ_arr = np.array(beam.GJ) if hasattr(beam.GJ, '__iter__') else np.full_like(beam.y, float(beam.GJ))
        twist_rate = torque_arr / (GJ_arr + 1e-10)  # rad/m
        # Trapezoidal integration
        dy = np.diff(beam.y)
        twist_increments = 0.5 * (twist_rate[:-1] + twist_rate[1:]) * dy
        torsion_twist_rad = np.sum(twist_increments)
    else:
        torsion_twist_rad = 0.0
    
    # Total tip twist (we care about magnitude)
    total_twist_rad = np.abs(bending_twist_rad) + np.abs(torsion_twist_rad)
    max_twist_rad = max_twist_deg * np.pi / 180.0
    opti.subject_to(total_twist_rad <= max_twist_rad)
    
    # === CONSTRAINT 3: Skin buckling via KS aggregation ===
    # Use the SAME calculation as _compute_buckling_data() for consistency
    # Evaluate at each spanwise station with local geometry
    # 
    # NOTE: When post-buckling is enabled, skin is ALLOWED to buckle, but
    # we still compute sigma_cr for post-buckling ratio limits and other checks.
    
    # Applied stress at each station (use absolute value of skin stress)
    sigma_applied = np.abs(beam.sigma_skin)
    
    # Calculate critical buckling stress at each station using symbolic function
    n_stations = len(beam.y)
    
    # Get local panel widths (affected by stringers)
    if n_stringers > 0:
        local_panel_widths = box_width / (n_stringers + 1)
    else:
        local_panel_widths = box_width
    
    # Compute sigma_cr at each station using LOCAL panel width
    # This matches exactly what _compute_buckling_data() does in the full analysis
    sigma_cr_list = []
    for i in range(n_stations):
        local_width = float(local_panel_widths[i])
        sigma_cr_i = calculate_orthotropic_buckling_stress_symbolic(
            panel_length=rib_spacing,
            panel_width=local_width,
            thickness=t_skin,
            material=skin_material,
            grain_spanwise=True,
            boundary_condition=boundary_condition,
        )
        sigma_cr_list.append(sigma_cr_i)
    
    # Stack into array (each element depends on symbolic t_skin)
    sigma_cr_skin = np.array(sigma_cr_list)
    
    # Check if post-buckling mode is enabled
    pb_config = post_buckling_config or PostBucklingConfig()
    
    if pb_config.enabled:
        # Post-buckling mode: skin is ALLOWED to buckle
        # Instead of preventing buckling, we limit the post-buckling ratio
        # Constraint: σ_applied / σ_cr <= max_skin_buckle_ratio
        # Reformulate: σ_applied - max_ratio * σ_cr <= 0
        max_ratio = pb_config.max_skin_buckle_ratio
        post_buckling_ratio_constraint = sigma_applied - max_ratio * sigma_cr_skin
        ks_pb_ratio = ks_aggregate(post_buckling_ratio_constraint, rho=50.0, minimize=False)
        opti.subject_to(ks_pb_ratio <= 0)
        
        # Also require stringers if configured
        if pb_config.require_stringers and n_stringers == 0:
            # This configuration is infeasible - no stringers but required
            # The optimizer will naturally avoid this by mass penalty,
            # but we could add an explicit constraint. For now, we rely on
            # the post-analysis feasibility check.
            pass
    else:
        # Normal mode: skin must not buckle (margin >= FOS)
        # Buckling margin: σ_cr / σ_applied >= FOS
        # Rewrite as constraint: FOS * σ_applied - σ_cr <= 0
        skin_buckling_constraint = factor_of_safety * sigma_applied - sigma_cr_skin
        
        # Use KS aggregation to get smooth max of constraint violations
        ks_skin = ks_aggregate(skin_buckling_constraint, rho=50.0, minimize=False)
        opti.subject_to(ks_skin <= 0)
    
    # === CONSTRAINT 4: Spar web shear buckling via KS aggregation ===
    # Use the SAME calculation as _compute_buckling_data() for consistency
    # NOW INCLUDES TORSIONAL SHEAR for combined shear stress
    
    # Bending shear stress in spar webs
    # τ_bending = 1.5 * V / A_web where A_web = 2 * h * t_spar
    A_web = 2 * box_height * t_spar
    tau_bending = 1.5 * np.abs(beam.shear_force) / (A_web + 1e-10)
    
    # Torsional shear stress in spar webs (from Bredt's formula)
    # τ_torsion = T / (2 * A_enclosed * t_spar)
    # The beam computes tau_torsion_spar directly
    tau_torsion = beam.tau_torsion_spar if hasattr(beam, 'tau_torsion_spar') else np.zeros_like(tau_bending)
    
    # Combined shear stress (conservative: direct addition)
    tau_applied = tau_bending + tau_torsion
    
    # Compute tau_cr at each station using LOCAL spar height
    tau_cr_list = []
    for i in range(n_stations):
        local_h = float(box_height[i])
        tau_cr_i = calculate_spar_shear_buckling_stress_symbolic(
            spar_height=local_h,
            thickness=t_spar,
            material=spar_material,
            boundary_condition=boundary_condition,
        )
        tau_cr_list.append(tau_cr_i)
    
    # Stack into array
    tau_cr_spar = np.array(tau_cr_list)
    
    # Shear buckling constraint: FOS * τ_applied - τ_cr <= 0
    spar_buckling_constraint = factor_of_safety * tau_applied - tau_cr_spar
    
    ks_spar = ks_aggregate(spar_buckling_constraint, rho=50.0, minimize=False)
    opti.subject_to(ks_spar <= 0)
    
    # === CONSTRAINT 5 & 6: Rib buckling and crushing (if ribs present) ===
    if rib_props is not None and t_rib is not None and rib_positions is not None and len(rib_positions) > 0:
        rib_material = rib_props.material
        lightening_fraction = rib_props.lightening_hole_fraction
        spar_cap_width = rib_props.spar_cap_width_m
        rib_bc = rib_props.boundary_condition
        
        # Get shear stress coefficient for rib buckling
        k_shear = {
            "simply_supported": 5.35,
            "semi_restrained": 7.0,
            "clamped": 8.98,
        }.get(rib_bc, 5.35)
        
        E_rib = rib_material.E_1
        nu_rib = rib_material.nu_12
        # Cross-grain compression strength for crushing
        sigma_2_comp = getattr(rib_material, 'sigma_2_compression', rib_material.sigma_1_compression)
        
        # Evaluate at each rib position
        rib_shear_constraints = []
        rib_crush_constraints = []
        
        for rib_y in rib_positions:
            # Get local box height at rib
            local_h = float(np.interp(rib_y, section_y, np.array([s.box_height for s in sections])))
            
            if local_h <= 0:
                continue
            
            # Get shear force and bending moment at rib location
            V_at_rib = np.interp(rib_y, beam.y, np.abs(beam.shear_force))
            M_at_rib = np.interp(rib_y, beam.y, np.abs(beam.bending_moment))
            
            # --- Rib Shear Buckling ---
            # τ_cr = k_s * π² * E * (t/h)² / (12 * (1 - ν²))
            # Apply lightening knockdown
            stiffness_knockdown = (1.0 - lightening_fraction) ** 1.5
            if lightening_fraction < 0.3:
                stress_knockdown = 1.0 / (1.0 + 1.5 * lightening_fraction)
            elif lightening_fraction < 0.5:
                stress_knockdown = 1.0 / (1.5 + 2.0 * (lightening_fraction - 0.3))
            else:
                stress_knockdown = 1.0 / (1.9 + 3.0 * (lightening_fraction - 0.5))
            
            tau_cr_rib = (k_shear * (np.pi ** 2) * E_rib / (12 * (1 - nu_rib ** 2)) 
                          * (t_rib / local_h) ** 2 * stiffness_knockdown * stress_knockdown)
            
            # Applied shear stress in rib: τ = V / (h * t_rib)
            # Use average shear stress approximation
            tau_applied_rib = V_at_rib / (local_h * t_rib + 1e-10)
            
            # Constraint: FOS * τ_applied - τ_cr <= 0
            rib_shear_constraints.append(factor_of_safety * tau_applied_rib - tau_cr_rib)
            
            # --- Rib Crushing ---
            # Bearing stress: σ_bearing = (M / h) / (t_rib * spar_cap_width)
            P_cap = M_at_rib / (local_h + 1e-10)
            sigma_bearing = P_cap / (t_rib * spar_cap_width + 1e-10)
            
            # Constraint: FOS * σ_bearing - σ_allowable <= 0
            rib_crush_constraints.append(factor_of_safety * sigma_bearing - sigma_2_comp)
        
        # Apply rib constraints using KS aggregation
        if rib_shear_constraints:
            ks_rib_shear = ks_aggregate(np.array(rib_shear_constraints), rho=50.0, minimize=False)
            opti.subject_to(ks_rib_shear <= 0)
        
        if rib_crush_constraints:
            ks_rib_crush = ks_aggregate(np.array(rib_crush_constraints), rho=50.0, minimize=False)
            opti.subject_to(ks_rib_crush <= 0)
    
    # === CONSTRAINT 7: Skin shear buckling from torsion (Priority 3) ===
    # Skin panels experience torsional shear stress that can cause shear buckling
    tau_torsion_skin = beam.tau_torsion_skin if hasattr(beam, 'tau_torsion_skin') else np.zeros_like(beam.y)
    tau_applied_skin = np.abs(tau_torsion_skin)
    
    # Compute tau_cr for skin shear buckling at each station
    tau_cr_skin_list = []
    for i in range(n_stations):
        local_width = float(local_panel_widths[i])
        tau_cr_skin_i = calculate_skin_shear_buckling_stress_symbolic(
            panel_length=rib_spacing,
            panel_width=local_width,
            thickness=t_skin,
            material=skin_material,
            grain_spanwise=True,
            boundary_condition=boundary_condition,
        )
        tau_cr_skin_list.append(tau_cr_skin_i)
    
    tau_cr_skin = np.array(tau_cr_skin_list)
    
    # Skin shear buckling constraint: FOS * τ_applied - τ_cr <= 0
    # Note: Shear buckling constraint applies regardless of post-buckling mode
    # (tension field handles post-buckled shear, but we still need to limit τ/τ_cr)
    skin_shear_constraint = factor_of_safety * tau_applied_skin - tau_cr_skin
    ks_skin_shear = ks_aggregate(skin_shear_constraint, rho=50.0, minimize=False)
    opti.subject_to(ks_skin_shear <= 0)
    
    # === CONSTRAINT 8: Combined biaxial buckling (Priority 5) ===
    # Skin panels under combined compression and shear have reduced buckling strength
    # Use interaction equation: (σ/σ_cr)² + (τ/τ_cr)² <= 1
    # Reformulate as: FOS² * [(σ/σ_cr)² + (τ/τ_cr)²] <= 1
    
    # Stress ratios at each station
    R_sigma = sigma_applied / (sigma_cr_skin + 1e-10)
    R_tau = tau_applied_skin / (tau_cr_skin + 1e-10)
    
    if pb_config.enabled:
        # Post-buckling mode: compression can exceed σ_cr (up to max ratio)
        # But shear interaction still applies
        # Use modified interaction: clamp R_sigma to max_ratio for interaction
        # If σ > σ_cr, we're in post-buckling - interaction is less relevant
        # For simplicity in optimization, we skip the combined constraint when post-buckling
        # is enabled (the post-analysis will still check all constraints)
        pass
    else:
        # Normal mode: full biaxial interaction constraint
        # Interaction sum (parabolic)
        interaction_sum = R_sigma**2 + R_tau**2
        
        # Constraint: FOS² * interaction_sum <= 1 (margin = 1/sqrt(interaction) >= FOS)
        # Reformulate: factor_of_safety² * interaction_sum - 1 <= 0
        combined_constraint = factor_of_safety**2 * interaction_sum - 1.0
        ks_combined = ks_aggregate(combined_constraint, rho=50.0, minimize=False)
        opti.subject_to(ks_combined <= 0)
    
    # === CONSTRAINT 9: Stringer crippling (Priority 6) ===
    # Stringers can fail by local crippling before global buckling
    if n_stringers > 0 and stringer_props is not None and stringer_props.material is not None:
        stringer_material = stringer_props.material
        
        # Calculate stringer crippling stress (doesn't depend on symbolic variables)
        b_over_t = stringer_props.crippling_b_over_t
        sigma_crippling = calculate_stringer_crippling_stress(
            b_over_t=b_over_t,
            material=stringer_material,
            section_type=stringer_props.section_type,
            edge_condition="one_edge_free",
        )
        
        # Column buckling stress (depends on rib_spacing but not symbolic vars)
        sigma_euler = calculate_stringer_column_buckling_stress(
            stringer_length=rib_spacing,
            stringer_props=stringer_props,
            end_fixity=2.0,  # Semi-fixed
        )
        
        # Stringer allowable = min(crippling, column buckling, yield)
        sigma_yield = stringer_material.sigma_1_compression
        sigma_allowable = min(sigma_crippling, sigma_euler, sigma_yield)
        
        # Calculate stringer stress at each station
        # Stringer stress scales with skin stress (both at same y-location from NA)
        # σ_stringer ≈ σ_skin * E_stringer / E_skin
        E_ratio = stringer_material.E_1 / (skin_material.E_1 + 1e-10)
        sigma_stringer = np.abs(beam.sigma_skin) * E_ratio
        
        # Stringer crippling constraint: FOS * σ_applied - σ_allowable <= 0
        stringer_constraint = factor_of_safety * sigma_stringer - sigma_allowable
        ks_stringer = ks_aggregate(stringer_constraint, rho=50.0, minimize=False)
        opti.subject_to(ks_stringer <= 0)
    
    # === SOLVE ===
    try:
        sol = opti.solve(verbose=verbose)  # Pass verbose through
        
        # Extract optimized values
        t_spar_opt = float(sol.value(t_spar))
        t_skin_opt = float(sol.value(t_skin))
        t_rib_opt = float(sol.value(t_rib)) if t_rib is not None else rib_thickness_init / 1000
        
        if verbose:
            print(f"  Solved: spar={t_spar_opt*1000:.2f}mm, skin={t_skin_opt*1000:.2f}mm, rib={t_rib_opt*1000:.2f}mm")
        
        # Create updated rib_props with optimized thickness
        optimized_rib_props = None
        if rib_props is not None:
            optimized_rib_props = RibProperties(
                thickness_m=t_rib_opt,
                material=rib_props.material,
                lightening_hole_fraction=rib_props.lightening_hole_fraction,
                spar_cap_width_m=rib_props.spar_cap_width_m,
                boundary_condition=rib_props.boundary_condition,
            )
        
        # Run full analysis with optimized thicknesses
        optimized_result = analyze_wingbox_beam(
            sections=sections,
            spar_thickness=t_spar_opt,
            skin_thickness=t_skin_opt,
            spar_material=spar_material,
            skin_material=skin_material,
            lift_distribution=lift_distribution,
            rib_positions=rib_positions,
            factor_of_safety=factor_of_safety,
            max_deflection_fraction=max_deflection_fraction,
            max_twist_deg=max_twist_deg,
            stringer_props=stringer_props,
            rib_props=optimized_rib_props,
            boundary_condition=boundary_condition,
            include_curvature=include_curvature,
            airfoils=airfoils,
            control_surface_props=control_surface_props,
            fastener_adhesive_fraction=fastener_adhesive_fraction,
            post_buckling_config=post_buckling_config,
        )
        
        return OptimizationResult(
            success=True,
            message="Optimization converged",
            spar_thickness_mm=t_spar_opt * 1000,
            skin_thickness_mm=t_skin_opt * 1000,
            rib_thickness_mm=t_rib_opt * 1000,
            stringer_count=n_stringers,
            initial_mass_kg=0.0,  # Will be set by caller
            optimized_mass_kg=optimized_result.mass_kg,
            mass_reduction_percent=0.0,  # Will be set by caller
            stress_margin=optimized_result.stress_margin,
            skin_buckling_margin=optimized_result.min_buckling_margin,
            spar_buckling_margin=optimized_result.min_spar_buckling_margin or float('inf'),
            rib_buckling_margin=optimized_result.min_rib_buckling_margin or float('inf'),
            rib_crushing_margin=optimized_result.min_rib_crushing_margin or float('inf'),
            skin_shear_buckling_margin=optimized_result.min_skin_shear_buckling_margin or float('inf'),
            combined_buckling_margin=optimized_result.min_combined_buckling_margin or float('inf'),
            stringer_crippling_margin=optimized_result.min_stringer_crippling_margin or float('inf'),
            stringer_failure_mode=optimized_result.stringer_failure_mode,
            tip_deflection_percent=abs(optimized_result.tip_deflection_m) / half_span * 100,
            tip_twist_deg=optimized_result.tip_twist_deg or 0.0,
            twist_margin=optimized_result.twist_margin or float('inf'),
            iterations=1,
            structural_result=optimized_result,
        )
        
    except Exception as e:
        return OptimizationResult(
            success=False,
            message=str(e),
            spar_thickness_mm=spar_thickness_init,
            skin_thickness_mm=skin_thickness_init,
            rib_thickness_mm=rib_thickness_init,
            stringer_count=n_stringers,
            initial_mass_kg=0.0,
            optimized_mass_kg=float('inf'),
            mass_reduction_percent=0.0,
            stress_margin=0.0,
            skin_buckling_margin=0.0,
            spar_buckling_margin=0.0,
            rib_buckling_margin=0.0,
            rib_crushing_margin=0.0,
            skin_shear_buckling_margin=0.0,
            combined_buckling_margin=0.0,
            stringer_crippling_margin=0.0,
            stringer_failure_mode=None,
            tip_deflection_percent=100.0,
            tip_twist_deg=0.0,
            twist_margin=0.0,
            iterations=0,
            structural_result=None,
        )
