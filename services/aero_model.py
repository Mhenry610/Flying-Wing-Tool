"""
Aerodynamic Model Factory for 6-DOF Dynamics Integration

Creates callable aero models that wrap AeroSandbox AeroBuildup analysis,
accepting control surface deflections and returning aerodynamic coefficients
suitable for FlyingWingDynamics6DOF.

This module bridges:
- Planform control surface definitions (core/models/planform.py)
- AeroSandbox geometry with control surfaces (services/geometry.py)
- 6-DOF dynamics aero_model interface (services/dynamics/aircraft_dynamics.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Any, Tuple, List
import numpy as np
import aerosandbox as asb
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
import time

if TYPE_CHECKING:
    from services.geometry import AeroSandboxService

# Import post-stall model parameters from stability_derivatives_ad
try:
    from scripts.stability_derivatives_ad import PostStallParams, ADStabilityAnalyzerWithPostStall
    HAS_AD_STABILITY = True
except ImportError:
    # Try adding project root to path
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from scripts.stability_derivatives_ad import PostStallParams, ADStabilityAnalyzerWithPostStall
        HAS_AD_STABILITY = True
    except ImportError as e:
        print(f"Warning: Could not import AD stability module: {e}")
        HAS_AD_STABILITY = False
        PostStallParams = None
        ADStabilityAnalyzerWithPostStall = None


@dataclass
class AeroPolarConfig:
    """Configuration for pre-computed aerodynamic polar tables."""
    alpha_min: float = -5.0      # Min angle of attack [deg]
    alpha_max: float = 15.0      # Max angle of attack [deg]
    alpha_steps: int = 21        # Number of alpha points (1° resolution)
    
    delta_e_min: float = -20.0   # Min elevator deflection [deg]
    delta_e_max: float = 10.0    # Max elevator deflection [deg]
    delta_e_steps: int = 7       # Number of elevator points (5° resolution)
    
    delta_a_min: float = -15.0   # Min aileron deflection [deg]
    delta_a_max: float = 15.0    # Max aileron deflection [deg]
    delta_a_steps: int = 7       # Number of aileron points (5° resolution)
    
    beta_nominal: float = 0.0    # Nominal sideslip for polar [deg]
    airspeed_nominal: float = 20.0  # Nominal airspeed for polar [m/s]
    
    include_aileron: bool = True  # Build 3D grid with aileron dimension
    
    def get_alpha_grid(self) -> np.ndarray:
        return np.linspace(self.alpha_min, self.alpha_max, self.alpha_steps)
    
    def get_delta_e_grid(self) -> np.ndarray:
        return np.linspace(self.delta_e_min, self.delta_e_max, self.delta_e_steps)
    
    def get_delta_a_grid(self) -> np.ndarray:
        return np.linspace(self.delta_a_min, self.delta_a_max, self.delta_a_steps)


class PrecomputedPolarAeroModel:
    """
    Aerodynamic model using pre-computed polar lookup tables.
    
    Computes CL, CD, CY, Cl, Cm, Cn over a grid of (alpha, delta_e, delta_a)
    at initialization using AeroBuildup, then uses fast interpolation during
    simulation.
    
    This provides:
    - Smooth curves (interpolated, no stair-steps from caching)
    - Fast simulation (~1μs interpolation vs ~200ms AeroBuildup per call)
    - Physically correct (same AeroBuildup data, just pre-computed)
    
    Trade-off: ~30s one-time cost at startup for a typical grid.
    
    For rate derivatives (Cmq, Clp, Cnr), uses linear approximations since
    these are expensive to tabulate and typically vary slowly with alpha.
    """
    
    def __init__(
        self,
        geometry_service: 'AeroSandboxService',
        config: Optional[AeroPolarConfig] = None,
        atmosphere: Optional[asb.Atmosphere] = None,
        xyz_ref: Optional[list] = None,
        Cmq: float = -15.0,   # Pitch rate damping (typical flying wing)
        Clp: float = -0.4,    # Roll rate damping
        Cnr: float = -0.1,    # Yaw rate damping
        verbose: bool = True,
    ):
        """
        Build pre-computed polar tables.
        
        Args:
            geometry_service: AeroSandboxService with wing geometry
            config: Polar grid configuration (default: 21x7 alpha x delta_e)
            atmosphere: AeroSandbox Atmosphere (default: sea level ISA)
            xyz_ref: Reference point for moments [x, y, z]
            Cmq: Pitch rate damping derivative (default: -15, typical for flying wings)
            Clp: Roll rate damping derivative
            Cnr: Yaw rate damping derivative
            verbose: Print progress during polar computation
        """
        self.geometry_service = geometry_service
        self.config = config or AeroPolarConfig()
        self.atmosphere = atmosphere or asb.Atmosphere(altitude=0)
        
        # Cache reference geometry
        self._cached_wing = geometry_service.build_wing()
        self._xyz_ref = xyz_ref
        if self._xyz_ref is None:
            ac = self._cached_wing.aerodynamic_center()
            self._xyz_ref = [ac[0], 0.0, 0.0]
        
        # Rate damping derivatives (used for body-rate contributions)
        self.Cmq = Cmq
        self.Clp = Clp
        self.Cnr = Cnr
        
        # Reference geometry for non-dimensionalization
        planform = geometry_service.wing_project.planform
        self.reference_chord = planform.mean_aerodynamic_chord
        self.reference_span = planform.span
        
        # Build the polar tables
        self._build_polars(verbose)
    
    def _build_polars(self, verbose: bool = True):
        """Pre-compute aerodynamic coefficient tables over the grid."""
        alpha_grid = self.config.get_alpha_grid()
        delta_e_grid = self.config.get_delta_e_grid()
        
        if self.config.include_aileron:
            delta_a_grid = self.config.get_delta_a_grid()
            shape = (len(alpha_grid), len(delta_e_grid), len(delta_a_grid))
            n_total = np.prod(shape)
        else:
            delta_a_grid = np.array([0.0])  # Single point
            shape = (len(alpha_grid), len(delta_e_grid))
            n_total = np.prod(shape)
        
        # Initialize coefficient arrays
        CL_data = np.zeros(shape)
        CD_data = np.zeros(shape)
        CY_data = np.zeros(shape)
        Cl_data = np.zeros(shape)  # Roll moment
        Cm_data = np.zeros(shape)  # Pitch moment
        Cn_data = np.zeros(shape)  # Yaw moment
        
        if verbose:
            print(f"Building aerodynamic polar: {shape} grid ({n_total} evaluations)...")
        
        # Compute polars
        count = 0
        for i, alpha in enumerate(alpha_grid):
            for j, delta_e in enumerate(delta_e_grid):
                for k, delta_a in enumerate(delta_a_grid):
                    try:
                        result = self._compute_single_point(
                            alpha=alpha,
                            delta_e=delta_e,
                            delta_a=delta_a,
                        )
                        
                        if self.config.include_aileron:
                            CL_data[i, j, k] = result['CL']
                            CD_data[i, j, k] = result['CD']
                            CY_data[i, j, k] = result['CY']
                            Cl_data[i, j, k] = result['Cl']
                            Cm_data[i, j, k] = result['Cm']
                            Cn_data[i, j, k] = result['Cn']
                        else:
                            CL_data[i, j] = result['CL']
                            CD_data[i, j] = result['CD']
                            CY_data[i, j] = result['CY']
                            Cl_data[i, j] = result['Cl']
                            Cm_data[i, j] = result['Cm']
                            Cn_data[i, j] = result['Cn']
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: failed at α={alpha:.1f}°, δe={delta_e:.1f}°, δa={delta_a:.1f}°: {e}")
                        # Use NaN, interpolator will handle it
                        if self.config.include_aileron:
                            CL_data[i, j, k] = np.nan
                        else:
                            CL_data[i, j] = np.nan
                    
                    count += 1
                    if verbose and count % 20 == 0:
                        print(f"  {count}/{n_total} points computed...")
        
        if verbose:
            print(f"Polar computation complete. Building interpolators...")
        
        # Build interpolators
        if self.config.include_aileron:
            grid_points = (alpha_grid, delta_e_grid, delta_a_grid)
        else:
            grid_points = (alpha_grid, delta_e_grid)
        
        self._interp_CL = RegularGridInterpolator(grid_points, CL_data, bounds_error=False, fill_value=None)
        self._interp_CD = RegularGridInterpolator(grid_points, CD_data, bounds_error=False, fill_value=None)
        self._interp_CY = RegularGridInterpolator(grid_points, CY_data, bounds_error=False, fill_value=None)
        self._interp_Cl = RegularGridInterpolator(grid_points, Cl_data, bounds_error=False, fill_value=None)
        self._interp_Cm = RegularGridInterpolator(grid_points, Cm_data, bounds_error=False, fill_value=None)
        self._interp_Cn = RegularGridInterpolator(grid_points, Cn_data, bounds_error=False, fill_value=None)
        
        # Store grids for reference
        self._alpha_grid = alpha_grid
        self._delta_e_grid = delta_e_grid
        self._delta_a_grid = delta_a_grid
        
        if verbose:
            print("Aerodynamic polar ready.")
    
    def _compute_single_point(
        self,
        alpha: float,
        delta_e: float,
        delta_a: float = 0.0,
    ) -> Dict[str, float]:
        """Compute aero coefficients at a single (alpha, delta_e, delta_a) point using AeroBuildup."""
        # Map control inputs to surface deflections
        control_deflections = {}
        planform = self.geometry_service.wing_project.planform
        
        for cs in planform.control_surfaces:
            surface_type = cs.surface_type.lower()
            
            if surface_type == 'elevon':
                control_deflections[f'{cs.name}_pitch'] = delta_e
                control_deflections[f'{cs.name}_roll'] = delta_a
            elif surface_type in ['elevator', 'flap']:
                control_deflections[cs.name] = delta_e
            elif surface_type == 'aileron':
                control_deflections[cs.name] = delta_a
        
        # Build wing with control deflections
        wing = self.geometry_service.build_wing(control_deflections=control_deflections)
        
        # Create airplane
        airplane = asb.Airplane(
            name="FlyingWing",
            wings=[wing],
            xyz_ref=self._xyz_ref,
        )
        
        # Create operating point (zero rates for polar)
        op_point = asb.OperatingPoint(
            atmosphere=self.atmosphere,
            velocity=self.config.airspeed_nominal,
            alpha=alpha,
            beta=self.config.beta_nominal,
            p=0.0, q=0.0, r=0.0,
        )
        
        # Run AeroBuildup
        aero = asb.AeroBuildup(airplane=airplane, op_point=op_point)
        result = aero.run()
        
        # Extract coefficients
        def get_coef(name: str, fallback: str = None) -> float:
            val = result.get(name, result.get(fallback, 0.0) if fallback else 0.0)
            if hasattr(val, 'item'):
                return float(val.item())
            return float(val)
        
        return {
            'CL': get_coef('CL', 'Cl'),
            'CD': get_coef('CD', 'Cd'),
            'CY': get_coef('CY', 'Cy'),
            'Cl': get_coef('Cl'),
            'Cm': get_coef('Cm'),
            'Cn': get_coef('Cn'),
        }
    
    def __call__(
        self,
        alpha: float,       # Angle of attack [deg]
        beta: float,        # Sideslip angle [deg]
        airspeed: float,    # True airspeed [m/s]
        p: float,           # Roll rate [rad/s]
        q: float,           # Pitch rate [rad/s]
        r: float,           # Yaw rate [rad/s]
        elevator: float = 0.0,  # Elevator deflection [deg]
        aileron: float = 0.0,   # Aileron deflection [deg]
        rudder: float = 0.0,    # Rudder deflection [deg] (ignored for now)
    ) -> Dict[str, float]:
        """
        Compute aerodynamic coefficients using interpolated polars.
        
        Interpolates base coefficients from pre-computed tables, then adds
        rate-dependent terms using linear damping derivatives.
        
        Args:
            alpha: Angle of attack [deg]
            beta: Sideslip angle [deg] (used for side force estimate)
            airspeed: True airspeed [m/s]
            p, q, r: Body angular rates [rad/s]
            elevator: Elevator/elevon pitch deflection [deg]
            aileron: Aileron/elevon roll deflection [deg]
            rudder: Rudder deflection [deg] (currently ignored)
        
        Returns:
            Dict with CL, CD, CY, Cl, Cm, Cn
        """
        V = max(airspeed, 0.1)  # Avoid division by zero
        c = self.reference_chord
        b = self.reference_span
        
        # Interpolate base coefficients from polar tables
        if self.config.include_aileron:
            point = np.array([[alpha, elevator, aileron]])
        else:
            point = np.array([[alpha, elevator]])
        
        CL_base = float(self._interp_CL(point)[0])
        CD_base = float(self._interp_CD(point)[0])
        CY_base = float(self._interp_CY(point)[0])
        Cl_base = float(self._interp_Cl(point)[0])
        Cm_base = float(self._interp_Cm(point)[0])
        Cn_base = float(self._interp_Cn(point)[0])
        
        # Non-dimensional rates
        p_hat = p * b / (2 * V)   # Non-dimensional roll rate
        q_hat = q * c / (2 * V)   # Non-dimensional pitch rate
        r_hat = r * b / (2 * V)   # Non-dimensional yaw rate
        
        # Add rate damping contributions
        Cm_total = Cm_base + self.Cmq * q_hat
        Cl_total = Cl_base + self.Clp * p_hat
        Cn_total = Cn_base + self.Cnr * r_hat
        
        # Simple beta effect on side force
        beta_rad = np.radians(beta)
        CY_total = CY_base - 0.3 * beta_rad  # Typical sideslip derivative
        
        return {
            'CL': CL_base,
            'CD': CD_base,
            'CY': CY_total,
            'Cl': Cl_total,
            'Cm': Cm_total,
            'Cn': Cn_total,
        }
    
    def get_interpolators(self) -> Dict[str, RegularGridInterpolator]:
        """Return the interpolator objects for direct access."""
        return {
            'CL': self._interp_CL,
            'CD': self._interp_CD,
            'CY': self._interp_CY,
            'Cl': self._interp_Cl,
            'Cm': self._interp_Cm,
            'Cn': self._interp_Cn,
        }
    
    def get_grids(self) -> Dict[str, np.ndarray]:
        """Return the grid arrays used for interpolation."""
        result = {
            'alpha': self._alpha_grid,
            'delta_e': self._delta_e_grid,
        }
        if self.config.include_aileron:
            result['delta_a'] = self._delta_a_grid
        return result


def create_precomputed_aero_model(
    geometry_service: 'AeroSandboxService',
    config: Optional[AeroPolarConfig] = None,
    atmosphere: Optional[asb.Atmosphere] = None,
    xyz_ref: Optional[list] = None,
    Cmq: float = -15.0,
    Clp: float = -0.4,
    Cnr: float = -0.1,
    verbose: bool = True,
) -> PrecomputedPolarAeroModel:
    """
    Factory function to create a pre-computed polar aero model.
    
    This is the recommended approach for simulation when you need:
    - Fast execution (many timesteps)
    - Smooth coefficient curves (no cache stair-stepping)
    - Physical accuracy (same AeroBuildup data, pre-computed)
    
    Trade-off: ~30s startup cost for default grid (21x7 = 147 AeroBuildup calls).
    
    Args:
        geometry_service: AeroSandboxService with wing geometry
        config: Grid configuration (see AeroPolarConfig)
        atmosphere: Atmosphere model
        xyz_ref: Moment reference point
        Cmq, Clp, Cnr: Rate damping derivatives
        verbose: Print progress
    
    Returns:
        PrecomputedPolarAeroModel instance (callable)
    
    Example:
        >>> geo = AeroSandboxService(wing_project)
        >>> aero = create_precomputed_aero_model(geo)  # ~30s polar build
        >>> # Now fast calls:
        >>> result = aero(alpha=5, beta=0, airspeed=20, p=0, q=0, r=0, elevator=-2)
    """
    return PrecomputedPolarAeroModel(
        geometry_service=geometry_service,
        config=config,
        atmosphere=atmosphere,
        xyz_ref=xyz_ref,
        Cmq=Cmq,
        Clp=Clp,
        Cnr=Cnr,
        verbose=verbose,
    )


# =============================================================================
# AD-Based Aero Model (uses ADStabilityAnalyzerWithPostStall from stability_derivatives_ad.py)
# =============================================================================

class ADStabilityAeroModel:
    """
    Aerodynamic model using CasADi AD-based stability analyzer with post-stall model.
    
    This wraps ADStabilityAnalyzerWithPostStall from stability_derivatives_ad.py,
    providing the mission simulator aero_model interface.
    
    Features:
    - CasADi AD for exact derivatives
    - Post-stall blending with flat-plate separated flow model
    - Control effectiveness fadeout beyond stall
    - ~μs evaluation time after one-time build (~5-10s)
    
    This is the recommended aero model for mission simulation when you need:
    - Accurate post-stall behavior
    - Control surface effects with proper fadeout
    - Physical pitch/climb limits from actual aerodynamics
    """
    
    def __init__(
        self,
        geometry_service: 'AeroSandboxService',
        x_cg: Optional[float] = None,
        airspeed: float = 20.0,
        alpha_stall_deg: float = 16.0,
        blend_width_deg: float = 4.0,
        eta_min: float = 0.3,
        verbose: bool = True,
    ):
        """
        Build AD-based aero model with post-stall blending.
        
        Args:
            geometry_service: AeroSandboxService with wing geometry
            x_cg: CG x-position [m]. If None, uses aerodynamic center.
            airspeed: Reference airspeed for building AD model [m/s]
            alpha_stall_deg: Stall angle of attack [deg]
            blend_width_deg: Width of stall transition [deg]
            eta_min: Minimum control effectiveness in deep stall [0-1]
            verbose: Print build progress
        """
        if not HAS_AD_STABILITY:
            raise ImportError(
                "ADStabilityAnalyzerWithPostStall not available. "
                "Ensure scripts/stability_derivatives_ad.py exists and CasADi is installed."
            )
        
        self.geometry_service = geometry_service
        self.verbose = verbose
        
        # Get reference geometry
        wing = geometry_service.build_wing()
        planform = geometry_service.wing_project.planform
        self.reference_chord = planform.mean_aerodynamic_chord
        self.reference_span = planform.span
        
        # CG position
        if x_cg is None:
            ac = wing.aerodynamic_center()
            x_cg = ac[0]
        self.x_cg = x_cg
        
        # Post-stall parameters
        self.post_stall_params = PostStallParams(
            alpha_stall_deg=alpha_stall_deg,
            blend_width_deg=blend_width_deg,
            eta_min=eta_min,
        )
        
        # Build the AD analyzer
        if verbose:
            print(f"Building AD-based aero model with post-stall blending...")
            print(f"  Stall: α_stall={alpha_stall_deg}°, blend_width={blend_width_deg}°")
            print(f"  Control fadeout: η_min={eta_min}")
        
        start = time.time()
        self._analyzer = ADStabilityAnalyzerWithPostStall(
            service=geometry_service,
            x_cg=x_cg,
            airspeed=airspeed,
            post_stall_params=self.post_stall_params,
        )
        elapsed = time.time() - start
        
        if verbose:
            print(f"AD aero model built in {elapsed:.1f}s")
            print(f"  Control surfaces: {self._analyzer.control_surface_names}")
    
    def __call__(
        self,
        alpha: float,       # Angle of attack [deg]
        beta: float,        # Sideslip angle [deg]
        airspeed: float,    # True airspeed [m/s]
        p: float,           # Roll rate [rad/s]
        q: float,           # Pitch rate [rad/s]
        r: float,           # Yaw rate [rad/s]
        elevator: float = 0.0,  # Elevator deflection [deg]
        aileron: float = 0.0,   # Aileron deflection [deg]
        rudder: float = 0.0,    # Rudder deflection [deg] (if present)
    ) -> Dict[str, float]:
        """
        Compute aerodynamic coefficients using AD-based post-stall model.
        
        Returns dict with: CL, CD, CY, Cl, Cm, Cn
        Plus diagnostics: w_blend, eta_ctrl
        """
        # Map elevator/aileron to control surface names
        control_deflections_deg = {}
        for name in self._analyzer.control_surface_names:
            if '_pitch' in name or name.lower() in ('elevator', 'flap'):
                control_deflections_deg[name] = elevator
            elif '_roll' in name or name.lower() == 'aileron':
                control_deflections_deg[name] = aileron
            elif name.lower() == 'rudder':
                control_deflections_deg[name] = rudder
        
        # Evaluate the AD model
        result = self._analyzer.evaluate(
            alpha_deg=alpha,
            beta_deg=beta,
            p=p,
            q=q,
            r=r,
            control_deflections_deg=control_deflections_deg,
        )
        
        # Return in standard format
        return {
            'CL': result['CL'],
            'CD': result['CD'],
            'CY': result['CY'],
            'Cl': result['Cl'],
            'Cm': result['Cm'],
            'Cn': result['Cn'],
            'w_blend': result.get('w_blend', 0.0),
            'eta_ctrl': result.get('eta_ctrl', 1.0),
        }
    
    @property
    def analyzer(self) -> 'ADStabilityAnalyzerWithPostStall':
        """Access underlying AD analyzer for derivatives."""
        return self._analyzer
    
    @property 
    def control_surface_names(self) -> List[str]:
        """List of control surface names."""
        return self._analyzer.control_surface_names


def create_ad_aero_model(
    geometry_service: 'AeroSandboxService',
    x_cg: Optional[float] = None,
    airspeed: float = 20.0,
    alpha_stall_deg: float = 16.0,
    blend_width_deg: float = 4.0,
    eta_min: float = 0.3,
    verbose: bool = True,
) -> ADStabilityAeroModel:
    """
    Factory function to create AD-based aero model with post-stall blending.
    
    This is the RECOMMENDED aero model for mission simulation. It uses:
    - CasADi automatic differentiation for exact derivatives
    - Post-stall blending (attached → separated flow)
    - Control effectiveness fadeout beyond stall
    
    Build time: ~5-10s (one-time)
    Evaluation time: ~μs per call
    
    Args:
        geometry_service: AeroSandboxService with wing geometry
        x_cg: CG x-position [m]. If None, uses aerodynamic center.
        airspeed: Reference airspeed [m/s]
        alpha_stall_deg: Stall angle [deg]
        blend_width_deg: Stall transition width [deg]
        eta_min: Minimum control effectiveness in stall
        verbose: Print progress
    
    Returns:
        ADStabilityAeroModel (callable)
    
    Example:
        >>> geo = AeroSandboxService(wing_project)
        >>> aero = create_ad_aero_model(geo, alpha_stall_deg=14)
        >>> result = aero(alpha=18, beta=0, airspeed=15, p=0, q=0, r=0, elevator=-5)
        >>> print(f"Post-stall CL: {result['CL']:.2f}, blend: {result['w_blend']:.2f}")
    """
    if not HAS_AD_STABILITY:
        raise ImportError(
            "AD stability module not available. Install CasADi: pip install casadi"
        )
    
    return ADStabilityAeroModel(
        geometry_service=geometry_service,
        x_cg=x_cg,
        airspeed=airspeed,
        alpha_stall_deg=alpha_stall_deg,
        blend_width_deg=blend_width_deg,
        eta_min=eta_min,
        verbose=verbose,
    )


def create_dynamics_aero_model(
    geometry_service: 'AeroSandboxService',
    atmosphere: Optional[asb.Atmosphere] = None,
    xyz_ref: Optional[list] = None,
) -> Callable:
    """
    Create an aerodynamic model callable for 6-DOF dynamics.
    
    Returns a function compatible with FlyingWingDynamics6DOF.aero_model.
    
    Args:
        geometry_service: AeroSandboxService with wing geometry
        atmosphere: AeroSandbox Atmosphere (default: sea level ISA)
        xyz_ref: Reference point for moments [x, y, z] in meters
                 If None, uses wing's aerodynamic center
    
    Returns:
        Callable with signature:
            aero_model(alpha, beta, airspeed, p, q, r, 
                      elevator, aileron, rudder) -> Dict[str, float]
        
        Returns dict with: CL, CD, CY, Cl, Cm, Cn
    
    Example:
        >>> geo = AeroSandboxService(wing_project)
        >>> aero_model = create_dynamics_aero_model(geo)
        >>> result = aero_model(alpha=5, beta=0, airspeed=20, 
        ...                     p=0, q=0, r=0, elevator=-2, aileron=3)
        >>> print(result['Cm'])  # Pitch moment coefficient
    """
    if atmosphere is None:
        atmosphere = asb.Atmosphere(altitude=0)
    
    # Cache the base wing for reference calculations
    _cached_wing = geometry_service.build_wing()
    _default_xyz_ref = xyz_ref
    if _default_xyz_ref is None:
        # Use aerodynamic center as default reference
        ac = _cached_wing.aerodynamic_center()
        _default_xyz_ref = [ac[0], 0.0, 0.0]
    
    def aero_model(
        alpha: float,       # Angle of attack [deg]
        beta: float,        # Sideslip angle [deg]
        airspeed: float,    # True airspeed [m/s]
        p: float,           # Roll rate [rad/s]
        q: float,           # Pitch rate [rad/s]
        r: float,           # Yaw rate [rad/s]
        elevator: float = 0.0,  # Elevator deflection [deg] (pitch control)
        aileron: float = 0.0,   # Aileron deflection [deg] (roll control)
        rudder: float = 0.0,    # Rudder deflection [deg] (yaw control, if present)
    ) -> Dict[str, float]:
        """
        Compute aerodynamic coefficients at given flight condition.
        
        For flying wings with elevons, the control mapping is:
        - 'elevator' controls symmetric deflection (pitch) on Elevon surfaces
        - 'aileron' controls anti-symmetric deflection (roll) on Elevon surfaces
        
        Implementation:
        - Each Elevon in the planform creates TWO internal control surfaces:
          - {name}_pitch: symmetric=True, receives elevator input
          - {name}_roll: symmetric=False, receives aileron input
        - AeroSandbox sums deflections from both, achieving proper mixing:
          - Right elevon = elevator + aileron
          - Left elevon = elevator - aileron
        
        Other surface types:
        - 'Flap'/'Elevator': symmetric only, responds to elevator
        - 'Aileron': anti-symmetric only, responds to aileron
        - 'Rudder': responds to rudder input
        
        Returns:
            Dict with keys: CL, CD, CY, Cl, Cm, Cn
            (lift, drag, side force, roll/pitch/yaw moment coefficients)
        """
        # Map control inputs to surface deflections
        # For a flying wing with elevons:
        # - Elevon surfaces create TWO control surfaces each:
        #   - {name}_pitch (symmetric) responds to elevator input
        #   - {name}_roll (anti-symmetric) responds to aileron input
        # - Pure Aileron surfaces respond only to aileron
        # - Pure Elevator/Flap surfaces respond only to elevator
        control_deflections = {}
        
        planform = geometry_service.wing_project.planform
        for cs in planform.control_surfaces:
            surface_type = cs.surface_type.lower()
            
            if surface_type == 'elevon':
                # Elevons respond to BOTH elevator (pitch) and aileron (roll)
                # geometry.py creates {name}_pitch and {name}_roll surfaces
                control_deflections[f'{cs.name}_pitch'] = elevator
                control_deflections[f'{cs.name}_roll'] = aileron
            elif surface_type in ['elevator', 'flap']:
                # Symmetric surfaces respond only to elevator
                control_deflections[cs.name] = elevator
            elif surface_type == 'aileron':
                # Anti-symmetric surfaces respond only to aileron
                control_deflections[cs.name] = aileron
            elif surface_type == 'rudder':
                control_deflections[cs.name] = rudder
        
        # Build wing with current control deflections
        wing = geometry_service.build_wing(control_deflections=control_deflections)
        
        # Create airplane
        airplane = asb.Airplane(
            name="FlyingWing",
            wings=[wing],
            xyz_ref=_default_xyz_ref,
        )
        
        # Create operating point
        op_point = asb.OperatingPoint(
            atmosphere=atmosphere,
            velocity=max(airspeed, 0.1),  # Avoid zero velocity
            alpha=alpha,
            beta=beta,
            p=p,
            q=q,
            r=r,
        )
        
        # Run AeroBuildup analysis
        aero = asb.AeroBuildup(airplane=airplane, op_point=op_point)
        result = aero.run()
        
        # Extract coefficients (handle both dict and object return)
        def get_coef(name: str, fallback: str = None) -> float:
            val = result.get(name, result.get(fallback, 0.0) if fallback else 0.0)
            # Handle array/scalar
            if hasattr(val, 'item'):
                return float(val.item())
            return float(val)
        
        return {
            'CL': get_coef('CL', 'Cl'),
            'CD': get_coef('CD', 'Cd'),
            'CY': get_coef('CY', 'Cy'),
            'Cl': get_coef('Cl'),  # Roll moment (note: same name as lift in some conventions)
            'Cm': get_coef('Cm'),  # Pitch moment
            'Cn': get_coef('Cn'),  # Yaw moment
        }
    
    return aero_model


def create_simple_flying_wing_aero_model(
    CL0: float = 0.3,
    CLa: float = 5.5,      # per radian
    CD0: float = 0.02,
    K: float = 0.04,       # Induced drag factor (CD = CD0 + K*CL^2)
    Cma: float = -0.8,     # Pitch stability (per radian, negative = stable)
    Cmq: float = -12.0,    # Pitch damping
    Cmde: float = -0.02,   # Elevator effectiveness (per deg)
    Clp: float = -0.4,     # Roll damping
    Clda: float = 0.002,   # Aileron roll effectiveness (per deg)
    Cnr: float = -0.1,     # Yaw damping
    Cndr: float = -0.001,  # Rudder yaw effectiveness (per deg)
    reference_chord: float = 0.3,
    reference_span: float = 2.0,
) -> Callable:
    """
    Create a simple linear aerodynamic model for testing.
    
    This model uses linear stability derivatives, suitable for:
    - Initial development and debugging
    - When AeroSandbox analysis is too slow
    - As a fallback when geometry is not fully defined
    
    The model includes rate damping terms normalized by reference dimensions.
    
    Args:
        CL0: Zero-alpha lift coefficient
        CLa: Lift curve slope (per radian)
        CD0: Zero-lift drag coefficient
        K: Induced drag factor
        Cma: Pitch moment slope (per radian)
        Cmq: Pitch rate damping
        Cmde: Elevator control derivative (per degree)
        Clp: Roll rate damping
        Clda: Aileron control derivative (per degree)
        Cnr: Yaw rate damping
        Cndr: Rudder control derivative (per degree)
        reference_chord: Mean aerodynamic chord [m]
        reference_span: Wingspan [m]
    
    Returns:
        Callable aero_model compatible with FlyingWingDynamics6DOF
    """
    import numpy as np
    
    c = reference_chord
    b = reference_span
    
    def aero_model(
        alpha: float,
        beta: float,
        airspeed: float,
        p: float, q: float, r: float,
        elevator: float = 0.0,
        aileron: float = 0.0,
        rudder: float = 0.0,
    ) -> Dict[str, float]:
        # Convert angles to radians for stability derivatives
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        
        # Avoid division by zero
        V = max(airspeed, 0.1)
        
        # Lift and drag
        CL = CL0 + CLa * alpha_rad
        CD = CD0 + K * CL**2
        
        # Side force (simple model)
        CY = -0.3 * beta_rad
        
        # Non-dimensional rates
        p_hat = p * b / (2 * V)   # Non-dimensional roll rate
        q_hat = q * c / (2 * V)   # Non-dimensional pitch rate
        r_hat = r * b / (2 * V)   # Non-dimensional yaw rate
        
        # Moments
        Cm = Cma * alpha_rad + Cmq * q_hat + Cmde * elevator
        Cl_moment = Clp * p_hat + Clda * aileron
        Cn = Cnr * r_hat + Cndr * rudder
        
        return {
            'CL': CL,
            'CD': CD,
            'CY': CY,
            'Cl': Cl_moment,  # Roll moment coefficient
            'Cm': Cm,         # Pitch moment coefficient
            'Cn': Cn,         # Yaw moment coefficient
        }
    
    return aero_model


# =============================================================================
# RigidBodyAeroModel for 3-DOF Mission Simulation
# =============================================================================


class RigidBodyAeroModel:
    """
    Aerodynamic model for 3-DOF mission analysis.

    Uses AeroBuildup to compute CL, CD, and CM at various angles of attack
    and control surface deflections.

    With use_precomputed_polars=True (default), builds a polar lookup table at
    initialization for smooth, fast interpolation during simulation.
    
    This model provides the same physics as the takeoff_analysis3DOF.py script,
    with matching verbose output format for consistency.
    """

    def __init__(
        self,
        wing_project: 'WingProject',
        ground_effect: bool = True,
        use_precomputed_polars: bool = True,
        alpha_range: Tuple[float, float, int] = (-5.0, 20.0, 26),  # 1° steps
        delta_e_range: Tuple[float, float, int] = (-25.0, 10.0, 8),  # ~5° steps
        verbose: bool = True,
    ):
        """
        Initialize aerodynamic model.

        Args:
            wing_project: WingProject definition
            ground_effect: Enable ground effect modeling
                            NOTE: This flag is currently stored but NOT USED.
                            Ground effect modeling has been removed from the
                            coefficient calculations. The flag remains for backward
                            compatibility but does not affect physics results.
            use_precomputed_polars: If True, build polar lookup table at init
                                    (~30s startup, ~μs per call during sim)
                                    If False, use AeroBuildup per call with caching
            alpha_range: (min, max, n_points) for alpha grid [deg]
            delta_e_range: (min, max, n_points) for elevator grid [deg]
            verbose: Print progress during polar computation
        """
        from services.geometry import AeroSandboxService
        
        self.wing_project = wing_project
        self.ground_effect = ground_effect
        self.use_precomputed_polars = use_precomputed_polars
        self.verbose = verbose
        self.aero_service = AeroSandboxService(wing_project)

        # Cache key parameters
        self.S = wing_project.planform.wing_area_m2
        self.span = wing_project.planform.actual_span()

        # Build wing to get AC and MAC
        wing = self.aero_service.build_wing()
        self.x_ac = wing.aerodynamic_center()[0]
        self.mac = wing.mean_aerodynamic_chord()

        # Calculate CG based on static margin
        static_margin = self.wing_project.twist_trim.static_margin_percent
        self.x_cg = self.x_ac - (static_margin / 100.0) * self.mac

        # Identify pitch control surface name
        self.pitch_control_name = "Elevon"  # Default fallback
        if self.wing_project.planform.control_surfaces:
            for cs in self.wing_project.planform.control_surfaces:
                if cs.surface_type.lower() in ["elevon", "elevator"]:
                    self.pitch_control_name = cs.name
                    break

        # Initialize polar storage
        self._polar_cache: Dict[
            Tuple[float, float, float, float], Tuple[float, float, float]
        ] = {}
        self._interp_CL = None
        self._interp_CD = None
        self._interp_CM = None
        self._interp_CMq = None
        self._alpha_grid = None
        self._delta_e_grid = None

        # Build precomputed polars if requested
        if use_precomputed_polars:
            self._build_polar_table(alpha_range, delta_e_range)

    def _build_polar_table(
        self,
        alpha_range: Tuple[float, float, int],
        delta_e_range: Tuple[float, float, int],
    ):
        """Build precomputed polar lookup table using AeroBuildup."""
        alpha_min, alpha_max, n_alpha = alpha_range
        delta_e_min, delta_e_max, n_delta_e = delta_e_range

        self._alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha)
        self._delta_e_grid = np.linspace(delta_e_min, delta_e_max, n_delta_e)

        n_total = n_alpha * n_delta_e

        if self.verbose:
            print(f"\nBuilding aerodynamic polar table (Vectorized)...")
            print(f"  Alpha: {alpha_min}deg to {alpha_max}deg ({n_alpha} points)")
            print(
                f"  Delta_e: {delta_e_min}deg to {delta_e_max}deg ({n_delta_e} points)"
            )
            print(f"  Total: {n_total} evaluations")

        # Initialize arrays
        CL_data = np.zeros((n_alpha, n_delta_e))
        CD_data = np.zeros((n_alpha, n_delta_e))
        CM_data = np.zeros((n_alpha, n_delta_e))
        CMq_data = np.zeros((n_alpha, n_delta_e))

        start_time = time.time()

        for j, delta_e in enumerate(self._delta_e_grid):
            try:
                # Build airplane once for this delta_e
                controls = {self.pitch_control_name: delta_e}
                airplane = self.aero_service.build_airplane(
                    xyz_ref=[self.x_cg, 0.0, 0.0], control_deflections=controls
                )

                # Vectorized alpha evaluation
                atmo = asb.Atmosphere(altitude=0.0)
                op_point = asb.OperatingPoint(
                    atmosphere=atmo,
                    velocity=20.0,
                    alpha=self._alpha_grid,
                )

                aero = asb.AeroBuildup(airplane, op_point)
                result = aero.run_with_stability_derivatives(q=True)

                # Reshape to ensure 1D array even if single alpha
                CL_data[:, j] = np.reshape(result.get("CL", result.get("Cl", 0.0)), -1)
                CD_data[:, j] = np.reshape(result.get("CD", result.get("Cd", 0.0)), -1)
                CM_data[:, j] = np.reshape(result.get("CM", result.get("Cm", 0.0)), -1)
                CMq_data[:, j] = np.reshape(result.get("Cmq", 0.0), -1)

                if self.verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"  [{j + 1}/{n_delta_e}] delta_e={delta_e:5.1f}deg | "
                        f"CL avg={np.mean(CL_data[:, j]):.3f} | {elapsed:.1f}s elapsed"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed at delta_e={delta_e:.1f}deg: {e}")
                CL_data[:, j] = np.nan
                CD_data[:, j] = np.nan
                CM_data[:, j] = np.nan
                CMq_data[:, j] = np.nan

        elapsed = time.time() - start_time

        # Build interpolators
        grid_points = (self._alpha_grid, self._delta_e_grid)
        self._interp_CL = RegularGridInterpolator(
            grid_points, CL_data, bounds_error=False, fill_value=None
        )
        self._interp_CD = RegularGridInterpolator(
            grid_points, CD_data, bounds_error=False, fill_value=None
        )
        self._interp_CM = RegularGridInterpolator(
            grid_points, CM_data, bounds_error=False, fill_value=None
        )
        self._interp_CMq = RegularGridInterpolator(
            grid_points, CMq_data, bounds_error=False, fill_value=None
        )

        if self.verbose:
            print(f"\nPolar table complete in {elapsed:.1f}s")
            print(f"  CL range: [{np.nanmin(CL_data):.3f}, {np.nanmax(CL_data):.3f}]")
            print(f"  CD range: [{np.nanmin(CD_data):.3f}, {np.nanmax(CD_data):.3f}]")
            print(f"  CM range: [{np.nanmin(CM_data):.3f}, {np.nanmax(CM_data):.3f}]")
            print(
                f"  CMq range: [{np.nanmin(CMq_data):.3f}, {np.nanmax(CMq_data):.3f}]"
            )
            print(f"  Ready for fast interpolation (~us per call)")

    def _compute_coefficients_raw(
        self, alpha_deg: float, delta_e: float, V: float = 20.0, altitude: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Compute coefficients using AeroBuildup (slow, ~200ms per call)."""
        controls = {self.pitch_control_name: delta_e}
        airplane = self.aero_service.build_airplane(
            xyz_ref=[self.x_cg, 0.0, 0.0], control_deflections=controls
        )

        atmo = asb.Atmosphere(altitude=max(0, altitude))
        op_point = asb.OperatingPoint(
            atmosphere=atmo,
            velocity=max(V, 1.0),
            alpha=alpha_deg,
        )

        aero = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point,
        )
        result = aero.run_with_stability_derivatives(q=True)

        CL = float(result.get("CL", result.get("Cl", 0.0)))
        CD = float(result.get("CD", result.get("Cd", 0.0)))
        CM = float(result.get("CM", result.get("Cm", 0.0)))
        CMq = float(result.get("Cmq", 0.0))

        return CL, CD, CM, CMq

    def get_coefficients(
        self, alpha_deg: float, V: float, delta_e: float = 0.0, altitude: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Get aerodynamic coefficients CL, CD, CM, CMq.

        Args:
            alpha_deg: Angle of attack [degrees]
            V: Airspeed [m/s]
            delta_e: Pitch control deflection [degrees] (positive = down)
            altitude: Altitude [m]
                    NOTE: Altitude is NOT USED for ground effect modeling.

        Returns:
            (CL, CD, CM, CMq) Tuple
        """
        if self.use_precomputed_polars and self._interp_CL is not None:
            # Check for NaNs before interpolation to prevent CasADi crash
            if not np.isfinite(alpha_deg) or not np.isfinite(delta_e) or not np.isfinite(V):
                return 0.0, 0.5, 0.0, 0.0  # Safe fallback (CL=0, CD=0.5)

            alpha_clamped = alpha_deg
            delta_e_clamped = delta_e
            if self._alpha_grid is not None:
                alpha_min = float(np.min(self._alpha_grid))
                alpha_max = float(np.max(self._alpha_grid))
                alpha_clamped = float(np.clip(alpha_deg, alpha_min, alpha_max))
            if self._delta_e_grid is not None:
                delta_min = float(np.min(self._delta_e_grid))
                delta_max = float(np.max(self._delta_e_grid))
                delta_e_clamped = float(np.clip(delta_e, delta_min, delta_max))
            point = np.array([[alpha_clamped, delta_e_clamped]])
            CL = float(self._interp_CL(point)[0])
            CD = float(self._interp_CD(point)[0])
            CM = float(self._interp_CM(point)[0])
            CMq = float(self._interp_CMq(point)[0])
            if not np.isfinite(CL) or not np.isfinite(CD) or not np.isfinite(CM) or not np.isfinite(CMq):
                CL, CD, CM, CMq = self._compute_coefficients_raw(
                    alpha_clamped, delta_e_clamped, V, altitude
                )
            return CL, CD, CM, CMq

        # Fallback to direct AeroBuildup with rounding for caching
        alpha_rounded = round(alpha_deg * 2) / 2  # 0.5 deg bins
        delta_e_rounded = round(delta_e * 2) / 2  # 0.5 deg bins

        cache_key = (alpha_rounded, delta_e_rounded, round(V), round(altitude))
        if cache_key not in self._polar_cache:
            CL, CD, CM, CMq = self._compute_coefficients_raw(
                alpha_deg, delta_e, V, altitude
            )
            self._polar_cache[cache_key] = (CL, CD, CM, CMq)

        return self._polar_cache[cache_key]

    def get_forces_moments(
        self,
        V: float,
        alpha_deg: float,
        delta_e: float,
        rho: float,
        altitude: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Compute aerodynamic forces and pitching moment.

        Args:
            V: Airspeed [m/s]
            alpha_deg: Angle of attack [degrees]
            delta_e: Elevator deflection [degrees]
            rho: Air density [kg/m^3]
            altitude: Altitude [m]

        Returns:
            (L, D, M, CMq) Tuple of lift [N], drag [N], moment [N-m], and CMq derivative
        """
        CL, CD, CM, CMq = self.get_coefficients(alpha_deg, V, delta_e, altitude)

        q = 0.5 * rho * V**2 * self.S
        L = CL * q
        D = CD * q
        M = CM * q * self.mac

        return L, D, M, CMq

    def get_forces(
        self, V: float, alpha_deg: float, rho: float, altitude: float = 0.0
    ) -> Tuple[float, float]:
        """Backward compatibility for other calculators."""
        L, D, M, CMq = self.get_forces_moments(V, alpha_deg, 0.0, rho, altitude)
        return L, D

    def __call__(
        self,
        alpha: float,
        beta: float,
        airspeed: float,
        p: float,
        q: float,
        r: float,
        elevator: float = 0.0,
        aileron: float = 0.0,
        rudder: float = 0.0,
    ) -> Dict[str, float]:
        """
        Callable interface compatible with MissionSimulator aero_model.
        
        Returns dict with CL, CD, CY, Cl, Cm, Cn for use in simulation.
        """
        CL, CD, CM, CMq = self.get_coefficients(alpha, airspeed, elevator, 0.0)
        
        # For 3-DOF, we ignore lateral/directional coefficients
        return {
            'CL': CL,
            'CD': CD,
            'CY': 0.0,
            'Cl': 0.0,  # Roll moment
            'Cm': CM,   # Pitch moment
            'Cn': 0.0,  # Yaw moment
            'CMq': CMq,
        }


def create_rigid_body_aero_model(
    wing_project: 'WingProject',
    ground_effect: bool = True,
    use_precomputed_polars: bool = True,
    alpha_range: Tuple[float, float, int] = (-5.0, 20.0, 26),
    delta_e_range: Tuple[float, float, int] = (-25.0, 10.0, 8),
    verbose: bool = True,
) -> RigidBodyAeroModel:
    """
    Factory function to create a RigidBodyAeroModel for 3-DOF mission simulation.
    
    This model is designed to match the physics and output format of the
    takeoff_analysis3DOF.py script, ensuring consistent results between
    the standalone script and GUI-based simulation.
    
    Build time: ~12s for default grid (26×8 = 208 evaluations)
    Evaluation time: ~μs per call
    
    Args:
        wing_project: WingProject definition
        ground_effect: Enable ground effect modeling (currently not implemented)
        use_precomputed_polars: Build lookup table at init (recommended)
        alpha_range: (min, max, n_points) for alpha grid [deg]
        delta_e_range: (min, max, n_points) for elevator grid [deg]
        verbose: Print progress during polar computation
    
    Returns:
        RigidBodyAeroModel instance (callable)
    
    Example:
        >>> aero = create_rigid_body_aero_model(wing_project)  # ~12s polar build
        >>> result = aero(alpha=5, beta=0, airspeed=20, p=0, q=0, r=0, elevator=-2)
        >>> print(f"CL: {result['CL']:.3f}, CD: {result['CD']:.4f}, Cm: {result['Cm']:.3f}")
    """
    return RigidBodyAeroModel(
        wing_project=wing_project,
        ground_effect=ground_effect,
        use_precomputed_polars=use_precomputed_polars,
        alpha_range=alpha_range,
        delta_e_range=delta_e_range,
        verbose=verbose,
    )
