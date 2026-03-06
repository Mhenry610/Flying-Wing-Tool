"""
Comprehensive Aerodynamic Analysis Script (AD + 2D Grid + Controls + Post-Stall Model)

Uses CasADi's automatic differentiation through AeroSandbox for EXACT derivatives,
with an extended post-stall model that blends attached-flow AeroBuildup coefficients
into a simple separated-flow model beyond stall, and applies control effectiveness
fadeout near/after stall.

**POST-STALL MODEL FEATURES**:
- Smooth blend from attached flow to separated flow using tanh transition
- Flat-plate normal/axial force model for separated flow (CN, CA)
- Control effectiveness fadeout beyond stall angle
- Plausible post-stall pitching moment behavior
- All derivatives remain smooth through stall transition

**CONTROL SURFACE MODEL**:
Instead of sweeping δ (which is 10-100x more expensive), we compute:
- ∂C/∂δ at δ=0 using AD (control derivatives already in Jacobian)
- Reconstruct: C(α,δ) ≈ C(α,0) + η(α) * Cδ(α) * sat(δ/δ_max)
- Where sat(x) = tanh(x) or clamp(x, -1, 1)

Sweep ranges (defaults):
- Pitch (alpha): -20 to +40 degrees (0.5° step = 121 points)
- Sideslip (beta): -25 to +25 degrees (0.5° step = 101 points)
- δ_max: 30 degrees (saturation limit for control reconstruction)

Usage:
    python scripts/stability_derivatives_ad.py
    python scripts/stability_derivatives_ad.py --alpha-stall 16 --blend-width 4
    python scripts/stability_derivatives_ad.py --delta-max 25
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CasADi
try:
    import casadi as cas
except ImportError:
    print("ERROR: CasADi not installed. Install with: pip install casadi")
    sys.exit(1)

import aerosandbox as asb
import aerosandbox.numpy as asnp
from services.geometry import AeroSandboxService


# =============================================================================
# Post-Stall Model Parameters (defaults, can be overridden via CLI)
# =============================================================================
@dataclass
class PostStallParams:
    """Parameters for post-stall aerodynamic blending model."""
    alpha_stall_deg: float = 16.0      # Stall angle of attack [deg]
    blend_width_deg: float = 4.0       # Width of tanh transition [deg]
    eta_min: float = 0.3               # Minimum control effectiveness in stall
    eta_power: float = 2.0             # Power for control fadeout curve
    
    # Separated flow model parameters (flat plate)
    CN_max: float = 1.8                # Peak normal force coefficient
    CA_0: float = 0.02                 # Base axial force (skin friction)
    CA_k: float = 0.8                  # Axial force due to separated flow
    
    # Post-stall pitching moment
    Cm_stall_offset: float = 0.0       # Cm offset at stall (will be computed)
    Cm_post_stall_slope: float = -0.02 # dCm/dalpha slope in separated flow [/deg]
    Cm_alpha_width_deg: float = 20.0   # Width of Cm transition
    
    # Lateral/directional (simplified)
    Cl_sep_factor: float = 0.3         # Reduction in roll moment in stall
    Cn_sep_factor: float = 0.3         # Reduction in yaw moment in stall
    CY_sep_factor: float = 0.3         # Reduction in side force in stall


# Thread-safe progress tracking
_progress_lock = threading.Lock()
_progress_counter = 0
_total_points = 0


def reset_progress(total: int = 0):
    global _progress_counter, _total_points
    with _progress_lock:
        _progress_counter = 0
        _total_points = total


def increment_progress() -> Tuple[int, int]:
    global _progress_counter
    with _progress_lock:
        _progress_counter += 1
        return _progress_counter, _total_points


class ADStabilityAnalyzerWithPostStall:
    """
    Automatic Differentiation-based stability derivative analyzer with post-stall model.
    
    Blends attached-flow (AeroBuildup) coefficients with separated-flow model
    using a smooth tanh transition. Control effectiveness fades beyond stall.
    
    The blending uses:
        w(alpha) = 0.5 * (1 + tanh((|alpha| - alpha_stall) / blend_width))
        
    Where w=0 is fully attached, w=1 is fully separated.
    """
    
    def __init__(
        self,
        service: AeroSandboxService,
        x_cg: float,
        airspeed: float,
        post_stall_params: Optional[PostStallParams] = None,
    ):
        self.service = service
        self.x_cg = x_cg
        self.airspeed = airspeed
        self.ps = post_stall_params or PostStallParams()
        
        # Get reference values
        wing = service.build_wing()
        self.c_ref = wing.mean_aerodynamic_chord()
        self.b_ref = wing.span()
        
        # Detect control surfaces
        self.control_surface_names = self._detect_control_surfaces()
        
        # Build symbolic functions with post-stall model
        self._build_symbolic_functions()
    
    def _detect_control_surfaces(self) -> List[str]:
        """Detect control surfaces from project."""
        planform = self.service.wing_project.planform
        control_surfaces = planform.control_surfaces
        
        names = []
        for cs in control_surfaces:
            surface_type = cs.surface_type.lower()
            if surface_type == 'elevon':
                names.append(f'{cs.name}_pitch')
                names.append(f'{cs.name}_roll')
            else:
                names.append(cs.name)
        
        if names:
            print(f"  Detected control surfaces: {names}")
        else:
            print(f"  No control surfaces detected")
        
        return names
    
    def _build_symbolic_functions(self):
        """
        Build CasADi symbolic functions with post-stall blending.
        
        The model computes:
        1. Attached flow coefficients at delta=0 (baseline)
        2. Attached flow coefficients at commanded delta
        3. Incremental control effects: dC = C_att(delta) - C_att(0)
        4. Control effectiveness factor: eta(alpha)
        5. Faded attached coefficients: C_att_faded = C_att(0) + eta * dC
        6. Separated flow coefficients (flat-plate model)
        7. Blend weight: w(alpha)
        8. Final: C = (1-w) * C_att_faded + w * C_sep
        """
        print("  Building symbolic AD functions with post-stall model...")
        start = time.time()
        
        ps = self.ps
        
        # === Symbolic inputs ===
        self.sym_alpha = cas.SX.sym('alpha')  # radians
        self.sym_beta = cas.SX.sym('beta')    # radians
        self.sym_p = cas.SX.sym('p')          # rad/s
        self.sym_q = cas.SX.sym('q')          # rad/s
        self.sym_r = cas.SX.sym('r')          # rad/s
        
        # Control surface variables
        self.sym_controls = {}
        for name in self.control_surface_names:
            self.sym_controls[name] = cas.SX.sym(name)  # radians
        
        # Stack all inputs
        input_list = [self.sym_alpha, self.sym_beta, self.sym_p, self.sym_q, self.sym_r]
        input_list.extend([self.sym_controls[name] for name in self.control_surface_names])
        self.sym_inputs = cas.vertcat(*input_list)
        
        self.input_names = ['alpha', 'beta', 'p', 'q', 'r'] + self.control_surface_names
        self.n_inputs = len(self.input_names)
        
        # Convert angles to degrees for AeroSandbox
        alpha_deg = self.sym_alpha * 180 / cas.pi
        beta_deg = self.sym_beta * 180 / cas.pi
        
        # === Post-stall blending parameters (convert to radians where needed) ===
        alpha_stall_rad = np.radians(ps.alpha_stall_deg)
        blend_width_rad = np.radians(ps.blend_width_deg)
        
        # Blend weight: w = 0.5 * (1 + tanh((|alpha| - alpha_stall) / blend_width))
        # Use absolute alpha for symmetric stall behavior
        alpha_abs = cas.fabs(self.sym_alpha)
        w_blend = 0.5 * (1 + cas.tanh((alpha_abs - alpha_stall_rad) / blend_width_rad))
        
        # Control effectiveness: eta = eta_min + (1 - eta_min) * (1 - w)^p
        eta_ctrl = ps.eta_min + (1 - ps.eta_min) * cas.power(1 - w_blend, ps.eta_power)

        # Signed stall reference angle for post-stall Cm anchoring
        stall_sign = cas.if_else(self.sym_alpha >= 0, 1.0, -1.0)
        alpha_stall_signed_rad = alpha_stall_rad * stall_sign
        alpha_stall_signed_deg = alpha_stall_signed_rad * 180 / cas.pi
        
        # === Build AeroBuildup for BASELINE (delta=0) ===
        control_deflections_zero = {name: 0.0 for name in self.control_surface_names}
        wing_baseline = self.service.build_wing(control_deflections=control_deflections_zero)
        
        airplane_baseline = asb.Airplane(
            name=self.service.wing_project.name + "_baseline",
            wings=[wing_baseline],
            xyz_ref=[self.x_cg, 0.0, 0.0],
        )
        
        op_point_baseline = asb.OperatingPoint(
            atmosphere=self.service.atmosphere,
            velocity=self.airspeed,
            alpha=alpha_deg,
            beta=beta_deg,
            p=self.sym_p,
            q=self.sym_q,
            r=self.sym_r,
        )
        
        aero_baseline = asb.AeroBuildup(airplane=airplane_baseline, op_point=op_point_baseline)
        result_baseline = aero_baseline.run()

        # Baseline reference at stall (anchored Cm at alpha_stall)
        op_point_stall = asb.OperatingPoint(
            atmosphere=self.service.atmosphere,
            velocity=self.airspeed,
            alpha=alpha_stall_signed_deg,
            beta=beta_deg,
            p=self.sym_p,
            q=self.sym_q,
            r=self.sym_r,
        )
        aero_stall = asb.AeroBuildup(airplane=airplane_baseline, op_point=op_point_stall)
        result_stall = aero_stall.run()
        
        # Baseline attached coefficients
        CL_att_0 = result_baseline['CL']
        CD_att_0 = result_baseline['CD']
        CY_att_0 = result_baseline['CY']
        Cl_att_0 = result_baseline['Cl']
        Cm_att_0 = result_baseline['Cm']
        Cn_att_0 = result_baseline['Cn']
        Cm_att_stall = result_stall['Cm']
        
        # === Build AeroBuildup with COMMANDED control deflections ===
        control_deflections_cmd = {}
        for name in self.control_surface_names:
            control_deflections_cmd[name] = self.sym_controls[name] * 180 / cas.pi
        
        wing_cmd = self.service.build_wing(control_deflections=control_deflections_cmd)
        
        airplane_cmd = asb.Airplane(
            name=self.service.wing_project.name + "_commanded",
            wings=[wing_cmd],
            xyz_ref=[self.x_cg, 0.0, 0.0],
        )
        
        op_point_cmd = asb.OperatingPoint(
            atmosphere=self.service.atmosphere,
            velocity=self.airspeed,
            alpha=alpha_deg,
            beta=beta_deg,
            p=self.sym_p,
            q=self.sym_q,
            r=self.sym_r,
        )
        
        aero_cmd = asb.AeroBuildup(airplane=airplane_cmd, op_point=op_point_cmd)
        result_cmd = aero_cmd.run()
        
        # Commanded attached coefficients
        CL_att_cmd = result_cmd['CL']
        CD_att_cmd = result_cmd['CD']
        CY_att_cmd = result_cmd['CY']
        Cl_att_cmd = result_cmd['Cl']
        Cm_att_cmd = result_cmd['Cm']
        Cn_att_cmd = result_cmd['Cn']
        
        # === Incremental control effects ===
        dCL_ctrl = CL_att_cmd - CL_att_0
        dCD_ctrl = CD_att_cmd - CD_att_0
        dCY_ctrl = CY_att_cmd - CY_att_0
        dCl_ctrl = Cl_att_cmd - Cl_att_0
        dCm_ctrl = Cm_att_cmd - Cm_att_0
        dCn_ctrl = Cn_att_cmd - Cn_att_0
        
        # === Faded attached coefficients (control effectiveness reduced in stall) ===
        CL_att_faded = CL_att_0 + eta_ctrl * dCL_ctrl
        CD_att_faded = CD_att_0 + eta_ctrl * dCD_ctrl
        CY_att_faded = CY_att_0 + eta_ctrl * dCY_ctrl
        Cl_att_faded = Cl_att_0 + eta_ctrl * dCl_ctrl
        Cm_att_faded = Cm_att_0 + eta_ctrl * dCm_ctrl
        Cn_att_faded = Cn_att_0 + eta_ctrl * dCn_ctrl
        
        # === Separated flow model (flat-plate) ===
        # Normal force: CN = CN_max * sin(2*alpha)
        # Axial force: CA = CA_0 + CA_k * sin(alpha)^2
        sin_alpha = cas.sin(self.sym_alpha)
        cos_alpha = cas.cos(self.sym_alpha)
        sin_2alpha = cas.sin(2 * self.sym_alpha)
        
        CN_sep = ps.CN_max * sin_2alpha
        CA_sep = ps.CA_0 + ps.CA_k * cas.power(sin_alpha, 2)
        
        # Convert to body-axis lift and drag
        # CL = CN * cos(alpha) - CA * sin(alpha)
        # CD = CN * sin(alpha) + CA * cos(alpha)
        CL_sep = CN_sep * cos_alpha - CA_sep * sin_alpha
        CD_sep_raw = CN_sep * sin_alpha + CA_sep * cos_alpha
        
        # Clamp CD to minimum value
        CD_sep = cas.fmax(CD_sep_raw, 0.02)
        
        # Side force in separated flow (simplified - reduced effectiveness)
        CY_sep = ps.CY_sep_factor * CY_att_0
        
        # Post-stall pitching moment
        # Cm_sep = Cm_at_stall + dCm * tanh((alpha - alpha_stall) / width)
        # We use the attached Cm at stall as reference, then add a pitch-down tendency
        Cm_stall_ref = Cm_att_stall  # Use attached Cm at stall as reference
        alpha_excess = self.sym_alpha - alpha_stall_rad * cas.sign(self.sym_alpha)
        Cm_post_stall_delta = ps.Cm_post_stall_slope * (alpha_abs - alpha_stall_rad) * 180 / cas.pi
        Cm_sep = Cm_stall_ref + Cm_post_stall_delta * cas.tanh(alpha_excess / np.radians(ps.Cm_alpha_width_deg))
        
        # Lateral/directional in separated flow (reduced effectiveness)
        Cl_sep = ps.Cl_sep_factor * Cl_att_0
        Cn_sep = ps.Cn_sep_factor * Cn_att_0
        
        # === Final blended coefficients ===
        # C = (1 - w) * C_att_faded + w * C_sep
        CL_final = (1 - w_blend) * CL_att_faded + w_blend * CL_sep
        CD_final = (1 - w_blend) * CD_att_faded + w_blend * CD_sep
        CY_final = (1 - w_blend) * CY_att_faded + w_blend * CY_sep
        Cl_final = (1 - w_blend) * Cl_att_faded + w_blend * Cl_sep
        Cm_final = (1 - w_blend) * Cm_att_faded + w_blend * Cm_sep
        Cn_final = (1 - w_blend) * Cn_att_faded + w_blend * Cn_sep
        
        # === Stack outputs ===
        self.sym_outputs = cas.vertcat(
            CL_final, CD_final, CY_final,
            Cl_final, Cm_final, Cn_final
        )
        self.output_names = ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']
        self.n_outputs = len(self.output_names)
        
        # Also output the blend weight and control effectiveness for diagnostics
        self.sym_w_blend = w_blend
        self.sym_eta_ctrl = eta_ctrl
        
        # === Compute Jacobian ===
        self.sym_jacobian = cas.jacobian(self.sym_outputs, self.sym_inputs)
        
        # === Create CasADi functions ===
        # Main function: coefficients + Jacobian
        self.combined_func = cas.Function(
            'aero_poststall',
            [self.sym_inputs],
            [self.sym_outputs, self.sym_jacobian, w_blend, eta_ctrl],
            ['inputs'],
            ['outputs', 'jacobian', 'w_blend', 'eta_ctrl']
        )
        
        elapsed = time.time() - start
        print(f"  Built post-stall model with {self.n_outputs} outputs × {self.n_inputs} inputs")
        print(f"  Stall parameters: α_stall={ps.alpha_stall_deg}°, blend_width={ps.blend_width_deg}°")
        print(f"  Control fadeout: η_min={ps.eta_min}, power={ps.eta_power}")
        print(f"  Build time: {elapsed:.2f}s")
    
    def evaluate(
        self,
        alpha_deg: float,
        beta_deg: float,
        p: float = 0.0,
        q: float = 0.0,
        r: float = 0.0,
        control_deflections_deg: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate blended coefficients and derivatives at a flight condition.
        """
        if control_deflections_deg is None:
            control_deflections_deg = {}
        
        # Convert to radians
        alpha_rad = np.radians(alpha_deg)
        beta_rad = np.radians(beta_deg)
        
        # Build input vector
        inputs = [alpha_rad, beta_rad, p, q, r]
        for name in self.control_surface_names:
            deflection_rad = np.radians(control_deflections_deg.get(name, 0.0))
            inputs.append(deflection_rad)
        inputs = np.array(inputs)
        
        # Evaluate
        outputs, jacobian, w_blend, eta_ctrl = self.combined_func(inputs)
        
        # Convert to numpy
        outputs = np.array(outputs).flatten()
        jacobian = np.array(jacobian)
        w_blend = float(w_blend)
        eta_ctrl = float(eta_ctrl)
        
        # Build result dict
        result = {
            'alpha': alpha_deg,
            'beta': beta_deg,
            'airspeed': self.airspeed,
            'w_blend': w_blend,      # 0=attached, 1=separated
            'eta_ctrl': eta_ctrl,    # Control effectiveness factor
        }
        
        for i, name in enumerate(self.output_names):
            result[name] = float(outputs[i])
        
        # Add control states
        for name in self.control_surface_names:
            result[f'd_{name}'] = control_deflections_deg.get(name, 0.0)
        
        # === Extract derivatives from Jacobian ===
        V = self.airspeed
        b = self.b_ref
        c = self.c_ref
        
        p_norm = b / (2 * V)
        q_norm = c / (2 * V)
        r_norm = b / (2 * V)
        
        for i, coef in enumerate(self.output_names):
            for j, inp in enumerate(self.input_names):
                deriv_value = float(jacobian[i, j])
                
                if inp == 'p':
                    deriv_value /= p_norm
                    deriv_name = f'{coef}p'
                elif inp == 'q':
                    deriv_value /= q_norm
                    deriv_name = f'{coef}q'
                elif inp == 'r':
                    deriv_value /= r_norm
                    deriv_name = f'{coef}r'
                elif inp == 'alpha':
                    deriv_name = f'{coef}a'
                elif inp == 'beta':
                    deriv_name = f'{coef}b'
                else:
                    short_inp = inp.replace('_pitch', 'e').replace('_roll', 'a')
                    short_inp = short_inp.replace('Surface1', 'elv')
                    deriv_name = f'{coef}d{short_inp}'
                
                result[deriv_name] = deriv_value
        
        return result


def sweep_2d_grid(
    analyzer: ADStabilityAnalyzerWithPostStall,
    alpha_range: Tuple[float, float, int],
    beta_range: Tuple[float, float, int],
    control_deflections_deg: Optional[Dict[str, float]] = None,
    max_workers: int = 4,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Sweep full 2D grid of alpha × beta combinations."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], int(alpha_range[2]))
    betas = np.linspace(beta_range[0], beta_range[1], int(beta_range[2]))
    
    n_alpha = len(alphas)
    n_beta = len(betas)
    total_points = n_alpha * n_beta
    
    print(f"\n2D GRID SWEEP: {n_alpha} alpha × {n_beta} beta = {total_points} points")
    print(f"  Alpha: {alpha_range[0]}° to {alpha_range[1]}° ({n_alpha} points)")
    print(f"  Beta:  {beta_range[0]}° to {beta_range[1]}° ({n_beta} points)")
    print(f"  Workers: {max_workers}")
    
    grid_points = [(alpha, beta) for beta in betas for alpha in alphas]
    reset_progress(total_points)
    
    def evaluate_point(point: Tuple[float, float]) -> Dict:
        alpha, beta = point
        try:
            result = analyzer.evaluate(
                alpha_deg=alpha,
                beta_deg=beta,
                control_deflections_deg=control_deflections_deg,
            )
            count, total = increment_progress()
            
            if count % 100 == 0 or count == 1 or count == total:
                pct = 100 * count / total
                w = result.get('w_blend', 0)
                print(f"  [{count:5d}/{total}] ({pct:5.1f}%) α={alpha:6.1f}°, β={beta:6.1f}°, w={w:.2f}")
            return result
        except Exception as e:
            increment_progress()
            return {'alpha': alpha, 'beta': beta, 'CL': np.nan, 'error': str(e)}
    
    results = []
    start_time = time.time()
    
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_point = {executor.submit(evaluate_point, pt): pt for pt in grid_points}
            for future in as_completed(future_to_point):
                results.append(future.result())
    else:
        for pt in grid_points:
            results.append(evaluate_point(pt))
    
    elapsed = time.time() - start_time
    rate = total_points / elapsed if elapsed > 0 else 0
    
    print(f"\n  Completed {total_points} points in {elapsed:.1f}s ({rate:.1f} pts/sec)")
    
    results.sort(key=lambda x: (x.get('beta', 0), x.get('alpha', 0)))
    return results, alphas, betas


def reshape_to_grid(results: List[Dict], alphas: np.ndarray, betas: np.ndarray, key: str) -> np.ndarray:
    """Reshape flat results to 2D grid array."""
    values = np.array([r.get(key, np.nan) for r in results])
    return values.reshape(len(betas), len(alphas))


def save_results_json(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {filepath}")


def save_results_csv(data: List[Dict], filepath: str):
    """Save flat results to CSV."""
    import csv
    
    if not data:
        return
    
    priority = ['alpha', 'beta', 'airspeed', 'w_blend', 'eta_ctrl',
                'CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn',
                'CLa', 'Cma', 'CYb', 'Cnb', 'Clb', 'Clp', 'Cmq', 'Cnr']
    
    all_keys = list(data[0].keys())
    ordered = [k for k in priority if k in all_keys]
    ordered += [k for k in all_keys if k not in priority and k != 'error']
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"CSV saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="2D Aero Database with Post-Stall Model (AD)"
    )
    parser.add_argument('--project', type=str, default='IntendedValidation.json')
    parser.add_argument('--airspeed', type=float, default=None)
    parser.add_argument('--alpha-range', type=str, default='-20,40,121',
                        help='Alpha sweep: min,max,n_points (default: -20,40,121 = 0.5° step)')
    parser.add_argument('--beta-range', type=str, default='-25,25,101',
                        help='Beta sweep: min,max,n_points (default: -25,25,101 = 0.5° step)')
    parser.add_argument('--delta-max', type=float, default=30.0,
                        help='Max control deflection for saturation model [deg] (default: 30)')
    parser.add_argument('--output', type=str, default='aero_database_poststall.json')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--serial', action='store_true')
    
    # Post-stall model parameters
    parser.add_argument('--alpha-stall', type=float, default=16.0,
                        help='Stall angle of attack [deg]')
    parser.add_argument('--blend-width', type=float, default=4.0,
                        help='Width of stall transition [deg]')
    parser.add_argument('--eta-min', type=float, default=0.3,
                        help='Minimum control effectiveness in stall [0-1]')
    parser.add_argument('--eta-power', type=float, default=2.0,
                        help='Power for control fadeout curve')
    parser.add_argument('--cn-max', type=float, default=1.8,
                        help='Peak normal force coefficient (flat plate)')
    
    args = parser.parse_args()
    
    # Build post-stall parameters
    ps_params = PostStallParams(
        alpha_stall_deg=args.alpha_stall,
        blend_width_deg=args.blend_width,
        eta_min=args.eta_min,
        eta_power=args.eta_power,
        CN_max=args.cn_max,
    )
    
    # Workers
    if args.workers is None:
        max_workers = min(multiprocessing.cpu_count(), 16)
    else:
        max_workers = args.workers
    if args.serial:
        max_workers = 1
    
    # Parse ranges
    alpha_range = tuple(map(float, args.alpha_range.split(',')))
    alpha_range = (alpha_range[0], alpha_range[1], int(alpha_range[2]))
    
    beta_range = tuple(map(float, args.beta_range.split(',')))
    beta_range = (beta_range[0], beta_range[1], int(beta_range[2]))
    
    total_points = int(alpha_range[2]) * int(beta_range[2])
    
    # Load project
    project_path = project_root / args.project
    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        sys.exit(1)
    
    # Import here to avoid circular import when this module is imported by aero_model.py
    from core.state import Project
    
    print(f"Loading project from: {project_path}")
    project = Project.load(str(project_path))
    
    # Create service
    service = AeroSandboxService(project.wing)
    airspeed = args.airspeed if args.airspeed else service.design_velocity()
    
    # Reference data
    wing = service.build_wing()
    x_np = wing.aerodynamic_center()[0]
    mac = wing.mean_aerodynamic_chord()
    static_margin = project.wing.twist_trim.static_margin_percent
    x_cg = x_np - (static_margin / 100.0) * mac
    
    # Header
    print(f"\n{'='*70}")
    print("2D AERO DATABASE WITH POST-STALL MODEL (Automatic Differentiation)")
    print(f"{'='*70}")
    print(f"\n=== Configuration ===")
    print(f"  Project: {project.wing.name}")
    print(f"  Wing Area: {project.wing.planform.wing_area_m2:.2f} m²")
    print(f"  Aspect Ratio: {project.wing.planform.aspect_ratio:.2f}")
    print(f"  Airspeed: {airspeed:.2f} m/s")
    print(f"  Grid: {int(alpha_range[2])} × {int(beta_range[2])} = {total_points} points")
    print(f"  Workers: {max_workers}")
    
    print(f"\n=== Post-Stall Parameters ===")
    print(f"  Stall angle: {ps_params.alpha_stall_deg}°")
    print(f"  Blend width: {ps_params.blend_width_deg}°")
    print(f"  Control η_min: {ps_params.eta_min}")
    print(f"  Control fadeout power: {ps_params.eta_power}")
    print(f"  Flat-plate CN_max: {ps_params.CN_max}")
    
    print(f"\n=== Reference ===")
    print(f"  MAC: {mac:.4f} m, Span: {wing.span():.4f} m")
    print(f"  Neutral Point: {x_np:.4f} m")
    print(f"  CG (SM={static_margin}%): {x_cg:.4f} m")
    
    # Create analyzer
    print(f"\n=== Building Post-Stall AD Analyzer ===")
    analyzer = ADStabilityAnalyzerWithPostStall(
        service=service,
        x_cg=x_cg,
        airspeed=airspeed,
        post_stall_params=ps_params,
    )
    
    # 2D sweep
    print("\n" + "="*70)
    print("FULL 2D GRID SWEEP WITH POST-STALL BLENDING")
    print("="*70)
    
    total_start = time.time()
    
    grid_results, alphas, betas = sweep_2d_grid(
        analyzer,
        alpha_range=alpha_range,
        beta_range=beta_range,
        max_workers=max_workers,
    )
    
    total_elapsed = time.time() - total_start
    
    # Build output
    results = {
        'project': project.wing.name,
        'method': 'automatic_differentiation_with_poststall',
        'grid_type': '2D_full_with_poststall',
        'control_surfaces': analyzer.control_surface_names,
        'post_stall_params': {
            'alpha_stall_deg': ps_params.alpha_stall_deg,
            'blend_width_deg': ps_params.blend_width_deg,
            'eta_min': ps_params.eta_min,
            'eta_power': ps_params.eta_power,
            'CN_max': ps_params.CN_max,
            'CA_0': ps_params.CA_0,
            'CA_k': ps_params.CA_k,
        },
        'configuration': {
            'wing_area_m2': project.wing.planform.wing_area_m2,
            'aspect_ratio': project.wing.planform.aspect_ratio,
            'taper_ratio': project.wing.planform.taper_ratio,
            'sweep_le_deg': project.wing.planform.sweep_le_deg,
            'dihedral_deg': project.wing.planform.dihedral_deg,
        },
        'reference': {
            'airspeed_m_s': airspeed,
            'mac_m': mac,
            'span_m': analyzer.b_ref,
            'x_np_m': x_np,
            'x_cg_m': x_cg,
            'static_margin_percent': static_margin,
        },
        'grid': {
            'alpha_deg': alphas.tolist(),
            'beta_deg': betas.tolist(),
            'n_alpha': len(alphas),
            'n_beta': len(betas),
            'total_points': len(grid_results),
        },
        'data': grid_results,
    }
    
    # Add 2D tables
    results['tables'] = {}
    key_coeffs = ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn', 'w_blend', 'eta_ctrl',
                  'CLa', 'Cma', 'Cnb', 'Clb', 'Clp', 'Cmq', 'Cnr']
    
    if analyzer.control_surface_names:
        for cs_name in analyzer.control_surface_names:
            short = cs_name.replace('_pitch', 'e').replace('_roll', 'a').replace('Surface1', 'elv')
            key_coeffs.extend([f'CLd{short}', f'Cmd{short}', f'Cld{short}', f'Cnd{short}'])
    
    for key in key_coeffs:
        if grid_results and key in grid_results[0]:
            results['tables'][key] = reshape_to_grid(grid_results, alphas, betas, key).tolist()
    
    # Add control saturation model info (for reconstructing C(α,δ) without sweeping)
    # C(α,δ) ≈ C(α,0) + η(α) * Cδ(α) * sat(δ)
    # The Jacobian already contains ∂C/∂δ at δ=0, so we just document the saturation function
    if analyzer.control_surface_names:
        results['control_model'] = {
            'description': 'Use C(α,δ) ≈ C(α,0) + η(α) * Cδ(α) * sat(δ/δ_max)',
            'delta_max_deg': args.delta_max,
            'saturation_function': 'sat(x) = tanh(x) or clamp(x, -1, 1)',
            'control_derivatives': [f'{coef}d{cs.replace("_pitch","e").replace("_roll","a").replace("Surface1","elv")}' 
                                    for cs in analyzer.control_surface_names 
                                    for coef in ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']],
            'effectiveness_column': 'eta_ctrl',
            'note': 'Control derivatives (Cδ) are computed at δ=0 via AD. '
                    'Reconstruct non-zero δ using: ΔC = η(α) * Cδ * sat(δ/δ_max)'
        }
    
    # Save
    output_path = project_root / 'scripts' / args.output
    save_results_json(results, str(output_path))
    
    csv_path = output_path.with_suffix('.csv')
    save_results_csv(grid_results, str(csv_path))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid_data = [d for d in grid_results if not np.isnan(d.get('CL', np.nan))]
    
    if valid_data:
        # Find key characteristics
        trim_data = [d for d in valid_data if abs(d['alpha']) < 3 and abs(d['beta']) < 3]
        stall_data = [d for d in valid_data if abs(d['alpha'] - ps_params.alpha_stall_deg) < 3]
        deep_stall = [d for d in valid_data if abs(d['alpha']) > 45 and abs(d['beta']) < 5]
        
        if trim_data:
            t = trim_data[0]
            print(f"\n  At Trim (α≈0°, β≈0°):")
            print(f"    CL = {t['CL']:.4f}, CD = {t['CD']:.4f}, Cm = {t['Cm']:.4f}")
            print(f"    w_blend = {t['w_blend']:.3f} (attached)")
            print(f"    η_ctrl = {t['eta_ctrl']:.3f} (full control)")
            print(f"    Cma = {t.get('Cma', 0):.4f} /rad")
        
        if stall_data:
            s = stall_data[0]
            print(f"\n  At Stall (α≈{ps_params.alpha_stall_deg}°):")
            print(f"    CL = {s['CL']:.4f}, CD = {s['CD']:.4f}")
            print(f"    w_blend = {s['w_blend']:.3f} (transitioning)")
            print(f"    η_ctrl = {s['eta_ctrl']:.3f} (reduced control)")
        
        if deep_stall:
            d = deep_stall[0]
            print(f"\n  Deep Stall (α≈{d['alpha']:.0f}°):")
            print(f"    CL = {d['CL']:.4f}, CD = {d['CD']:.4f}")
            print(f"    w_blend = {d['w_blend']:.3f} (separated)")
            print(f"    η_ctrl = {d['eta_ctrl']:.3f} (minimal control)")
    
    n_derivs = analyzer.n_outputs * analyzer.n_inputs
    print(f"\n  Performance:")
    print(f"    Total time: {total_elapsed:.1f}s")
    print(f"    Grid points: {len(grid_results)}")
    print(f"    Derivatives per point: {n_derivs}")
    
    print(f"\n  Control Model:")
    if analyzer.control_surface_names:
        print(f"    Surfaces: {analyzer.control_surface_names}")
        print(f"    δ_max: {args.delta_max}° (for saturation)")
        print(f"    Formula: C(α,δ) ≈ C(α,0) + η(α) * Cδ * sat(δ/δ_max)")
    else:
        print(f"    No control surfaces detected")
    
    print(f"\n  Outputs:")
    print(f"    JSON: {output_path}")
    print(f"    CSV:  {csv_path}")
    
    print("\n" + "="*70)
    print("Complete! (With Post-Stall Blending Model)")
    print("="*70)


if __name__ == "__main__":
    main()
