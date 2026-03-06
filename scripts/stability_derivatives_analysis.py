"""
Comprehensive Aerodynamic Analysis Script (Multithreaded)

Generates polars, stability derivatives, and damping coefficients for the geometry
defined in IntendedValidation.json across a wide range of flight conditions.

Sweep ranges:
- Pitch (alpha): -90 to +90 degrees
- Sideslip (beta): -90 to +90 degrees  
- Roll/Pitch/Yaw rates for damping derivatives

Outputs:
- Full aerodynamic polars (CL, CD, CY, Cl, Cm, Cn vs alpha/beta)
- Static stability derivatives (CLa, Cma, Cnb, etc.)
- Dynamic damping derivatives (Clp, Cmq, Cnr, etc.)
- Results saved to JSON and CSV for post-processing

Usage:
    python scripts/stability_derivatives_analysis.py [--airspeed 20] [--output results.json]
    python scripts/stability_derivatives_analysis.py --workers 8  # Use 8 threads
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aerosandbox as asb
from core.state import Project
from services.geometry import AeroSandboxService


# Thread-local storage for progress tracking
_progress_lock = threading.Lock()
_progress_counter = 0


def reset_progress():
    global _progress_counter
    with _progress_lock:
        _progress_counter = 0


def increment_progress() -> int:
    global _progress_counter
    with _progress_lock:
        _progress_counter += 1
        return _progress_counter


@dataclass
class StabilityDerivatives:
    """Container for stability derivatives at a single flight condition."""
    alpha: float  # deg
    beta: float   # deg
    airspeed: float  # m/s
    
    # Force coefficients
    CL: float
    CD: float
    CY: float
    
    # Moment coefficients
    Cl: float  # Roll
    Cm: float  # Pitch
    Cn: float  # Yaw
    
    # Static derivatives (w.r.t. alpha, per rad)
    CLa: float = 0.0
    CDa: float = 0.0
    CYa: float = 0.0
    Cla: float = 0.0
    Cma: float = 0.0
    Cna: float = 0.0
    
    # Static derivatives (w.r.t. beta, per rad)
    CLb: float = 0.0
    CDb: float = 0.0
    CYb: float = 0.0
    Clb: float = 0.0
    Cmb: float = 0.0
    Cnb: float = 0.0
    
    # Dynamic/damping derivatives (w.r.t. normalized rates)
    CLp: float = 0.0  # Roll rate
    CDp: float = 0.0
    CYp: float = 0.0
    Clp: float = 0.0  # Roll damping
    Cmp: float = 0.0
    Cnp: float = 0.0  # Yaw due to roll
    
    CLq: float = 0.0  # Pitch rate
    CDq: float = 0.0
    CYq: float = 0.0
    Clq: float = 0.0
    Cmq: float = 0.0  # Pitch damping
    Cnq: float = 0.0
    
    CLr: float = 0.0  # Yaw rate
    CDr: float = 0.0
    CYr: float = 0.0
    Clr: float = 0.0  # Roll due to yaw
    Cmr: float = 0.0
    Cnr: float = 0.0  # Yaw damping
    
    # Neutral point
    x_np: float = 0.0


def build_airplane(service: AeroSandboxService, x_cg: float) -> Tuple[asb.Airplane, float, float]:
    """
    Build an AeroSandbox Airplane object from the geometry service.
    
    Returns:
        (airplane, c_ref, b_ref)
    """
    wing = service.build_wing()
    
    airplane = asb.Airplane(
        name=service.wing_project.name,
        wings=[wing],
        xyz_ref=[x_cg, 0.0, 0.0],
    )
    
    c_ref = wing.mean_aerodynamic_chord()
    b_ref = wing.span()
    
    return airplane, c_ref, b_ref


def run_single_aerobuildup(
    airplane: asb.Airplane,
    atmosphere: asb.Atmosphere,
    airspeed: float,
    alpha: float,
    beta: float,
    compute_derivatives: bool = True,
) -> Dict[str, float]:
    """
    Run AeroBuildup analysis for a single flight condition.
    Thread-safe wrapper for parallel execution.
    
    Args:
        airplane: AeroSandbox Airplane object
        atmosphere: Atmosphere model
        airspeed: True airspeed [m/s]
        alpha: Angle of attack [deg]
        beta: Sideslip angle [deg]
        compute_derivatives: If True, compute stability derivatives
        
    Returns:
        Dict with all coefficients and derivatives
    """
    op_point = asb.OperatingPoint(
        atmosphere=atmosphere,
        velocity=max(airspeed, 0.1),
        alpha=alpha,
        beta=beta,
        p=0.0,
        q=0.0,
        r=0.0,
    )
    
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
    )
    
    if compute_derivatives:
        result = aero.run_with_stability_derivatives(
            alpha=True,
            beta=True,
            p=True,
            q=True,
            r=True,
        )
    else:
        result = aero.run()
    
    # Extract values, handling numpy arrays
    def extract(key, default=0.0):
        val = result.get(key, default)
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val) if val is not None else default
    
    output = {k: extract(k) for k in result.keys() if not k.endswith('_components')}
    output['alpha'] = alpha
    output['beta'] = beta
    output['airspeed'] = airspeed
    
    return output


def _run_alpha_point(args: Tuple) -> Dict:
    """Worker function for parallel alpha sweep."""
    airplane, atmosphere, airspeed, alpha, beta, total_points = args
    try:
        result = run_single_aerobuildup(
            airplane, atmosphere, airspeed,
            alpha=alpha, beta=beta,
            compute_derivatives=True,
        )
        count = increment_progress()
        if count % 10 == 0 or count == 1:
            print(f"  [{count}/{total_points}] α={alpha:6.1f}°: CL={result.get('CL', 0):7.4f}, CD={result.get('CD', 0):7.4f}, Cm={result.get('Cm', 0):7.4f}")
        return result
    except Exception as e:
        increment_progress()
        return {
            'alpha': alpha, 'beta': beta, 'airspeed': airspeed,
            'CL': np.nan, 'CD': np.nan, 'CY': np.nan,
            'Cl': np.nan, 'Cm': np.nan, 'Cn': np.nan,
            'error': str(e),
        }


def _run_beta_point(args: Tuple) -> Dict:
    """Worker function for parallel beta sweep."""
    airplane, atmosphere, airspeed, alpha, beta, total_points = args
    try:
        result = run_single_aerobuildup(
            airplane, atmosphere, airspeed,
            alpha=alpha, beta=beta,
            compute_derivatives=True,
        )
        count = increment_progress()
        if count % 10 == 0 or count == 1:
            print(f"  [{count}/{total_points}] β={beta:6.1f}°: CY={result.get('CY', 0):7.4f}, Cl={result.get('Cl', 0):7.4f}, Cn={result.get('Cn', 0):7.4f}")
        return result
    except Exception as e:
        increment_progress()
        return {
            'alpha': alpha, 'beta': beta, 'airspeed': airspeed,
            'CL': np.nan, 'CD': np.nan, 'CY': np.nan,
            'Cl': np.nan, 'Cm': np.nan, 'Cn': np.nan,
            'error': str(e),
        }


def sweep_alpha_parallel(
    airplane: asb.Airplane,
    atmosphere: asb.Atmosphere,
    airspeed: float,
    alpha_range: Tuple[float, float, int] = (-90, 90, 37),
    beta: float = 0.0,
    max_workers: int = 4,
) -> List[Dict]:
    """
    Sweep angle of attack and collect polars (multithreaded).
    
    Args:
        airplane: AeroSandbox Airplane
        atmosphere: Atmosphere model
        airspeed: Airspeed [m/s]
        alpha_range: (min, max, num_points)
        beta: Fixed sideslip [deg]
        max_workers: Number of parallel threads
        
    Returns:
        List of result dicts for each alpha (sorted by alpha)
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], int(alpha_range[2]))
    total_points = len(alphas)
    
    print(f"\nSweeping alpha from {alpha_range[0]}° to {alpha_range[1]}° ({total_points} points), beta={beta}°")
    print(f"  Using {max_workers} parallel workers...")
    
    reset_progress()
    
    # Prepare arguments for each point
    args_list = [
        (airplane, atmosphere, airspeed, alpha, beta, total_points)
        for alpha in alphas
    ]
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_alpha = {
            executor.submit(_run_alpha_point, args): args[3]  # args[3] is alpha
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_alpha):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    print(f"  Completed {total_points} points in {elapsed:.1f}s ({total_points/elapsed:.1f} pts/sec)")
    
    # Sort by alpha to maintain order
    results.sort(key=lambda x: x.get('alpha', 0))
    
    return results


def sweep_beta_parallel(
    airplane: asb.Airplane,
    atmosphere: asb.Atmosphere,
    airspeed: float,
    beta_range: Tuple[float, float, int] = (-90, 90, 37),
    alpha: float = 0.0,
    max_workers: int = 4,
) -> List[Dict]:
    """
    Sweep sideslip angle and collect polars (multithreaded).
    
    Args:
        airplane: AeroSandbox Airplane
        atmosphere: Atmosphere model
        airspeed: Airspeed [m/s]
        beta_range: (min, max, num_points)
        alpha: Fixed angle of attack [deg]
        max_workers: Number of parallel threads
        
    Returns:
        List of result dicts for each beta (sorted by beta)
    """
    betas = np.linspace(beta_range[0], beta_range[1], int(beta_range[2]))
    total_points = len(betas)
    
    print(f"\nSweeping beta from {beta_range[0]}° to {beta_range[1]}° ({total_points} points), alpha={alpha}°")
    print(f"  Using {max_workers} parallel workers...")
    
    reset_progress()
    
    # Prepare arguments for each point
    args_list = [
        (airplane, atmosphere, airspeed, alpha, beta, total_points)
        for beta in betas
    ]
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_beta = {
            executor.submit(_run_beta_point, args): args[4]  # args[4] is beta
            for args in args_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_beta):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    print(f"  Completed {total_points} points in {elapsed:.1f}s ({total_points/elapsed:.1f} pts/sec)")
    
    # Sort by beta to maintain order
    results.sort(key=lambda x: x.get('beta', 0))
    
    return results


def _run_damping_point(args: Tuple) -> Dict:
    """Worker function for parallel damping derivative computation."""
    airplane, atmosphere, airspeed, alpha, beta, c_ref, b_ref = args
    
    V = max(airspeed, 0.1)
    
    # Perturbation magnitudes (in normalized rate units)
    d_phat = 0.001
    d_qhat = 0.001
    d_rhat = 0.001
    
    # Convert to actual rates (rad/s)
    dp = d_phat * (2 * V / b_ref)
    dq = d_qhat * (2 * V / c_ref)
    dr = d_rhat * (2 * V / b_ref)
    
    def run_at_rates(p, q, r):
        op = asb.OperatingPoint(
            atmosphere=atmosphere,
            velocity=V,
            alpha=alpha,
            beta=beta,
            p=p,
            q=q,
            r=r,
        )
        aero = asb.AeroBuildup(airplane=airplane, op_point=op)
        res = aero.run()
        
        def extract(key):
            val = res.get(key, 0.0)
            if hasattr(val, 'item'):
                return float(val.item())
            return float(val) if val is not None else 0.0
        
        return {
            'CL': extract('CL'), 'CD': extract('CD'), 'CY': extract('CY'),
            'Cl': extract('Cl'), 'Cm': extract('Cm'), 'Cn': extract('Cn'),
        }
    
    # Baseline and perturbed cases
    base = run_at_rates(0, 0, 0)
    p_plus = run_at_rates(dp, 0, 0)
    q_plus = run_at_rates(0, dq, 0)
    r_plus = run_at_rates(0, 0, dr)
    
    # Finite difference derivatives
    derivs = {'alpha': alpha}
    for coef in ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn']:
        derivs[f'{coef}p'] = (p_plus[coef] - base[coef]) / d_phat
        derivs[f'{coef}q'] = (q_plus[coef] - base[coef]) / d_qhat
        derivs[f'{coef}r'] = (r_plus[coef] - base[coef]) / d_rhat
    
    return derivs


def compute_damping_derivatives_parallel(
    airplane: asb.Airplane,
    atmosphere: asb.Atmosphere,
    airspeed: float,
    alpha_values: List[float],
    beta: float = 0.0,
    c_ref: float = 1.0,
    b_ref: float = 1.0,
    max_workers: int = 4,
) -> List[Dict[str, float]]:
    """
    Compute damping derivatives at multiple alpha values (multithreaded).
    
    Args:
        airplane: AeroSandbox Airplane
        atmosphere: Atmosphere model
        airspeed: Airspeed [m/s]
        alpha_values: List of alpha values to evaluate
        beta: Fixed sideslip [deg]
        c_ref: Reference chord [m]
        b_ref: Reference span [m]
        max_workers: Number of parallel threads
        
    Returns:
        List of dicts with damping derivatives for each alpha
    """
    print(f"\nComputing damping derivatives at {len(alpha_values)} alpha values...")
    print(f"  Using {max_workers} parallel workers...")
    
    args_list = [
        (airplane, atmosphere, airspeed, alpha, beta, c_ref, b_ref)
        for alpha in alpha_values
    ]
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_alpha = {
            executor.submit(_run_damping_point, args): args[3]
            for args in args_list
        }
        
        for future in as_completed(future_to_alpha):
            result = future.result()
            results.append(result)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")
    
    # Sort by alpha
    results.sort(key=lambda x: x.get('alpha', 0))
    
    return results


def compute_damping_derivatives(
    airplane: asb.Airplane,
    atmosphere: asb.Atmosphere,
    airspeed: float,
    alpha: float = 0.0,
    beta: float = 0.0,
    c_ref: float = 1.0,
    b_ref: float = 1.0,
) -> Dict[str, float]:
    """
    Compute damping derivatives at a single condition (non-parallel version).
    """
    return _run_damping_point((airplane, atmosphere, airspeed, alpha, beta, c_ref, b_ref))


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
    """Save tabular results to CSV."""
    import csv
    
    if not data:
        return
    
    # Get all keys from first record
    keys = list(data[0].keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"CSV saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive stability derivatives analysis (multithreaded)")
    parser.add_argument('--project', type=str, default='IntendedValidation.json',
                        help='Path to project JSON file')
    parser.add_argument('--airspeed', type=float, default=None,
                        help='Airspeed for analysis [m/s]. If not specified, uses cruise speed.')
    parser.add_argument('--alpha-range', type=str, default='-90,90,37',
                        help='Alpha sweep range: min,max,num_points')
    parser.add_argument('--beta-range', type=str, default='-90,90,37',
                        help='Beta sweep range: min,max,num_points')
    parser.add_argument('--output', type=str, default='stability_analysis_results.json',
                        help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--skip-derivatives', action='store_true',
                        help='Skip computing stability derivatives (faster)')
    
    args = parser.parse_args()
    
    # Determine worker count
    if args.workers is None:
        import multiprocessing
        max_workers = min(multiprocessing.cpu_count(), 16)  # Cap at 16
    else:
        max_workers = args.workers
    
    # Parse sweep ranges
    alpha_range = tuple(map(float, args.alpha_range.split(',')))
    alpha_range = (alpha_range[0], alpha_range[1], int(alpha_range[2]))
    
    beta_range = tuple(map(float, args.beta_range.split(',')))
    beta_range = (beta_range[0], beta_range[1], int(beta_range[2]))
    
    # Load project
    project_path = project_root / args.project
    if not project_path.exists():
        print(f"Error: Project file not found: {project_path}")
        sys.exit(1)
    
    print(f"Loading project from: {project_path}")
    project = Project.load(str(project_path))
    
    # Create geometry service
    service = AeroSandboxService(project.wing)
    atmosphere = service.atmosphere
    
    # Determine airspeed
    if args.airspeed is not None:
        airspeed = args.airspeed
    else:
        airspeed = service.design_velocity()
    
    print(f"\n{'='*60}")
    print("MULTITHREADED STABILITY DERIVATIVES ANALYSIS")
    print(f"{'='*60}")
    print(f"\n=== Analysis Configuration ===")
    print(f"  Project: {project.wing.name}")
    print(f"  Wing Area: {project.wing.planform.wing_area_m2:.2f} m²")
    print(f"  Aspect Ratio: {project.wing.planform.aspect_ratio:.2f}")
    print(f"  Airspeed: {airspeed:.2f} m/s")
    print(f"  Alpha range: {alpha_range[0]}° to {alpha_range[1]}° ({int(alpha_range[2])} points)")
    print(f"  Beta range: {beta_range[0]}° to {beta_range[1]}° ({int(beta_range[2])} points)")
    print(f"  Parallel workers: {max_workers}")
    
    # Get CG location
    wing = service.build_wing()
    x_np = wing.aerodynamic_center()[0]
    mac = wing.mean_aerodynamic_chord()
    static_margin = project.wing.twist_trim.static_margin_percent
    x_cg = x_np - (static_margin / 100.0) * mac
    
    print(f"\n=== Reference Data ===")
    print(f"  MAC: {mac:.4f} m")
    print(f"  Span: {wing.span():.4f} m")
    print(f"  Neutral Point: {x_np:.4f} m")
    print(f"  CG (at {static_margin}% SM): {x_cg:.4f} m")
    
    # Build airplane
    airplane, c_ref, b_ref = build_airplane(service, x_cg)
    
    total_start = time.time()
    
    results = {
        'project': project.wing.name,
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
            'span_m': b_ref,
            'x_np_m': x_np,
            'x_cg_m': x_cg,
            'static_margin_percent': static_margin,
        },
        'alpha_sweep': [],
        'beta_sweep': [],
        'damping_derivatives': {},
    }
    
    # Run alpha sweep at beta=0 (PARALLEL)
    print("\n" + "="*60)
    print("ALPHA SWEEP (β=0°) - PARALLEL")
    print("="*60)
    alpha_results = sweep_alpha_parallel(
        airplane, atmosphere, airspeed,
        alpha_range=alpha_range,
        beta=0.0,
        max_workers=max_workers,
    )
    results['alpha_sweep'] = alpha_results
    
    # Run beta sweep at alpha=0 (PARALLEL)
    print("\n" + "="*60)
    print("BETA SWEEP (α=0°) - PARALLEL")
    print("="*60)
    beta_results = sweep_beta_parallel(
        airplane, atmosphere, airspeed,
        beta_range=beta_range,
        alpha=0.0,
        max_workers=max_workers,
    )
    results['beta_sweep'] = beta_results
    
    # Additional beta sweeps at different alpha values (PARALLEL)
    for alpha_test in [5.0, 10.0, -5.0]:
        print(f"\n--- Additional beta sweep at α={alpha_test}° ---")
        additional_beta = sweep_beta_parallel(
            airplane, atmosphere, airspeed,
            beta_range=beta_range,
            alpha=alpha_test,
            max_workers=max_workers,
        )
        results[f'beta_sweep_alpha{int(alpha_test)}'] = additional_beta
    
    # Compute damping derivatives at trim condition
    print("\n" + "="*60)
    print("DAMPING DERIVATIVES - PARALLEL")
    print("="*60)
    
    damping = compute_damping_derivatives(
        airplane, atmosphere, airspeed,
        alpha=0.0, beta=0.0,
        c_ref=c_ref, b_ref=b_ref,
    )
    results['damping_derivatives'] = damping
    
    print("\n  Rate Derivatives (normalized) at α=0°:")
    print(f"    Roll damping  Clp = {damping.get('Clp', 0):8.4f}")
    print(f"    Pitch damping Cmq = {damping.get('Cmq', 0):8.4f}")
    print(f"    Yaw damping   Cnr = {damping.get('Cnr', 0):8.4f}")
    print(f"    Yaw due to roll Cnp = {damping.get('Cnp', 0):8.4f}")
    print(f"    Roll due to yaw Clr = {damping.get('Clr', 0):8.4f}")
    
    # Damping at multiple alpha values (PARALLEL)
    alpha_test_values = list(np.linspace(-20, 20, 9))
    damping_vs_alpha = compute_damping_derivatives_parallel(
        airplane, atmosphere, airspeed,
        alpha_values=alpha_test_values,
        beta=0.0,
        c_ref=c_ref, b_ref=b_ref,
        max_workers=max_workers,
    )
    results['damping_vs_alpha'] = damping_vs_alpha
    
    print("\n  Damping vs Alpha:")
    for d in damping_vs_alpha:
        print(f"    α={d['alpha']:5.1f}°: Clp={d['Clp']:7.4f}, Cmq={d['Cmq']:7.4f}, Cnr={d['Cnr']:7.4f}")
    
    total_elapsed = time.time() - total_start
    
    # Save results
    output_path = project_root / 'scripts' / args.output
    save_results_json(results, str(output_path))
    
    # Also save CSV of alpha sweep for easy plotting
    csv_path = output_path.with_suffix('.csv')
    save_results_csv(alpha_results, str(csv_path))
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find key values from alpha sweep
    alpha_data = results['alpha_sweep']
    valid_data = [d for d in alpha_data if not np.isnan(d.get('CL', np.nan))]
    
    if valid_data:
        # Find CL max and corresponding alpha
        cl_values = [d['CL'] for d in valid_data]
        alpha_values = [d['alpha'] for d in valid_data]
        
        cl_max_idx = np.argmax(cl_values)
        cl_max = cl_values[cl_max_idx]
        alpha_cl_max = alpha_values[cl_max_idx]
        
        # Find zero-lift alpha (interpolate)
        for i in range(len(cl_values) - 1):
            if cl_values[i] * cl_values[i+1] < 0:
                alpha_zero_lift = alpha_values[i] - cl_values[i] * (alpha_values[i+1] - alpha_values[i]) / (cl_values[i+1] - cl_values[i])
                break
        else:
            alpha_zero_lift = 0.0
        
        # Lift curve slope (at alpha=0 or near)
        center_idx = len(valid_data) // 2
        if center_idx > 0:
            dCL = cl_values[center_idx + 1] - cl_values[center_idx - 1]
            dalpha = alpha_values[center_idx + 1] - alpha_values[center_idx - 1]
            CLa = dCL / np.radians(dalpha) if dalpha != 0 else 0
        else:
            CLa = 0
        
        print(f"\n  Lift Characteristics:")
        print(f"    CL_max = {cl_max:.3f} at α = {alpha_cl_max:.1f}°")
        print(f"    α_0L = {alpha_zero_lift:.2f}°")
        print(f"    CL_α = {CLa:.3f} /rad ({CLa * np.pi/180:.4f} /deg)")
        
        # Stability derivatives at alpha=0
        alpha0_data = [d for d in valid_data if abs(d['alpha']) < 3]
        if alpha0_data:
            Cma = alpha0_data[0].get('Cma', 0)
            print(f"\n  Static Stability (at α≈0°):")
            print(f"    Cm_α = {Cma:.4f} /rad")
            print(f"    {'STABLE' if Cma < 0 else 'UNSTABLE'} (longitudinal)")
    
    print(f"\n  Damping Derivatives (at trim):")
    print(f"    Cl_p = {damping.get('Clp', 0):.4f} (roll damping)")
    print(f"    Cm_q = {damping.get('Cmq', 0):.4f} (pitch damping)")
    print(f"    Cn_r = {damping.get('Cnr', 0):.4f} (yaw damping)")
    
    print(f"\n  Total analysis time: {total_elapsed:.1f}s")
    
    # Estimate speedup
    total_points = int(alpha_range[2]) + int(beta_range[2]) * 4 + len(alpha_test_values)
    estimated_serial = total_points * 0.5  # Rough estimate: 0.5s per point serial
    speedup = estimated_serial / total_elapsed if total_elapsed > 0 else 1
    print(f"  Estimated speedup vs serial: ~{speedup:.1f}x")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
