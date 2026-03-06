"""
Run full analysis on a project file and save all results to JSON.

This script:
1. Loads a project from JSON
2. Runs aerodynamic polars (CL/CD/CM vs alpha for cruise and takeoff)
3. Runs performance metrics calculation
4. Runs structural analysis (Timoshenko beam with full buckling checks)
5. Saves all results back to the JSON file

Usage:
    python run_full_analysis.py IntendedValidation.json
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.state import Project
from services.geometry import AeroSandboxService


def run_full_analysis(filepath: str, load_factor: float = 2.5):
    """
    Run complete analysis suite on a project file.
    
    Args:
        filepath: Path to project JSON file
        load_factor: Load factor for structural analysis (default 2.5g)
    """
    print(f"Loading project from: {filepath}")
    project = Project.load(filepath)
    
    print(f"Project: {project.wing.name}")
    print(f"  Wing Area: {project.wing.planform.wing_area_m2:.2f} m²")
    print(f"  Aspect Ratio: {project.wing.planform.aspect_ratio:.2f}")
    print(f"  Gross Weight: {project.wing.twist_trim.gross_takeoff_weight_kg:.1f} kg")
    
    # Create service
    service = AeroSandboxService(project.wing)
    
    # 1. Calculate performance metrics
    print("\n[1/3] Calculating performance metrics...")
    metrics = service.calculate_performance_metrics()
    project.analysis.performance_metrics = metrics
    print(f"  Cruise: V={metrics['cruise_velocity']:.2f} m/s, CL={metrics['cruise_cl']:.3f}, L/D={metrics['cruise_l_d']:.1f}")
    print(f"  Takeoff: V={metrics['takeoff_velocity']:.2f} m/s, CL={metrics['takeoff_cl']:.3f}")
    
    # 2. Calculate full polars
    print("\n[2/3] Calculating aerodynamic polars...")
    polars = service.calculate_performance_polars(alpha_range=(-5, 15), num_points=21)
    project.analysis.polars = polars
    print(f"  Alpha range: {polars['alpha'][0]:.1f}° to {polars['alpha'][-1]:.1f}° ({len(polars['alpha'])} points)")
    print(f"  Cruise CL range: {min(polars['cruise']['CL']):.3f} to {max(polars['cruise']['CL']):.3f}")
    print(f"  Cruise CD range: {min(polars['cruise']['CD']):.4f} to {max(polars['cruise']['CD']):.4f}")
    
    # 3. Run structural analysis
    print(f"\n[3/3] Running structural analysis (load factor = {load_factor}g)...")
    try:
        result_dict = service.run_aerostructural_analysis(
            flight_condition={'load_factor': load_factor}
        )
        
        if 'error' in result_dict:
            print(f"  WARNING: Structural analysis error: {result_dict['error']}")
            project.analysis.structural_analysis = result_dict
        else:
            # Store the full result dict (already serialized via as_dict())
            project.analysis.structural_analysis = result_dict
            struct_data = result_dict.get('structure', {})
            print(f"  Mass: {struct_data.get('mass_kg', 0):.3f} kg")
            print(f"  Tip deflection: {struct_data.get('tip_deflection_m', 0)*1000:.1f} mm")
            print(f"  Max stress: {struct_data.get('max_stress_MPa', 0):.1f} MPa")
            print(f"  Min buckling margin: {struct_data.get('min_buckling_margin', 0):.2f}")
            print(f"  Stress margin: {struct_data.get('stress_margin', 0):.2f}")
            print(f"  Feasible: {result_dict.get('feasible', False)}")
    except Exception as e:
        print(f"  WARNING: Structural analysis failed: {e}")
        project.analysis.structural_analysis = {"error": str(e)}
    
    # 4. Store x_cg if not already set
    if project.analysis.x_cg is None:
        wing = service.build_wing()
        x_np = wing.aerodynamic_center()[0]
        mac = wing.mean_aerodynamic_chord()
        static_margin = project.wing.twist_trim.static_margin_percent
        project.analysis.x_cg = x_np - (static_margin / 100.0) * mac
        print(f"\n  CG location: {project.analysis.x_cg:.4f} m")
    
    # 5. Save back to file
    print(f"\nSaving results to: {filepath}")
    project.save(filepath)
    print("Done!")
    
    return project


if __name__ == "__main__":
    if len(sys.argv) < 2:
        filepath = "IntendedValidation.json"
    else:
        filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    run_full_analysis(filepath)
