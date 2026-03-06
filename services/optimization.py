from typing import List, Dict, Any
import numpy as np
from core.state import Project
from services.aero_analysis import AeroAnalysisService

class OptimizationService:
    def __init__(self):
        self.aero_service = AeroAnalysisService()

    def run_twist_optimization(self, project: Project) -> Project:
        """
        Run the twist optimization loop.
        1. Uses AeroSandboxService to calculate optimized twist for target lift distribution.
        2. Calculates performance metrics and polars.
        3. Updates project state.
        """
        print("Starting twist optimization...")
        
        try:
            # Initialize service with the project
            service = self.aero_service
            # Note: AeroAnalysisService in optimization.py seems to be the one from services.aero_analysis
            # But we need the one from services.geometry which has the wing logic!
            # Let's fix the import in the file header first or import locally.
            
            from services.geometry import AeroSandboxService as GeometryService
            geo_service = GeometryService(project)
            
            print("Calculating optimal twist distribution...")
            # Calculate twist required for the target lift distribution (Elliptical/Bell)
            optimized_twist = geo_service.calculate_optimized_twist()
            
            # Update Project
            project.wing.optimized_twist_deg = optimized_twist
            
            # Calculate and store performance metrics
            print("Calculating performance metrics...")
            try:
                metrics = geo_service.calculate_performance_metrics()
                project.analysis.performance_metrics = metrics
                print(f"  Cruise: V={metrics.get('cruise_velocity', 0):.2f} m/s, L/D={metrics.get('cruise_l_d', 0):.1f}")
            except Exception as e:
                print(f"  Warning: Could not calculate metrics: {e}")
            
            # Calculate and store performance polars
            print("Calculating aerodynamic polars...")
            try:
                polars = geo_service.calculate_performance_polars(alpha_range=(-5, 15), num_points=21)
                project.analysis.polars = polars
                print(f"  Polars computed for {len(polars.get('alpha', []))} alpha points")
            except Exception as e:
                print(f"  Warning: Could not calculate polars: {e}")
            
            # Store x_cg location
            try:
                wing = geo_service.build_wing()
                x_np = wing.aerodynamic_center()[0]
                mac = wing.mean_aerodynamic_chord()
                static_margin = project.wing.twist_trim.static_margin_percent
                project.analysis.x_cg = x_np - (static_margin / 100.0) * mac
            except Exception as e:
                print(f"  Warning: Could not calculate x_cg: {e}")
            
            print("Optimization complete. Twist distribution and analysis data updated.")
            return project
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            raise e
