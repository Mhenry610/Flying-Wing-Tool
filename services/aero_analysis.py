"""
AeroAnalysis Service
Extracted from AeroAnalysis/nf_sweep_gui.py

Provides services for:
1. Airfoil creation/loading (ASB)
2. NeuralFoil evaluation
3. Kulfan (CST) optimization
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
from core.naca_generator.naca456 import generate_naca_airfoil

# Lazy import helper for aerosandbox
def _import_asb():
    try:
        import aerosandbox as asb
        return asb
    except ImportError:
        raise RuntimeError("AeroSandbox not installed. Please run: pip install aerosandbox neuralfoil")

@dataclass
class OptimizationObjectives:
    max_ld: bool = True
    min_cd: bool = False
    target_cl: Optional[float] = None
    target_cm: Optional[float] = None
    
    w_max_ld: float = 1.0
    w_min_cd: float = 1.0
    w_target_cl: float = 1.0
    w_target_cm: float = 1.0

@dataclass
class OptimizationConstraints:
    min_thickness: float = 0.012  # Min t/c
    wiggliness_cap: float = 2.0   # Multiplier on initial wiggliness
    optimize_alpha: bool = True
    alpha_bounds: Tuple[float, float] = (-20.0, 20.0)

class AeroAnalysisService:
    def __init__(self):
        self.asb = _import_asb()

    def make_airfoil(self, user_str: str):
        """
        Try to build an asb.Airfoil from a name or a .dat filepath.
        Includes NACA generation backup.
        """
        asb = self.asb
        # If empty, default
        if not user_str.strip():
            return asb.Airfoil("naca2412")

        s = user_str.strip()

        # Try new NACA generator first
        try:
            clean_name = s.lower().replace("naca", "").strip()
            x, y = generate_naca_airfoil(clean_name, n_points=200)
            coords = np.column_stack((x, y))
            return asb.Airfoil(s, coordinates=coords)
        except Exception:
            pass

        # Try name through ASB
        try:
            return asb.Airfoil(s)
        except Exception:
            pass

        # Try file path through asb directly
        try:
            return asb.Airfoil(file=s)
        except Exception:
            pass

        # Fallback: naive .dat parser for 2-column coordinate files
        if os.path.isfile(s):
            try:
                # Read all floats from lines containing at least 2 numbers
                xs, ys = [], []
                with open(s, "r") as f:
                    for line in f:
                        parts = line.strip().replace(",", " ").split()
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                xs.append(x)
                                ys.append(y)
                            except ValueError:
                                continue
                coords = np.column_stack([xs, ys])
                if len(coords) < 10:
                    raise ValueError("Too few numeric points parsed from file.")
                return asb.Airfoil(name=os.path.basename(s), coordinates=coords)
            except Exception as e:
                raise RuntimeError(f"Failed to parse airfoil from file:\n{s}\nError: {e}")

        raise RuntimeError(f"Could not resolve airfoil '{s}'. Use a known name (e.g., 'naca2412') or a valid .dat path.")

    def eval_neuralfoil(self, airfoil, alpha_deg_grid, Re_grid, Mach=0.0):
        """
        Evaluate NeuralFoil over a grid.
        """
        alpha_arr = np.asarray(alpha_deg_grid)
        Re_arr = np.asarray(Re_grid)
        try:
            a_b, r_b = np.broadcast_arrays(alpha_arr, Re_arr)
        except Exception:
            if alpha_arr.ndim == 1 and Re_arr.ndim == 1:
                r_b, a_b = np.meshgrid(Re_arr, alpha_arr)
            else:
                raise RuntimeError(f"Alpha shape {alpha_arr.shape} and Re shape {Re_arr.shape} are not broadcastable.")

        a_flat = a_b.ravel()
        r_flat = r_b.ravel()

        aero = None
        # Prefer using NeuralFoil directly
        try:
            import neuralfoil as nf
        except ImportError:
            nf = None

        if nf is not None:
            if hasattr(nf, "get_aero_from_airfoil"):
                try:
                    aero = nf.get_aero_from_airfoil(airfoil=airfoil, alpha=a_flat, Re=r_flat)
                except Exception:
                    aero = None
            if aero is None and hasattr(nf, "get_aero_from_coordinates"):
                try:
                    coords = getattr(airfoil, "coordinates", None)
                    if coords is None:
                        try:
                            coords = np.column_stack([airfoil.get_x(), airfoil.get_y()])
                        except Exception:
                            coords = None
                    if coords is not None:
                        aero = nf.get_aero_from_coordinates(coordinates=coords, alpha=a_flat, Re=r_flat)
                except Exception:
                    aero = None

        # Fallback: AeroSandbox wrapper
        if aero is None:
            Mach_arr = np.full(a_flat.shape, float(Mach))
            try:
                aero = airfoil.get_aero_from_neuralfoil(alpha=a_flat, Re=r_flat, Mach=Mach_arr)
            except TypeError:
                try:
                    aero = airfoil.get_aero_from_neuralfoil(alpha=a_flat, Re=r_flat, mach=Mach_arr)
                except Exception as e:
                    raise RuntimeError(f"NeuralFoil evaluation failed via AeroSandbox. {e}")

        # Normalize keys
        def k(d, *names):
            for n in names:
                if n in d:
                    return np.asarray(d[n])
            raise KeyError(f"Missing keys {names} in NeuralFoil result: {list(d.keys())}")

        CL = k(aero, "CL", "Cl", "cl")
        CD = k(aero, "CD", "Cd", "cd")
        try:
            CM = k(aero, "CM", "Cm", "cm")
        except KeyError:
            CM = np.zeros_like(CL)

        shp = a_b.shape
        return {"CL": CL.reshape(shp), "CD": CD.reshape(shp), "CM": CM.reshape(shp)}

    def optimize_kulfan(self, 
                        initial_airfoil, 
                        alpha: float, 
                        Re: float, 
                        Mach: float,
                        objectives: OptimizationObjectives,
                        constraints: OptimizationConstraints) -> Tuple[Any, Dict[str, float]]:
        """
        Run Kulfan (CST) optimization on the given airfoil.
        Returns (optimized_airfoil, performance_dict)
        """
        asb = self.asb
        
        # Convert to Kulfan if needed
        try:
            if isinstance(initial_airfoil, asb.KulfanAirfoil):
                kulfan_init = initial_airfoil
            else:
                kulfan_init = initial_airfoil.to_kulfan_airfoil()
        except Exception as e:
            raise RuntimeError(f"Kulfan conversion failed: {e}")

        # Initial guess
        try:
            import numpy as _np
            lw0 = _np.resize(kulfan_init.lower_weights, 8)
            uw0 = _np.resize(kulfan_init.upper_weights, 8)
            le0 = float(getattr(kulfan_init, 'leading_edge_weight', 0.0))
            te0 = float(getattr(kulfan_init, 'TE_thickness', 0.0))
        except Exception:
            lw0 = np.zeros(8)
            uw0 = np.zeros(8)
            le0 = 0.0
            te0 = 0.0

        # Setup Opti
        opti = asb.Opti()
        lw = opti.variable(init_guess=lw0)
        uw = opti.variable(init_guess=uw0)
        le = opti.variable(init_guess=le0)
        te = te0 # Fixed TE thickness for stability
        
        if constraints.optimize_alpha:
            alpha_var = opti.variable(init_guess=alpha, lower_bound=constraints.alpha_bounds[0], upper_bound=constraints.alpha_bounds[1])
        else:
            alpha_var = alpha

        optimized_airfoil = asb.KulfanAirfoil(
            lower_weights=lw,
            upper_weights=uw,
            leading_edge_weight=le,
            TE_thickness=te,
        )

        # Aero evaluation
        aero = optimized_airfoil.get_aero_from_neuralfoil(
            alpha=alpha_var,
            Re=Re,
            mach=Mach
        )
        CL, CD, CM = aero["CL"], aero["CD"], aero["CM"]

        # Constraints
        # Positive thickness
        xs = np.linspace(0.01, 0.99, 50)
        opti.subject_to(optimized_airfoil.local_thickness(xs) > 0)
        
        # Min thickness at spar
        if constraints.min_thickness > 0:
            opti.subject_to(optimized_airfoil.local_thickness(np.array([0.33])) >= constraints.min_thickness)
            
        # Confidence
        if "analysis_confidence" in aero:
            opti.subject_to(aero["analysis_confidence"] > 0.85)

        # Wiggliness
        def _wiggle_np(arr):
            arr = np.asarray(arr).ravel()
            if arr.size < 3: return 0.0
            d2 = arr[2:] - 2 * arr[1:-1] + arr[:-2]
            return float(np.sum(d2 ** 2))

        def wigglesum(arr):
            try:
                import casadi as cas
                d2 = arr[2:] - 2 * arr[1:-1] + arr[:-2]
                return cas.sumsqr(d2)
            except Exception:
                return _wiggle_np(arr)

        wig0 = _wiggle_np(lw0) + _wiggle_np(uw0)
        wig = wigglesum(lw) + wigglesum(uw)
        if wig0 > 1e-6:
            opti.subject_to(wig <= constraints.wiggliness_cap * wig0)

        # Objective
        eps = 1e-6
        obj = 0
        if objectives.max_ld:
            obj += (-objectives.w_max_ld) * (CL / (CD + eps))
        if objectives.min_cd:
            obj += objectives.w_min_cd * CD
        if objectives.target_cl is not None:
            obj += objectives.w_target_cl * (CL - objectives.target_cl) ** 2
        if objectives.target_cm is not None:
            obj += objectives.w_target_cm * (CM - objectives.target_cm) ** 2
        
        obj += 0.001 * wig
        opti.minimize(obj)

        # Solve
        try:
            sol = opti.solve()
        except Exception:
            # Fallback logic could go here, but for now we raise
            raise RuntimeError("Optimization failed to converge")

        # Extract results
        lw_b = opti.debug.value(lw)
        uw_b = opti.debug.value(uw)
        le_b = opti.debug.value(le)
        alpha_b = float(opti.debug.value(alpha_var)) if constraints.optimize_alpha else alpha

        af_best = asb.KulfanAirfoil(
            lower_weights=lw_b,
            upper_weights=uw_b,
            leading_edge_weight=le_b,
            TE_thickness=te,
        )
        
        # Final aero
        aero_final = af_best.get_aero_from_neuralfoil(
            alpha=alpha_b,
            Re=Re,
            mach=Mach
        )
        
        def _first(x):
            try:
                return float(np.ravel(np.array(x))[0])
            except Exception:
                return float(x)

        results = {
            "CL": _first(aero_final["CL"]),
            "CD": max(_first(aero_final["CD"]), 1e-6),
            "CM": _first(aero_final["CM"]),
            "alpha": alpha_b,
            "Re": Re,
            "Mach": Mach
        }

        return af_best, results
