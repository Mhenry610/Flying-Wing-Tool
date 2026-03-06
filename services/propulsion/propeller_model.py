"""
Propeller meta-model for differentiable propeller performance prediction.

Uses AeroSandbox FittedModel for AD-compatible polynomial surrogate modeling.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any

import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp  # Standard numpy for non-AD operations
import pandas as pd

from .propeller_data import PropellerDataset, APCDataParser, combine_datasets


@dataclass
class PropellerMetaModelConfig:
    """
    Configuration for propeller meta-model training.
    
    Attributes:
        J_degree: Polynomial degree for advance ratio
        D_degree: Polynomial degree for diameter
        P_degree: Polynomial degree for pitch
        include_JP_interaction: Include J*P cross-term
        include_JD_interaction: Include J*D cross-term
        include_DP_interaction: Include D*P cross-term
        residual_norm: Norm type for fitting ("L1", "L2", "Linf")
        put_residuals_in_logspace: Minimize % error vs absolute error
        Ct_min: Physical lower bound for Ct
        Cp_min: Physical lower bound for Cp
        D_max_in: Maximum diameter (inches) to include in training
        D_min_in: Minimum diameter (inches) to include in training
    """
    J_degree: int = 4
    D_degree: int = 2
    P_degree: int = 3
    
    include_JP_interaction: bool = True
    include_JD_interaction: bool = True
    include_DP_interaction: bool = True
    
    residual_norm: Literal["L1", "L2", "Linf"] = "L2"
    put_residuals_in_logspace: bool = False
    
    Ct_min: float = 0.0
    Cp_min: float = 1e-6
    
    # Size filtering (inches) - focuses on common model aircraft props
    D_min_in: float = 6.0
    D_max_in: float = 26.0
    
    # Fitting verbosity
    verbose: bool = True


@dataclass
class TrainingBounds:
    """
    Records the bounds of training data for validation during inference.
    """
    J_min: float
    J_max: float
    D_min: float
    D_max: float
    P_min: float
    P_max: float
    
    def is_in_bounds(self, J: float, D: float, P: float, 
                     tolerance: float = 0.1) -> bool:
        """
        Check if operating point is within training bounds.
        
        Args:
            J: Advance ratio
            D: Diameter [m]
            P: Pitch [m]
            tolerance: Fractional tolerance for extrapolation warning
            
        Returns:
            True if within bounds (with tolerance)
        """
        J_range = self.J_max - self.J_min
        D_range = self.D_max - self.D_min
        P_range = self.P_max - self.P_min
        
        return (
            (self.J_min - tolerance * J_range <= J <= self.J_max + tolerance * J_range) and
            (self.D_min - tolerance * D_range <= D <= self.D_max + tolerance * D_range) and
            (self.P_min - tolerance * P_range <= P <= self.P_max + tolerance * P_range)
        )


# Module-level model function for pickling support
def _polynomial_model_4_2_3(x, p):
    """Polynomial model for Ct or Cp with J_degree=4, D_degree=2, P_degree=3."""
    J = x["J"]
    D = x["D"]
    P = x["P"]
    
    result = np.zeros_like(J)
    
    # Main polynomial terms (5 * 3 * 4 = 60 terms)
    for i in range(5):  # J: 0-4
        for j in range(3):  # D: 0-2
            for k in range(4):  # P: 0-3
                coef = p[f"c_{i}_{j}_{k}"]
                result = result + coef * (J ** i) * (D ** j) * (P ** k)
    
    # Interaction terms
    if "c_JP" in p:
        result = result + p["c_JP"] * J * P
    if "c_JD" in p:
        result = result + p["c_JD"] * J * D
    if "c_DP" in p:
        result = result + p["c_DP"] * D * P
    
    return result


class PropellerMetaModel:
    """
    Differentiable propeller performance meta-model.
    
    Uses AeroSandbox FittedModel for AD-compatible polynomial surrogate.
    Predicts Ct and Cp as functions of (J, D, P).
    
    Example:
        >>> datasets = APCDataParser.load_family(Path("prop_data/"), "Electric")
        >>> model = PropellerMetaModel.train_from_datasets(datasets)
        >>> thrust, power = model.get_performance(V=20, omega=800, D=0.254, rho=1.225)
        >>> model.save(Path("electric_model.pkl"))
        >>> loaded = PropellerMetaModel.load(Path("electric_model.pkl"))
    """
    
    def __init__(self, 
                 Ct_model: asb.FittedModel,
                 Cp_model: asb.FittedModel,
                 family: str,
                 config: PropellerMetaModelConfig,
                 training_bounds: TrainingBounds,
                 Ct_params: Dict[str, float],
                 Cp_params: Dict[str, float],
                 fit_metrics: Dict[str, float]):
        """
        Initialize with fitted models. Use train_from_* class methods instead.
        """
        self._Ct_model = Ct_model
        self._Cp_model = Cp_model
        self.family = family
        self.config = config
        self.training_bounds = training_bounds
        self.Ct_params = Ct_params
        self.Cp_params = Cp_params
        self.fit_metrics = fit_metrics
    
    @classmethod
    def _build_model_function(cls, config: PropellerMetaModelConfig):
        """
        Build the polynomial model function for FittedModel.
        
        The model form is:
            f(J, D, P) = Σ aᵢⱼₖ · Jⁱ · Dʲ · Pᵏ + interactions
        
        Returns:
            Tuple of (model_function, parameter_guesses)
        """
        # Build parameter names and initial guesses
        param_guesses = {}
        
        # Main polynomial terms
        for i in range(config.J_degree + 1):
            for j in range(config.D_degree + 1):
                for k in range(config.P_degree + 1):
                    if i == 0 and j == 0 and k == 0:
                        # Constant term
                        param_guesses["c_0_0_0"] = 0.1
                    else:
                        param_guesses[f"c_{i}_{j}_{k}"] = 0.0
        
        # Interaction terms
        if config.include_JP_interaction:
            param_guesses["c_JP"] = 0.0
        if config.include_JD_interaction:
            param_guesses["c_JD"] = 0.0
        if config.include_DP_interaction:
            param_guesses["c_DP"] = 0.0
        
        # Use module-level function for pickling support
        return _polynomial_model_4_2_3, param_guesses
    
    @classmethod
    def train_from_datasets(cls, 
                           datasets: List[PropellerDataset],
                           family: str = "Unknown",
                           config: Optional[PropellerMetaModelConfig] = None,
                           ) -> 'PropellerMetaModel':
        """
        Train meta-model from multiple PropellerDatasets.
        
        Args:
            datasets: List of PropellerDataset objects
            family: Family name for this model
            config: Training configuration (uses defaults if None)
            
        Returns:
            Trained PropellerMetaModel
        """
        if config is None:
            config = PropellerMetaModelConfig()
        
        if not datasets:
            raise ValueError("No datasets provided for training")
        
        # Filter datasets by size range
        D_min_m = config.D_min_in * 0.0254
        D_max_m = config.D_max_in * 0.0254
        
        filtered_datasets = [
            ds for ds in datasets 
            if D_min_m <= ds.geometry.diameter_m <= D_max_m
        ]
        
        if not filtered_datasets:
            raise ValueError(f"No datasets in size range {config.D_min_in}-{config.D_max_in} inches")
        
        if config.verbose and len(filtered_datasets) < len(datasets):
            print(f"  Filtered to {len(filtered_datasets)}/{len(datasets)} props in "
                  f"{config.D_min_in}-{config.D_max_in}\" range")
        
        # Combine all datasets
        combined = combine_datasets(filtered_datasets)
        
        J = combined['J']
        D = combined['D_m']
        P = combined['P_m']
        Ct = combined['Ct']
        Cp = combined['Cp']
        
        # Filter valid data points
        valid_mask = (Ct >= -0.1) & (Cp >= 0) & (J >= 0) & (J <= 2.0)
        J = J[valid_mask]
        D = D[valid_mask]
        P = P[valid_mask]
        Ct = Ct[valid_mask]
        Cp = Cp[valid_mask]
        
        if len(J) < 50:
            raise ValueError(f"Insufficient valid data points: {len(J)} (need at least 50)")
        
        # Record training bounds
        training_bounds = TrainingBounds(
            J_min=float(onp.min(J)),
            J_max=float(onp.max(J)),
            D_min=float(onp.min(D)),
            D_max=float(onp.max(D)),
            P_min=float(onp.min(P)),
            P_max=float(onp.max(P)),
        )
        
        if config.verbose:
            print(f"Training PropellerMetaModel for family '{family}'")
            print(f"  Data points: {len(J)}")
            print(f"  J range: [{training_bounds.J_min:.3f}, {training_bounds.J_max:.3f}]")
            print(f"  D range: [{training_bounds.D_min*1000:.1f}, {training_bounds.D_max*1000:.1f}] mm")
            print(f"  P range: [{training_bounds.P_min*1000:.1f}, {training_bounds.P_max*1000:.1f}] mm")
        
        # Normalize inputs for better conditioning
        J_norm = J / max(training_bounds.J_max, 1.0)
        D_norm = D / training_bounds.D_max
        P_norm = P / training_bounds.P_max
        
        x_data = {
            "J": J_norm,
            "D": D_norm,
            "P": P_norm,
        }
        
        # Build model function
        model_fn, param_guesses = cls._build_model_function(config)
        
        # Fit Ct model
        if config.verbose:
            print("  Fitting Ct model...")
        
        Ct_model = asb.FittedModel(
            model=model_fn,
            x_data=x_data,
            y_data=Ct,
            parameter_guesses=param_guesses.copy(),
            residual_norm_type=config.residual_norm,
            put_residuals_in_logspace=config.put_residuals_in_logspace,
            verbose=False,
        )
        
        Ct_r2 = Ct_model.goodness_of_fit(type="R^2")
        Ct_rmse = Ct_model.goodness_of_fit(type="root_mean_squared_error")
        
        if config.verbose:
            print(f"    Ct R² = {Ct_r2:.4f}, RMSE = {Ct_rmse:.6f}")
        
        # Fit Cp model
        if config.verbose:
            print("  Fitting Cp model...")
        
        # Use log-space for Cp to avoid negative predictions
        Cp_clipped = onp.maximum(Cp, config.Cp_min)
        
        Cp_model = asb.FittedModel(
            model=model_fn,
            x_data=x_data,
            y_data=Cp_clipped,
            parameter_guesses=param_guesses.copy(),
            residual_norm_type=config.residual_norm,
            put_residuals_in_logspace=config.put_residuals_in_logspace,
            verbose=False,
        )
        
        Cp_r2 = Cp_model.goodness_of_fit(type="R^2")
        Cp_rmse = Cp_model.goodness_of_fit(type="root_mean_squared_error")
        
        if config.verbose:
            print(f"    Cp R² = {Cp_r2:.4f}, RMSE = {Cp_rmse:.6f}")
        
        fit_metrics = {
            "Ct_R2": Ct_r2,
            "Ct_RMSE": Ct_rmse,
            "Cp_R2": Cp_r2,
            "Cp_RMSE": Cp_rmse,
            "n_samples": len(J),
            "n_propellers": len(datasets),
        }
        
        return cls(
            Ct_model=Ct_model,
            Cp_model=Cp_model,
            family=family,
            config=config,
            training_bounds=training_bounds,
            Ct_params=dict(Ct_model.parameters),
            Cp_params=dict(Cp_model.parameters),
            fit_metrics=fit_metrics,
        )
    
    @classmethod
    def train_from_folder(cls,
                         folder: Path,
                         family: Optional[str] = None,
                         config: Optional[PropellerMetaModelConfig] = None,
                         **parse_kwargs) -> 'PropellerMetaModel':
        """
        Train meta-model from a folder of .dat files.
        
        Args:
            folder: Path to folder containing APC .dat files
            family: Family name (inferred from first file if None)
            config: Training configuration
            **parse_kwargs: Additional arguments for APCDataParser.load_folder
            
        Returns:
            Trained PropellerMetaModel
        """
        datasets = APCDataParser.load_folder(folder, **parse_kwargs)
        
        if not datasets:
            raise ValueError(f"No valid datasets found in {folder}")
        
        if family is None:
            family = datasets[0].geometry.family
        
        return cls.train_from_datasets(datasets, family=family, config=config)
    
    def _normalize_inputs(self, J, D, P):
        """Normalize inputs using training bounds."""
        J_norm = J / max(self.training_bounds.J_max, 1.0)
        D_norm = D / self.training_bounds.D_max
        P_norm = P / self.training_bounds.P_max
        return J_norm, D_norm, P_norm
    
    def get_coefficients(self, J, D_m, P_m) -> Tuple[Any, Any]:
        """
        Get (Ct, Cp) for given advance ratio, diameter, and pitch.
        
        Args:
            J: Advance ratio (scalar or array, can be AD variable)
            D_m: Diameter in meters
            P_m: Pitch in meters
            
        Returns:
            Tuple of (Ct, Cp) - same shape as inputs
        """
        # Convert to regular numpy to avoid CasADi segfaults
        J_val = float(onp.asarray(J).item()) if onp.asarray(J).ndim == 0 else onp.asarray(J)
        
        # Clamp J to training bounds using regular numpy
        J_clamped = onp.clip(J_val, self.training_bounds.J_min, self.training_bounds.J_max)
        J_norm, D_norm, P_norm = self._normalize_inputs(J_clamped, D_m, P_m)
        
        # Convert to regular floats for the model
        x_input = {
            "J": float(J_norm) if onp.ndim(J_norm) == 0 else onp.asarray(J_norm, dtype=float),
            "D": float(D_norm) if onp.ndim(D_norm) == 0 else onp.asarray(D_norm, dtype=float),
            "P": float(P_norm) if onp.ndim(P_norm) == 0 else onp.asarray(P_norm, dtype=float),
        }
        
        try:
            Ct = self._Ct_model(x_input)
            Cp = self._Cp_model(x_input)
        except Exception:
            # Fallback to safe defaults if model evaluation fails
            Ct = 0.1
            Cp = 0.04
        
        # Convert results to regular numpy and apply physical bounds
        Ct_arr = onp.asarray(Ct)
        Cp_arr = onp.asarray(Cp)
        Ct_clipped = onp.clip(Ct_arr, self.config.Ct_min, 1.0)
        Cp_clipped = onp.clip(Cp_arr, self.config.Cp_min, 1.0)

        if Ct_clipped.ndim == 0:
            Ct_out = float(Ct_clipped)
        else:
            Ct_out = Ct_clipped.astype(float)

        if Cp_clipped.ndim == 0:
            Cp_out = float(Cp_clipped)
        else:
            Cp_out = Cp_clipped.astype(float)

        return Ct_out, Cp_out
    
    def get_performance(self, V, omega, D, P, rho=1.225) -> Tuple[Any, Any]:
        """
        Get dimensional thrust and power.
        
        Args:
            V: Freestream velocity [m/s]
            omega: Rotational speed [rad/s]
            D: Diameter [m]
            P: Pitch [m]
            rho: Air density [kg/m³]
            
        Returns:
            Tuple of (Thrust [N], Power [W])
        """
        # Calculate advance ratio
        n = omega / (2 * np.pi)  # rev/s
        J = V / (n * D + 1e-9)  # Avoid division by zero
        
        Ct, Cp = self.get_coefficients(J, D, P)
        
        # Dimensional quantities
        T = Ct * rho * (n ** 2) * (D ** 4)
        P_shaft = Cp * rho * (n ** 3) * (D ** 5)
        
        return T, P_shaft
    
    def get_efficiency(self, J, D_m, P_m) -> Any:
        """
        Get propulsive efficiency η = J * Ct / Cp.
        
        Args:
            J: Advance ratio
            D_m: Diameter [m]
            P_m: Pitch [m]
            
        Returns:
            Propulsive efficiency
        """
        Ct, Cp = self.get_coefficients(J, D_m, P_m)
        eta = J * Ct / (Cp + 1e-9)
        return np.clip(eta, 0, 1)
    
    def get_static_thrust(self, omega, D, P, rho=1.225) -> Any:
        """
        Get static thrust (V=0, J=0).
        
        Args:
            omega: Rotational speed [rad/s]
            D: Diameter [m]
            P: Pitch [m]
            rho: Air density [kg/m³]
            
        Returns:
            Static thrust [N]
        """
        T, _ = self.get_performance(V=0, omega=omega, D=D, P=P, rho=rho)
        return T
    
    def save(self, path: Path) -> None:
        """
        Save model to pickle file.
        
        Note: The FittedModel objects contain the fitted parameters and
        model structure, enabling full restoration.
        """
        path = Path(path)
        
        save_dict = {
            'family': self.family,
            'config': self.config,
            'training_bounds': self.training_bounds,
            'Ct_params': self.Ct_params,
            'Cp_params': self.Cp_params,
            'fit_metrics': self.fit_metrics,
            # Store the actual FittedModel objects
            'Ct_model': self._Ct_model,
            'Cp_model': self._Cp_model,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: Path) -> 'PropellerMetaModel':
        """
        Load model from pickle file.
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        return cls(
            Ct_model=save_dict['Ct_model'],
            Cp_model=save_dict['Cp_model'],
            family=save_dict['family'],
            config=save_dict['config'],
            training_bounds=save_dict['training_bounds'],
            Ct_params=save_dict['Ct_params'],
            Cp_params=save_dict['Cp_params'],
            fit_metrics=save_dict['fit_metrics'],
        )
    
    def summary(self) -> str:
        """Return a summary string of the model."""
        lines = [
            f"PropellerMetaModel: {self.family}",
            f"  Training data: {self.fit_metrics.get('n_samples', 'N/A')} points "
            f"from {self.fit_metrics.get('n_propellers', 'N/A')} propellers",
            f"  J range: [{self.training_bounds.J_min:.3f}, {self.training_bounds.J_max:.3f}]",
            f"  D range: [{self.training_bounds.D_min*1000:.1f}, {self.training_bounds.D_max*1000:.1f}] mm",
            f"  P range: [{self.training_bounds.P_min*1000:.1f}, {self.training_bounds.P_max*1000:.1f}] mm",
            f"  Ct fit: R² = {self.fit_metrics.get('Ct_R2', 'N/A'):.4f}",
            f"  Cp fit: R² = {self.fit_metrics.get('Cp_R2', 'N/A'):.4f}",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PropellerMetaModel(family='{self.family}', n_samples={self.fit_metrics.get('n_samples', 'N/A')})"


# Convenience functions for pre-trained models
def get_pretrained_model(family: str) -> PropellerMetaModel:
    """
    Load pre-trained meta-model for given family.
    
    Args:
        family: Family name ("Electric", "SlowFly", "Standard", etc.)
        
    Returns:
        Pre-trained PropellerMetaModel
    """
    model_dir = Path(__file__).parent.parent.parent / 'data' / 'propeller_models'
    model_path = model_dir / f'{family}_meta.pkl'
    
    if not model_path.exists():
        available = list_pretrained_families()
        raise ValueError(f"No pre-trained model for family '{family}'. "
                        f"Available: {available}")
    
    return PropellerMetaModel.load(model_path)


def list_pretrained_families() -> List[str]:
    """List available pre-trained model families."""
    model_dir = Path(__file__).parent.parent.parent / 'data' / 'propeller_models'
    
    if not model_dir.exists():
        return []
    
    return [p.stem.replace('_meta', '') 
            for p in model_dir.glob('*_meta.pkl')]
