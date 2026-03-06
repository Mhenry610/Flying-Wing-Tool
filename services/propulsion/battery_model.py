"""
Differentiable Battery Model with SOC Dynamics

Models battery pack with state-of-charge tracking, internal resistance variation,
and thermal effects. Designed for AD-compatible optimization.

Key equations:
    V_terminal = OCV(SOC) - I * R_internal(SOC, T)
    dSOC/dt = -I / (3600 * capacity_Ah)
    Q_heat = I^2 * R_internal

Features:
    - OCV-SOC polynomial curves for different chemistries
    - Temperature and SOC-dependent internal resistance
    - C-rate limits with thermal derating
    - Pack configuration (series/parallel)

References:
    - Tremblay et al. "A Generic Battery Model for EV Simulation"
    - Typical LiPo/LiFe/Li-ion cell datasheets
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Callable
import numpy as np

# Type alias for AD-compatible arrays
ArrayLike = Union[float, np.ndarray]


# =============================================================================
# OCV-SOC Curves
# =============================================================================

def lipo_ocv_curve(SOC: ArrayLike, V_min: float = 3.0, V_max: float = 4.2) -> ArrayLike:
    """
    Open Circuit Voltage curve for LiPo cells.
    
    Polynomial fit based on typical LiPo discharge curves.
    Characteristic: Relatively flat in 20-80% SOC range, steep at extremes.
    
    Args:
        SOC: State of charge [0, 1]
        V_min: Cell voltage at 0% SOC (default 3.0V for LiPo)
        V_max: Cell voltage at 100% SOC (default 4.2V for LiPo)
        
    Returns:
        Open circuit voltage [V]
    """
    SOC = np.clip(SOC, 0, 1)
    
    # Polynomial coefficients (empirical fit to typical LiPo curve)
    # OCV = a0 + a1*SOC + a2*SOC^2 + a3*SOC^3 + a4*SOC^4
    # Normalized to [0, 1] range, then scaled to [V_min, V_max]
    a = [0.0, 1.2, 0.8, -1.5, 0.5]  # Gives characteristic LiPo shape
    
    normalized = a[0] + a[1]*SOC + a[2]*SOC**2 + a[3]*SOC**3 + a[4]*SOC**4
    normalized = np.clip(normalized, 0, 1)
    
    return V_min + normalized * (V_max - V_min)


def life_ocv_curve(SOC: ArrayLike, V_min: float = 2.5, V_max: float = 3.65) -> ArrayLike:
    """
    Open Circuit Voltage curve for LiFePO4 (LiFe) cells.
    
    LiFe has a very flat discharge curve - nearly constant voltage
    from 10% to 90% SOC.
    
    Args:
        SOC: State of charge [0, 1]
        V_min: Cell voltage at 0% SOC (default 2.5V for LiFe)
        V_max: Cell voltage at 100% SOC (default 3.65V for LiFe)
        
    Returns:
        Open circuit voltage [V]
    """
    SOC = np.clip(SOC, 0, 1)
    
    # LiFe characteristic: very flat plateau around 3.2-3.3V
    # Steep drop below 10% and above 95%
    V_plateau = 3.25
    
    # Piecewise approximation with smooth transitions
    # Below 10%: drops to V_min
    # 10-95%: nearly flat at V_plateau  
    # Above 95%: rises to V_max
    
    low_region = V_min + (V_plateau - V_min) * (SOC / 0.10)
    mid_region = V_plateau + (3.35 - V_plateau) * ((SOC - 0.10) / 0.85)
    high_region = 3.35 + (V_max - 3.35) * ((SOC - 0.95) / 0.05)
    
    # Blend regions smoothly
    OCV = np.where(SOC < 0.10, low_region,
           np.where(SOC < 0.95, mid_region, high_region))
    
    return OCV


def liion_ocv_curve(SOC: ArrayLike, V_min: float = 2.5, V_max: float = 4.2) -> ArrayLike:
    """
    Open Circuit Voltage curve for Li-ion cells (18650/21700 type).
    
    Similar to LiPo but with slightly different curve shape.
    Typically used for energy cells (lower C-rate, higher capacity).
    
    Args:
        SOC: State of charge [0, 1]
        V_min: Cell voltage at 0% SOC (default 2.5V)
        V_max: Cell voltage at 100% SOC (default 4.2V)
        
    Returns:
        Open circuit voltage [V]
    """
    SOC = np.clip(SOC, 0, 1)
    
    # Li-ion curve: similar to LiPo but slightly more linear in mid-range
    a = [0.0, 0.9, 1.0, -1.2, 0.3]
    
    normalized = a[0] + a[1]*SOC + a[2]*SOC**2 + a[3]*SOC**3 + a[4]*SOC**4
    normalized = np.clip(normalized, 0, 1)
    
    return V_min + normalized * (V_max - V_min)


# =============================================================================
# Cell Parameters
# =============================================================================

@dataclass
class BatteryCellParameters:
    """
    Parameters for a single battery cell.
    
    Covers LiPo, LiFe, and Li-ion chemistries with typical values.
    
    Attributes:
        capacity_Ah: Cell capacity in Amp-hours
        V_nominal: Nominal voltage [V]
        V_max: Maximum (fully charged) voltage [V]
        V_min: Minimum (cutoff) voltage [V]
        mass: Cell mass [kg]
        R_internal_mOhm: Internal resistance at 25C, 50% SOC [mOhm]
        R_temp_coeff: Resistance increase per degree below 25C [1/C]
        R_soc_factor_low: Resistance multiplier at 0% SOC
        R_soc_factor_high: Resistance multiplier at 100% SOC
        C_rate_max_continuous: Maximum continuous discharge C-rate
        C_rate_max_burst: Maximum burst discharge C-rate (10s)
        thermal_capacitance: Heat capacity [J/C]
        thermal_resistance: Cell to ambient [C/W]
        T_max: Maximum operating temperature [C]
        T_min: Minimum operating temperature [C]
        chemistry: Cell chemistry identifier
        ocv_curve: Function to compute OCV from SOC
    
    Example:
        >>> cell = BatteryCellParameters(
        ...     capacity_Ah=2.2,
        ...     V_nominal=3.7,
        ...     V_max=4.2,
        ...     V_min=3.0,
        ...     mass=0.045,
        ...     R_internal_mOhm=8.0,
        ...     C_rate_max_continuous=25,
        ... )
        >>> print(f"Max current: {cell.I_max_continuous:.1f} A")
    """
    
    # Capacity and voltage
    capacity_Ah: float
    V_nominal: float
    V_max: float
    V_min: float
    mass: float                         # [kg]
    
    # Internal resistance
    R_internal_mOhm: float              # At 25C, 50% SOC [mOhm]
    R_temp_coeff: float = 0.005         # Resistance increase per C below 25C
    R_soc_factor_low: float = 1.5       # R multiplier at 0% SOC
    R_soc_factor_high: float = 1.2      # R multiplier at 100% SOC
    
    # Current limits (C-rates)
    C_rate_max_continuous: float = 25.0
    C_rate_max_burst: float = 40.0      # 10-second burst
    
    # Thermal properties
    thermal_capacitance: float = 50.0   # [J/C]
    thermal_resistance: float = 10.0    # [C/W]
    T_max: float = 60.0                 # [C]
    T_min: float = -20.0                # [C]
    
    # Chemistry identifier
    chemistry: str = "LiPo"
    
    # OCV curve function (set in __post_init__)
    ocv_curve: Optional[Callable] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Set default OCV curve based on chemistry."""
        if self.ocv_curve is None:
            if self.chemistry.lower() in ['lipo', 'li-po', 'lithium polymer']:
                self.ocv_curve = lambda soc: lipo_ocv_curve(soc, self.V_min, self.V_max)
            elif self.chemistry.lower() in ['life', 'lifepo4', 'lfp']:
                self.ocv_curve = lambda soc: life_ocv_curve(soc, self.V_min, self.V_max)
            elif self.chemistry.lower() in ['liion', 'li-ion', 'lithium ion']:
                self.ocv_curve = lambda soc: liion_ocv_curve(soc, self.V_min, self.V_max)
            else:
                # Default to LiPo
                self.ocv_curve = lambda soc: lipo_ocv_curve(soc, self.V_min, self.V_max)
    
    @property
    def R_internal(self) -> float:
        """Internal resistance in Ohms (at reference conditions)."""
        return self.R_internal_mOhm / 1000.0
    
    @property
    def I_max_continuous(self) -> float:
        """Maximum continuous discharge current [A]."""
        return self.capacity_Ah * self.C_rate_max_continuous
    
    @property
    def I_max_burst(self) -> float:
        """Maximum burst discharge current [A]."""
        return self.capacity_Ah * self.C_rate_max_burst
    
    @property
    def energy_Wh(self) -> float:
        """Cell energy capacity [Wh]."""
        return self.capacity_Ah * self.V_nominal
    
    @property
    def specific_energy(self) -> float:
        """Specific energy [Wh/kg]."""
        return self.energy_Wh / self.mass if self.mass > 0 else 0.0
    
    def get_ocv(self, SOC: ArrayLike) -> ArrayLike:
        """Get open circuit voltage for given SOC."""
        return self.ocv_curve(SOC)
    
    def get_resistance(
        self, 
        SOC: ArrayLike, 
        T_cell: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Get internal resistance adjusted for SOC and temperature.
        
        R increases at low/high SOC and at low temperatures.
        
        Args:
            SOC: State of charge [0, 1]
            T_cell: Cell temperature [C]
            
        Returns:
            Internal resistance [Ohm]
        """
        SOC = np.clip(SOC, 0, 1)
        
        # SOC factor: higher R at extremes
        # Interpolate between low, mid (1.0), and high factors
        soc_factor = np.where(
            SOC < 0.5,
            1.0 + (self.R_soc_factor_low - 1.0) * (0.5 - SOC) / 0.5,
            1.0 + (self.R_soc_factor_high - 1.0) * (SOC - 0.5) / 0.5
        )
        
        # Temperature factor: R increases below 25C
        temp_factor = 1.0 + self.R_temp_coeff * np.maximum(0, 25.0 - T_cell)
        
        return self.R_internal * soc_factor * temp_factor
    
    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            f"Cell: {self.chemistry}",
            f"  Capacity: {self.capacity_Ah:.2f} Ah ({self.energy_Wh:.1f} Wh)",
            f"  Voltage: {self.V_min:.2f} - {self.V_max:.2f} V (nom {self.V_nominal:.2f})",
            f"  R_internal: {self.R_internal_mOhm:.1f} mOhm",
            f"  Mass: {self.mass*1000:.0f} g",
            f"  C_max: {self.C_rate_max_continuous:.0f}C continuous, {self.C_rate_max_burst:.0f}C burst",
            f"  I_max: {self.I_max_continuous:.0f} A continuous",
            f"  Specific energy: {self.specific_energy:.0f} Wh/kg",
        ]
        return "\n".join(lines)


# =============================================================================
# Cell Presets
# =============================================================================

CELL_PRESETS: Dict[str, BatteryCellParameters] = {
    # Standard LiPo - typical RC hobby cells
    'LiPo_Standard': BatteryCellParameters(
        capacity_Ah=2.2,
        V_nominal=3.7,
        V_max=4.2,
        V_min=3.0,
        mass=0.045,
        R_internal_mOhm=5.0,
        C_rate_max_continuous=25,
        C_rate_max_burst=40,
        thermal_capacitance=45.0,
        thermal_resistance=12.0,
        T_max=60.0,
        chemistry="LiPo",
    ),
    
    # High-C LiPo - racing/performance cells
    'LiPo_HighC': BatteryCellParameters(
        capacity_Ah=1.5,
        V_nominal=3.7,
        V_max=4.2,
        V_min=3.0,
        mass=0.040,
        R_internal_mOhm=3.0,
        C_rate_max_continuous=50,
        C_rate_max_burst=75,
        thermal_capacitance=40.0,
        thermal_resistance=10.0,
        T_max=60.0,
        chemistry="LiPo",
    ),
    
    # High-capacity LiPo - endurance cells
    'LiPo_HighCap': BatteryCellParameters(
        capacity_Ah=5.0,
        V_nominal=3.7,
        V_max=4.2,
        V_min=3.0,
        mass=0.095,
        R_internal_mOhm=4.0,
        C_rate_max_continuous=15,
        C_rate_max_burst=25,
        thermal_capacitance=90.0,
        thermal_resistance=8.0,
        T_max=60.0,
        chemistry="LiPo",
    ),
    
    # LiFePO4 (A123 style)
    'LiFe_A123': BatteryCellParameters(
        capacity_Ah=2.5,
        V_nominal=3.3,
        V_max=3.65,
        V_min=2.5,
        mass=0.070,
        R_internal_mOhm=6.0,
        C_rate_max_continuous=30,
        C_rate_max_burst=50,
        thermal_capacitance=70.0,
        thermal_resistance=8.0,
        T_max=70.0,
        T_min=-30.0,
        chemistry="LiFe",
    ),
    
    # Standard LiFe cell
    'LiFe_Standard': BatteryCellParameters(
        capacity_Ah=2.3,
        V_nominal=3.3,
        V_max=3.65,
        V_min=2.5,
        mass=0.065,
        R_internal_mOhm=8.0,
        C_rate_max_continuous=20,
        C_rate_max_burst=35,
        thermal_capacitance=65.0,
        thermal_resistance=10.0,
        T_max=70.0,
        T_min=-30.0,
        chemistry="LiFe",
    ),
    
    # Li-ion 18650 (Samsung 25R style - high drain)
    'LiIon_18650_HighDrain': BatteryCellParameters(
        capacity_Ah=2.5,
        V_nominal=3.6,
        V_max=4.2,
        V_min=2.5,
        mass=0.045,
        R_internal_mOhm=20.0,
        C_rate_max_continuous=8,  # 20A continuous
        C_rate_max_burst=12,
        thermal_capacitance=45.0,
        thermal_resistance=15.0,
        T_max=60.0,
        chemistry="LiIon",
    ),
    
    # Li-ion 18650 (Panasonic NCR style - high capacity)
    'LiIon_18650_HighCap': BatteryCellParameters(
        capacity_Ah=3.4,
        V_nominal=3.6,
        V_max=4.2,
        V_min=2.5,
        mass=0.048,
        R_internal_mOhm=45.0,
        C_rate_max_continuous=2,  # ~7A max
        C_rate_max_burst=3,
        thermal_capacitance=48.0,
        thermal_resistance=15.0,
        T_max=60.0,
        chemistry="LiIon",
    ),
    
    # Li-ion 21700 (Tesla style)
    'LiIon_21700': BatteryCellParameters(
        capacity_Ah=5.0,
        V_nominal=3.6,
        V_max=4.2,
        V_min=2.5,
        mass=0.070,
        R_internal_mOhm=15.0,
        C_rate_max_continuous=6,  # 30A
        C_rate_max_burst=9,
        thermal_capacitance=70.0,
        thermal_resistance=12.0,
        T_max=60.0,
        chemistry="LiIon",
    ),
}


def get_cell_preset(name: str) -> BatteryCellParameters:
    """
    Get a pre-defined cell by name.
    
    Args:
        name: Cell preset name
        
    Returns:
        BatteryCellParameters instance (copy)
    """
    if name not in CELL_PRESETS:
        available = ", ".join(sorted(CELL_PRESETS.keys()))
        raise ValueError(f"Unknown cell preset '{name}'. Available: {available}")
    
    import copy
    return copy.deepcopy(CELL_PRESETS[name])


def list_cell_presets() -> list:
    """List available cell preset names."""
    return sorted(CELL_PRESETS.keys())


# =============================================================================
# Battery Pack Configuration
# =============================================================================

@dataclass
class BatteryPackConfig:
    """
    Battery pack configuration from cells.
    
    Defines series/parallel arrangement and pack-level properties.
    
    Attributes:
        cell: Cell parameters
        n_series: Number of cells in series (determines voltage)
        n_parallel: Number of cells in parallel (determines capacity)
        pack_overhead_factor: Mass overhead for wiring, case, BMS (default 1.15 = 15%)
        
    Example:
        >>> cell = get_cell_preset('LiPo_Standard')
        >>> pack = BatteryPackConfig(cell, n_series=6, n_parallel=1)  # 6S 2200mAh
        >>> print(f"Pack: {pack.V_nominal:.1f}V, {pack.capacity_Ah:.2f}Ah")
        Pack: 22.2V, 2.20Ah
    """
    
    cell: BatteryCellParameters
    n_series: int
    n_parallel: int
    pack_overhead_factor: float = 1.15
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_series < 1:
            raise ValueError(f"n_series must be >= 1, got {self.n_series}")
        if self.n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {self.n_parallel}")
    
    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.n_series * self.n_parallel
    
    @property
    def V_nominal(self) -> float:
        """Pack nominal voltage [V]."""
        return self.cell.V_nominal * self.n_series
    
    @property
    def V_max(self) -> float:
        """Pack maximum voltage [V]."""
        return self.cell.V_max * self.n_series
    
    @property
    def V_min(self) -> float:
        """Pack minimum voltage [V]."""
        return self.cell.V_min * self.n_series
    
    @property
    def capacity_Ah(self) -> float:
        """Pack capacity [Ah]."""
        return self.cell.capacity_Ah * self.n_parallel
    
    @property
    def capacity_mAh(self) -> float:
        """Pack capacity [mAh]."""
        return self.capacity_Ah * 1000
    
    @property
    def energy_Wh(self) -> float:
        """Pack energy [Wh]."""
        return self.capacity_Ah * self.V_nominal
    
    @property
    def R_internal(self) -> float:
        """Pack internal resistance [Ohm] at reference conditions."""
        # Series adds, parallel divides
        return self.cell.R_internal * self.n_series / self.n_parallel
    
    @property
    def I_max_continuous(self) -> float:
        """Pack maximum continuous discharge current [A]."""
        return self.cell.I_max_continuous * self.n_parallel
    
    @property
    def I_max_burst(self) -> float:
        """Pack maximum burst discharge current [A]."""
        return self.cell.I_max_burst * self.n_parallel
    
    @property
    def P_max_continuous(self) -> float:
        """Pack maximum continuous power [W] (at nominal voltage)."""
        return self.I_max_continuous * self.V_nominal
    
    @property
    def mass(self) -> float:
        """Pack mass including overhead [kg]."""
        return self.cell.mass * self.n_cells * self.pack_overhead_factor
    
    @property
    def specific_energy(self) -> float:
        """Pack specific energy [Wh/kg]."""
        return self.energy_Wh / self.mass if self.mass > 0 else 0.0
    
    @property
    def config_string(self) -> str:
        """Standard RC notation (e.g., '6S2P')."""
        if self.n_parallel == 1:
            return f"{self.n_series}S"
        return f"{self.n_series}S{self.n_parallel}P"
    
    def get_resistance(
        self, 
        SOC: ArrayLike, 
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Get pack internal resistance adjusted for SOC and temperature.
        
        Args:
            SOC: State of charge [0, 1]
            T_battery: Battery temperature [C]
            
        Returns:
            Pack internal resistance [Ohm]
        """
        cell_R = self.cell.get_resistance(SOC, T_battery)
        return cell_R * self.n_series / self.n_parallel
    
    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            f"Battery Pack: {self.config_string} ({self.n_cells} cells)",
            f"  Cell: {self.cell.chemistry}",
            f"  Capacity: {self.capacity_mAh:.0f} mAh ({self.energy_Wh:.1f} Wh)",
            f"  Voltage: {self.V_min:.1f} - {self.V_max:.1f} V (nom {self.V_nominal:.1f})",
            f"  R_internal: {self.R_internal*1000:.1f} mOhm",
            f"  I_max: {self.I_max_continuous:.0f} A continuous",
            f"  P_max: {self.P_max_continuous:.0f} W",
            f"  Mass: {self.mass*1000:.0f} g",
            f"  Specific energy: {self.specific_energy:.0f} Wh/kg",
        ]
        return "\n".join(lines)


# =============================================================================
# Differentiable Battery Model
# =============================================================================

class DifferentiableBatteryModel:
    """
    Differentiable battery model with SOC dynamics.
    
    Provides AD-compatible methods for:
    - Terminal voltage under load
    - SOC derivative (discharge rate)
    - Heat generation
    - Current limits with derating
    
    All methods support both scalar and array inputs for use
    in optimization and simulation.
    
    Example:
        >>> cell = get_cell_preset('LiPo_Standard')
        >>> pack = BatteryPackConfig(cell, n_series=6, n_parallel=1)
        >>> battery = DifferentiableBatteryModel(pack)
        >>> 
        >>> V = battery.get_terminal_voltage(SOC=0.8, I_discharge=10, T_battery=25)
        >>> print(f"Terminal voltage at 10A: {V:.2f} V")
    """
    
    def __init__(self, pack: BatteryPackConfig):
        """
        Initialize battery model.
        
        Args:
            pack: Battery pack configuration
        """
        self.pack = pack
    
    def get_ocv(self, SOC: ArrayLike) -> ArrayLike:
        """
        Get pack open-circuit voltage for given SOC.
        
        Args:
            SOC: State of charge [0, 1]
            
        Returns:
            Open circuit voltage [V]
        """
        cell_ocv = self.pack.cell.get_ocv(SOC)
        return cell_ocv * self.pack.n_series
    
    def get_resistance(
        self, 
        SOC: ArrayLike, 
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Get pack internal resistance for given SOC and temperature.
        
        Args:
            SOC: State of charge [0, 1]
            T_battery: Battery temperature [C]
            
        Returns:
            Internal resistance [Ohm]
        """
        return self.pack.get_resistance(SOC, T_battery)
    
    def get_terminal_voltage(
        self, 
        SOC: ArrayLike, 
        I_discharge: ArrayLike,
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Get terminal voltage under load.
        
        V_terminal = OCV(SOC) - I * R_internal(SOC, T)
        
        Args:
            SOC: State of charge [0, 1]
            I_discharge: Discharge current [A] (positive = discharging)
            T_battery: Battery temperature [C]
            
        Returns:
            Terminal voltage [V]
        """
        OCV = self.get_ocv(SOC)
        R = self.get_resistance(SOC, T_battery)
        
        return OCV - I_discharge * R
    
    def get_current_for_power(
        self,
        SOC: ArrayLike,
        P_demand: ArrayLike,
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Calculate discharge current required to deliver given power.
        
        Solves: P = V * I = (OCV - I*R) * I
        Which gives: R*I^2 - OCV*I + P = 0
        
        Args:
            SOC: State of charge [0, 1]
            P_demand: Demanded power [W]
            T_battery: Battery temperature [C]
            
        Returns:
            Required discharge current [A]
        """
        OCV = self.get_ocv(SOC)
        R = self.get_resistance(SOC, T_battery)
        
        # Quadratic: R*I^2 - OCV*I + P = 0
        # I = (OCV - sqrt(OCV^2 - 4*R*P)) / (2*R)
        # Take lower root for efficiency (less I^2*R loss)
        
        discriminant = OCV**2 - 4 * R * P_demand
        discriminant = np.maximum(discriminant, 0)  # Clamp for AD
        
        I_discharge = (OCV - np.sqrt(discriminant)) / (2 * R)
        
        return I_discharge
    
    def get_power_for_current(
        self,
        SOC: ArrayLike,
        I_discharge: ArrayLike,
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Calculate power output for given discharge current.
        
        P = V * I = (OCV - I*R) * I
        
        Args:
            SOC: State of charge [0, 1]
            I_discharge: Discharge current [A]
            T_battery: Battery temperature [C]
            
        Returns:
            Power output [W]
        """
        V_terminal = self.get_terminal_voltage(SOC, I_discharge, T_battery)
        return V_terminal * I_discharge
    
    def get_soc_derivative(self, I_discharge: ArrayLike) -> ArrayLike:
        """
        Calculate rate of change of SOC.
        
        dSOC/dt = -I / (3600 * capacity_Ah)
        
        Negative because discharging decreases SOC.
        
        Args:
            I_discharge: Discharge current [A] (positive = discharging)
            
        Returns:
            dSOC/dt [1/s]
        """
        return -I_discharge / (3600 * self.pack.capacity_Ah)
    
    def get_time_to_discharge(
        self, 
        SOC_start: float, 
        SOC_end: float, 
        I_discharge: float
    ) -> float:
        """
        Estimate time to discharge from SOC_start to SOC_end.
        
        Assumes constant current (approximation).
        
        Args:
            SOC_start: Starting SOC [0, 1]
            SOC_end: Ending SOC [0, 1]
            I_discharge: Discharge current [A]
            
        Returns:
            Time [s]
        """
        delta_SOC = SOC_start - SOC_end
        if I_discharge <= 0 or delta_SOC <= 0:
            return np.inf
            
        return delta_SOC * 3600 * self.pack.capacity_Ah / I_discharge
    
    def get_heat_generation(
        self,
        I_discharge: ArrayLike,
        SOC: ArrayLike,
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Calculate heat generation rate.
        
        Q = I^2 * R (Joule heating)
        
        Args:
            I_discharge: Discharge current [A]
            SOC: State of charge [0, 1]
            T_battery: Battery temperature [C]
            
        Returns:
            Heat generation rate [W]
        """
        R = self.get_resistance(SOC, T_battery)
        return I_discharge**2 * R
    
    def get_efficiency(
        self,
        I_discharge: ArrayLike,
        SOC: ArrayLike,
        T_battery: ArrayLike = 25.0
    ) -> ArrayLike:
        """
        Calculate discharge efficiency.
        
        eta = P_out / P_total = (OCV - I*R)*I / (OCV*I) = 1 - I*R/OCV
        
        Args:
            I_discharge: Discharge current [A]
            SOC: State of charge [0, 1]
            T_battery: Battery temperature [C]
            
        Returns:
            Efficiency [0, 1]
        """
        OCV = self.get_ocv(SOC)
        R = self.get_resistance(SOC, T_battery)
        
        # Avoid division by zero
        efficiency = np.where(
            OCV > 1e-6,
            1.0 - I_discharge * R / OCV,
            0.0
        )
        
        return np.clip(efficiency, 0, 1)
    
    def get_limits(
        self,
        SOC: float,
        T_battery: float = 25.0
    ) -> Dict[str, float]:
        """
        Get operational limits for current conditions.
        
        Includes derating for:
        - Low SOC (voltage limit)
        - High temperature
        - Low temperature (increased resistance)
        
        Args:
            SOC: State of charge [0, 1]
            T_battery: Battery temperature [C]
            
        Returns:
            Dict with I_max, P_max, derate_factor, V_terminal_min, etc.
        """
        cell = self.pack.cell
        
        # Base limits
        I_max_base = self.pack.I_max_continuous
        
        # Temperature derating
        if T_battery > cell.T_max * 0.8:
            # Linear derating from 80% to 100% of T_max
            T_margin = (cell.T_max - T_battery) / (cell.T_max * 0.2)
            temp_derate = np.clip(T_margin, 0, 1)
        elif T_battery < 0:
            # Derate at low temp due to increased resistance
            temp_derate = np.clip(1.0 + T_battery / 20, 0.5, 1.0)
        else:
            temp_derate = 1.0
        
        # SOC-based voltage limit
        # Don't allow current that would drop voltage below V_min
        OCV = self.get_ocv(SOC)
        R = self.get_resistance(SOC, T_battery)
        I_voltage_limit = (OCV - self.pack.V_min) / R if R > 0 else np.inf
        
        # Combined limit
        I_max = min(I_max_base * temp_derate, I_voltage_limit)
        I_max = max(I_max, 0)
        
        # Power limit
        V_at_limit = self.get_terminal_voltage(SOC, I_max, T_battery)
        P_max = V_at_limit * I_max
        
        return {
            'I_max': I_max,
            'I_max_base': I_max_base,
            'P_max': P_max,
            'V_terminal_at_limit': V_at_limit,
            'V_open_circuit': float(OCV),
            'R_internal': float(R),
            'derate_factor': temp_derate,
            'SOC': SOC,
            'T_battery': T_battery,
        }
    
    def simulate_discharge(
        self,
        I_profile: np.ndarray,
        dt: float,
        SOC_initial: float = 1.0,
        T_battery: float = 25.0,
        SOC_cutoff: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate battery discharge with given current profile.
        
        Simple Euler integration for quick analysis.
        
        Args:
            I_profile: Array of discharge currents [A]
            dt: Time step [s]
            SOC_initial: Starting SOC [0, 1]
            T_battery: Battery temperature (constant) [C]
            SOC_cutoff: Stop simulation when SOC reaches this value
            
        Returns:
            Dict with time, SOC, voltage, power, heat arrays
        """
        n_steps = len(I_profile)
        
        time = np.zeros(n_steps)
        SOC = np.zeros(n_steps)
        voltage = np.zeros(n_steps)
        power = np.zeros(n_steps)
        heat = np.zeros(n_steps)
        
        SOC[0] = SOC_initial
        voltage[0] = float(self.get_terminal_voltage(SOC[0], I_profile[0], T_battery))
        power[0] = voltage[0] * I_profile[0]
        heat[0] = float(self.get_heat_generation(I_profile[0], SOC[0], T_battery))
        
        for i in range(1, n_steps):
            time[i] = time[i-1] + dt
            
            # Update SOC
            dSOC = float(self.get_soc_derivative(I_profile[i-1])) * dt
            SOC[i] = np.clip(SOC[i-1] + dSOC, 0, 1)
            
            # Calculate outputs for this step
            voltage[i] = float(self.get_terminal_voltage(SOC[i], I_profile[i], T_battery))
            power[i] = voltage[i] * I_profile[i]
            heat[i] = float(self.get_heat_generation(I_profile[i], SOC[i], T_battery))
            
            # Check cutoff after computing values
            if SOC[i] <= SOC_cutoff:
                # Truncate arrays (keeping this last computed point)
                time = time[:i+1]
                SOC = SOC[:i+1]
                voltage = voltage[:i+1]
                power = power[:i+1]
                heat = heat[:i+1]
                break
        
        return {
            'time': time,
            'SOC': SOC,
            'voltage': voltage,
            'power': power,
            'heat': heat,
            'current': I_profile[:len(time)],
            'energy_Wh': np.trapz(power, time) / 3600,
        }
    
    def summary(self) -> str:
        """Return formatted summary."""
        return self.pack.summary()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_lipo_pack(
    n_series: int,
    capacity_mAh: float,
    c_rating: float = 25.0,
    n_parallel: int = 1
) -> BatteryPackConfig:
    """
    Create a LiPo pack configuration with common parameters.
    
    Args:
        n_series: Number of cells in series (e.g., 6 for 6S)
        capacity_mAh: Pack capacity in mAh
        c_rating: C-rate for continuous discharge
        n_parallel: Number of cells in parallel (default 1)
        
    Returns:
        BatteryPackConfig
    """
    # Calculate per-cell capacity
    cell_capacity = capacity_mAh / 1000 / n_parallel
    
    # Estimate cell mass from capacity (empirical: ~20g per 1000mAh for LiPo)
    cell_mass = cell_capacity * 0.020
    
    # Estimate internal resistance (empirical: ~5mOhm for 2Ah, scales inversely)
    cell_R = 5.0 * (2.0 / cell_capacity) if cell_capacity > 0 else 5.0
    
    cell = BatteryCellParameters(
        capacity_Ah=cell_capacity,
        V_nominal=3.7,
        V_max=4.2,
        V_min=3.0,
        mass=cell_mass,
        R_internal_mOhm=cell_R,
        C_rate_max_continuous=c_rating,
        C_rate_max_burst=c_rating * 1.5,
        chemistry="LiPo",
    )
    
    return BatteryPackConfig(cell, n_series, n_parallel)
