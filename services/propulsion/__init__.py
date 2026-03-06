"""Propulsion system components: propeller meta-models, motor, battery, thermal."""

from .propeller_data import (
    PropellerGeometry,
    PropellerDataset,
    APCDataParser,
    combine_datasets,
)
from .propeller_model import (
    PropellerMetaModelConfig,
    PropellerMetaModel,
    TrainingBounds,
    get_pretrained_model,
    list_pretrained_families,
)
from .motor_model import (
    MotorParameters,
    DifferentiableMotorModel,
    MOTOR_PRESETS,
    get_motor_preset,
    list_motor_presets,
    estimate_motor_params_from_spec,
)
from .battery_model import (
    BatteryCellParameters,
    BatteryPackConfig,
    DifferentiableBatteryModel,
    CELL_PRESETS,
    get_cell_preset,
    list_cell_presets,
    create_lipo_pack,
    lipo_ocv_curve,
    life_ocv_curve,
    liion_ocv_curve,
)
from .thermal_model import (
    ThermalNode,
    ThermalConnection,
    ThermalNetworkModel,
    ConnectionType,
    ThermalState,
    HeatInputs,
    create_propulsion_thermal_network,
    compute_propulsion_heat_inputs,
)
from .propulsion_system import (
    MotorMount,
    create_twin_motor_mounts,
    create_single_motor_mount,
    MAX_MOTORS,
    ESCParameters,
    ESC_PRESETS,
    get_esc_preset,
    list_esc_presets,
    PropellerSpec,
    PropulsionSystemConfig,
    IntegratedPropulsionSystem,
    create_simple_propulsion_system,
)

__all__ = [
    # Data parsing
    "PropellerGeometry",
    "PropellerDataset",
    "APCDataParser",
    "combine_datasets",
    # Propeller meta-model
    "PropellerMetaModelConfig",
    "PropellerMetaModel",
    "TrainingBounds",
    "get_pretrained_model",
    "list_pretrained_families",
    # Motor model
    "MotorParameters",
    "DifferentiableMotorModel",
    "MOTOR_PRESETS",
    "get_motor_preset",
    "list_motor_presets",
    "estimate_motor_params_from_spec",
    # Battery model
    "BatteryCellParameters",
    "BatteryPackConfig",
    "DifferentiableBatteryModel",
    "CELL_PRESETS",
    "get_cell_preset",
    "list_cell_presets",
    "create_lipo_pack",
    "lipo_ocv_curve",
    "life_ocv_curve",
    "liion_ocv_curve",
    # Thermal model
    "ThermalNode",
    "ThermalConnection",
    "ThermalNetworkModel",
    "ConnectionType",
    "ThermalState",
    "HeatInputs",
    "create_propulsion_thermal_network",
    "compute_propulsion_heat_inputs",
    # Integrated propulsion system
    "MotorMount",
    "create_twin_motor_mounts",
    "create_single_motor_mount",
    "MAX_MOTORS",
    "ESCParameters",
    "ESC_PRESETS",
    "get_esc_preset",
    "list_esc_presets",
    "PropellerSpec",
    "PropulsionSystemConfig",
    "IntegratedPropulsionSystem",
    "create_simple_propulsion_system",
]
