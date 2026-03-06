"""
Thermal Network Model

Lumped-parameter thermal network for propulsion system components.
Models heat transfer between motor, ESC, battery, and ambient air
with airspeed-dependent forced convection.

Key concepts:
    - Thermal nodes: Components with thermal mass (capacitance)
    - Thermal connections: Heat paths between nodes (conductance)
    - Convective cooling: Airspeed-dependent heat transfer coefficient
    - Derating: Power limiting based on temperature

Governing equation for each node:
    C * dT/dt = Q_in - sum(G_ij * (T_i - T_j))

Where:
    C = thermal capacitance [J/K]
    Q_in = heat input (losses) [W]
    G_ij = thermal conductance between nodes i and j [W/K]
    T = temperature [C or K]

References:
    - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"
    - Typical BLDC motor thermal modeling practices
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import numpy as np

# Type alias for AD-compatible arrays
ArrayLike = Union[float, np.ndarray]


class ConnectionType(Enum):
    """Type of thermal connection."""
    CONDUCTIVE = auto()     # Fixed conductance (solid path)
    CONVECTIVE = auto()     # Airspeed-dependent (to ambient)
    RADIATIVE = auto()      # Temperature-dependent (usually small)


@dataclass
class ThermalNode:
    """
    A node in the thermal network representing a component.
    
    Each node has thermal mass (capacitance) and temperature limits.
    Heat is added to nodes from component losses.
    
    Attributes:
        name: Unique identifier for this node
        thermal_capacitance: Heat capacity [J/K]
        T_initial: Initial temperature [C]
        T_max: Maximum allowable temperature [C]
        T_derate_start: Temperature where derating begins [C]
        description: Optional description
    
    Example:
        >>> motor_node = ThermalNode(
        ...     name="motor",
        ...     thermal_capacitance=50.0,  # J/K
        ...     T_initial=25.0,
        ...     T_max=150.0,
        ...     T_derate_start=120.0,
        ... )
    """
    
    name: str
    thermal_capacitance: float          # [J/K]
    T_initial: float = 25.0             # [C]
    T_max: float = 100.0                # [C]
    T_derate_start: Optional[float] = None  # Defaults to 0.8 * T_max
    description: str = ""
    
    def __post_init__(self):
        """Set default derate start if not specified."""
        if self.T_derate_start is None:
            self.T_derate_start = 0.8 * self.T_max
        
        if self.thermal_capacitance <= 0:
            raise ValueError(f"thermal_capacitance must be positive, got {self.thermal_capacitance}")
    
    def get_derate_factor(self, T: ArrayLike) -> ArrayLike:
        """
        Calculate derating factor based on temperature.
        
        Linear derating from T_derate_start to T_max:
            factor = 1.0 when T <= T_derate_start
            factor = 0.0 when T >= T_max
            factor = linear interpolation between
        
        Args:
            T: Current temperature [C]
            
        Returns:
            Derating factor [0, 1]
        """
        if self.T_derate_start >= self.T_max:
            return np.where(T < self.T_max, 1.0, 0.0)
        
        derate_range = self.T_max - self.T_derate_start
        factor = (self.T_max - T) / derate_range
        return np.clip(factor, 0.0, 1.0)
    
    def time_constant(self, total_conductance: float) -> float:
        """
        Calculate thermal time constant.
        
        tau = C / G
        
        Args:
            total_conductance: Sum of all conductances from this node [W/K]
            
        Returns:
            Time constant [s]
        """
        if total_conductance <= 0:
            return np.inf
        return self.thermal_capacitance / total_conductance


@dataclass
class ThermalConnection:
    """
    Connection between thermal nodes (or to ambient).
    
    Represents a heat transfer path with either fixed or
    airspeed-dependent conductance.
    
    Attributes:
        from_node: Source node name
        to_node: Destination node name ("ambient" for environment)
        connection_type: CONDUCTIVE, CONVECTIVE, or RADIATIVE
        conductance: Base thermal conductance [W/K]
        
        # For convective connections:
        h_base: Base heat transfer coefficient [W/(m^2*K)]
        h_velocity_coeff: Additional h per m/s airspeed [W/(m^2*K)/(m/s)]
        area: Heat transfer area [m^2]
        
    Notes:
        For CONDUCTIVE: G = conductance (constant)
        For CONVECTIVE: G = h * A, where h = h_base + h_velocity_coeff * V
    
    Example:
        >>> motor_to_ambient = ThermalConnection(
        ...     from_node="motor",
        ...     to_node="ambient",
        ...     connection_type=ConnectionType.CONVECTIVE,
        ...     h_base=5.0,           # Natural convection
        ...     h_velocity_coeff=10.0, # Forced convection coefficient
        ...     area=0.01,            # 100 cm^2 surface
        ... )
    """
    
    from_node: str
    to_node: str
    connection_type: ConnectionType = ConnectionType.CONDUCTIVE
    conductance: float = 0.0            # Base conductance [W/K]
    
    # Convective parameters
    h_base: float = 5.0                 # Base h [W/(m^2*K)]
    h_velocity_coeff: float = 10.0      # Additional h per m/s
    area: float = 0.01                  # Heat transfer area [m^2]
    
    # Optional description
    description: str = ""
    
    def get_conductance(self, airspeed: ArrayLike = 0.0) -> ArrayLike:
        """
        Get thermal conductance, possibly airspeed-dependent.
        
        Args:
            airspeed: Air velocity over component [m/s]
            
        Returns:
            Thermal conductance [W/K]
        """
        if self.connection_type == ConnectionType.CONDUCTIVE:
            return self.conductance
        
        elif self.connection_type == ConnectionType.CONVECTIVE:
            # h increases with airspeed (forced convection)
            h = self.h_base + self.h_velocity_coeff * airspeed
            return h * self.area
        
        elif self.connection_type == ConnectionType.RADIATIVE:
            # Simplified: treat as fixed small conductance
            # Full radiative would need T^4 terms
            return self.conductance
        
        return self.conductance


class ThermalNetworkModel:
    """
    Lumped-parameter thermal network model.
    
    Computes temperature derivatives for integration in dynamics
    simulations. Supports multiple nodes with various connection
    types including airspeed-dependent forced convection.
    
    Network topology for propulsion:
        Motor <---> Ambient (forced convection, airspeed-dependent)
        ESC <---> Ambient (forced convection)
        Battery <---> Ambient (natural + forced convection)
        Motor <---> ESC (conductive, through mounting)
    
    Example:
        >>> # Create network
        >>> network = ThermalNetworkModel()
        >>> network.add_node(ThermalNode("motor", thermal_capacitance=50, T_max=150))
        >>> network.add_node(ThermalNode("esc", thermal_capacitance=20, T_max=100))
        >>> network.add_connection(ThermalConnection(
        ...     "motor", "ambient", ConnectionType.CONVECTIVE,
        ...     h_base=5, h_velocity_coeff=10, area=0.01
        ... ))
        >>> 
        >>> # Compute derivatives
        >>> dT = network.get_temperature_derivatives(
        ...     temperatures={"motor": 80, "esc": 50},
        ...     heat_inputs={"motor": 20, "esc": 5},
        ...     airspeed=15,
        ...     T_ambient=25
        ... )
    """
    
    def __init__(self):
        """Initialize empty thermal network."""
        self.nodes: Dict[str, ThermalNode] = {}
        self.connections: List[ThermalConnection] = []
        self._node_connections: Dict[str, List[ThermalConnection]] = {}
    
    def add_node(self, node: ThermalNode) -> None:
        """
        Add a thermal node to the network.
        
        Args:
            node: ThermalNode to add
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists")
        self.nodes[node.name] = node
        self._node_connections[node.name] = []
    
    def add_connection(self, connection: ThermalConnection) -> None:
        """
        Add a thermal connection between nodes.
        
        Args:
            connection: ThermalConnection to add
        """
        # Validate nodes exist (except "ambient")
        if connection.from_node != "ambient" and connection.from_node not in self.nodes:
            raise ValueError(f"Node '{connection.from_node}' not found")
        if connection.to_node != "ambient" and connection.to_node not in self.nodes:
            raise ValueError(f"Node '{connection.to_node}' not found")
        
        self.connections.append(connection)
        
        # Track connections per node
        if connection.from_node in self._node_connections:
            self._node_connections[connection.from_node].append(connection)
        if connection.to_node in self._node_connections:
            self._node_connections[connection.to_node].append(connection)
    
    def get_node_names(self) -> List[str]:
        """Get list of node names."""
        return list(self.nodes.keys())
    
    def get_initial_temperatures(self) -> Dict[str, float]:
        """Get initial temperatures for all nodes."""
        return {name: node.T_initial for name, node in self.nodes.items()}
    
    def get_conductance_matrix(
        self, 
        airspeed: float = 0.0
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Build conductance matrix for current airspeed.
        
        Returns:
            Tuple of:
                - Dict of node-to-node conductances
                - Dict of node-to-ambient conductances
        """
        # Initialize
        node_to_node: Dict[str, Dict[str, float]] = {
            name: {} for name in self.nodes
        }
        node_to_ambient: Dict[str, float] = {
            name: 0.0 for name in self.nodes
        }
        
        for conn in self.connections:
            G = conn.get_conductance(airspeed)
            
            if conn.to_node == "ambient":
                # Connection to ambient
                node_to_ambient[conn.from_node] += G
            elif conn.from_node == "ambient":
                node_to_ambient[conn.to_node] += G
            else:
                # Node-to-node connection (bidirectional)
                if conn.to_node not in node_to_node[conn.from_node]:
                    node_to_node[conn.from_node][conn.to_node] = 0.0
                if conn.from_node not in node_to_node[conn.to_node]:
                    node_to_node[conn.to_node][conn.from_node] = 0.0
                
                node_to_node[conn.from_node][conn.to_node] += G
                node_to_node[conn.to_node][conn.from_node] += G
        
        return node_to_node, node_to_ambient
    
    def get_temperature_derivatives(
        self,
        temperatures: Dict[str, ArrayLike],
        heat_inputs: Dict[str, ArrayLike],
        airspeed: ArrayLike = 0.0,
        T_ambient: ArrayLike = 25.0
    ) -> Dict[str, ArrayLike]:
        """
        Compute temperature derivatives for all nodes.
        
        dT_i/dt = (Q_i - sum_j(G_ij * (T_i - T_j)) - G_amb * (T_i - T_amb)) / C_i
        
        Args:
            temperatures: Current temperature of each node [C]
            heat_inputs: Heat input (losses) to each node [W]
            airspeed: Freestream airspeed [m/s]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Dict of temperature derivatives [C/s] for each node
        """
        derivatives: Dict[str, ArrayLike] = {}
        
        # Get conductances for current airspeed
        node_to_node, node_to_ambient = self.get_conductance_matrix(float(airspeed) if np.isscalar(airspeed) else float(airspeed.mean()))
        
        for name, node in self.nodes.items():
            T_i = temperatures.get(name, node.T_initial)
            Q_i = heat_inputs.get(name, 0.0)
            
            # Heat flow to other nodes
            Q_out_nodes = 0.0
            for other_name, G in node_to_node[name].items():
                T_j = temperatures.get(other_name, self.nodes[other_name].T_initial)
                Q_out_nodes += G * (T_i - T_j)
            
            # Heat flow to ambient
            G_amb = node_to_ambient[name]
            Q_out_ambient = G_amb * (T_i - T_ambient)
            
            # Net heat and derivative
            Q_net = Q_i - Q_out_nodes - Q_out_ambient
            dT_dt = Q_net / node.thermal_capacitance
            
            derivatives[name] = dT_dt
        
        return derivatives
    
    def get_steady_state_temperatures(
        self,
        heat_inputs: Dict[str, float],
        airspeed: float = 0.0,
        T_ambient: float = 25.0
    ) -> Dict[str, float]:
        """
        Calculate steady-state temperatures analytically.
        
        At steady state: dT/dt = 0 for all nodes
        This gives a system of linear equations.
        
        For single node: T_ss = T_amb + Q / G_total
        For multiple nodes: solve linear system
        
        Args:
            heat_inputs: Constant heat input to each node [W]
            airspeed: Freestream airspeed [m/s]
            T_ambient: Ambient temperature [C]
            
        Returns:
            Steady-state temperature of each node [C]
        """
        node_names = list(self.nodes.keys())
        n = len(node_names)
        
        if n == 0:
            return {}
        
        # Build conductance matrix
        node_to_node, node_to_ambient = self.get_conductance_matrix(airspeed)
        
        # Build linear system: A * T = b
        # For node i: sum_j(G_ij * (T_i - T_j)) + G_amb_i * (T_i - T_amb) = Q_i
        # Rearranging: (sum_j(G_ij) + G_amb_i) * T_i - sum_j(G_ij * T_j) = Q_i + G_amb_i * T_amb
        
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i, name_i in enumerate(node_names):
            # Diagonal: sum of all conductances from this node
            G_total = node_to_ambient[name_i]
            for G in node_to_node[name_i].values():
                G_total += G
            A[i, i] = G_total
            
            # Off-diagonal: negative conductance to other nodes
            for name_j, G in node_to_node[name_i].items():
                j = node_names.index(name_j)
                A[i, j] = -G
            
            # RHS: heat input + ambient term
            Q = heat_inputs.get(name_i, 0.0)
            b[i] = Q + node_to_ambient[name_i] * T_ambient
        
        # Solve
        try:
            T_ss = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular matrix - return ambient
            T_ss = np.full(n, T_ambient)
        
        return {name: float(T_ss[i]) for i, name in enumerate(node_names)}
    
    def get_derated_steady_state_temperatures(
        self,
        heat_inputs_base: Dict[str, float],
        airspeed: float = 0.0,
        T_ambient: float = 25.0,
        max_iterations: int = 20,
        tolerance: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calculate steady-state temperatures WITH derating feedback.
        
        Uses analytical solution for equilibrium in the derating region.
        For linear derating: derate(T) = (T_max - T) / (T_max - T_start)
        
        At equilibrium: T = T_amb + Q_base * derate(T) / G
        
        Solving for T in the derating region:
            T = (T_amb + k * T_max) / (1 + k)
            where k = Q_base / ((T_max - T_start) * G_total)
        
        For coupled nodes, we iterate with analytical per-node solutions
        until the inter-node heat flows stabilize.
        
        Args:
            heat_inputs_base: Base (underated) heat input to each node [W]
            airspeed: Freestream airspeed [m/s]
            T_ambient: Ambient temperature [C]
            max_iterations: Max iterations for coupled-node convergence
            tolerance: Convergence tolerance [C]
            
        Returns:
            Steady-state temperature of each node [C] with derating applied
        """
        node_names = list(self.nodes.keys())
        n = len(node_names)
        
        if n == 0:
            return {}
        
        # Get conductance structure
        node_to_node, node_to_ambient = self.get_conductance_matrix(airspeed)
        
        # First, compute underated steady-state to check if derating is needed
        T_underated = self.get_steady_state_temperatures(
            heat_inputs_base, airspeed, T_ambient
        )
        
        # Check if any node needs derating
        needs_derating = any(
            T_underated[name] > self.nodes[name].T_derate_start
            for name in node_names
        )
        
        if not needs_derating:
            return T_underated
        
        # Initialize temperatures for coupled iteration
        T_current = {name: T_ambient for name in node_names}
        
        for iteration in range(max_iterations):
            T_new = {}
            
            for name in node_names:
                node = self.nodes[name]
                Q_base = heat_inputs_base.get(name, 0.0)
                
                # Compute total conductance to ambient + other nodes
                G_to_ambient = node_to_ambient[name]
                
                # Heat flow to/from other nodes (using current temps)
                Q_from_others = 0.0
                G_to_others = 0.0
                for other_name, G in node_to_node[name].items():
                    G_to_others += G
                    # Heat flowing INTO this node from others
                    Q_from_others += G * T_current.get(other_name, T_ambient)
                
                # Effective ambient: accounts for heat from other nodes
                # Node sees: G_amb*(T - T_amb) + G_others*(T - T_others) = Q
                # Rearranging: (G_amb + G_others)*T = Q + G_amb*T_amb + sum(G_j*T_j)
                # T = (Q + G_amb*T_amb + sum(G_j*T_j)) / (G_amb + G_others)
                
                G_total = G_to_ambient + G_to_others
                
                if G_total <= 0:
                    # No heat path - temperature would go to infinity
                    T_new[name] = node.T_max
                    continue
                
                # Effective source temperature (ambient + coupled nodes)
                T_eff_source = (G_to_ambient * T_ambient + Q_from_others) / G_total
                
                # Underated equilibrium for this node
                T_underated_node = T_eff_source + Q_base / G_total
                
                # Check which regime we're in
                T_start = node.T_derate_start
                T_max = node.T_max
                derate_range = T_max - T_start
                
                if derate_range <= 0:
                    # No derating range - hard cutoff at T_max
                    T_new[name] = min(T_underated_node, T_max)
                    continue
                
                if T_underated_node <= T_start:
                    # No derating needed - equilibrium is below T_start
                    T_new[name] = T_underated_node
                    
                elif Q_base <= 0:
                    # No heat input - temperature equals source
                    T_new[name] = T_eff_source
                    
                else:
                    # Equilibrium is in derating region - solve analytically
                    # T = T_source + Q_base * derate(T) / G_total
                    # where derate(T) = (T_max - T) / (T_max - T_start)
                    #
                    # T = T_source + Q_base * (T_max - T) / (derate_range * G_total)
                    # Let k = Q_base / (derate_range * G_total)
                    # T = T_source + k * (T_max - T)
                    # T = T_source + k*T_max - k*T
                    # T(1 + k) = T_source + k*T_max
                    # T = (T_source + k*T_max) / (1 + k)
                    
                    k = Q_base / (derate_range * G_total)
                    T_equilibrium = (T_eff_source + k * T_max) / (1 + k)
                    
                    # Clamp to valid range
                    T_new[name] = np.clip(T_equilibrium, T_start, T_max)
            
            # Check convergence
            max_change = max(abs(T_new[name] - T_current[name]) for name in node_names)
            T_current = T_new
            
            if max_change < tolerance:
                break
        
        return T_current
    
    def get_derating_factors(
        self, 
        temperatures: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get derating factors for all nodes.
        
        Args:
            temperatures: Current temperature of each node [C]
            
        Returns:
            Dict of derating factors [0, 1] for each node
        """
        factors = {}
        for name, node in self.nodes.items():
            T = temperatures.get(name, node.T_initial)
            factors[name] = float(node.get_derate_factor(T))
        return factors
    
    def simulate(
        self,
        heat_profile: Dict[str, np.ndarray],
        dt: float,
        airspeed_profile: Optional[np.ndarray] = None,
        T_ambient: float = 25.0,
        T_initial: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate thermal transient with given heat profile.
        
        Uses Euler integration for simplicity.
        
        Args:
            heat_profile: Dict of heat input arrays [W] for each node
            dt: Time step [s]
            airspeed_profile: Airspeed array [m/s], or None for static
            T_ambient: Ambient temperature [C]
            T_initial: Initial temperatures, or None for defaults
            
        Returns:
            Dict with time and temperature arrays for each node
        """
        # Determine number of steps from first heat profile
        first_key = next(iter(heat_profile.keys()))
        n_steps = len(heat_profile[first_key])
        
        # Initialize
        if T_initial is None:
            T_initial = self.get_initial_temperatures()
        
        if airspeed_profile is None:
            airspeed_profile = np.zeros(n_steps)
        
        # Allocate output arrays
        time = np.zeros(n_steps)
        temperatures = {name: np.zeros(n_steps) for name in self.nodes}
        
        # Set initial conditions
        for name in self.nodes:
            temperatures[name][0] = T_initial.get(name, self.nodes[name].T_initial)
        
        # Integrate
        for i in range(1, n_steps):
            time[i] = time[i-1] + dt
            
            # Current state
            T_current = {name: temperatures[name][i-1] for name in self.nodes}
            Q_current = {name: heat_profile.get(name, np.zeros(n_steps))[i-1] 
                        for name in self.nodes}
            V_current = airspeed_profile[i-1]
            
            # Compute derivatives
            dT = self.get_temperature_derivatives(
                T_current, Q_current, V_current, T_ambient
            )
            
            # Update temperatures
            for name in self.nodes:
                temperatures[name][i] = temperatures[name][i-1] + dT[name] * dt
        
        result = {'time': time}
        result.update(temperatures)
        return result
    
    def summary(self) -> str:
        """Return formatted network summary."""
        lines = [f"Thermal Network: {len(self.nodes)} nodes, {len(self.connections)} connections"]
        
        lines.append("\nNodes:")
        for name, node in self.nodes.items():
            lines.append(f"  {name}: C={node.thermal_capacitance:.1f} J/K, "
                        f"T_max={node.T_max:.0f}C")
        
        lines.append("\nConnections:")
        for conn in self.connections:
            if conn.connection_type == ConnectionType.CONVECTIVE:
                lines.append(f"  {conn.from_node} -> {conn.to_node}: "
                           f"convective (h={conn.h_base}+{conn.h_velocity_coeff}*V, A={conn.area}m2)")
            else:
                lines.append(f"  {conn.from_node} -> {conn.to_node}: "
                           f"G={conn.conductance:.2f} W/K")
        
        return "\n".join(lines)


# =============================================================================
# Propulsion System Thermal Network Factory
# =============================================================================

def create_propulsion_thermal_network(
    motor_capacitance: float = 50.0,
    motor_T_max: float = 150.0,
    motor_area: float = 0.008,
    esc_capacitance: float = 20.0,
    esc_T_max: float = 100.0,
    esc_area: float = 0.004,
    battery_capacitance: float = 100.0,
    battery_T_max: float = 60.0,
    battery_area: float = 0.02,
    h_base: float = 15.0,
    h_velocity_coeff: float = 10.0,
    motor_esc_conductance: float = 0.5,
) -> ThermalNetworkModel:
    """
    Create a standard propulsion thermal network.
    
    Network topology:
        Motor <--convective--> Ambient
        ESC <--convective--> Ambient  
        Battery <--convective--> Ambient
        Motor <--conductive--> ESC
    
    Args:
        motor_capacitance: Motor thermal mass [J/K]
        motor_T_max: Motor max temperature [C]
        motor_area: Motor cooling surface area [m^2]
        esc_capacitance: ESC thermal mass [J/K]
        esc_T_max: ESC max temperature [C]
        esc_area: ESC cooling surface area [m^2]
        battery_capacitance: Battery thermal mass [J/K]
        battery_T_max: Battery max temperature [C]
        battery_area: Battery cooling surface area [m^2]
        h_base: Base convection coefficient [W/(m^2*K)]
        h_velocity_coeff: Forced convection coefficient [W/(m^2*K)/(m/s)]
        motor_esc_conductance: Conductive path motor-ESC [W/K]
        
    Returns:
        Configured ThermalNetworkModel
    """
    network = ThermalNetworkModel()
    
    # Add nodes
    network.add_node(ThermalNode(
        name="motor",
        thermal_capacitance=motor_capacitance,
        T_max=motor_T_max,
        description="BLDC motor windings"
    ))
    
    network.add_node(ThermalNode(
        name="esc",
        thermal_capacitance=esc_capacitance,
        T_max=esc_T_max,
        description="Electronic speed controller"
    ))
    
    network.add_node(ThermalNode(
        name="battery",
        thermal_capacitance=battery_capacitance,
        T_max=battery_T_max,
        description="Battery pack"
    ))
    
    # Add convective connections to ambient
    network.add_connection(ThermalConnection(
        from_node="motor",
        to_node="ambient",
        connection_type=ConnectionType.CONVECTIVE,
        h_base=h_base,
        h_velocity_coeff=h_velocity_coeff,
        area=motor_area,
        description="Motor to ambient (forced convection)"
    ))
    
    network.add_connection(ThermalConnection(
        from_node="esc",
        to_node="ambient",
        connection_type=ConnectionType.CONVECTIVE,
        h_base=h_base,
        h_velocity_coeff=h_velocity_coeff * 0.8,  # Slightly less exposed
        area=esc_area,
        description="ESC to ambient (forced convection)"
    ))
    
    network.add_connection(ThermalConnection(
        from_node="battery",
        to_node="ambient",
        connection_type=ConnectionType.CONVECTIVE,
        h_base=h_base * 0.7,  # More insulated
        h_velocity_coeff=h_velocity_coeff * 0.5,
        area=battery_area,
        description="Battery to ambient (partial convection)"
    ))
    
    # Add conductive connection motor-ESC
    if motor_esc_conductance > 0:
        network.add_connection(ThermalConnection(
            from_node="motor",
            to_node="esc",
            connection_type=ConnectionType.CONDUCTIVE,
            conductance=motor_esc_conductance,
            description="Motor-ESC thermal coupling"
        ))
    
    return network


# =============================================================================
# Convenience Classes
# =============================================================================

@dataclass
class ThermalState:
    """
    Container for thermal system state.
    
    Useful for passing thermal state between components.
    """
    T_motor: float = 25.0
    T_esc: float = 25.0
    T_battery: float = 25.0
    T_ambient: float = 25.0
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dict for thermal network."""
        return {
            "motor": self.T_motor,
            "esc": self.T_esc,
            "battery": self.T_battery,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float], T_ambient: float = 25.0) -> 'ThermalState':
        """Create from dict."""
        return cls(
            T_motor=d.get("motor", 25.0),
            T_esc=d.get("esc", 25.0),
            T_battery=d.get("battery", 25.0),
            T_ambient=T_ambient,
        )
    
    def max_temperature(self) -> float:
        """Get maximum component temperature."""
        return max(self.T_motor, self.T_esc, self.T_battery)


@dataclass  
class HeatInputs:
    """
    Container for heat inputs to thermal nodes.
    
    Computed from component losses (motor I^2R, ESC switching, battery I^2R).
    """
    Q_motor: float = 0.0       # Motor losses [W]
    Q_esc: float = 0.0         # ESC losses [W]
    Q_battery: float = 0.0     # Battery losses [W]
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dict for thermal network."""
        return {
            "motor": self.Q_motor,
            "esc": self.Q_esc,
            "battery": self.Q_battery,
        }
    
    def total(self) -> float:
        """Total heat generation."""
        return self.Q_motor + self.Q_esc + self.Q_battery


def compute_propulsion_heat_inputs(
    motor_current: float,
    motor_resistance: float,
    battery_current: float,
    battery_resistance: float,
    esc_power_in: float,
    esc_efficiency: float = 0.95,
) -> HeatInputs:
    """
    Compute heat inputs from propulsion system operating point.
    
    Args:
        motor_current: Motor phase current [A]
        motor_resistance: Motor winding resistance [Ohm]
        battery_current: Battery discharge current [A]
        battery_resistance: Battery internal resistance [Ohm]
        esc_power_in: Power into ESC [W]
        esc_efficiency: ESC efficiency [0-1]
        
    Returns:
        HeatInputs container
    """
    Q_motor = motor_current**2 * motor_resistance
    Q_battery = battery_current**2 * battery_resistance
    Q_esc = esc_power_in * (1 - esc_efficiency)
    
    return HeatInputs(
        Q_motor=Q_motor,
        Q_esc=Q_esc,
        Q_battery=Q_battery,
    )
