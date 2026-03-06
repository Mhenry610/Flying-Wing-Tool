"""
Simple Autopilot for Mission Simulation

PID-based autopilot providing control commands based on phase targets.
Supports all MissionPhaseType modes with phase-specific control logic.

Control outputs:
- throttle: 0-1 (motor power command)
- elevator: degrees (pitch control, positive = nose up)
- aileron: degrees (roll control, positive = roll right)
- rudder: degrees (yaw control, positive = nose right)
- brake: 0-1 (wheel brake command, ground phases only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .mission_definition import MissionPhase, MissionPhaseType, Waypoint


@dataclass
class PIDController:
    """
    Simple PID controller with anti-windup.
    
    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        output_min: Minimum output limit
        output_max: Maximum output limit
        integral_max: Maximum integral accumulator (anti-windup)
    """
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -float('inf')
    output_max: float = float('inf')
    integral_max: float = 100.0
    
    # State
    _integral: float = field(default=0.0, repr=False)
    _prev_error: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute control output.
        
        Args:
            error: Current error (setpoint - measurement)
            dt: Time step [s]
        
        Returns:
            Control output (clamped to limits)
        """
        if dt <= 0:
            return 0.0
        
        # Proportional
        p_term = self.kp * error
        
        # Integral with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self._integral
        
        # Derivative (skip on first call)
        if self._initialized:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0
            self._initialized = True
        
        self._prev_error = error
        
        # Sum and clamp
        output = p_term + i_term + d_term
        return float(np.clip(output, self.output_min, self.output_max))


@dataclass 
class AutopilotGains:
    """
    Collection of autopilot PID gains.
    
    Default values are suitable for typical 1-5 kg flying wings.
    Tuned for stable cascaded control with reduced oscillation.
    """
    # Altitude control (outer loop) -> pitch command
    # Reduced gains for smoother altitude tracking
    altitude_kp: float = 0.08
    altitude_ki: float = 0.01
    altitude_kd: float = 0.3
    
    # Climb rate control (inner loop) -> pitch command
    # Reduced gains to prevent overshoot
    climb_rate_kp: float = 2.0
    climb_rate_ki: float = 0.2
    climb_rate_kd: float = 0.5
    
    # Speed control -> throttle command
    speed_kp: float = 0.15
    speed_ki: float = 0.02
    speed_kd: float = 0.05

    # Cruise speed control -> throttle command
    cruise_speed_kp: float = 0.1
    cruise_speed_ki: float = 0.015
    cruise_speed_kd: float = 0.03
    
    # Heading control -> bank angle command
    heading_kp: float = 1.5
    heading_ki: float = 0.05
    heading_kd: float = 0.3
    
    # Bank angle control -> aileron command
    bank_kp: float = 1.0
    bank_ki: float = 0.05
    bank_kd: float = 0.2
    
    # Pitch angle control -> elevator command
    # Reduced gains for smoother pitch response
    pitch_kp: float = 1.0
    pitch_ki: float = 0.1
    pitch_kd: float = 0.3


class SimpleAutopilot:
    """
    PID-based autopilot for mission simulation.
    
    Provides control commands based on phase targets with phase-specific logic.
    
    Control architecture:
    - Altitude -> Climb rate -> Pitch angle -> Elevator (cascaded)
    - Speed -> Throttle (direct)
    - Heading -> Bank angle -> Aileron (cascaded)
    
    Attributes:
        gains: AutopilotGains instance
        max_elevator: Maximum elevator deflection [deg]
        max_aileron: Maximum aileron deflection [deg]
        max_rudder: Maximum rudder deflection [deg]
        max_bank: Maximum bank angle [deg]
        max_pitch: Maximum pitch angle [deg]
    """
    
    def __init__(
        self,
        gains: Optional[AutopilotGains] = None,
        max_elevator: float = 20.0,
        max_aileron: float = 25.0,
        max_rudder: float = 15.0,
        max_bank: float = 45.0,
        max_pitch: float = 20.0,
    ):
        self.gains = gains or AutopilotGains()
        self.max_elevator = max_elevator
        self.max_aileron = max_aileron
        self.max_rudder = max_rudder
        self.max_bank = max_bank
        self.max_pitch = max_pitch
        
        # Initialize PID controllers
        self._init_controllers()
        self._last_phase_key = None
        self._cruise_alt_integral = 0.0
    
    def _init_controllers(self):
        """Initialize all PID controllers."""
        g = self.gains
        
        # Altitude -> climb rate command
        self.altitude_pid = PIDController(
            kp=g.altitude_kp, ki=g.altitude_ki, kd=g.altitude_kd,
            output_min=-3.0, output_max=3.0,  # Reduced climb rate limits [m/s]
        )
        
        # Climb rate -> pitch command
        self.climb_rate_pid = PIDController(
            kp=g.climb_rate_kp, ki=g.climb_rate_ki, kd=g.climb_rate_kd,
            output_min=-self.max_pitch, output_max=self.max_pitch,
        )
        
        # Speed -> throttle
        self.speed_pid = PIDController(
            kp=g.speed_kp, ki=g.speed_ki, kd=g.speed_kd,
            output_min=0.0, output_max=1.0,
        )

        self.cruise_speed_pid = PIDController(
            kp=g.cruise_speed_kp, ki=g.cruise_speed_ki, kd=g.cruise_speed_kd,
            output_min=-0.3, output_max=1.0,
        )
        
        # Heading -> bank command
        self.heading_pid = PIDController(
            kp=g.heading_kp, ki=g.heading_ki, kd=g.heading_kd,
            output_min=-self.max_bank, output_max=self.max_bank,
        )
        
        # Bank -> aileron
        self.bank_pid = PIDController(
            kp=g.bank_kp, ki=g.bank_ki, kd=g.bank_kd,
            output_min=-self.max_aileron, output_max=self.max_aileron,
        )
        
        # Pitch -> elevator
        self.pitch_pid = PIDController(
            kp=g.pitch_kp, ki=g.pitch_ki, kd=g.pitch_kd,
            output_min=-self.max_elevator, output_max=self.max_elevator,
        )
    
    def reset(self):
        """Reset all controller states."""
        self.altitude_pid.reset()
        self.climb_rate_pid.reset()
        self.speed_pid.reset()
        self.cruise_speed_pid.reset()
        self.heading_pid.reset()
        self.bank_pid.reset()
        self.pitch_pid.reset()
        self._cruise_alt_integral = 0.0
    
    def compute_controls(
        self,
        state: Dict[str, float],
        phase: 'MissionPhase',
        dt: float,
    ) -> Dict[str, float]:
        """
        Compute control commands for current state and phase.
        
        Args:
            state: Current aircraft state with keys:
                - altitude: Current altitude AGL [m]
                - airspeed: True airspeed [m/s]
                - climb_rate: Vertical speed [m/s] (positive = up)
                - heading: Current heading [deg]
                - pitch: Pitch angle [deg]
                - bank: Bank angle [deg]
                - x, y: Position [m] (for waypoint tracking)
            phase: Current mission phase
            dt: Time step [s]
        
        Returns:
            Dict with control commands:
                - throttle: 0-1
                - elevator: degrees
                - aileron: degrees
                - rudder: degrees
                - brake: 0-1
        """
        from .mission_definition import MissionPhaseType
        
        # Default outputs
        controls = {
            'throttle': 0.0,
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0,
            'brake': 0.0,
        }
        
        phase_type = phase.phase_type
        phase_key = (phase_type, phase.name)
        if phase_key != self._last_phase_key:
            self.reset()
            self._last_phase_key = phase_key
        
        # Phase-specific control logic
        if phase_type == MissionPhaseType.GROUND_IDLE:
            controls = self._control_ground_idle(state, phase, dt)
        
        elif phase_type == MissionPhaseType.TAKEOFF_ROLL:
            controls = self._control_takeoff_roll(state, phase, dt)
        
        elif phase_type == MissionPhaseType.ROTATION:
            controls = self._control_rotation(state, phase, dt)
        
        elif phase_type == MissionPhaseType.CLIMB:
            controls = self._control_climb(state, phase, dt)
        
        elif phase_type == MissionPhaseType.CRUISE:
            controls = self._control_cruise(state, phase, dt)
        
        elif phase_type == MissionPhaseType.DESCENT:
            controls = self._control_descent(state, phase, dt)
        
        elif phase_type == MissionPhaseType.LOITER:
            controls = self._control_loiter(state, phase, dt)
        
        elif phase_type == MissionPhaseType.WAYPOINT_TRANSIT:
            controls = self._control_waypoint_transit(state, phase, dt)
        
        elif phase_type == MissionPhaseType.APPROACH:
            controls = self._control_approach(state, phase, dt)
        
        elif phase_type == MissionPhaseType.LANDING_FLARE:
            controls = self._control_landing_flare(state, phase, dt)
        
        elif phase_type == MissionPhaseType.LANDING_ROLL:
            controls = self._control_landing_roll(state, phase, dt)
        
        # Apply phase throttle limits
        controls['throttle'] = np.clip(
            controls['throttle'],
            phase.throttle_min,
            phase.throttle_max,
        )
        
        # Apply control surface limits
        controls['elevator'] = np.clip(controls['elevator'], -self.max_elevator, self.max_elevator)
        controls['aileron'] = np.clip(controls['aileron'], -self.max_aileron, self.max_aileron)
        controls['rudder'] = np.clip(controls['rudder'], -self.max_rudder, self.max_rudder)
        controls['brake'] = np.clip(controls['brake'], 0.0, 1.0)
        
        return controls
    
    # =========================================================================
    # Phase-Specific Control Methods
    # =========================================================================
    
    def _control_ground_idle(self, state, phase, dt) -> Dict[str, float]:
        """Ground idle: brakes on, zero throttle."""
        return {
            'throttle': 0.0,
            'elevator': 0.0,
            'aileron': 0.0,
            'rudder': 0.0,
            'brake': 1.0,
        }
    
    def _control_takeoff_roll(self, state, phase, dt) -> Dict[str, float]:
        """Takeoff roll: full throttle, maintain heading, slight nose-up."""
        # Heading control for runway tracking
        heading_error = self._normalize_heading_error(
            phase.target_heading or 0.0, state.get('heading', 0.0)
        )
        rudder = self.heading_pid.update(heading_error, dt) * 0.5  # Reduced authority on ground
        
        # Progressive elevator as speed builds
        airspeed = state.get('airspeed', 0.0)
        rotation_speed = phase.end_speed or 12.0
        elevator = 2.0 * (airspeed / rotation_speed)  # Gradual nose-up
        
        return {
            'throttle': 1.0,
            'elevator': np.clip(elevator, 0.0, 5.0),
            'aileron': 0.0,
            'rudder': np.clip(rudder, -5.0, 5.0),
            'brake': 0.0,
        }
    
    def _control_rotation(self, state, phase, dt) -> Dict[str, float]:
        """Rotation: smoothly pitch up for liftoff."""
        # Simple proportional control for rotation - no PID, just direct pitch command
        target_speed = phase.target_speed or phase.end_speed or 12.0
        current_speed = state.get('airspeed', 0.0)
        speed_error = target_speed - current_speed
        pitch_bias = np.clip(-0.5 * speed_error, 0.0, 5.0)
        target_pitch = 10.0 + pitch_bias  # Favor height over speed when fast
        current_pitch = state.get('pitch', 0.0)
        
        # Proportional elevator with rate limiting
        pitch_error = target_pitch - current_pitch
        elevator = 0.8 * pitch_error  # Proportional gain only
        elevator = np.clip(elevator, -10.0, 15.0)  # Limit authority
        
        return {
            'throttle': 1.0,
            'elevator': elevator,
            'aileron': 0.0,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_climb(self, state, phase, dt) -> Dict[str, float]:
        """Climb: prioritize climb rate, maintain safe speed."""
        # Simple proportional control for climb - avoid PID instability
        
        # Target climb rate
        if phase.target_climb_rate is not None:
            target_climb_rate = phase.target_climb_rate
        else:
            target_climb_rate = 2.0  # Default
        
        target_climb_rate = np.clip(target_climb_rate, 0.5, 4.0)
        
        # Current climb rate
        current_climb_rate = state.get('climb_rate', 0.0)

        # Target speed for climb
        target_speed = phase.target_speed or 15.0
        current_speed = state.get('airspeed', 10.0)
        max_climb_speed = target_speed * 1.1
        
        # Simple relationship: pitch ≈ base + correction for climb rate error
        target_pitch = 8.0 + 2.0 * (target_climb_rate - current_climb_rate)
        speed_error = target_speed - current_speed
        speed_pitch = np.clip(-0.5 * speed_error, 0.0, 5.0)
        if current_speed > max_climb_speed:
            overspeed = current_speed - max_climb_speed
            speed_pitch += np.clip(overspeed * 0.8, 0.0, 6.0)
        target_pitch += speed_pitch
        target_pitch = np.clip(target_pitch, 5.0, 15.0)  # Reasonable climb pitch
        
        # Elevator to achieve pitch - simple proportional
        current_pitch = state.get('pitch', 0.0)
        pitch_error = target_pitch - current_pitch
        elevator = 0.6 * pitch_error
        elevator = np.clip(elevator, -10.0, 15.0)
        
        # Speed control during climb (avoid overspeed)
        throttle = self.speed_pid.update(speed_error, dt)
        if throttle <= 0.0 and speed_error > 0.0:
            self.speed_pid.reset()
            throttle = self.speed_pid.update(speed_error, dt)
        if current_speed < target_speed * 0.9:
            throttle = max(throttle, 0.7)
        if current_speed > max_climb_speed:
            throttle = min(throttle, 0.3)
        elif current_speed > target_speed * 1.05:
            throttle = min(throttle, 0.4)
        if current_speed > target_speed * 1.15:
            throttle = min(throttle, 0.2)
        
        return {
            'throttle': throttle,
            'elevator': elevator,
            'aileron': 0.0,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_cruise(self, state, phase, dt) -> Dict[str, float]:
        """Cruise: altitude and speed hold with simplified control."""
        # Cruise altitude hold: PI on altitude -> climb rate -> pitch -> elevator
        if phase.target_altitude is not None:
            target_altitude = phase.target_altitude
        elif phase.end_altitude is not None:
            target_altitude = phase.end_altitude
        else:
            target_altitude = state.get('altitude', 100.0)
        alt_error = target_altitude - state.get('altitude', 0.0)
        self._cruise_alt_integral += alt_error * dt
        self._cruise_alt_integral = np.clip(self._cruise_alt_integral, -200.0, 200.0)

        climb_rate = state.get('climb_rate', 0.0)
        target_climb_rate = 0.3 * alt_error + 0.02 * self._cruise_alt_integral
        target_climb_rate = np.clip(target_climb_rate, -1.5, 1.5)
        climb_rate_error = target_climb_rate - climb_rate

        target_pitch = np.clip(3.0 * climb_rate_error, -8.0, 10.0)
        pitch_error = target_pitch - state.get('pitch', 0.0)
        elevator = np.clip(0.8 * pitch_error, -12.0, 12.0)
        
        # Speed -> throttle (PID speed hold)
        target_speed = phase.target_speed or 18.0
        speed_error = target_speed - state.get('airspeed', 0.0)
        throttle = self.cruise_speed_pid.update(speed_error, dt)
        
        # Heading hold
        aileron = self._compute_heading_hold(state, phase, dt)
        
        return {
            'throttle': throttle,
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_descent(self, state, phase, dt) -> Dict[str, float]:
        """Descent: controlled descent rate with speed management.
        
        Uses simple PD controller to achieve target descent rate.
        Positive elevator (nose up) reduces descent rate.
        Negative elevator (nose down) increases descent rate.
        """
        # Target descent rate (positive value = descending)
        target_descent_rate = phase.target_descent_rate or 2.0  # m/s down
        current_climb_rate = state.get('climb_rate', 0.0)  # positive = climbing
        
        # Error: if we want to descend at 2 m/s, climb_rate should be -2 m/s
        # climb_rate_error = desired - actual = (-2) - current
        target_climb_rate = -target_descent_rate
        climb_rate_error = target_climb_rate - current_climb_rate
        
        # Simple PD controller: 
        # - If climbing when should descend: climb_rate_error is negative -> need negative elevator (nose down)
        # - If descending too fast: climb_rate_error is positive -> need positive elevator (nose up)
        kp = 3.0  # degrees per (m/s) error
        kd = 0.5  # damping
        
        # Use pitch rate for damping (q)
        pitch_rate = state.get('q', 0.0)
        
        elevator = kp * climb_rate_error - kd * pitch_rate
        elevator = np.clip(elevator, -10.0, 10.0)  # Limit to ±10 degrees
        
        # Speed control: reduce throttle during descent
        target_speed = phase.target_speed or 15.0
        speed_error = target_speed - state.get('airspeed', 0.0)
        # Low throttle during descent, let gravity help
        throttle = 0.15 + 0.02 * speed_error
        
        aileron = self._compute_heading_hold(state, phase, dt)
        
        return {
            'throttle': np.clip(throttle, 0.0, 0.4),
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_loiter(self, state, phase, dt) -> Dict[str, float]:
        """Loiter: altitude hold with constant bank for orbit."""
        # Altitude hold
        target_altitude = phase.target_altitude or state.get('altitude', 100.0)
        alt_error = target_altitude - state.get('altitude', 0.0)
        target_climb_rate = self.altitude_pid.update(alt_error, dt)
        
        climb_rate_error = target_climb_rate - state.get('climb_rate', 0.0)
        target_pitch = self.climb_rate_pid.update(climb_rate_error, dt)
        
        pitch_error = target_pitch - state.get('pitch', 0.0)
        elevator = self.pitch_pid.update(pitch_error, dt)
        
        # Speed hold
        target_speed = phase.target_speed or 15.0
        speed_error = target_speed - state.get('airspeed', 0.0)
        throttle = 0.4 + self.speed_pid.update(speed_error, dt) * 0.3
        
        # Constant bank for orbit
        target_bank = phase.target_bank_angle or 15.0
        bank_error = target_bank - state.get('bank', 0.0)
        aileron = self.bank_pid.update(bank_error, dt)
        
        return {
            'throttle': np.clip(throttle, 0.2, 0.7),
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_waypoint_transit(self, state, phase, dt) -> Dict[str, float]:
        """Waypoint transit: navigate to waypoint with altitude/speed control."""
        if phase.waypoint is None:
            return self._control_cruise(state, phase, dt)
        
        wp = phase.waypoint
        
        # Compute bearing to waypoint
        dx = wp.x - state.get('x', 0.0)
        dy = wp.y - state.get('y', 0.0)
        target_heading = np.degrees(np.arctan2(dx, dy)) % 360.0
        
        # Heading -> bank -> aileron
        heading_error = self._normalize_heading_error(target_heading, state.get('heading', 0.0))
        target_bank = self.heading_pid.update(heading_error, dt)
        target_bank = np.clip(target_bank, -phase.bank_max, phase.bank_max)
        
        bank_error = target_bank - state.get('bank', 0.0)
        aileron = self.bank_pid.update(bank_error, dt)
        
        # Altitude control
        target_altitude = phase.target_altitude or wp.altitude
        alt_error = target_altitude - state.get('altitude', 0.0)
        target_climb_rate = self.altitude_pid.update(alt_error, dt)
        
        climb_rate_error = target_climb_rate - state.get('climb_rate', 0.0)
        target_pitch = self.climb_rate_pid.update(climb_rate_error, dt)
        
        pitch_error = target_pitch - state.get('pitch', 0.0)
        elevator = self.pitch_pid.update(pitch_error, dt)
        
        # Speed control
        target_speed = phase.target_speed or wp.speed or 18.0
        speed_error = target_speed - state.get('airspeed', 0.0)
        throttle = 0.5 + self.speed_pid.update(speed_error, dt)
        
        return {
            'throttle': np.clip(throttle, 0.3, 0.9),
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_approach(self, state, phase, dt) -> Dict[str, float]:
        """Approach: glide slope tracking with simple PD control."""
        # Target descent rate for approach (gentler than cruise descent)
        target_descent_rate = phase.target_descent_rate or 1.5  # m/s down
        current_climb_rate = state.get('climb_rate', 0.0)
        
        target_climb_rate = -target_descent_rate
        climb_rate_error = target_climb_rate - current_climb_rate
        
        # Simple PD controller for approach
        kp = 4.0  # slightly more aggressive for approach
        kd = 0.5
        pitch_rate = state.get('q', 0.0)
        
        elevator = kp * climb_rate_error - kd * pitch_rate
        elevator = np.clip(elevator, -10.0, 10.0)
        
        # Speed control (approach speed, low throttle)
        target_speed = phase.target_speed or 13.0
        speed_error = target_speed - state.get('airspeed', 0.0)
        throttle = 0.15 + 0.02 * speed_error
        
        # Heading hold to runway
        aileron = self._compute_heading_hold(state, phase, dt)
        
        return {
            'throttle': np.clip(throttle, 0.0, 0.4),
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_landing_flare(self, state, phase, dt) -> Dict[str, float]:
        """Landing flare: reduce sink rate while descending to touchdown.
        
        Uses simple PD controller to achieve target descent rate.
        """
        target_descent_rate = phase.target_descent_rate or 0.5
        current_alt = state.get('altitude', 0.0)
        current_climb_rate = state.get('climb_rate', 0.0)
        
        # Modulate target based on altitude - descend faster when higher
        if current_alt > 5.0:
            target_climb_rate = -1.5  # Still descending
        elif current_alt > 2.0:
            target_climb_rate = -0.8  # Slowing descent
        else:
            target_climb_rate = -target_descent_rate  # Gentle touchdown
        
        climb_rate_error = target_climb_rate - current_climb_rate
        
        # Simple PD controller for flare
        # Higher gains for precise touchdown control
        kp = 5.0  # more aggressive for flare
        kd = 1.0  # more damping
        pitch_rate = state.get('q', 0.0)
        
        elevator = kp * climb_rate_error - kd * pitch_rate
        elevator = np.clip(elevator, -10.0, 10.0)  # Limit to ±10 degrees
        
        # Throttle - near idle during flare
        throttle = 0.05
        
        # Heading hold to runway centerline
        aileron = self._compute_heading_hold(state, phase, dt)
        
        return {
            'throttle': throttle,
            'elevator': elevator,
            'aileron': aileron,
            'rudder': 0.0,
            'brake': 0.0,
        }
    
    def _control_landing_roll(self, state, phase, dt) -> Dict[str, float]:
        """Landing roll: zero throttle, full brake."""
        # Heading control with rudder
        heading_error = self._normalize_heading_error(
            phase.target_heading or 0.0, state.get('heading', 0.0)
        )
        rudder = heading_error * 0.5
        
        return {
            'throttle': 0.0,
            'elevator': -2.0,  # Slight nose-down for ground contact
            'aileron': 0.0,
            'rudder': np.clip(rudder, -10.0, 10.0),
            'brake': 1.0,
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _normalize_heading_error(self, target: float, current: float) -> float:
        """Compute heading error normalized to [-180, 180]."""
        error = target - current
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        return error
    
    def _compute_heading_hold(self, state, phase, dt) -> float:
        """Compute aileron for heading hold."""
        target_heading = phase.target_heading
        if target_heading is None:
            # Maintain current heading
            return 0.0
        
        heading_error = self._normalize_heading_error(target_heading, state.get('heading', 0.0))
        target_bank = self.heading_pid.update(heading_error, dt)
        target_bank = np.clip(target_bank, -phase.bank_max, phase.bank_max)
        
        bank_error = target_bank - state.get('bank', 0.0)
        aileron = self.bank_pid.update(bank_error, dt)
        
        return aileron
