"""
Python wrapper for CRM C++ physics engine.

Provides a unified interface that uses the compiled C++ bindings when available,
or falls back to simplified dynamics for development/testing.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Try to import compiled C++ bindings
try:
    from . import crm_python
    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False


@dataclass
class CatheterParams:
    """Catheter physical parameters."""
    # Damping coefficients (6 elements): [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
    # Values from CRMDYN_test.cpp for consistency with C++ dynamics
    damping: np.ndarray = field(default_factory=lambda: np.array([
        12.1761626666366, 12.1761626666366, 284.429938756989,
        0.0304776127617393, 0.0304776127617393, 0.00502712804532508
    ]))

    # Tube radii [outer, inner] in mm
    radii: np.ndarray = field(default_factory=lambda: np.array([1.5875, 0.9906]))

    # Material properties [Young's modulus, Shear modulus] in MPa
    E: np.ndarray = field(default_factory=lambda: np.array([5.3948, 2.3881]))

    # Coil alignment angles [theta_x, theta_y] in radians
    coil_align: np.ndarray = field(default_factory=lambda: np.array([3.1631, -3.0989]))

    # Turn-area matrix diagonal [axial, side1, side2]
    coil_turnarea: np.ndarray = field(default_factory=lambda: np.array([3.8037, 2.7781, 1.7728]))

    # Coil mass in kg
    mass: float = 7.0293e-6

    # Segment lengths [proximal_flex, coil, distal_flex] in mm
    segment_lengths: np.ndarray = field(default_factory=lambda: np.array([18.5, 18.3, 57.50]))

    # Magnetic field strength (T)
    B0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.5]))

    # Number of actuator sets
    num_act_set: int = 1

    @classmethod
    def from_dict(cls, params: Dict) -> 'CatheterParams':
        """Create parameters from dictionary."""
        default_damping = [
            12.1761626666366, 12.1761626666366, 284.429938756989,
            0.0304776127617393, 0.0304776127617393, 0.00502712804532508
        ]
        return cls(
            damping=np.array(params.get('damping', default_damping)),
            radii=np.array(params.get('radii', [1.5875, 0.9906])),
            E=np.array(params.get('E', [5.3948, 2.3881])),
            coil_align=np.array(params.get('coil_align', [3.1631, -3.0989])),
            coil_turnarea=np.array(params.get('coil_turnarea', [3.8037, 2.7781, 1.7728])),
            mass=params.get('mass', 7.0293e-6),
            segment_lengths=np.array(params.get('segment_lengths', [18.5, 18.3, 57.50])),
            B0=np.array(params.get('B0', [0.0, 0.0, 1.5]))
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'damping': self.damping.tolist(),
            'radii': self.radii.tolist(),
            'E': self.E.tolist(),
            'coil_align': self.coil_align.tolist(),
            'coil_turnarea': self.coil_turnarea.tolist(),
            'mass': self.mass,
            'segment_lengths': self.segment_lengths.tolist(),
            'B0': self.B0.tolist()
        }


@dataclass
class CatheterState:
    """Catheter state variables."""
    # Tip position (spatial frame) in mm
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 80.0]))

    # Tip velocity in mm/s
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Rotation matrix at tip (3x3, flattened)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3).flatten())

    # Curvature at base
    curvature: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Convergence flag
    converged: bool = True

    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix as 3x3 array."""
        return self.rotation.reshape(3, 3)

    def copy(self) -> 'CatheterState':
        """Create a deep copy of the state."""
        return CatheterState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            rotation=self.rotation.copy(),
            curvature=self.curvature.copy(),
            converged=self.converged
        )


class CRMWrapper:
    """
    Wrapper for CRM C++ physics engine.

    Uses compiled C++ bindings when available, otherwise falls back to
    simplified dynamics model.
    """

    def __init__(
        self,
        param_file: Optional[str] = None,
        config_file: Optional[str] = None,
        params: Optional[CatheterParams] = None,
        use_cpp: bool = True,
        damping: Optional[np.ndarray] = None,
        flip_third_current: bool = False,
        disable_cpp_fallback: bool = False,
        strict_cpp: bool = False
    ):
        """
        Initialize CRM wrapper.

        Args:
            param_file: Path to catheter parameter file
            config_file: Path to catheter configuration file
            params: Catheter parameters (for simplified model)
            use_cpp: Whether to use C++ bindings (if available)
            damping: Optional damping coefficients
            flip_third_current: Apply dataset-specific flip on third current channel
        """
        self.params = params or CatheterParams()
        self.use_cpp = use_cpp and HAS_CPP_BINDINGS
        self.initialized = False
        self._initial_use_cpp = self.use_cpp
        self._cpp_available = False
        self._cpp_failures = 0
        self._cpp_disabled_due_to_failure = False
        self._flip_third_current = flip_third_current
        self._disable_cpp_fallback = disable_cpp_fallback
        self._strict_cpp = strict_cpp
        # Default damping matches values used in CRMDYN_test.cpp
        self._default_damping = damping if damping is not None else np.array([
            12.1761626666366, 12.1761626666366, 284.429938756989,
            0.0304776127617393, 0.0304776127617393, 0.00502712804532508
        ])

        # Default file paths
        if param_file is None:
            # Use dynamics-tuned parameters by default for stability
            param_file = "data/catheter_params/CatheterParameterSet_1_dyn.txt"
        if config_file is None:
            config_file = "data/catheter_params/CatheterSpatialConfiguration_1.txt"

        self.param_file = param_file
        self.config_file = config_file

        if self.use_cpp:
            self._init_cpp()
        else:
            self._init_simplified()
        self._cpp_available = self.use_cpp and self.initialized

    def _init_cpp(self):
        """Initialize C++ bindings."""
        self._cpp_kinematics = crm_python.CRMKinematics()
        self._cpp_dynamics = crm_python.CRMDynamics()

        # Try to load data/simulation_parameters
        if Path(self.param_file).exists() and Path(self.config_file).exists():
            self.initialized = self._cpp_kinematics.load_parameters(
                self.param_file, self.config_file
            )
            if self.initialized:
                self._cpp_dynamics.load_parameters(self.param_file, self.config_file)
                # Apply default damping to stabilize coil dynamics
                self._cpp_dynamics.set_damping(self._default_damping)
                # Slightly smaller integration step improves convergence robustness
                self._cpp_kinematics.integration_step_size = 0.1
                self._cpp_dynamics.integration_step_size = 0.1
        else:
            print(f"Warning: Parameter files not found, falling back to simplified model")
            self.use_cpp = False
            self._init_simplified()

    def _init_simplified(self):
        """Initialize simplified dynamics model."""
        # Simplified model.parameters
        self._mass = 0.001  # kg
        self._damping = 10.0  # N·s/m
        self._stiffness = 100.0  # N/m
        self._current_to_force = np.array([5.0, 5.0, 3.0])  # N/A
        self._equilibrium = np.array([0.0, 0.0, 80.0])  # mm
        self._dt = 0.02  # Default timestep

        # State
        self._position = self._equilibrium.copy()
        self._velocity = np.zeros(3)
        self._rotation = np.eye(3)

        self.initialized = True

    def forward_kinematics(
        self,
        currents: np.ndarray,
        insertion_length: float = 94.3,
        deltau0_initialguess: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute forward kinematics.

        Args:
            currents: Applied currents (num_act_set * 3,)
            insertion_length: Inserted length in mm

        Returns:
            Dictionary with tip_position, tip_rotation, delta_u0, converged
        """
        currents = np.asarray(currents, dtype=np.float64).flatten()
        if self._flip_third_current and len(currents) >= 3:
            currents = currents.copy()
            currents[2] *= -1.0

        if self.use_cpp and self.initialized:
            if deltau0_initialguess is not None and hasattr(self._cpp_kinematics, "forward_kinematics_with_guess"):
                return self._cpp_kinematics.forward_kinematics_with_guess(
                    currents, insertion_length, np.asarray(deltau0_initialguess, dtype=np.float64).reshape(3)
                )
            return self._cpp_kinematics.forward_kinematics(currents, insertion_length)
        else:
            if self._strict_cpp:
                raise RuntimeError("C++ FK requested (strict_cpp=True) but C++ path is unavailable.")
            # Simplified FK: tip deflection proportional to current
            tip_pos = self._equilibrium.copy()
            for i in range(min(len(currents), 3)):
                tip_pos[i] += currents[i] * self._current_to_force[i]

            return {
                'tip_position': tip_pos,
                'tip_rotation': np.eye(3).flatten(),
                'delta_u0': np.zeros(3),
                'potential_energy': 0.0,
                'converged': True
            }

    def compute_jacobian(
        self,
        currents: np.ndarray,
        insertion_length: float = 94.3
    ) -> np.ndarray:
        """
        Compute analytical Jacobian.

        Args:
            currents: Applied currents
            insertion_length: Inserted length in mm

        Returns:
            Jacobian matrix (dp/dI)
        """
        currents = np.asarray(currents, dtype=np.float64).flatten()
        if self._flip_third_current and len(currents) >= 3:
            currents = currents.copy()
            currents[2] *= -1.0

        if self.use_cpp and self.initialized:
            return self._cpp_kinematics.compute_jacobian(currents, insertion_length)
        else:
            if self._strict_cpp:
                raise RuntimeError("C++ Jacobian requested (strict_cpp=True) but C++ path is unavailable.")
            # Simplified Jacobian: constant diagonal mapping
            return np.diag(self._current_to_force)

    def fk_and_jacobian(
        self,
        currents: np.ndarray,
        insertion_length: float = 94.3,
        deltau0_initialguess: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Fast path: compute FK and analytical Jacobian in one call (no second FK solve).

        Returns a dict with at least: tip_position, tip_rotation, delta_u0, converged, jacobian.
        """
        currents = np.asarray(currents, dtype=np.float64).flatten()
        if self._flip_third_current and len(currents) >= 3:
            currents = currents.copy()
            currents[2] *= -1.0

        if not (self.use_cpp and self.initialized):
            raise RuntimeError("fk_and_jacobian requires C++ bindings (use_cpp=True).")

        guess = np.array([], dtype=np.float64)
        if deltau0_initialguess is not None:
            guess = np.asarray(deltau0_initialguess, dtype=np.float64).reshape(3)
        if not hasattr(self._cpp_kinematics, "fk_and_jacobian"):
            # Backward-compatible fallback (slower).
            fk = self.forward_kinematics(currents, insertion_length, deltau0_initialguess=deltau0_initialguess)
            J = self.compute_jacobian(currents, insertion_length)
            fk["jacobian"] = J
            return fk

        return self._cpp_kinematics.fk_and_jacobian(currents, insertion_length, guess)

    def step_dynamics(
        self,
        currents: np.ndarray,
        insertion_length: float = 94.3,
        dt: Optional[float] = None
    ) -> Dict:
        """
        Step dynamics forward in time.

        Args:
            currents: Applied currents
            insertion_length: Inserted length in mm
            dt: Time step (uses default if None)

        Returns:
            Dictionary with tip_position, tip_velocity, converged
        """
        currents = np.asarray(currents, dtype=np.float64).flatten()
        if self._flip_third_current and len(currents) >= 3:
            currents = currents.copy()
            currents[2] *= -1.0

        if self.use_cpp and self.initialized:
            if dt is not None:
                self._cpp_dynamics.set_timestep(dt)
            
            # First attempt
            result = self._cpp_dynamics.step(currents, insertion_length)

            # If the first attempt fails, try to recover by re-initializing from FK
            if not result.get('converged', True):
                # print("Dynamics failed. Attempting recovery with FK re-initialization...")
                init_success = self.initialize_dynamics(currents, insertion_length)

                if init_success:
                    # Second attempt after re-initialization
                    result = self._cpp_dynamics.step(currents, insertion_length)

            # If still not converged after recovery attempt, handle failure
            if not result.get('converged', True):
                self._cpp_failures += 1
                if self._cpp_failures == 1: # Only print warning once to avoid spam
                    localmin = result.get('localmin', 'N/A')
                    step_size = self._cpp_dynamics.integration_step_size if hasattr(self._cpp_dynamics, 'integration_step_size') else 'N/A'
                    print(f"Warning: CRM C++ dynamics did not converge. "
                          f"currents={currents}, insertion_length={insertion_length}, "
                          f"dt={dt}, step_size={step_size}, localmin={localmin}. "
                          f"Falling back to simplified model.")
                
                if self._disable_cpp_fallback:
                    return result
                
                self._fallback_to_simplified(result)
                return self._step_simplified(currents, dt)
            
            # If converged, reset failure count
            self._cpp_failures = 0
            return result
        else:
            if self._strict_cpp:
                raise RuntimeError("C++ dynamics requested (strict_cpp=True) but C++ path is unavailable.")
            return self._step_simplified(currents, dt)

    def _step_simplified(self, currents: np.ndarray, dt: Optional[float]) -> Dict:
        """Step the simplified mass-spring dynamics."""
        if dt is None:
            dt = self._dt

        # Magnetic force
        F_magnetic = np.zeros(3)
        for i in range(min(len(currents), 3)):
            F_magnetic[i] = currents[i] * self._current_to_force[i]

        # Spring force
        displacement = (self._position - self._equilibrium) / 1000  # mm to m
        F_spring = -self._stiffness * displacement

        # Damping force
        F_damping = -self._damping * self._velocity / 1000

        # Total force and acceleration
        F_total = F_magnetic + F_spring + F_damping
        acceleration = F_total / self._mass * 1000  # mm/s^2
        # Clamp acceleration to avoid numerical blow-up when falling back from C++
        acc_norm = np.linalg.norm(acceleration)
        if acc_norm > 5e4:  # 50 m/s^2 in mm units
            acceleration = acceleration / acc_norm * 5e4

        # Semi-implicit Euler
        self._velocity = self._velocity + acceleration * dt
        # Clamp velocity to avoid numerical blow-up (max 500 mm/s)
        vel_norm = np.linalg.norm(self._velocity)
        if vel_norm > 500.0:
            self._velocity = self._velocity / vel_norm * 500.0
        self._position = self._position + self._velocity * dt

        return {
            'tip_position': self._position.copy(),
            'tip_velocity': self._velocity.copy(),
            'converged': True
        }

    def reset(self):
        """Reset dynamics state to zeros."""
        # Re-enable C++ path if it was available before a fallback
        if self._cpp_available and not self._cpp_disabled_due_to_failure:
            self.use_cpp = True
        self._cpp_failures = 0
        if self.use_cpp and self.initialized:
            self._cpp_dynamics.reset()
        else:
            self._position = self._equilibrium.copy()
            self._velocity = np.zeros(3)
            self._rotation = np.eye(3)

    def _fallback_to_simplified(self, last_result: Optional[Dict] = None):
        """
        Switch from C++ dynamics to simplified model after a convergence failure.

        Prevents repeated C++ solver attempts (and log spam) while keeping the
        simulation running from the latest known state.
        """
        # Initialize simplified model.parameters/state
        self._init_simplified()
        if last_result is not None:
            # If the solver reported non-convergence, discard its state entirely
            if not last_result.get('converged', True):
                last_result = None
        if last_result is not None:
            pos = np.array(last_result.get('tip_position', self._equilibrium), dtype=float).copy()
            vel = np.array(last_result.get('tip_velocity', np.zeros(3)), dtype=float).copy()
            # Reset to equilibrium if the failed step produced invalid or extreme values
            if (not np.all(np.isfinite(pos))) or (np.linalg.norm(pos) > 1e3):
                pos = self._equilibrium.copy()
            if (not np.all(np.isfinite(vel))) or (np.linalg.norm(vel) > 1e3):
                vel = np.zeros(3)
            self._position = pos
            self._velocity = vel
        else:
            self._position = self._equilibrium.copy()
            self._velocity = np.zeros(3)
        self._rotation = np.eye(3)
        self.use_cpp = False
        self._cpp_disabled_due_to_failure = True
        # Keep knowledge that C++ exists so reset() can re-enable if desired

    def initialize_dynamics(
        self,
        currents: np.ndarray,
        insertion_length: float = 94.3
    ) -> bool:
        """
        Initialize dynamics from a valid forward kinematics solution.

        This MUST be called before step_dynamics() when using C++ bindings
        to ensure the solver starts from a physically valid configuration
        and avoids convergence issues ("Coil integration Unbounded!!").

        Args:
            currents: Applied currents for initial configuration
            insertion_length: Inserted length in mm

        Returns:
            True if initialization succeeded, False otherwise.
        """
        currents = np.asarray(currents, dtype=np.float64).flatten()
        if self._flip_third_current and len(currents) >= 3:
            currents = currents.copy()
            currents[2] *= -1.0

        if self.use_cpp and self.initialized:
            # Try requested currents first
            success = self._cpp_dynamics.initialize_from_kinematics(currents, insertion_length)
            if not success:
                # Retry with reduced step and zero currents for robustness
                prev_step = self._cpp_dynamics.integration_step_size
                self._cpp_dynamics.integration_step_size = min(prev_step, 0.05)
                zero_curr = np.zeros_like(currents)
                success = self._cpp_dynamics.initialize_from_kinematics(zero_curr, insertion_length)
                # Restore step size
                self._cpp_dynamics.integration_step_size = prev_step

            if success:
                self._cpp_disabled_due_to_failure = False
                self.use_cpp = True
                self._cpp_failures = 0
            return success
        else:
            # For simplified model, just compute initial position from FK
            result = self.forward_kinematics(currents, insertion_length)
            self._position = result['tip_position'].copy()
            self._velocity = np.zeros(3)
            return True

    def bvp_initialize_with_seed(
        self,
        currents: np.ndarray,
        insertion_length: float,
        p_L: np.ndarray,
        R_L: np.ndarray,
        xf_seed: np.ndarray,
        mL: Optional[np.ndarray] = None,
        nL: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        integration_step: Optional[float] = None
    ) -> bool:
        """
        Advanced initializer that mimics CRMDYNTest: seed BVP with known coil/tip state.

        Args:
            currents: applied currents (len 3)
            insertion_length: insertion length in mm
            p_L: coil position seed (3,)
            R_L: coil rotation seed (9,)
            xf_seed: tip state seed (15,) matching CRMDYN_test.cpp order
            mL, nL: optional moment/force seeds (3,)
            damping: optional damping to set before init
            dt: optional timestep to set
            integration_step: optional integration step size to set
        """
        if not (self.use_cpp and self.initialized):
            return False
        if damping is not None:
            self.set_damping(damping)
        if dt is not None:
            self._cpp_dynamics.dt = dt
        if integration_step is not None:
            self._cpp_dynamics.integration_step_size = integration_step

        kwargs = {}
        if mL is not None:
            kwargs['mL_in'] = np.asarray(mL, dtype=np.float64)
        if nL is not None:
            kwargs['nL_in'] = np.asarray(nL, dtype=np.float64)
        try:
            self._cpp_dynamics.initialize_from_seed(
                np.asarray(currents, dtype=np.float64),
                insertion_length,
                np.asarray(p_L, dtype=np.float64),
                np.asarray(R_L, dtype=np.float64),
                np.asarray(xf_seed, dtype=np.float64),
                **kwargs,
            )
            self._cpp_disabled_due_to_failure = False
            self._cpp_failures = 0
            return True
        except Exception:
            return False

    def get_tip_position(self) -> np.ndarray:
        """Get current tip position."""
        if self.use_cpp and self.initialized:
            return self._cpp_dynamics.get_tip_position()
        else:
            return self._position.copy()

    def set_damping(self, damping: np.ndarray):
        """Set damping coefficients."""
        if self.use_cpp and self.initialized:
            self._cpp_dynamics.set_damping(np.asarray(damping, dtype=np.float64))
        else:
            self._damping = float(np.mean(damping))

    def set_timestep(self, dt: float):
        """Set simulation timestep."""
        if self.use_cpp and self.initialized:
            self._cpp_dynamics.set_timestep(dt)
        else:
            self._dt = dt

    def debug_seed_dynamics(
        self,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        R: np.ndarray,
        xf: np.ndarray,
        mL: Optional[np.ndarray] = None,
        nL: Optional[np.ndarray] = None
    ):
        """
        Debug helper to manually seed C++ dynamics state.

        This mirrors the hardcoded seeds used in C++ tests and should only be
        called in diagnostics; production code should rely on normal initialization.
        """
        if not (self.use_cpp and self.initialized):
            raise RuntimeError("C++ dynamics not available for debug seeding.")
        kwargs = {}
        if mL is not None:
            kwargs['mL_in'] = np.asarray(mL, dtype=np.float64)
        if nL is not None:
            kwargs['nL_in'] = np.asarray(nL, dtype=np.float64)
        self._cpp_dynamics.debug_seed_state(
            np.asarray(v, dtype=np.float64),
            np.asarray(w, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
            np.asarray(R, dtype=np.float64),
            np.asarray(xf, dtype=np.float64),
            **kwargs
        )

    @property
    def is_using_cpp(self) -> bool:
        """Check if using C++ bindings."""
        return self.use_cpp and self.initialized


class CRMSimulator:
    """
    High-level simulator for catheter trajectories.

    Provides convenient methods for simulating trajectories and
    generating training data.
    """

    def __init__(
        self,
        param_file: Optional[str] = None,
        config_file: Optional[str] = None,
        dt: float = 0.02,
        use_cpp: bool = True,
        damping: Optional[np.ndarray] = None,
        flip_third_current: bool = False
    ):
        """
        Initialize simulator.

        Args:
            param_file: Path to catheter parameter file
            config_file: Path to catheter configuration file
            dt: Simulation timestep
            use_cpp: Whether to use C++ bindings
        """
        self.wrapper = CRMWrapper(
            param_file=param_file,
            config_file=config_file,
            use_cpp=use_cpp,
            damping=damping,
            flip_third_current=flip_third_current
        )
        self.dt = dt
        self.wrapper.set_timestep(dt)
        # Ensure damping is applied for C++ dynamics
        if damping is not None:
            self.wrapper.set_damping(damping)
        self.state = CatheterState()
        self.history = []

    def reset(
        self,
        initial_position: Optional[np.ndarray] = None,
        initial_currents: Optional[np.ndarray] = None,
        insertion_length: float = 94.3
    ):
        """
        Reset simulator to initial state.

        For C++ dynamics, this initializes from a forward kinematics solution
        to ensure the solver starts from a valid physical configuration.

        Args:
            initial_position: Optional initial tip position (for simplified model)
            initial_currents: Initial currents for FK initialization (default: zeros)
            insertion_length: Insertion length for FK initialization
        """
        # First reset internal state
        self.wrapper.reset()

        # For C++ dynamics, initialize from FK to avoid convergence issues
        if self.wrapper.is_using_cpp:
            if initial_currents is None:
                initial_currents = np.zeros(3)
            success = self.wrapper.initialize_dynamics(initial_currents, insertion_length)
            if not success:
                print("Warning: Dynamics initialization from FK failed; falling back to simplified dynamics.")
                self.wrapper._fallback_to_simplified()

        # Set position
        if initial_position is not None:
            self.state.position = np.array(initial_position)
        else:
            self.state.position = self.wrapper.get_tip_position()

        self.state.velocity = np.zeros(3)
        self.history = []

    def step(self, currents: np.ndarray, insertion_length: float = 94.3) -> CatheterState:
        """
        Execute one simulation step.

        Args:
            currents: Coil currents [I1, I2, I3]
            insertion_length: Inserted length in mm

        Returns:
            New state after step
        """
        result = self.wrapper.step_dynamics(currents, insertion_length)
        self.state.position = result['tip_position']
        self.state.velocity = result['tip_velocity']
        self.state.converged = result['converged']

        self.history.append({
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'currents': np.asarray(currents).copy()
        })
        return self.state

    def simulate_trajectory(
        self,
        currents_sequence: np.ndarray,
        insertion_length: float = 94.3,
        initial_position: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate a full trajectory.

        Args:
            currents_sequence: Array of currents (T, 3) or (3, T)
            insertion_length: Inserted length in mm
            initial_position: Initial tip position (optional)

        Returns:
            Dict with trajectory data
        """
        # Ensure shape is (T, 3)
        currents = np.array(currents_sequence)
        if currents.shape[0] == 3 and currents.shape[1] != 3:
            currents = currents.T

        T = len(currents)

        # Reset
        self.reset(
            initial_position=initial_position,
            insertion_length=insertion_length
        )

        # Storage
        positions = np.zeros((T + 1, 3))
        velocities = np.zeros((T + 1, 3))
        times = np.arange(T + 1) * self.dt

        # Initial state
        positions[0] = self.state.position
        velocities[0] = self.state.velocity

        # Simulate
        for t in range(T):
            state = self.step(currents[t], insertion_length)
            positions[t + 1] = state.position
            velocities[t + 1] = state.velocity

        return {
            'positions': positions,
            'velocities': velocities,
            'times': times,
            'currents': currents,
            'dt': self.dt,
            'num_steps': T
        }

    def generate_random_trajectories(
        self,
        num_trajectories: int = 100,
        trajectory_length: int = 200,
        max_current: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Generate random trajectories for training data.

        Args:
            num_trajectories: Number of trajectories to generate
            trajectory_length: Length of each trajectory
            max_current: Maximum current magnitude

        Returns:
            Dict with all trajectory data
        """
        all_positions = []
        all_currents = []

        for _ in range(num_trajectories):
            # Generate smooth random currents
            currents = self._generate_smooth_currents(trajectory_length, max_current)
            result = self.simulate_trajectory(currents)
            all_positions.append(result['positions'])
            all_currents.append(result['currents'])

        return {
            'positions': np.array(all_positions),  # (num_traj, traj_len+1, 3)
            'currents': np.array(all_currents),    # (num_traj, traj_len, 3)
            'num_trajectories': num_trajectories,
            'trajectory_length': trajectory_length
        }

    def _generate_smooth_currents(
        self,
        length: int,
        max_current: float
    ) -> np.ndarray:
        """Generate smooth random current trajectory."""
        t = np.linspace(0, 2 * np.pi, length)
        currents = np.zeros((length, 3))

        for i in range(3):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.1, 1.0) * max_current
            currents[:, i] = amplitude * np.sin(freq * t + phase)

        return currents

    @property
    def is_using_cpp(self) -> bool:
        """Check if using C++ bindings."""
        return self.wrapper.is_using_cpp


# Convenience function
def create_simulator(use_cpp: bool = True, dt: float = 0.02) -> CRMSimulator:
    """
    Create a CRM simulator with default parameters.

    Args:
        use_cpp: Whether to use C++ bindings
        dt: Simulation timestep

    Returns:
        CRMSimulator instance
    """
    return CRMSimulator(dt=dt, use_cpp=use_cpp)


if __name__ == "__main__":
    print("Testing CRM Wrapper...")
    print(f"C++ bindings available: {HAS_CPP_BINDINGS}")

    # Test wrapper
    wrapper = CRMWrapper(use_cpp=True)
    print(f"Using C++ bindings: {wrapper.is_using_cpp}")

    # Test forward kinematics
    currents = np.array([0.1, 0.0, 0.0])
    result = wrapper.forward_kinematics(currents, insertion_length=94.3)
    print(f"\nForward kinematics result:")
    print(f"  Tip position: {result['tip_position']}")
    print(f"  Converged: {result['converged']}")

    # Test dynamics with proper initialization
    print("\nTesting dynamics with FK initialization...")
    wrapper.reset()
    # IMPORTANT: Initialize from FK before stepping dynamics
    init_success = wrapper.initialize_dynamics(currents, insertion_length=94.3)
    print(f"  Dynamics initialization: {'success' if init_success else 'failed'}")

    for i in range(10):
        result = wrapper.step_dynamics(currents, insertion_length=94.3)
        if not result['converged']:
            print(f"  Step {i}: convergence failed")
            break
    print(f"\nAfter 10 dynamics steps:")
    print(f"  Tip position: {result['tip_position']}")
    print(f"  Tip velocity: {result['tip_velocity']}")
    print(f"  Converged: {result['converged']}")

    # Test simulator (uses automatic FK initialization in reset)
    print("\nTesting simulator...")
    sim = CRMSimulator(dt=0.02, use_cpp=True)

    # Generate current sequence
    T = 100
    t = np.linspace(0, 2, T)
    currents_seq = np.column_stack([
        0.1 * np.sin(2 * np.pi * 0.5 * t),
        0.1 * np.cos(2 * np.pi * 0.5 * t),
        np.zeros(T)
    ])

    trajectory = sim.simulate_trajectory(currents_seq)
    print(f"Simulated trajectory: {trajectory['positions'].shape}")
    print(f"Final position: {trajectory['positions'][-1]}")

    # Check for NaN values (indicates unbounded integration)
    has_nan = np.any(np.isnan(trajectory['positions']))
    print(f"Contains NaN values: {has_nan}")

    print("\nCRM Wrapper testing complete!")
