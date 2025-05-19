import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
import math
import os
# import json # Not needed directly in sim bundle if AppSimConfig is passed fully formed
from collections import deque
from typing import Optional, List, Dict, Any, Tuple, Literal, Union

from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import imageio.v2 as imageio
# from PIL import Image # imageio might need Pillow - keep if imageio has issues
import glob
from matplotlib.patches import Ellipse # For PF visualization


# --- Core Constants ---
CORE_STATE_DIM = 9 # agent_x, agent_y, agent_vx, agent_vy, agent_heading_rad, landmark_x, landmark_y, landmark_depth, current_range
CORE_ACTION_DIM = 1 # yaw_change_normalized
TRAJECTORY_REWARD_DIM = 1 # For world state encoding

# --- Configuration Models (Simplified for Simulation) ---

class LocationConfig(BaseModel):
    x: float = 0.0
    y: float = 0.0
    depth: float = 0.0

class VelocityConfig(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class RandomizationRangeConfig(BaseModel):
    x_range: Tuple[float, float] = Field((-100.0, 100.0))
    y_range: Tuple[float, float] = Field((-100.0, 100.0))
    depth_range: Tuple[float, float] = Field((0.0, 300.0))

class VelocityRandomizationRangeConfig(BaseModel):
    vx_range: Tuple[float, float] = Field((-0.5, 0.5))
    vy_range: Tuple[float, float] = Field((-0.5, 0.5))
    vz_range: Tuple[float, float] = Field((-0.1, 0.1))

class ParticleFilterSimConfig(BaseModel):
    num_particles: int = Field(1000)
    initial_range_stddev: float = Field(0.02)
    initial_velocity_guess: float = Field(0.1)
    max_particle_range: float = Field(250.0)
    process_noise_pos: float = Field(0.02)
    process_noise_orient: float = Field(0.2)
    process_noise_vel: float = Field(0.02)
    measurement_noise_stddev: float = Field(5.0)
    resampling_method: int = Field(2)
    pf_eval_max_mean_range_error_factor: float = Field(0.1) 
    pf_eval_dispersion_threshold: float = Field(5.0)      

class LeastSquaresSimConfig(BaseModel):
    history_size: int = Field(10)
    min_points_required: int = Field(3)
    location_smoothing_factor: float = Field(0.8)
    position_buffer_size: int = Field(5)
    velocity_smoothing: int = Field(3)
    min_observer_movement: float = Field(0.5)

class SACSimConfig(BaseModel):
    state_dim: int = Field(CORE_STATE_DIM)
    action_dim: int = Field(CORE_ACTION_DIM)
    hidden_dims: List[int] = Field([64, 64])
    log_std_min: int = Field(-20)
    log_std_max: int = Field(1)
    use_rnn: bool = Field(False)
    rnn_type: Literal['lstm', 'gru'] = Field('lstm')
    rnn_hidden_size: int = Field(128)
    rnn_num_layers: int = Field(1)

class PPOSimConfig(BaseModel):
    state_dim: int = Field(CORE_STATE_DIM)
    action_dim: int = Field(CORE_ACTION_DIM)
    hidden_dim: int = Field(256)
    log_std_min: int = Field(-20)
    log_std_max: int = Field(1)
    use_rnn: bool = Field(False)
    rnn_type: Literal['lstm', 'gru'] = Field('lstm')
    rnn_hidden_size: int = Field(128)
    rnn_num_layers: int = Field(1)

class WorldSimConfig(BaseModel):
    dt: float = Field(1.0)
    agent_speed: float = Field(2.5)
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 6, math.pi / 6))
    world_x_bounds: Tuple[float, float] = Field((-150.0, 150.0))
    world_y_bounds: Tuple[float, float] = Field((-150.0, 150.0))
    landmark_depth_bounds: Tuple[float, float] = Field((0.0, 300.0))
    normalize_state: bool = Field(True)

    agent_initial_location: LocationConfig = Field(default_factory=LocationConfig)
    landmark_initial_location: LocationConfig = Field(default_factory=lambda: LocationConfig(x=42.0, y=42.0, depth=42.0))
    landmark_initial_velocity: VelocityConfig = Field(default_factory=VelocityConfig)

    randomize_agent_initial_location: bool = Field(True)
    randomize_landmark_initial_location: bool = Field(True)
    randomize_landmark_initial_velocity: bool = Field(False) 

    agent_randomization_ranges: RandomizationRangeConfig = Field(
        default_factory=lambda: RandomizationRangeConfig(x_range=(-100.0,100.0), y_range=(-100.0,100.0), depth_range=(0.0, 0.0))
    )
    landmark_randomization_ranges: RandomizationRangeConfig = Field(
        default_factory=lambda: RandomizationRangeConfig(x_range=(-100.0,100.0), y_range=(-100.0,100.0), depth_range=(0.0,300.0))
    )
    landmark_velocity_randomization_ranges: VelocityRandomizationRangeConfig = Field(
        default_factory=VelocityRandomizationRangeConfig
    )

    trajectory_length: int = Field(10)
    trajectory_feature_dim: int = Field(CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM)

    range_measurement_base_noise: float = Field(0.01)
    range_measurement_distance_factor: float = Field(0.001)
    success_threshold: float = Field(0.5) # Default matched to WorldConfig
    collision_threshold: float = Field(0.5)
    new_measurement_probability: float = Field(0.75) # Default matched to WorldConfig
    
    estimator_config: Union[ParticleFilterSimConfig, LeastSquaresSimConfig] = Field(default_factory=LeastSquaresSimConfig)

class VisualizationSimConfig(BaseModel):
    figure_size: tuple = Field((10, 8))
    max_trajectory_points: int = Field(100)
    gif_frame_duration: float = Field(0.2)
    delete_frames_after_gif: bool = Field(True)

class AppSimConfig(BaseModel):
    sac: Optional[SACSimConfig] = None
    ppo: Optional[PPOSimConfig] = None
    world: WorldSimConfig
    particle_filter: ParticleFilterSimConfig = Field(default_factory=ParticleFilterSimConfig) 
    least_squares: LeastSquaresSimConfig = Field(default_factory=LeastSquaresSimConfig) 
    visualization: VisualizationSimConfig
    cuda_device: str = Field("cpu")
    algorithm: str = Field("sac")

    class Config:
        arbitrary_types_allowed = True


# --- World Objects (from world_objects.py) ---
class Velocity:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    def is_moving(self) -> bool: return self.x != 0 or self.y != 0 or self.z != 0
    def __str__(self) -> str: return f"Vel:(vx:{self.x:.2f}, vy:{self.y:.2f}, vz:{self.z:.2f})"

class Location:
    def __init__(self, x: float, y: float, depth: float):
        self.x = x
        self.y = y
        self.depth = depth
    def update(self, velocity: Velocity, dt: float = 1.0):
        self.x += velocity.x * dt
        self.y += velocity.y * dt
        self.depth += velocity.z * dt
    def __str__(self) -> str: return f"Pos:(x:{self.x:.2f}, y:{self.y:.2f}, d:{self.depth:.2f})"

class Object:
    def __init__(self, location: Location, velocity: Velocity = None, name: str = None):
        self.name = name if name else "Unnamed Object"
        self.location = location
        self.velocity = velocity if velocity is not None else Velocity(0.0, 0.0, 0.0)
    def update_position(self, dt: float = 1.0):
        if self.velocity and self.velocity.is_moving():
            self.location.update(self.velocity, dt)
    def __str__(self) -> str: return f"{self.name}: {self.location}, {self.velocity}"

# --- Estimators ---
class TrackedTargetLS:
    def __init__(self, config: LeastSquaresSimConfig):
        self.config = config
        self.history_size = config.history_size
        self.min_points_required = max(config.min_points_required, 3)
        self.position_buffer_size = config.position_buffer_size
        self.velocity_smoothing = max(config.velocity_smoothing, 1)
        self.min_observer_movement = config.min_observer_movement
        self.location_smoothing_factor = config.location_smoothing_factor

        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None
        self._is_initialized = False
        self._observer_locations: deque[Location] = deque(maxlen=self.history_size)
        self._range_measurements: deque[float] = deque(maxlen=self.history_size)
        self._timestamps_of_measurements: deque[float] = deque(maxlen=self.history_size)
        self._position_history: deque[Tuple[float, Location]] = deque(maxlen=self.position_buffer_size)
        self._current_timestamp = 0.0

    def update(self, dt: float, has_new_range: bool, range_measurement: float, observer_location: Location, perform_update_step: bool = True):
        self._current_timestamp += dt
        new_measurement_added = False
        if has_new_range and range_measurement > 0 and perform_update_step:
            has_moved_enough = (not self._observer_locations or
                                self._calculate_distance(observer_location, self._observer_locations[-1]) > self.min_observer_movement)
            if has_moved_enough:
                observer_2d_loc = Location(x=observer_location.x, y=observer_location.y, depth=0.0)
                self._observer_locations.append(observer_2d_loc)
                self._range_measurements.append(range_measurement)
                self._timestamps_of_measurements.append(self._current_timestamp)
                new_measurement_added = True
        if self.estimated_location and self.estimated_velocity:
            predicted_location = Location(
                x=self.estimated_location.x + self.estimated_velocity.x * dt,
                y=self.estimated_location.y + self.estimated_velocity.y * dt,
                depth=self.estimated_location.depth )
            self.estimated_location = predicted_location
        raw_ls_estimate_xy = None
        if perform_update_step and new_measurement_added and len(self._observer_locations) >= self.min_points_required:
            raw_ls_estimate_xy = self._solve_least_squares_linearized()
        if raw_ls_estimate_xy is not None:
            raw_ls_location = Location(x=raw_ls_estimate_xy[0], y=raw_ls_estimate_xy[1], depth=0.0) # LS is 2D
            if not self._is_initialized or self.estimated_location is None:
                self.estimated_location = raw_ls_location # Use 2D LS depth
                self._is_initialized = True
            else:
                alpha = self.location_smoothing_factor
                smooth_x = alpha * raw_ls_location.x + (1 - alpha) * self.estimated_location.x
                smooth_y = alpha * raw_ls_location.y + (1 - alpha) * self.estimated_location.y
                # Preserve existing estimated depth if any, otherwise use LS's (0.0)
                current_depth = self.estimated_location.depth if self.estimated_location.depth is not None else 0.0
                self.estimated_location = Location(x=smooth_x, y=smooth_y, depth=current_depth) 
            self._position_history.append((self._current_timestamp, self.estimated_location))
            self._update_velocity_estimate()
    def _solve_least_squares_linearized(self) -> Optional[np.ndarray]:
        n_points = len(self._observer_locations);
        if n_points < 3: return None
        ref_loc = self._observer_locations[0]; ref_range_sq = self._range_measurements[0]**2
        x1, y1 = ref_loc.x, ref_loc.y
        A = np.zeros((n_points - 1, 2)); b = np.zeros(n_points - 1)
        for i in range(1, n_points):
            obs_loc = self._observer_locations[i]; range_sq = self._range_measurements[i]**2
            xi, yi = obs_loc.x, obs_loc.y
            A[i-1, 0] = 2 * (x1 - xi); A[i-1, 1] = 2 * (y1 - yi)
            b[i-1] = (range_sq - ref_range_sq) - (xi**2 - x1**2) - (yi**2 - y1**2)
        try: return np.linalg.pinv(A) @ b
        except np.linalg.LinAlgError: return None
        except Exception: return None # Catch any other unexpected math errors
    def _update_velocity_estimate(self):
        if len(self._position_history) < 2:
            if self.estimated_velocity is None: self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)
            return
        num_diffs = min(self.velocity_smoothing, len(self._position_history) - 1)
        if num_diffs < 1: num_diffs = 1
        older_idx = -num_diffs - 1; newer_idx = -1
        try:
            older_timestamp, older_pos = self._position_history[older_idx]
            newer_timestamp, newer_pos = self._position_history[newer_idx]
        except IndexError: # Should not happen with deque if len >=2 and num_diffs >=1
             older_timestamp, older_pos = self._position_history[0]
             newer_timestamp, newer_pos = self._position_history[-1]
        time_diff = newer_timestamp - older_timestamp
        if time_diff > 1e-6:
            vx = (newer_pos.x - older_pos.x) / time_diff; vy = (newer_pos.y - older_pos.y) / time_diff
            self.estimated_velocity = Velocity(x=vx, y=vy, z=0.0) # Velocity is 2D for LS
        elif not self.estimated_velocity: self.estimated_velocity = Velocity(x=0.0, y=0.0, z=0.0)
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)**0.5
    def encode_state(self) -> Dict[str, Any]:
        state = {"is_initialized": self._is_initialized}
        if self.estimated_location: state["estimated_location"] = (self.estimated_location.x, self.estimated_location.y, self.estimated_location.depth)
        if self.estimated_velocity: state["estimated_velocity"] = (self.estimated_velocity.x, self.estimated_velocity.y, self.estimated_velocity.z)
        return state
    def decode_state(self, state_dict: Dict[str, Any]):
        self._is_initialized = state_dict.get("is_initialized", False)
        if "estimated_location" in state_dict:
            x,y,d = state_dict["estimated_location"]; self.estimated_location = Location(x,y,d)
        if "estimated_velocity" in state_dict:
            x,y,z = state_dict["estimated_velocity"]; self.estimated_velocity = Velocity(x,y,z)

class ParticleFilterCore:
    def __init__(self, config: ParticleFilterSimConfig):
        self.state_dimension = 4
        self.config = config
        self.num_particles = config.num_particles
        self.max_particle_range = config.max_particle_range
        self.process_noise_position = config.process_noise_pos
        self.process_noise_orientation = config.process_noise_orient
        self.process_noise_velocity = config.process_noise_vel
        self.measurement_noise_stddev = config.measurement_noise_stddev
        self.particles_state = np.zeros((self.num_particles, self.state_dimension))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None
        self.position_covariance_matrix = np.eye(2)
        self.position_covariance_eigenvalues = np.array([0.02, 0.02])
        self.position_covariance_orientation = 0.0
        self.is_initialized = False
    def initialize_particles(self, observer_location: Location, initial_range_guess: float):
        for i in range(self.num_particles):
            angle = random.uniform(0, 2 * np.pi)
            radius = random.gauss(initial_range_guess, self.config.initial_range_stddev)
            self.particles_state[i, 0] = observer_location.x + radius * np.cos(angle)
            self.particles_state[i, 2] = observer_location.y + radius * np.sin(angle)
            initial_orientation = random.uniform(0, 2 * np.pi)
            vel_mag_std_dev = self.config.initial_velocity_guess / 2.0 + 1e-6 # Avoid std_dev = 0
            velocity_magnitude = abs(random.gauss(self.config.initial_velocity_guess, vel_mag_std_dev))
            self.particles_state[i, 1] = velocity_magnitude * np.cos(initial_orientation)
            self.particles_state[i, 3] = velocity_magnitude * np.sin(initial_orientation)
        self.weights.fill(1.0 / self.num_particles)
        self.estimate_target_state()
        self.is_initialized = True
    def predict(self, dt: float):
        if not self.is_initialized: return
        for i in range(self.num_particles):
            vx, vy = self.particles_state[i, 1], self.particles_state[i, 3]
            current_orientation = np.arctan2(vy, vx)
            orientation_noise = random.uniform(-self.process_noise_orientation, self.process_noise_orientation)
            new_orientation = (current_orientation + orientation_noise + 2 * np.pi) % (2 * np.pi) # Ensure positive
            current_velocity_magnitude = np.sqrt(vx**2 + vy**2)
            velocity_noise = random.uniform(-self.process_noise_velocity, self.process_noise_velocity)
            new_velocity_magnitude = max(0.0, current_velocity_magnitude + velocity_noise)
            distance_travelled = new_velocity_magnitude * dt # Use new_velocity_magnitude for distance
            position_noise = random.uniform(-self.process_noise_position, self.process_noise_position)
            effective_distance = distance_travelled + position_noise
            self.particles_state[i, 0] += effective_distance * np.cos(new_orientation)
            self.particles_state[i, 2] += effective_distance * np.sin(new_orientation)
            self.particles_state[i, 1] = new_velocity_magnitude * np.cos(new_orientation)
            self.particles_state[i, 3] = new_velocity_magnitude * np.sin(new_orientation)
    def _calculate_likelihood(self, predicted_range: float, measured_range: float) -> float:
        if measured_range <= 0: return 1.0
        variance = self.measurement_noise_stddev**2
        if variance < 1e-9: variance = 1e-9
        exponent = -((predicted_range - measured_range)**2) / (2 * variance)
        return np.exp(exponent) + 1e-9
    def update_weights(self, measured_range: float, observer_location: Location):
        if not self.is_initialized: return
        dx = self.particles_state[:, 0] - observer_location.x
        dy = self.particles_state[:, 2] - observer_location.y
        predicted_ranges = np.sqrt(dx**2 + dy**2)
        for i in range(self.num_particles):
            self.weights[i] *= self._calculate_likelihood(predicted_ranges[i], measured_range)
        total_weight = np.sum(self.weights)
        if total_weight > 1e-9: self.weights /= total_weight
        else: self.weights.fill(1.0 / self.num_particles)
    def resample_particles(self, method: int):
        if not self.is_initialized: return
        new_particles_state = np.zeros_like(self.particles_state)
        n = self.num_particles
        
        # Simplified systematic resampling (original had Neff check)
        cumulative_sum = np.cumsum(self.weights)
        step = 1.0 / n; u = random.uniform(0, step); i = 0
        for j in range(n):
            while i < n -1 and u > cumulative_sum[i]: i += 1 # Ensure i doesn't exceed valid index for cumulative_sum
            idx_to_copy = i 
            new_particles_state[j, :] = self.particles_state[idx_to_copy, :]
            u += step
        self.particles_state = new_particles_state
        self.weights.fill(1.0 / self.num_particles)
    def estimate_target_state(self, method: int = 2):
        if not self.is_initialized or self.num_particles == 0:
            self.estimated_location = None; self.estimated_velocity = None; return
        total_weight = np.sum(self.weights)
        mean_state = np.average(self.particles_state, axis=0, weights=self.weights) if total_weight > 1e-9 else np.mean(self.particles_state, axis=0)
        self.estimated_location = Location(x=mean_state[0], y=mean_state[2], depth=0.0) # PF is 2D
        self.estimated_velocity = Velocity(x=mean_state[1], y=mean_state[3], z=0.0) # PF is 2D
        try:
            if total_weight > 1e-9:
                x_coords, y_coords = self.particles_state[:, 0], self.particles_state[:, 2]
                avg_x, avg_y = self.estimated_location.x, self.estimated_location.y
                cov_xx = np.sum(self.weights * (x_coords - avg_x)**2)
                cov_yy = np.sum(self.weights * (y_coords - avg_y)**2)
                cov_xy = np.sum(self.weights * (x_coords - avg_x) * (y_coords - avg_y))
                self.position_covariance_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            else: self.position_covariance_matrix = np.cov(self.particles_state[:, 0], self.particles_state[:, 2], ddof=0) # Use population covariance
            
            if np.any(np.isnan(self.position_covariance_matrix)) or np.any(np.isinf(self.position_covariance_matrix)):
                 raise np.linalg.LinAlgError("Covariance matrix contains NaN/Inf.")

            vals, vecs = np.linalg.eigh(self.position_covariance_matrix)
            order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
            self.position_covariance_eigenvalues = np.sqrt(np.maximum(0, vals)) # Ensure non-negative before sqrt
            self.position_covariance_orientation = np.arctan2(vecs[1, 0], vecs[0, 0])
        except np.linalg.LinAlgError:
            self.position_covariance_matrix = np.eye(2) * 1e-6 # Small identity matrix
            self.position_covariance_eigenvalues = np.array([1e-3, 1e-3])
            self.position_covariance_orientation = 0.0
    def evaluate_filter_quality(self, observer_location: Location, measured_range: float):
        if not self.is_initialized or measured_range <= 0: return
        max_mean_range_error = self.config.pf_eval_max_mean_range_error_factor * self.max_particle_range
        dx = self.particles_state[:, 0] - observer_location.x; dy = self.particles_state[:, 2] - observer_location.y
        particle_ranges = np.sqrt(dx**2 + dy**2)
        total_weight = np.sum(self.weights)
        mean_particle_range = np.average(particle_ranges, weights=self.weights) if total_weight > 1e-9 else np.mean(particle_ranges)
        mean_range_error = abs(mean_particle_range - measured_range)
        
        # Use a small default dispersion if eigenvalues are problematic
        dispersion = self.config.pf_eval_dispersion_threshold * 2 
        if self.position_covariance_eigenvalues is not None and len(self.position_covariance_eigenvalues) == 2:
            ellipse_axis1 = self.position_covariance_eigenvalues[0] * 1.96 # 95% confidence
            ellipse_axis2 = self.position_covariance_eigenvalues[1] * 1.96
            dispersion = np.sqrt(ellipse_axis1**2 + ellipse_axis2**2) # Geometric mean of axes lengths

        if mean_range_error > max_mean_range_error and dispersion < self.config.pf_eval_dispersion_threshold:
            self.is_initialized = False

class TrackedTargetPF:
    def __init__(self, config: ParticleFilterSimConfig):
        self.config = config; self.resampling_method = config.resampling_method
        self.pf_core = ParticleFilterCore(config=config)
        self.estimated_location: Optional[Location] = None
        self.estimated_velocity: Optional[Velocity] = None
    def update(self, dt: float, has_new_range: bool, range_measurement: float, observer_location: Location, perform_update_step: bool = True):
        if not self.pf_core.is_initialized:
            if has_new_range and range_measurement > 0:
                self.pf_core.initialize_particles(observer_location, range_measurement)
                # Update public attributes after initialization
                if self.pf_core.is_initialized:
                    self.estimated_location = self.pf_core.estimated_location
                    self.estimated_velocity = self.pf_core.estimated_velocity
                return
            else: return # Cannot initialize without a valid range measurement
        
        self.pf_core.predict(dt)
        if perform_update_step:
            effective_range = range_measurement if has_new_range else -1.0
            self.pf_core.update_weights(effective_range, observer_location)
            self.pf_core.resample_particles(method=self.resampling_method)
            self.pf_core.evaluate_filter_quality(observer_location, effective_range) # Check after resample
            if not self.pf_core.is_initialized: # If quality check failed
                 self.estimated_location = None; self.estimated_velocity = None; return
        
        self.pf_core.estimate_target_state(method=2) # Always estimate after predict/update
        self.estimated_location = self.pf_core.estimated_location
        self.estimated_velocity = self.pf_core.estimated_velocity
        # Removed self.current_particles_state, access via pf_core.particles_state if needed by viz

    def encode_state(self) -> Dict[str, Any]:
        state = {"is_initialized": self.pf_core.is_initialized}
        if self.estimated_location: state["estimated_location"] = (self.estimated_location.x, self.estimated_location.y, self.estimated_location.depth)
        if self.estimated_velocity: state["estimated_velocity"] = (self.estimated_velocity.x, self.estimated_velocity.y, self.estimated_velocity.z)
        return state
    def decode_state(self, state_dict: Dict[str, Any]):
        self.pf_core.is_initialized = state_dict.get("is_initialized", False)
        if "estimated_location" in state_dict:
            x,y,d = state_dict["estimated_location"]; self.estimated_location = Location(x,y,d)
            self.pf_core.estimated_location = self.estimated_location
        if "estimated_velocity" in state_dict:
            x,y,z = state_dict["estimated_velocity"]; self.estimated_velocity = Velocity(x,y,z)
            self.pf_core.estimated_velocity = self.estimated_velocity

# --- RL AGENT NETWORKS (EVALUATION VERSIONS) ---
class ActorNetEval(nn.Module): # For SAC
    def __init__(self, config: SACSimConfig, world_config: WorldSimConfig): # world_config not strictly needed here, but kept for signature
        super(ActorNetEval, self).__init__(); self.config = config;
        self.use_rnn = config.use_rnn; self.state_dim = config.state_dim; self.action_dim = config.action_dim
        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size; self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # RNN input is the basic state part of trajectory
            if config.rnn_type == 'lstm': self.rnn = nn.LSTM(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru': self.rnn = nn.GRU(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN: {config.rnn_type}")
            mlp_input_dim = config.rnn_hidden_size
        else: mlp_input_dim = self.state_dim; self.rnn = None # MLP input is just the basic state
        
        self.layers = nn.ModuleList(); current_dim = mlp_input_dim
        for hidden_dim_val in config.hidden_dims: # Iterate over list of hidden_dims
            self.layers.append(nn.Linear(current_dim, hidden_dim_val))
            self.layers.append(nn.ReLU())
            current_dim = hidden_dim_val
        self.mean = nn.Linear(current_dim, self.action_dim); self.log_std = nn.Linear(current_dim, self.action_dim)
        self.log_std_min = config.log_std_min; self.log_std_max = config.log_std_max

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn: 
            # network_input for RNN actor is (batch, seq_len, state_dim)
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            mlp_input = rnn_output[:, -1, :] # Use last output of RNN sequence
        else: 
            # network_input for MLP actor is (batch, state_dim)
            mlp_input = network_input
        
        x = mlp_input
        for layer in self.layers: x = layer(x)
        mean = self.mean(x); log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        mean, _, next_hidden_state = self.forward(network_input, hidden_state) # log_std not used for deterministic eval
        action_normalized = torch.tanh(mean)
        return action_normalized, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm': return (h_zeros, torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device))
        return h_zeros

class PolicyNetworkNetEval(nn.Module): # For PPO
    def __init__(self, ppo_config: PPOSimConfig, world_config: WorldSimConfig): # world_config not strictly needed
        super(PolicyNetworkNetEval, self).__init__(); self.ppo_config = ppo_config;
        self.use_rnn = ppo_config.use_rnn; self.state_dim = ppo_config.state_dim; self.action_dim = ppo_config.action_dim
        if self.use_rnn:
            self.rnn_hidden_size = ppo_config.rnn_hidden_size; self.rnn_num_layers = ppo_config.rnn_num_layers
            rnn_input_dim = self.state_dim # PPO RNN input is (batch, seq_len=1, state_dim) for single step action
            if ppo_config.rnn_type == 'lstm': self.rnn = nn.LSTM(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True)
            elif ppo_config.rnn_type == 'gru': self.rnn = nn.GRU(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN: {ppo_config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size
        else: self.rnn = None; mlp_input_dim = self.state_dim # PPO MLP input is (batch, state_dim)

        self.fc1 = nn.Linear(mlp_input_dim, ppo_config.hidden_dim)
        self.fc2 = nn.Linear(ppo_config.hidden_dim, ppo_config.hidden_dim)
        self.mean_layer = nn.Linear(ppo_config.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim)) # Part of model state
        self.log_std_min = ppo_config.log_std_min; self.log_std_max = ppo_config.log_std_max

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn:
            # network_input is (batch, 1, state_dim) for single step action
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            mlp_features = rnn_output[:, -1, :] # Use last (and only) output
        else:
            # network_input is (batch, state_dim)
            mlp_features = network_input
        x = F.relu(self.fc1(mlp_features)); x = F.relu(self.fc2(x))
        action_mean = self.mean_layer(x)
        return action_mean, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        mean, next_hidden_state = self.forward(network_input, hidden_state)
        action_normalized = torch.tanh(mean)
        return action_normalized, next_hidden_state
        
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
        if self.ppo_config.rnn_type == 'lstm': return (h_zeros, torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device))
        return h_zeros


# --- RL AGENT MAIN CLASSES (EVALUATION VERSIONS) ---
class SACAgentEval:
    def __init__(self, agent_config: SACSimConfig, world_config: WorldSimConfig, device_str: str): # world_config for potential future use
        self.config = agent_config
        self.device = torch.device("cpu") # Always CPU for streamlit bundle
        self.use_rnn = agent_config.use_rnn
        self.actor = ActorNetEval(agent_config, world_config).to(self.device)
        self.log_alpha = torch.tensor(0.0).to(self.device)
    def select_action(self, state_dict: Dict[str, Any], actor_hidden_state: Optional[Tuple] = None) -> Tuple[float, Optional[Tuple]]:
        with torch.no_grad():
            if self.use_rnn:
                # SAC RNN Actor expects a sequence of basic states: (batch=1, seq_len, state_dim)
                actor_input = torch.FloatTensor(state_dict['full_trajectory'][:, :self.config.state_dim]).to(self.device).unsqueeze(0)
            else:
                # SAC MLP Actor expects the last basic state: (batch=1, state_dim)
                actor_input = torch.FloatTensor(state_dict['basic_state']).to(self.device).unsqueeze(0)
            self.actor.eval()
            action_normalized, next_actor_hidden_state = self.actor.sample(actor_input, actor_hidden_state)
        return action_normalized.detach().cpu().numpy()[0, 0], next_actor_hidden_state # Action is 1D
    def load_model(self, path: str):
        if not os.path.exists(path): print(f"SAC Model file not found: {path}"); return
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        if 'log_alpha' in checkpoint: self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.actor.eval()
        print(f"SAC Eval Agent loaded actor from {path}")

class PPOAgentEval:
    def __init__(self, agent_config: PPOSimConfig, world_config: WorldSimConfig, device_str: str): # world_config for potential future use
        self.config = agent_config
        self.device = torch.device("cpu") # Always CPU for streamlit bundle
        self.use_rnn = agent_config.use_rnn
        self.actor = PolicyNetworkNetEval(agent_config, world_config).to(self.device)
    def select_action(self, norm_basic_state_tuple: Tuple, actor_hidden_state: Optional[Tuple]=None) -> Tuple[float, Optional[Tuple]]:
        with torch.no_grad():
            if self.use_rnn:
                # PPO RNN Actor expects current basic state shaped as (batch=1, seq_len=1, state_dim)
                network_input_tensor = torch.FloatTensor(norm_basic_state_tuple).to(self.device).unsqueeze(0).unsqueeze(0)
            else:
                # PPO MLP Actor expects current basic state shaped as (batch=1, state_dim)
                network_input_tensor = torch.FloatTensor(norm_basic_state_tuple).to(self.device).unsqueeze(0)
            self.actor.eval()
            action_normalized, next_actor_h_detached = self.actor.sample(network_input_tensor, actor_hidden_state)
        return action_normalized.detach().cpu().numpy().item(), next_actor_h_detached # Action is 1D
    def load_model(self, path: str):
        if not os.path.exists(path): print(f"PPO Model file not found: {path}"); return
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        print(f"PPO Eval Agent loaded actor from {path}")


# --- World Simulation (adapted from world.py) ---
class WorldSim:
    def __init__(self, world_sim_config: WorldSimConfig,
                 initial_agent_loc_override: Optional[Tuple[float, float]] = None,
                 initial_landmark_loc_override: Optional[Tuple[float, float, float]] = None):
        self.world_config = world_sim_config
        self.initial_agent_loc_override = initial_agent_loc_override
        self.initial_landmark_loc_override = initial_landmark_loc_override
        self.dt = world_sim_config.dt
        self.success_threshold = world_sim_config.success_threshold
        self.agent_speed = world_sim_config.agent_speed
        self.max_yaw_change = world_sim_config.yaw_angle_range[1]
        self.trajectory_length = world_sim_config.trajectory_length
        self.feature_dim = world_sim_config.trajectory_feature_dim # Should be CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM
        self.new_measurement_probability = world_sim_config.new_measurement_probability
        self.max_world_diagonal_range = np.sqrt(
            (world_sim_config.world_x_bounds[1] - world_sim_config.world_x_bounds[0])**2 +
            (world_sim_config.world_y_bounds[1] - world_sim_config.world_y_bounds[0])**2)
        if self.max_world_diagonal_range == 0: self.max_world_diagonal_range = 1.0

        if isinstance(world_sim_config.estimator_config, LeastSquaresSimConfig):
            self.estimated_landmark = TrackedTargetLS(config=world_sim_config.estimator_config)
        elif isinstance(world_sim_config.estimator_config, ParticleFilterSimConfig):
            self.estimated_landmark = TrackedTargetPF(config=world_sim_config.estimator_config)
        else:
            print(f"Warning: Unknown estimator config type for WorldSim: {type(world_sim_config.estimator_config)}. Defaulting to LS.")
            self.estimated_landmark = TrackedTargetLS(config=LeastSquaresSimConfig())
        self.reset()

    def _initialize_world_objects(self):
        if self.initial_landmark_loc_override:
            lx, ly, ldepth = self.initial_landmark_loc_override
            true_landmark_location = Location(x=lx, y=ly, depth=ldepth)
        elif self.world_config.randomize_landmark_initial_location:
            ranges = self.world_config.landmark_randomization_ranges
            true_landmark_location = Location(x=random.uniform(*ranges.x_range), y=random.uniform(*ranges.y_range), depth=random.uniform(*ranges.depth_range))
        else:
            loc_cfg = self.world_config.landmark_initial_location
            true_landmark_location = Location(x=loc_cfg.x, y=loc_cfg.y, depth=loc_cfg.depth)

        if self.world_config.randomize_landmark_initial_velocity:
            vranges = self.world_config.landmark_velocity_randomization_ranges
            true_landmark_velocity = Velocity(x=random.uniform(*vranges.vx_range), y=random.uniform(*vranges.vy_range), z=random.uniform(*vranges.vz_range))
        else:
            vcfg = self.world_config.landmark_initial_velocity
            true_landmark_velocity = Velocity(x=vcfg.x, y=vcfg.y, z=vcfg.z)
        self.true_landmark = Object(location=true_landmark_location, velocity=true_landmark_velocity, name="true_landmark")

        if self.initial_agent_loc_override:
            ax, ay = self.initial_agent_loc_override
            agent_location = Location(x=ax, y=ay, depth=self.world_config.agent_initial_location.depth) # Use configured agent depth
        elif self.world_config.randomize_agent_initial_location:
            ranges = self.world_config.agent_randomization_ranges
            agent_location = Location(x=random.uniform(*ranges.x_range), y=random.uniform(*ranges.y_range), depth=random.uniform(*ranges.depth_range))
        else:
            loc_cfg = self.world_config.agent_initial_location
            agent_location = Location(x=loc_cfg.x, y=loc_cfg.y, depth=loc_cfg.depth)

        initial_heading = random.uniform(0, 2 * math.pi)
        agent_velocity = Velocity(x=self.agent_speed * math.cos(initial_heading), y=self.agent_speed * math.sin(initial_heading), z=0.0)
        self.agent = Object(location=agent_location, velocity=agent_velocity, name="agent")
        self.objects = [self.true_landmark, self.agent]

    def reset(self):
        self._initialize_world_objects()
        self.current_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.reward = 0.0
        self.error_dist = float('inf')
        self.done = False
        self._update_error_dist()
        
        if isinstance(self.world_config.estimator_config, LeastSquaresSimConfig):
             self.estimated_landmark = TrackedTargetLS(config=self.world_config.estimator_config)
        elif isinstance(self.world_config.estimator_config, ParticleFilterSimConfig):
             self.estimated_landmark = TrackedTargetPF(config=self.world_config.estimator_config)
        
        self._trajectory_history = deque(maxlen=self.trajectory_length)
        self._initialize_trajectory_history()

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val: return 0.0 if value == min_val else np.sign(value - min_val)
        return np.clip(2 * (value - min_val) / (max_val - min_val) - 1, -1.0, 1.0)

    def _normalize_basic_state(self, raw_state_tuple: Tuple) -> Tuple:
        if not self.world_config.normalize_state: return raw_state_tuple
        ax_r, ay_r, avx_r, avy_r, ah_r, lx_r, ly_r, ld_r, r_r = raw_state_tuple
        ax_n = self._normalize_value(ax_r, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
        ay_n = self._normalize_value(ay_r, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
        avx_n = np.clip(avx_r / self.agent_speed if self.agent_speed > 1e-6 else 0.0, -1.0, 1.0)
        avy_n = np.clip(avy_r / self.agent_speed if self.agent_speed > 1e-6 else 0.0, -1.0, 1.0)
        ah_n = np.clip(ah_r / math.pi, -1.0, 1.0) # Heading from [-pi, pi] to [-1, 1]
        lx_n = self._normalize_value(lx_r, self.world_config.world_x_bounds[0], self.world_config.world_x_bounds[1])
        ly_n = self._normalize_value(ly_r, self.world_config.world_y_bounds[0], self.world_config.world_y_bounds[1])
        ld_n = self._normalize_value(ld_r, self.world_config.landmark_depth_bounds[0], self.world_config.landmark_depth_bounds[1])
        r_n = np.clip(r_r / self.max_world_diagonal_range if self.max_world_diagonal_range > 1e-6 else 0.0, 0.0, 1.0) # Range to [0,1]
        return (ax_n, ay_n, avx_n, avy_n, ah_n, lx_n, ly_n, ld_n, r_n)

    def _calculate_planar_range_measurement(self, loc1: Location, loc2: Location) -> float:
        return np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

    def _calculate_true_range_measurement(self, loc1: Location, loc2: Location) -> float:
        return np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.depth - loc2.depth)**2)

    def _get_noisy_range_measurement(self, loc1: Location, loc2: Location) -> float:
        planar_range = self._calculate_planar_range_measurement(loc1, loc2)
        true_range = self._calculate_true_range_measurement(loc1, loc2) # Noise depends on true 3D range
        noise_std = self.world_config.range_measurement_base_noise + self.world_config.range_measurement_distance_factor * true_range
        return max(0.1, planar_range + np.random.normal(0, noise_std)) # Noisy planar range

    def _update_error_dist(self):
        if self.estimated_landmark and self.estimated_landmark.estimated_location:
            est_loc, true_loc = self.estimated_landmark.estimated_location, self.true_landmark.location
            self.error_dist = np.sqrt((est_loc.x - true_loc.x)**2 + (est_loc.y - true_loc.y)**2) # 2D error
        else: self.error_dist = float('inf')

    def _get_basic_state_tuple(self) -> Tuple: # Raw values
        agent_loc, agent_vel = self.agent.location, self.agent.velocity
        agent_h_rad = math.atan2(agent_vel.y, agent_vel.x)
        est_loc = self.estimated_landmark.estimated_location if self.estimated_landmark.estimated_location else Location(0,0,0)
        # Use estimated depth if available (PF), else 0 (LS). Estimators are 2D, so depth from them is 0.
        # True landmark depth isn't part of basic state, only estimated landmark depth.
        lmk_depth_state = est_loc.depth if hasattr(est_loc, 'depth') and est_loc.depth is not None else 0.0
        return (agent_loc.x, agent_loc.y, agent_vel.x, agent_vel.y, agent_h_rad,
                est_loc.x, est_loc.y, lmk_depth_state, self.current_range)

    def _initialize_trajectory_history(self):
        initial_raw_basic_state = self._get_basic_state_tuple()
        for _ in range(self.trajectory_length):
             self._trajectory_history.append((initial_raw_basic_state, 0.0, 0.0))

    def step(self, yaw_change_normalized: float, terminal_step: bool = False):
        s_t_raw = self._get_basic_state_tuple(); a_t_raw = yaw_change_normalized
        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_vx, current_vy = self.agent.velocity.x, self.agent.velocity.y
        current_heading = math.atan2(current_vy, current_vx)
        new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]
        self.agent.velocity = Velocity(self.agent_speed * math.cos(new_heading), self.agent_speed * math.sin(new_heading), 0.0)
        self.agent.update_position(self.dt); self.true_landmark.update_position(self.dt)
        
        noisy_range = self._get_noisy_range_measurement(self.agent.location, self.true_landmark.location)
        self.current_range = noisy_range # This becomes part of s_{t+1}

        has_new_range = np.random.rand() <= self.new_measurement_probability
        self.estimated_landmark.update(self.dt, has_new_range, noisy_range, self.agent.location)
        self._update_error_dist() # Based on new estimate
        
        r_t_raw = self._calculate_reward()
        self._trajectory_history.append((s_t_raw, a_t_raw, r_t_raw))
        
        true_agent_landmark_dist = self._calculate_planar_range_measurement(self.agent.location, self.true_landmark.location)
        self.done = terminal_step or true_agent_landmark_dist < self.world_config.collision_threshold

    def _calculate_reward(self) -> float:
        """ Calculate reward based on current state (AFTER step). Returns raw reward.
            This matches the reward function from the provided world.py.
        """
        current_reward = 0.0
        estimation_error = self.error_dist
        true_agent_landmark_dist = self._calculate_planar_range_measurement(
            self.agent.location, self.true_landmark.location
        )

        if estimation_error != float('inf') and self.estimated_landmark.estimated_location is not None:
            current_reward += np.clip(np.log(1/max(estimation_error, 1e-6)) + 1, -1, 5) 
        current_reward *= 0.05 

        if true_agent_landmark_dist < 1: 
            current_reward -= 1

        current_reward -= 0.0001 * true_agent_landmark_dist
        
        self.reward = current_reward
        return current_reward

    def encode_state(self) -> Dict[str, Any]:
        raw_traj_tuples = list(self._trajectory_history)
        # feature_dim = CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM
        output_traj_features = np.zeros((self.trajectory_length, self.feature_dim), dtype=np.float32)
        for i in range(self.trajectory_length):
            s_raw, a_raw, r_raw = raw_traj_tuples[i]
            s_norm_tuple = self._normalize_basic_state(s_raw)
            output_traj_features[i, :CORE_STATE_DIM] = np.array(s_norm_tuple, dtype=np.float32)
            output_traj_features[i, CORE_STATE_DIM] = float(a_raw) # Action
            output_traj_features[i, CORE_STATE_DIM + CORE_ACTION_DIM] = float(r_raw) # Reward
        
        last_norm_basic_state_tuple = tuple(output_traj_features[-1, :CORE_STATE_DIM])
        return {'basic_state': last_norm_basic_state_tuple, 'full_trajectory': output_traj_features}


# --- VISUALIZATION (adapted from visualization.py) ---
_agent_trajectory_sim = []
_landmark_trajectory_sim = []

def reset_trajectories_sim():
    global _agent_trajectory_sim, _landmark_trajectory_sim
    _agent_trajectory_sim = []; _landmark_trajectory_sim = []

def visualize_world_sim(world: WorldSim, vis_config: VisualizationSimConfig, vis_output_dir: str, filename_suffix: str):
    global _agent_trajectory_sim, _landmark_trajectory_sim
    max_traj_pts = vis_config.max_trajectory_points
    if world.agent and world.agent.location: _agent_trajectory_sim.append((world.agent.location.x, world.agent.location.y))
    if world.true_landmark and world.true_landmark.location: _landmark_trajectory_sim.append((world.true_landmark.location.x, world.true_landmark.location.y))
    if len(_agent_trajectory_sim) > max_traj_pts: _agent_trajectory_sim = _agent_trajectory_sim[-max_traj_pts:]
    if len(_landmark_trajectory_sim) > max_traj_pts: _landmark_trajectory_sim = _landmark_trajectory_sim[-max_traj_pts:]

    fig, ax = plt.subplots(figsize=vis_config.figure_size)
    if len(_agent_trajectory_sim) > 1: ax.plot(*zip(*_agent_trajectory_sim), 'b-', lw=1.5, alpha=0.6, label='Agent Traj.')
    if len(_landmark_trajectory_sim) > 1: ax.plot(*zip(*_landmark_trajectory_sim), 'r-', lw=1.5, alpha=0.6, label='Landmark Traj.')
    if world.agent and world.agent.location: ax.scatter(world.agent.location.x, world.agent.location.y, c='b', marker='o', s=100, label=f'Agent (Z:{world.agent.location.depth:.1f})')
    if world.true_landmark and world.true_landmark.location:
        ax.scatter(world.true_landmark.location.x, world.true_landmark.location.y, c='r', marker='^', s=100, label=f'True Lmk (Z:{world.true_landmark.location.depth:.1f})')
        if world.agent and world.agent.location: ax.plot([world.agent.location.x, world.true_landmark.location.x], [world.agent.location.y, world.true_landmark.location.y], 'r--', alpha=0.5, label=f'True Range ({world.current_range:.1f})')
    
    if world.estimated_landmark and world.estimated_landmark.estimated_location:
        est_loc = world.estimated_landmark.estimated_location
        ax.scatter(est_loc.x, est_loc.y, c='g', marker='x', s=100, label=f'Est. Lmk (Z:{est_loc.depth:.1f})')
        
        if isinstance(world.estimated_landmark, TrackedTargetPF) and \
           hasattr(world.estimated_landmark, 'pf_core') and world.estimated_landmark.pf_core and \
           world.estimated_landmark.pf_core.particles_state is not None:
            
            pf_core = world.estimated_landmark.pf_core
            particles = pf_core.particles_state
            num_particles_to_plot = pf_core.num_particles

            if num_particles_to_plot < 500:
                ax.scatter(particles[:num_particles_to_plot, 0], particles[:num_particles_to_plot, 2], color='gray', marker='.', s=1, alpha=0.3, label='Particles')
            
            if hasattr(pf_core, 'position_covariance_matrix') and pf_core.position_covariance_eigenvalues is not None and est_loc is not None:
                try:
                    eigvals = pf_core.position_covariance_eigenvalues; angle = np.degrees(pf_core.position_covariance_orientation)
                    safe_eigvals = np.maximum(eigvals, 1e-9); width = safe_eigvals[0]**0.5 * 2 * 1.96; height = safe_eigvals[1]**0.5 * 2 * 1.96
                    ellipse = Ellipse(xy=(est_loc.x, est_loc.y), width=width, height=height, angle=angle, edgecolor='purple', fc='None', lw=1, ls='--', label='95% Conf.')
                    ax.add_patch(ellipse)
                except Exception: pass 
    elif world.estimated_landmark and not world.estimated_landmark._is_initialized : # Check _is_initialized for LS as well
        if world.agent and world.agent.location: ax.text(0.5, 0.02, "Estimator not initialized", ha='center', transform=ax.transAxes, c='orange', fontsize=10)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    title_info = f"Reward: {world.reward:.2f}, Err: {world.error_dist:.2f}, Step: {filename_suffix}"
    ax.set_title(f'World State\n{title_info}')
    
    points_x, points_y = [], []
    if world.agent and world.agent.location: points_x.append(world.agent.location.x); points_y.append(world.agent.location.y)
    if world.true_landmark and world.true_landmark.location: points_x.append(world.true_landmark.location.x); points_y.append(world.true_landmark.location.y)
    if world.estimated_landmark and world.estimated_landmark.estimated_location: points_x.append(world.estimated_landmark.estimated_location.x); points_y.append(world.estimated_landmark.estimated_location.y)
    if _agent_trajectory_sim: traj_x, traj_y = zip(*_agent_trajectory_sim); points_x.extend(traj_x); points_y.extend(traj_y)
    if _landmark_trajectory_sim: traj_x, traj_y = zip(*_landmark_trajectory_sim); points_x.extend(traj_x); points_y.extend(traj_y)
    
    if not points_x or not points_y: min_x, max_x, min_y, max_y = -10, 10, -10, 10
    else: min_x, max_x = min(points_x), max(points_x); min_y, max_y = min(points_y), max(points_y)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    range_x_val, range_y_val = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0) # Renamed to avoid conflict
    max_range_val = max(range_x_val, range_y_val, 20.0); padding = max_range_val * 0.2
    ax.set_xlim(center_x - (max_range_val / 2 + padding), center_x + (max_range_val / 2 + padding))
    ax.set_ylim(center_y - (max_range_val / 2 + padding), center_y + (max_range_val / 2 + padding))
    ax.set_aspect('equal', adjustable='box'); ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    filename = f"sim_frame_{filename_suffix}.png"
    full_path = os.path.join(vis_output_dir, filename)
    try:
        os.makedirs(vis_output_dir, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(full_path); plt.close(fig)
        return full_path
    except Exception as e: print(f"Error saving viz: {e}"); plt.close(fig); return None

def save_gif_sim(output_filename: str, vis_config: VisualizationSimConfig, vis_output_dir:str, frame_paths: list):
    if not frame_paths: return None
    output_path = os.path.join(vis_output_dir, output_filename)
    try:
        images = [imageio.imread(fp) for fp in frame_paths if os.path.exists(fp)]
        if not images: return None
        imageio.mimsave(output_path, images, duration=vis_config.gif_frame_duration)
        if vis_config.delete_frames_after_gif:
            for fp in frame_paths:
                try:
                    if os.path.exists(fp): os.remove(fp)
                except OSError: pass
        return output_path
    except Exception as e: print(f"Error creating GIF: {e}"); return None


# --- MAIN SIMULATION RUNNER ---
def run_simulation_for_streamlit(
    app_sim_config: AppSimConfig,
    model_path: str,
    initial_agent_loc_data: Optional[Tuple[float, float]],
    initial_landmark_loc_data: Optional[Tuple[float, float, float]],
    random_initialization: bool,
    num_simulation_steps: int,
    visualization_output_dir: str,
    st_progress_bar: Optional[Any] = None # Added parameter for Streamlit progress bar
) -> Tuple[Optional[str], float, float, List[float]]:

    world_cfg = app_sim_config.world
    
    if not random_initialization:
        world_cfg.randomize_agent_initial_location = False
        world_cfg.randomize_landmark_initial_location = False
    else:
        world_cfg.randomize_agent_initial_location = True
        world_cfg.randomize_landmark_initial_location = True

    world = WorldSim(
        world_sim_config=world_cfg,
        initial_agent_loc_override=initial_agent_loc_data if not random_initialization else None,
        initial_landmark_loc_override=initial_landmark_loc_data if not random_initialization else None
    )

    agent: Any = None
    if app_sim_config.algorithm.lower() == "sac":
        if not app_sim_config.sac: raise ValueError("SAC config missing in AppSimConfig")
        agent = SACAgentEval(app_sim_config.sac, world_cfg, app_sim_config.cuda_device)
    elif app_sim_config.algorithm.lower() == "ppo":
        if not app_sim_config.ppo: raise ValueError("PPO config missing in AppSimConfig")
        agent = PPOAgentEval(app_sim_config.ppo, world_cfg, app_sim_config.cuda_device)
    else:
        raise ValueError(f"Unknown algorithm for simulation: {app_sim_config.algorithm}")
    
    agent.load_model(model_path)

    reset_trajectories_sim()
    frame_paths = []
    episode_rewards_raw = []
    
    state_dict = world.encode_state()
    actor_hidden_state = None

    if agent.use_rnn:
        if app_sim_config.algorithm.lower() == "sac":
            actor_hidden_state = agent.actor.get_initial_hidden_state(1, agent.device)
        elif app_sim_config.algorithm.lower() == "ppo":
            actor_hidden_state = agent.actor.get_initial_hidden_state(1, agent.device)

    initial_frame_path = visualize_world_sim(world, app_sim_config.visualization, visualization_output_dir, "000_initial")
    if initial_frame_path: frame_paths.append(initial_frame_path)

    for step_num in range(num_simulation_steps):
        action_norm: float = 0.0
        next_actor_hidden_state: Optional[Tuple] = None
        
        if app_sim_config.algorithm.lower() == "sac":
            action_norm, next_actor_hidden_state = agent.select_action(state_dict, actor_hidden_state)
        elif app_sim_config.algorithm.lower() == "ppo":
            norm_basic_state_for_ppo = state_dict['basic_state']
            action_norm, next_actor_hidden_state = agent.select_action(norm_basic_state_for_ppo, actor_hidden_state)

        world.step(action_norm, terminal_step=(step_num == num_simulation_steps - 1))
        state_dict = world.encode_state()
        episode_rewards_raw.append(world.reward)

        if agent.use_rnn:
            actor_hidden_state = next_actor_hidden_state

        if st_progress_bar:
            current_progress_value = (step_num + 1) / num_simulation_steps
            st_progress_bar.progress(current_progress_value)

        frame_path = visualize_world_sim(world, app_sim_config.visualization, visualization_output_dir, f"{step_num+1:03d}")
        if frame_path: frame_paths.append(frame_path)

        if world.done:
            if st_progress_bar: # Ensure progress bar reaches 100% if loop breaks early
                st_progress_bar.progress(1.0)
            break
            
    if st_progress_bar and not world.done: # If loop completed fully without early break
        st_progress_bar.progress(1.0)

    gif_filename = f"simulation_run_{time.strftime('%Y%m%d-%H%M%S')}.gif"
    gif_path = save_gif_sim(gif_filename, app_sim_config.visualization, visualization_output_dir, frame_paths)
    
    final_error = world.error_dist
    total_raw_reward = sum(episode_rewards_raw) if episode_rewards_raw else 0.0

    return gif_path, final_error, total_raw_reward, episode_rewards_raw