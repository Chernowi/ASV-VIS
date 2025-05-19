from typing import Dict, Literal, Tuple, List, Optional
from pydantic import BaseModel, Field
import math

# Core dimensions
CORE_STATE_DIM = 9 # agent_x, agent_y, agent_vx, agent_vy, agent_heading_rad, landmark_x, landmark_y, landmark_depth, current_range
CORE_ACTION_DIM = 1 # yaw_change_normalized
TRAJECTORY_REWARD_DIM = 1

class SACConfig(BaseModel):
    """Configuration for the SAC agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple within the trajectory")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dims: List[int] = Field([64, 64], description="List of hidden layer dimensions for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(1, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-5, description="Learning rate for the actor network")
    critic_lr: float = Field(5e-5, description="Learning rate for the critic network")
    alpha_lr: float = Field(5e-5, description="Learning rate for the alpha temperature parameter (if auto-tuning)")
    gamma: float = Field(0.99, description="Discount factor")
    tau: float = Field(0.005, description="Target network update rate")
    alpha: float = Field(0.2, description="Temperature parameter (Initial value if auto-tuning)")
    auto_tune_alpha: bool = Field(True, description="Whether to auto-tune the alpha parameter")
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic (Recommended for trajectory state)")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")

    # --- Prioritized Experience Replay (PER) settings ---
    use_per: bool = Field(False, description="Whether to use Prioritized Experience Replay")
    per_alpha: float = Field(0.6, description="PER: Alpha exponent for priority calculation (0=uniform, 1=full priority)")
    per_beta_start: float = Field(0.4, description="PER: Initial beta for importance sampling correction")
    per_beta_end: float = Field(1.0, description="PER: Final beta for importance sampling correction (annealed to this value)")
    per_beta_anneal_steps: int = Field(100000, description="PER: Number of agent steps to anneal beta from start to end")
    per_epsilon: float = Field(1e-5, description="PER: Small constant added to priorities to ensure non-zero probability")


class PPOConfig(BaseModel):
    """Configuration for the PPO agent"""
    state_dim: int = Field(CORE_STATE_DIM, description="Dimension of the basic state tuple (PPO uses the last state of trajectory if not RNN)")
    action_dim: int = Field(CORE_ACTION_DIM, description="Action dimension (yaw_change)")
    hidden_dim: int = Field(256, description="Hidden layer dimension for MLP part")
    log_std_min: int = Field(-20, description="Minimum log std for action distribution")
    log_std_max: int = Field(1, description="Maximum log std for action distribution")
    actor_lr: float = Field(5e-6, description="Actor learning rate")
    critic_lr: float = Field(1e-3, description="Critic learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    policy_clip: float = Field(0.05, description="PPO clipping parameter")
    n_epochs: int = Field(3, description="Number of optimization epochs per update")
    entropy_coef: float = Field(0.015, description="Entropy coefficient for exploration")
    value_coef: float = Field(0.5, description="Value loss coefficient")
    batch_size: int = Field(64, description="Batch size for training (number of sequences per batch if RNN, or transitions if MLP)") # Adjusted for RNN
    steps_per_update: int = Field(2048, description="Environment steps between PPO updates (rollout length for recurrent PPO)") # Typically one full rollout

    # --- RNN settings for PPO ---
    use_rnn: bool = Field(False, description="Whether to use RNN layers in Actor/Critic")
    rnn_type: Literal['lstm', 'gru'] = Field('lstm', description="Type of RNN cell (Only used if use_rnn is True)")
    rnn_hidden_size: int = Field(128, description="Hidden size of RNN layers (Only used if use_rnn is True)")
    rnn_num_layers: int = Field(1, description="Number of RNN layers (Only used if use_rnn is True)")
    # For recurrent PPO, max_seq_len_in_batch might be useful if optimizing memory for batches,
    # but usually derived from steps_per_update or actual rollout lengths.
    # rollout_buffer_size: int = Field(10, description="Number of full rollouts to store before an update (if steps_per_update is one rollout)")


class ReplayBufferConfig(BaseModel):
    """Configuration for the replay buffer (SAC specific)"""
    capacity: int = Field(1000000, description="Maximum capacity of replay buffer (stores full trajectories)")
    gamma: float = Field(0.99, description="Discount factor for returns")

class TrainingConfig(BaseModel):
    """Configuration for training"""
    num_episodes: int = Field(30000, description="Number of episodes to train")
    max_steps: int = Field(300, description="Maximum steps per episode")
    # batch_size for SAC is in SACConfig, PPO uses its own ppo_config.batch_size
    sac_batch_size: int = Field(32, description="Batch size for SAC training (Number of trajectories sampled)") 
    save_interval: int = Field(1000, description="Interval (in episodes) for saving models")
    log_frequency: int = Field(10, description="Frequency (in episodes) for logging to TensorBoard")
    models_dir: str = Field("models/default_models/", description="Default directory if experiment structure fails or for old scripts (less used now)")
    experiment_base_dir: str = Field("experiments", description="Base directory for saving all experiment data (logs, models, config)")
    learning_starts: int = Field(8000, description="Number of steps to collect before starting training updates (SAC)")
    train_freq: int = Field(30, description="Update the policy every n environment steps (SAC)")
    gradient_steps: int = Field(20, description="How many gradient steps to perform when training frequency is met (SAC)")
    normalize_rewards: bool = Field(True, description="Whether to normalize rewards using running mean/std (agent-side)")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    num_episodes: int = Field(1, description="Number of episodes for evaluation")
    max_steps: int = Field(300, description="Maximum steps per evaluation episode")
    render: bool = Field(True, description="Whether to render the evaluation")

class Position(BaseModel):
    """Position configuration"""
    x: float = 0
    y: float = 0
    depth: float = 0

class Velocity(BaseModel):
    """Velocity configuration"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class RandomizationRange(BaseModel):
    """Defines ranges for random initialization of position"""
    x_range: Tuple[float, float] = Field((-100.0, 100.0), description="Min/Max X range for randomization")
    y_range: Tuple[float, float] = Field((-100.0, 100.0), description="Min/Max Y range for randomization")
    depth_range: Tuple[float, float] = Field((0, 300), description="Min/Max Depth range for randomization")

class VelocityRandomizationRange(BaseModel):
    """Defines ranges for random initialization of velocity"""
    vx_range: Tuple[float, float] = Field((-0.5, 0.5), description="Min/Max Vx range for randomization")
    vy_range: Tuple[float, float] = Field((-0.5, 0.5), description="Min/Max Vy range for randomization")
    vz_range: Tuple[float, float] = Field((-0.1, 0.1), description="Min/Max Vz range for randomization")

class ParticleFilterConfig(BaseModel):
    """Configuration for the particle filter"""
    num_particles: int = Field(1000, description="Number of particles")
    initial_range_stddev: float = Field(0.02, description="Standard deviation for initial particle spread")
    initial_velocity_guess: float = Field(0.1, description="Initial velocity guess for particles")
    max_particle_range: float = Field(250.0, description="Maximum range for particles (used in area method or init)")
    process_noise_pos: float = Field(0.02, description="Process noise for position")
    process_noise_orient: float = Field(0.2, description="Process noise for orientation")
    process_noise_vel: float = Field(0.02, description="Process noise for velocity")
    measurement_noise_stddev: float = Field(5.0, description="Standard deviation for measurement noise")
    resampling_method: int = Field(2, description="Method for resampling")
    pf_eval_max_mean_range_error_factor: float = Field(0.1, description="Factor of max_particle_range used as threshold for PF quality check")
    pf_eval_dispersion_threshold: float = Field(5.0, description="Dispersion threshold for PF quality check")

class LeastSquaresConfig(BaseModel):
    """Configuration for the Least Squares estimator"""
    history_size: int = Field(10, description="Number of measurements to keep in history")
    min_points_required: int = Field(3, description="Minimum number of points required for estimation")
    location_smoothing_factor: float = Field(0.8, 
        description="Smoothing factor for location updates (0=no update from LS, 1=instant update, lower values mean smoother/slower updates)"
    )
    position_buffer_size: int = Field(5, description="Number of position estimates to keep for velocity calculation")
    velocity_smoothing: int = Field(3, description="Number of position points to use for velocity smoothing")
    min_observer_movement: float = Field(0.5, description="Minimum movement required between measurement points")

class VisualizationConfig(BaseModel):
    """Configuration for visualization"""
    save_dir: str = Field("world_snapshots", description="Directory for saving visualizations (relative to experiment dir or absolute)")
    figure_size: tuple = Field((10, 8), description="Figure size for visualizations")
    max_trajectory_points: int = Field(100, description="Max trajectory points to display")
    gif_frame_duration: float = Field(0.2, description="Duration of each frame in generated GIFs")
    delete_frames_after_gif: bool = Field(True, description="Delete individual PNG frames after creating GIF")

class WorldConfig(BaseModel):
    """Configuration for the world"""
    # --- Basic World Dynamics ---
    dt: float = Field(1.0, description="Time step")
    agent_speed: float = Field(2.5, description="Constant speed of the agent")
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 6, math.pi / 6), description="Range of possible yaw angle changes per step [-max_change, max_change]")

    # --- World Boundaries for Normalization ---
    world_x_bounds: Tuple[float, float] = Field((-150.0, 150.0), description="Min/Max X boundaries of the world for normalization")
    world_y_bounds: Tuple[float, float] = Field((-150.0, 150.0), description="Min/Max Y boundaries of the world for normalization")
    landmark_depth_bounds: Tuple[float, float] = Field((0.0, 300.0), description="Min/Max depth for landmark, used for normalization")
    normalize_state: bool = Field(True, description="Whether the world should provide normalized states to the agent.")


    # --- Initial Conditions & Randomization ---
    agent_initial_location: Position = Field(default_factory=Position, description="Initial agent position (used if randomization is false)")
    landmark_initial_location: Position = Field(default_factory=lambda: Position(x=42, y=42, depth=42), description="Initial landmark position (used if randomization is false)")
    landmark_initial_velocity: Velocity = Field(default_factory=Velocity, description="Initial landmark velocity (used if randomization is false)")

    randomize_agent_initial_location: bool = Field(True, description="Randomize agent initial location?")
    randomize_landmark_initial_location: bool = Field(True, description="Randomize landmark initial location?")
    randomize_landmark_initial_velocity: bool = Field(False, description="Randomize landmark initial velocity?")

    agent_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(-100.0,100.0), y_range=(-100.0,100.0), depth_range=(0.0, 0.0)), description="Ranges for agent location randomization")
    landmark_randomization_ranges: RandomizationRange = Field(default_factory=lambda: RandomizationRange(x_range=(-100.0,100.0), y_range=(-100.0,100.0), depth_range=(0.0,300.0)), description="Ranges for landmark location randomization")
    landmark_velocity_randomization_ranges: VelocityRandomizationRange = Field(default_factory=VelocityRandomizationRange, description="Ranges for landmark velocity randomization")

    # --- State Representation ---
    trajectory_length: int = Field(10, description="Number of steps (N) included in the trajectory state for fixed-history SAC/PPO-MLP. PPO-RNN uses full rollouts.")
    trajectory_feature_dim: int = Field(CORE_STATE_DIM + CORE_ACTION_DIM + TRAJECTORY_REWARD_DIM, description="Dimension of features per step in trajectory state (basic_state + prev_action + prev_reward)") # 9+1+1=11

    # --- Observations & Noise ---
    range_measurement_base_noise: float = Field(0.01, description="Base standard deviation of range measurement noise")
    range_measurement_distance_factor: float = Field(0.001, description="Factor by which range noise std dev increases with distance")

    # --- Termination Conditions ---
    success_threshold: float = Field(0.5, description="Estimation error (2D distance) below which the episode is considered successful")
    collision_threshold: float = Field(0.5, description="Agent-Landmark true distance below which a collision occurs (terminates episode)")

    # --- Reward Function Parameters ---
    reward_error_threshold: float = Field(1.0, description="Estimation error threshold for bonus reward")
    low_error_bonus: float = Field(1.0, description="Reward bonus when estimation error is below reward_error_threshold")
    high_error_penalty_factor: float = Field(0.1, description="Penalty multiplier for estimation error exceeding threshold")
    uninitialized_penalty: float = Field(1.0, description="Penalty if estimator hasn't produced a valid estimate")
    reward_distance_threshold: float = Field(15.0, description="True distance threshold for close distance bonus")
    close_distance_bonus: float = Field(1.0, description="Reward bonus when true distance is below reward_distance_threshold")
    distance_reward_scale: float = Field(0.0001, description="Scaling factor for distance reward shaping")
    max_distance_for_reward: float = Field(50.0, description="Maximum true distance for distance shaping reward")
    max_observable_range: float = Field(100.0, description="Maximum true distance considered 'in range' for penalty")
    out_of_range_penalty: float = Field(0.1, description="Penalty if true distance exceeds max_observable_range")
    landmark_collision_penalty: float = Field(1.0, description="Penalty for collision")
    success_bonus: float = Field(30.0, description="Bonus reward upon success")
    new_measurement_probability: float = Field(0.75, description="Probability at each step of recieving a new range measurement")


    # --- Landmark Estimator ---
    estimator_config: ParticleFilterConfig | LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Configuration for the landmark estimator")


class DefaultConfig(BaseModel):
    """Default configuration for the entire application"""
    sac: SACConfig = Field(default_factory=SACConfig, description="SAC agent configuration")
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO agent configuration")
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig, description="Replay buffer configuration (SAC)")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    world: WorldConfig = Field(default_factory=WorldConfig, description="World configuration")
    particle_filter: ParticleFilterConfig = Field(default_factory=ParticleFilterConfig, description="Particle filter configuration")
    least_squares: LeastSquaresConfig = Field(default_factory=LeastSquaresConfig, description="Least Squares estimator configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")
    cuda_device: str = Field("cuda:0", description="CUDA device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    algorithm: str = Field("sac", description="RL algorithm to use ('sac' or 'ppo')")

    # Using model_post_init for Pydantic V2 compatibility
    def model_post_init(self, __context):
        self._sync_dependent_configs()
        self._resolve_estimator_config()
        self._update_models_dir_per_algo()

    def _sync_dependent_configs(self):
        if self.world.agent_randomization_ranges.x_range[0] < self.world.world_x_bounds[0] or \
           self.world.agent_randomization_ranges.x_range[1] > self.world.world_x_bounds[1]:
            print(f"Warning: Agent X randomization range {self.world.agent_randomization_ranges.x_range} may exceed world X bounds {self.world.world_x_bounds}. Clamping randomization to bounds.")
            self.world.agent_randomization_ranges.x_range = (
                max(self.world.agent_randomization_ranges.x_range[0], self.world.world_x_bounds[0]),
                min(self.world.agent_randomization_ranges.x_range[1], self.world.world_x_bounds[1])
            )
        if self.world.agent_randomization_ranges.y_range[0] < self.world.world_y_bounds[0] or \
           self.world.agent_randomization_ranges.y_range[1] > self.world.world_y_bounds[1]:
            print(f"Warning: Agent Y randomization range {self.world.agent_randomization_ranges.y_range} may exceed world Y bounds {self.world.world_y_bounds}. Clamping randomization to bounds.")
            self.world.agent_randomization_ranges.y_range = (
                max(self.world.agent_randomization_ranges.y_range[0], self.world.world_y_bounds[0]),
                min(self.world.agent_randomization_ranges.y_range[1], self.world.world_y_bounds[1])
            )
        
        if self.world.landmark_randomization_ranges.depth_range[0] < self.world.landmark_depth_bounds[0] or \
           self.world.landmark_randomization_ranges.depth_range[1] > self.world.landmark_depth_bounds[1]:
            print(f"Warning: Landmark depth randomization range {self.world.landmark_randomization_ranges.depth_range} may exceed landmark depth bounds {self.world.landmark_depth_bounds}. Clamping randomization to bounds.")
            self.world.landmark_randomization_ranges.depth_range = (
                max(self.world.landmark_randomization_ranges.depth_range[0], self.world.landmark_depth_bounds[0]),
                min(self.world.landmark_randomization_ranges.depth_range[1], self.world.landmark_depth_bounds[1])
            )


    def _resolve_estimator_config(self):
        if isinstance(self.world.estimator_config, type): 
            if self.world.estimator_config == ParticleFilterConfig:
                self.world.estimator_config = self.particle_filter
            elif self.world.estimator_config == LeastSquaresConfig:
                self.world.estimator_config = self.least_squares
            else: 
                 self.world.estimator_config = self.least_squares
        elif not isinstance(self.world.estimator_config, (ParticleFilterConfig, LeastSquaresConfig)):
             print("Warning: Invalid estimator_config type in WorldConfig, defaulting to LeastSquares.")
             self.world.estimator_config = self.least_squares
    
    def _update_models_dir_per_algo(self):
        if self.algorithm.lower() == "sac":
            self.training.models_dir = "models/sac/"
        elif self.algorithm.lower() == "ppo":
            self.training.models_dir = "models/ppo/"

# --- Define Base Default Configurations ---
default_config = DefaultConfig() # This is SAC MLP by default

# --- Signal Variation Configs (based on default SAC MLP) ---
default_config_poor_signal = default_config.model_copy(deep=True)
default_config_poor_signal.world.range_measurement_base_noise = 0.05
default_config_poor_signal.world.range_measurement_distance_factor = 0.005
default_config_poor_signal.world.new_measurement_probability = 0.5

default_config_good_signal = default_config.model_copy(deep=True)
default_config_good_signal.world.range_measurement_base_noise = 0.002
default_config_good_signal.world.range_measurement_distance_factor = 0.0002
default_config_good_signal.world.new_measurement_probability = 0.95

# --- Algorithm & Feature Specific Base Configs ---
sac_rnn_config = DefaultConfig(algorithm="sac")
sac_rnn_config.sac.use_rnn = True

sac_per_config = default_config.model_copy(deep=True) # Based on SAC MLP
sac_per_config.sac.use_per = True

ppo_mlp_config = DefaultConfig(algorithm="ppo") # This is PPO MLP
ppo_mlp_config.ppo.use_rnn = False # Ensure MLP

ppo_rnn_config = DefaultConfig(algorithm="ppo") 
ppo_rnn_config.ppo.use_rnn = True
ppo_rnn_config.ppo.steps_per_update = 256 
ppo_rnn_config.ppo.batch_size = 16        

# --- Initialize CONFIGS dictionary ---
CONFIGS: Dict[str, DefaultConfig] = {
    "default": default_config, # SAC MLP
    "default_poor_signal": default_config_poor_signal, # SAC MLP, poor signal
    "default_good_signal": default_config_good_signal, # SAC MLP, good signal
    "sac_rnn": sac_rnn_config,
    "sac_per": sac_per_config, # SAC MLP with PER
    "ppo_mlp": ppo_mlp_config, # Renamed from ppo_config_obj for clarity
    "ppo_rnn": ppo_rnn_config, 
}

# --- SAC MLP Hyperparameter Variations (based on `default_config`) ---
sac_mlp_variations_list = [
    ("actor_lr_low", "sac.actor_lr", 1e-5), ("actor_lr_high", "sac.actor_lr", 1e-4),
    ("critic_lr_low", "sac.critic_lr", 1e-5), ("critic_lr_high", "sac.critic_lr", 1e-4),
    ("gamma_low", "sac.gamma", 0.95), ("gamma_high", "sac.gamma", 0.995),
    ("tau_low", "sac.tau", 0.001), ("tau_high", "sac.tau", 0.01),
    ("hidden_dims_small", "sac.hidden_dims", [32, 32]),
    ("hidden_dims_large", "sac.hidden_dims", [128, 128]),
    ("alpha_low", "sac.alpha", 0.1), ("alpha_high", "sac.alpha", 0.5),
]
for name_suffix, param_path, value in sac_mlp_variations_list:
    config_name = f"sac_mlp_{name_suffix}"
    new_config = default_config.model_copy(deep=True) # Base is sac_mlp
    new_config.algorithm = "sac" # Explicitly set
    new_config.sac.use_rnn = False # Ensure MLP
    parts = param_path.split(".")
    attr_to_set = new_config
    for part in parts[:-1]:
        attr_to_set = getattr(attr_to_set, part)
    setattr(attr_to_set, parts[-1], value)
    CONFIGS[config_name] = new_config

# --- PPO MLP Hyperparameter Variations (based on `ppo_mlp_config`) ---
ppo_mlp_variations_list = [
    ("actor_lr_low", "ppo.actor_lr", 1e-6), ("actor_lr_high", "ppo.actor_lr", 1e-5),
    ("critic_lr_low", "ppo.critic_lr", 5e-4), ("critic_lr_high", "ppo.critic_lr", 5e-3),
    ("gae_lambda_low", "ppo.gae_lambda", 0.90), ("gae_lambda_high", "ppo.gae_lambda", 0.98),
    ("policy_clip_low", "ppo.policy_clip", 0.02), ("policy_clip_high", "ppo.policy_clip", 0.1),
    ("entropy_coef_low", "ppo.entropy_coef", 0.005), ("entropy_coef_high", "ppo.entropy_coef", 0.05),
    ("hidden_dim_small", "ppo.hidden_dim", 128), ("hidden_dim_large", "ppo.hidden_dim", 512),
    ("n_epochs_low", "ppo.n_epochs", 2), ("n_epochs_high", "ppo.n_epochs", 5),
]
for name_suffix, param_path, value in ppo_mlp_variations_list:
    config_name = f"ppo_mlp_{name_suffix}"
    new_config = ppo_mlp_config.model_copy(deep=True) # Base is ppo_mlp
    new_config.algorithm = "ppo" # Explicitly set
    new_config.ppo.use_rnn = False # Ensure MLP
    parts = param_path.split(".")
    attr_to_set = new_config
    for part in parts[:-1]:
        attr_to_set = getattr(attr_to_set, part)
    setattr(attr_to_set, parts[-1], value)
    CONFIGS[config_name] = new_config

# --- Re-run post_init for ALL configs to apply updates and sync dependent params ---
config_instances_to_initialize = list(CONFIGS.values()) # Get all instances from the dict

for cfg_instance in config_instances_to_initialize:
    cfg_instance.model_post_init(None)


# --- List of all defined configurations ---
# Base Configurations:
# "default": DefaultConfig() - SAC MLP with standard signal
# "default_poor_signal": DefaultConfig() - SAC MLP with poor signal/high noise
# "default_good_signal": DefaultConfig() - SAC MLP with good signal/low noise
# "sac_rnn": DefaultConfig(algorithm="sac", sac.use_rnn=True)
# "sac_per": DefaultConfig(sac.use_per=True) - SAC MLP with Prioritized Experience Replay
# "ppo_mlp": DefaultConfig(algorithm="ppo", ppo.use_rnn=False)
# "ppo_rnn": DefaultConfig(algorithm="ppo", ppo.use_rnn=True, ppo.steps_per_update=256, ppo.batch_size=16)
#
# SAC MLP Hyperparameter Variations (based on "default"):
# "sac_mlp_actor_lr_low"
# "sac_mlp_actor_lr_high"
# "sac_mlp_critic_lr_low"
# "sac_mlp_critic_lr_high"
# "sac_mlp_gamma_low"
# "sac_mlp_gamma_high"
# "sac_mlp_tau_low"
# "sac_mlp_tau_high"
# "sac_mlp_hidden_dims_small"
# "sac_mlp_hidden_dims_large"
# "sac_mlp_alpha_low"
# "sac_mlp_alpha_high"
#
# PPO MLP Hyperparameter Variations (based on "ppo_mlp"):
# "ppo_mlp_actor_lr_low"
# "ppo_mlp_actor_lr_high"
# "ppo_mlp_critic_lr_low"
# "ppo_mlp_critic_lr_high"
# "ppo_mlp_gae_lambda_low"
# "ppo_mlp_gae_lambda_high"
# "ppo_mlp_policy_clip_low"
# "ppo_mlp_policy_clip_high"
# "ppo_mlp_entropy_coef_low"
# "ppo_mlp_entropy_coef_high"
# "ppo_mlp_hidden_dim_small"
# "ppo_mlp_hidden_dim_large"
# "ppo_mlp_n_epochs_low"
# "ppo_mlp_n_epochs_high"