import streamlit as st
import os
import json
import shutil
import time
from typing import Optional, List, Dict, Tuple, Any

from simulation_bundle import (
    AppSimConfig, WorldSimConfig, SACSimConfig, PPOSimConfig,
    ParticleFilterSimConfig, LeastSquaresSimConfig, VisualizationSimConfig,
    LocationConfig as SimLocationConfig,
    VelocityConfig as SimVelocityConfig,
    RandomizationRangeConfig as SimRandomizationRangeConfig,
    VelocityRandomizationRangeConfig as SimVelocityRandomizationRangeConfig,
    run_simulation_for_streamlit
)

from configs import DefaultConfig, ParticleFilterConfig, LeastSquaresConfig # For type checking loaded config

EXPERIMENTS_DIR = "experiments"
TEMP_VIS_DIR_BASE = "streamlit_temp_vis"

# Define the base agent types for selection
BASE_AGENT_TYPES = {
    "SAC_MLP": "SAC MLP (Most Recent)",
    "SAC_RNN": "SAC RNN (Most Recent)",
    "PPO_MLP": "PPO MLP (Most Recent)",
    "PPO_RNN": "PPO RNN (Most Recent)",
}

st.set_page_config(layout="wide")

def get_experiment_details() -> List[Dict[str, Any]]:
    """
    Scans the EXPERIMENTS_DIR, reads config.json for each, and extracts relevant details.
    Returns a list of dictionaries, each representing an experiment.
    """
    experiment_details_list = []
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []

    for exp_dir_name in os.listdir(EXPERIMENTS_DIR):
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_dir_name)
        if not os.path.isdir(exp_path):
            continue

        config_json_path = os.path.join(exp_path, "config.json")
        
        # Look for model.pt inside a "models" subdirectory first, then directly in exp_path
        model_path_to_use = None
        models_subdir = os.path.join(exp_path, "models")
        if os.path.isdir(models_subdir):
            pt_files = [f for f in os.listdir(models_subdir) if f.endswith(".pt")]
            if pt_files:
                # Sort by modification time (newest first) or name if needed
                pt_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_subdir, f)), reverse=True)
                model_path_to_use = os.path.join(models_subdir, pt_files[0])
        
        if not model_path_to_use: # Fallback: check for model.pt directly in experiment_path
            model_pt_path_direct = os.path.join(exp_path, "model.pt") # Example direct name
            if os.path.isfile(model_pt_path_direct):
                 model_path_to_use = model_pt_path_direct
            # Fallback for other common names like "actor.pt" or "final_model.pt" could be added
            # For now, we'll assume it's in models/ or a single "model.pt"

        if os.path.exists(config_json_path) and model_path_to_use: # Must have config and a model
            try:
                with open(config_json_path, 'r') as f:
                    config_data = json.load(f)
                
                algo = config_data.get("algorithm", "unknown").lower()
                use_rnn = False
                if algo == "sac":
                    use_rnn = config_data.get("sac", {}).get("use_rnn", False)
                elif algo == "ppo":
                    use_rnn = config_data.get("ppo", {}).get("use_rnn", False)

                agent_key = f"{algo.upper()}_{'RNN' if use_rnn else 'MLP'}"
                
                experiment_details_list.append({
                    "dir_name": exp_dir_name,
                    "full_path": exp_path,
                    "config_path": config_json_path,
                    "model_path": model_path_to_use,
                    "algorithm": algo,
                    "use_rnn": use_rnn,
                    "agent_key": agent_key,
                    "timestamp": os.path.getmtime(exp_path)
                })
            except Exception as e:
                print(f"Error processing experiment {exp_dir_name}: {e}")
    
    experiment_details_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return experiment_details_list


def find_representative_experiment(
    all_experiments: List[Dict[str, Any]], 
    target_agent_key: str
) -> Optional[Dict[str, Any]]:
    for exp_detail in all_experiments:
        if exp_detail["agent_key"] == target_agent_key:
            return exp_detail
    return None


def main():
    st.title("RL Range Only Tracker Simulation")

    all_exp_details = get_experiment_details()
    if not all_exp_details:
        st.error(f"No valid experiments (with config.json and a model file) found in '{EXPERIMENTS_DIR}'.")
        return

    available_agent_display_names = []
    display_name_to_key_map = {} 
    
    for key, display_name in BASE_AGENT_TYPES.items():
        if any(exp["agent_key"] == key for exp in all_exp_details):
            available_agent_display_names.append(display_name)
            display_name_to_key_map[display_name] = key

    if not available_agent_display_names:
        st.error("No experiments found matching the predefined base agent types (SAC/PPO, MLP/RNN).")
        return

    st.sidebar.header("Simulation Setup")
    selected_display_name = st.sidebar.selectbox(
        "Select Agent Type",
        available_agent_display_names
    )

    if selected_display_name:
        selected_agent_key = display_name_to_key_map[selected_display_name]
        representative_exp = find_representative_experiment(all_exp_details, selected_agent_key)

        if not representative_exp:
            st.sidebar.error(f"Could not find a representative experiment for {selected_display_name}.")
            return

        config_json_path = representative_exp["config_path"]
        model_file_path = representative_exp["model_path"]
        
        st.sidebar.info(f"Selected: {selected_display_name}")
        st.sidebar.caption(f"Using experiment: {representative_exp['dir_name']}")
        st.sidebar.caption(f"Model: {os.path.relpath(model_file_path, EXPERIMENTS_DIR)}")


        with open(config_json_path, 'r') as f:
            exp_config_dict = json.load(f)
        
        try:
            # Load into the project's DefaultConfig to access all original structured data
            original_exp_config = DefaultConfig(**exp_config_dict)
        except Exception as e:
            st.error(f"Error parsing experiment's config.json from {representative_exp['dir_name']}: {e}")
            st.exception(e)
            return

        st.sidebar.subheader("Initial Conditions")
        use_random_init = st.sidebar.checkbox("Random Initial Positions", value=True)

        agent_x_init, agent_y_init = 0.0, 0.0
        landmark_x_init, landmark_y_init, landmark_depth_init = 42.0, 42.0, 42.0

        if not use_random_init:
            agent_x_init = st.sidebar.number_input("Agent Initial X", value=original_exp_config.world.agent_initial_location.x, step=1.0)
            agent_y_init = st.sidebar.number_input("Agent Initial Y", value=original_exp_config.world.agent_initial_location.y, step=1.0)
            st.sidebar.markdown("---")
            landmark_x_init = st.sidebar.number_input("Landmark Initial X", value=original_exp_config.world.landmark_initial_location.x, step=1.0)
            landmark_y_init = st.sidebar.number_input("Landmark Initial Y", value=original_exp_config.world.landmark_initial_location.y, step=1.0)
            landmark_depth_init = st.sidebar.number_input("Landmark Initial Depth", value=original_exp_config.world.landmark_initial_location.depth, step=1.0)
        
        default_steps = 200
        if hasattr(original_exp_config, 'evaluation') and hasattr(original_exp_config.evaluation, 'max_steps'):
            default_steps = original_exp_config.evaluation.max_steps or 200
        num_steps = st.sidebar.slider("Number of Simulation Steps", 50, 500, default_steps)
        
        world_sim_cfg = WorldSimConfig(
            dt=original_exp_config.world.dt,
            agent_speed=original_exp_config.world.agent_speed,
            yaw_angle_range=original_exp_config.world.yaw_angle_range,
            world_x_bounds=original_exp_config.world.world_x_bounds,
            world_y_bounds=original_exp_config.world.world_y_bounds,
            landmark_depth_bounds=original_exp_config.world.landmark_depth_bounds,
            normalize_state=original_exp_config.world.normalize_state,
            agent_initial_location=SimLocationConfig(**original_exp_config.world.agent_initial_location.model_dump()),
            landmark_initial_location=SimLocationConfig(**original_exp_config.world.landmark_initial_location.model_dump()),
            landmark_initial_velocity=SimVelocityConfig(**original_exp_config.world.landmark_initial_velocity.model_dump()),
            randomize_agent_initial_location=use_random_init, 
            randomize_landmark_initial_location=use_random_init, 
            randomize_landmark_initial_velocity=original_exp_config.world.randomize_landmark_initial_velocity,
            agent_randomization_ranges=SimRandomizationRangeConfig(**original_exp_config.world.agent_randomization_ranges.model_dump()),
            landmark_randomization_ranges=SimRandomizationRangeConfig(**original_exp_config.world.landmark_randomization_ranges.model_dump()),
            landmark_velocity_randomization_ranges=SimVelocityRandomizationRangeConfig(**original_exp_config.world.landmark_velocity_randomization_ranges.model_dump()),
            trajectory_length=original_exp_config.world.trajectory_length,
            trajectory_feature_dim=original_exp_config.world.trajectory_feature_dim,
            range_measurement_base_noise=original_exp_config.world.range_measurement_base_noise,
            range_measurement_distance_factor=original_exp_config.world.range_measurement_distance_factor,
            success_threshold=original_exp_config.world.success_threshold,
            collision_threshold=original_exp_config.world.collision_threshold,
            new_measurement_probability=original_exp_config.world.new_measurement_probability
            # Removed unused reward parameters from here
        )

        # Determine and set the estimator configuration for the simulation
        # original_exp_config.world.estimator_config can be a Pydantic model or a dict (if loaded from JSON)
        estimator_source = original_exp_config.world.estimator_config
        if isinstance(estimator_source, ParticleFilterConfig) or \
           (isinstance(estimator_source, dict) and 'num_particles' in estimator_source):
            data_to_use = estimator_source.model_dump() if isinstance(estimator_source, ParticleFilterConfig) else estimator_source
            world_sim_cfg.estimator_config = ParticleFilterSimConfig(**data_to_use)
        elif isinstance(estimator_source, LeastSquaresConfig) or \
             (isinstance(estimator_source, dict) and 'history_size' in estimator_source): # Assuming 'history_size' implies LS
            data_to_use = estimator_source.model_dump() if isinstance(estimator_source, LeastSquaresConfig) else estimator_source
            world_sim_cfg.estimator_config = LeastSquaresSimConfig(**data_to_use)
        else:
            st.warning("Could not determine specific estimator type from config, defaulting to LeastSquares for simulation.")
            world_sim_cfg.estimator_config = LeastSquaresSimConfig()


        sac_sim_cfg = None; ppo_sim_cfg = None
        if original_exp_config.algorithm.lower() == "sac":
            sac_orig = original_exp_config.sac
            sac_sim_cfg = SACSimConfig(
                state_dim=sac_orig.state_dim, action_dim=sac_orig.action_dim, 
                hidden_dims=sac_orig.hidden_dims, log_std_min=sac_orig.log_std_min, 
                log_std_max=sac_orig.log_std_max, use_rnn=sac_orig.use_rnn, 
                rnn_type=sac_orig.rnn_type, rnn_hidden_size=sac_orig.rnn_hidden_size, 
                rnn_num_layers=sac_orig.rnn_num_layers
            )
        elif original_exp_config.algorithm.lower() == "ppo":
            ppo_orig = original_exp_config.ppo
            ppo_sim_cfg = PPOSimConfig(
                state_dim=ppo_orig.state_dim, action_dim=ppo_orig.action_dim, 
                hidden_dim=ppo_orig.hidden_dim, log_std_min=ppo_orig.log_std_min, 
                log_std_max=ppo_orig.log_std_max, use_rnn=ppo_orig.use_rnn, 
                rnn_type=ppo_orig.rnn_type, rnn_hidden_size=ppo_orig.rnn_hidden_size, 
                rnn_num_layers=ppo_orig.rnn_num_layers
            )

        app_sim_config = AppSimConfig(
            sac=sac_sim_cfg, ppo=ppo_sim_cfg, world=world_sim_cfg,
            # These particle_filter and least_squares fields are top-level defaults in AppSimConfig,
            # but world_sim_cfg.estimator_config is the one that will be used by WorldSim.
            particle_filter=ParticleFilterSimConfig(**original_exp_config.particle_filter.model_dump()), 
            least_squares=LeastSquaresSimConfig(**original_exp_config.least_squares.model_dump()), 
            visualization=VisualizationSimConfig(**original_exp_config.visualization.model_dump()),
            cuda_device=original_exp_config.cuda_device, # Will default to CPU in agent eval
            algorithm=original_exp_config.algorithm
        )
        
        if st.sidebar.button("Run Simulation"):
            run_timestamp = time.strftime("%Y%m%d-%H%M%S")
            current_run_vis_dir = os.path.join(TEMP_VIS_DIR_BASE, f"{representative_exp['dir_name']}_{selected_agent_key}_{run_timestamp}")
            os.makedirs(current_run_vis_dir, exist_ok=True)

            st.markdown("---"); st.subheader("Simulation Results")
            status_text = st.empty()
            # Create the progress bar widget here. It will be updated by the simulation function.
            progress_bar_widget = st.progress(0)
            gif_display = st.empty()
            metrics_display = st.empty(); reward_plot_display = st.empty()
            
            status_text.info("Running simulation, please wait...")

            try:
                gif_path, final_error, total_reward, episode_rewards = run_simulation_for_streamlit(
                    app_sim_config=app_sim_config, model_path=model_file_path,
                    initial_agent_loc_data=(agent_x_init, agent_y_init) if not use_random_init else None,
                    initial_landmark_loc_data=(landmark_x_init, landmark_y_init, landmark_depth_init) if not use_random_init else None,
                    random_initialization=use_random_init, num_simulation_steps=num_steps,
                    visualization_output_dir=current_run_vis_dir,
                    st_progress_bar=progress_bar_widget # Pass the progress bar widget
                )
                status_text.success("Simulation complete!")
                if gif_path and os.path.exists(gif_path): gif_display.image(gif_path, caption="Simulation Animation")
                else: gif_display.warning("GIF could not be generated or found.")
                metrics_display.markdown(f"- **Final Error (2D):** {final_error:.2f}\n- **Total Reward:** {total_reward:.2f}")
                if episode_rewards:
                    import pandas as pd
                    reward_df = pd.DataFrame({'Step': range(len(episode_rewards)), 'Raw Reward': episode_rewards})
                    reward_plot_display.line_chart(reward_df.set_index('Step'))
            except Exception as e:
                status_text.error(f"Simulation error: {e}"); import traceback; st.error(traceback.format_exc())
                progress_bar_widget.progress(1.0) # Ensure progress shows 100% on error too
            finally:
                # The simulation bundle itself will set progress to 1.0 on normal completion or early exit.
                # The except block above handles errors from the simulation function.
                if os.path.exists(current_run_vis_dir):
                    try: shutil.rmtree(current_run_vis_dir)
                    except Exception as e_clean: print(f"Warn: Cleanup failed {current_run_vis_dir}: {e_clean}")

if __name__ == "__main__":
    if not os.path.exists(TEMP_VIS_DIR_BASE): os.makedirs(TEMP_VIS_DIR_BASE)
    main()