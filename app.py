import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import math
from collections import deque
from typing import Dict, Tuple, Any, List, Optional, Literal
from scipy.spatial import ConvexHull, Delaunay
import warnings
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import imageio.v2 as imageio
from PIL import Image
import matplotlib.animation as animation
import time
import traceback # For more detailed error printing

# --- Pydantic Models ---
from pydantic import BaseModel, Field

class SACConfigPydantic(BaseModel):
    state_dim: int
    action_dim: int
    hidden_dims: List[int]
    log_std_min: int
    log_std_max: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    alpha: float
    auto_tune_alpha: bool
    use_rnn: bool
    rnn_type: Literal['lstm', 'gru']
    rnn_hidden_size: int
    rnn_num_layers: int
    use_state_normalization: bool
    use_reward_normalization: bool
    use_per: bool
    per_alpha: float
    per_beta_start: float
    per_beta_frames: int
    per_epsilon: float

class PositionPydantic(BaseModel):
    x: float
    y: float

class RandomizationRangePydantic(BaseModel):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

class MapperConfigPydantic(BaseModel):
    min_oil_points_for_estimate: int

class WorldConfigPydantic(BaseModel):
    CORE_STATE_DIM: int = Field(8)
    CORE_ACTION_DIM: int = Field(1)
    TRAJECTORY_REWARD_DIM: int = Field(1)
    dt: float
    world_size: Tuple[float, float]
    normalize_coords: bool
    agent_speed: float
    yaw_angle_range: Tuple[float, float]
    num_sensors: int
    sensor_distance: float
    sensor_radius: float
    agent_initial_location: PositionPydantic
    randomize_agent_initial_location: bool
    agent_randomization_ranges: RandomizationRangePydantic
    num_oil_points: int
    num_water_points: int
    oil_cluster_std_dev_range: Tuple[float, float]
    randomize_oil_cluster: bool
    oil_center_randomization_range: RandomizationRangePydantic
    initial_oil_center: PositionPydantic
    initial_oil_std_dev: float
    min_initial_separation_distance: float
    trajectory_length: int
    trajectory_feature_dim: int
    success_metric_threshold: float
    terminate_on_success: bool
    terminate_out_of_bounds: bool
    metric_improvement_scale: float
    step_penalty: float
    new_oil_detection_bonus: float
    out_of_bounds_penalty: float
    success_bonus: float
    uninitialized_mapper_penalty: float
    mapper_config: MapperConfigPydantic
    seeds: List[int]
    max_steps: int = Field(300)

class EvaluationConfigPydantic(BaseModel):
    num_episodes: int
    max_steps: int
    render: bool
    use_stochastic_policy_eval: bool

class VisualizationConfigPydantic(BaseModel):
    save_dir: str = "streamlit_outputs"
    figure_size: Tuple[int, int]
    max_trajectory_points: int
    output_format: Literal['gif', 'mp4']
    video_fps: int
    delete_png_frames: bool
    sensor_marker_size: int
    sensor_color_oil: str
    sensor_color_water: str
    plot_oil_points: bool
    plot_water_points: bool
    point_marker_size: int

class AppConfigPydantic(BaseModel):
    sac: SACConfigPydantic
    world: WorldConfigPydantic
    visualization: VisualizationConfigPydantic
    evaluation: EvaluationConfigPydantic
    cuda_device: str = "cpu"

# --- World Objects ---
class Velocity:
    def __init__(self, x: float, y: float): self.x = x; self.y = y
    def is_moving(self) -> bool: return self.x != 0 or self.y != 0
    def get_heading(self) -> float: return math.atan2(self.y, self.x) if self.is_moving() else 0.0

class Location:
    def __init__(self, x: float, y: float): self.x = x; self.y = y
    def update(self, velocity: Velocity, dt: float = 1.0): self.x += velocity.x * dt; self.y += velocity.y * dt
    def distance_to(self, other_loc: 'Location') -> float: return math.sqrt((self.x - other_loc.x)**2 + (self.y - other_loc.y)**2)
    def get_normalized(self, world_size: Tuple[float, float]) -> Tuple[float, float]:
        ws_x, ws_y = world_size
        norm_x = max(0.0, min(1.0, self.x / ws_x)) if ws_x > 0 else 0.0
        norm_y = max(0.0, min(1.0, self.y / ws_y)) if ws_y > 0 else 0.0
        return norm_x, norm_y

class WorldObject:
    def __init__(self, location: Location, velocity: Optional[Velocity] = None, name: Optional[str] = None):
        self.name = name or "Unnamed"; self.location = location
        self.velocity = velocity or Velocity(0.0, 0.0)
    def update_position(self, dt: float = 1.0):
        if self.velocity and self.velocity.is_moving(): self.location.update(self.velocity, dt)
    def get_heading(self) -> float: return self.velocity.get_heading()

# --- Mapper ---
class Mapper:
    def __init__(self, config: MapperConfigPydantic): self.config = config; self.reset()
    def reset(self): self.oil_sensor_locations: List[Location] = []; self.estimated_hull: Optional[ConvexHull] = None; self.hull_vertices: Optional[np.ndarray] = None
    def add_measurement(self, sensor_loc: Location, is_oil: bool):
        if is_oil and not any(abs(p.x-sensor_loc.x)<1e-6 and abs(p.y-sensor_loc.y)<1e-6 for p in self.oil_sensor_locations):
            self.oil_sensor_locations.append(sensor_loc)
    def estimate_spill(self):
        self.estimated_hull = None; self.hull_vertices = None
        if len(self.oil_sensor_locations) < self.config.min_oil_points_for_estimate: return
        pts = np.array([[p.x, p.y] for p in self.oil_sensor_locations]); unique_pts = np.unique(pts, axis=0)
        if unique_pts.shape[0] < 3: return
        try:
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); hull = ConvexHull(unique_pts, qhull_options='QJ')
            self.estimated_hull = hull; self.hull_vertices = unique_pts[hull.vertices]
        except Exception: pass
    def is_inside_estimate(self, point: Location) -> bool:
        if self.hull_vertices is None or len(self.hull_vertices) < 3: return False
        try: return Delaunay(self.hull_vertices, qhull_options='QJ').find_simplex(np.array([point.x, point.y])) >= 0
        except Exception: return False

# --- World Environment ---
class World:
    def __init__(self, world_config: WorldConfigPydantic):
        self.world_config = wc = world_config; self.mapper_config = wc.mapper_config
        self.dt=wc.dt; self.agent_speed=wc.agent_speed; self.max_yaw_change=wc.yaw_angle_range[1]
        self.num_sensors=wc.num_sensors; self.sensor_distance=wc.sensor_distance; self.sensor_radius=wc.sensor_radius
        self.trajectory_length=wc.trajectory_length; self.CORE_STATE_DIM=wc.CORE_STATE_DIM
        self.CORE_ACTION_DIM=wc.CORE_ACTION_DIM; self.TRAJECTORY_REWARD_DIM=wc.TRAJECTORY_REWARD_DIM
        self.feature_dim = self.CORE_STATE_DIM + self.CORE_ACTION_DIM + self.TRAJECTORY_REWARD_DIM
        self.world_size=wc.world_size; self.normalize_coords=wc.normalize_coords
        self.agent: Optional[WorldObject]=None; self.true_oil_points:List[Location]=[]; self.mapper=Mapper(self.mapper_config)
        self.current_seed=None; self.reward=0.0; self.performance_metric=0.0; self.done=False; self.current_step=0
        self._trajectory_history=deque(maxlen=self.trajectory_length)

    def _seed_environment(self, seed:Optional[int]=None):
        if seed is None: seed=random.randint(0,2**32-1)
        self.current_seed=seed; random.seed(seed); np.random.seed(seed)

    def reset(self, seed:Optional[int]=None, custom_agent_loc:Optional[Tuple[float,float]]=None, custom_oil_center:Optional[Tuple[float,float]]=None):
        self.current_step=0; self.done=False; self.reward=0.0; self.performance_metric=0.0
        self._seed_environment(seed); self.true_oil_points=[]; world_w,world_h = self.world_size
        if custom_oil_center:
            oil_center = Location(x=custom_oil_center[0],y=custom_oil_center[1])
            oil_std_dev = np.mean(self.world_config.oil_cluster_std_dev_range) if self.world_config.oil_cluster_std_dev_range else self.world_config.initial_oil_std_dev
        else:
            oil_center = Location(x=self.world_config.initial_oil_center.x, y=self.world_config.initial_oil_center.y)
            oil_std_dev = self.world_config.initial_oil_std_dev
        for _ in range(self.world_config.num_oil_points):
            px=np.random.normal(oil_center.x,oil_std_dev); py=np.random.normal(oil_center.y,oil_std_dev)
            self.true_oil_points.append(Location(max(0.,min(world_w,px)),max(0.,min(world_h,py))))
        if custom_agent_loc: agent_loc = Location(x=custom_agent_loc[0],y=custom_agent_loc[1])
        else: agent_loc = Location(x=self.world_config.agent_initial_location.x,y=self.world_config.agent_initial_location.y)
        h=random.uniform(-math.pi,math.pi); v=Velocity(self.agent_speed*math.cos(h),self.agent_speed*math.sin(h))
        self.agent=WorldObject(location=agent_loc,velocity=v); self.mapper.reset(); self._initialize_trajectory_history()
        s_loc,s_read = self._get_sensor_readings()
        for loc,read in zip(s_loc,s_read): self.mapper.add_measurement(loc,read)
        self.mapper.estimate_spill(); self._calculate_performance_metric()
        return self.encode_state()

    def _get_sensor_locations(self) -> List[Location]:
        if not self.agent: return []
        s_locs=[]; al=self.agent.location; ah=self.agent.get_heading()
        ao=np.linspace(-math.pi/2,math.pi/2,self.num_sensors) if self.num_sensors>1 else [0.]
        for off in ao: 
            sa=ah+off; sx=al.x+self.sensor_distance*math.cos(sa); sy=al.y+self.sensor_distance*math.sin(sa)
            s_locs.append(Location(max(0.,min(self.world_size[0],sx)),max(0.,min(self.world_size[1],sy))))
        return s_locs

    def _get_sensor_readings(self) -> Tuple[List[Location],List[bool]]:
        sl=self._get_sensor_locations(); sr=[False]*self.num_sensors
        if not self.true_oil_points: return sl,sr
        for i,s_loc in enumerate(sl):
            for op in self.true_oil_points:
                if s_loc.distance_to(op)<=self.sensor_radius: sr[i]=True; break
        return sl,sr

    def _calculate_performance_metric(self):
        if self.mapper.hull_vertices is None or not self.true_oil_points: self.performance_metric=0.; return
        pi=sum(1 for op in self.true_oil_points if self.mapper.is_inside_estimate(op))
        self.performance_metric=pi/len(self.true_oil_points) if len(self.true_oil_points) > 0 else 0.0


    def _get_basic_state_tuple_normalized(self) -> Tuple:
        if not self.agent: return tuple([0.]*self.CORE_STATE_DIM)
        _,sr_bool=self._get_sensor_readings(); sr_f=[1. if r else 0. for r in sr_bool]
        aln=self.agent.location.get_normalized(self.world_size); ahn=self.agent.get_heading()/math.pi
        return tuple(sr_f+list(aln)+[ahn])

    def _initialize_trajectory_history(self):
        if not self.agent: self.agent = WorldObject(Location(0,0),Velocity(0,0))
        ibsn=self._get_basic_state_tuple_normalized()
        feat=np.concatenate([np.array(ibsn,dtype=np.float32),np.zeros(self.CORE_ACTION_DIM,dtype=np.float32),np.zeros(self.TRAJECTORY_REWARD_DIM,dtype=np.float32)])
        self._trajectory_history.clear()
        for _ in range(self.trajectory_length): self._trajectory_history.append(feat)

    def step(self, yaw_norm:float):
        if self.done or not self.agent: return self.encode_state()
        pbsn=self._get_basic_state_tuple_normalized(); pa=yaw_norm; r_prev=self.reward
        s_loc,s_read=self._get_sensor_readings()
        yc=yaw_norm*self.max_yaw_change; ch=self.agent.get_heading(); nh=(ch+yc+math.pi)%(2*math.pi)-math.pi
        self.agent.velocity=Velocity(self.agent_speed*math.cos(nh),self.agent_speed*math.sin(nh))
        self.agent.update_position(self.dt)
        oob=not(0<=self.agent.location.x<=self.world_size[0] and 0<=self.agent.location.y<=self.world_size[1])
        if oob and self.world_config.terminate_out_of_bounds: self.done=True; self.reward=-self.world_config.out_of_bounds_penalty
        else:
            if oob: # Agent went out of bounds but we don't terminate, just penalize and clamp
                self.agent.location.x = max(0.0, min(self.world_size[0], self.agent.location.x))
                self.agent.location.y = max(0.0, min(self.world_size[1], self.agent.location.y))
                self.reward = -self.world_config.out_of_bounds_penalty 
            else: # Normal step
                self.reward=self.performance_metric-self.world_config.step_penalty # Base reward
            
            for l,r in zip(s_loc,s_read): self.mapper.add_measurement(l,r)
            self.mapper.estimate_spill(); self._calculate_performance_metric()
            
            # Add metric improvement bonus if not OOB
            if not oob:
                # metric_delta = self.performance_metric - self.previous_performance_metric # previous_performance_metric not stored yet
                # self.reward += self.world_config.metric_improvement_scale * max(0, metric_delta)
                 self.reward += self.world_config.metric_improvement_scale * self.performance_metric # Simplified to current metric scaled


            if self.performance_metric>=self.world_config.success_metric_threshold:
                self.reward+=self.world_config.success_bonus
                if self.world_config.terminate_on_success: self.done=True
        
        self.current_step+=1
        if self.current_step >= self.world_config.max_steps : self.done=True
        
        cfv=np.concatenate([np.array(pbsn,dtype=np.float32),np.array([pa],dtype=np.float32),np.array([r_prev],dtype=np.float32)])
        self._trajectory_history.append(cfv)
        return self.encode_state()

    def encode_state(self) -> Dict[str,Any]:
        bsn=self._get_basic_state_tuple_normalized(); ft=np.array(self._trajectory_history,dtype=np.float32)
        if ft.shape!=(self.trajectory_length,self.feature_dim): self._initialize_trajectory_history(); ft=np.array(self._trajectory_history,dtype=np.float32)
        return {"basic_state":bsn, "full_trajectory":ft}

# --- SAC Agent (Lightweight - RNN variant) ---
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.device = device or torch.device("cpu"); self.mean = torch.zeros(shape,dtype=torch.float32,device=self.device)
        self.var = torch.ones(shape,dtype=torch.float32,device=self.device); self.count = torch.tensor(epsilon,dtype=torch.float32,device=self.device)
        self.epsilon = epsilon; self._is_eval = False
    def update(self, x:torch.Tensor):
        if self._is_eval:return
        x=x.to(self.device);
        if x.dim()==1: x=x.unsqueeze(0)
        if x.shape[0]==0: return
        bm=torch.mean(x,dim=0); bv=torch.var(x,dim=0,unbiased=False); bc=torch.tensor(x.shape[0],dtype=torch.float32,device=self.device)
        d=bm-self.mean;tc=self.count+bc;nm=self.mean+d*bc/tc;ma=self.var*self.count;mb=bv*bc;M2=ma+mb+torch.square(d)*self.count*bc/tc
        self.mean=nm;self.var=torch.clamp(M2/tc,min=0.);self.count=tc
    def normalize(self,x:torch.Tensor)->torch.Tensor: x=x.to(self.device);return torch.clamp((x-self.mean)/torch.sqrt(self.var+self.epsilon),-10.,10.)
    def state_dict(self):return{'mean':self.mean.cpu(),'var':self.var.cpu(),'count':self.count.cpu()}
    def load_state_dict(self,sd):self.mean=sd['mean'].to(self.device);self.var=sd['var'].to(self.device);self.count=sd['count'].to(self.device)
    def eval(self):self._is_eval=True
    def train(self):self._is_eval=False

class Actor(nn.Module):
    def __init__(self, config: SACConfigPydantic, world_config: WorldConfigPydantic, target_device_str: str = "cpu"):
        super().__init__()
        self.config = config
        # self.wc = world_config # No longer needed with direct use of world_config
        self.use_rnn = config.use_rnn
        
        # Use world_config directly for these attributes
        self.state_dim = world_config.CORE_STATE_DIM
        self.action_dim = world_config.CORE_ACTION_DIM
        
        dev = torch.device(target_device_str)

        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size
            self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim # Use self.state_dim
            if config.rnn_type == 'gru':
                self.rnn = nn.GRU(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True, device=dev)
            elif config.rnn_type == 'lstm':
                self.rnn = nn.LSTM(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True, device=dev)
            else:
                raise ValueError(f"Unsupported RNN type: {config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size
        else:
            mlp_input_dim = self.state_dim # Use self.state_dim
            self.rnn = None

        # --- Revert attribute names to match original model ---
        self.layers = nn.ModuleList() # Changed 'ls' back to 'layers'
        current_dim = mlp_input_dim
        hidden_dims_to_use = config.hidden_dims if config.hidden_dims else [256, 256]
        
        for hidden_dim_idx, hidden_dim in enumerate(hidden_dims_to_use):
            linear_layer = nn.Linear(current_dim, hidden_dim, device=dev)
            self.layers.append(linear_layer) # Appending to self.layers
            self.layers.append(nn.ReLU())    # Appending to self.layers
            current_dim = hidden_dim
        
        self.mean = nn.Linear(current_dim, self.action_dim, device=dev) # Changed 'm' back to 'mean'
        self.log_std = nn.Linear(current_dim, self.action_dim, device=dev) # Changed 'lgstd' back to 'log_std'
        # --- End name reversions ---
        
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, net_in: torch.Tensor, hs: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        nhs = None
        if self.use_rnn and self.rnn:
            no, nhs = self.rnn(net_in, hs)
            mi = no[:, -1, :]
        else:
            mi = net_in
        x = mi
        # Iterate through self.layers (which contains Linear and ReLU)
        for layer_idx in range(0, len(self.layers), 2): # Assuming Linear, ReLU pairs
            x = self.layers[layer_idx](x) # Linear
            x = self.layers[layer_idx+1](x) # ReLU
            
        mean_val = self.mean(x) # Use self.mean
        log_std_val = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max) # Use self.log_std
        return mean_val, log_std_val, nhs # Return the correct variables

    def sample(self, net_in: torch.Tensor, hs: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        mean_val, log_std_val, nhs = self.forward(net_in, hs) # Use correct returned variables
        std = log_std_val.exp()
        norm = Normal(mean_val, std)
        xt = norm.rsample()
        an = torch.tanh(xt)
        lpu = norm.log_prob(xt)
        ct = an.clamp(-1 + 1e-6, 1 - 1e-6)
        ldj = torch.log(1 - ct.pow(2) + 1e-7)
        lp = (lpu - ldj).sum(1, keepdim=True)
        return an, lp, torch.tanh(mean_val), nhs # Use torch.tanh(mean_val)

    def get_initial_hidden_state(self, bs: int, dev: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        # Ensure self.rnl and self.rh are used if they were assigned from config
        h0_num_layers = self.rnn_num_layers if hasattr(self, 'rnn_num_layers') else self.config.rnn_num_layers
        h0_hidden_size = self.rnn_hidden_size if hasattr(self, 'rnn_hidden_size') else self.config.rnn_hidden_size

        h0 = torch.zeros(h0_num_layers, bs, h0_hidden_size, device=dev)
        return (h0, torch.zeros_like(h0)) if self.config.rnn_type == 'lstm' else h0

class SACAgent:
    def __init__(self, config: SACConfigPydantic, world_config: WorldConfigPydantic, device_str: str = "cpu"):
        self.config=config; self.wc=world_config; self.dev_str=device_str
        self.device=torch.device(device_str)
        self.actor: Optional[Actor] = None 
        self.state_normalizer: Optional[RunningMeanStd] = None
        if config.use_state_normalization:
            self.state_normalizer = RunningMeanStd(shape=(world_config.CORE_STATE_DIM,), device=self.device)

    def _ensure_actor_instantiated(self):
        if self.actor is None:
            self.actor = Actor(self.config, self.wc, target_device_str=self.dev_str).to(self.device)

    def select_action(self, state:dict, actor_hs:Optional[Tuple]=None, evaluate:bool=True)->Tuple[float,Optional[Tuple]]:
        self._ensure_actor_instantiated()
        assert self.actor is not None, "Actor not instantiated in select_action"
        st_tr=state['full_trajectory']; stf=torch.FloatTensor(st_tr).to(self.device).unsqueeze(0)
        with torch.no_grad():
            raw_in_st=stf[:,:,:self.wc.CORE_STATE_DIM]
            act_in=raw_in_st if self.config.use_rnn else raw_in_st[:,-1,:]
            if self.config.use_state_normalization and self.state_normalizer:
                self.state_normalizer.eval(); norm_act_in=self.state_normalizer.normalize(act_in)
            else: norm_act_in=act_in
            self.actor.eval()
            if evaluate: _,_,ams,nhs=self.actor.sample(norm_act_in,actor_hs); an=ams
            else: an,_,_,nhs=self.actor.sample(norm_act_in,actor_hs)
            self.actor.train() # Set back to train mode after sampling
        return an.detach().cpu().numpy()[0,0],nhs

    def load_model(self, path: str):
        self._ensure_actor_instantiated()
        assert self.actor is not None, "Actor not instantiated before load_model"
        if not os.path.exists(path): st.error(f"Model NF: {path}"); return
        try:
            chkpt = torch.load(path, map_location=torch.device('cpu')) # Always map to CPU first
            self.actor.load_state_dict(chkpt['actor_state_dict'])
            self.actor.to(self.device) # Move to agent's target device
            if self.config.use_state_normalization and self.state_normalizer and 'state_normalizer_state_dict' in chkpt:
                self.state_normalizer.load_state_dict(chkpt['state_normalizer_state_dict'])
                self.state_normalizer.mean = self.state_normalizer.mean.to(self.device)
                self.state_normalizer.var = self.state_normalizer.var.to(self.device)
                self.state_normalizer.count = self.state_normalizer.count.to(self.device)
            st.sidebar.success(f"Model loaded from {os.path.basename(path)} to actor device: {next(self.actor.parameters()).device}")
        except Exception as e: st.error(f"Error loading model: {e}"); st.text(traceback.format_exc())

# --- Matplotlib visualization function ---
_agent_trajectory_viz: List[tuple[float,float]] = []
def visualize_world_mpl(world:World, vis_config:VisualizationConfigPydantic, fig, ax):
    global _agent_trajectory_viz
    if world.agent: _agent_trajectory_viz.append((world.agent.location.x,world.agent.location.y))
    if len(_agent_trajectory_viz)>vis_config.max_trajectory_points: _agent_trajectory_viz=_agent_trajectory_viz[-vis_config.max_trajectory_points:]
    ax.clear()
    if len(_agent_trajectory_viz)>1: tx,ty=zip(*_agent_trajectory_viz); ax.plot(tx,ty,'g-',lw=1,alpha=0.6,label='Agent Traj.')
    if vis_config.plot_oil_points and world.true_oil_points: ox,oy=zip(*[(p.x,p.y) for p in world.true_oil_points]); ax.scatter(ox,oy,c='k',marker='.',s=vis_config.point_marker_size,alpha=0.7,label='Oil')
    if world.mapper and world.mapper.hull_vertices is not None: hull_p=Polygon(world.mapper.hull_vertices,ec='r',fc='r',alpha=0.2,lw=1.5,ls='--',label=f'Est. Hull ({world.performance_metric:.2%})'); ax.add_patch(hull_p)
    if world.agent:
        ax.scatter(world.agent.location.x,world.agent.location.y,c='b',marker='o',s=60,zorder=5,label='Agent')
        h=world.agent.get_heading();al=3.;ax.arrow(world.agent.location.x,world.agent.location.y,al*math.cos(h),al*math.sin(h),head_width=1.,head_length=1.5,fc='b',ec='b',alpha=0.7,zorder=5)
        s_loc,s_read=world._get_sensor_readings()
        for i,loc in enumerate(s_loc): c=vis_config.sensor_color_oil if s_read[i] else vis_config.sensor_color_water; ax.scatter(loc.x,loc.y,c=c,marker='s',s=vis_config.sensor_marker_size,ec='k',lw=0.5,zorder=4,label='Sensors' if i==0 else "")
    ww,wh=world.world_size;ax.set_xlabel('X');ax.set_ylabel('Y')
    t1=f"Step: {world.current_step}, Metric: {world.performance_metric:.3f}";
    if world.current_seed is not None:t1+=f", Seed: {world.current_seed}"
    ax.set_title(f'Oil Spill Mapping\n{t1}');pad=5.;ax.set_xlim(-pad,ww+pad);ax.set_ylim(-pad,wh+pad);ax.set_aspect('equal',adjustable='box');ax.legend(fontsize='small',loc='upper left',bbox_to_anchor=(1.02,1.));fig.tight_layout(rect=[0,0,0.85,1])

def reset_viz_trajectories(): global _agent_trajectory_viz; _agent_trajectory_viz = []

# --- Streamlit Application UI and Logic ---
st.set_page_config(layout="wide", page_title="RL Agent Oil Spill Mapping Demo")
st.title("🛢️ RL Agent Demo: Oil Spill Mapping")
st.markdown("Set initial oil and agent positions using the number inputs, then run the simulation.")

MODEL_DIR="assets"; CONFIG_FILE_PATH=os.path.join(MODEL_DIR,"config.json"); MODEL_FILE_PATH=os.path.join(MODEL_DIR,"sac_rnn_oil_spill_model.pt")

@st.cache_data
def load_app_config(config_path):
    if not os.path.exists(config_path): st.error(f"Config NF: {config_path}"); return None
    with open(config_path,'r') as f: config_dict=json.load(f)
    try:
        eval_data = config_dict.get('evaluation', {})
        eval_config = EvaluationConfigPydantic(
            num_episodes=eval_data.get('num_episodes', 1),
            max_steps=eval_data.get('max_steps', 300),
            render=eval_data.get('render', False),
            use_stochastic_policy_eval=eval_data.get('use_stochastic_policy_eval', False)
        )
        # Make sure world config also has max_steps for the eval loop
        world_data = config_dict['world']
        if 'max_steps' not in world_data: # Add if missing from original json
            world_data['max_steps'] = eval_config.max_steps

        app_conf = AppConfigPydantic(
            sac=config_dict['sac'], 
            world=world_data, 
            visualization=config_dict['visualization'], 
            evaluation=eval_config, 
            cuda_device="cpu" 
        )
        return app_conf
    except Exception as e: st.error(f"Error parsing config: {e}"); st.text(traceback.format_exc()); return None

app_config = load_app_config(CONFIG_FILE_PATH)
if not app_config: st.stop()


# --- Sidebar UI ---
st.sidebar.header("Simulation Parameters")
max_eval_steps_default = app_config.evaluation.max_steps if app_config and app_config.evaluation else 300
max_eval_steps=st.sidebar.slider("Max Eval Steps",10, int(app_config.world.max_steps * 1.5 if app_config else 500) , max_eval_steps_default)

num_oil_points_default = app_config.world.num_oil_points if app_config else 200
num_oil_points_sidebar=st.sidebar.slider("Num Oil Pts",50,500,num_oil_points_default)

oil_std_dev_default = float(np.mean(app_config.world.oil_cluster_std_dev_range) if app_config and app_config.world.oil_cluster_std_dev_range else app_config.world.initial_oil_std_dev if app_config else 10.0)
oil_std_dev_sidebar=st.sidebar.slider("Oil Std Dev",1.,20.,oil_std_dev_default)

agent_speed_default = app_config.world.agent_speed if app_config else 3.0
agent_speed_sidebar=st.sidebar.slider("Agent Speed",1.,10.,agent_speed_default)

output_format_default_idx = ["mp4","gif"].index(app_config.visualization.output_format if app_config and app_config.visualization else "mp4")
output_format_sidebar=st.sidebar.selectbox("Output Format",["mp4","gif"],index=output_format_default_idx)

stochastic_policy_default = app_config.evaluation.use_stochastic_policy_eval if app_config and app_config.evaluation else False
stochastic_policy=st.sidebar.checkbox("Use Stochastic Policy",value=stochastic_policy_default)

# --- Main UI for Initial Conditions ---
st.subheader("Set Initial Conditions")
world_w_main, world_h_main = (app_config.world.world_size if app_config else (100.0, 100.0))

if 'oil_center_canvas' not in st.session_state:st.session_state.oil_center_canvas=None
if 'agent_start_canvas' not in st.session_state:st.session_state.agent_start_canvas=None

default_oil_x_main = st.session_state.oil_center_canvas[0] if st.session_state.oil_center_canvas else (app_config.world.initial_oil_center.x if app_config else 50.0)
default_oil_y_main = st.session_state.oil_center_canvas[1] if st.session_state.oil_center_canvas else (app_config.world.initial_oil_center.y if app_config else 50.0)
default_agent_x_main = st.session_state.agent_start_canvas[0] if st.session_state.agent_start_canvas else (app_config.world.agent_initial_location.x if app_config else 10.0)
default_agent_y_main = st.session_state.agent_start_canvas[1] if st.session_state.agent_start_canvas else (app_config.world.agent_initial_location.y if app_config else 10.0)

col1_main, col2_main = st.columns(2)
with col1_main:
    oil_x_input=st.number_input("Oil Center X",0.0,float(world_w_main),float(default_oil_x_main),1.0, key="oil_x")
    oil_y_input=st.number_input("Oil Center Y",0.0,float(world_h_main),float(default_oil_y_main),1.0, key="oil_y")
with col2_main:
    agent_x_input=st.number_input("Agent Start X",0.0,float(world_w_main),float(default_agent_x_main),1.0, key="agent_x")
    agent_y_input=st.number_input("Agent Start Y",0.0,float(world_h_main),float(default_agent_y_main),1.0, key="agent_y")

st.session_state.oil_center_canvas=(oil_x_input,oil_y_input)
st.session_state.agent_start_canvas=(agent_x_input,agent_y_input)

fig_preview_main,ax_preview_main=plt.subplots();ax_preview_main.set_xlim(0,world_w_main);ax_preview_main.set_ylim(0,world_h_main);ax_preview_main.set_title("Initial Setup Preview");ax_preview_main.set_aspect('equal',adjustable='box');ax_preview_main.grid(True)
if st.session_state.oil_center_canvas:ax_preview_main.plot(st.session_state.oil_center_canvas[0],st.session_state.oil_center_canvas[1],'ko',ms=10,label='Oil Center')
if st.session_state.agent_start_canvas:ax_preview_main.plot(st.session_state.agent_start_canvas[0],st.session_state.agent_start_canvas[1],'bo',ms=10,label='Agent Start')
if st.session_state.oil_center_canvas or st.session_state.agent_start_canvas:ax_preview_main.legend()
st.pyplot(fig_preview_main);plt.close(fig_preview_main)

# --- Run Simulation Button and Logic ---
if st.button("🚀 Run Simulation"):
    if not agent or agent.actor is None: st.error("Agent not loaded or actor is None. Check sidebar status.")
    elif not app_config: st.error("App configuration not loaded.") # Should not happen due to st.stop()
    else:
        custom_oil_world_coords=st.session_state.oil_center_canvas
        custom_agent_world_coords=st.session_state.agent_start_canvas
        
        # Create a deep copy of world_config for this specific run
        run_world_config = app_config.world.model_copy(deep=True)
        run_world_config.randomize_agent_initial_location=False
        run_world_config.randomize_oil_cluster=False
        run_world_config.agent_initial_location=PositionPydantic(x=custom_agent_world_coords[0],y=custom_agent_world_coords[1])
        run_world_config.initial_oil_center=PositionPydantic(x=custom_oil_world_coords[0],y=custom_oil_world_coords[1])
        run_world_config.max_steps = max_eval_steps # Use sidebar value for this run
        run_world_config.num_oil_points=num_oil_points_sidebar
        run_world_config.initial_oil_std_dev=oil_std_dev_sidebar
        run_world_config.oil_cluster_std_dev_range=(oil_std_dev_sidebar,oil_std_dev_sidebar) # Make it a range
        run_world_config.agent_speed=agent_speed_sidebar
        
        run_vis_config = app_config.visualization.model_copy(deep=True)
        run_vis_config.output_format=output_format_sidebar
        
        os.makedirs(run_vis_config.save_dir,exist_ok=True)
        world_instance=World(world_config=run_world_config)
        
        st.info(f"Running: Oil@{custom_oil_world_coords}, Agent@{custom_agent_world_coords}")
        progress_bar_ui=st.progress(0); status_text_ui=st.empty(); episode_frames_paths=[]
        
        current_state=world_instance.reset(custom_agent_loc=custom_agent_world_coords,custom_oil_center=custom_oil_world_coords)
        actor_hidden_state_sim=None
        if agent.config.use_rnn and agent.actor: actor_hidden_state_sim=agent.actor.get_initial_hidden_state(1,agent.device)
        
        reset_viz_trajectories()
        output_filename_base_sim=f"oil_spill_sim_{int(time.time())}"
        temp_frame_dir_sim=os.path.join(run_vis_config.save_dir,f"{output_filename_base_sim}_frames")
        os.makedirs(temp_frame_dir_sim,exist_ok=True)
        
        fig_sim_run,ax_sim_run=plt.subplots(figsize=run_vis_config.figure_size)
        
        for sim_step_idx in range(max_eval_steps):
            status_text_ui.text(f"Simulating step {sim_step_idx+1}/{max_eval_steps}...")
            action_from_agent,actor_hidden_state_sim = agent.select_action(current_state,actor_hidden_state_sim,evaluate=not stochastic_policy)
            current_state=world_instance.step(action_from_agent)
            
            visualize_world_mpl(world_instance,run_vis_config,fig_sim_run,ax_sim_run)
            frame_path_sim=os.path.join(temp_frame_dir_sim,f"frame_{sim_step_idx:04d}.png")
            fig_sim_run.savefig(frame_path_sim)
            episode_frames_paths.append(frame_path_sim)
            
            progress_bar_ui.progress((sim_step_idx+1)/max_eval_steps)
            if world_instance.done: status_text_ui.text(f"Simulation ended at step {sim_step_idx+1} (Done). Metric: {world_instance.performance_metric:.3f}"); break
        if not world_instance.done: status_text_ui.text(f"Simulation ended at step {max_eval_steps} (Max steps reached). Metric: {world_instance.performance_metric:.3f}")
        plt.close(fig_sim_run)
        
        if episode_frames_paths:
            media_placeholder_ui=st.empty(); media_placeholder_ui.info(f"Generating {run_vis_config.output_format.upper()}...")
            output_media_filename_sim=f"{output_filename_base_sim}.{run_vis_config.output_format}"
            output_media_path_sim=os.path.join(run_vis_config.save_dir,output_media_filename_sim)
            images_for_media_sim=[imageio.imread(f) for f in episode_frames_paths]
            
            if run_vis_config.output_format=='gif':
                imageio.mimsave(output_media_path_sim,images_for_media_sim,fps=run_vis_config.video_fps)
                media_placeholder_ui.image(output_media_path_sim)
            elif run_vis_config.output_format=='mp4':
                try: 
                    imageio.mimsave(output_media_path_sim,images_for_media_sim,fps=run_vis_config.video_fps,format='FFMPEG',output_params=['-vcodec','libx264'])
                    media_placeholder_ui.video(output_media_path_sim)
                except Exception as e_mp4_sim:
                    st.warning(f"MP4 creation fail: {e_mp4_sim}. Try GIF or install ffmpeg."); gfp_sim=output_media_path_sim.replace(".mp4",".gif")
                    imageio.mimsave(gfp_sim,images_for_media_sim,fps=run_vis_config.video_fps); media_placeholder_ui.image(gfp_sim); st.info("Fell back to GIF.")
            
            if run_vis_config.delete_png_frames:
                for frame_file_to_del in episode_frames_paths:
                    try:os.remove(frame_file_to_del)
                    except:pass
                try:os.rmdir(temp_frame_dir_sim)
                except:pass
            st.success(f"Simulation complete. Final Performance Metric: {world_instance.performance_metric:.3f}")
        else: st.error("No frames generated for the video.")

st.sidebar.markdown("---")
st.sidebar.markdown("Oil Spill Mapping Demo")