from dataclasses import dataclass, field
from typing import Optional, List, Dict

from legged_gym.envs.base import legged_robot_config, humanoid_robot_config

@dataclass
class InitState( legged_robot_config.InitState ):
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        'left_hip_pitch_joint': -0.1,
        'left_hip_roll_joint': 0.,
        'left_hip_yaw_joint': 0.,
        'left_knee_joint': 0.3,
        'left_ankle_pitch_joint': -0.2,
        'left_ankle_roll_joint': 0.,
        'right_hip_pitch_joint': -0.1,
        'right_hip_roll_joint': 0.,
        'right_hip_yaw_joint': 0.,
        'right_knee_joint': 0.3,
        'right_ankle_pitch_joint': -0.2,
        'right_ankle_roll_joint': 0.,
        'waist_yaw_joint': 0.,
        'waist_roll_joint': 0.,
        'waist_pitch_joint' : 0.,
        'left_shoulder_pitch_joint': 0.,
        'left_shoulder_roll_joint': 0.,
        'left_shoulder_yaw_joint': 0.,
        'left_elbow_joint': 0.,
        'right_shoulder_pitch_joint': 0.,
        'right_shoulder_roll_joint': 0.,
        'right_shoulder_yaw_joint': 0.,
        'right_elbow_joint': 0.,
    })

@dataclass
class Env(legged_robot_config.Env):
    # 3 + 3 + 3 + 23 * 3 + 2 = 80
    num_observations: int = 80
    # 3 + 3 + 3 + 23 * 3 + 2 + 3(base lin vel) = 83
    num_privileged_obs: Optional[int] = 83
    num_actions: int = 23
    
@dataclass
class DomainRand(legged_robot_config.DomainRand):
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.5, 1.25])
    randomize_base_mass: bool = True
    added_mass_range: List[float] = field(default_factory=lambda: [-1.0, 3.0])
    push_robots: bool = True
    push_interval_s: float = 5.0
    max_push_vel_xy: float = 1.5

@dataclass
class Control(legged_robot_config.Control):
    # PD Drive parameters:
    control_type: str = 'P'
    stiffness: Dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 100,
        'hip_roll': 100,
        'hip_pitch': 100,
        'knee': 200,
        'ankle_pitch': 20,
        'ankle_roll': 20,
        'waist_yaw': 400,
        'waist_roll': 400,
        'waist_pitch': 400,
        'shoulder_pitch': 90,
        'shoulder_roll': 60,
        'shoulder_yaw': 20,
        'elbow': 60,
    })
    damping: Dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 2.5,
        'hip_roll': 2.5,
        'hip_pitch': 2.5,
        'knee': 5.0,
        'ankle_pitch': 0.2,
        'ankle_roll': 0.1,
        'waist_yaw': 5.0,
        'waist_roll': 5.0,
        'waist_pitch': 5.0,
        'shoulder_pitch': 2.0,
        'shoulder_roll': 1.0,
        'shoulder_yaw': 0.4,
        'elbow': 1.0,
    })
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale: float = 0.25
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation: int = 4

@dataclass
class Asset(humanoid_robot_config.Asset):
    file: str = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29dof_anneal_23dof.urdf'
    name: str = "g1"
    foot_name: str = "ankle"
    penalize_contacts_on: List[str] = field(default_factory=lambda: ["pelvis", "shoulder", "hip"])
    terminate_after_contacts_on: List[str] = field(default_factory=lambda: ["pelvis", "shoulder", "hip"])
    self_collisions: int = 1 # 1 to disable, 0 to enable...bitwise filter
    flip_visual_attachments: bool = False
    
    hip_roll_name: Optional[str] = "hip_roll"
    hip_yaw_name: Optional[str] = "hip_yaw"
    hip_pitch_name: Optional[str] = "hip_pitch"
    
    shoulder_roll_name: Optional[str] = "shoulder_roll"
    shoulder_yaw_name: Optional[str] = "shoulder_yaw"
    shoulder_pitch_name: Optional[str] = "shoulder_pitch"
    
    elbow_name: Optional[str] = "elbow"
    knee_name: Optional[str] = "knee"
    
    ankle_pitch_name: Optional[str] = "ankle_pitch"
    ankle_roll_name: Optional[str] = "ankle_roll"
    
    waist_yaw_name: Optional[str] = "waist_yaw"
    waist_roll_name: Optional[str] = "waist_roll"
    waist_pitch_name: Optional[str] = "waist_pitch"

@dataclass
class Rewards(humanoid_robot_config.Rewards):
    only_positive_rewards: bool = False # important for rl training
    soft_dof_pos_limit: float = 0.9
    base_height_target: float = 1.05
    fixed_upper_body_positive_sigma: float = 1. 
    scales: Dict[str, float] = field(default_factory=lambda: {
        'tracking_lin_vel': 1.0,
        'tracking_ang_vel': 0.5,
        'lin_vel_z': -2.0,
        'ang_vel_xy': -0.05,
        'orientation': -1.0,
        'base_height': -10.0,
        'dof_acc': -2.5e-7,
        'feet_air_time': 0.0,
        'collision': -1.0,
        'action_rate': -0.01,
        'torques': 0.0,
        'dof_pos_limits': -5.0,
        'alive': 0.15,
        'hip_pos': -1.0,
        'contact_no_vel': -0.2,
        'feet_swing_height': -20.0,
        'contact': 0.18,
        'fixed_upper_body_positive': 1,
    })

@dataclass
class G1Cfg:
    env: Env = field(default_factory=Env)
    terrain: legged_robot_config.Terrain = field(default_factory=legged_robot_config.Terrain)
    commands: legged_robot_config.Commands = field(default_factory=legged_robot_config.Commands)
    init_state: InitState = field(default_factory=InitState)
    control: Control = field(default_factory=Control)
    asset: Asset = field(default_factory=Asset)
    domain_rand: DomainRand = field(default_factory=DomainRand)
    rewards: Rewards = field(default_factory=Rewards)
    normalization: legged_robot_config.Normalization = field(default_factory=legged_robot_config.Normalization)
    noise: legged_robot_config.Noise = field(default_factory=legged_robot_config.Noise)
    viewer: legged_robot_config.Viewer = field(default_factory=legged_robot_config.Viewer)
    sim: legged_robot_config.Sim = field(default_factory=legged_robot_config.Sim)
    
    seed: int = field(init=False)
    name = 'g1'

@dataclass
class Policy( legged_robot_config.Policy ):
    init_noise_std: float = 0.8

@dataclass
class Algorithm( legged_robot_config.Algorithm ):
    entropy_coef: float = 0.01
    
@dataclass
class Runner( legged_robot_config.Runner ):
    policy_class_name: str = "ActorCritic"
    max_iterations: int = 10000
    run_name: str = ''
    experiment_name: str = 'g1'

@dataclass
class G1PPOCfg:    
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    policy: Policy = field(default_factory=Policy)
    algorithm: Algorithm = field(default_factory=Algorithm)
    runner: Runner = field(default_factory=Runner)