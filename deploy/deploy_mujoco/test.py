## reference: https://github.com/LeCAR-Lab/ASAP/issues/23

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
import joblib
import traceback # For detailed error reporting

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.motions.motion_lib_g1 import MotionLibG1
from rsl_rl.modules.actor_critic import ActorCritic


# Define base directory relative to the script location
ASAP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# --- Helper Functions ---

# Helper for quaternion multiplication (wxyz format)
def multiply_quaternions_wxyz(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    return np.array([w, x, y, z]) / norm if norm > 1e-8 else np.array([1., 0., 0., 0.])

# Helper for rotating vector by quaternion (wxyz format)
def rotate_vector_by_quat_wxyz(q, v):
     q_norm = np.sqrt(np.sum(q**2))
     if q_norm < 1e-8: q = np.array([1.0, 0.0, 0.0, 0.0])
     else: q = q / q_norm
     q_v = np.concatenate(([0.], v)); q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
     q_rotated_v = multiply_quaternions_wxyz(multiply_quaternions_wxyz(q, q_v), q_conj)
     return q_rotated_v[1:]

# Standard Gravity Projection Calculation
def get_gravity_orientation_standard(quat_wxyz: np.ndarray) -> np.ndarray:
    """Calculates the projected gravity vector using standard quaternion rotation."""
    gravity_vec_world = np.array([0., 0., -1.])
    quat_inv_wxyz = np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]])
    projected_gravity = rotate_vector_by_quat_wxyz(quat_inv_wxyz, gravity_vec_world)
    return projected_gravity.astype(np.float32)

def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
               target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """Calculates PD control torque."""
    target_q = np.asarray(target_q); q = np.asarray(q); kp = np.asarray(kp)
    target_dq = np.asarray(target_dq); dq = np.asarray(dq); kd = np.asarray(kd)
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_obs(d: mujoco.MjData, motion_dict: dict, motion_frame: int, device: str = "cpu") -> dict:
    """Extracts required observations from MuJoCo data."""

    ### parse reference motion
    motion_root_trans_offset = motion_dict["root_trans_offset"].astype(np.float32)[motion_frame]
    motion_root_rot = motion_dict["root_rot"].astype(np.float32)[motion_frame]
    motion_dof = motion_dict["dof"].astype(np.float32)[motion_frame]
    

    obs = {}

    vel = d.qvel
    pos = d.qpos

    obs["base_ang_vel"] = torch.tensor(d.qvel[3:6].astype(np.float32), device=device).numpy()

    projected_gravity_np = get_gravity_orientation_standard(d.qpos[3:7])
    obs["projected_gravity"] = torch.tensor(projected_gravity_np, device=device).numpy()

    obs["dof_pos"] = torch.tensor(d.qpos[7:].astype(np.float32), device=device).numpy()

    obs["dof_vel"] = torch.tensor(d.qvel[6:].astype(np.float32), device=device).numpy()

    obs["base_lin_vel"] = torch.tensor(d.qvel[:3].astype(np.float32), device=device).numpy()

    ### task
    obs["diff_local_body_rot_flat"] = 0.0
    obs["diff_local_body_rot_vel"] = 0.0
    obs["diff_local_body_rot_ang_vel"] = 0.0
    obs["dof_diff"] = 0.0
    obs["dof_vel_diff"] = 0.0

    import ipdb 
    ipdb.set_trace()
    
    return obs

#    obs_buf = np.concatenate((  obs_dict['base_ang_vel'] * 1.0 , 
#                                                 obs_dict['projected_gravity'], 
#                                                 obs_dict['dof_pos'] * 1.0,
#                                                 obs_dict['dof_vel'] * 1.0,
#                                                 actions,
#                                                 obs_dict['base_lin_vel']* 1.0,
#                                                 0.0,
#                                                 0.0,

#                                                 ### task obs
#                                                 obs_dict['diff_local_body_rot_flat']* 1.0,
#                                                 obs_dict['diff_local_root_vel']* 1.0,
#                                                 obs_dict['dif_local_root_ang_vel']* 1.0,
#                                                 obs_dict['dof_diff']* 1.0,
#                                                 obs_dict['dof_vel_diff']* 1.0,
#                     ), axis=-1)

def get_policy(policy_path):
    policy_cfg = {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}
    actor_critic = ActorCritic( 138,
                                138,
                                23,
                                **policy_cfg).to("cpu")

    loaded_dict = torch.load(policy_path)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.eval()
    policy = actor_critic.act_inference
    return policy
    
# --- Main Execution ---

def main():
    """Main function to load configs, setup simulation, and run."""

    # ================================================================
    # !!!!! DEBUG MODE SELECTOR !!!!!
    # 0: Normal Execution (Policy Controlled, Target=Action+Default)
    # 1: Log Policy I/O & Individual Observations (Detailed Log)
    # 2: Zero-Action Test (No Policy, Target=Default Pose)
    # 3: Reference Tracking Test (No Policy, Target=Reference Pose)

    # !!!!! PD GAIN SCALING !!!!!
    KP_SCALE = 1 # Default 1.0. Try reducing (e.g., 0.1) or tuning.
    KD_SCALE = 1 # Default 1.0. Try reducing (e.g., 0.1) or tuning.
    # ================================================================
    print(f"***** Running with Kp Scale {KP_SCALE}, Kd Scale {KD_SCALE} *****")

    # --- 1. Load Configurations ---
    config_path1 = f"/home/pengyang/codebase/playground/humanoid-dancer/deploy/deploy_mujoco/configs/g1_23.yaml"
    try:
        with open(config_path1, "r") as f1:
            config = yaml.safe_load(f1)
    except Exception as e: print(f"Config Error: {e}"); return

    # --- 2. Extract Parameters ---
    try:
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        motion_file = '/home/pengyang/data/motion/g1/LAFAN1/dance1_subject2_v2.pkl'
        simulation_duration = float(config["simulation_duration"])
        control_decimation = int(config["control_decimation"])
        # Apply Gain Scaling
        kps = np.array(config["kps"], dtype=np.float32) * KP_SCALE
        kds = np.array(config["kds"], dtype=np.float32) * KD_SCALE

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        num_actions = int(config["num_actions"])

        action_scale = float(config["action_scale"])

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
       
        print(f"Action Scale: {action_scale}")
        if not all(len(arr) == num_actions for arr in [kps, kds, default_angles]):
             raise ValueError("Dimension mismatch: kps/kds/default_angles vs num_actions")
    except Exception as e: print(f"Parameter Error: {e}"); return

    # --- 3. Load Motion Data ---
    try:
        motion_data = joblib.load(motion_file)
        motion_key = list(motion_data.keys())[0]
        motion_dof = motion_data[motion_key]["dof"].astype(np.float32)


        motion_dict = motion_data[motion_key]
        num_motion_frames = motion_dof.shape[0]
        if motion_dof.shape[1] != num_actions: raise ValueError(f"Motion DOF dim mismatch")


        _motion_lib = MotionLibG1(
            motion_file='/home/pengyang/data/motion/g1/LAFAN1/dance1_subject2_v2.pkl', device='cpu', 
            masterfoot_conifg=None, fix_height=False,
            multi_thread=False, mjcf_file='/home/pengyang/codebase/playground/humanoid-dancer/legged_gym/resources/robots/g1/xml/g1_29dof_anneal_23dof_fitmotionONLY.xml', 
            sim_timestep=config["simulation_dt"],
        )
        print(f"Loaded motion: {motion_key} ({num_motion_frames} frames)")



    except Exception as e: print(f"Motion File Error: {e}"); return

    # --- 4. Setup MuJoCo Simulation ---
    try:
        m = mujoco.MjModel.from_xml_path(xml_path); d = mujoco.MjData(m)
        m.opt.timestep = config["simulation_dt"]
        control_step_time = m.opt.timestep * control_decimation
        ref_motion_length = num_motion_frames * control_step_time
        print(f"MuJoCo Timestep (dt): {m.opt.timestep:.5f}")
        print(f"Control Frequency: {1.0/control_step_time:.1f} Hz (Decimation: {control_decimation})")
    except Exception as e: print(f"MuJoCo Model Error: {e}"); return

    # --- 5. Load  Policy Model ---
    policy = None
    try:
        policy = get_policy(policy_path=policy_path)
    except Exception as e: print(f"Policy Model Error: {e}"); return

    # --- 6. Initialize Simulation Variables ---
    target_dof_pos = default_angles.copy()
    motion_index = 0; motion_times = 0.0
    prev_action = np.zeros(num_actions, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)

    obs = np.zeros(138, dtype=np.float32)

    hardcoded_effort_limits = [ 88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 88.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0 ]
    if len(hardcoded_effort_limits) == num_actions:
         tau_limit = np.array(hardcoded_effort_limits, dtype=np.double); print("Using hardcoded torque limits.")
    else: print(f"Error: Torque limit length mismatch. Using fallback."); tau_limit = 200. * np.ones(num_actions, dtype=np.double)

    ref_motion_phase = 0

    # --- 7. Run Simulation Loop ---
    sim_start_time = time.time(); sim_step_counter = 0; viewer_instance = None
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer_instance:
            while viewer_instance.is_running() and time.time() - sim_start_time < simulation_duration:
                step_start_time = time.time()
                mujoco.mj_step(m, d)

                # Set initial pose from motion[0]
                if sim_step_counter == 0:
                    # d.qpos[7:] = motion_dof[0]
                    d.qpos[7:] = default_angles
                    d.qvel[6:] = 0.0
                    print("Initial pose set to first motion frame.")

                # --- Control Logic (Runs at lower frequency) ---
                if sim_step_counter % control_decimation == 0:

                    motion_times = (motion_times + control_step_time) % ref_motion_length
                    current_phase = motion_times / ref_motion_length if ref_motion_length > 0 else 0.0

                    print(f"motion time: {motion_times}")
                    # -- Calculate Observations & History (Only if policy is used) --
                    import ipdb;ipdb.set_trace()
                    obs_dict = get_obs(d, motion_dict, motion_index, device="cpu") # Gets ABSOLUTE dof_pos
                    dof_pos = obs_dict["dof_pos"] * 1.0
                    dof_vel = obs_dict["dof_vel"] * 0.05
                    base_ang_vel = obs_dict["base_ang_vel"]* 0.25
                    projected_gravity = obs_dict["projected_gravity"] * 1.0


                    motion_index += 1 

                 
                            # # # self.obs_buf = torch.cat((  
                            # # #         # self obs
                            # # #         self.base_ang_vel  * self.obs_scales.ang_vel,
                            # # #         self.projected_gravity,
                            # # #         # self.commands[:, :3] * self.commands_scale * 0, # do not use commands
                            # # #         (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            # # #         self.dof_vel * self.obs_scales.dof_vel,
                            # # #         self.actions,
                            # # #         self.base_lin_vel * self.obs_scales.lin_vel,
                            # # #         sin_phase,
                            # # #         cos_phase,
                                    
                            # # #         # task obs (6 + 3 + 3 + 23 * 2 = 58)
                            # # #         torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1),
                            # # #         diff_local_root_vel.view(B, -1) * self.obs_scales.lin_vel,
                            # # #         diff_local_root_ang_vel.view(B, -1) * self.obs_scales.ang_vel,
                            # # #         dof_diff.view(B, -1) * self.obs_scales.dof_pos,
                            # # #         dof_vel_diff.view(B, -1) * self.obs_scales.dof_vel,
                            # # #         ),dim=-1)

                    obs_buf = np.concatenate((  obs_dict['base_ang_vel'] * 1.0 , 
                                                obs_dict['projected_gravity'], 
                                                obs_dict['dof_pos'] * 1.0,
                                                obs_dict['dof_vel'] * 1.0,
                                                actions,
                                                obs_dict['base_lin_vel']* 1.0,
                                                0.0,
                                                0.0,

                                                ### task obs
                                                obs_dict['diff_local_body_rot_flat']* 1.0,
                                                obs_dict['diff_local_root_vel']* 1.0,
                                                obs_dict['dif_local_root_ang_vel']* 1.0,
                                                obs_dict['dof_diff']* 1.0,
                                                obs_dict['dof_vel_diff']* 1.0,
                    ), axis=-1)
                      

                    obs_array = torch.from_numpy(obs_buf).cpu().numpy()
                    actions = policy(obs_array)

                    target_dof_pos = actions * 0.25 + default_angles


                # --- Apply PD Control (Using Kp*Scale, Kd*Scale) ---
                tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                                 np.zeros_like(kds), d.qvel[6:], kds) # Use original kds (already scaled)

                tau = np.clip(tau, -tau_limit, tau_limit)
                d.ctrl[:] = tau

                # --- Sync Viewer and Sleep ---
                viewer_instance.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start_time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                sim_step_counter += 1

    except KeyboardInterrupt: print("\nSimulation interrupted by user.")
    except Exception as e: print(f"\nAn error occurred during simulation:\n{traceback.format_exc()}")
    finally:
        if viewer_instance and viewer_instance.is_running(): viewer_instance.close()
        elapsed_time = time.time() - sim_start_time
        print(f"Simulation finished after {elapsed_time:.2f} seconds ({sim_step_counter} steps).")

if __name__ == "__main__":
    main()