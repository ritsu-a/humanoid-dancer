import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


from legged_gym.legged_gym.motions.motion_lib_g1 import MotionLibG1
from legged_gym.legged_gym.utils import torch_utils
from rsl_rl.modules.actor_critic import ActorCritic
import tyro


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def main():
     # get config file name from command line

    config_file = "/home/pengyang/codebase/playground/humanoid-dancer/deploy/deploy_mujoco/configs/g1_23.yaml"
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

   
    # load policy
    policy_cfg = {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}
    actor_critic = ActorCritic( 138,
                                138,
                                23,
                                **policy_cfg).to("cpu")

    loaded_dict = torch.load(policy_path)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.eval()
    policy = actor_critic.act_inference


    # load motion 
    _motion_lib = MotionLibG1(
            motion_file='/home/pengyang/data/motion/g1/LAFAN1/dance1_subject2_v2.pkl', device='cpu', 
            masterfoot_conifg=None, fix_height=False,
            multi_thread=False, mjcf_file='/home/pengyang/codebase/playground/humanoid-dancer/legged_gym/resources/robots/g1/xml/g1_29dof_anneal_23dof_fitmotionONLY.xml', 
            sim_timestep=simulation_dt,
        )


    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                root_vel = d.qvel[:3]


                qj = (qj - default_angles) 
                dqj = dqj 
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega 
                root_vel = root_vel 

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                import ipdb 
                ipdb.set_trace()

                obs[:3] = omega * ang_vel_scale
                obs[3:6] = gravity_orientation
                obs[6 : 6 + num_actions] = qj * dof_pos_scale
                obs[6 + num_actions : 6 + 2 * num_actions] = dqj * dof_vel_scale
                obs[6 + 2 * num_actions : 6 + 3 * num_actions] = action
                obs[6 + 3 * num_actions : 9 + 3 * num_actions] = root_vel * lin_vel_scale

                obs[9 + 3 * num_actions : 11 + 3 * num_actions] = np.array([sin_phase, cos_phase])


                ### task obs

                    # # heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
                    # # heading_rot = torch_utils.calc_heading_quat(root_rot)
                    
                    # # diff_global_body_rot = torch_utils.quat_mul(ref_body_rot[:, 0], torch_utils.quat_conjugate(root_rot))
                    # # diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot.view(-1, 4))
                motion_res = _motion_lib
                
                heading_inv_rot = torch_utils.calc_heading_quat_inv(torch.from_numpy(quat)).numpy()
                heading_rot = torch_utils.calc_heading_quat(torch.from_numpy(quat)).numpy()




                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                                    
                                    # # task obs (6 + 3 + 3 + 23 * 2 = 58)
                                    # torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1),
                                    # diff_local_root_vel.view(B, -1) * self.obs_scales.lin_vel,
                                    # diff_local_root_ang_vel.view(B, -1) * self.obs_scales.ang_vel,
                                    # dof_diff.view(B, -1) * self.obs_scales.dof_pos,
                                    # dof_vel_diff.view(B, -1) * self.obs_scales.dof_vel,

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":

    main()
   