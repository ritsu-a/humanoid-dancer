from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from rsl_rl.modules.actor_critic import ActorCritic
from legged_gym.motions.motion_lib_g1 import MotionLibG1
from legged_gym.utils import torch_utils

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch


from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

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

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = get_policy(policy_path=config.policy_path)



        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = np.array(config.default_angles, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0


        # Load reference motion
        # TODO: write paths in config
        self._motion_lib =MotionLibG1(
            motion_file='/home/pengyang/data/motion/g1/LAFAN1/dance1_subject2_v2.pkl', device='cpu', 
            masterfoot_conifg=None, fix_height=False,
            multi_thread=False, mjcf_file='/home/pengyang/codebase/playground/humanoid-dancer/legged_gym/resources/robots/g1/xml/g1_29dof_anneal_23dof_fitmotionONLY.xml', 
            sim_timestep=self.config.simulation_dt,
        )

        skeleton_file = '/home/pengyang/codebase/playground/humanoid-dancer/legged_gym/resources/robots/g1/xml/g1_29dof_anneal_23dof_fitmotionONLY.xml'
        sk_tree = SkeletonTree.from_mjcf(skeleton_file)
        self._motion_lib.load_motions(
                skeleton_trees=[sk_tree], gender_betas=[torch.zeros(17)], 
                limb_weights=[np.zeros(10)], 
                random_sample=False, start_idx=0
        )


        ###
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.dof_idx + self.config.waist_idx
        kps = self.config.kps + self.config.waist_kps
        kds = self.config.kds + self.config.waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.dof_idx)):
                motor_idx = self.config.dof_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.waist_idx)):
                motor_idx = self.config.waist_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def compute_observation(self, quat, ang_vel, gravity_orientation, motion_res, device='cpu'):
       
        B = 1


        ref_body_pos = motion_res["rg_pos"] 
        ref_body_pos_extend = motion_res["rg_pos_t"]
        ref_body_vel_subset = motion_res["body_vel"] # [num_envs, num_markers, 3]
        ref_body_vel = ref_body_vel_subset
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        ref_body_rot = motion_res["rb_rot"] # [num_envs, num_markers, 4]
        ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel = motion_res["body_ang_vel"] # [num_envs, num_markers, 3]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_dof_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_dof_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        
        
        ref_root_vel = ref_body_vel[:, 0] # [num_envs, 3]
        ref_root_ang_vel = ref_body_ang_vel[:, 0]

        root_rot = torch.from_numpy(np.array(quat).astype(np.float32)) 
        root_rot = torch.cat((root_rot[1:], root_rot[:1]), dim=0)[None, :] # Convert to xyzw format

        root_ang_vel = torch.from_numpy(ang_vel)
        
        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        

        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot[:, 0], torch_utils.quat_conjugate(root_rot))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot.view(-1, 4))
    
        ref_local_root_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), ref_root_vel.view(B, 3))

        diff_global_root_ang_vel = ref_root_ang_vel.view(B, 1, 3) - root_ang_vel.view(B, 1, 3)
        diff_local_root_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_ang_vel.view(-1, 3))

        dof_pos = torch.tensor(self.qj.astype(np.float32), device=device)
        dof_vel = torch.tensor(self.dqj.astype(np.float32), device=device)

        dof_diff = ref_dof_pos.view(B, 1, -1) - dof_pos.view(B, 1, -1)
        dof_vel_diff = ref_dof_vel.view(B, 1, -1) - dof_vel.view(B, 1, -1)

        obs = {}

        obs["base_ang_vel"] = root_ang_vel
        obs["projected_gravity"] = gravity_orientation
        obs["dof_pos"] = dof_pos.numpy()
        obs["dof_vel"] = dof_vel.numpy()


        ### task
        obs["diff_local_body_rot_flat"] = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(-1).numpy()
        obs["ref_local_root_vel"] = ref_local_root_vel.view(-1).numpy()
        obs["diff_local_root_ang_vel"] = diff_local_root_ang_vel.view(-1).numpy()
        obs["dof_diff"] = dof_diff.view(-1).numpy()
        obs["dof_vel_diff"] = dof_vel_diff.view(-1).numpy()
        return obs

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.dof_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.dof_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.dof_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)

        ### TODO specialized case for lafan_dance12
        if (self.counter * self.config.control_dt + 4.0) > self._motion_lib.get_motion_length()[0].numpy():
            self.counter = 0
        
        motion_times = (self.counter * self.config.control_dt + 4.0) % self._motion_lib.get_motion_length()[0].numpy()
        
        current_phase = motion_times / self._motion_lib.get_motion_length().numpy() if self._motion_lib.get_motion_length().numpy() > 0 else 0.0

        phase = current_phase 

        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        motion_res = self._motion_lib.get_motion_state(
                        motion_ids=[0],
                        motion_times=torch.tensor([motion_times], dtype=torch.float32),
                        offset=None
                    )

        obs_dict = self.compute_observation(quat, ang_vel, gravity_orientation, motion_res, device='cpu')

        

        num_actions = self.config.num_actions

        obs_dict = self.compute_observation(quat, ang_vel, gravity_orientation, motion_res, device='cpu')
        ### TODO: write the scales into config
        obs_buf = np.concatenate((  obs_dict['base_ang_vel'].view(-1) * 0.25 , 
                                    obs_dict['projected_gravity'], 
                                    (obs_dict['dof_pos'] - self.config.default_angles)* 1.0,
                                    obs_dict['dof_vel'] * 0.05,
                                    self.action,
                                    np.zeros(3)* 0.0,
                                    sin_phase,
                                    cos_phase,

                                    ### task obs
                                    obs_dict['diff_local_body_rot_flat'],
                                    obs_dict['ref_local_root_vel']* 2.0,
                                    obs_dict['diff_local_root_ang_vel']* 0.25,
                                    obs_dict['dof_diff']* 1.0,
                                    obs_dict['dof_vel_diff']* 0.05,
        ), axis=-1).astype(np.float32)

        # Get the action from the policy network
        obs_array = torch.from_numpy(obs_buf).cpu()
        self.action = self.policy(obs_array).detach().numpy()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale * 0.6

        # Build low cmd
        for i in range(len(self.config.dof_idx)):
            motor_idx = self.config.dof_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.waist_idx)):
            motor_idx = self.config.waist_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("--config", type=str, help="config file name in the configs folder", default="g1_23.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/../deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
