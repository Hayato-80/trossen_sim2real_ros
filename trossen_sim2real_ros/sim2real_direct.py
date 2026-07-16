import time
import math
import subprocess
import tempfile
import numpy as np
import torch
import pinocchio as pin
import trossen_arm

def get_urdf():
    print("Generating URDF via xacro...")
    xacro_cmd = [
        'xacro',
        '/home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_description/urdf/wxai.urdf.xacro',
        'use_fake_hardware:=false'
    ]
    result = subprocess.run(xacro_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate URDF from xacro:\n{result.stderr}")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as f:
        f.write(result.stdout)
        return f.name

class Sim2RealDirect:
    def __init__(self):
        # Parameters
        self.policy_path = '/home/hayato/rl_ws/trossen_ai_isaac/logs/rsl_rl/wxai_reach/2026-07-14_02-13-52/exported/policy.pt'
        self.control_freq = 30.0

        print(f"Loading PyTorch policy from {self.policy_path}...")
        self.policy = torch.jit.load(self.policy_path)
        self.policy.eval()
        print("Policy loaded successfully.")

        # Trossen Driver Initialization
        print("Initializing Trossen Arm Driver...")
        self.driver = trossen_arm.TrossenArmDriver()
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            "192.168.1.2",
            False
        )
        self.driver.set_arm_modes(trossen_arm.Mode.position)
        print("Arm is configured and in position mode.")

        # Joints configuration
        self.arm_joint_names = [
            "joint_0", "joint_1", "joint_2",
            "joint_3", "joint_4", "joint_5"
        ]
        self.all_joint_names = self.arm_joint_names + ["left_carriage_joint", "right_carriage_joint"]

        self.default_joint_pos = {
            "joint_0": 0.0,
            "joint_1": math.pi / 3,
            "joint_2": math.pi / 6,
            "joint_3": math.pi / 5,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "left_carriage_joint": 0.0,
            "right_carriage_joint": 0.0,
        }

        self.last_actions = np.zeros(len(self.arm_joint_names), dtype=np.float32)

        # Pinocchio Model setup
        print("Building Pinocchio model...")
        urdf_path = get_urdf()
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame = 'ee_gripper_link'
        self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        print(f"Pinocchio model built. EE frame ID: {self.ee_frame_id}")

        # The real arm current positions
        self.current_joint_pos = {name: 0.0 for name in self.all_joint_names}
        self.current_joint_vel = {name: 0.0 for name in self.all_joint_names}

        # Safe target pose for Reach task
        self.target_pose = [0.45, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0]

    def update_joint_states_from_robot(self):
        # Driver returns list of floats matching the internal order
        jp = self.driver.get_joint_positions()
        # Assumes the first 6 elements correspond to arm_joint_names
        for i in range(6):
            self.current_joint_pos[self.arm_joint_names[i]] = jp[i]
        
        # We don't have velocities from basic API, so assume 0
        for name in self.all_joint_names:
            self.current_joint_vel[name] = 0.0

    def loop(self):
        print("Starting control loop. Press Ctrl+C to stop.")
        dt = 1.0 / self.control_freq

        while True:
            start_time = time.time()
            
            self.update_joint_states_from_robot()

            # 1. joint_pos_rel
            joint_pos_rel = []
            for name in self.all_joint_names:
                joint_pos_rel.append(self.current_joint_pos[name] - self.default_joint_pos[name])
            
            # 2. joint_vel_rel
            joint_vel_rel = []
            for name in self.all_joint_names:
                joint_vel_rel.append(self.current_joint_vel[name])

            # 3. pose_command
            pose_command = self.target_pose

            # 4. actions
            actions_obs = self.last_actions.tolist()

            obs_list = joint_pos_rel + joint_vel_rel + pose_command + actions_obs
            obs_tensor = torch.tensor([obs_list], dtype=torch.float32)

            with torch.no_grad():
                action_tensor = self.policy(obs_tensor)
                actions = action_tensor[0].cpu().numpy()
            
            actions = np.clip(actions, -1.0, 1.0)
            self.last_actions = actions

            # --- DIFFERENTIAL IK ---
            q = np.zeros(self.model.nq)
            for name in self.all_joint_names:
                if self.model.existJointName(name):
                    idx = self.model.joints[self.model.getJointId(name)].idx_q
                    q[idx] = self.current_joint_pos[name]
            
            arm_idx_q = []
            for name in self.arm_joint_names:
                if self.model.existJointName(name):
                    arm_idx_q.append(self.model.joints[self.model.getJointId(name)].idx_q)

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
            J_arm = J[:, arm_idx_q]
            
            # Scale actions to 'Cobot Speed'
            delta_pose = np.zeros(6)
            delta_pose[:3] = actions[:3] * 0.01
            delta_pose[3:] = actions[3:] * 0.02
            
            lambda_matrix = (0.01**2) * np.eye(6)
            J_T = J_arm.T
            inv_term = np.linalg.inv(J_arm @ J_T + lambda_matrix)
            dq_arm = J_T @ inv_term @ delta_pose

            # Target positions
            current_jp = self.driver.get_joint_positions()
            for i in range(6):
                current_jp[i] += float(dq_arm[i])

            # Send to robot using Joint Interpolation
            self.driver.set_joint_positions(current_jp, trossen_arm.InterpolationSpace.joint)

            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    node = Sim2RealDirect()
    try:
        node.loop()
    except KeyboardInterrupt:
        print("\nStopping...")
