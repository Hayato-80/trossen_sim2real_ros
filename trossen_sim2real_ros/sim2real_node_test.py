#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl

# TF2 for tracking error
import tf2_ros

# Standard ROS 2 interactive marker server
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

import torch
import pinocchio as pin


class Sim2RealNode(Node):
    def __init__(self):
        super().__init__('sim2real_node')

        # Parameters
        self.declare_parameter('policy_path', '/home/hayato/rl_ws/trossen_ai_isaac/logs/rsl_rl/wxai_reach/2026-07-14_02-13-52/exported/policy.pt')
        self.declare_parameter('control_freq', 30.0)

        policy_path = self.get_parameter('policy_path').value
        self.control_freq = self.get_parameter('control_freq').value

        self.get_logger().info(f"Loading PyTorch policy from {policy_path}...")
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            self.get_logger().info("Policy loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise e

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

        # State storage
        self.current_joint_pos = {name: 0.0 for name in self.all_joint_names}
        self.current_joint_vel = {name: 0.0 for name in self.all_joint_names}
        self.last_actions = np.zeros(len(self.arm_joint_names), dtype=np.float32)
        self.joint_states_received = False
        self.marker_initialized = False

        # --- Timed Waypoint Sequence ---
        # Waypoints: [x, y, z, qw, qx, qy, qz], reach_threshold_meters
        self.waypoint_sequence = [
            # ([0.5453, -0.000, 0.262, 1.0, 0.0, 0.0, 0.0], 0.0127855),  # Step 1: Approach switch
            ([0.5453, -0.000, 0.262, 1.0, 0.0, 0.0, 0.0], 0.012),
            ([0.545,   0.000, 0.35, 1.0, 0.0, 0.0, 0.0], 0.02),  # Step 2: Push up
            ([0.38,   -0.000, 0.25,  1.0, 0.0, 0.0, 0.0], None),  # Step 3: Retract (final)
        ]
        self.waypoint_index = 0
        self.sequence_started = False
        self.sequence_timer = None

        # Target pose (default: a safe reachable pose)
        self.target_pose = [0.5453, -0.000, 0.262, 1.0, 0.0, 0.0, 0.0]  # pos (3) + quat (w, x, y, z)

        # CLI Target Parameters (set these via --ros-args -p to specify target at launch)
        self.declare_parameter('use_target_param', False)
        self.declare_parameter('target_x', 0.5453)
        self.declare_parameter('target_y', -0.000)
        self.declare_parameter('target_z', 0.262)
        self.declare_parameter('target_qw', 1.0)
        self.declare_parameter('target_qx', 0.0)
        self.declare_parameter('target_qy', 0.0)
        self.declare_parameter('target_qz', 0.0)

        # Pinocchio Model
        self.model = None
        self.data = None
        self.ee_frame_id = None
        self.ee_frame = 'ee_gripper_link'

        # Interactive Marker Server
        self.setup_interactive_marker()

        # TF Listener for error logging
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.base_frame = 'base_link'
        self.log_counter = 0

        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.robot_desc_sub = self.create_subscription(
            String,
            '/robot_description',
            self.robot_desc_callback,
            qos_profile)

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_pose_callback,
            10)
            
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.target_pose_callback,
            10)

        # Publisher
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)

        # Timer
        timer_period = 1.0 / self.control_freq
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info("Sim2Real Node is initialized and running.")

    def robot_desc_callback(self, msg):
        if self.model is None:
            self.get_logger().info("Received URDF. Building Pinocchio model...")
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as f:
                    f.write(msg.data)
                    tmp_urdf_path = f.name
                
                self.model = pin.buildModelFromUrdf(tmp_urdf_path)
                self.data = self.model.createData()
                
                if self.model.existFrame(self.ee_frame):
                    self.ee_frame_id = self.model.getFrameId(self.ee_frame)
                    self.get_logger().info(f"Pinocchio model built. EE frame ID: {self.ee_frame_id}")
                else:
                    self.get_logger().error(f"Frame {self.ee_frame} not found in URDF!")
            except Exception as e:
                self.get_logger().error(f"Failed to build Pinocchio model: {e}")

    def setup_interactive_marker(self):
        self.server = InteractiveMarkerServer(self, 'target_pose_marker')
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "target_goal"
        int_marker.description = "Drag to set Target Pose"
        int_marker.scale = 0.2

        int_marker.pose.position.x = float(self.target_pose[0])
        int_marker.pose.position.y = float(self.target_pose[1])
        int_marker.pose.position.z = float(self.target_pose[2])
        int_marker.pose.orientation.w = float(self.target_pose[3])
        int_marker.pose.orientation.x = float(self.target_pose[4])
        int_marker.pose.orientation.y = float(self.target_pose[5])
        int_marker.pose.orientation.z = float(self.target_pose[6])

        # Add 6DOF controls
        control_x = InteractiveMarkerControl()
        control_x.orientation.w = 1.0
        control_x.orientation.x = 1.0
        control_x.orientation.y = 0.0
        control_x.orientation.z = 0.0
        control_x.name = "rotate_x"
        control_x.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_x)

        control_move_x = InteractiveMarkerControl()
        control_move_x.orientation.w = 1.0
        control_move_x.orientation.x = 1.0
        control_move_x.orientation.y = 0.0
        control_move_x.orientation.z = 0.0
        control_move_x.name = "move_x"
        control_move_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_move_x)

        control_z = InteractiveMarkerControl()
        control_z.orientation.w = 1.0
        control_z.orientation.x = 0.0
        control_z.orientation.y = 1.0
        control_z.orientation.z = 0.0
        control_z.name = "rotate_z"
        control_z.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_z)

        control_move_z = InteractiveMarkerControl()
        control_move_z.orientation.w = 1.0
        control_move_z.orientation.x = 0.0
        control_move_z.orientation.y = 1.0
        control_move_z.orientation.z = 0.0
        control_move_z.name = "move_z"
        control_move_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_move_z)

        control_y = InteractiveMarkerControl()
        control_y.orientation.w = 1.0
        control_y.orientation.x = 0.0
        control_y.orientation.y = 0.0
        control_y.orientation.z = 1.0
        control_y.name = "rotate_y"
        control_y.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_y)

        control_move_y = InteractiveMarkerControl()
        control_move_y.orientation.w = 1.0
        control_move_y.orientation.x = 0.0
        control_move_y.orientation.y = 0.0
        control_move_y.orientation.z = 1.0
        control_move_y.name = "move_y"
        control_move_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_move_y)

        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        self.target_pose = [
            feedback.pose.position.x,
            feedback.pose.position.y,
            feedback.pose.position.z,
            feedback.pose.orientation.w,
            feedback.pose.orientation.x,
            feedback.pose.orientation.y,
            feedback.pose.orientation.z
        ]
        self.server.setPose("target_goal", feedback.pose)
        self.server.applyChanges()

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.all_joint_names:
                self.current_joint_pos[name] = msg.position[i]
                if len(msg.velocity) > i:
                    self.current_joint_vel[name] = msg.velocity[i]
        self.joint_states_received = True

    def target_pose_callback(self, msg):
        self.target_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ]

    def log_tracking_error(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time())
            
            ee_x = t.transform.translation.x
            ee_y = t.transform.translation.y
            ee_z = t.transform.translation.z
            ee_qw = t.transform.rotation.w
            ee_qx = t.transform.rotation.x
            ee_qy = t.transform.rotation.y
            ee_qz = t.transform.rotation.z

            tg_x, tg_y, tg_z = self.target_pose[0], self.target_pose[1], self.target_pose[2]
            tg_qw, tg_qx, tg_qy, tg_qz = self.target_pose[3], self.target_pose[4], self.target_pose[5], self.target_pose[6]

            if (tg_qw*ee_qw + tg_qx*ee_qx + tg_qy*ee_qy + tg_qz*ee_qz) < 0:
                ee_qw, ee_qx, ee_qy, ee_qz = -ee_qw, -ee_qx, -ee_qy, -ee_qz

            err_x, err_y, err_z = tg_x - ee_x, tg_y - ee_y, tg_z - ee_z
            err_qw, err_qx, err_qy, err_qz = tg_qw - ee_qw, tg_qx - ee_qx, tg_qy - ee_qy, tg_qz - ee_qz

            self.get_logger().info(
                f"--- Tracking Error [{self.ee_frame}] ---\n"
                f"Target: Pos [{tg_x:.3f}, {tg_y:.3f}, {tg_z:.3f}], Quat [{tg_qw:.3f}, {tg_qx:.3f}, {tg_qy:.3f}, {tg_qz:.3f}]\n"
                f"Actual: Pos [{ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f}], Quat [{ee_qw:.3f}, {ee_qx:.3f}, {ee_qy:.3f}, {ee_qz:.3f}]\n"
                f"Error:  Pos [{err_x:.3f}, {err_y:.3f}, {err_z:.3f}], Quat [{err_qw:.3f}, {err_qx:.3f}, {err_qy:.3f}, {err_qz:.3f}]"
            )
        except Exception as e:
            self.get_logger().debug(f"Could not compute tracking error: {e}")

    def _execute_waypoint(self, index):
        """Set the target to waypoint[index]. Advancement happens via distance check in control_loop."""
        if index >= len(self.waypoint_sequence):
            self.get_logger().info("[Waypoint Sequence] All waypoints complete!")
            return

        pose, threshold = self.waypoint_sequence[index]
        self.target_pose = list(pose)
        self.waypoint_index = index

        # Update interactive marker to match new target
        from geometry_msgs.msg import Pose
        pose_msg = Pose()
        pose_msg.position.x = float(pose[0])
        pose_msg.position.y = float(pose[1])
        pose_msg.position.z = float(pose[2])
        pose_msg.orientation.w = float(pose[3])
        pose_msg.orientation.x = float(pose[4])
        pose_msg.orientation.y = float(pose[5])
        pose_msg.orientation.z = float(pose[6])
        self.server.setPose("target_goal", pose_msg)
        self.server.applyChanges()

        self.get_logger().info(
            f"[Waypoint {index + 1}/{len(self.waypoint_sequence)}] "
            f"Target → Pos [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]"
            + (f" — advance when dist < {threshold:.3f}m" if threshold else " — FINAL waypoint")
        )

    def control_loop(self):
        if not self.joint_states_received or self.model is None or self.ee_frame_id is None:
            # Wait for joint states and model to be ready
            return

        if not self.marker_initialized:
            # Compute current end-effector pose using Pinocchio forward kinematics
            q = np.zeros(self.model.nq)
            for name in self.all_joint_names:
                if self.model.existJointName(name):
                    idx = self.model.joints[self.model.getJointId(name)].idx_q
                    q[idx] = self.current_joint_pos[name]
            
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            oMf = self.data.oMf[self.ee_frame_id]
            pos = oMf.translation
            quat = pin.Quaternion(oMf.rotation)
            
            use_target_param = self.get_parameter('use_target_param').value
            if use_target_param:
                tx = self.get_parameter('target_x').value
                ty = self.get_parameter('target_y').value
                tz = self.get_parameter('target_z').value
                tqw = self.get_parameter('target_qw').value
                tqx = self.get_parameter('target_qx').value
                tqy = self.get_parameter('target_qy').value
                tqz = self.get_parameter('target_qz').value
                self.target_pose = [tx, ty, tz, tqw, tqx, tqy, tqz]
                self.get_logger().info(f"Initialized target pose from params: {self.target_pose}")
            else:
                self.target_pose = [
                    pos[0], pos[1], pos[2],
                    quat.w, quat.x, quat.y, quat.z
                ]
                self.get_logger().info(f"Initialized target pose to current EE: {self.target_pose}")
            
            # Sync interactive marker server pose
            from geometry_msgs.msg import Pose
            pose_msg = Pose()
            pose_msg.position.x = float(self.target_pose[0])
            pose_msg.position.y = float(self.target_pose[1])
            pose_msg.position.z = float(self.target_pose[2])
            pose_msg.orientation.w = float(self.target_pose[3])
            pose_msg.orientation.x = float(self.target_pose[4])
            pose_msg.orientation.y = float(self.target_pose[5])
            pose_msg.orientation.z = float(self.target_pose[6])
            self.server.setPose("target_goal", pose_msg)
            self.server.applyChanges()
            
            self.marker_initialized = True
            self.get_logger().info(f"Initialized target pose to current end-effector: {self.target_pose}")

            # Start the waypoint sequence once the marker is ready
            self._execute_waypoint(0)

        # 1. joint_pos_rel (8)
        joint_pos_rel = []
        for name in self.all_joint_names:
            joint_pos_rel.append(self.current_joint_pos[name] - self.default_joint_pos[name])
        
        # 2. joint_vel_rel (8)
        joint_vel_rel = []
        for name in self.all_joint_names:
            joint_vel_rel.append(self.current_joint_vel[name])
        
        # 3. pose_command (7)
        pose_command = self.target_pose
        
        # 4. actions (6)
        actions_obs = self.last_actions.tolist()

        # Combine to single flat observation (total 29)
        obs_list = joint_pos_rel + joint_vel_rel + pose_command + actions_obs
        obs_tensor = torch.tensor([obs_list], dtype=torch.float32)

        # Policy Inference
        with torch.no_grad():
            try:
                action_tensor = self.policy(obs_tensor)
                actions = action_tensor[0].cpu().numpy()
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")
                return

        # Save raw actions for the next observation
        self.last_actions = actions.copy()

        # --- DISTANCE-BASED WAYPOINT ADVANCEMENT ---
        if self.marker_initialized and self.waypoint_index < len(self.waypoint_sequence):
            _, threshold = self.waypoint_sequence[self.waypoint_index]
            if threshold is not None:  # None means it's the final waypoint — no advancement
                oMf_now = self.data.oMf[self.ee_frame_id]
                ee_pos_now = oMf_now.translation
                tgt = self.target_pose[:3]
                dist = np.linalg.norm(ee_pos_now - np.array(tgt))
                if dist < threshold:
                    next_idx = self.waypoint_index + 1
                    self.get_logger().info(
                        f"[Waypoint {self.waypoint_index + 1}] Reached! (dist={dist:.3f}m) "
                        f"→ Advancing to waypoint {next_idx + 1}"
                    )
                    self._execute_waypoint(next_idx)
        
        # --- DIFFERENTIAL IK (Pinocchio) ---
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
        
        # J_arm has shape (6, num_arm_joints)
        J_arm = J[:, arm_idx_q]
        
        # Damped Least Squares
        # Scale actions to 'Cobot Speed'
        # Translation (x, y, z): 0.01 * 30Hz = 0.3 m/s = 30cm/s
        # Rotation (rx, ry, rz): 0.02 * 30Hz = 0.6 rad/s = 34.4 deg/s
        delta_pose = np.zeros(6)
        delta_pose[:3] = actions[:3] * 0.005
        delta_pose[3:] = actions[3:] * 0.005
        lambda_val = 0.01
        lambda_matrix = (lambda_val**2) * np.eye(6)
        
        J_T = J_arm.T
        inv_term = np.linalg.inv(J_arm @ J_T + lambda_matrix)
        dq_arm = J_T @ inv_term @ delta_pose
        
        # Publish Trajectory
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = []
        
        for i, name in enumerate(self.arm_joint_names):
            target_pos = self.current_joint_pos[name] + dq_arm[i]
            point.positions.append(float(target_pos))
            
        point.time_from_start.sec = 0
        # Multiply by 1.5 to prevent the trajectory controller from reaching the end of the 
        # spline and stopping before the next 30Hz frame arrives (this causes the shaking!)
        point.time_from_start.nanosec = int((1.5 / self.control_freq) * 1e9)

        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)
        
        # Log error once per second
        self.log_counter += 1
        if self.log_counter >= self.control_freq:
            self.log_counter = 0
            self.log_tracking_error()


def main(args=None):
    rclpy.init(args=args)
    node = Sim2RealNode()
    
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
