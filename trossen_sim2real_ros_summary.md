# trossen_sim2real_ros: A ROS 2 Package for Sim-to-Real Transfer of RL Policies on the Trossen WidowX AI Arm

## Abstract

`trossen_sim2real_ros` is a ROS 2 (Robot Operating System 2) package designed to bridge the sim-to-real gap for reinforcement learning (RL) policies trained in simulation and deployed on the Trossen WidowX AI (wxai) robotic arm. The package integrates a TorchScript-serialized RL policy with real-time robot kinematics via Pinocchio, and provides multiple control interfaces compatible with mock components, NVIDIA Isaac Sim, and physical hardware. It exposes an interactive 6-DOF target pose marker in RViz for real-time task specification and implements Differential Inverse Kinematics (Diff-IK) using a Damped Least Squares (DLS) solver to convert policy outputs into executable joint commands.

---

## 1. Overview

### 1.1 Package Identification

| Field | Value |
|---|---|
| **Package Name** | `trossen_sim2real_ros` |
| **Build System** | `ament_python` (ROS 2 Python package) |
| **Maintainer** | Hayato Kojima (`hayato.kojima80@gmail.com`) |
| **Target Robot** | Trossen WidowX AI arm (wxai), 6-DOF |
| **ROS 2 Distribution** | Jazzy / Humble |
| **Control Frequency** | 30 Hz (configurable) |

### 1.2 Purpose

The package solves the sim-to-real transfer problem for reach-type manipulation tasks. A neural network policy—exported from an Isaac Lab / RSL-RL training pipeline as a TorchScript `.pt` file—is loaded at runtime and queried at 30 Hz. The policy outputs 6-dimensional end-effector velocity commands (3 translational, 3 rotational), which are converted to joint angle increments via Differential IK and dispatched to the robot hardware through the `ros2_control` stack.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
[RViz / Target Param]    ──► target_pose (7D)
[/joint_states topic]    ──► joint_pos_rel (8D), joint_vel (8D)
[last_actions]           ──► actions_prev (6D)
                                │
                                ▼
                       ┌─────────────────┐
                       │  RL Policy      │  (TorchScript, 30 Hz)
                       │  Input: 29D/36D │
                       │  Output: 6D act │
                       └────────┬────────┘
                                │ actions
                                ▼
                     ┌──────────────────────┐
                     │  State Machine / EMA │  (Switch task only)
                     └──────────┬───────────┘
                                │ filtered actions
                                ▼
                     ┌──────────────────────┐
                     │  Differential IK     │  (Pinocchio)
                     │  Damped Least Squares│
                     └──────────┬───────────┘
                                │ dq_arm ∈ ℝ^6
                                ▼
               ┌────────────────────────────────────┐
               │ ros2_control / JointTrajectory      │
               └────────────────────────────────────┘
                                │
                                ▼
               [Mock Components | Isaac Sim | Real HW]
```

### 2.2 Observation Space: Reach Task (29-Dimensional)

The reach policy receives a 29-dimensional flat observation vector assembled at each control step:

| Segment | Dimension | Description |
|---|---|---|
| `joint_pos_rel` | 8 | Joint positions relative to default pose (6 arm + 2 gripper carriage joints) |
| `joint_vel` | 8 | Joint velocities (rad/s) |
| `pose_command` | 7 | Target end-effector pose: `[x, y, z, qw, qx, qy, qz]` in base frame |
| `last_actions` | 6 | Action history (raw unclipped previous policy outputs) |
| **Total** | **29** | |

### 2.3 Observation Space: Switch Task (36-Dimensional)

The control panel switch policy adds the current end-effector pose to the observation vector, resulting in 36 dimensions:

| Segment | Dimension | Description |
|---|---|---|
| `joint_pos_rel` | 8 | Joint positions relative to default pose (6 arm + 2 gripper carriage joints) |
| `joint_vel` | 8 | Joint velocities (rad/s) |
| `ee_pos` | 7 | Current end-effector pose: `[x, y, z, qw, qx, qy, qz]` in base frame |
| `last_actions` | 6 | Action history (raw unclipped previous policy outputs) |
| `pose_command` | 7 | Target switch pose: `[x, y, z, qw, qx, qy, qz]` in base frame |
| **Total** | **36** | |

The joint default pose used for relative encoding in both tasks is:

| Joint | Default (rad) |
|---|---|
| `joint_0` | 0.0 |
| `joint_1` | π/3 ≈ 1.047 |
| `joint_2` | π/6 ≈ 0.524 |
| `joint_3` | π/5 ≈ 0.628 |
| `joint_4` | 0.0 |
| `joint_5` | 0.0 |
| `left_carriage_joint` | 0.0 |
| `right_carriage_joint` | 0.0 |

### 2.4 Action Space and Differential IK

The policy outputs a 6-dimensional action vector. These are scaled to desired end-effector velocity increments:
- Translational (x, y, z): `Δp = action[:3] × 0.005` (m/step) (or `0.008` / `0.01` under custom tuning)
- Rotational (rx, ry, rz): `Δω = action[3:] × 0.005` (rad/step) (or `0.01` under custom tuning)

**Damped Least Squares IK:**

The arm Jacobian `J_arm ∈ ℝ^{6×6}` is computed by Pinocchio for the `ee_gripper_link` frame (`LOCAL_WORLD_ALIGNED`). Joint velocity increments are solved as:

$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} \Delta \xi$$

where $\lambda = 0.01$ is the damping coefficient.

---

## 3. Package Structure

```
trossen_sim2real_ros/
├── README.md                         # English documentation
├── README_ja.md                      # Japanese documentation
├── package.xml                       # ROS 2 package manifest
├── setup.py                          # Python package setup / entry points
├── setup.cfg                         # ament test configuration
├── resource/
│   └── trossen_sim2real_ros          # ROS 2 ament resource marker
├── models/                           # (reserved for future model storage)
├── test/                             # ament linting test stubs
└── trossen_sim2real_ros/
    ├── __init__.py
    ├── sim2real_node.py              # Primary ROS 2 node (JointTrajectoryController)
    ├── sim2real_node_switch.py       # Switch policy node (adds state machine and EMA filter)
    ├── sim2real_node_trapezoidal.py  # Alternate node (JointGroupPositionController)
    └── sim2real_direct.py            # Standalone script (direct Trossen SDK)
```

---

## 4. Node Descriptions

### 4.1 `sim2real_node` (Primary Node)

**Entry point:** `trossen_sim2real_ros.sim2real_node:main`  
**ROS 2 Node name:** `sim2real_node`  
**Controller interface:** `JointTrajectoryController`  
**Output topic:** `/arm_controller/joint_trajectory`

This is the primary, zero-configuration node. It publishes `JointTrajectory` messages, each containing a single `JointTrajectoryPoint` with incremented joint positions. The `time_from_start` is set to `1.5 / control_freq` seconds (i.e., 50 ms at 30 Hz) to prevent the trajectory controller from reaching the spline endpoint and causing motion artifacts ("shaking") before the next command arrives.

**Key ROS 2 interfaces:**

| Interface | Type | Direction | Topic/Service |
|---|---|---|---|
| Robot description | `std_msgs/String` | Subscribe | `/robot_description` |
| Joint states | `sensor_msgs/JointState` | Subscribe | `/joint_states` |
| Target pose (external) | `geometry_msgs/PoseStamped` | Subscribe | `/target_pose`, `/goal_pose` |
| Interactive marker | `visualization_msgs/*` | Bidirectional | `target_pose_marker` server |
| Joint trajectory | `trajectory_msgs/JointTrajectory` | Publish | `/arm_controller/joint_trajectory` |
| TF transform | TF2 | Subscribe | `base_link` → `ee_gripper_link` |

**ROS 2 Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `policy_path` | (hardcoded path) | Absolute path to exported TorchScript policy `.pt` file |
| `control_freq` | `30.0` | Control loop frequency in Hz |
| `use_sim_time` | `false` | Use simulation clock (set `true` for Isaac Sim) |

**Lifecycle:**
1. Load TorchScript policy via `torch.jit.load()`; set to `eval()` mode.
2. Subscribe to `/robot_description` (TRANSIENT_LOCAL QoS) to build a Pinocchio kinematic model from URDF.
3. On first control step, compute current FK pose and initialize the interactive marker at that position.
4. At 30 Hz: assemble observation → policy inference → DLS IK → publish trajectory.
5. Every 1 second, log tracking error via TF2 lookup of `ee_gripper_link` in `base_link`.

### 4.2 `sim2real_node_switch` (Switch Task Node)

**Entry point:** `trossen_sim2real_ros.sim2real_node_switch:main`  
**ROS 2 Node name:** `sim2real_node_switch`  
**Controller interface:** `JointTrajectoryController`  
**Output topic:** `/arm_controller/joint_trajectory`

This node is specifically designed for the control panel breaker-switch flipping task. It loads a 36-dimensional observation policy and implements a state machine and low-pass filter (Exponential Moving Average) to ensure stable and safe execution.

**Key ROS 2 interfaces:**

| Interface | Type | Direction | Topic/Service |
|---|---|---|---|
| Robot description | `std_msgs/String` | Subscribe | `/robot_description` |
| Joint states | `sensor_msgs/JointState` | Subscribe | `/joint_states` |
| Target pose (external) | `geometry_msgs/PoseStamped` | Subscribe | `/target_pose` |
| Joint trajectory | `trajectory_msgs/JointTrajectory` | Publish | `/arm_controller/joint_trajectory` |

**Target Pose Parameters (CLI Initialization):**
To run the switch node with a custom target directly from the command line (bypassing interactive markers), the node exposes the following parameters:
- `use_target_param` (bool, default: `true`): If `true`, the target pose is initialized using parameters instead of defaulting to the current end-effector position.
- `target_x`, `target_y`, `target_z` (float): Target coordinates.
- `target_qw`, `target_qx`, `target_qy`, `target_qz` (float): Target orientation quaternion.

**State Machine Logic:**
To prevent the robot from slamming into the control panel repeatedly, a state machine is implemented:
1. `APPROACHING`: The robot uses the RL policy to reach the target switch position.
2. `PUSHING`: Triggered when the end-effector is within 2 cm of the target (`dist < 0.02`). It records the start time and allows the policy to push the switch up for exactly 1.0 second.
3. `DONE`: After the pushing phase, the joint commands are frozen (`actions` set to zero) to hold the pose and prevent further motion.

*Note:* Publishing a new target pose resets the state machine back to `APPROACHING`.

**Action Filtering / Exponential Moving Average (EMA):**
To damp high-frequency jerks and smooth physical joint acceleration without modifying the RL policy's action scale, a low-pass filter is applied to the joint positions $q_{\text{new}}$:
$$q_{\text{filtered}} = (1 - \alpha) \cdot q_{\text{filtered}} + \alpha \cdot q_{\text{new}}$$
where $\alpha = 0.15$ is the default smoothing factor. Lowering $\alpha$ yields smoother but slower physical motion.

### 4.3 `sim2real_node_trapezoidal` (Alternate Node)

**Entry point:** `trossen_sim2real_ros.sim2real_node_trapezoidal:main`  
**ROS 2 Node name:** `sim2real_node_trapezoidal`  
**Controller interface:** `JointGroupPositionController`  
**Output topic:** `/forward_position_controller/commands`

This node bypasses the trajectory generator and publishes raw `Float64MultiArray` position commands directly to a `JointGroupPositionController`. This leverages the controller's own trapezoidal velocity/acceleration profile for smoother, higher-bandwidth motion. External pose topic subscriptions are intentionally disabled in this node to prevent overriding the interactive marker during testing.

The controller switch procedure (via `/controller_manager/switch_controller` service) must be performed before launching this node, as it requires `forward_position_controller` to be active and `arm_controller` to be deactivated.

**Differences from `sim2real_node`:**

| Aspect | `sim2real_node` | `sim2real_node_trapezoidal` |
|---|---|---|
| Output message type | `JointTrajectory` | `Float64MultiArray` |
| Output topic | `/arm_controller/joint_trajectory` | `/forward_position_controller/commands` |
| Velocity scaling (translation) | 0.005 m/step | 0.010 m/step |
| Velocity scaling (rotation) | 0.010 rad/step | 0.020 rad/step |
| External pose topic | Enabled | Disabled |
| Controller type | `JointTrajectoryController` | `JointGroupPositionController` |
| Setup complexity | Zero-config | Requires controller swap |

### 4.4 `sim2real_direct` (Standalone Script)

**File:** `sim2real_direct.py`  
**Not a ROS 2 node** — runs as a standalone Python script.

This script bypasses ROS 2 entirely and communicates with the physical robot using the `trossen_arm` Python SDK directly. It connects to the arm via IP (192.168.1.2), configures it to position mode, and runs the same observation → policy → DLS IK control loop in a `while True` busy-wait loop. This is intended for low-latency deployment without ROS overhead.

---

## 5. Kinematics Engine: Pinocchio

All three nodes use `pinocchio` (a C++ rigid body dynamics library with Python bindings) for online kinematics computation. The URDF is either:
- Received from the `/robot_description` topic (ROS 2 nodes), or
- Generated at startup via `xacro` subprocess call (standalone script).

**Pinocchio operations performed per step:**
1. `pin.forwardKinematics(model, data, q)` — compute joint placements
2. `pin.updateFramePlacements(model, data)` — update frame positions
3. `pin.computeJointJacobians(model, data, q)` — compute full Jacobian
4. `pin.getFrameJacobian(..., LOCAL_WORLD_ALIGNED)` — extract 6×n frame Jacobian

The end-effector frame is `ee_gripper_link`. Only the 6 arm joint columns of the Jacobian (`J_arm`) are used for IK; gripper carriage joints are excluded from actuation.

---

## 6. Interactive Marker Interface

A 6-DOF interactive marker (via `interactive_markers` ROS 2 package) is created in the `base_link` frame, accessible from RViz. It supports:
- Translation along X, Y, Z axes
- Rotation around X, Y, Z axes

On startup, the marker is initialized to the current FK end-effector pose (not a hardcoded default), ensuring continuity. The marker's pose feedback directly updates the `target_pose` field (7D: position + quaternion) used in the observation vector.

---

## 7. Operating Modes

The package supports three hardware backends without code changes, controlled via launch arguments:

| Mode | `ros2_control_hardware_type` | Additional Arg | Use Case |
|---|---|---|---|
| Mock Components | `mock_components` | — | RViz visualization, policy debugging |
| Isaac Sim | `isaac` | `use_sim_time:=true` | Full simulation with physics |
| Real Hardware | `real` | `ip_address:=<IP>` | Physical robot deployment |

### Isaac Sim Clock Synchronization

When using Isaac Sim, it acts as the master simulation clock. All nodes must be launched with `use_sim_time:=true` to avoid TF timestamp errors (`TF_OLD_DATA`). The QoS profile for `/robot_description` uses `TRANSIENT_LOCAL` durability to ensure the node receives the URDF even if it subscribes after the publisher has already sent it.

---

## 8. Software Dependencies

| Dependency | Category | Role |
|---|---|---|
| `rclpy` | ROS 2 | Python client library |
| `sensor_msgs` | ROS 2 | `JointState` message |
| `trajectory_msgs` | ROS 2 | `JointTrajectory` message |
| `geometry_msgs` | ROS 2 | `PoseStamped` message |
| `visualization_msgs` | ROS 2 | Interactive marker messages |
| `interactive_markers` | ROS 2 | Interactive marker server |
| `tf2_ros` | ROS 2 | Transform lookup for error logging |
| `torch` | Python | TorchScript policy inference |
| `pinocchio` | Python | Forward kinematics, Jacobian computation |
| `numpy` | Python | Numerical computation |
| `trossen_arm` | Python SDK | Direct hardware communication (standalone only) |
| `ros2_control` | ROS 2 | Hardware abstraction layer |
| `ros2_controllers` | ROS 2 | Trajectory and position controllers |
| `topic_based_ros2_control` | ROS 2 | Isaac Sim topic bridge |

---

## 9. Executor Configuration

Both ROS 2 nodes use a `MultiThreadedExecutor` with 4 threads, allowing the control loop timer, subscription callbacks (joint states, URDF, marker feedback), and the TF listener to execute concurrently without blocking each other.

---

## 10. Tracking Error Logging

Every second (at 30 Hz, after 30 control steps), the node performs a TF2 lookup from `base_link` to `ee_gripper_link` and logs:
- Current target pose (position + quaternion)
- Actual end-effector pose (from TF)
- Positional error (Δx, Δy, Δz)
- Quaternion error (component-wise difference, with double-cover correction)

The quaternion double-cover issue (q and -q represent the same rotation) is handled by flipping the sign of the actual quaternion when its dot product with the target quaternion is negative.

---

## 11. Design Decisions and Engineering Notes

1. **Action smoothing via `last_actions`:** The previous action is fed back as part of the observation, providing the policy with temporal context. This reduces jerk and improves tracking stability.

2. **Trajectory duration factor (1.5×):** The `time_from_start` of each trajectory point is set to 1.5 control periods rather than 1.0. This prevents the `JointTrajectoryController` from reaching the trajectory endpoint before the next command arrives, which would cause the robot to stop and restart, producing visible oscillation.

3. **Action clipping:** Policy outputs are clipped to `[-1, 1]` before applying the velocity scale, providing a hard safety bound on per-step joint displacement.

4. **Pinocchio model initialization deferral:** The kinematic model is built lazily upon receipt of the `/robot_description` message rather than at node startup. This ensures robustness across different launch orderings and hardware modes.

5. **Marker initialization from FK:** The interactive marker is initialized to the current FK pose of the end-effector (not a static default), preventing a sudden jump when the control loop first activates.

6. **Damping coefficient (λ = 0.01):** The DLS damping prevents joint velocity blowup near kinematic singularities, at the cost of a slight accuracy reduction in those configurations.

---

## 12. Repository Information

- **Repository:** `https://github.com/Hayato-80/trossen_sim2real_ros`
- **Dependency Repository:** `https://github.com/TrossenRobotics/trossen_arm_ros` (jazzy branch)
- **RL Training Framework:** Isaac Lab / RSL-RL (`wxai_reach` task)
- **Policy Format:** TorchScript (`.pt`), exported via `torch.jit.save()`
