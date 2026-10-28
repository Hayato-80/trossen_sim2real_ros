# Reach-Task RL Training and Sim-to-Real Deployment for Breaker-Switch Flipping

## Abstract

This document describes the complete pipeline for training a reinforcement-learning (RL) reach policy in NVIDIA Isaac Lab and deploying it on a physical Trossen WidowX AI (wxai) 6-DOF robotic arm to perform a sequential breaker-switch flipping task. The pipeline consists of two stages: (1) training a goal-conditioned RL policy in simulation using the `trossen_ai_isaac` package, and (2) deploying it with the `trossen_sim2real_ros` package via the `sim2real_node_test` ROS 2 node, which automates a three-phase motion sequence controlled entirely by the learned policy.

---

## 1. Training: The Reach Task in Isaac Lab

### 1.1 Task Overview

The reach task trains the robot's end-effector (EE) to track randomly sampled Cartesian pose targets. The policy learns to follow commanded poses within the robot's reachable workspace. No object is manipulated during training; the goal is for the arm to become a reliable pose-tracking controller that generalises to novel targets at deployment.

Training is performed in NVIDIA Isaac Lab using the RSL-RL on-policy PPO framework with 4,096 parallel environments for massively parallel data collection.

### 1.2 Simulation Environment

The environment is configured by the class hierarchy:

```
ReachEnvCfg  (base: reach_env_cfg.py)
    └── WXAIReachEnvCfg  (joint_pos_env_cfg.py)
            └── WXAIReachEnvCfg (ik_rel_env_cfg.py)  <- used for training
```

**Scene:**
- Ground plane at -1.05 m (beneath the table surface)
- Seattle Lab Table at (0.55, 0, 0), rotated 90 degrees; robot sits on this table
- Trossen WidowX AI arm in High-PD mode (WXAI_HIGH_PD_CFG) - gravity disabled via high gains to enable stiff Cartesian-space IK control
- Dome light (intensity 2500, grey)

**Initial Joint Configuration (default pose, applied at every episode reset):**

| Joint | Default (rad) | Degrees |
|---|---|---|
| joint_0 | 0.0 | 0 |
| joint_1 | pi/3 = 1.047 | 60 |
| joint_2 | pi/6 = 0.524 | 30 |
| joint_3 | pi/5 = 0.628 | 36 |
| joint_4 | 0.0 | 0 |
| joint_5 | 0.0 | 0 |
| left_carriage_joint | 0.0 | 0 (gripper closed) |

At each episode reset, all joint positions are randomised by scaling this default by a factor uniformly drawn from [0.5, 1.5], providing diverse initial conditions for robust generalisation.

**Episode Length:** 12 seconds (720 physics steps at 60 Hz; 360 policy steps at 30 Hz)

### 1.3 Action Space

Training uses Differential Inverse Kinematics (Diff-IK) with relative pose control:

```python
DifferentialInverseKinematicsActionCfg(
    joint_names = ["joint_[0-5]"],
    body_name   = "ee_gripper_link",
    controller  = DifferentialIKControllerCfg(
        command_type      = "pose",
        use_relative_mode = True,
        ik_method         = "dls",
    ),
    scale = 0.01,
)
```

The policy outputs a 6-dimensional action a in [-1, 1]^6:
- a[0:3] -> delta position (x, y, z) in metres, scaled by 0.01 per step -> max 0.01 m/step = 0.3 m/s at 30 Hz
- a[3:6] -> delta orientation (roll, pitch, yaw), scaled by 0.01 per step -> max 0.01 rad/step = 0.3 rad/s at 30 Hz

Isaac Lab's internal DLS IK solver converts relative EE deltas to joint position commands with damping coefficient lambda = 0.01.

The decimation is 2, meaning the policy runs at 30 Hz while the physics engine runs at 60 Hz.

### 1.4 Observation Space (29-Dimensional)

The policy observation vector is assembled at every 30 Hz step as a flat 29-dimensional tensor:

| Index | Term | Dim | Description |
|---|---|---|---|
| 0-7 | joint_pos_rel | 8 | Joint positions relative to default pose; noise +/- 0.01 rad |
| 8-15 | joint_vel_rel | 8 | Joint velocities (rad/s); noise +/- 0.01 rad/s |
| 16-22 | pose_command | 7 | Target EE pose [x, y, z, qw, qx, qy, qz] in base frame |
| 23-28 | actions | 6 | Previous policy output (action history) |

All 8 joints are included: 6 arm revolute joints plus left_carriage_joint and right_carriage_joint (gripper rails, both held at 0).

### 1.5 Command Sampling (Target Pose Distribution)

Target poses are drawn uniformly and resampled every 4 seconds during training:

| Axis | Range |
|---|---|
| x (forward) | [0.35, 0.55] m |
| y (lateral) | [-0.20, 0.20] m |
| z (vertical) | [0.10, 0.45] m |
| roll | [-pi/3, pi/3] rad |
| pitch | [0, pi/3] rad |
| yaw | [-pi/3, pi/3] rad |

### 1.6 Reward Function

| Term | Function | Weight | Description |
|---|---|---|---|
| ee_pos_track | position_command_error (L2) | -1.0 | Penalises EE-target distance |
| ee_pos_track_fine | position_command_error_tanh (std=0.5) | +1.5 | Dense saturating position reward |
| ee_ori_track | orientation_command_error (L2) | -0.1 | Penalises orientation mismatch |
| ee_ori_track_fine | orientation_command_error_tanh (std=0.1) | +0.2 | Dense orientation reward |
| action_rate | L2 of action change | -0.0001 to -0.005 | Smoothness penalty (curriculum) |
| joint_vel | L2 of joint velocities | -0.0001 to -0.001 | Velocity penalty (curriculum) |

Position is prioritised strongly over orientation (ratio ~7.5:1) to ensure the arm first reaches the correct location.

### 1.7 Curriculum

- action_rate: grows from -0.0001 to -0.005 over 4500 iterations
- joint_vel: grows from -0.0001 to -0.001 over 4500 iterations
- ee_ori_track: weight grows to -0.1 over 24,000 iterations

### 1.8 Training Configuration (PPO)

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (RSL-RL) |
| Parallel environments | 4,096 |
| Steps per env per update | 24 |
| Mini-batches | 4 |
| Learning epochs per update | 8 |
| Learning rate | 1e-3 (adaptive) |
| Discount factor gamma | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.001 |
| Clip parameter | 0.2 |
| Network | Actor-Critic, MLP [64, 64, 64], ELU |

---

## 2. Deployment: sim2real_node_test and the Switch-Flip Scenario

### 2.1 Deployment Architecture

The sim2real_node_test ROS 2 node bridges the trained reach policy to both Isaac Sim and the physical robot.

```
/joint_states --> Obs Assembly (29D) --> Policy (TorchScript) --> actions (6D)
                                                                       |
                                                         Pinocchio DLS IK --> dq (6D)
                                                                       |
                                               JointTrajectory --> ros2_control
                                                                       |
                                                          Isaac Sim / Physical Robot
```

### 2.2 Observation Vector Assembly (Real-Time)

```python
joint_pos_rel = [current_pos[j] - default_pos[j] for j in all_joint_names]  # 8D
joint_vel_rel = [current_vel[j] for j in all_joint_names]                    # 8D
pose_command  = self.target_pose   # [x, y, z, qw, qx, qy, qz]              # 7D
actions_obs   = self.last_actions.tolist()                                    # 6D

obs = joint_pos_rel + joint_vel_rel + pose_command + actions_obs              # 29D
```

**Critical:** self.last_actions stores the raw, unclipped network output immediately after inference. This matches the Isaac Lab training convention. Storing the clipped or IK-processed value would create a training/deployment distribution mismatch.

### 2.3 Differential IK at Deployment

```python
delta_pose[:3] = actions[:3] * 0.005   # translation [m/step], max 0.15 m/s
delta_pose[3:] = actions[3:] * 0.005   # rotation [rad/step], max 0.15 rad/s

dq = J_arm^T * inv(J_arm * J_arm^T + lambda^2 * I) * delta_pose
```

The 0.005 scale is conservatively lower than training (0.01) to account for real-robot dynamics.

### 2.4 Breaker-Switch Flipping Scenario

The sim2real_node_test node implements an autonomous three-phase motion sequence for flipping a circuit breaker switch. The same reach policy drives all three phases by simply updating the goal observation:

1. **Approach:** Position EE in front of the switch at the correct height
2. **Push:** Move EE upward to flip the switch ON
3. **Retract:** Move the arm back to a safe home position

### 2.5 Waypoint Sequence Definition

```python
self.waypoint_sequence = [
    ([0.5453, 0.000, 0.262, 1.0, 0.0, 0.0, 0.0], 0.013),  # Phase 1: Approach
    ([0.545,  0.000, 0.350, 1.0, 0.0, 0.0, 0.0], 0.030),  # Phase 2: Push up
    ([0.380,  0.000, 0.250, 1.0, 0.0, 0.0, 0.0], None),   # Phase 3: Retract (final)
]
```

| Phase | x (m) | y (m) | z (m) | Description |
|---|---|---|---|---|
| 1 - Approach | 0.545 | 0.000 | 0.262 | Tip aligned with switch actuator |
| 2 - Push | 0.545 | 0.000 | 0.350 | Switch pushed upward (+8.8 cm) |
| 3 - Retract | 0.380 | 0.000 | 0.250 | Safe standoff position |

### 2.6 Distance-Based Phase Transitions

```python
dist = np.linalg.norm(ee_pos - np.array(tgt))
if dist < threshold:
    self._execute_waypoint(next_index)
```

Transitions are triggered by EE-to-target Euclidean distance (computed via Pinocchio FK), not time. This ensures the robot has actually reached the target before proceeding.

### 2.7 Sequence Flow

```
Start -> Load policy -> Wait for joint_states + URDF -> Init marker
    |
    -> Phase 1 (Approach): pose_command = (0.545, 0, 0.262)
         |
         -> dist < 0.013 m -> Phase 2 (Push): pose_command = (0.545, 0, 0.350)
              |
              -> dist < 0.030 m -> Phase 3 (Retract): pose_command = (0.380, 0, 0.250)
                   |
                   -> threshold = None -> holds pose indefinitely
```

### 2.8 Trajectory Controller Interface

```python
point.time_from_start.nanosec = int((1.5 / control_freq) * 1e9)  # 50 ms at 30 Hz
```

time_from_start is set to 1.5 control periods (not 1.0) to prevent the JointTrajectoryController from reaching its spline endpoint before the next command arrives, which causes visible arm oscillation.

### 2.9 Isaac Sim Compatibility

```bash
ros2 run trossen_sim2real_ros sim2real_node_test --ros-args -p use_sim_time:=true
```

The node is fully compatible with Isaac Sim via topic_based_ros2_control. The only change is enabling the simulation clock.

---

## 3. Sim-to-Real Transfer Considerations

### 3.1 Action Scale Gap
Training uses scale=0.01, deployment uses 0.005 (0.5x). The policy adapts implicitly because it observes its own raw outputs as last_actions.

### 3.2 Observation Noise as Domain Randomisation
Training noise +/-0.01 rad on joint_pos and joint_vel compensates for real-world sensor noise and encoder differentiation.

### 3.3 Gripper State
Both gripper joints are included in the 8D joint observation even though they do not move, ensuring exact observation vector dimension match.

### 3.4 FK-Based Distance Sensing
Waypoint advancement uses Pinocchio FK directly from /joint_states, eliminating TF2 pipeline latency.

---

## 4. Key Parameters Reference

| Parameter | Training | Deployment | Note |
|---|---|---|---|
| Control frequency | 30 Hz | 30 Hz | Must match |
| Physics frequency | 60 Hz (decimation 2) | - | Sim only |
| Action scale (trans) | 0.01 m/step | 0.005 m/step | Conservative for safety |
| Action scale (rot) | 0.01 rad/step | 0.005 rad/step | Conservative for safety |
| DLS damping lambda | 0.01 | 0.01 | Singularity robustness |
| Observation vector | 29D | 29D | Must match exactly |
| Network | MLP [64,64,64] ELU | Same (loaded) | TorchScript |
| Approach threshold | - | 0.013 m | Phase 1->2 |
| Push threshold | - | 0.030 m | Phase 2->3 |

---

## 5. RL Training Run: Detailed Summary

**Policy file:** `logs/rsl_rl/wxai_reach/2026-07-14_02-13-52/exported/policy.pt`
**Task variant:** `wxai_reach` — IK-Relative pose control (`ik_rel_env_cfg.py`)
**Total training iterations:** 3000 (model saved every 50 iterations, final: `model_2999.pt`)
**Seed:** 42  |  **Device:** `cuda:0`

---

### 5.1 PPO Agent Hyperparameters (from `agent.yaml`)

| Hyperparameter | Value | Notes |
|---|---|---|
| Algorithm | PPO (`OnPolicyRunner`) | On-policy, synchronous |
| Policy class | `ActorCritic` | Shared feature extractor |
| Actor hidden dims | [64, 64, 64] | 3-layer MLP, ELU |
| Critic hidden dims | [64, 64, 64] | 3-layer MLP, ELU |
| Activation | ELU | Applied after each hidden layer |
| Initial noise std | 1.0 (scalar) | Exploration noise initialisation |
| Obs normalisation | false | No running normalisation applied |
| Steps per env per update | 24 | Rollout horizon per environment |
| Max iterations | 3000 | Wall-clock training duration |
| Learning epochs per update | 8 | PPO update epochs per rollout |
| Mini-batches | 4 | Batch splits per epoch |
| Learning rate | 0.001 | Adaptive schedule (KL target = 0.01) |
| Discount factor γ | 0.99 | Long-horizon credit assignment |
| GAE λ | 0.95 | Advantage estimation smoothing |
| Entropy coefficient | 0.001 | Exploration bonus |
| Clip parameter | 0.2 | PPO policy clipping |
| Value loss coefficient | 1.0 | Critic loss weight |
| Clipped value loss | true | Value function PPO clipping |
| Max gradient norm | 1.0 | Gradient clipping |
| Advantage normalisation | false per mini-batch | |
| Logger | TensorBoard | |

---

### 5.2 Environment Parameters (from `env.yaml`)

| Parameter | Value |
|---|---|
| Parallel environments (play/eval run) | 32 |
| Episode length | 12.0 s |
| Physics dt | 1/60 s = 0.01667 s |
| Policy decimation | 2 (policy runs at 30 Hz) |
| Joint position reset range | [0.5, 1.5] × default |
| Joint velocity reset | 0.0 |
| Gravity | (0, 0, -9.81) m/s² |
| Physics solver | PhysX TGS (solver_type=1) |
| Self-collision | enabled |

---

### 5.3 Action Space (from `env.yaml`)

| Property | Value |
|---|---|
| Action class | `DifferentialInverseKinematicsAction` |
| Controlled joints | `joint_[0-5]` (6 arm joints) |
| Body (EE frame) | `ee_gripper_link` |
| Command type | `pose` (6D: Δpos + Δori) |
| IK mode | Relative (`use_relative_mode: true`) |
| IK method | DLS (`dls`) |
| DLS damping λ | 0.01 |
| Action scale | **0.01** (applied before IK solve) |
| Action clipping | none |
| Gripper action | null (gripper not controlled) |

The action scale of 0.01 means a saturated action of ±1.0 produces a Cartesian EE displacement of ±0.01 m per step (at 30 Hz → max 0.3 m/s translation, 0.3 rad/s rotation).

---

### 5.4 Observation Space (from `env.yaml`)

Total dimension: **29**

| Index | Name | Dim | Function | Noise |
|---|---|---|---|---|
| 0–7 | `joint_pos` | 8 | `joint_pos_rel` | Uniform ±0.01 rad |
| 8–15 | `joint_vel` | 8 | `joint_vel_rel` | Uniform ±0.01 rad/s |
| 16–22 | `pose_command` | 7 | `generated_commands("ee_pose")` | none |
| 23–28 | `actions` | 6 | `last_action` | none |

- `joint_pos_rel` and `joint_vel_rel` include **all 8 articulation joints**: `joint_0`–`joint_5` plus `left_carriage_joint` and `right_carriage_joint` (gripper rails, fixed at 0).
- `pose_command` encodes the commanded EE pose as `[x, y, z, qw, qx, qy, qz]` in the robot base frame.
- `actions` is the **raw, unclipped** network output from the previous timestep.
- `enable_corruption: true` — noise is applied during training; disabled in play/eval mode.

---

### 5.5 Reward Function (from `env.yaml`)

| Term | Function | Initial Weight | Final Weight (curriculum) |
|---|---|---|---|
| `end_effector_position_tracking` | `position_command_error` (L2 distance) | **−1.0** | −1.0 (fixed) |
| `end_effector_position_tracking_fine_grained` | `position_command_error_tanh` (std=0.5) | **+1.5** | +1.5 (fixed) |
| `end_effector_orientation_tracking` | `orientation_command_error` (L2) | **−0.1** | **−0.1** (curriculum target, over 24000 steps) |
| `end_effector_orientation_tracking_fine_grained` | `orientation_command_error_tanh` (std=0.1) | **+0.2** | +0.2 (fixed) |
| `action_rate` | `action_rate_l2` | **−0.0001** | **−0.005** (over 4500 steps) |
| `joint_vel` | `joint_vel_l2` | **−0.0001** | **−0.001** (over 4500 steps) |

**Reward design notes:**
- The coarse position term (`weight=−1.0`) provides a shaped gradient over the full workspace.
- The fine-grained tanh term (`weight=+1.5`) saturates near the target, providing a strong dense bonus within ~0.5 m of the goal.
- Orientation tracking starts at −0.1 and is ramped up only after the policy converges on position tracking (curriculum over 24,000 steps).
- Action rate and joint velocity penalties begin small and are ramped up during training to produce progressively smoother motions.

---

### 5.6 Target Pose Command Sampling (from `env.yaml`)

| Axis | Range | Unit |
|---|---|---|
| x (forward) | [0.35, 0.55] | m |
| y (lateral) | [−0.20, 0.20] | m |
| z (vertical) | [0.10, 0.45] | m |
| roll | [−π/4, +π/4] ≈ [−45°, +45°] | rad |
| pitch | [0, π/4] ≈ [0°, 45°] | rad |
| yaw | [−π/4, +π/4] ≈ [−45°, +45°] | rad |

Target poses are sampled from `UniformPoseCommand` and resampled every **4.0 seconds** (fixed interval). The policy must generalise to any pose within this box, which covers the arm's primary forward reachable volume.

---

### 5.7 Curriculum Schedule

| Term | Initial Weight | Final Weight | Steps |
|---|---|---|---|
| `action_rate` | −0.0001 | −0.005 | 4500 |
| `joint_vel` | −0.0001 | −0.001 | 4500 |
| `end_effector_orientation_tracking` | 0.0 | −0.1 | 24000 |

The orientation tracking curriculum delays orientation penalty until position tracking is learned (~early 24000 steps). The smoothness penalties grow early (4500 steps) to prevent jerky motions.

---

### 5.8 Default Robot Pose (from `env.yaml`)

```yaml
joint_pos:
  joint_0: 0.0         # 0.0°
  joint_1: 1.0472      # 60.0°  (pi/3)
  joint_2: 0.5236      # 30.0°  (pi/6)
  joint_3: 0.6283      # 36.0°  (pi/5)
  joint_4: 0.0         # 0.0°
  joint_5: 0.0         # 0.0°
  left_carriage_joint: 0.0   # gripper closed
```

At each episode reset, all joints are scaled by a factor drawn from U(0.5, 1.5) to diversify the initial condition, while joint velocities are always reset to zero.

