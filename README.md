# Trossen WidowX AI Sim2Real ROS 2 Package

<table>
  <thead>
    <tr>
      <th style="text-align:center">English</th>
      <th style="text-align:center"><a href="README_ja.md">日本語</a></th>
    </tr>
  </thead>
</table>

This ROS 2 package enables sim-to-real transfer and interactive marker control of the Trossen WidowX AI arm. It integrates PyTorch RL policies with physical or simulated hardware control interfaces.

---

## Requirements & Installation

### Hardware Requirements
* **Robot**: Trossen WidowX AI arm (wxai) or compatible WidowX series arm.
* **Workstation**: PC running Ubuntu 24.04 LTS (or Ubuntu 22.04 depending on ROS 2 distro).
* **GPU**: NVIDIA GPU (Required if running Isaac Sim).

### Software Dependencies
Ensure you have the following packages installed:
* **ROS 2**: Jazzy (or Humble)
* **Python Packages**:
  ```bash
  pip install torch pinocchio numpy
  ```
* **ROS 2 Control & Controller Packages**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-topic-based-ros2-control
  ```
  *(Note: Replace `jazzy` with `humble` in the package names if you are using ROS 2 Humble.)*

### Git Clone & Setup
Create a ROS 2 workspace, clone the repositories, and import dependencies using `vcs` and `rosdep`:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 1. Clone the Trossen robot config & bringup package (jazzy branch)
git clone -b jazzy https://github.com/TrossenRobotics/trossen_arm_ros.git

# 2. Clone this Sim2Real package
git clone https://github.com/Hayato-80/trossen_sim2real_ros.git

# 3. Import remaining dependencies (automatically clones trossen_arm and trossen_arm_description)
cd ~/ros2_ws
vcs import src < src/trossen_arm_ros/dependencies.repos

# 4. Install system dependencies
rosdep install --from-paths src --ignore-src -r -y
```

### Build & Source Workspace
Build the workspace and source it:

```bash
cd ~/ros2_ws
colcon build --symlink-install

# Source the workspace
source ~/ros2_ws/install/setup.bash
```

> [!TIP]
> You can add this line to your `~/.bashrc` file to automatically source the workspace every time you open a new terminal:
> ```bash
> echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## Operating Modes & Usage (Standard)

These instructions run the standard trajectory-based policy node (`sim2real_node`). No extra configurations are required since the robot's default `arm_controller` is active by default.

### Mode 1: Mock Components (Offline/RViz Testing)

Uses fake hardware interfaces. Commands are reflected directly back to joint states for visualization in RViz.

1. **Launch the robot hardware interface (Mock mode)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=mock_components
   ```
2. **Run the policy control node**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node
   ```

### Mode 2: Isaac Sim (Simulation)

Enables control inside the NVIDIA Isaac Sim environment via the topic-based bridge.

> [!IMPORTANT]
> Because Isaac Sim acts as the master simulation clock, all ROS 2 nodes must run with `use_sim_time:=true` to prevent clock synchronization and transform errors (e.g., `TF_OLD_DATA`).

1. **Launch the robot hardware interface (Isaac Sim mode)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=isaac use_sim_time:=true
   ```
2. **Run the policy control node with Sim Time**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node --ros-args -p use_sim_time:=true
   ```

### Mode 3: Real World Robot

Controls the physical WidowX AI robot arm.

1. **Launch the robot hardware interface (Real hardware mode)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=real ip_address:=<ROBOT_IP>
   ```
2. **Run the policy control node**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node
   ```

---

## Node Descriptions

* **`sim2real_node`**: Interacts with the default `JointTrajectoryController`. Calculates target positions from the RL policy and creates a step-by-step joint trajectory target. This works out of the box with zero external configuration.

---

## [Optional] Appendix: Trapezoidal Profile Control

The `sim2real_node_trapezoidal` node bypasses the trajectory generator and publishes raw position arrays directly to `JointGroupPositionController` with built-in trapezoidal velocity/acceleration limits for fast, smooth control. 

This requires changing hardware parameters outside of the package.

### Part 1: External Config Changes

#### 1. Controller Manager Configuration
Add the `forward_position_controller` definitions to [controllers.yaml](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/config/controllers.yaml):

```yaml
controller_manager:
  ros__parameters:
    # ... (existing controllers) ...
    forward_position_controller:
      type: position_controllers/JointGroupPositionController

forward_position_controller:
  ros__parameters:
    joints:
      - joint_0
      - joint_1
      - joint_2
      - joint_3
      - joint_4
      - joint_5
    interface_name: position
```

#### 2. Robot Launch Script
Modify [trossen_arm.launch.py](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/launch/trossen_arm.launch.py) to spawn the position controller as `--inactive` by default to avoid startup conflicts:

```python
    controller_spawner_nodes.append(
        Node(
            name='forward_position_controller_spawner',
            package='controller_manager',
            executable='spawner',
            arguments=[
                'forward_position_controller',
                '--inactive'
            ],
            output={'both': 'screen'},
        )
    )
```

#### 3. Rebuild Workspace
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Part 2: Switching Controllers & Running

Since raw position control uses a different active controller, you must deactivate the default trajectory controller and activate the position controller before running the node:

1. **Launch the robot hardware interface** (Mock, Isaac with sim_time, or Real).
2. **Switch controllers**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **Run the trapezoidal node**:
   * For **Mock/Real**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * For **Isaac Sim**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal --ros-args -p use_sim_time:=true
     ```