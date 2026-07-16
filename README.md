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

## External Setup & Configuration (Required Changes Outside This Package)

To run the nodes in this package, the hardware configuration in the `trossen_arm_bringup` package must be updated to support raw joint group position control alongside joint trajectory control.

### 1. Controller Manager Configuration
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

### 2. Robot Launch Script
Modify [trossen_arm.launch.py](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/launch/trossen_arm.launch.py) to spawn the `forward_position_controller` as `--inactive` by default. This avoids hardware interface conflicts at launch:

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

### 3. Install Controller Dependencies
Ensure the necessary ROS 2 controller plugins are installed on your host system or inside the Docker container:
```bash
sudo apt-get update
sudo apt-get install -y ros-jazzy-position-controllers ros-jazzy-forward-command-controller
```

---

## Operating Modes & Usage

Build the workspace after making modifications:
```bash
colcon build --packages-select trossen_sim2real_ros trossen_arm_bringup
source install/setup.bash
```

Depending on the environment, follow the steps below.

### Mode 1: Mock Components (Offline/RViz Testing)

Uses fake hardware interfaces. Commands are reflected directly back to joint states for visualization in RViz.

1. **Launch the robot hardware interface (Mock mode)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=mock_components
   ```
2. **Switch controllers** (Required for the raw position/trapezoidal control nodes):
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **Run a controller node**:
   * **Trapezoidal Profile (Custom Interpolation for RL)**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * **Standard Policy Control**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node
     ```

---

### Mode 2: Isaac Sim (Simulation)

Enables control inside the NVIDIA Isaac Sim environment via the topic-based bridge.

> [!IMPORTANT]
> Because Isaac Sim acts as the master simulation clock, all ROS 2 nodes must run with `use_sim_time:=true` to prevent clock synchronization and transform errors (e.g., `TF_OLD_DATA`).

1. **Launch the robot hardware interface (Isaac Sim mode)**:
   ```bash
   # Re-source setup to ensure the new position controller plugins are discovered by AMENT
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   
   # Launch with sim time enabled
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=isaac use_sim_time:=true
   ```
2. **Switch controllers**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **Run a controller node with Sim Time**:
   * **Trapezoidal Profile**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal --ros-args -p use_sim_time:=true
     ```
   * **Standard Policy Control**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node --ros-args -p use_sim_time:=true
     ```

---

### Mode 3: Real World Robot

Controls the physical WidowX AI robot arm.

1. **Launch the robot hardware interface (Real hardware mode)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=real ip_address:=<ROBOT_IP>
   ```
2. **Switch controllers**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **Run a controller node**:
   * **Trapezoidal Profile**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * **Standard Policy Control**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node
     ```

---

## Node Descriptions

* **`sim2real_node`**: Interacts with `JointTrajectoryController`. Calculates target positions from the RL policy and creates a step-by-step joint trajectory target.
* **`sim2real_node_trapezoidal`**: Bypasses the trajectory generator by publishing raw position arrays to `JointGroupPositionController`. Features built-in custom trapezoidal acceleration and velocity limiting to keep RL execution safe, fast, and smooth on physical links.