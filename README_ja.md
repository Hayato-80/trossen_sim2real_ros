# Trossen WidowX AI Sim2Real ROS 2 パッケージ

<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README.md">English</a></th>
      <th style="text-align:center">日本語</th>
    </tr>
  </thead>
</table>

この ROS 2 パッケージは、Trossen WidowX AI アームの Sim-to-Real（シミュレーションから実機への移行）およびインタラクティブマーカーによる制御を可能にします。PyTorch で学習された強化学習（RL）ポリシーと、実機またはシミュレータのハードウェア制御インターフェースを統合します。

## 外部セットアップと設定（本パッケージ外で必要な変更）

本パッケージのノードを実行する前に、`trossen_arm_bringup` パッケージ内のハードウェア設定を更新し、通常の関節軌道制御（Joint Trajectory Control）に加えて、生の関節グループ位置制御（Raw Joint Group Position Control）をサポートできるようにする必要があります。

### 1. コントローラーマネージャーの設定
[controllers.yaml](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/config/controllers.yaml) に `forward_position_controller` の定義を追加します：

```yaml
controller_manager:
  ros__parameters:
    # ... (既存のコントローラー) ...
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

### 2. ロボット起動スクリプトの修正
[trossen_arm.launch.py](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/launch/trossen_arm.launch.py) を修正し、起動時に `forward_position_controller` がデフォルトで非アクティブ (`--inactive`) としてスポーンされるようにします。これにより、起動時のハードウェアインターフェースの競合を防ぎます：

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

### 3. コントローラー依存関係のインストール
ホストシステムまたは Docker コンテナに必要な ROS 2 コントローラープラグインがインストールされていることを確認します：
```bash
sudo apt-get update
sudo apt-get install -y ros-jazzy-position-controllers ros-jazzy-forward-command-controller
```

---

## 実行モードと使用方法

変更を加えた後、ワークスペースをビルドします：
```bash
colcon build --packages-select trossen_sim2real_ros trossen_arm_bringup
source install/setup.bash
```

使用環境に応じて、以下の手順を実行してください。

### モード 1: Mock コンポーネント（オフライン/RViz テスト用）

仮想的なハードウェアインターフェースを使用します。コマンドは直接関節状態（joint states）に反映され、RViz 上でシミュレーション表示されます。

1. **ロボットハードウェアインターフェースの起動 (Mock モード)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=mock_components
   ```
2. **コントローラーの切り替え** (生位置制御/台形制御ノードに必要)：
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **制御ノードの実行**:
   * **台形プロファイル制御（強化学習向けのカスタム補間）**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * **通常のポリシー制御**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node
     ```

---

### モード 2: Isaac Sim (シミュレーション)

トピックベースのブリッジを介して、NVIDIA Isaac Sim 環境内で制御を行います。

> [!IMPORTANT]
> Isaac Sim がマスターシミュレーションクロックとして機能するため、クロック同期やトランスフォームのエラー（例：`TF_OLD_DATA`）を防ぐために、すべての ROS 2 ノードを `use_sim_time:=true` で実行する必要があります。

1. **ロボットハードウェアインターフェースの起動 (Isaac Sim モード)**:
   ```bash
   # 新しくインストールした位置コントローラープラグインが AMENT に検出されるように設定をリロード
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   
   # シミュレーション時間を有効にして起動
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=isaac use_sim_time:=true
   ```
2. **コントローラーの切り替え**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **シミュレーション時間を指定して制御ノードを実行**:
   * **台形プロファイル制御**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal --ros-args -p use_sim_time:=true
     ```
   * **通常のポリシー制御**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node --ros-args -p use_sim_time:=true
     ```

---

### モード 3: 実機ロボット

物理的な WidowX AI ロボットアームを制御します。

1. **ロボットハードウェアインターフェースの起動 (実機モード)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=real ip_address:=<ROBOT_IP>
   ```
2. **コントローラーの切り替え**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **制御ノードの実行**:
   * **台形プロファイル制御**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * **通常のポリシー制御**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node
     ```

---

## ノード解説

* **`sim2real_node`**: `JointTrajectoryController` と対話します。RLポリシーから目標位置を計算し、ステップごとの関節軌道ターゲットを作成します。
* **`sim2real_node_trapezoidal`**: 軌道生成器をバイパスして、生の位置配列を直接 `JointGroupPositionController` にパブリッシュします。内蔵された独自の台形加速度・速度制限機能により、物理アーム上の制御を安全かつ高速、滑らかに保ちます。
