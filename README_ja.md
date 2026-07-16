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

---

## 必要条件とインストール

### ハードウェア要件
* **ロボット**: Trossen WidowX AI アーム (wxai) または互換性のある WidowX シリーズアーム。
* **ワークステーション**: Ubuntu 24.04 LTS（または ROS 2 のディストリビューションに応じた Ubuntu 22.04）が動作する PC。
* **GPU**: NVIDIA GPU（Isaac Sim を実行する場合に必須）。

### ソフトウェア依存関係
以下のパッケージがインストールされていることを確認してください：
* **ROS 2**: Jazzy (または Humble)
* **Python パッケージ**:
  ```bash
  pip install torch pinocchio numpy
  ```
* **ROS 2 Control およびコントローラー関連パッケージ**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-topic-based-ros2-control
  ```
  *(注: ROS 2 Humble を使用している場合は、パッケージ名の中の `jazzy` を `humble` に置き換えてください。)*

### Git クローンとセットアップ
ROS 2 ワークスペースを作成し、リポジトリをクローンして、`vcs` と `rosdep` を使用して依存関係をインポートします：

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 1. Trossen ロボットの設定 & bringup パッケージのクローン (jazzy ブランチ)
git clone -b jazzy https://github.com/TrossenRobotics/trossen_arm_ros.git

# 2. この Sim2Real パッケージのクローン
git clone https://github.com/Hayato-80/trossen_sim2real_ros.git

# 3. 残りの依存関係をインポート（trossen_arm と trossen_arm_description は自動的にクローンされます）
cd ~/ros2_ws
vcs import src < src/trossen_arm_ros/dependencies.repos

# 4. システム依存関係のインストール
rosdep install --from-paths src --ignore-src -r -y
```

### ワークスペースのビルドとソース
ワークスペースをビルドしてソースします：

```bash
cd ~/ros2_ws
colcon build --symlink-install

# ワークスペースのソース
source ~/ros2_ws/install/setup.bash
```

> [!TIP]
> 新しい端末を開くたびにワークスペースが自動的にソースされるように、この行を `~/.bashrc` ファイルに追加することができます：
> ```bash
> echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
> ```

---

## 実行モードと使用方法（標準）

これらの手順では、標準の軌道ベースのポリシーノード（`sim2real_node`）を実行します。ロボットのデフォルトの関節軌道コントローラー（`arm_controller`）が起動時からアクティブになっているため、追加の設定は不要です。

### モード 1: Mock コンポーネント（オフライン/RViz テスト用）

仮想的なハードウェアインターフェースを使用します。コマンドは直接関節状態（joint states）に反映され、RViz 上でシミュレーション表示されます。

1. **ロボットハードウェアインターフェースの起動 (Mock モード)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=mock_components
   ```
2. **制御ノードの実行**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node
   ```

### モード 2: Isaac Sim (シミュレーション)

トピックベースのブリッジを介して、NVIDIA Isaac Sim 環境内で制御を行います。

> [!IMPORTANT]
> Isaac Sim がマスターシミュレーションクロックとして機能するため、クロック同期やトランスフォームのエラー（例：`TF_OLD_DATA`）を防ぐために、すべての ROS 2 ノードを `use_sim_time:=true` で実行する必要があります。

1. **ロボットハードウェアインターフェースの起動 (Isaac Sim モード)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=isaac use_sim_time:=true
   ```
2. **シミュレーション時間を指定して制御ノードを実行**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node --ros-args -p use_sim_time:=true
   ```

### モード 3: 実機ロボット

物理的な WidowX AI ロボットアームを制御します。

1. **ロボットハードウェアインターフェースの起動 (実機モード)**:
   ```bash
   ros2 launch trossen_arm_bringup trossen_arm.launch.py ros2_control_hardware_type:=real ip_address:=<ROBOT_IP>
   ```
2. **制御ノードの実行**:
   ```bash
   ros2 run trossen_sim2real_ros sim2real_node
   ```

---

## ノード解説

* **`sim2real_node`**: デフォルトの `JointTrajectoryController` に対してコマンドを送信します。RLポリシーから目標関節位置を計算し、関節軌道（Joint Trajectory）の指令値を作成・送信します。特別な設定なしでそのまま動作します。

---

## [任意] 付録: 台形プロファイル制御

`sim2real_node_trapezoidal` ノードは、軌道生成器をバイパスして、生の位置配列を直接 `JointGroupPositionController` にパブリッシュします。内蔵された独自の台形速度・加速度制限機能により、高速で滑らかな制御を可能にします。

このノードを使用するには、パッケージ外の設定ファイルを修正する必要があります。

### パート 1: 外部設定ファイルの修正

#### 1. コントローラーマネージャーの設定
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

#### 2. ロボット起動スクリプトの修正
[trossen_arm.launch.py](file:///home/hayato/rl_ws/isaac_ros-dev/src/trossen_arm_ros/trossen_arm_bringup/launch/trossen_arm.launch.py) を修正し、起動時に位置コントローラーがデフォルトで非アクティブ (`--inactive`) としてスポーンされるようにします。これにより起動時の競合を防ぎます：

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

#### 3. ワークスペースのビルド
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### パート 2: コントローラー切り替えと実行

生の関節位置制御を行うには、使用するコントローラーを切り替える必要があるため、ノードを実行する前にデフォルトの軌道コントローラーを非アクティブ化し、位置コントローラーをアクティブ化します。

1. **ロボットハードウェアインターフェースの起動** (Mock, Isaac Sim with sim_time, または実機)
2. **コントローラーの切り替え**:
   ```bash
   ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{activate_controllers: ['forward_position_controller'], deactivate_controllers: ['arm_controller'], strictness: 1}"
   ```
3. **台形ノードの実行**:
   * **Mock/実機の場合**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal
     ```
   * **Isaac Sim の場合**:
     ```bash
     ros2 run trossen_sim2real_ros sim2real_node_trapezoidal --ros-args -p use_sim_time:=true
     ```
