# Drone Fencing with Control Barrier Functions

A demo for the Stanford Robotics Center opening, November 2024

## Installation

```
# Prereqs
# Install Ubuntu 22.04
# Install ROS Humble
# Install vrpn-mocap (sudo apt install ros-humble-vrpn-mocap)

# Virtual environment
pyenv install 3.10.8
pyenv virtualenv 3.10.8 drone_fencing

# Construct the ROS workspace
mkdir -p drone_fencing_ws/src
cd drone_fencing_ws/src

# Clone the repo + build TrajBridge
# Note that this should be done outside of the controller environment
git clone https://github.com/danielpmorton/drone_fencing
cd drone_fencing # TODO: Decide if this folder needs to be named TrajBridge
source /opt/ros/humble/setup.bash
git submodule update --init --recursive
cd TrajBridge
colcon build

# Install dependencies
cd ../cbf_controller
pyenv local drone_fencing
pip install -e .
```

### Additional setup steps:

To reduce issues from multiple people publishing to drone ROS2 topics on the same network, we use the ROS_DOMAIN_ID to isolate our rostopics from other devices on the network. To do this,
- On the drone companion computer AND the ground station computer, update the `~/.bashrc` to include the line `export ROS_DOMAIN_ID=123`. `123` is just an arbitrary choice that should not conflict with anyone else. 
- On the drone's PX4 (connected via USB to a computer running qgroundcontrol), set the `UXRCE_DDS_DOM_ID` parameter to the same value (`123`)

The version of px4_msgs must also match between all devices (the companion computer, ground station, AND the px4). While the px4 does not use ROS directly, the version of the firmware on the px4 must be compatible with the version of px4_msgs being used. 
- For the TrajBridge state machine code which worked on the drone's companion computer, see [this link](https://github.com/danielpmorton/trajbridge_fencing)
- For the PX4 firmware which is compatible with this codebase, see [this link](https://drive.google.com/file/d/1fcM4B2ZX2NEXhMTd7MIiBsnSKaGJbeuI/view?usp=drive_link). Note that this is built for a `fmu` namespace, rather than a drone-specific namespace

To not have to constantly source the ROS 2 files, add the following lines to the `~/.bashrc` on both the companion computer and ground station: `source /opt/ros/humble/setup.bash` 


## Operation

Terminal 1 (Computer): Mocap
```
cd drone_fencing_ws/src/TrajBridge/TrajBridge
source install/setup.bash
ros2 launch px4_comm src_mocap.launch.py
```
Terminal 2 (Drone): MicroXRCE
```
# ssh asl@drone11.local
sudo micro-xrce-dds-agent serial --dev /dev/ttyTHS1 --baudrate 921600 
```
Terminal 3 (Drone): TrajBridge
```
# ssh asl@drone11.local
cd ~/StanfordMSL/TrajBridge/TrajBridge
source install/setup.bash
ros2 launch px4_comm trajbridge.launch.py
```
Terminal 4 (Computer): Controller
```
cd drone_fencing_ws/src/TrajBridge/TrajBridge
source install/setup.bash
cd ../cbf_controller
# pyenv shell drone_fencing if not already activated
python cbf_controller/cbf_node.py
```

## SRC vs Flightroom notes

Optitrack in the flightroom has Z-up, but in SRC, it was set to Y-up. To account for this, this required adjusting:
- The transforms in the mocap node (see [this commit](https://github.com/danielpmorton/drone_fencing/commit/a2faf303e8ff1dbb1ecb119dfbcb5b255214821c))
- The obstacle transform (mocap frame to drone frame) (see [this commit](https://github.com/danielpmorton/drone_fencing/commit/def7f226dd01ed4a1ef6fbec79e3b0fdfa348fff))

Some other things that needed adjusting were:
- The IP address of the optitrack computer
- The room limits in the TrajBridge state machine on the drone (see [this commit](https://github.com/danielpmorton/trajbridge_fencing/commit/dde5424490852301d6f67326d0eb744492c877c4)) and the CBF (see [this commit](https://github.com/danielpmorton/drone_fencing/commit/def7f226dd01ed4a1ef6fbec79e3b0fdfa348fff))

## Assorted notes

- Sometimes, the EKF on the PX4 would fail to get a lock on the mocap data. A few reboots / power cycles of the full system (px4, companion computer, ground station) typically fixed this, though I still don't know exactly what caused it. In general, it's good to run a pre-flight check that the px4 odometry data looks correct and stable. To do so, run `ros2 topic echo /fmu/out/vehicle_odometry` and check that it looks correct. It's useful to move the drone around a bit and check the data.
- I updated the px4 acceleration parameters (`MPC_ACC_HOR` and `MPC_ACC_HOR_MAX`) to make the drone a bit more responsive to velocity commands in the horizontal direction. For the final exported PX4 parameters, see [this link](https://drive.google.com/file/d/1av-afm3RLSJTzFyXxPk46MqBAAfgHmd-/view?usp=sharing)

## Debugging

- If the EKF estimate on the drone is randomly flipping signs, the firmware is probably bad.
- If the TrajBridge state machine is not transitioning states, there might be a version mismatch between the px4_msgs between the companion computer, ground station, and px4
- If you get a message about QoS incompatibility, check how this was done in the cbf_node file. The mocap uses a QoS which is different than the default

## Additional resources

- [TrajBridge wiki page](https://github.com/StanfordMSL/TrajBridge/wiki)
- [MSL Quad Build Guide](https://docs.google.com/presentation/d/1GplwG5dU9iBMCfxJVyR89wg4f9wwwXytaCo_LqSN5lc/edit?usp=sharing)