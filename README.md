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

## Operation

Terminal 1 (Computer): Mocap
```
cd drone_fencing_ws/src/Trajbridge/Trajbridge
source install/setup.bash
ros2 launch px4_comm mocap.launch.py
```
Terminal 2 (Drone): MicroXRCE
```
# ssh asl@drone10.local
micro-xrce-dds-agent serial --dev /dev/ttyTHS0 -b 921600 
# Note that ttyTHS0 may need to be replaced by ttyTHS1
```
Terminal 3 (Drone): TrajBridge
```
# ssh asl@drone10.local
cd drone_fencing_ws/src/TrajBridge/TrajBridge
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


See the [TrajBridge wiki page](https://github.com/StanfordMSL/TrajBridge/wiki) for additional info
