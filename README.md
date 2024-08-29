# Drone Fencing with Control Barrier Functions

A demo for the Stanford Robotics Center opening, November 2024

## Installation

```
# Virtual environment
pyenv install 3.10.8
pyenv virtualenv 3.10.8 drone_fencing
pyenv shell drone_fencing

# Construct the ROS workspace
mkdir -p drone_fencing_ws/src
cd drone_fencing_ws/src

# Clone the repo + build TrajBridge
git clone https://github.com/danielpmorton/drone_fencing
cd TrajBridge
git submodule update --init --recursive
cd TrajBridge
colcon build

# Install dependencies
cd ../cbf_controller
pip install -e .
```

## Operation

Terminal 1
```
cd drone_fencing_ws/src/TrajBridge/TrajBridge
source install/setup.bash
ros2 launch px4_comm trajbridge.launch.py
```
Terminal 2
```
pyenv shell drone_fencing
cd drone_fencing_ws/src/TrajBridge/TrajBridge
source install/setup.bash
cd ../cbf_controller
python cbf_controller/cbf_node.py
```

See the [TrajBridge wiki page](https://github.com/StanfordMSL/TrajBridge/wiki) for additional info