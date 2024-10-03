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
cd drone_fencing_ws/src/TrajBridge/TrajBridge
source install/setup.bash
ros2 launch px4_comm mocap.launch.py
```
Terminal 2 (Drone): MicroXRCE
```
# ssh asl@drone10.local
sudo micro-xrce-dds-agent serial --dev /dev/ttyTHS1 -b 921600 
```
Terminal 3 (Drone): TrajBridge
```
# ssh asl@drone10.local
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


See the [TrajBridge wiki page](https://github.com/StanfordMSL/TrajBridge/wiki) for additional info


## Debugging notes

- Had to change the ROS_DOMAIN_ID env var to 123 in the `~/.bashrc` to avoid other people running things from interfering with our code
- Also, changed the UXRCE_DDS_DOM_ID parameter on the microcontroller (via qgroundcontrol) to the same value
- Changed the Trajbridge code to match Keiko's older version (with an older version of the px4_comm/p4_msgs packages) -- to fix a bug where the state machine would be subscribing to a message with a different structure than what was actually being published (from a different version of the firmware on the microcontroller + the micro xrce bridge)
- Added `source /opt/ros/humble/setup.bash` to the `~/.bashrc` files on the drone and laptop

## TODO
- Update the drone11 px4 UXRCE_DDS_DOM_ID parameter
- Check to make sure that the bashrc is the same on drone10/11
- Try changing the waypoint ready in trajbridge **on the drone, not on the laptop**
- Echo the fmu/in/... rostopic to see if the command signal from trajbridge to drone is correct. The drone will need to be sitting still near the waypoint ready and then you can activate the cbf control node (without arming the drone)


## Notes on local network in dance studio

TODO
- Get the router set up in a high area away from anything metal. Command strips/hooks?

Config/params
- Wifi: TP-Link_BDE8
- Hostname for MSL laptop: mslxps2 
- Hostname for mocap computer: SRC-campc-030
- Hostname for drone: drone10 or drone11
- IP address for drone10: 192.168.0.238

Notes
- Remember to enable cameras and flip the "VRPN Enable" switch in settings
- Getting the mocap to work just required creating a new mocap launch file with the SRC optitrack IP address in the server line
- There was some noticeable lag at 120Hz -- 360 seems better. Note that if you go too high on frequency then you lose camera frame size, so 360 seems to strike a good balance between image size and responsiveness
- Optitrack defaults to y up, but the flight room uses z up. I switched it to z-up in the streaming settings
- Connecting the laptop to the LAN via ethernet is ideal to minimize any communication lags
- SSH into drone 10 as `ssh asl@192.168.0.238`
- All px4_msgs must be the same version!! Can't have a newer version on one computer and an older version on a drone, for instance -- the message definitions need to match up or else it won't work


HACKS
- NOTE: hacked the topics ON THE DRONE to use drone10 as a namespace, since keiko gave me the wrong firmware
- Removed the vehicle state checks in trajbridge due to the incorrect firmware -- there seems to be an odd mismatch between the px4_msgs topics (only for vehicle status) which caused this message to never be read, and thus it never transitioned to the waypoint state


WEIRD STUFF
- The mocap was sign flipping again, which is incredibly frustrating ........ this sign flip seems to happen when the micro-xrce agent restarts its publishers, and I don't know why this happens yet. It seems to only happen when I am running trajbridge though



## Peflight checks
- Ensure that mocap is publishing and that the values look correct
- Ensure that the vehicle EKF state estimate is listening to the mocap



SRC: changed ip address to SRC cameras to 192.168.0.119 in network and internet settings 

WIFI PW: fencingsrcdemo
WIFI: drone_fencing_5G
drone11 ip: 192.168.0.223 (can also access via drone11.local)

Changed the COM_POWER_COUNT to 0 (from 1) since failed to arm without connecting to computer