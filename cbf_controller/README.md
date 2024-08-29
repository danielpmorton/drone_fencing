# CBFpy + ROS2 

```
mkdir -p cbfpy_ros_ws/src
cd cbfpy_ros_ws/src
git clone https://github.com/danielpmorton/cbfpy_ros
```


TODO add notes about colcon building


## Launching an example

- Open three terminals
- In each one, run the following
    ```
    cd cbfpy_ros_ws
    source /opt/ros/humble/setup.bash
    # source install/local_setup.bash ???? -- check if necessary
    # Activate venv
    cd src/cbfpy_ros
    ```
Then run the following in the three terminals as follows:

- Terminal 1: `python cbfpy_ros/state_node.py`
- Terminal 2: `python cbfpy_ros/control_node.py`
- Terminal 3: `python cbfpy_ros/cbf_node.py`

