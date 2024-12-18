"""Main ROS2 node for the drone fencing demo. 

Assumes we have a PX4 drone and an obstacle (sword tip) tracked by motion capture.
"""

# TODO: Tune the following
# - Publishing frequency
# - QoS depth
# - Velocity limits
# - Bounding box limits
# - Obstacle radius/padding
# - Nominal controller gains
# - Obstacle velocity buffer depth

from collections import deque

import numpy as np
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from jax import Array
from cbfpy import CBF, CBFConfig

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry

jax.config.update("jax_enable_x64", True)

NODE_NAME = "cbf_node"
VEHICLE_ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
OBSTACLE_MOCAP_TOPIC = "/vrpn_mocap/obstacle/pose"
VELOCITY_SETPOINT_TOPIC = "/setpoint_control/velocity_with_ff"


class DroneConfig(CBFConfig):
    """Configuration for the SRC drone demo: Dynamic obstacle avoidance with drones and CBFs

    We assume that the drone is a point robot and that we have direct control over its velocity.

    i.e. z = [x, y, z, vx, vy, vz]
    and z_dot = f(z) + g(z)u = [0, 0, 0, 0, 0, 0] + [vx, vy, vz, 0, 0, 0]
    where u  = [vx, vy, vz]

    Args:
        pos_min (ArrayLike, optional): XYZ Lower bound of the safe-set box. Defaults to (-2, -2, -1.5)
        pos_max (ArrayLike, optional): XYZ Upper bound of the safe-set box. Defaults to (2, 2, -0.5)
        drone_radius (float, optional): Radius of the drone. Defaults to 0.175.
        obstacle_radius (float, optional): Radius of the obstacle. Defaults to 0.04.
        padding (float, optional): Padding between the drone and the obstacle. Defaults to 0.25.
        lookahead_time (float, optional): Time horizon for obstacle avoidance. Defaults to 2.0.
    """

    # SRC field robotics room safety box is (5m, 4m, 2.3m)
    # We'll also tighten this constraint for the CBF
    # Remember that it is negative up for drone z coordinates

    def __init__(
        self,
        pos_min: ArrayLike = (-2.25, -1.75, -2.0),  # (-2.25, -1.5, -2.3),
        pos_max: ArrayLike = (2.25, 1.75, -0.5),  # (2.25, 1.5, -0.5),
        drone_radius: float = 0.175,
        obstacle_radius: float = 0.15,  # Bump this up? + reduce padding?
        padding: float = 0.25,
        lookahead_time: float = 2.0,
    ):
        self.mass = 1.0
        self.pos_min = jnp.asarray(pos_min, dtype=jnp.float64)
        self.pos_max = jnp.asarray(pos_max, dtype=jnp.float64)
        self.drone_radius = drone_radius
        self.obstacle_radius = obstacle_radius
        self.padding = padding
        self.lookahead_time = lookahead_time
        init_z_obs = jnp.array([-2.0, 0.0, -1.0, -0.1, -0.1, -0.1])
        super().__init__(
            n=6,
            m=3,
            # NOTE: This code was written for an older version of cbfpy. num_barr and relative_degree are no longer used
            num_barr=7,
            relative_degree=1,
            u_min=jnp.array([-8.0, -8.0, -4.0]),
            u_max=jnp.array([8.0, 8.0, 4.0]),
            relax_cbf=True,
            init_args=(init_z_obs,),
            cbf_relaxation_penalty=1e6,
        )

    def f(self, z):
        # Assume we are directly controlling the robot's velocity
        return jnp.zeros(6)

    def g(self, z):
        # Assume we are directly controlling the robot's velocity
        return jnp.block([[jnp.eye(3)], [jnp.zeros((3, 3))]])

    # NOTE: This code was written for an older version of cbfpy. h should be replaced with h_1
    def h(self, z, z_obs):
        pos_robot = z[:3]
        vel_robot = z[3:]
        pos_obs = z_obs[:3]
        vel_obs = z_obs[3:]
        dist_between_centers = jnp.linalg.norm(pos_obs - pos_robot)
        dir_obs_to_robot = (pos_robot - pos_obs) / dist_between_centers
        collision_velocity_component = (vel_obs - vel_robot).T @ dir_obs_to_robot
        h_obstacle_avoidance = jnp.array(
            [
                dist_between_centers
                - collision_velocity_component * self.lookahead_time
                - self.obstacle_radius
                - self.drone_radius
                - self.padding
            ]
        )
        h_box_containment = jnp.concatenate(
            [self.pos_max - z[:3], z[:3] - self.pos_min]
        )
        return jnp.concatenate([h_obstacle_avoidance, h_box_containment])


def nominal_controller(z: Array, z_des: Array) -> Array:
    """A simple PD controller to control the velocity of the drone to achieve a desired state

    Args:
        z (np.ndarray): Current state of the point-mass reduced model of the drone
            (position + velocity), shape (6,)
        z_des (np.ndarray): Desired state of the point-mass reduced model of the drone
            (position + velocity), shape (6,)

    Returns:
        np.ndarray: Velocity control input, shape (3,)
    """
    Kp = 1.0
    Kv = 1.0
    return Kp * (z_des[:3] - z[:3]) + Kv * (z_des[3:] - z[3:])


# NOTE: the desired state of the drone should be static for the full demo
# (we always want it to go to the same position with no velocity)
# TODO decide on if we should mark this as static
# @partial(jax.jit, static_argnums=(2,))
@jax.jit
def safe_controller(cbf: CBF, z: Array, z_des: Array, z_obs: Array) -> Array:
    """CBF safe controller for the drone

    Args:
        cbf (CBF): Control barrier function for the drone
        z (Array): State of the point-mass reduced model of the drone
            (position + velocity), shape (6,)
        z_des (Array): Desired state of the point-mass reduced model of the drone
            (position + velocity), shape (6,)
        z_obs (Array): State of the obstacle (position + velocity), shape (6,)

    Returns:
        Array: Velocity control input, shape (3,)
    """
    u = nominal_controller(z, z_des)
    return cbf.safety_filter(z, u, z_obs)


class CBFNode(Node):
    """ROS2 node for the CBF drone controller

    Args:
        cbf_config (CBFConfig): Configuration for the CBF safety filter
        z_des (ArrayLike, optional): Desired state of the point-mass reduced model of the drone
            (position + velocity), shape (6,). Defaults to (0, 0, -1, 0, 0, 0).
        vel_buffer_depth (int, optional): Depth of the buffer for filtering the obstacle velocity.
        control_freq (float, optional): Control frequency of the drone. Defaults to 100 Hz.
    """

    def __init__(
        self,
        cbf_config: CBFConfig,
        z_des: ArrayLike = (0, 0, -1.25, 0, 0, 0),
        vel_buffer_depth: int = 5,
        control_freq: float = 100,
    ):
        super().__init__(NODE_NAME)
        assert isinstance(cbf_config, CBFConfig)
        self.cbf = CBF.from_config(cbf_config)
        self.z_des = jnp.asarray(z_des, dtype=jnp.float64)
        self.control_freq = control_freq
        # QoS profile for the publisher and subscribers
        # TODO decide on if the depth should be modified
        default_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        mocap_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Publisher: Sends velocity control commands
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, VELOCITY_SETPOINT_TOPIC, default_qos_profile
        )
        # Subscribers: Listens to state of the drone and the obstacle
        self.vehicle_odometry_sub = self.create_subscription(
            VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self.vehicle_odometry_callback,
            default_qos_profile,
        )
        self.obstacle_mocap_sub = self.create_subscription(
            PoseStamped,
            OBSTACLE_MOCAP_TOPIC,
            self.mocap_callback,
            mocap_qos_profile,
        )
        # Set up cache for last known states of the drone and the obstacle
        # Also, store a buffer for filtering the obstacle velocity
        self.last_z = None
        # self.last_z_time = None
        self.velocity_buffer = deque(maxlen=vel_buffer_depth)
        self.last_obstacle_time = None
        self.last_obstacle_position = None
        self.last_z_obs = None

        # Publish the control input at the desired frequency
        self.timer = self.create_timer(1 / self.control_freq, self.publish_control)

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """Callback when we have new information on the state of the drone"""
        self.last_z = np.concatenate([msg.position, msg.velocity])
        # self.last_z_time = msg.timestamp / 1e6
        # self.publish_control()

    def mocap_callback(self, msg: PoseStamped):
        """Callback when we have new information on the state of the obstacle"""
        # Filter the velocity because we only have an instantaneous position measurement
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # NOTE: Mocap has a different frame convention than the drone
        # We'll update the obstacle position to match the drone frame
        # This just involves inverting the y and z coordinates
        # NOTE NEW!! Field bay has y up. X is the same, though !!!!!!!!!!!!!!!!!!!!
        # Flight room was (x, -y, -z)
        current_position = np.array(
            [msg.pose.position.x, msg.pose.position.z, -msg.pose.position.y]
        )
        if (
            self.last_obstacle_position is not None
            and self.last_obstacle_time is not None
        ):
            delta_time = current_time - self.last_obstacle_time
            delta_position = current_position - self.last_obstacle_position
            instantaneous_velocity = delta_position / delta_time
            self.velocity_buffer.append(instantaneous_velocity)
        else:
            self.velocity_buffer.append(np.zeros(3))
        filtered_velocity = np.mean(self.velocity_buffer, axis=0)

        # Update stored parameters
        self.last_obstacle_time = current_time
        self.last_obstacle_position = current_position
        self.last_z_obs = np.concatenate([current_position, filtered_velocity])
        # self.publish_control()

    def publish_control(self):
        """Publish the CBF safe velocity control input to the drone"""
        if self.last_z is None or self.last_z_obs is None:
            return
        velocity = safe_controller(
            self.cbf,
            jnp.asarray(self.last_z, dtype=jnp.float64),
            jnp.asarray(self.z_des, dtype=jnp.float64),
            jnp.asarray(self.last_z_obs, dtype=jnp.float64),
        ).__array__()
        position = np.array([np.nan, np.nan, np.nan])
        msg = TrajectorySetpoint(
            position=position,
            velocity=velocity,
            timestamp=int(self.get_clock().now().nanoseconds / 1000),
        )
        self.setpoint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CBFNode(DroneConfig())
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
