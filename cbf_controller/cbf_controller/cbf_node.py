from collections import deque
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from jax import Array
from cbfpy import CBF, CBFConfig

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry

jax.config.update("jax_enable_x64", True)

# TODO check on the exact topics and the types of each message

NODE_NAME = "cbf_node"
VEHICLE_ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
OBSTACLE_MOCAP_TOPIC = "/vrpn_mocap/obstacle/pose"
VEHICLE_SETPOINT_TOPIC = "/fmu/in/vehicle_setpoint"


class DroneConfig(CBFConfig):
    """Configuration for the SRC drone demo: Dynamic obstacle avoidance with drones and CBFs

    We assume that the drone is a point robot and that we have direct control over its velocity.

    i.e. z = [x, y, z, vx, vy, vz]
    and z_dot = f(z) + g(z)u = [0, 0, 0, 0, 0, 0] + [vx, vy, vz, 0, 0, 0]
    where u  = [vx, vy, vz]
    """

    def __init__(
        self,
        pos_min: ArrayLike = (-2, -2, -2),
        pos_max: ArrayLike = (2, 2, 2),
        drone_radius: float = 0.175,
        obstacle_radius: float = 0.04,
        padding: float = 0.15,
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
            num_barr=7,
            relative_degree=1,
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
        cbf (CBF): _description_
        z (Array): _description_
        z_des (Array): _description_
        z_obs (Array): _description_

    Returns:
        Array: _description_
    """
    u = nominal_controller(z, z_des)
    return cbf.safety_filter(z, u, z_obs)


class CBFNode(Node):

    def __init__(self, cbf_config: CBFConfig):
        super().__init__(NODE_NAME)
        assert isinstance(cbf_config, CBFConfig)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.cbf = CBF.from_config(cbf_config)
        # Publisher
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, VEHICLE_SETPOINT_TOPIC, qos_profile
        )
        # Subscribers
        self.vehicle_odometry_sub = self.create_subscription(
            VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self.vehicle_odometry_callback,
            qos_profile,
        )
        self.obstacle_mocap_sub = self.create_subscription(
            Pose, OBSTACLE_MOCAP_TOPIC, self.mocap_callback, qos_profile
        )
        self.z_des = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # TODO CHECK ON Z HEIGHT
        self.last_z = None
        self.velocity_buffer = deque(maxlen=5)
        self.last_obstacle_time = None
        self.last_obstacle_position = None
        self.last_z_obs = None

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """Callback when we have new information on the state of the drone"""
        self.last_z = np.concatenate([msg.position, msg.velocity])
        self.publish_control()

    def mocap_callback(self, msg: Pose):
        """Callback when we have new information on the state of the obstacle"""
        # Filter the velocity because we only have an instantaneous position measurement
        current_time = self.get_clock().now().to_msg()
        time_in_seconds = current_time.sec + current_time.nanosec * 1e-9
        current_position = np.array([msg.position.x, msg.position.y, msg.position.z])
        if self.last_obstacle_time is not None and self.last_obstacle_time is not None:
            delta_time = time_in_seconds - self.last_obstacle_time
            delta_position = current_position - self.last_obstacle_position
            instantaneous_velocity = delta_position / delta_time
            self.velocity_buffer.append(instantaneous_velocity)
        filtered_velocity = np.mean(self.velocity_buffer, axis=0)
        self.last_z_obs = np.concatenate([current_position, filtered_velocity])
        self.publish_control()

    def publish_control(self):
        if self.last_z is None or self.last_z_obs is None:
            return
        msg = TrajectorySetpoint(
            velocity=safe_controller(
                self.cbf, self.last_z, self.z_des, self.last_z_obs
            ).__array__()
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
