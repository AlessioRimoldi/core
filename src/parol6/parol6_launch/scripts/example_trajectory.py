#!/usr/bin/env python3
"""Send a simple trajectory to the joint_trajectory_controller.

Usage (inside the container):
  python3 /ros2_ws/core/src/parol6/parol6_launch/scripts/example_trajectory.py
"""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINTS = ["L1", "L2", "L3", "L4", "L5", "L6"]

# Sequence of waypoints: (positions_rad, time_from_start_sec)
WAYPOINTS = [
    ([0.5, 0.3, -0.3, 0.0, 0.0, 0.0], 2.0),
    ([0.5, 0.3, -0.5, 0.8, -0.5, 0.0], 4.0),
    ([-0.5, -0.3, 0.5, -0.8, 0.5, 1.0], 6.0),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8.0),
]


class TrajectoryClient(Node):
    def __init__(self):
        super().__init__("example_trajectory")
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

    def send(self):
        self.get_logger().info("Waiting for action server...")
        self._client.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS

        for positions, t in WAYPOINTS:
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.velocities = [0.0] * len(JOINTS)
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(pt)

        self.get_logger().info(f"Sending trajectory with {len(WAYPOINTS)} waypoints...")
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted, executing...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Trajectory completed successfully!")
        else:
            self.get_logger().error(f"Trajectory failed with error code: {result.error_code}")


def main():
    rclpy.init()
    node = TrajectoryClient()
    try:
        node.send()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
