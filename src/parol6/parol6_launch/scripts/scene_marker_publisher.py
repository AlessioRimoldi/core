#!/usr/bin/env python3
"""Publish RViz MarkerArray for scene objects defined in a YAML file.

Reads object definitions (shape, size, color) from the scene YAML and listens
to TF for each object's pose.  Publishes a visualization_msgs/MarkerArray on
/scene_markers at 10 Hz.
"""
import yaml

import rclpy
import rclpy.time
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


class SceneMarkerPublisher(Node):
    def __init__(self):
        super().__init__("scene_marker_publisher")
        self.declare_parameter("scene_file_path", "")
        scene_path = self.get_parameter("scene_file_path").get_parameter_value().string_value

        if not scene_path:
            self.get_logger().error("No scene_file_path parameter provided")
            return

        with open(scene_path, "r") as f:
            scene = yaml.safe_load(f)

        self.objects = scene.get("objects", [])
        if not self.objects:
            self.get_logger().warn("No objects found in scene file")
            return

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(MarkerArray, "/scene_markers", 10)
        self.timer = self.create_timer(0.1, self.publish_markers)  # 10 Hz

        self.get_logger().info(f"Scene marker publisher started with {len(self.objects)} objects")

    def publish_markers(self):
        marker_array = MarkerArray()

        for idx, obj in enumerate(self.objects):
            name = obj["name"]

            marker = Marker()
            marker.header.frame_id = name
            # stamp=0 tells RViz to use the latest available TF
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = "scene_objects"
            marker.id = idx
            marker.action = Marker.ADD

            obj_type = obj["type"]
            if obj_type == "box":
                marker.type = Marker.CUBE
                sx, sy, sz = obj["size"]
                marker.scale.x = sx * 2.0
                marker.scale.y = sy * 2.0
                marker.scale.z = sz * 2.0
            elif obj_type == "sphere":
                marker.type = Marker.SPHERE
                r = obj["radius"]
                marker.scale.x = r * 2.0
                marker.scale.y = r * 2.0
                marker.scale.z = r * 2.0
            elif obj_type == "cylinder":
                marker.type = Marker.CYLINDER
                r = obj["radius"]
                l = obj["length"]
                marker.scale.x = r * 2.0
                marker.scale.y = r * 2.0
                marker.scale.z = l
            else:
                continue

            rgba = obj.get("color", [0.5, 0.5, 0.5, 1.0])
            marker.color.r = float(rgba[0])
            marker.color.g = float(rgba[1])
            marker.color.b = float(rgba[2])
            marker.color.a = float(rgba[3])

            # Identity pose — the marker frame is the object frame from TF
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SceneMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
