"""
PAROL6 launch file — starts ros2_control with MuJoCo sim or real hardware.

Usage:
  ros2 launch parol6_launch parol6.launch.py                          # defaults to sim
  ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── URDF generation ──────────────────────────────────────────────────────────
# The upstream PAROL6 URDF (from PCrnjak) has no ros2_control tags and uses
# ROS1-style transmissions.  We inject the <ros2_control> block at launch time
# so the same URDF works for both sim and real.

# Path inside the Docker container where the upstream URDF lives
_UPSTREAM_URDF = (
    "/ros2_ws/core/thirdparty/parol6_upstream/"
    "PAROL6_URDF/PAROL6/urdf/PAROL6.urdf"
)

_UPSTREAM_MESH_DIR = (
    "/ros2_ws/core/thirdparty/parol6_upstream/"
    "PAROL6_URDF/PAROL6/meshes"
)

_JOINTS = ["L1", "L2", "L3", "L4", "L5", "L6"]


def _ros2_control_block(hw_type: str, gains_file: str, mjcf_path: str, serial_port: str) -> str:
    """Return the <ros2_control> XML block for the chosen backend."""
    joint_xml = ""
    for j in _JOINTS:
        joint_xml += f"""
    <joint name="{j}">
      <command_interface name="position"/>
      <command_interface name="velocity"/>
      <command_interface name="effort"/>
      <command_interface name="p_gain"/>
      <command_interface name="d_gain"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>"""

    params = f'<param name="hardware_interface_type">{hw_type}</param>\n'
    params += f'      <param name="gains_file_path">{gains_file}</param>\n'
    if hw_type == "sim":
        params += f'      <param name="mjcf_file_path">{mjcf_path}</param>\n'
    else:
        params += f'      <param name="serial_port">{serial_port}</param>\n'
        params += '      <param name="baudrate">3000000</param>\n'

    return f"""
  <ros2_control name="parol6_hardware" type="system">
    <hardware>
      <plugin>parol6_hardware_interface/Parol6HardwareInterface</plugin>
      {params}
    </hardware>
    {joint_xml}
  </ros2_control>"""


def _make_mjcf_from_urdf(urdf_path: str, mesh_dir: str) -> str:
    """Create a patched URDF in the mesh directory for MuJoCo loading.

    MuJoCo's URDF parser strips directory components from mesh ``filename``
    attributes and resolves them relative to the model file.  We replace the
    ``package://`` URIs with bare filenames and write the patched URDF into
    the mesh directory so MuJoCo finds the STL files alongside it.
    """
    with open(urdf_path, "r") as f:
        urdf_text = f.read()

    # Strip to bare filenames — MuJoCo resolves relative to the model file
    urdf_text = urdf_text.replace("package://parol6/meshes/", "")

    # Write into the mesh directory so STL files are adjacent
    tmp_path = os.path.join(mesh_dir, "_parol6_mujoco.urdf")
    with open(tmp_path, "w") as f:
        f.write(urdf_text)
    return tmp_path


def _strip_legacy_blocks(urdf_text: str) -> str:
    """Remove ROS1-era <transmission>, <gazebo>, and Gazebo plugin blocks.

    These are vestigial from the SolidWorks URDF exporter and can confuse
    Foxglove's URDF parser (and are unused in ros2_control).
    """
    import re
    urdf_text = re.sub(r'<transmission[\s\S]*?</transmission\s*>', '', urdf_text)
    urdf_text = re.sub(r'<gazebo[\s\S]*?</gazebo\s*>', '', urdf_text)
    return urdf_text


def _build_robot_description(urdf_path: str, hw_type: str, gains_file: str,
                              mjcf_path: str, serial_port: str) -> str:
    """Read upstream URDF, strip closing </robot>, append ros2_control block."""
    with open(urdf_path, "r") as f:
        urdf_text = f.read()

    # Strip legacy ROS1/Gazebo blocks that confuse some URDF parsers
    urdf_text = _strip_legacy_blocks(urdf_text)

    # Remove any existing </robot> closing tag (we'll re-add it)
    urdf_text = urdf_text.replace("</robot>", "")

    ros2_ctrl = _ros2_control_block(hw_type, gains_file, mjcf_path, serial_port)
    return urdf_text + "\n" + ros2_ctrl + "\n</robot>\n"


# ── Launch setup (OpaqueFunction so we can read resolved args) ───────────────

def _launch_setup(context):
    hw_type = LaunchConfiguration("hardware_interface_type").perform(context)
    serial_port = LaunchConfiguration("serial_port").perform(context)

    description_share = get_package_share_directory("parol6_description")
    launch_share = get_package_share_directory("parol6_launch")

    gains_file = os.path.join(description_share, "config", "parol6_gains.yaml")
    controllers_file = os.path.join(launch_share, "config", "controllers.yaml")

    # Resolve the MJCF path (only used in sim mode)
    mjcf_path = ""
    if hw_type == "sim":
        mjcf_path = _make_mjcf_from_urdf(_UPSTREAM_URDF, _UPSTREAM_MESH_DIR)

    # Build the URDF with ros2_control tags
    robot_description = _build_robot_description(
        _UPSTREAM_URDF, hw_type, gains_file, mjcf_path, serial_port
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            controllers_file,
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # Sequence: wait for joint_state_broadcaster before spawning trajectory controller
    delay_jtc = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_trajectory_controller_spawner],
        )
    )

    nodes = [
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        delay_jtc,
    ]

    foxglove = LaunchConfiguration("foxglove").perform(context)
    if foxglove == "true":
        foxglove_bridge = Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            parameters=[{
                "port": 8765,
                "address": "0.0.0.0",
                "capabilities": ["clientPublish", "connectionGraph", "assets"],
                "asset_uri_allowlist": ["^package://.*"],
            }],
            output="screen",
        )
        nodes.append(foxglove_bridge)

    return nodes


# ── Entry point ──────────────────────────────────────────────────────────────

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "hardware_interface_type",
            default_value="sim",
            description="Backend type: 'sim' (MuJoCo) or 'real' (PAROL6 serial)",
            choices=["sim", "real"],
        ),
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyACM0",
            description="Serial device path (only used when hardware_interface_type:=real)",
        ),
        DeclareLaunchArgument(
            "foxglove",
            default_value="true",
            description="Launch Foxglove Bridge for visualization",
            choices=["true", "false"],
        ),
        OpaqueFunction(function=_launch_setup),
    ])
