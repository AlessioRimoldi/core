"""
PAROL6 launch file — starts ros2_control with MuJoCo sim or real hardware.

Usage:
  ros2 launch parol6_launch parol6.launch.py                          # defaults to sim
  ros2 launch parol6_launch parol6.launch.py hardware_interface_type:=real
  ros2 launch parol6_launch parol6.launch.py scene_file:=/path/to/scene.yaml
"""
import math
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
import yaml


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


def _ros2_control_block(hw_type: str, gains_file: str, mjcf_path: str,
                        serial_port: str, scene_file_path: str = "") -> str:
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
        if scene_file_path:
            params += f'      <param name="scene_file_path">{scene_file_path}</param>\n'
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


def _make_mjcf_from_urdf(urdf_path: str, mesh_dir: str, scene_file_path: str = "") -> str:
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

    # Inject scene objects into the MuJoCo URDF (not the RViz one)
    if scene_file_path:
        urdf_text = _inject_scene_objects(urdf_text, scene_file_path)

    # Write into the mesh directory so STL files are adjacent
    tmp_path = os.path.join(mesh_dir, "_parol6_mujoco.urdf")
    with open(tmp_path, "w") as f:
        f.write(urdf_text)
    return tmp_path


def _strip_legacy_blocks(urdf_text: str) -> str:
    """Remove ROS1-era <transmission>, <gazebo>, and Gazebo plugin blocks.

    These are vestigial from the SolidWorks URDF exporter and are unused
    in ros2_control.
    """
    import re
    urdf_text = re.sub(r'<transmission[\s\S]*?</transmission\s*>', '', urdf_text)
    urdf_text = re.sub(r'<gazebo[\s\S]*?</gazebo\s*>', '', urdf_text)
    return urdf_text


def _inject_scene_objects(urdf_text: str, scene_file_path: str) -> str:
    """Inject scene objects from YAML as MuJoCo bodies into the URDF.

    Each object becomes a link+joint in the URDF so MuJoCo's URDF
    parser creates corresponding bodies in the physics model.
    Objects with ``dynamic: true`` get a floating joint (6-DOF, affected
    by gravity and contacts); others get a fixed joint (welded to world).
    """
    with open(scene_file_path, "r") as f:
        scene = yaml.safe_load(f)

    if not scene or "objects" not in scene:
        return urdf_text

    # Remove closing </robot> tag — we'll re-add it after
    urdf_text = urdf_text.rstrip()
    if urdf_text.endswith("</robot>"):
        urdf_text = urdf_text[: -len("</robot>")]

    # Tell MuJoCo not to fuse static (fixed-joint) bodies into their parent,
    # so our scene objects remain as separate bodies we can track via mj_name2id.
    urdf_text += '\n  <mujoco><compiler fusestatic="false"/></mujoco>\n'

    for obj in scene["objects"]:
        name = obj["name"]
        pos = obj.get("position", [0, 0, 0])
        rpy = obj.get("orientation", [0, 0, 0])
        obj_type = obj["type"]
        dynamic = obj.get("dynamic", False)

        # Build geometry for the visual/collision
        if obj_type == "box":
            sx, sy, sz = obj["size"]
            geom = f'<geometry><box size="{sx*2} {sy*2} {sz*2}"/></geometry>'
        elif obj_type == "sphere":
            r = obj["radius"]
            geom = f'<geometry><sphere radius="{r}"/></geometry>'
        elif obj_type == "cylinder":
            r = obj["radius"]
            l = obj["length"]
            geom = f'<geometry><cylinder radius="{r}" length="{l}"/></geometry>'
        else:
            continue

        rgba = obj.get("color", [0.5, 0.5, 0.5, 1.0])
        mass = obj.get("mass", 0.1 if dynamic else 0.01)

        # Compute inertia from shape (uniform density approximation)
        if obj_type == "box":
            sx, sy, sz = [s * 2 for s in obj["size"]]  # full extents
            ixx = mass / 12.0 * (sy**2 + sz**2)
            iyy = mass / 12.0 * (sx**2 + sz**2)
            izz = mass / 12.0 * (sx**2 + sy**2)
        elif obj_type == "sphere":
            r = obj["radius"]
            ixx = iyy = izz = 2.0 / 5.0 * mass * r**2
        elif obj_type == "cylinder":
            r = obj["radius"]
            h = obj["length"]
            ixx = iyy = mass / 12.0 * (3 * r**2 + h**2)
            izz = mass / 2.0 * r**2
        else:
            ixx = iyy = izz = mass * 0.001

        inertial = (
            f'<inertial><mass value="{mass}"/>'
            f'<inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>'
            f'</inertial>'
        )

        joint_type = "floating" if dynamic else "fixed"

        urdf_text += f"""
  <link name="{name}">
    {inertial}
    <visual>
      {geom}
      <material name="{name}_mat">
        <color rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
      </material>
    </visual>
    <collision>
      {geom}
    </collision>
  </link>
  <joint name="{name}_joint" type="{joint_type}">
    <parent link="world"/>
    <child link="{name}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
  </joint>
"""

    urdf_text += "</robot>\n"
    return urdf_text


def _build_robot_description(urdf_path: str, hw_type: str, gains_file: str,
                              mjcf_path: str, serial_port: str,
                              scene_file_path: str = "") -> str:
    """Read upstream URDF, strip closing </robot>, append ros2_control block."""
    with open(urdf_path, "r") as f:
        urdf_text = f.read()

    # Strip legacy ROS1/Gazebo blocks that confuse some URDF parsers
    urdf_text = _strip_legacy_blocks(urdf_text)

    # Remove any existing </robot> closing tag (we'll re-add it)
    urdf_text = urdf_text.replace("</robot>", "")

    ros2_ctrl = _ros2_control_block(hw_type, gains_file, mjcf_path, serial_port,
                                     scene_file_path)
    return urdf_text + "\n" + ros2_ctrl + "\n</robot>\n"


# ── Launch setup (OpaqueFunction so we can read resolved args) ───────────────

def _launch_setup(context):
    hw_type = LaunchConfiguration("hardware_interface_type").perform(context)
    serial_port = LaunchConfiguration("serial_port").perform(context)
    scene_file = LaunchConfiguration("scene_file").perform(context)

    description_share = get_package_share_directory("parol6_description")
    launch_share = get_package_share_directory("parol6_launch")

    # Resolve scene file: bare filename is looked up in parol6_launch/config/
    if scene_file and not os.path.isabs(scene_file):
        scene_file = os.path.join(launch_share, "config", scene_file)

    gains_file = os.path.join(description_share, "config", "parol6_gains.yaml")
    controllers_file = os.path.join(launch_share, "config", "controllers.yaml")

    # Resolve the MJCF path (only used in sim mode)
    mjcf_path = ""
    if hw_type == "sim":
        mjcf_path = _make_mjcf_from_urdf(_UPSTREAM_URDF, _UPSTREAM_MESH_DIR,
                                          scene_file)

    # Build the URDF with ros2_control tags
    robot_description = _build_robot_description(
        _UPSTREAM_URDF, hw_type, gains_file, mjcf_path, serial_port,
        scene_file,
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

    rviz = LaunchConfiguration("rviz").perform(context)
    if rviz == "true":
        rviz_config = os.path.join(launch_share, "config", "parol6.rviz")
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
        )
        nodes.append(rviz_node)

    # Launch scene marker publisher when a scene file is provided
    if scene_file:
        marker_script = os.path.join(launch_share, "scripts", "scene_marker_publisher.py")
        marker_node = Node(
            package="parol6_launch",
            executable="scene_marker_publisher",
            parameters=[{"scene_file_path": scene_file}],
            output="screen",
        )
        nodes.append(marker_node)

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
            "rviz",
            default_value="true",
            description="Launch RViz2 with pre-configured robot view",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "scene_file",
            default_value="",
            description="Scene YAML filename (resolved from parol6_launch/config/) or absolute path",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
