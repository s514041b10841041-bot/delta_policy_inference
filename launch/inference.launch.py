import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("ros2_policy_inference")
    params_file = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("policy_name", default_value="dummy"),
        DeclareLaunchArgument(
            "checkpoint", default_value="",
            description="Checkpoint folder for act/depth_act/diffusion*/depth_inference*/diffusion_depth_only*, e.g. diffusion_50k; "
                        "empty = the policy's config.yaml checkpoint_dir"),
        DeclareLaunchArgument(
            "reset_on_handoff", default_value="true",
            description="Reset the policy and its gripper state when the sim's IL grasp step starts"),
        DeclareLaunchArgument(
            "gradcam_enable", default_value="false",
            description="Publish Grad-CAM overlays of what the policy looks at (gradcam_topic)"),
        DeclareLaunchArgument(
            "gradcam_target", default_value="motion",
            description="Action dims Grad-CAM explains: motion, position, rotation, gripper or all"),
        Node(
            package="ros2_policy_inference",
            executable="inference_node",
            name="policy_inference_node",
            output="screen",
            parameters=[
                params_file,
                {
                    "policy_name": LaunchConfiguration("policy_name"),
                    "checkpoint": ParameterValue(LaunchConfiguration("checkpoint"), value_type=str),
                    "reset_on_handoff": ParameterValue(
                        LaunchConfiguration("reset_on_handoff"), value_type=bool),
                    "gradcam_enable": ParameterValue(LaunchConfiguration("gradcam_enable"), value_type=bool),
                    "gradcam_target": ParameterValue(LaunchConfiguration("gradcam_target"), value_type=str),
                },
            ],
        ),
    ])
