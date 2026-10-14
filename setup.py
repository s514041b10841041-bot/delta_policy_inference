from glob import glob

from setuptools import find_packages, setup

package_name = "ros2_policy_inference"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    # Policies load their config.yaml from next to their own module
    # (os.path.dirname(__file__)); without this, colcon installs only the .py
    # files and the policy fails to import.
    package_data={package_name: ["policies/*/config.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Zhong-Zi-Heng",
    maintainer_email="ximen0806449@gmail.com",
    description=(
        "Standalone ROS2 node that subscribes to real-robot camera/"
        "proprioception topics, preprocesses them for an IL/VLA policy, "
        "runs inference, and publishes the predicted EE-offset action."
    ),
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = ros2_policy_inference.inference_node:main",
        ],
    },
)
