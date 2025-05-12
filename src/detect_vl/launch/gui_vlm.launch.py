# File: gui_vlm.launch.py
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Launch the PyQt GUI
    gui = ExecuteProcess(
        cmd=[
            '/bin/python3',
            '/home/tamir/autonomousService/src/GUI/robot_chat_gui.py'
        ],
        output='screen'
    )

    # Launch the VLM detection node in GUI mode
    vlm = Node(
        package='detect_vl',
        executable='start_service',
        name='detect_vl_node',
        output='screen',
        arguments=['--use-gui']
    )

    return LaunchDescription([
        gui,
        vlm
    ])
