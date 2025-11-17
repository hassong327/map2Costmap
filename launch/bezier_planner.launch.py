from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bezier_path_planner',
            executable='bezier_planner_node',
            name='bezier_planner_node',
            output='screen',
            parameters=[
                # 필요하면 여기서 파라미터 추가
            ]
        )
    ])
