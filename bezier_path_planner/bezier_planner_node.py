#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from nav2_msgs.msg import Costmap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped

# 기존
from .utils import (
    plan_bezier_from_start_goal,
    bezier_eval,
    Obstacle,
)



class CostmapBezierPlanner(Node):
    def __init__(self):
        super().__init__('costmap_bezier_planner')

        # ===== 파라미터 =====
        self.declare_parameter('costmap_topic', '/global_costmap/costmap_raw')
        self.declare_parameter('planner_path_topic', 'planner/path')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('degree', 3)            # 기본은 cubic
        self.declare_parameter('cost_threshold', 50)   # occupancy/cost 임계값

        costmap_topic = self.get_parameter('costmap_topic').get_parameter_value().string_value
        planner_path_topic = self.get_parameter('planner_path_topic').get_parameter_value().string_value
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value

        # ===== Subscriber / Publisher =====
        self.costmap_sub = self.create_subscription(
            Costmap,
            costmap_topic,
            self.costmap_callback,
            10
        )

        self.clicked_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            planner_path_topic,
            10
        )

        self.costmap = None
        self.costmap_meta = None

        # nav_msgs/OccupancyGrid 비슷한 래퍼 (QP 코드의 og 인자로 넘기기 위함)
        self.og = None

        # QP 쪽에서 필요로 하는 기하학 장애물 리스트
        # (일단은 빈 리스트로 두고, 나중에 costmap에서 추출해서 채우면 됨)
        self.obstacles: List[Obstacle] = []

        # RViz에서 클릭한 점 저장 (2개: start, goal)
        self.clicked_points: List[Tuple[float, float]] = []

        self.get_logger().info(
            f'CostmapBezierPlanner initialized. '
            f'Sub: {costmap_topic}, /clicked_point, '
            f'Pub: {planner_path_topic}'
        )

        # 주기적으로 plan 시도
        self.timer = self.create_timer(0.1, self.timer_callback)

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------
    def costmap_callback(self, msg: Costmap):
        self.costmap_meta = msg.metadata
        data = np.array(msg.data, dtype=np.int8)
        self.costmap = data.reshape(
            (msg.metadata.size_y, msg.metadata.size_x)
        )

        # QP 코드에서 사용하는 형태로 OccupancyGrid 비슷하게 감싸기
        class Info:  # 최소 필드만
            pass
        info = Info()
        info.width = msg.metadata.size_x
        info.height = msg.metadata.size_y
        info.resolution = msg.metadata.resolution
        info.origin = msg.metadata.origin

        class OG:
            pass
        og = OG()
        og.info = info
        og.data = msg.data  # 1D list

        self.og = og

        self.get_logger().info('Costmap received')

    def clicked_point_callback(self, msg: PointStamped):
        if msg.header.frame_id != self.global_frame:
            self.get_logger().warn(
                f'clicked_point frame_id={msg.header.frame_id}, '
                f'global_frame={self.global_frame} (변환 필요할 수도 있음)'
            )

        x = msg.point.x
        y = msg.point.y

        self.clicked_points.append((x, y))
        if len(self.clicked_points) > 2:
            self.clicked_points = self.clicked_points[-2:]

        self.get_logger().info(
            f'Clicked point ({x:.3f}, {y:.3f}) - total {len(self.clicked_points)}'
        )

    def timer_callback(self):
        self.get_logger().info('timer_callback called')

        if self.costmap is None or self.costmap_meta is None or self.og is None:
            self.get_logger().warn('costmap / og not ready yet')
            return
        if len(self.clicked_points) < 2:
            self.get_logger().warn(f'clicked_points < 2 (len={len(self.clicked_points)})')
            return

        start = self.clicked_points[0]
        goal = self.clicked_points[1]

        degree = self.get_parameter('degree').get_parameter_value().integer_value
        if degree < 1:
            self.get_logger().warn('degree < 1 인 경우는 의미가 없으므로 1로 강제 설정')
            degree = 1

        # ======= 🔴 여기서부터가 핵심 변경 부분 =======
        # start, goal, degree → 직선 위 control points 만들고
        # recursive QP 기반 push-away로 장애물/맵을 피하게 수정
        ctrl_final, intervals, info = plan_bezier_from_start_goal(
            start=start,
            goal=goal,
            degree=degree,
            obstacles=self.obstacles,   # 아직 없으면 [], 나중에 채워넣기
            og=self.og,                 # /global_costmap 기반 OccupancyGrid 래퍼
            occ_th=50,                  # cost_threshold와 맞춰줌
            plot=False,
            verbose=False,
        )

        # 곡선을 샘플링해서 Path로 변환
        ts = np.linspace(0.0, 1.0, 80)
        pts = [bezier_eval(ctrl_final, float(t)) for t in ts]
        pts_np = np.array(pts, dtype=float)

        path_msg = self.build_path_msg(pts_np)
        self.get_logger().info(f'Publishing path with {len(path_msg.poses)} poses '
                               f'(status={info.get("status", "")})')
        self.path_pub.publish(path_msg)
        # ======= 🔴 여기까지 변경 =======

    # ------------------------------------------------------------------
    # Path 생성 유틸
    # ------------------------------------------------------------------
    def build_path_msg(self, points_xy: np.ndarray) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.global_frame

        for x, y in points_xy:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # yaw=0
            path.poses.append(pose)

        return path


def main(args=None):
    rclpy.init(args=args)
    node = CostmapBezierPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
