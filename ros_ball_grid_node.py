"""
ROS 2 Humble node: probability grid fusion of ball detections (pixel coords + depth).

Assumptions:
- Subscribe to topic `/ball_detection` with message geometry_msgs/msg/Point
  (x = pixel u, y = pixel v, z = depth in meters). If depth is unavailable,
  publish z<=0 to skip update.
- Camera intrinsics provided via parameters fx, fy, cx, cy.
- Camera height/pitch provided via parameters camera_height (m), camera_pitch_rad (downward-positive).
- Grid size/resolution configurable; log-odds updated per detection with decay.

Run:
  ros2 run your_pkg ros_ball_grid_node.py
Or directly:
  python ros_ball_grid_node.py --ros-args -p fx:=700.0 -p fy:=700.0 -p cx:=640.0 -p cy:=360.0
"""

import math
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header


class Grid:
  def __init__(self, w: int, h: int, res: float, l_hit=0.7, l_miss=-0.4, l_decay=0.98):
    self.w, self.h, self.res = w, h, res
    self.log_odds = np.zeros((h, w), dtype=np.float32)
    self.l_hit, self.l_miss, self.l_decay = l_hit, l_miss, l_decay

  def decay(self):
    self.log_odds *= self.l_decay

  def update(self, x: int, y: int, hit: bool):
    if 0 <= x < self.w and 0 <= y < self.h:
      self.log_odds[y, x] += self.l_hit if hit else self.l_miss

  def best(self) -> Tuple[int, int, float]:
    idx = np.unravel_index(np.argmax(self.log_odds), self.log_odds.shape)
    score = float(self.log_odds[idx])
    return idx[1], idx[0], score

  def to_occupancy(self) -> list:
    probs = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))  # logistic
    return (probs * 100).clip(0, 100).astype(np.int8).flatten().tolist()


class BallGridNode(Node):
  def __init__(self):
    super().__init__("ball_grid_node")

    # Parameters
    self.declare_parameter("fx", 700.0)
    self.declare_parameter("fy", 700.0)
    self.declare_parameter("cx", 640.0)
    self.declare_parameter("cy", 360.0)
    self.declare_parameter("camera_height", 0.45)
    self.declare_parameter("camera_pitch_rad", 0.0)  # downward-positive
    self.declare_parameter("grid_w", 200)
    self.declare_parameter("grid_h", 200)
    self.declare_parameter("grid_res", 0.05)
    self.declare_parameter("min_depth", 0.2)
    self.declare_parameter("max_depth", 8.0)
    self.declare_parameter("decay_rate", 0.98)
    self.declare_parameter("update_rate_hz", 10.0)

    self.fx = self.get_parameter("fx").value
    self.fy = self.get_parameter("fy").value
    self.cx = self.get_parameter("cx").value
    self.cy = self.get_parameter("cy").value
    self.camera_height = self.get_parameter("camera_height").value
    self.camera_pitch = self.get_parameter("camera_pitch_rad").value
    self.min_depth = self.get_parameter("min_depth").value
    self.max_depth = self.get_parameter("max_depth").value

    w = int(self.get_parameter("grid_w").value)
    h = int(self.get_parameter("grid_h").value)
    res = float(self.get_parameter("grid_res").value)
    decay_rate = float(self.get_parameter("decay_rate").value)

    self.grid = Grid(w, h, res, l_decay=decay_rate)
    self.get_logger().info(f"Grid {w}x{h} res={res}m; decay={decay_rate}")

    self.sub = self.create_subscription(Point, "/ball_detection", self.on_ball, 10)
    self.pub_grid = self.create_publisher(OccupancyGrid, "/ball_grid", 1)

    update_rate = float(self.get_parameter("update_rate_hz").value)
    self.create_timer(1.0 / update_rate, self.publish_grid)

  def on_ball(self, msg: Point):
    # msg.x, msg.y = pixel u,v ; msg.z = depth (m). If depth<=0, skip update.
    depth = msg.z
    if depth <= 0 or not math.isfinite(depth):
      return
    if not (self.min_depth < depth < self.max_depth):
      return

    x_cam = (msg.x - self.cx) * depth / self.fx
    y_cam = (msg.y - self.cy) * depth / self.fy
    z_cam = depth

    cos_p = math.cos(self.camera_pitch)
    sin_p = math.sin(self.camera_pitch)
    y_r = cos_p * y_cam - sin_p * z_cam
    z_r = sin_p * y_cam + cos_p * z_cam
    x_r = x_cam

    if y_r <= 1e-3:
      return
    scale = self.camera_height / y_r
    ground_x = x_r * scale
    ground_z = z_r * scale

    gx = int(ground_x / self.grid.res) + self.grid.w // 2
    gy = int(ground_z / self.grid.res)
    self.grid.decay()
    self.grid.update(gx, gy, True)

  def publish_grid(self):
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.info.resolution = self.grid.res
    msg.info.width = self.grid.w
    msg.info.height = self.grid.h
    # Origin: center laterally, origin at robot base; shift so grid x=0 at left edge
    msg.info.origin.position.x = -self.grid.w * self.grid.res * 0.5
    msg.info.origin.position.y = 0.0
    msg.info.origin.position.z = 0.0
    msg.data = self.grid.to_occupancy()
    self.pub_grid.publish(msg)


def main(args=None):
  rclpy.init(args=args)
  node = BallGridNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
