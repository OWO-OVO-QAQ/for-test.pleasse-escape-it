"""
Minimal Python example: ZED Mini probability grid mapping with OpenCV.
Dependencies:
  - ZED SDK Python API (pyzed-sl)
  - OpenCV (`pip install opencv-python`)

Run:
  python zedmini_prob_grid.py
"""

import math
import numpy as np
import cv2
import pyzed.sl as sl


class Grid:
  def __init__(self, w: int, h: int, res: float):
    self.w, self.h, self.res = w, h, res
    self.log_odds = np.zeros((h, w), dtype=np.float32)
    self.l_hit, self.l_miss, self.l_decay = 0.7, -0.4, 0.98

  def decay(self):
    self.log_odds *= self.l_decay

  def update(self, x: int, y: int, hit: bool):
    if 0 <= x < self.w and 0 <= y < self.h:
      self.log_odds[y, x] += self.l_hit if hit else self.l_miss

  def best(self):
    idx = np.unravel_index(np.argmax(self.log_odds), self.log_odds.shape)
    score = self.log_odds[idx]
    return (idx[1], idx[0], float(score))  # (x, y, score)


def _find_contours(mask):
  # OpenCV 4 returns (contours, hierarchy); OpenCV 3 returns (image, contours, hierarchy)
  out = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return out[1] if len(out) == 3 else out[0]


def detect_ball(bgr: np.ndarray):
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, (5, 120, 120), (25, 255, 255))  # orange example
  cnts = _find_contours(mask)
  MIN_BALL_AREA = 80.0
  best_area = 0.0
  best = None
  for c in cnts:
    area = cv2.contourArea(c)
    if area > MIN_BALL_AREA and area > best_area:
      m = cv2.moments(c)
      if m["m00"] != 0:
        best_area = area
        best = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
  return best


def is_valid_depth(depth_value, error_code, min_depth, max_depth):
  return error_code == sl.ERROR_CODE.SUCCESS and math.isfinite(depth_value) and min_depth < depth_value < max_depth


def main():
  MIN_VALID_DEPTH = 0.2
  MAX_VALID_DEPTH = 8.0
  CAMERA_HEIGHT = 0.45  # meters
  CAMERA_PITCH_RAD = 0.0  # downward-positive: +X rotation in RIGHT_HANDED_Z_UP_X_FWD tilts camera down
  MIN_RAY_DOWN = 1e-3
  DRAW_RADIUS, DRAW_THICKNESS = 6, 2
  DRAW_COLOR = (0, 255, 0)

  grid = Grid(200, 200, 0.05)  # 10m x 10m

  zed = sl.Camera()
  init_params = sl.InitParameters(
      camera_resolution=sl.RESOLUTION.HD720,
      depth_mode=sl.DEPTH_MODE.PERFORMANCE,  # faster; use ULTRA for higher accuracy
      coordinate_units=sl.UNIT.METER,
      coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
  )
  if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED")
    return

  cam_info = zed.get_camera_information()
  calib = cam_info.camera_configuration.calibration_parameters.left_cam
  fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

  image = sl.Mat()
  depth = sl.Mat()

  print("Press ESC to quit.")
  while True:
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
      continue

    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

    bgr = image.get_data()[:, :, :3].copy()
    grid.decay()

    px = detect_ball(bgr)
    if px:
      depth_value, error_code = depth.get_value(px[0], px[1])
      if is_valid_depth(depth_value, error_code, MIN_VALID_DEPTH, MAX_VALID_DEPTH):
        x_cam = (px[0] - cx) * depth_value / fx
        y_cam = (px[1] - cy) * depth_value / fy
        z_cam = depth_value

        cos_p, sin_p = math.cos(CAMERA_PITCH_RAD), math.sin(CAMERA_PITCH_RAD)
        y_r = cos_p * y_cam - sin_p * z_cam
        z_r = sin_p * y_cam + cos_p * z_cam
        x_r = x_cam

        if y_r > MIN_RAY_DOWN:
          scale = CAMERA_HEIGHT / y_r
          gx = int((x_r * scale) / grid.res) + grid.w // 2
          gy = int((z_r * scale) / grid.res)
          grid.update(gx, gy, True)
        cv2.circle(bgr, px, DRAW_RADIUS, DRAW_COLOR, DRAW_THICKNESS)

    bx, by, score = grid.best()
    target_x = (bx - grid.w // 2) * grid.res
    target_z = by * grid.res
    msg = f"Target: {target_x:.2f}m, {target_z:.2f}m, s={score:.2f}"
    print(msg)
    cv2.putText(bgr, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ZED Ball Detection", bgr)
    if cv2.waitKey(1) == 27:
      break

  zed.close()


if __name__ == "__main__":
  main()
