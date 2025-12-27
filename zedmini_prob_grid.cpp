// Minimal standalone example: probability grid mapping with ZED Mini + OpenCV.
// Dependencies: ZED SDK, OpenCV (core/imgproc/highgui). Build example:
//   g++ -std=c++17 zedmini_prob_grid.cpp -o zedmini_prob_grid \ 
//       -I/usr/local/zed/include -L/usr/local/zed/lib -lsl_zed \
//       `pkg-config --cflags --libs opencv4`
// Run near-field only; adjust grid size/resolution for your field dimensions.

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>

struct Grid {
  int w, h;
  float res; // meters per cell
  std::vector<float> log_odds;
  float l_hit = 0.7f, l_miss = -0.4f, l_decay = 0.98f;

  Grid(int w_, int h_, float res_) : w(w_), h(h_), res(res_), log_odds(w_ * h_, 0.0f) {}
  int idx(int x, int y) const { return y * w + x; }
  void decay() { for (auto &v : log_odds) v *= l_decay; }
  void updateCell(int x, int y, bool hit) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    log_odds[idx(x, y)] += hit ? l_hit : l_miss;
  }
  bool bestCell(int &bx, int &by, float &score) const {
    score = -std::numeric_limits<float>::infinity();
    bool found = false;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        float v = log_odds[idx(x, y)];
        if (v > score) {
          score = v;
          bx = x;
          by = y;
          found = true;
        }
      }
    }
    return found;
  }
};

bool detectBall(const cv::Mat &bgr, cv::Point &px_center) {
  cv::Mat hsv, mask;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  // Example threshold for orange ball; tune for lighting/ball color.
  cv::inRange(hsv, cv::Scalar(5, 120, 120), cv::Scalar(25, 255, 255), mask);
  std::vector<std::vector<cv::Point>> cnts;
  cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  static constexpr double MIN_BALL_AREA = 80.0;
  double bestArea = 0;
  cv::Point best(0, 0);
  for (auto &c : cnts) {
    double a = cv::contourArea(c);
    if (a > MIN_BALL_AREA && a > bestArea) {
      bestArea = a;
      cv::Moments m = cv::moments(c);
      if (m.m00 != 0) {
        best = {int(m.m10 / m.m00), int(m.m01 / m.m00)};
      }
    }
  }
  if (bestArea == 0) return false;
  px_center = best;
  return true;
}

int main() {
  static constexpr float MIN_VALID_DEPTH = 0.2f;
  static constexpr float MAX_VALID_DEPTH = 8.0f;
  static constexpr float CAMERA_HEIGHT = 0.45f;      // meters above ground
  static constexpr float CAMERA_PITCH_RAD = 0.0f;    // downward-positive pitch
  static constexpr float MIN_RAY_DOWNWARD_COMPONENT = 1e-3f;
  static constexpr int DRAW_RADIUS = 6;
  static constexpr int DRAW_THICKNESS = 2;
  const cv::Scalar DRAW_COLOR(0, 255, 0);

  sl::Camera zed;
  sl::InitParameters p;
  p.camera_resolution = sl::RESOLUTION::HD720;
  p.depth_mode = sl::DEPTH_MODE::PERFORMANCE; // faster, lower depth accuracy; switch to sl::DEPTH_MODE::ULTRA if precision is needed
  p.coordinate_units = sl::UNIT::METER;
  p.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  if (zed.open(p) != sl::ERROR_CODE::SUCCESS) {
    std::cerr << "Failed to open ZED\n";
    return 1;
  }

  auto calib = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
  float fx = calib.fx, fy = calib.fy, cx = calib.cx, cy = calib.cy;

  // 10m x 10m area centered laterally; forward is +Z.
  Grid grid(200, 200, 0.05f);
  sl::Mat zedImage, zedDepth;

  std::cout << "Press ESC to quit.\n";
  while (true) {
    if (zed.grab() != sl::ERROR_CODE::SUCCESS) continue;
    zed.retrieveImage(zedImage, sl::VIEW::LEFT);
    zed.retrieveMeasure(zedDepth, sl::MEASURE::DEPTH);

    cv::Mat rgba(zedImage.getHeight(), zedImage.getWidth(), CV_8UC4, zedImage.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_BGRA2BGR);

    grid.decay();
    cv::Point px;
    bool seen = detectBall(bgr, px);
    if (seen) {
      float depth = 0.0f;
      sl::ERROR_CODE err = zedDepth.getValue(px.x, px.y, &depth);
      if (err == sl::ERROR_CODE::SUCCESS && std::isfinite(depth) && depth > MIN_VALID_DEPTH && depth < MAX_VALID_DEPTH) {
        float X = (px.x - cx) * depth / fx; // right
        float Y = (px.y - cy) * depth / fy; // down
        float Z = depth;                    // forward

        // Apply pitch about camera X (right) axis to account for mounting angle.
        float cosP = std::cos(CAMERA_PITCH_RAD);
        float sinP = std::sin(CAMERA_PITCH_RAD);
        float Yr = cosP * Y - sinP * Z;
        float Zr = sinP * Y + cosP * Z;
        float Xr = X;

        // Intersect ray with ground plane at camera height; skip if ray is parallel or pointing upward.
        if (Yr > MIN_RAY_DOWNWARD_COMPONENT) {
          float scale = CAMERA_HEIGHT / Yr;
          float ground_x = Xr * scale; // lateral
          float ground_z = Zr * scale; // forward
          int gx = int(ground_x / grid.res) + grid.w / 2;
          int gy = int(ground_z / grid.res);
          grid.updateCell(gx, gy, true);
        }
        cv::circle(bgr, px, DRAW_RADIUS, DRAW_COLOR, DRAW_THICKNESS);
      }
    }

    int bx = 0, by = 0;
    float score = 0;
    if (grid.bestCell(bx, by, score)) {
      float target_x = (bx - grid.w / 2) * grid.res;
      float target_z = by * grid.res;
      std::cout << "Target cell (" << target_x << "m lateral, " << target_z
                << "m forward), score=" << score << "\n";
    }

    cv::imshow("ZED Ball Detection", bgr);
    if (cv::waitKey(1) == 27) break; // ESC
  }

  zed.close();
  return 0;
}
