# RoboCup 人型（KidSize）决策与找球调研

本文档汇总可用于 RoboCup Humanoid KidSize 机器人决策（找球、踢球策略）的开源方案，并给出基于 ZED Mini 双目深度相机的概率栅格找球示例代码框架。

## 可用开源库对比

| 名称 | 领域/能力 | 语言 | 优势 | 劣势 | 入口 |
| --- | --- | --- | --- | --- | --- |
| ROS 2 + BehaviorTree.CPP + Nav2 | 决策/行为树、导航、任务调度 | C++ | 社区活跃、插件丰富，行为树便于可视化调试，Nav2 支持约束路径规划 | 需要裁剪以适配人形步态；配置成本高 | <https://github.com/BehaviorTree/BehaviorTree.CPP> |
| NUbots（RoboCup Humanoid 开源代码） | 视觉、定位、行为、步态，基于 NUClear 架构 | C++ | 针对 humanoid soccer，含球检测、策略与步态示例，可直接复用模块 | 工程较大，需时间熟悉 NUClear 框架 | <https://github.com/NUbots/NUbots> |
| Rhoban / WalkKick (Team Rhoban) | 步态与踢球控制 | C++ | 针对人形机器人行走与踢球，包含参数化 walk-kick | 聚焦运动控制，需要自行集成视觉/决策 | <https://github.com/rhoban/walk-pattern> |
| ZED SDK + zed-ros2-wrapper | 双目深度感知、相机标定、点云/深度流 | C++ | 对 ZED Mini 直接支持，ROS 2 插件输出彩色+深度+点云 | SDK 依赖 CUDA，嵌入式需注意算力 | <https://github.com/stereolabs/zed-ros2-wrapper> |
| OpenCV + ONNX/TensorRT 推理 | 视觉检测（球/机器人） | C++/Python | 轻量易用，可与深度数据融合 | 需自训练模型或调色阈值 | <https://opencv.org> |

> 建议：决策层采用 BehaviorTree.CPP（行为树便于策略扩展与可视化），感知层使用 ZED SDK/ROS2 Wrapper 提供的深度和彩色图，结合轻量球检测（阈值或小型 CNN）更新概率栅格；运动层可结合现有步态/踢球库（如 Rhoban walk-kick）。

## 概率栅格找球（ZED Mini）流程

1. **相机输入**：使用 ZED SDK 获取彩色图与深度图（或 ROS2 `/zed/*` 话题）。  
2. **球检测**：轻量检测（HSV/形状阈值）或小型 CNN（ONNX/TensorRT）。输出球像素位置与置信度。  
3. **3D 投影**：利用深度将像素反投影到相机坐标系；转换到机器人坐标系。  
4. **概率栅格更新**：对平面网格使用 log-odds 更新：命中区域提升概率，未命中区域衰减；加入时间衰减避免陈旧信息。  
5. **策略**：  
   - 栅格最高置信单元作为目标方向；  
   - 若全局低置信，切换搜索（原地旋转/行走扫描）；  
   - 目标高置信时切换接近与对齐踢球。

## 示例代码框架（C++，使用 ZED SDK + OpenCV）

```cpp
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct Grid {
  int w, h; float res;               // 分辨率 m/格
  std::vector<float> log_odds;       // 初值 0
  float l_hit = 0.7f, l_miss = -0.4f, l_decay = 0.98f;
  Grid(int w_, int h_, float res_) : w(w_), h(h_), res(res_), log_odds(w_*h_, 0.0f) {}
  int idx(int x, int y) const { return y * w + x; }
  void decay() { for (auto &v : log_odds) v *= l_decay; }
  void updateCell(int x, int y, bool hit) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    log_odds[idx(x,y)] += hit ? l_hit : l_miss;
  }
};

bool detectBall(const cv::Mat &bgr, cv::Point &px_center) {
  cv::Mat hsv, mask;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  // 示例：橙球阈值，根据实际调节
  cv::inRange(hsv, cv::Scalar(5,120,120), cv::Scalar(25,255,255), mask);
  std::vector<std::vector<cv::Point>> cnts;
  cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  double bestArea = 0; cv::Point best(0, 0);
  for (auto &c : cnts) {
    double a = cv::contourArea(c);
    if (a > 80 && a > bestArea) {
      bestArea = a;
      cv::Moments m = cv::moments(c);
      if (m.m00 != 0) {
        best = {int(m.m10/m.m00), int(m.m01/m.m00)};
      }
    }
  }
  if (bestArea == 0) return false;
  px_center = best;
  return true;
}

int main() {
  sl::Camera zed;
  sl::InitParameters p; p.camera_resolution = sl::RESOLUTION::VGA;
  p.depth_mode = sl::DEPTH_MODE::ULTRA; // 精度高，算力不足时可用 PERFORMANCE
  if (zed.open(p) != sl::ERROR_CODE::SUCCESS) return 1;

  Grid grid(120, 120, 0.05f); // 6m x 6m 覆盖区
  sl::Mat zedImage, zedDepth;
  auto calib = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
  float fx = calib.fx, fy = calib.fy, cx = calib.cx, cy = calib.cy;

  while (true) {
    if (zed.grab() != sl::ERROR_CODE::SUCCESS) continue;
    zed.retrieveImage(zedImage, sl::VIEW::LEFT);
    zed.retrieveMeasure(zedDepth, sl::MEASURE::DEPTH);
    cv::Mat bgr(zedImage.getHeight(), zedImage.getWidth(), CV_8UC4, zedImage.getPtr<sl::uchar1>(sl::MEM::CPU));
    cv::Point px;
    grid.decay();
    if (detectBall(bgr, px)) {
      float depth;
      zedDepth.getValue(px.x, px.y, &depth);
      if (std::isfinite(depth) && depth > 0.2f && depth < 6.0f) {
        // 相机系到机器人平面坐标（使用内参矩阵 K）
        float X = (px.x - cx) * depth / fx;
        float Y = (px.y - cy) * depth / fy;
        float Z = depth;
        int gx = int(X / grid.res) + grid.w/2;
        int gy = int(Z / grid.res);
        grid.updateCell(gx, gy, true);
      }
    }
    // 未命中区域可根据视锥投影衰减/标记 miss
    // 最高 log-odds 单元即当前目标方向
  }
}
```

> 说明：示例仅展示流程骨架，实际项目需使用相机内参矩阵将像素反投影到 3D，并将相机系转换到机器人基座系；可将栅格发布为 ROS2 `nav_msgs/OccupancyGrid` 供决策节点消费。

## 建议的模块化结构

- **感知节点**：ZED Wrapper 输出彩色+深度；球检测节点发布 `ball_pose`；概率栅格节点发布 `ball_grid`。  
- **决策节点（行为树）**：状态包含“搜索/转向/接近/对齐/踢球”；条件节点读取 `ball_grid` 置信度和机器人定位。  
- **运动节点**：行走与踢球动作调用现有步态/踢球库，带超时与失败回退。  

以上内容可直接落地到现有 ROS 2 项目或独立 C++ 代码，为 KidSize 找球与踢球提供可操作的起点。
