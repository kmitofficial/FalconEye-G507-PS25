#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>


// ---------------- CONFIG ----------------
constexpr int FRAME_WIDTH  = 640;
constexpr int FRAME_HEIGHT = 480;

constexpr float X_DEADZONE_RATIO = 0.08f;   // 8% frame width
constexpr float AREA_TOLERANCE   = 0.15f;   // ±15%

constexpr float KP_X = 0.003f;   // yaw gain
constexpr float KP_D = 0.8f;     // distance gain

constexpr float MAX_LINEAR  = 1.0f;
constexpr float MAX_ANGULAR = 0.8f;

constexpr int CONTROL_HZ = 30;
// ---------------------------------------

// Bounding box
struct BBox {
    float x, y, w, h;
};

// Motor command
struct MotorCmd {
    float left;
    float right;
};

// Clamp helper
float clamp(float v, float min_v, float max_v) {
    return std::max(min_v, std::min(v, max_v));
}

// Compute bbox center X
float bbox_center_x(const BBox& b) {
    return b.x + b.w / 2.0f;
}

// Compute bbox area
float bbox_area(const BBox& b) {
    return b.w * b.h;
}

// ---------------- CONTROL CORE ----------------
MotorCmd compute_control(
    const BBox& bbox,
    float ref_area
) {
    float frame_center_x = FRAME_WIDTH / 2.0f;
    float obj_center_x   = bbox_center_x(bbox);
    float dx             = obj_center_x - frame_center_x;

    float x_deadzone = FRAME_WIDTH * X_DEADZONE_RATIO;

    // -------- YAW CONTROL (CENTERING) --------
    float omega = 0.0f;
    if (std::fabs(dx) > x_deadzone) {
        omega = KP_X * dx;
    }

    omega = clamp(omega, -MAX_ANGULAR, MAX_ANGULAR);

    // -------- DISTANCE CONTROL (AREA) --------
    float curr_area  = bbox_area(bbox);
    float area_ratio = curr_area / ref_area;

    float v = 0.0f;

    if (area_ratio < (1.0f - AREA_TOLERANCE)) {
        // Too far → move forward
        v = KP_D * (1.0f - area_ratio);
    }
    else if (area_ratio > (1.0f + AREA_TOLERANCE)) {
        // Too close → move backward
        v = -KP_D * (area_ratio - 1.0f);
    }

    v = clamp(v, -MAX_LINEAR, MAX_LINEAR);

    // -------- COMBINE MOTION --------
    MotorCmd cmd;
    cmd.left  = v - omega;
    cmd.right = v + omega;

    cmd.left  = clamp(cmd.left,  -1.0f, 1.0f);
    cmd.right = clamp(cmd.right, -1.0f, 1.0f);

    return cmd;
}

// ---------------- MAIN LOOP ----------------
int main() {
    // Reference bounding box (measured at ~30 cm)
    BBox reference_bbox = {200, 120, 120, 160};
    float reference_area = bbox_area(reference_bbox);

    std::cout << "[INIT] Reference area = " << reference_area << std::endl;

while (true) {
    BBox bbox;

    if (!(std::cin >> bbox.x >> bbox.y >> bbox.w >> bbox.h)) {
        std::cerr << "[WARN] Input stream closed. Exiting.\n";
        break;
    }

    if (bbox.w == 0 || bbox.h == 0) {
        std::cout << "LEFT: 0 | RIGHT: 0\n";
        continue;
    }

    MotorCmd cmd = compute_control(bbox, reference_area);

    std::cout << "LEFT: " << cmd.left
              << " | RIGHT: " << cmd.right << std::endl;

    std::this_thread::sleep_for(
        std::chrono::milliseconds(1000 / CONTROL_HZ)
    );
}
    return 0;
}