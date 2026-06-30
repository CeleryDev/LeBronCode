# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# (License text omitted for brevity)

import logging
import time
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pprint import pformat

import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.zmq import ZMQCameraConfig
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

# ==========================================
# --- SHARED STATE & THREADING ---
# ==========================================

SHARED_STATE = {
    "cmd_capture_start": False,
    "cmd_capture_end": False,
    "ui_msg": "",
    "start_frame": None,
    "end_frame": None,
    "stop_teleop": False
}

# ==========================================
# --- COMPUTER VISION EVALUATION SCRIPT ---
# ==========================================

MAX_TOLERANCE_PX = 30.0
TOLERANCE_THRESHOLD_PX = 15.0
LEFT_TRIM_PX = 130
EDGE_BUFFER_PX = 50

def auto_crop_whiteboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img.shape[1], img.shape[0]
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    margin = 15
    h_img, w_img = img.shape[:2]
    
    x1 = max(0, x + LEFT_TRIM_PX)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    
    return x1, y1, (x2 - x1), (y2 - y1)

def get_estimated_curve(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = thresh_img.shape
    centroids = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 800:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if (cx > EDGE_BUFFER_PX and cx < (w - EDGE_BUFFER_PX) and 
                    cy > EDGE_BUFFER_PX and cy < (h - EDGE_BUFFER_PX)):
                    centroids.append((cx, cy))
                
    if len(centroids) < 3:
        return thresh_img
        
    start_idx = np.argmin([p[1] for p in centroids])
    sorted_pts = [centroids.pop(start_idx)]
    
    while centroids:
        last = sorted_pts[-1]
        dists = [np.linalg.norm(np.array(last) - np.array(p)) for p in centroids]
        next_idx = np.argmin(dists)
        sorted_pts.append(centroids.pop(next_idx))
        
    sorted_pts = np.array(sorted_pts)
    curve_mask = np.zeros_like(thresh_img)
    
    try:
        tck, u = splprep([sorted_pts[:,0], sorted_pts[:,1]], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        curve_points = np.vstack((x_new, y_new)).T.astype(np.int32)
    except Exception:
        curve_points = sorted_pts
    
    cv2.polylines(curve_mask, [curve_points], False, 255, 3)
    return curve_mask

def evaluate_tracing(img_start, img_end):
    gray_start = cv2.cvtColor(img_start, cv2.COLOR_BGR2GRAY)
    gray_end = cv2.cvtColor(img_end, cv2.COLOR_BGR2GRAY)

    _, thresh_start = cv2.threshold(gray_start, 100, 255, cv2.THRESH_BINARY_INV)
    target_path = get_estimated_curve(thresh_start)

    diff = cv2.absdiff(gray_end, gray_start)
    _, drawn_mask = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)
    
    kernel_clean = np.ones((3, 3), np.uint8)
    drawn_mask = cv2.morphologyEx(drawn_mask, cv2.MORPH_OPEN, kernel_clean)

    contours, _ = cv2.findContours(drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_drawn_mask = np.zeros_like(drawn_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 150: 
            cv2.drawContours(clean_drawn_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    drawn_mask = clean_drawn_mask

    h, w = drawn_mask.shape
    drawn_mask[0:EDGE_BUFFER_PX, :] = 0
    drawn_mask[h-EDGE_BUFFER_PX:h, :] = 0
    drawn_mask[:, 0:EDGE_BUFFER_PX] = 0
    drawn_mask[:, w-EDGE_BUFFER_PX:w] = 0

    inv_target = cv2.bitwise_not(target_path)
    dist_transform = cv2.distanceTransform(inv_target, cv2.DIST_L2, 5)
    error_pixels = dist_transform[drawn_mask > 0]
    
    if len(error_pixels) == 0:
        return 0, 0, 0.0, 0.0, target_path, drawn_mask
        
    mean_error = np.mean(error_pixels)
    max_error = np.max(error_pixels)
    accuracy_mean = max(0.0, 100.0 - ((mean_error / MAX_TOLERANCE_PX) * 100.0))
    in_bounds_pixels = np.sum(error_pixels <= TOLERANCE_THRESHOLD_PX)
    accuracy_in_bounds = (in_bounds_pixels / len(error_pixels)) * 100.0

    return mean_error, max_error, accuracy_mean, accuracy_in_bounds, target_path, drawn_mask

def create_dashboard(crop_start, crop_end, target_mask, drawn_mask, mean_err, max_err, acc_mean, acc_bounds):
    h, w, _ = crop_start.shape
    footer_height = 180
    dash = np.ones((h + footer_height, w * 2, 3), dtype=np.uint8) * 255
    
    radius = int(TOLERANCE_THRESHOLD_PX)
    k_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tolerance_mask = cv2.dilate(target_mask, kernel)

    left_panel = crop_start.copy()
    left_overlay = left_panel.copy()
    left_overlay[tolerance_mask > 0] = [170, 255, 170]
    cv2.addWeighted(left_overlay, 0.4, left_panel, 0.6, 0, left_panel)
    left_panel[target_mask > 0] = [0, 200, 0] 
    dash[0:h, 0:w] = left_panel
    
    right_panel = crop_end.copy()
    right_overlay = right_panel.copy()
    right_overlay[tolerance_mask > 0] = [170, 255, 170] 
    cv2.addWeighted(right_overlay, 0.4, right_panel, 0.6, 0, right_panel)
    right_panel[target_mask > 0] = [0, 200, 0] 
    dash[0:h, w:w*2] = right_panel
    
    cv2.line(dash, (w, 0), (w, h), (0, 0, 0), 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.rectangle(dash, (10, 10), (450, 100), (255, 255, 255), -1)
    cv2.putText(dash, "FINAL TRACE", (30, 50), font, 1.2, (0, 0, 0), 3)
    cv2.putText(dash, "(ESTIMATED CURVE)", (30, 90), font, 0.9, (50, 50, 50), 2)
    
    cv2.rectangle(dash, (w + 10, 10), (w + 400, 100), (255, 255, 255), -1)
    cv2.putText(dash, "FINAL TRACE", (w + 30, 50), font, 1.2, (0, 0, 0), 3)
    cv2.putText(dash, "(ACTUAL RESULT)", (w + 30, 90), font, 0.9, (50, 50, 50), 2)
    
    cv2.rectangle(dash, (0, h), (w * 2, h + footer_height), (45, 40, 40), -1)
    cv2.putText(dash, "PERFORMANCE EVALUATION:", (40, h + 50), font, 1.2, (200, 200, 200), 2)
    cv2.putText(dash, f"MEAN SCORE:    {acc_mean:.1f}%", (40, h + 105), font, 1.5, (255, 255, 255), 3)
    cv2.putText(dash, f"WITHIN {int(TOLERANCE_THRESHOLD_PX)}PX:    {acc_bounds:.1f}%", (40, h + 155), font, 1.5, (150, 255, 150), 3)
    
    box_x = int(w * 1.3)
    cv2.rectangle(dash, (box_x, h + 30), (box_x + 350, h + 150), (60, 55, 55), -1)
    cv2.rectangle(dash, (box_x, h + 30), (box_x + 350, h + 150), (100, 100, 100), 2)
    
    cv2.putText(dash, f"Mean Error: {mean_err:.2f} px", (box_x + 20, h + 70), font, 0.8, (255, 255, 255), 2)
    cv2.putText(dash, f"Max Error:  {max_err:.2f} px", (box_x + 20, h + 110), font, 0.8, (255, 255, 255), 2)
    
    status_text = "PASS" if acc_bounds >= 85.0 else "FAIL"
    status_color = (100, 200, 100) if status_text == "PASS" else (50, 50, 255)
    cv2.putText(dash, status_text, (box_x + 200, h + 140), font, 1.2, status_color, 3)

    return dash

def convert_lerobot_to_cv2(img_data):
    """Converts a LeRobot camera observation tensor/array to a standard OpenCV BGR image."""
    if hasattr(img_data, 'cpu'):
        img_data = img_data.cpu().numpy()
        
    if img_data.ndim == 3 and img_data.shape[0] == 3:
        img_data = np.transpose(img_data, (1, 2, 0))
        
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)
            
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

def run_evaluation_from_memory(raw_start, raw_end):
    print("\n--- Running Computer Vision Evaluation ---")
    img_start = convert_lerobot_to_cv2(raw_start)
    img_end = convert_lerobot_to_cv2(raw_end)
    
    x, y, w, h = auto_crop_whiteboard(img_start)
    crop_start = img_start[y:y+h, x:x+w]
    crop_end = img_end[y:y+h, x:x+w]
    
    mean_err, max_err, acc_mean, acc_bounds, target_mask, drawn_mask = evaluate_tracing(crop_start, crop_end)
    dashboard_img = create_dashboard(crop_start, crop_end, target_mask, drawn_mask, mean_err, max_err, acc_mean, acc_bounds)
    
    print(f"Mean Score:      {acc_mean:.1f}%")
    print(f"Within {int(TOLERANCE_THRESHOLD_PX)}px:      {acc_bounds:.1f}%")
    
    if dashboard_img.shape[0] > 900:
        scale = 900 / dashboard_img.shape[0]
        dashboard_img = cv2.resize(dashboard_img, (0,0), fx=scale, fy=scale)
    
    print("\nEvaluation complete! Close the image window to fully exit.")
    cv2.imshow("ACT Policy Evaluation", dashboard_img)
    cv2.waitKey(0)

# ==========================================
# --- TKINTER UI THREAD ---
# ==========================================

def start_ui_thread():
    root = tk.Tk()
    root.title("LeRobot Evaluator")
    root.geometry("350x220")
    root.configure(padx=20, pady=20)

    title_label = tk.Label(root, text="Line Tracing Evaluator", font=("Helvetica", 14, "bold"))
    title_label.pack(pady=(0, 15))

    status_var = tk.StringVar(value="Status: Ready")
    
    def btn_start():
        SHARED_STATE["cmd_capture_start"] = True
        status_var.set("Status: Waiting to grab START frame...")
        
    def btn_end():
        if SHARED_STATE["start_frame"] is None:
            status_var.set("Status: Error! Capture START first.")
            return
            
        SHARED_STATE["cmd_capture_end"] = True
        status_var.set("Status: Capturing END & Evaluating...")

    tk.Button(root, text="1. Capture START Image", command=btn_start, width=25, height=2, bg="#e0e0e0").pack(pady=5)
    tk.Button(root, text="2. Capture END & Evaluate", command=btn_end, width=25, height=2, bg="#4CAF50", fg="black").pack(pady=10)
    
    status_label = tk.Label(root, textvariable=status_var, font=("Helvetica", 10, "italic"), fg="blue")
    status_label.pack(side=tk.BOTTOM)

    def check_status():
        if SHARED_STATE["ui_msg"]:
            status_var.set(f"Status: {SHARED_STATE['ui_msg']}")
            SHARED_STATE["ui_msg"] = ""
            
        # Automatically close the Tkinter window once teleop stops
        if SHARED_STATE["stop_teleop"]:
            root.quit()
            return
            
        root.after(100, check_status)

    check_status()
    root.mainloop()


# ==========================================
# --- TELEOPERATION LOGIC ---
# ==========================================

@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False

def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    
    try:
        while not SHARED_STATE["stop_teleop"]:
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            # --- IN-MEMORY UI CAPTURE TRIGGERS ---
            if SHARED_STATE["cmd_capture_start"]:
                if "topRight" in obs:
                    frame = obs["topRight"]
                    SHARED_STATE["start_frame"] = frame.clone() if hasattr(frame, 'clone') else frame.copy()
                    SHARED_STATE["ui_msg"] = "Start Image Captured!"
                    print("\n[EVALUATOR] Start image captured from memory.")
                else:
                    SHARED_STATE["ui_msg"] = "Error: 'topRight' camera not found."
                SHARED_STATE["cmd_capture_start"] = False
                
            if SHARED_STATE["cmd_capture_end"]:
                if "topRight" in obs:
                    frame = obs["topRight"]
                    SHARED_STATE["end_frame"] = frame.clone() if hasattr(frame, 'clone') else frame.copy()
                    SHARED_STATE["ui_msg"] = "End Image Captured!"
                    print("\n[EVALUATOR] End image captured. Exiting teleop loop...")
                    # Setting this to True breaks the loop immediately
                    SHARED_STATE["stop_teleop"] = True 
                else:
                    SHARED_STATE["ui_msg"] = "Error: 'topRight' camera not found."
                SHARED_STATE["cmd_capture_end"] = False

            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            raw_action = teleop.get_action()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action_to_send = robot_action_processor((teleop_action, obs))
            _ = robot.send_action(robot_action_to_send)

            if display_data:
                obs_transition = robot_observation_processor(obs)
                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            if duration is not None and time.perf_counter() - start >= duration:
                break
                
    except KeyboardInterrupt:
        print("\n\nTeleoperation interrupted by user (Ctrl+C).")
        SHARED_STATE["stop_teleop"] = True


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    # Launch Tkinter UI in a background daemon thread
    ui_thread = threading.Thread(target=start_ui_thread, daemon=True)
    ui_thread.start()

    try:
        # This will run continuously until you click "Capture END" in the UI (or hit Ctrl+C)
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    finally:
        if cfg.display_data:
            shutdown_rerun()
            
        print("\nDisconnecting hardware safely before running CV Evaluation...")
        teleop.disconnect()
        robot.disconnect()
        
        # If both frames were successfully captured, run the evaluation popup!
        if SHARED_STATE["start_frame"] is not None and SHARED_STATE["end_frame"] is not None:
            run_evaluation_from_memory(SHARED_STATE["start_frame"], SHARED_STATE["end_frame"])
        else:
            print("\nEvaluation skipped: Start or End frame was not captured.")

def main():
    register_third_party_plugins()
    teleoperate()

if __name__ == "__main__":
    main()
