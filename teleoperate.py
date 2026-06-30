import cv2
import numpy as np
import argparse
import sys
from scipy.interpolate import splprep, splev

# --- CONFIGURATION ---
MAX_TOLERANCE_PX = 30.0       # Max pixel distance for the mean accuracy scaling (0% score)
TOLERANCE_THRESHOLD_PX = 15.0 # Radius in pixels for the strict "In-Bounds" percentage calculation
LEFT_TRIM_PX = 130            # Trims the robot arm from the left
EDGE_BUFFER_PX = 50           # Ignores noise/shadows on the extreme edges of the whiteboard

def auto_crop_whiteboard(img):
    """Automatically finds the whiteboard and applies a left-side trim."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: Could not automatically detect the whiteboard. Using full image.")
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
    """Finds dashes, filters out edge noise, and estimates a smooth SciPy curve."""
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
        print("Warning: Not enough valid dashes detected to fit a curve.")
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
    except Exception as e:
        print(f"Warning: Curve fitting failed ({e}).")
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

    # Filter out stray noise (keep only large continuous chunks of ink)
    contours, _ = cv2.findContours(drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_drawn_mask = np.zeros_like(drawn_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 150: 
            cv2.drawContours(clean_drawn_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    drawn_mask = clean_drawn_mask

    # Apply edge buffer blackout to evaluation
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
    
    # --- Generate the Faded Tolerance Mask ---
    # Expand the target line out by the threshold radius
    radius = int(TOLERANCE_THRESHOLD_PX)
    k_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    tolerance_mask = cv2.dilate(target_mask, kernel)

    # --- LEFT PANEL (Ideal Overlay) ---
    left_panel = crop_start.copy()
    
    # 1. Overlay translucent faded green tolerance zone
    left_overlay = left_panel.copy()
    left_overlay[tolerance_mask > 0] = [170, 255, 170] # Light pastel green
    cv2.addWeighted(left_overlay, 0.4, left_panel, 0.6, 0, left_panel)
    
    # 2. Draw solid target line on top
    left_panel[target_mask > 0] = [0, 200, 0] # Slightly darker solid green
    
    dash[0:h, 0:w] = left_panel
    
    # --- RIGHT PANEL (Result Overlay) ---
    right_panel = crop_end.copy()
    
    # 1. Overlay translucent faded green tolerance zone over the drawn ink
    right_overlay = right_panel.copy()
    right_overlay[tolerance_mask > 0] = [170, 255, 170] 
    cv2.addWeighted(right_overlay, 0.4, right_panel, 0.6, 0, right_panel)
    
    # 2. Draw solid target line on top
    right_panel[target_mask > 0] = [0, 200, 0] 
    
    dash[0:h, w:w*2] = right_panel
    
    # --- UI Elements ---
    cv2.line(dash, (w, 0), (w, h), (0, 0, 0), 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.rectangle(dash, (10, 10), (450, 100), (255, 255, 255), -1)
    cv2.putText(dash, "FINAL TRACE", (30, 50), font, 1.2, (0, 0, 0), 3)
    cv2.putText(dash, "(ESTIMATED CURVE)", (30, 90), font, 0.9, (50, 50, 50), 2)
    
    cv2.rectangle(dash, (w + 10, 10), (w + 400, 100), (255, 255, 255), -1)
    cv2.putText(dash, "FINAL TRACE", (w + 30, 50), font, 1.2, (0, 0, 0), 3)
    cv2.putText(dash, "(ACTUAL RESULT)", (w + 30, 90), font, 0.9, (50, 50, 50), 2)
    
    # --- Footer ---
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy line tracing.")
    parser.add_argument("--start", required=True, help="Path to baseline image.")
    parser.add_argument("--end", required=True, help="Path to final image.")
    parser.add_argument("--output", default="evaluation_dashboard.jpg", help="Path to save output.")
    args = parser.parse_args()

    img_start = cv2.imread(args.start)
    img_end = cv2.imread(args.end)

    if img_start is None or img_end is None:
        print("Error: Could not load one or both images.")
        sys.exit(1)

    if img_start.shape != img_end.shape:
        print("Error: Image dimensions must match exactly.")
        sys.exit(1)

    x, y, w, h = auto_crop_whiteboard(img_start)
    crop_start = img_start[y:y+h, x:x+w]
    crop_end = img_end[y:y+h, x:x+w]
    
    print(f"Cropped to whiteboard (with {LEFT_TRIM_PX}px left trim): {w}x{h}")

    mean_err, max_err, acc_mean, acc_bounds, target_mask, drawn_mask = evaluate_tracing(crop_start, crop_end)
    
    dashboard_img = create_dashboard(crop_start, crop_end, target_mask, drawn_mask, mean_err, max_err, acc_mean, acc_bounds)
    
    print(f"\n[EVALUATION COMPLETE]")
    print(f"Mean Score:      {acc_mean:.1f}%")
    print(f"Within {int(TOLERANCE_THRESHOLD_PX)}px:      {acc_bounds:.1f}%")
    print(f"Mean Error:      {mean_err:.2f} px")
    print(f"Max Error:       {max_err:.2f} px")
    
    display_img = dashboard_img.copy()
    if display_img.shape[0] > 900:
        scale = 900 / display_img.shape[0]
        display_img = cv2.resize(display_img, (0,0), fx=scale, fy=scale)

    cv2.imwrite(args.output, dashboard_img)
    cv2.imshow("ACT Policy Evaluation", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()