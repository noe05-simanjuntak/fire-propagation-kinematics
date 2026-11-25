import cv2
import time
import math
import csv
from ultralytics import YOLO

# 1. Initialize AI Model and Video Capture
model = YOLO('best.pt')
cap = cv2.VideoCapture("dataset_video/experiment_1.mp4")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Dynamic Fire Metrology System Active.")
print("Press 'q' to terminate the system.")
print("Press 'p' to pause/resume playback.")
print("Press 's' to capture a SCREENSHOT (Saved automatically).")
print("Press 'r' to START/STOP recording kinematics data to CSV.")

# 2. Spatiotemporal Calculus Memory Registry
previous_area = 0.0
previous_cx = 0
previous_cy = 0
previous_time = time.time()
history_dx = []
history_dy = []
WINDOW_SIZE = 15

# Global HUD Variables
hud_area = 0.0
hud_rate = 0.0
hud_cx, hud_cy = 0, 0
hud_vel = 0.0
hud_angle = 0.0

# ==========================================
# CSV LOGGER MANAGEMENT VARIABLES
# ==========================================
is_recording = False
csv_file = None
csv_writer = None
start_record_time = 0.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Automatically stop when video ends

    current_time = time.time()
    dt = current_time - previous_time
    previous_time = current_time

    results = model(frame, conf=0.35, verbose=False)

    # ---------------------------------------------------------
    # UNIFIED BOUNDING BOX ALGORITHM (Fire Center of Mass)
    # ---------------------------------------------------------
    global_x1, global_y1 = float('inf'), float('inf')
    global_x2, global_y2 = 0, 0
    fire_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Find the outer boundaries across all detected fire regions
            global_x1 = min(global_x1, x1)
            global_y1 = min(global_y1, y1)
            global_x2 = max(global_x2, x2)
            global_y2 = max(global_y2, y2)
            fire_detected = True

    if fire_detected and dt > 0:
        width = global_x2 - global_x1
        height = global_y2 - global_y1
        current_area = width * height

        cx = int(global_x1 + width / 2)
        cy = int(global_y1 + height / 2)

        da_dt = (current_area - previous_area) / dt

        # Vector Calculation with Moving Average
        if previous_cx != 0 and previous_cy != 0:
            dx = cx - previous_cx
            dy = cy - previous_cy

            history_dx.append(dx)
            history_dy.append(dy)

            if len(history_dx) > WINDOW_SIZE:
                history_dx.pop(0)
                history_dy.pop(0)

            avg_dx = sum(history_dx) / len(history_dx)
            avg_dy = sum(history_dy) / len(history_dy)

            distance = math.sqrt(avg_dx**2 + avg_dy**2)
            velocity = distance / dt
            angle_deg = math.degrees(math.atan2(avg_dy, avg_dx))

            hud_vel = velocity
            hud_angle = angle_deg

            # Stable Prediction Zone Calculation
            prediction_horizon = 1.0
            shift_x = int((avg_dx / dt) * prediction_horizon)
            shift_y = int((avg_dy / dt) * prediction_horizon)

            pred_x1 = int(global_x1) + shift_x
            pred_y1 = int(global_y1) + shift_y
            pred_x2 = int(global_x2) + shift_x
            pred_y2 = int(global_y2) + shift_y
            pred_cx = cx + shift_x
            pred_cy = cy + shift_y

            # Render Velocity Vector & Predicted Zone
            cv2.rectangle(frame, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 255), 2)
            cv2.arrowedLine(frame, (cx, cy), (pred_cx, pred_cy), (0, 255, 0), 3, tipLength=0.2)

        # Update Global HUD & Memory
        hud_area = current_area
        hud_rate = da_dt
        previous_area = current_area
        previous_cx = cx
        previous_cy = cy

        # Render Unified Bounding Box & Centroid
        cv2.rectangle(frame, (int(global_x1), int(global_y1)), (int(global_x2), int(global_y2)), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # ---------------------------------------------------------
    # RENDER LABORATORY HUD OVERLAY
    # ---------------------------------------------------------
    frame_height, frame_width = frame.shape[:2]

    # Dark background panel for clear telemetry readability
    cv2.rectangle(frame, (10, 10), (340, 140), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (340, 140), (255, 255, 255), 1)

    def draw_hud_text(img, text, pos, color):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    draw_hud_text(frame, f"Area      : {hud_area:.0f} px^2", (20, 35), (255, 255, 255))
    draw_hud_text(frame, f"Exp. Rate : {hud_rate:.1f} px^2/s", (20, 65), (255, 255, 255))
    draw_hud_text(frame, f"Velocity  : {hud_vel:.1f} px/s", (20, 95), (0, 255, 255))
    draw_hud_text(frame, f"Direction : {hud_angle:.1f} deg", (20, 125), (0, 255, 255))

    if is_recording:
        cv2.putText(frame, "REC [DATA LOGGING]", (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==========================================
    # CSV DATA EXTRACTION LOGIC
    # ==========================================
    if is_recording and csv_writer is not None:
        elapsed_time = current_time - start_record_time
        csv_writer.writerow([
            round(elapsed_time, 2),
            round(hud_area, 2),
            round(hud_rate, 2),
            round(hud_vel, 2),
            round(hud_angle, 2)
        ])

    cv2.imshow("Fire Metrology Dashboard v1.0", frame)

    # ==========================================
    # PERIPHERAL CONTROLS
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("--> PAUSE MODE.")
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord('p'):
                previous_time = time.time()
                break
            elif key2 == ord('s'):
                filename = f"Detection_Result_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"--> SCREENSHOT: {filename} saved!")
    elif key == ord('s'):
        filename = f"Detection_Result_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"--> SCREENSHOT: {filename} saved!")
    elif key == ord('r'):
        if not is_recording:
            is_recording = True
            start_record_time = time.time()
            filename_csv = f"Kinematics_Data_{int(time.time())}.csv"
            csv_file = open(filename_csv, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Time (s)", "Area (px^2)", "Expansion Rate (px^2/s)", "Velocity (px/s)", "Direction (deg)"])
            print(f"--> RECORDING STARTED: {filename_csv}")
        else:
            is_recording = False
            if csv_file:
                csv_writer.writerow(["---END OF DATA---", "", "", "", ""])
                csv_file.flush()
                csv_file.close()
                csv_file = None
                csv_writer = None
            print("--> RECORDING STOPPED.")

if csv_file:
    csv_file.close()
cap.release()
cv2.destroyAllWindows()
