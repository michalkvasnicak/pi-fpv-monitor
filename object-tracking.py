#!/usr/bin/env python3
"""
Person detection and tracking script.
Shows only person detections and allows clicking to start tracking.
"""

import cv2
import sys
from ultralytics import YOLO 
import time


# Global variables for mouse callback
selected_person_box = None
tracking_active = False
tracker = None
window_name = 'Person Detection & Tracking'


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select a person for tracking."""
    global selected_person_box, tracking_active, tracker, current_detections, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Find which person bounding box was clicked
        for box in current_detections:
            x1, y1, x2, y2 = box[:4]
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Person clicked! Start tracking
                selected_person_box = [x1, y1, x2, y2]
                tracking_active = True
                # Initialize tracker with the selected bounding box
                tracker = cv2.TrackerCSRT_create()
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                tracker.init(current_frame, bbox)
                print(f"Started tracking person at ({x1}, {y1}, {x2}, {y2})")
                break


def filter_person_detections(results):
    """Filter YOLO results to only include person class (class 0)."""
    person_detections = []
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Check if this is a person (class 0)
                cls = int(box.cls[0])
                if cls == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    person_detections.append([x1, y1, x2, y2, conf])
    
    return person_detections


def draw_person_detections(frame, detections, selected_box=None):
    """Draw person bounding boxes on the frame."""
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Check if this is the selected person
        is_selected = (selected_box is not None and 
                      abs(x1 - selected_box[0]) < 5 and 
                      abs(y1 - selected_box[1]) < 5)
        
        # Color: green for selected/tracked, blue for others
        color = (0, 255, 0) if is_selected else (255, 0, 0)
        thickness = 3 if is_selected else 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"Person {conf:.2f}"
        if is_selected:
            label = f"TRACKING: {label}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw text background
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def main():
    global selected_person_box, tracking_active, tracker, current_detections, current_frame
    
    # Try to open the default camera (usually /dev/video0 on Linux or 0 on other systems)
    camera_index = 0

    # Load a YOLO11n PyTorch model
    model = YOLO("yolo11n.pt")

    # Export the model to NCNN format
    model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

    # Load the exported NCNN model
    ncnn_model = YOLO("yolo11n_ncnn_model")
    
    # Try /dev/video0 first (common on Linux/Raspberry Pi)
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture("/dev/video0")
        if not cap.isOpened():
            # Fallback to index 0
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Make sure a USB camera is connected and try again.")
        sys.exit(1)
    
    # Set camera properties (optional - adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Control exposure settings to prevent overexposure
    # Disable auto-exposure (set to manual mode)
    # Value: 0.25 = manual mode, 0.75 = auto mode (varies by camera)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # Set exposure value (lower = darker, higher = brighter)
    # Typical range: -13 to -1 (logarithmic scale) or 1-10000 (linear scale)
    # Start with a lower value to reduce overexposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    
    # Set gain (amplification, lower = less noise but darker)
    # Typical range: 0-100
    cap.set(cv2.CAP_PROP_GAIN, 0)
    
    # Set brightness (0-100, default is often 50)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    
    # Print current camera settings for debugging
    print("Camera opened successfully!")
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Gain: {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print("Click on a person to start tracking. Press 'q' to quit, 'r' to reset tracking.")
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    fps_t0 = time.time()
    frames = 0
    fps = 0.0
    current_detections = []
    current_frame = None
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break

            current_frame = frame.copy()
            
            # Run detection only on person class
            results = ncnn_model(frame, classes=[0])  # classes=[0] filters to person only
            
            # Filter to get person detections
            current_detections = filter_person_detections(results)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # If tracking is active, update tracker
            if tracking_active and tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    # Update selected_person_box from tracker
                    x, y, w, h = [int(v) for v in bbox]
                    selected_person_box = [x, y, x + w, y + h]
                else:
                    # Tracking lost
                    print("Tracking lost!")
                    tracking_active = False
                    tracker = None
                    selected_person_box = None
            
            # Draw all person detections (tracked person will be highlighted in green)
            annotated_frame = draw_person_detections(annotated_frame, current_detections, selected_person_box)
            
            # If tracking, also draw a prominent tracking box
            if tracking_active and selected_person_box:
                x1, y1, x2, y2 = selected_person_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Draw thick green tracking box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated_frame, "TRACKING", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # FPS counter
            frames += 1
            dt = time.time() - fps_t0

            if dt >= 0.5:
                fps = frames / dt
                fps_t0 = time.time()
                frames = 0

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instruction text
            if not tracking_active:
                cv2.putText(annotated_frame, "Click on a person to track", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                tracking_active = False
                tracker = None
                selected_person_box = None
                print("Tracking reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()

