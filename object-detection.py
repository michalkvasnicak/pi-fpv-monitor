#!/usr/bin/env python3
"""
Simple USB camera feed viewer.
Displays video from a USB camera in a window.
"""

import cv2
import sys
from ultralytics import YOLO 
import time 


def main():
    # Try to open the default camera (usually /dev/video0 on Linux or 0 on other systems)
    # You can change this to a specific device path or index if needed
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
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")

    fps_t0 = time.time()
    frames = 0
    fps = 0.0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Run the model on the grayscale frame
            results = ncnn_model(gray_frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Convert annotated frame to grayscale for display
            if len(annotated_frame.shape) == 3:
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)

            # FPS counter
            frames += 1
            dt = time.time() - fps_t0

            if dt >= 0.5:
                fps = frames / dt
                fps_t0 = time.time()
                frames = 0

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the grayscale frame
            cv2.imshow('USB Camera Feed', annotated_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")


if __name__ == "__main__":
    main()

