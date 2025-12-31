#!/usr/bin/env python3
"""
Simple USB camera feed viewer.
Displays video from a USB camera in a window.
"""

import cv2
import sys


def main():
    # Try to open the default camera (usually /dev/video0 on Linux or 0 on other systems)
    # You can change this to a specific device path or index if needed
    camera_index = 0
    
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
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Display the frame
            cv2.imshow('USB Camera Feed', frame)
            
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

