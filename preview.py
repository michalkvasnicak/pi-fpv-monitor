#!/usr/bin/env python3
"""
Simple preview program for troubleshooting FPV setup.
Tunes RX5808 to 5825 MHz and displays video feed.
"""
import time
import cv2
from gpiozero import DigitalOutputDevice

# RX5808 pin mapping (same as main.py)
PIN_DATA = 17  # RX5808 CH1
PIN_CLK  = 22  # RX5808 CH3
PIN_LE   = 27  # RX5808 CH2

T = 20e-6
SPI_ADDRESS_SYNTH_B = 0x01
FREQ_MHZ = 5825  # Fixed frequency for troubleshooting


def get_synth_register_b_freq_mhz(freq_mhz: int) -> int:
    x = (freq_mhz - 479) // 2
    return ((x // 32) << 7) | (x % 32)


def build_25bit_word(address_bits: int, data_bits: int) -> int:
    return (address_bits & 0x0F) | (1 << 4) | ((data_bits & 0xFFFF) << 5)


class RX5808Tuner:
    """LSB-first, latch on LE falling edge (same as main.py)."""
    def __init__(self, pin_data: int, pin_clk: int, pin_le: int, le_idle: int = 1):
        self.data = DigitalOutputDevice(pin_data, initial_value=False)
        self.clk  = DigitalOutputDevice(pin_clk,  initial_value=False)
        self.le   = DigitalOutputDevice(pin_le,   initial_value=bool(le_idle))
        self.le_idle = bool(le_idle)

    def hard_init(self, freq_mhz: int):
        # Force known idle states
        self.data.off()
        self.clk.off()
        self.le.on()   # LE idle HIGH (critical)

        time.sleep(0.01)

        # Send a few dummy clocks with LE high
        for _ in range(10):
            self.clk.on()
            time.sleep(T)
            self.clk.off()
            time.sleep(T)

        # Send a valid tune twice (any frequency)
        self.tune_mhz(freq_mhz)
        time.sleep(0.01)
        self.tune_mhz(freq_mhz)

    def _sleep(self):
        time.sleep(T)

    def _clk_pulse(self):
        self.clk.on();  self._sleep()
        self.clk.off(); self._sleep()

    def _shift_25_lsb_first(self, word: int):
        for bit in range(25):
            self.data.value = bool((word >> bit) & 1)
            self._sleep()
            self._clk_pulse()

    def write_word(self, word: int):
        self.le.on()
        self._sleep()
        self._shift_25_lsb_first(word)
        self.le.off()  # falling latch
        self._sleep()
        self.le.value = self.le_idle
        self._sleep()

    def tune_mhz(self, freq_mhz: int):
        data_bits = get_synth_register_b_freq_mhz(freq_mhz)
        word = build_25bit_word(SPI_ADDRESS_SYNTH_B, data_bits)
        for _ in range(2):
            self.write_word(word)
            time.sleep(0.005)


def build_gst_pipeline_mjpg(device="/dev/video0", width=800, height=600, fps=30) -> str:
    """Build GStreamer pipeline for MJPEG video capture."""
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def main():
    import os
    
    print("FPV Preview - Troubleshooting Mode")
    
    # Wait for video device to be available (important after reboot)
    video_device = "/dev/video0"
    max_wait = 50  # Wait up to 5 seconds
    print(f"Waiting for {video_device}...")
    for i in range(max_wait):
        if os.path.exists(video_device):
            break
        time.sleep(0.1)
    else:
        print(f"ERROR: Video device {video_device} not found after waiting")
        return
    
    print(f"Initializing RX5808 tuner...")
    
    # Initialize tuner
    tuner = RX5808Tuner(PIN_DATA, PIN_CLK, PIN_LE, le_idle=1)
    
    # Hard init and tune to frequency BEFORE opening video device
    tuner.hard_init(FREQ_MHZ)
    time.sleep(0.2)
    
    # Apply tuning multiple times to ensure lock
    for _ in range(3):
        tuner.tune_mhz(FREQ_MHZ)
        time.sleep(0.1)
    
    print(f"Starting video capture...")
    
    # Build GStreamer pipeline
    pipeline = build_gst_pipeline_mjpg(
        device="/dev/video0",
        width=800,
        height=600,
        fps=30
    )
    
    # Open video capture
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Failed to open video device!")
        print("Try running:")
        print("  gst-launch-1.0 v4l2src device=/dev/video0 ! "
              "image/jpeg,width=800,height=600,framerate=30/1 ! "
              "jpegdec ! videoconvert ! fakesink")
        return
    
    # CRITICAL: Retune AFTER opening video device (device opening might affect tuner state)
    print("Retuning after video device opened...")
    for _ in range(3):
        tuner.tune_mhz(FREQ_MHZ)
        time.sleep(0.1)
    
    time.sleep(0.5)  # Let video device lock onto signal
    
    print(f"Video feed active. Frequency: {FREQ_MHZ} MHz")
    print("Press 'q' to quit")
    
    # Warmup: try to read a few frames
    print("Warming up video capture...")
    warmup_success = False
    for i in range(50):  # Try for up to 5 seconds
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_success = True
            print("Video capture ready!")
            break
        time.sleep(0.1)
        # Retune periodically during warmup
        if i % 10 == 0:
            tuner.tune_mhz(FREQ_MHZ)
    
    if not warmup_success:
        print("WARNING: Could not read frames during warmup, continuing anyway...")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_count += 1
                # Retune periodically if we're not getting frames
                if frame_count % 30 == 0:  # Every ~3 seconds at 10fps
                    print("Retuning...")
                    tuner.tune_mhz(FREQ_MHZ)
                time.sleep(0.1)
                continue
            
            frame_count = 0  # Reset counter on success
            
            # Display frame
            cv2.imshow("FPV Preview", frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Preview closed")


if __name__ == "__main__":
    main()

