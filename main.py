#!/usr/bin/env python3
import time
import datetime
import os
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QMessageBox, QComboBox
)

from gpiozero import DigitalOutputDevice

# =========================
# RX5808 pin mapping (your confirmed working mapping)
# =========================
PIN_DATA = 17  # RX5808 CH1
PIN_CLK  = 22  # RX5808 CH3
PIN_LE   = 27  # RX5808 CH2

T = 20e-6
SPI_ADDRESS_SYNTH_B = 0x01
BITRATE = 3000 # lower to 2000 if you have a slow CPU
DEFAULT_FREQ_MHZ = 5825

CHANNELS = [
    ("A1", 5865), ("A2", 5845), ("A3", 5825), ("A4", 5805), ("A5", 5785), ("A6", 5765), ("A7", 5745), ("A8", 5725),
    ("B1", 5733), ("B2", 5752), ("B3", 5771), ("B4", 5790), ("B5", 5809), ("B6", 5828), ("B7", 5847), ("B8", 5866),
    ("E1", 5705), ("E2", 5685), ("E3", 5665), ("E4", 5645), ("E5", 5885), ("E6", 5905), ("E7", 5925), ("E8", 5945),
    ("F1", 5740), ("F2", 5760), ("F3", 5780), ("F4", 5800), ("F5", 5820), ("F6", 5840), ("F7", 5860), ("F8", 5880),
    ("R1", 5658), ("R2", 5695), ("R3", 5732), ("R4", 5769), ("R5", 5806), ("R6", 5843), ("R7", 5880), ("R8", 5917),
]

SCREENSHOTS_DIR = Path("screenshots")
VIDEOS_DIR = Path("videos")


def get_synth_register_b_freq_mhz(freq_mhz: int) -> int:
    x = (freq_mhz - 479) // 2
    return ((x // 32) << 7) | (x % 32)

def build_25bit_word(address_bits: int, data_bits: int) -> int:
    return (address_bits & 0x0F) | (1 << 4) | ((data_bits & 0xFFFF) << 5)

class RX5808Tuner:
    """LSB-first, latch on LE rising edge, LE_idle=0 (confirmed working configuration)."""
    def __init__(self, pin_data: int, pin_clk: int, pin_le: int, le_idle: int = 0):
        self.data = DigitalOutputDevice(pin_data, initial_value=False)
        self.clk  = DigitalOutputDevice(pin_clk,  initial_value=False)
        self.le   = DigitalOutputDevice(pin_le,   initial_value=bool(le_idle))
        self.le_idle = bool(le_idle)

    def hard_init(self, freq_mhz: int):
        # Force known idle states
        self.data.off()
        self.clk.off()
        self.le.off()   # LE idle LOW (le_idle=0)

        time.sleep(0.01)

        # Send a few dummy clocks with LE low
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
        # Set idle state first
        self.le.value = bool(self.le_idle)
        self._sleep()
        # Keep LE low during shift (for rising latch)
        self.le.off()
        self._sleep()
        # Shift data bits
        self._shift_25_lsb_first(word)
        # Raise LE to latch (rising edge)
        self.le.on()
        self._sleep()
        # Return to idle state
        self.le.value = bool(self.le_idle)
        self._sleep()

    def tune_mhz(self, freq_mhz: int):
        data_bits = get_synth_register_b_freq_mhz(freq_mhz)
        word = build_25bit_word(SPI_ADDRESS_SYNTH_B, data_bits)
        for _ in range(2):
            self.write_word(word)
            time.sleep(0.005)


# =========================
# Video utils
# =========================
def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

def letterbox(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if target_w <= 0 or target_h <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

def format_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# =========================
# GStreamer tee pipeline (MJPG device)
# =========================
def build_gst_pipeline_mjpg(device="/dev/video0",
                            width=800, height=600, fps=30,
                            record_path: str | None = None) -> str:
    src = (
        f"v4l2src device={device} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"tee name=t "
    )

    display = (
        "t. ! queue leaky=downstream max-size-buffers=1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

    if record_path is None:
        return src + display

    record = (
        " t. ! queue ! "
        "videoconvert ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={BITRATE} key-int-max=30 ! "
        "h264parse config-interval=1 ! mpegtsmux ! "
        f"filesink location={record_path} sync=false"
    )

    return src + display + record


# =========================
# Main GUI (minimal overlay UI)
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV Monitor")

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        # RX5808
        self.tuner = RX5808Tuner(PIN_DATA, PIN_CLK, PIN_LE, le_idle=0)
        self.channel_idx = next((i for i, (_, mhz) in enumerate(CHANNELS) if mhz == DEFAULT_FREQ_MHZ), 0)

        # State
        self.cap = None
        self.last_frame = None
        self.recording = False
        self.record_path = None
        self.record_start_monotonic = None
        self.rssi_percent = 0  # placeholder

        # Root - use a container widget for absolute positioning
        root = QWidget()
        root.setStyleSheet("background: black;")
        self.setCentralWidget(root)

        # Video label fills entire widget
        self.video_label = QLabel(root)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(QSize(1024, 600))
        self.video_label.show()  # Ensure it's visible

        # Top overlay row container
        top_overlay = QWidget(root)
        top_row = QHBoxLayout(top_overlay)
        top_row.setContentsMargins(10, 10, 10, 0)
        top_row.setSpacing(10)

        self.lbl_status = QLabel()
        self.lbl_status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_status.setStyleSheet(self._overlay_label_css())

        # Optional dropdown (small)
        self.combo_channel = QComboBox()
        for i, (name, mhz) in enumerate(CHANNELS):
            self.combo_channel.addItem(f"{name}  {mhz} MHz", userData=i)
        self.combo_channel.setCurrentIndex(self.channel_idx)
        self.combo_channel.currentIndexChanged.connect(self.on_channel_selected)
        self.combo_channel.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                padding: 6px 10px;
                background: rgba(0,0,0,150);
                color: white;
                border: 1px solid rgba(255,255,255,80);
                border-radius: 10px;
                min-height: 38px;
            }
            QComboBox QAbstractItemView {
                background: #111;
                color: white;
                selection-background-color: #333;
            }
        """)

        self.lbl_rec = QLabel("")
        self.lbl_rec.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.lbl_rec.setStyleSheet(self._overlay_label_css())

        top_row.addWidget(self.lbl_status, stretch=0, alignment=Qt.AlignLeft | Qt.AlignTop)
        top_row.addWidget(self.combo_channel, stretch=0, alignment=Qt.AlignLeft | Qt.AlignTop)
        top_row.addStretch(1)
        top_row.addWidget(self.lbl_rec, stretch=0, alignment=Qt.AlignRight | Qt.AlignTop)

        # Bottom overlay row container
        bottom_overlay = QWidget(root)
        bottom_row = QHBoxLayout(bottom_overlay)
        bottom_row.setContentsMargins(10, 0, 10, 10)
        bottom_row.setSpacing(0)

        # Bottom-left stacked +/- like Google Maps zoom
        left_stack = QVBoxLayout()
        left_stack.setSpacing(8)

        self.btn_plus = QPushButton("+")
        self.btn_minus = QPushButton("−")
        self.btn_plus.clicked.connect(lambda: self.step_channel(+1))
        self.btn_minus.clicked.connect(lambda: self.step_channel(-1))
        for b in (self.btn_plus, self.btn_minus):
            b.setMinimumSize(70, 70)
            b.setStyleSheet(self._round_button_css(font_px=30))
        left_stack.addWidget(self.btn_plus, alignment=Qt.AlignLeft | Qt.AlignBottom)
        left_stack.addWidget(self.btn_minus, alignment=Qt.AlignLeft | Qt.AlignBottom)

        # Bottom-right stacked Record/Stop toggle + Screenshot
        right_stack = QVBoxLayout()
        right_stack.setSpacing(8)

        self.btn_record_toggle = QPushButton("Record")
        self.btn_shot = QPushButton("Screenshot")
        self.btn_fullscreen = QPushButton("Fullscreen")
        self.btn_record_toggle.clicked.connect(self.toggle_recording)
        self.btn_shot.clicked.connect(self.screenshot)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        self.btn_record_toggle.setMinimumSize(150, 70)
        self.btn_shot.setMinimumSize(150, 70)
        self.btn_fullscreen.setMinimumSize(150, 70)

        self.btn_record_toggle.setStyleSheet(self._pill_button_css())
        self.btn_shot.setStyleSheet(self._pill_button_css())
        self.btn_fullscreen.setStyleSheet(self._pill_button_css())

        right_stack.addWidget(self.btn_record_toggle, alignment=Qt.AlignRight | Qt.AlignBottom)
        right_stack.addWidget(self.btn_shot, alignment=Qt.AlignRight | Qt.AlignBottom)
        right_stack.addWidget(self.btn_fullscreen, alignment=Qt.AlignRight | Qt.AlignBottom)

        bottom_row.addLayout(left_stack)
        bottom_row.addStretch(1)
        bottom_row.addLayout(right_stack)

        # Make overlay widgets transparent backgrounds so video shows through
        top_overlay.setStyleSheet("background: transparent;")
        bottom_overlay.setStyleSheet("background: transparent;")
        
        # Position overlays absolutely on top of video
        top_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        bottom_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        # Wait for video device to be available (important after reboot)
        video_device = "/dev/video0"
        max_wait = 50  # Wait up to 5 seconds
        for i in range(max_wait):
            if os.path.exists(video_device):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Video device {video_device} not found after waiting")

        # Initialize RX5808 tuner BEFORE starting video capture
        # This ensures the video device is properly tuned when capture starts
        name, mhz = CHANNELS[self.channel_idx]
        self.tuner.hard_init(mhz)
        time.sleep(0.2)  # Give the tuner time to stabilize
        
        # Apply the channel tuning multiple times to ensure lock
        for _ in range(3):
            self.tuner.tune_mhz(mhz)
            time.sleep(0.1)
        
        time.sleep(0.5)  # Additional delay to let video device lock onto signal

        # Init capture AFTER tuner is ready
        if not self.restart_capture(record_path=None):
            raise RuntimeError("Failed to start GStreamer capture pipeline.")

        # Warmup: Try to read a few frames to ensure video device is ready
        # This is critical after reboot when the device needs time to initialize
        self._warmup_frames = 0
        self._warmup_timer = QTimer(self)
        self._warmup_timer.timeout.connect(self._warmup_capture)
        self._warmup_timer.start(50)  # Check every 50ms

        # Timer for frames (will be started after warmup)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        # Prime RX5808 after capture is up (helps after reboot / USB power timing)
        self._prime_tries = 0
        
        # Continue priming sequence for stability
        QTimer.singleShot(500, self._start_radio_init_sequence)

        # Shortcuts
        self.addAction(self._make_action("A", lambda: self.step_channel(-1)))
        self.addAction(self._make_action("D", lambda: self.step_channel(+1)))
        self.addAction(self._make_action("F", self.toggle_fullscreen))
        self.addAction(self._make_action("S", self.screenshot))
        self.addAction(self._make_action("R", self.toggle_recording))
        
        # Store overlay widgets for resize handling
        self.top_overlay = top_overlay
        self.bottom_overlay = bottom_overlay
        
        # Initial positioning
        self._position_overlays()

    def _warmup_capture(self):
        """Try to read frames from the video device until successful."""
        if self.cap is None:
            return
        
        ok, frame = self.cap.read()
        if ok and frame is not None:
            # Successfully read a frame, video device is ready
            self._warmup_timer.stop()
            self.last_frame = frame
            # Start the main timer
            self.timer.start(15)
            # Update display immediately (use frame dimensions if label not sized yet)
            w = self.video_label.width()
            h = self.video_label.height()
            if w <= 0 or h <= 0:
                w, h = 1024, 600  # Use default size
            out = letterbox(frame, w, h)
            self.video_label.setPixmap(QPixmap.fromImage(bgr_to_qimage(out)))
        else:
            # Keep trying, but limit attempts
            self._warmup_frames += 1
            # Retune every 20 attempts (1 second) in case signal was lost
            if self._warmup_frames % 20 == 0:
                name, mhz = CHANNELS[self.channel_idx]
                self.tuner.tune_mhz(mhz)
            if self._warmup_frames >= 100:  # 5 seconds max (100 * 50ms)
                # Give up and start timer anyway
                self._warmup_timer.stop()
                self.timer.start(15)

    def _start_radio_init_sequence(self):
        # Force a clean init AFTER RX5808 is powered
        self.tuner.hard_init(CHANNELS[self.channel_idx][1])

        # Prime retunes (helps PLL lock / power timing)
        self._prime_tries = 0
        if not hasattr(self, "_prime_timer"):
            self._prime_timer = QTimer(self)
            self._prime_timer.timeout.connect(self._prime_rx5808)
        self._prime_timer.start(200)

        # Also apply the channel once a bit later
        QTimer.singleShot(600, self.apply_channel)


    def _prime_rx5808(self):
        name, mhz = CHANNELS[self.channel_idx]
        self.tuner.tune_mhz(mhz)
        self._prime_tries += 1
        if self._prime_tries >= 20:  # ~4 seconds
            self._prime_timer.stop()

    def _make_action(self, key, fn):
        act = QAction(self)
        act.setShortcut(key)
        act.triggered.connect(fn)
        return act

    def _overlay_label_css(self) -> str:
        return """
        QLabel {
            font-size: 18px;
            padding: 6px 10px;
            color: white;
            background: rgba(0,0,0,150);
            border: 1px solid rgba(255,255,255,80);
            border-radius: 10px;
        }
        """

    def _round_button_css(self, font_px: int = 26) -> str:
        return f"""
        QPushButton {{
            font-size: {font_px}px;
            font-weight: 600;
            color: white;
            background: rgba(0,0,0,170);
            border: 1px solid rgba(255,255,255,90);
            border-radius: 16px;
        }}
        QPushButton:pressed {{
            background: rgba(255,255,255,60);
        }}
        """

    def _pill_button_css(self) -> str:
        return """
        QPushButton {
            font-size: 18px;
            font-weight: 600;
            color: white;
            background: rgba(0,0,0,170);
            border: 1px solid rgba(255,255,255,90);
            border-radius: 16px;
            padding: 10px 14px;
        }
        QPushButton:pressed {
            background: rgba(255,255,255,60);
        }
        """

    def apply_channel(self):
        name, mhz = CHANNELS[self.channel_idx]
        self.tuner.tune_mhz(mhz)
        self.update_status_label()

    def update_status_label(self):
        name, mhz = CHANNELS[self.channel_idx]
        self.lbl_status.setText(f"{name}  {mhz} MHz   RSSI: {self.rssi_percent}%")

    def sync_channel_ui(self):
        self.combo_channel.blockSignals(True)
        self.combo_channel.setCurrentIndex(self.channel_idx)
        self.combo_channel.blockSignals(False)

    def on_channel_selected(self, idx: int):
        if idx < 0:
            return
        self.channel_idx = idx
        self.apply_channel()

    def step_channel(self, delta: int):
        self.channel_idx = (self.channel_idx + delta) % len(CHANNELS)
        self.sync_channel_ui()
        self.apply_channel()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("Windowed")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("Fullscreen")

    def restart_capture(self, record_path: str | None) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        pipeline = build_gst_pipeline_mjpg(
            device="/dev/video0",
            width=800,
            height=600,
            fps=30,
            record_path=record_path
        )

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            QMessageBox.critical(
                self,
                "Video error",
                "Could not start GStreamer pipeline.\n\n"
                "Try:\n"
                "gst-launch-1.0 v4l2src device=/dev/video0 ! "
                "image/jpeg,width=800,height=600,framerate=30/1 ! jpegdec ! videoconvert ! fakesink"
            )
            return False

        self.cap = cap
        
        # Retune RX5808 after opening video device (device opening might affect tuner state)
        self.apply_channel()
        time.sleep(0.1)
        
        return True

    def update_rec_label(self):
        if self.recording and self.record_start_monotonic is not None:
            elapsed = int(time.monotonic() - self.record_start_monotonic)
            self.lbl_rec.setText(f"● REC {format_hms(elapsed)}")
            # red-ish tint for rec label background
            self.lbl_rec.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    padding: 6px 10px;
                    color: white;
                    background: rgba(120,0,0,170);
                    border: 1px solid rgba(255,255,255,90);
                    border-radius: 10px;
                }
            """)
        else:
            self.lbl_rec.setText("")
            self.lbl_rec.setStyleSheet(self._overlay_label_css())

    def tick(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            return

        self.last_frame = frame

        # Render letterboxed video
        w = self.video_label.width()
        h = self.video_label.height()
        out = letterbox(frame, w, h)
        self.video_label.setPixmap(QPixmap.fromImage(bgr_to_qimage(out)))

        # Update overlays
        # RSSI still placeholder (0), but keep updating in case you add ADC later.
        self.update_status_label()
        self.update_rec_label()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.recording:
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = str(VIDEOS_DIR / f"fpv_{ts}.ts")

        if not self.restart_capture(record_path=self.record_path):
            self.record_path = None
            return

        self.recording = True
        self.record_start_monotonic = time.monotonic()
        self.btn_record_toggle.setText("Stop")
        self.btn_record_toggle.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: 700;
                color: white;
                background: rgba(140,0,0,200);
                border: 1px solid rgba(255,255,255,90);
                border-radius: 16px;
                padding: 10px 14px;
            }
            QPushButton:pressed {
                background: rgba(255,80,80,160);
            }
        """)

        QTimer.singleShot(200, self._start_radio_init_sequence)

    def stop_recording(self):
        if not self.recording:
            return

        if not self.restart_capture(record_path=None):
            return

        self.recording = False
        self.record_start_monotonic = None
        self.record_path = None
        self.btn_record_toggle.setText("Record")
        self.btn_record_toggle.setStyleSheet(self._pill_button_css())

        QTimer.singleShot(200, self._start_radio_init_sequence)

    def screenshot(self):
        if self.last_frame is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        cv2.imwrite(str(path), self.last_frame)

    def showEvent(self, event):
        """Handle window show to position overlays after widget is sized."""
        super().showEvent(event)
        # Use QTimer to position overlays after window is fully shown
        QTimer.singleShot(0, self._position_overlays)
    
    def resizeEvent(self, event):
        """Handle window resize to reposition overlays."""
        super().resizeEvent(event)
        self._position_overlays()
    
    def _position_overlays(self):
        """Position overlay widgets absolutely on top of video."""
        root = self.centralWidget()
        if root is None or root.width() <= 0 or root.height() <= 0:
            return
        
        # Video label fills entire widget (bottom layer)
        self.video_label.setGeometry(0, 0, root.width(), root.height())
        self.video_label.lower()  # Ensure video is on bottom layer
        
        # Position top overlay at top
        if hasattr(self, 'top_overlay') and self.top_overlay:
            top_height = max(self.top_overlay.sizeHint().height(), 60)  # Minimum height
            self.top_overlay.setGeometry(0, 0, root.width(), top_height)
            self.top_overlay.raise_()  # Ensure overlay is on top
        
        # Position bottom overlay at bottom
        if hasattr(self, 'bottom_overlay') and self.bottom_overlay:
            bottom_height = max(self.bottom_overlay.sizeHint().height(), 100)  # Minimum height
            self.bottom_overlay.setGeometry(0, root.height() - bottom_height, root.width(), bottom_height)
            self.bottom_overlay.raise_()  # Ensure overlay is on top
    
    def closeEvent(self, event):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication([])

    win = MainWindow()
    win.resize(1024, 600)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
