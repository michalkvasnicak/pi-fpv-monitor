#!/usr/bin/env python3
import time
import datetime
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
    """LSB-first, latch on LE falling edge (your board)."""
    def __init__(self, pin_data: int, pin_clk: int, pin_le: int, le_idle: int = 1):
        self.data = DigitalOutputDevice(pin_data, initial_value=False)
        self.clk  = DigitalOutputDevice(pin_clk,  initial_value=False)
        self.le   = DigitalOutputDevice(pin_le,   initial_value=bool(le_idle))
        self.le_idle = bool(le_idle)

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
        self.tuner = RX5808Tuner(PIN_DATA, PIN_CLK, PIN_LE, le_idle=1)
        self.channel_idx = next((i for i, (_, mhz) in enumerate(CHANNELS) if mhz == 5825), 0)

        # State
        self.cap = None
        self.last_frame = None
        self.recording = False
        self.record_path = None
        self.record_start_monotonic = None
        self.rssi_percent = 0  # placeholder

        # Root
        root = QWidget()
        root.setStyleSheet("background: black;")
        self.setCentralWidget(root)

        # Main layout (video + overlays)
        main_v = QVBoxLayout(root)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)

        # Video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(QSize(1024, 600))
        main_v.addWidget(self.video_label, stretch=1)

        # Top overlay row
        top_row = QHBoxLayout()
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

        # Bottom overlay row: left buttons + right buttons
        bottom_row = QHBoxLayout()
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

        # Add overlays into main layout
        main_v.addLayout(top_row)
        main_v.addStretch(0)
        main_v.addLayout(bottom_row)

        # Init tuner + capture
        self.apply_channel()
        if not self.restart_capture(record_path=None):
            raise RuntimeError("Failed to start GStreamer capture pipeline.")

        # Timer for frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

        # Shortcuts
        self.addAction(self._make_action("A", lambda: self.step_channel(-1)))
        self.addAction(self._make_action("D", lambda: self.step_channel(+1)))
        self.addAction(self._make_action("F", self.toggle_fullscreen))
        self.addAction(self._make_action("S", self.screenshot))
        self.addAction(self._make_action("R", self.toggle_recording))

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

    def screenshot(self):
        if self.last_frame is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        cv2.imwrite(str(path), self.last_frame)

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
