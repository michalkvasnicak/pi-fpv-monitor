import time
import threading
import cv2

from gpiozero import Button, DigitalOutputDevice

# ----------------------------
# GPIO CONFIG (BCM numbering)
# ----------------------------
# RX5808 control pins (after SPI hack)
PIN_DATA = 17  # Pi -> RX5808 DATA  (to CH1)
PIN_CLK  = 27  # Pi -> RX5808 CLK   (to CH2)
PIN_LE   = 22  # Pi -> RX5808 LE/CS (to CH3)

# Buttons (to GND when pressed)
PIN_BTN_PLUS  = 5
PIN_BTN_MINUS = 6

# ----------------------------
# Channel list (5.8GHz analog)
# Bands: A, B, E, F, R (Raceband)
# Source frequency table: Oscar Liang channel chart
# ----------------------------
CHANNELS = [
    ("A1", 5865), ("A2", 5845), ("A3", 5825), ("A4", 5805), ("A5", 5785), ("A6", 5765), ("A7", 5745), ("A8", 5725),
    ("B1", 5733), ("B2", 5752), ("B3", 5771), ("B4", 5790), ("B5", 5809), ("B6", 5828), ("B7", 5847), ("B8", 5866),
    ("E1", 5705), ("E2", 5685), ("E3", 5665), ("E4", 5645), ("E5", 5885), ("E6", 5905), ("E7", 5925), ("E8", 5945),
    ("F1", 5740), ("F2", 5760), ("F3", 5780), ("F4", 5800), ("F5", 5820), ("F6", 5840), ("F7", 5860), ("F8", 5880),
    ("R1", 5658), ("R2", 5695), ("R3", 5732), ("R4", 5769), ("R5", 5806), ("R6", 5843), ("R7", 5880), ("R8", 5917),
]
# :contentReference[oaicite:2]{index=2}

# ----------------------------
# RX5808 serial write helpers
# ----------------------------
SPI_ADDRESS_SYNTH_B = 0x01  # from common RX5808 examples
# :contentReference[oaicite:3]{index=3}

def get_synth_register_b_freq_mhz(freq_mhz: int) -> int:
    """
    Calculate RX5808 Synth Register B value for a given frequency in MHz.
    Formula used in common RX5808 examples. :contentReference[oaicite:4]{index=4}
    """
    # ((((f - 479) / 2) / 32) << 7) | (((f - 479) / 2) % 32);
    x = (freq_mhz - 479) // 2
    return ((x // 32) << 7) | (x % 32)

def build_25bit_word(address_bits: int, data_bits: int) -> int:
    """
    Build 25-bit word: address(4 bits) | (1<<4) | (data_bits<<5)
    Matches common RX5808 write examples. :contentReference[oaicite:5]{index=5}
    """
    return (address_bits & 0x0F) | (1 << 4) | ((data_bits & 0xFFFF) << 5)

class RX5808Tuner:
    def __init__(self, pin_data: int, pin_clk: int, pin_le: int):
        self.data = DigitalOutputDevice(pin_data, initial_value=False)
        self.clk  = DigitalOutputDevice(pin_clk,  initial_value=False)
        self.le   = DigitalOutputDevice(pin_le,   initial_value=True)  # idle high is common
        self.lock = threading.Lock()

        # Conservative bit-bang timing (microseconds). Python isn't real-time; slower is safer.
        self.t_setup = 2e-6
        self.t_hold  = 2e-6

    def _clock_pulse(self):
        self.clk.on()
        time.sleep(self.t_hold)
        self.clk.off()
        time.sleep(self.t_hold)

    def write_25bits_lsb_first(self, word: int):
        """
        Shift out 25 bits, LSB first, latch on LE rising edge. :contentReference[oaicite:6]{index=6}
        """
        # Begin transaction: LE low
        self.le.off()
        time.sleep(self.t_setup)

        for bit in range(25):
            if (word >> bit) & 1:
                self.data.on()
            else:
                self.data.off()
            time.sleep(self.t_setup)
            self._clock_pulse()

        # Latch: LE high
        self.le.on()
        time.sleep(self.t_setup)

    def tune_mhz(self, freq_mhz: int):
        data_bits = get_synth_register_b_freq_mhz(freq_mhz)
        word = build_25bit_word(SPI_ADDRESS_SYNTH_B, data_bits)

        with self.lock:
            self.write_25bits_lsb_first(word)

# ----------------------------
# Main app
# ----------------------------
def main():
    tuner = RX5808Tuner(PIN_DATA, PIN_CLK, PIN_LE)

    # Buttons with internal pull-ups (wire button to GND)
    btn_plus = Button(PIN_BTN_PLUS, pull_up=True, bounce_time=0.08)
    btn_minus = Button(PIN_BTN_MINUS, pull_up=True, bounce_time=0.08)

    idx = 0
    name, mhz = CHANNELS[idx]
    tuner.tune_mhz(mhz)

    def set_channel(new_idx: int):
        nonlocal idx, name, mhz
        idx = new_idx % len(CHANNELS)
        name, mhz = CHANNELS[idx]
        tuner.tune_mhz(mhz)
        # small settle delay helps video lock
        time.sleep(0.01)

    btn_plus.when_pressed = lambda: set_channel(idx + 1)
    btn_minus.when_pressed = lambda: set_channel(idx - 1)

    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0. Check your capture device and permissions.")

    print("Running. Use buttons for +/- channels. Press 'q' in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        # Overlay current channel
        overlay = f"{name}  {mhz} MHz   (idx {idx+1}/{len(CHANNELS)})"
        cv2.putText(frame, overlay, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("RX5808 Viewer (/dev/video0)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()