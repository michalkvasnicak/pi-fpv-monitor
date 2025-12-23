#!/usr/bin/env python3
import time
import itertools
from gpiozero import DigitalOutputDevice

FREQ_MHZ = 5825

# RUN WITH
# 
# ffplay -fflags nobuffer -flags low_delay -framedrop \
#   -f v4l2 -input_format mjpeg -video_size 800x600 -framerate 30 \
#   /dev/video0

# Put here the *Pi GPIO (BCM) numbers* you physically wired to RX5808 CH1/CH2/CH3.
# Example (you can change these):
#   Pi GPIO17 -> RX CH1
#   Pi GPIO27 -> RX CH2
#   Pi GPIO22 -> RX CH3
PIN_TO_CH = {
    "CH1": 17,   # <-- change if needed
    "CH2": 27,   # <-- change if needed
    "CH3": 22,   # <-- change if needed
}

# Bit-bang speed. If nothing ever works, increase to 20e-6 or even 50e-6.
T = 20e-6

SPI_ADDRESS_SYNTH_B = 0x01

def get_synth_register_b_freq_mhz(freq_mhz: int) -> int:
    # Common RX5808/RTC6715 formula used in many projects
    x = (freq_mhz - 479) // 2
    return ((x // 32) << 7) | (x % 32)

def build_25bit_word(address_bits: int, data_bits: int) -> int:
    # 25-bit word: addr(4) | (1<<4) | (data<<5)
    return (address_bits & 0x0F) | (1 << 4) | ((data_bits & 0xFFFF) << 5)

class BitBang:
    def __init__(self, pin_data: int, pin_clk: int, pin_le: int):
        self.data = DigitalOutputDevice(pin_data, initial_value=False)
        self.clk  = DigitalOutputDevice(pin_clk,  initial_value=False)
        self.le   = DigitalOutputDevice(pin_le,   initial_value=False)

    def cleanup(self):
        try:
            self.data.close()
            self.clk.close()
            self.le.close()
        except Exception:
            pass

    def _sleep(self):
        time.sleep(T)

    def _clk_pulse(self):
        self.clk.on();  self._sleep()
        self.clk.off(); self._sleep()

    def shift_25(self, word: int, bit_order: str):
        """Shift 25 bits on DATA with CLK pulses. bit_order: 'lsb' or 'msb'."""
        if bit_order == "lsb":
            bits = [(word >> i) & 1 for i in range(25)]
        else:
            bits = [(word >> i) & 1 for i in range(24, -1, -1)]

        for b in bits:
            self.data.value = bool(b)
            self._sleep()
            self._clk_pulse()

    def write_word(self, word: int, bit_order: str, le_idle: int, latch_edge: str):
        """
        le_idle: 0 or 1 (what LE sits at when idle)
        latch_edge: 'rising' or 'falling' (which transition latches)
        """
        # Set idle
        self.le.value = bool(le_idle)
        self._sleep()

        if latch_edge == "rising":
            # keep LE low during shift, then raise to latch
            self.le.off()
            self._sleep()
            self.shift_25(word, bit_order)
            self.le.on()
            self._sleep()
            # return to idle
            self.le.value = bool(le_idle)
        else:
            # keep LE high during shift, then drop to latch
            self.le.on()
            self._sleep()
            self.shift_25(word, bit_order)
            self.le.off()
            self._sleep()
            # return to idle
            self.le.value = bool(le_idle)

def main():
    data_bits = get_synth_register_b_freq_mhz(FREQ_MHZ)
    word = build_25bit_word(SPI_ADDRESS_SYNTH_B, data_bits)

    ch_pins = {
        "CH1": PIN_TO_CH["CH1"],
        "CH2": PIN_TO_CH["CH2"],
        "CH3": PIN_TO_CH["CH3"],
    }

    roles = ["DATA", "CLK", "LE"]
    ch_names = ["CH1", "CH2", "CH3"]

    print("RX5808 brute-force tuner")
    print(f"Target frequency: {FREQ_MHZ} MHz")
    print("Your wiring (Pi GPIO BCM):", ch_pins)
    print("Watch your video display. Press Ctrl+C when you see a good picture.\n")

    try:
        test_num = 0
        for perm in itertools.permutations(ch_names, 3):
            # perm assigns CHx to roles in order DATA, CLK, LE
            mapping = dict(zip(roles, perm))
            pin_data = ch_pins[mapping["DATA"]]
            pin_clk  = ch_pins[mapping["CLK"]]
            pin_le   = ch_pins[mapping["LE"]]

            for bit_order in ["lsb", "msb"]:
                for le_idle in [0, 1]:
                    for latch_edge in ["rising", "falling"]:
                        test_num += 1
                        print(
                            f"[{test_num:03d}] DATA={mapping['DATA']} (GPIO{pin_data}), "
                            f"CLK={mapping['CLK']} (GPIO{pin_clk}), "
                            f"LE={mapping['LE']} (GPIO{pin_le}) | "
                            f"order={bit_order} | LE_idle={le_idle} | latch={latch_edge}"
                        )

                        bb = BitBang(pin_data, pin_clk, pin_le)
                        # Send the tune word a few times to be safe
                        for _ in range(3):
                            bb.write_word(word, bit_order, le_idle, latch_edge)
                            time.sleep(0.01)
                        bb.cleanup()

                        # Give you time to observe the screen
                        time.sleep(4)

    except KeyboardInterrupt:
        print("\nStopped. If you saw video, scroll up to the last printed configuration.")
        return

if __name__ == "__main__":
    main()
