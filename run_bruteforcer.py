#!/usr/bin/env python3
"""
Script to run ffplay and bruteforcer together, with proper GPIO cleanup on exit.
"""
import subprocess
import signal
import sys
import threading
import time
from gpiozero import DigitalOutputDevice

# Import from bruteforcer
from rx58080_bruteforcer import (
    PIN_TO_CH, FREQ_MHZ, T, SPI_ADDRESS_SYNTH_B,
    get_synth_register_b_freq_mhz, build_25bit_word, BitBang
)
import itertools

# GPIO pins that need to be reset
GPIO_PINS = {
    "CH1": PIN_TO_CH["CH1"],
    "CH2": PIN_TO_CH["CH2"],
    "CH3": PIN_TO_CH["CH3"],
}

# Global references for cleanup
ffplay_process = None
bruteforcer_thread = None
bruteforcer_running = False
bitbang_instances = []


def reset_gpio_pins():
    """Reset all GPIO pins to safe default states."""
    print("\nResetting GPIO pins to default states...")
    
    # Reset all pins used by bruteforcer to safe defaults
    # Default: DATA=LOW, CLK=LOW, LE=HIGH (idle state)
    pins_to_reset = []
    
    try:
        # Small delay to ensure all previous GPIO operations are complete
        time.sleep(0.05)
        
        # Reset all pins - set all to LOW first (safest state)
        for ch_name, pin_num in GPIO_PINS.items():
            try:
                pin = DigitalOutputDevice(pin_num, initial_value=False)
                pins_to_reset.append((pin, ch_name, pin_num))
                pin.off()  # Ensure LOW
            except Exception as e:
                print(f"Warning: Could not reset GPIO{pin_num} ({ch_name}): {e}")
        
        time.sleep(0.01)
        
        # Set CH2 (commonly used for LE) to HIGH as default idle state
        # This matches the typical hardware configuration where LE has pull-up
        ch2_pin_num = GPIO_PINS.get("CH2")
        if ch2_pin_num:
            try:
                # Close the LOW instance first if it exists
                for pin, name, pnum in pins_to_reset[:]:
                    if pnum == ch2_pin_num:
                        try:
                            pin.close()
                        except Exception:
                            pass
                        pins_to_reset.remove((pin, name, pnum))
                        break
                
                # Create new instance with HIGH (idle state)
                le_pin = DigitalOutputDevice(ch2_pin_num, initial_value=True)
                le_pin.on()  # Ensure HIGH
                pins_to_reset.append((le_pin, "LE", ch2_pin_num))
                time.sleep(0.01)
            except Exception as e:
                print(f"Warning: Could not set LE pin (GPIO{ch2_pin_num}) HIGH: {e}")
        
        time.sleep(0.01)
        
        # Close all pins (they will return to their default pull-up/pull-down states)
        for pin, name, pnum in pins_to_reset:
            try:
                pin.close()
            except Exception:
                pass
        
        print(f"GPIO pins reset: GPIO{GPIO_PINS['CH1']}, GPIO{GPIO_PINS['CH2']}, GPIO{GPIO_PINS['CH3']}")
        print("  (All pins closed, returning to default pull-up/pull-down states)")
        
    except Exception as e:
        print(f"Error during GPIO reset: {e}")


def run_bruteforcer():
    """Run the bruteforcer in a separate thread."""
    global bruteforcer_running, bitbang_instances
    
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
    
    bruteforcer_running = True
    
    try:
        test_num = 0
        for perm in itertools.permutations(ch_names, 3):
            if not bruteforcer_running:
                break
                
            # perm assigns CHx to roles in order DATA, CLK, LE
            mapping = dict(zip(roles, perm))
            pin_data = ch_pins[mapping["DATA"]]
            pin_clk  = ch_pins[mapping["CLK"]]
            pin_le   = ch_pins[mapping["LE"]]
            
            for bit_order in ["lsb", "msb"]:
                if not bruteforcer_running:
                    break
                    
                for le_idle in [0, 1]:
                    if not bruteforcer_running:
                        break
                        
                    for latch_edge in ["rising", "falling"]:
                        if not bruteforcer_running:
                            break
                            
                        test_num += 1
                        print(
                            f"[{test_num:03d}] DATA={mapping['DATA']} (GPIO{pin_data}), "
                            f"CLK={mapping['CLK']} (GPIO{pin_clk}), "
                            f"LE={mapping['LE']} (GPIO{pin_le}) | "
                            f"order={bit_order} | LE_idle={le_idle} | latch={latch_edge}"
                        )
                        
                        bb = BitBang(pin_data, pin_clk, pin_le)
                        bitbang_instances.append(bb)
                        
                        # Ensure pins are in correct state before tuning
                        bb.data.off()
                        bb.clk.off()
                        
                        # Send the tune word a few times to be safe
                        data_bits = get_synth_register_b_freq_mhz(FREQ_MHZ)
                        word = build_25bit_word(SPI_ADDRESS_SYNTH_B, data_bits)
                        for _ in range(3):
                            if not bruteforcer_running:
                                break
                            bb.write_word(word, bit_order, le_idle, latch_edge)
                            time.sleep(0.01)
                        
                        # Cleanup this instance
                        bb.cleanup()
                        bitbang_instances.remove(bb)
                        
                        # Give you time to observe the screen
                        if bruteforcer_running:
                            time.sleep(4)
    
    except Exception as e:
        print(f"\nBruteforcer error: {e}")
    finally:
        bruteforcer_running = False
        # Cleanup any remaining bitbang instances
        for bb in bitbang_instances[:]:
            try:
                bb.cleanup()
            except Exception:
                pass
        bitbang_instances.clear()


def cleanup_all():
    """Cleanup function called on exit."""
    global ffplay_process, bruteforcer_running, bitbang_instances
    
    print("\nCleaning up...")
    
    # Stop bruteforcer
    bruteforcer_running = False
    
    # Cleanup all bitbang instances
    for bb in bitbang_instances[:]:
        try:
            bb.cleanup()
        except Exception:
            pass
    bitbang_instances.clear()
    
    # Stop ffplay
    if ffplay_process:
        try:
            ffplay_process.terminate()
            ffplay_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffplay_process.kill()
        except Exception:
            pass
        ffplay_process = None
    
    # Reset GPIO pins
    reset_gpio_pins()
    
    print("Cleanup complete.")


def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    cleanup_all()
    sys.exit(0)


def main():
    global ffplay_process, bruteforcer_thread
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build ffplay command
    ffplay_cmd = [
        "ffplay",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", "800x600",
        "-framerate", "30",
        "/dev/video0"
    ]
    
    print("Starting ffplay and bruteforcer...")
    print("Press Ctrl+C to stop and reset GPIO pins.\n")
    
    try:
        # Start ffplay
        ffplay_process = subprocess.Popen(
            ffplay_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Start bruteforcer in a separate thread
        bruteforcer_thread = threading.Thread(target=run_bruteforcer, daemon=True)
        bruteforcer_thread.start()
        
        # Wait for ffplay to finish (user closes window) or for interrupt
        try:
            ffplay_process.wait()
        except KeyboardInterrupt:
            pass
        
    except FileNotFoundError:
        print("Error: ffplay not found. Please install ffmpeg.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup_all()


if __name__ == "__main__":
    main()

