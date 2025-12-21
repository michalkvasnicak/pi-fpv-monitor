# Raspberry Pi 5 FPV Monitor

## Requirements

- Raspberry Pi 5
- USB Analog Video Capture Device (USB Camera)
- RX5808 board with SPI interface

## Setup (Raspberry Pi OS 64-bit)

```bash
git clone <repo>
cd pi-fpv-monitor
./scripts/setup.sh
./scripts/run.sh
```

## Running over SSH but showing on Pi's display

```bash
export DISPLAY=:0
./scripts/run.sh
```
