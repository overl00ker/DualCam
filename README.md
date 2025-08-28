# DualCamQt for Raspberry Pi 5 (IMX296, Qt6, GStreamer, OpenCV)

## System requirements
- Raspberry Pi OS Bookworm (64-bit)
- Raspberry Pi 5
- Dual IMX296 camera modules (via libcamera)

---

## 1. Enable GUI environment
```bash
sudo apt update
sudo apt install -y raspberrypi-ui-mods
sudo systemctl enable lightdm
sudo systemctl set-default graphical.target
sudo reboot
```

## 2. Install dependencies
```bash
sudo apt install -y \
  cmake \
  qt6-base-dev \
  libopencv-dev \
  gstreamer1.0-libcamera \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev
```

## 3. Fix permissions (if needed)
```bash
sudo chmod 700 /run/user/1000
```

---

## 4. Clone project and build
```bash
git clone https://github.com/overl00ker/DualCamQt
cd DualCamQt
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 5. Get camera names
```bash
libcamera-hello --list-cameras
```

Example output:
```
0 : imx296 [.../i2c@88000/imx296@1a]
1 : imx296 [.../i2c@80000/imx296@1a]
```

---

## 6. Run app (example resolution 1280x960 @60fps)
```bash
./DualCamQt 1280 960 60
```

---

## Hotkeys
- `Esc`   — exit
