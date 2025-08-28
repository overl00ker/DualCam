# DualCam

A small Qt application for simultaneous capture of two video streams, displaying them in a 2×2 tiled view, computing a sharpness metric (Laplacian variance) for each stream, and visualizing the point where the sharpness curves intersect.

Supported sources:
- **V4L2 (Linux/Raspberry Pi)** — native capture via `mmap` with support for MJPG/YUYV/GREY/Y16/BGR24.
- **GStreamer** — arbitrary pipelines (e.g. `libcamerasrc` for CSI cameras on RPi).
- **OpenCV VideoCapture** — fallback path (MSMF/DirectShow on Windows, V4L2/GStreamer on Linux).

---

## Features
- Auto-detection of `/dev/video*` on Linux, fallback to `libcamerasrc` (2 pipelines by default).
- “2×2 tile” mode: top-left = CAM0, top-right = CAM1, bottom-left = absolute frame difference, bottom-right = sharpness plot with “equality” marker.
- Works with 8-bit mono and 16-bit Y16 (converted to 8-bit for display/metrics).
- Immediate frame updates without buffering (`buffersize=1`, `drop=true`).
- Exit with `Esc`/`Q`.

---

## Project structure
```
DualCam/
 ├─ CMakeLists.txt
 ├─ main.cpp
 ├─ DualCam.h / DualCam.cpp         # window logic, UI and frame processing
 ├─ CaptureBackend.h / .cpp         # unified I/O interface + backends (V4L2, OpenCV)
 └─ README.md                       # this file
```

---

## Quick Start (Windows 11, MSVC)
**Dependencies:**
- Qt 6.9.x (**MSVC 2022 64‑bit** kit)
- OpenCV 4.11.x (MSVC build; `opencv_world4110.dll` or equivalent)
- Visual Studio 2022 (C++ Desktop workload)

**Build:**
1. Open the project folder in Visual Studio (CMake project).
2. Ensure CMake can find Qt: in `CMakeLists.txt` set the path to MSVC Qt:
   ```
   set(CMAKE_PREFIX_PATH "C:/Qt/6.9.2/msvc2022_64")
   set(Qt6_DIR "C:/Qt/6.9.2/msvc2022_64/lib/cmake/Qt6")
   ```
3. Run **CMake → Delete Cache and Reconfigure**.
4. Build **x64‑Release**.

**Deploy & Run:**
1. Go to the folder containing `DualCam.exe` (e.g. `out/build/x64-release`).
2. Run:
   ```
   "C:/Qt/6.9.2/msvc2022_64/bin/windeployqt.exe" --release DualCam.exe
   ```
3. Ensure `platforms/qwindows.dll`, `imageformats/`, and DLLs (`D3DCompiler_47.dll`, `libEGL.dll`, `libGLESv2.dll`, `opengl32sw.dll`) are copied.
4. Place OpenCV DLLs (`opencv_world4110.dll`) next to `DualCam.exe`.
5. Launch `DualCam.exe`.

> Without connected cameras, the window will show black tiles and counters S0/S1=0 — this is expected.

---

## Quick Start (Raspberry Pi OS Bookworm, RPi 5)
**Dependencies (APT):**
```
sudo apt update
sudo apt install -y build-essential cmake qt6-base-dev libopencv-dev libv4l-dev     gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} libcamerasrc0
```

**Build:**
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

**Run** (see source selection below).

---

## Video sources: how to select
The factory `create_capture(...)` opens sources in this order:
1. Explicit **V4L2**: `v4l2:/dev/videoX` or `/dev/videoX` (Linux).
2. **GStreamer** if the string looks like a pipeline (`contains '!'` or `gst:` prefix).
3. **OpenCV VideoCapture** (any platform).

### Auto-detection (default)
- On Linux, `/dev/video*` devices are used first.
- If none are found, default `libcamerasrc` pipelines are substituted.

### Explicit GStreamer (RPi CSI)
Set environment variables before running:
```
GST_PIPELINE_CAM0="libcamerasrc ! video/x-raw,width=1280,height=720,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true" GST_PIPELINE_CAM1="libcamerasrc camera-id=1 ! video/x-raw,width=1280,height=720,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true" ./DualCam
```

### Explicit V4L2 (USB/UVC)
```
./DualCam                     # auto: uses /dev/video0 and /dev/video1 if available
```
> On Windows, devices `0`, `1` will be opened via MSMF/DirectShow through OpenCV.

---

## Controls
- `Esc` or `Q` — exit.
- Top-left overlay — current sharpness metrics `S0`, `S1` and difference `D`.

---

## Format notes
- **MJPG** — decoded to BGR.
- **YUYV** — converted to BGR via `COLOR_YUV2BGR_YUY2`.
- **GREY (8‑bit)** — stays single-channel; displayed as BGR.
- **Y16 (16‑bit mono)** — downscaled to 8‑bit for preview (`/256`). For proper scientific use, modify conversion (stretch min/max, gamma, etc.).

---

## Typical scenarios
- **RPi + CSI:** use `libcamerasrc`, set `camera-id=0/1`.
- **RPi + USB/UVC:** plug cameras → `/dev/video0`, `/dev/video1` → run `./DualCam`.
- **Windows + USB:** connect two webcams, run the app (detected as 0/1 by OpenCV).

---

## Debugging & troubleshooting
- Error *Qt platform plugin missing* on Windows → run `windeployqt`, ensure `platforms/qwindows.dll` is present.
- Blank window on Windows with no cameras → expected, plug a USB cam.
- `Y16` looks too dark → add auto-contrast/normalization in conversion (`CaptureBackend.cpp`).
- GStreamer errors on `libcamerasrc` → install package, check `/dev/media*` access.
- RPi Bookworm: don’t mix `libcamerasrc` and raw V4L2 for the same sensor.

---

## Build from source (cross-platform)
```
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release       -DQt6_DIR="C:/Qt/6.9.2/msvc2022_64/lib/cmake/Qt6"

# Build
cmake --build build -j

# Run
./build/DualCam                 # Linux
./build/Release/DualCam.exe     # Windows (after windeployqt)
```

---

## Roadmap
- CLI args `--cam0=... --cam1=... --size=WxH --fps=...`.
- Hardware timestamp & FPS meter.
- Optional dynamic range stretch for Y16.
- Hotkey source/pipeline switching.

---

## License
Check Qt and OpenCV licenses before redistributing binaries. Intended primarily for development and internal use.
