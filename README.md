<div align="center">

<img src="icon.png" alt="DualCam logo" width="120" />

# DualCam — Dual-Camera Analysis Tool

**Real-time synchronous capture, denoising, alignment and differential analysis for two-camera setups.**

Built with C++17 · Qt 6 · OpenCV 4 — runs on Windows 11 desktops and Raspberry Pi 5.

<sub>Developed for the scientific and metrology work of the Institute of Physics, National Academy of Sciences of Ukraine (IF NASU).</sub>

</div>

---

> **A note on the two READMEs.** The original Ukrainian write-up is preserved as [`README.uk.md`](README.uk.md). This file is the English reference and goes a little deeper into how the pipeline actually works.

---

## Table of contents

- [What it does](#what-it-does)
- [Why it exists](#why-it-exists)
- [Screenshots & figures](#screenshots--figures)
- [Feature tour](#feature-tour)
- [How the pipeline works](#how-the-pipeline-works)
- [The interface, panel by panel](#the-interface-panel-by-panel)
- [Saving data: snapshots, sidecars and EXIF](#saving-data-snapshots-sidecars-and-exif)
- [Dependencies](#dependencies)
- [Building from source](#building-from-source)
- [Running on a Raspberry Pi 5](#running-on-a-raspberry-pi-5)
- [Hotkeys & presets](#hotkeys--presets)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [License & credits](#license--credits)

---

## What it does

DualCam takes two camera feeds, runs them side by side, and gives you the tools to compare them *quantitatively* in real time. You can denoise each stream, measure how sharp each frame is, line the two views up on top of each other (automatically or by hand), and then look at the difference between them as a heat map. When something is worth keeping, one keystroke saves the frames together with a complete record of every setting that produced them.

It was written for lab work where two cameras look at the same scene — for example a reference channel and a measurement channel — and the interesting signal lives in the *difference* between them rather than in either image on its own.

## Why it exists

Most off-the-shelf camera viewers show you a picture. They don't tell you how sharp it is, they don't keep two streams in lock-step, and they certainly don't subtract one from the other while compensating for the fact that the two cameras are never mounted perfectly. DualCam was built to close that gap: a single, responsive desktop application that treats two cameras as one instrument and records enough metadata that a measurement can be reproduced later.

The whole computer-vision workload runs on a dedicated worker thread, so the heavy lifting (denoising, focus scoring, motion detection) never freezes the interface. Rendering is handed to the GPU through OpenGL, which keeps the UI fluid even at full frame rate on modest hardware.

---

## Screenshots & figures

> 📸 **This is where your own images go.** The application is highly visual, so a few good screenshots make the README far more convincing. Drop the files into [`docs/img/`](docs/img/) using the names below and they will appear automatically. Suggested captures, roughly in order of usefulness:
>
> | File to add | What it should show | Why it helps |
> |---|---|---|
> | `docs/img/hero.png` | The full main window with the side navigation visible and cameras off (clean placeholder views) | The single most important image — a calm, idle UI reads more clearly than a busy live scene; put it right under the title |
> | `docs/img/dual-mode.png` | Both cameras running side by side (the standard dual-view layout) | Shows the core two-camera workflow |
> | `docs/img/diff-mode.png` | Diff mode active, with the Jet heat map in the result view | Demonstrates the headline feature |
> | `docs/img/visualization-settings.png` | The visualization-settings dialog (noise floor, intensity stretch, colormap controls) | Documents how the difference view is tuned |
>
> If you later add **measurement results** (focus-vs-time curves, before/after denoise comparisons, a difference map next to the raw frames), those are exactly the kind of figures that belong in a "Results" or "Example session" section — tell me and I'll add one.

<!--
Once you add the files, uncomment this block:

<div align="center">
  <img src="docs/img/hero.png" alt="DualCam main window" width="80%" />
</div>
-->

---

## Feature tour

### Camera handling

- **Two streams, one instrument.** Cameras can be started and stopped independently or together. The capture backend adapts to the platform: GStreamer with `libcamerasrc` on the Raspberry Pi, and the default OpenCV backends (DirectShow / V4L2) on Windows and Linux.
- **Geometry correction at the source.** The second camera can be flipped horizontally or vertically on the fly, which is usually how you fix a mirror or an upside-down sensor mount before doing anything else.
- **Exposure control.** Gain and shutter can be driven manually, either shared across both cameras or set per-camera, which matters when you need the two channels radiometrically comparable.

### Image processing

- **Focus / sharpness scoring.** Every frame gets a sharpness number computed from the *variance of its Laplacian* — a standard, fast focus metric. Higher means sharper. Both cameras are scored continuously.
- **Temporal denoising.** A running exponential moving average (EMA) over the last *N* frames smooths out sensor noise. The buffer size `N` is adjustable; internally the blend factor is `α = 1/N`, so a larger buffer means heavier smoothing and more lag.
- **Spatial denoising.** An optional bilateral filter cleans up noise while preserving edges, with adjustable strength.
- **Motion detection.** A MOG2 background subtractor flags movement in the scene, with a tunable threshold, so you can tell a static measurement from one where something moved.

### Alignment (registration)

- **Automatic.** With one click the tool detects features in both frames and computes the homography that warps one onto the other. It tries **SIFT + FLANN** matching first and falls back to **ORB + brute-force Hamming** matching, with **RANSAC** rejecting bad correspondences and a sanity check that throws out degenerate transforms. (The UI labels the active-alignment indicator "ECC" for historical reasons.)
- **Manual, 6 degrees of freedom.** When you'd rather do it by eye, the alignment dialog gives you translation (Tx, Ty), scale/zoom, and three rotations (pitch, yaw, roll). The values are composed into a homography and applied live.

### Analysis & visualization

- **Live focus chart.** A Qt Charts plot at the bottom of the window traces the focus score of both cameras over time, with an adjustable history length.
- **Difference mode.** Subtracts the two aligned streams and renders the result as a **Jet colormap** heat map. A noise-floor control suppresses sensor noise below a threshold, and an intensity-stretch option normalizes the dynamic range so faint differences become visible.
- **Peak tracking.** Optionally finds and marks the brightest point(s) in the frame — handy for spot/beam tracking.
- **Profile and surface viewers.** Captured frames can be opened in auxiliary viewers: a line-profile plot (intensity along a drawn line) and a 3D surface plot of a region of interest.

### Workflow

- **GPU-accelerated rendering** through `QOpenGLWidget`, so display never bottlenecks the pipeline.
- **Fully remappable hotkeys** — every action can be bound to a key combination of your choice.
- **Presets** capture the entire processing configuration (pipeline parameters, alignment matrices, hotkeys) and reload it in one click.

---

## How the pipeline works

A frame's journey from sensor to screen, in order:

1. **Capture.** `CameraWorker`, a dedicated `QThread`, pulls frames from two `cv::VideoCapture` sources. Keeping this off the UI thread is what makes the interface stay responsive under load.
2. **Format conversion.** Each frame is converted to the chosen working format (native grayscale, OpenCV grayscale, or color).
3. **Temporal denoise.** If the time buffer is larger than 1, the frame is blended into a per-camera EMA accumulator via `cv::accumulateWeighted` with `α = 1/N`.
4. **Focus scoring.** The frame is run through a Laplacian, and the variance of the result (`stddev²`) becomes the focus score. This is cheap enough to do on every frame.
5. **Motion detection.** A `BackgroundSubtractorMOG2` produces a foreground mask; the proportion of moving pixels against the threshold decides whether motion is flagged.
6. **Hand-off.** The worker emits the two processed frames plus their focus scores and the motion flag back to the main window, which owns all the display and analysis logic.
7. **Alignment (optional).** If alignment is on, camera 2 is warped onto camera 1 using either the computed homography or the manual 6-DOF adjustment.
8. **Composition.** Depending on the mode, the main window shows the raw views, a fused overlay, or — in diff mode — the Jet-colored absolute difference, optionally noise-gated and contrast-stretched.
9. **Render.** The result is uploaded to a `GpuImageView` (an OpenGL widget) for drawing.

The automatic-alignment routine deserves a footnote: it runs on its own throwaway thread so a slow feature match never stalls the capture loop, detects features with SIFT (FLANN-matched) or ORB (brute-force-matched) as a fallback, estimates the homography with `cv::findHomography(..., RANSAC, ...)`, and rejects the result if the resulting transform isn't geometrically sane.

---

## The interface, panel by panel

**Main viewports.** Two camera views (Cam 1 / Cam 2) show the raw or processed streams; a Result view shows the composed output — overlay, alignment, or difference map.

**Side navigation.** A collapsible rail switches between the functional sections: Capture, Pipeline, Diff, Focus, Snapshot, Presets and Gallery.

**Capture.** Source selection (camera index or GStreamer pipeline), independent Start/Stop, the flip checkboxes, and exposure controls.

**Pipeline.** The denoise controls (time buffer / EMA and bilateral strength), the motion threshold, and the alignment controls — both the "calibrate automatically" button and the entry point to the manual 6-DOF dialog.

**Diff.** Noise-floor threshold, intensity-stretch toggle, and peak tracking for the difference view.

**Focus.** The large per-camera focus readouts and the real-time focus chart, with an adjustable history window.

**Snapshot.** Naming, the choice of which parameters to fold into the filename, and a recent-captures preview.

**Presets.** Save, load and delete named configuration profiles.

**Floating controls.** Persistent quick-access buttons (stream toggle, snapshot, mode toggle, focus toggle) plus status "pills" showing live FPS and alignment state. The **?** button opens the Actions & Keybinds menu and help.

---

## Saving data: snapshots, sidecars and EXIF

A snapshot is never just a picture — it's a picture plus a record of how it was made.

**Image formats.** You can save the two cameras *combined* into one image, *separately* as two files, or save the *difference* map.

**JSON sidecar.** Every snapshot is accompanied by a `.json` file written next to it, capturing the full state: capture mode and timestamp, an optional user-supplied name, color mode, flip flags, time-buffer size, motion threshold, fusion state, bilateral filter settings, noise floor, intensity-stretch and peak-tracking flags, the alignment matrices, per-camera exposure (gain and shutter), the live focus scores, the motion-active flag, diff-mode state and the frame count. In short: enough to reproduce the measurement.

**Embedded EXIF.** For JPEG output the application writes EXIF metadata directly into the file with a small hand-rolled writer (`exif_writer.cpp`) — image description, make/model, software tag, exposure time and ISO — so the key parameters travel *inside* the image even if the sidecar is separated from it.

---

## Dependencies

| Component | Requirement |
|---|---|
| Compiler | C++17 (MSVC 2022, or GCC 10+) |
| Build system | CMake 3.16 or newer |
| Qt 6 | `Core`, `Gui`, `Widgets`, `Charts`, `OpenGL`, `OpenGLWidgets` (and `DataVisualization` for the integrated surface viewer) |
| OpenCV 4.x | `core`, `imgproc`, `video` (background segmentation), `videoio`, `features2d`/`calib3d` (alignment) |
| Raspberry Pi only | `libcamera`, `gstreamer1.0` |

> The bundled `CMakeLists.txt` points `OpenCV_DIR` and `Qt6_DIR` at local Windows install paths (`D:/opencv/build/...`, `D:/Qt/6.10.0/...`). Adjust these to your machine, or pass them on the command line as shown below.

---

## Building from source

### Windows 11 (MSVC + Ninja)

1. Install Qt 6 (via the Qt Maintenance Tool) and OpenCV (prebuilt or self-compiled).
2. Open the **Developer Command Prompt for VS 2022**.
3. Configure and build:

```cmd
mkdir build && cd build
cmake -G "Ninja" ^
  -DQt6_DIR="C:/Qt/6.10.0/msvc2022_64/lib/cmake/Qt6" ^
  -DOpenCV_DIR="C:/opencv/build" ^
  -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

On Windows the build automatically runs `windeployqt` and copies the required OpenCV DLLs next to the executable, so the `build/` output is ready to run. A redistributable copy of the deployed application also lives in [`APP/`](APP/) (`DualCam.exe` with all its Qt/OpenCV DLLs).

### Linux desktop

```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

> **Build targets.** Besides the main `DualCam` executable, the project defines `test_exif` (a tiny EXIF round-trip check) and, on non-Windows builds, an optional standalone `DualCamViewer` controlled by the `DUALCAM_SEPARATE_VIEWER` option.

---

## Running on a Raspberry Pi 5

Install the toolchain and Qt/OpenCV from the distribution packages:

```bash
sudo apt update
sudo apt install qt6-base-dev qt6-charts-dev libopencv-dev \
                 cmake ninja-build libgstreamer1.0-dev
```

Then build as above. On the Pi the capture path uses a GStreamer `libcamerasrc` pipeline and expects two `libcamera`-visible cameras; if fewer than two are found the app warns and attempts a fallback.

---

## Hotkeys & presets

Every action in the app — starting cameras, taking a snapshot, toggling diff or focus mode, opening dialogs — is a named command that you can bind to any key combination. Open **Actions & Keybinds** from the **?** button to see and remap them.

Your keybindings, pipeline parameters and calibration matrices are persisted to the OS configuration location (`AppConfigLocation`) automatically, and named **presets** let you keep several complete configurations and switch between them instantly — useful when you routinely run more than one kind of measurement.

---

## Project layout

```
DualCamQt/
├── main.cpp              # Application entry point
├── mainwindow.h/.cpp     # UI, rendering, analysis, presets, hotkeys,
│                         #   plus the CameraWorker capture/processing thread
├── exif_writer.h/.cpp    # Minimal JPEG EXIF reader/writer
├── viewer_*.cpp/.h        # Auxiliary profile / surface viewers (optional target)
├── test_exif.cpp         # EXIF round-trip test target
├── CMakeLists.txt        # Build configuration
├── CMakePresets.json     # CMake preset definitions
├── resources.qrc         # Qt resources (icons)
├── appicon.rc            # Windows icon resource
├── DESIGN.md             # The "Adrenaline" dark UI design system
├── APP/                  # Deployed, ready-to-run Windows build
└── docs/img/             # Screenshots & figures (add your own here)
```

The bulk of the application lives in `mainwindow.cpp` — it owns the window, the OpenGL views, the charts, the snapshot logic and the `CameraWorker` thread that does the actual capture and per-frame processing.

---

## Troubleshooting

**The app builds but can't find Qt or OpenCV at configure time.** Point `Qt6_DIR` and `OpenCV_DIR` at your own install paths, either by editing `CMakeLists.txt` or by passing `-D...` on the `cmake` command line.

**It runs but a camera view stays blank.** Check the source selection in the Capture panel — on Windows the camera index must match a device DirectShow can see; on the Pi the two cameras must be visible to `libcamera`.

**Automatic alignment fails or looks wrong.** Feature-based registration needs texture to lock onto. On low-contrast or near-identical scenes it can fail the internal sanity check; fall back to the manual 6-DOF dialog and align by eye.

**The difference map is all noise.** Raise the noise floor in the Diff panel, and make sure the two streams are actually aligned first — an un-aligned difference is dominated by registration error, not by the signal you care about.

---

## License & credits

Developed for the scientific and research work of the **Institute of Physics, National Academy of Sciences of Ukraine (IF NASU)**.

Built on the shoulders of the [Qt](https://www.qt.io/) and [OpenCV](https://opencv.org/) projects. The visual design follows the in-house "Adrenaline" dark theme documented in [`DESIGN.md`](DESIGN.md).

> Add a `LICENSE` file (e.g. MIT, or your institution's preferred terms) and reference it here once chosen.
