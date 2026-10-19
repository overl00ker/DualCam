#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [[ "${1:-}" == "--update" ]]; then
    git pull --ff-only
fi

have() { apt-cache show "$1" >/dev/null 2>&1; }

pkgs=(git build-essential cmake ninja-build pkg-config
      qt6-base-dev qt6-charts-dev libopencv-dev
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good)

datavis=""
for p in qt6-datavis3d-dev qt6-datavisualization-dev; do
    if have "$p"; then datavis="$p"; break; fi
done
if [[ -z "$datavis" ]]; then
    echo "Qt 6 Data Visualization (qt6-datavis3d-dev / qt6-datavisualization-dev) not found in apt." >&2
    exit 1
fi
pkgs+=("$datavis")

if have gstreamer1.0-libcamera; then pkgs+=(gstreamer1.0-libcamera); fi
if ! command -v rpicam-hello >/dev/null 2>&1 && have rpicam-apps; then pkgs+=(rpicam-apps); fi

sudo apt-get update
sudo apt-get install -y "${pkgs[@]}"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

app="$PWD/build/DualCam"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/dualcam.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=DualCam
Comment=Dual-channel imaging polarimeter
Exec=$app
Path=$PWD/build
Icon=$PWD/icon.png
Terminal=false
Categories=Science;Graphics;
EOF

echo
echo "DualCam built: $app"
echo "Start it from the applications menu (DualCam) or with: $app"
