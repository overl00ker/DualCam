# DualCam

**DualCam** — настільний застосунок на C++/Qt + OpenCV для **Raspberry Pi 5**, який одночасно показує **два відеопотоки**, будує **карту різниці** та відстежує **різкість у реальному часі**.

Підтримуються будь‑які комбінації камер:
- 2× **libcamera** (офіційні модулі Raspberry Pi);
- 2× **V4L2** (наприклад, VEYE);
- **libcamera + V4L2**.

Обробка відео — через **GStreamer**:
- для RPi‑камер: `libcamerasrc -> videoconvert -> BGR/GRAY8 -> appsink`;
- для V4L2: `v4l2src -> (GRAY8/BGR) -> appsink`.

---

## Можливості

- **Дві камери паралельно.** Відображення «пліч‑о‑пліч» (Side‑by‑Side).
- **Карта різниці** між кадрами (diff) у псевдокольорі.
- **Графік різкості** (метрика: дисперсія Лапласа).
- **Freeze history.** У меню *Advanced* можна «заморозити» історію — графік перестає додавати нові точки.
- **Гнучкі параметри запуску.** Backend на кожну камеру (AUTO/libcamera/V4L2), розмір, FPS, caps‑формат, повний override GStreamer‑pipeline.
- **Безпечне відображення.** Всі кадри приводяться до BGR для коректної візуалізації.
- **Сумісність із VEYE.** Фіксований режим GRAY8 1440×1088 через V4L2.

---

## Залежності (runtime)

Для **Raspberry Pi OS (Bookworm)** встановіть:

```bash
sudo apt update

# Qt (рантайм)
sudo apt install -y libqt6widgets6 libqt6gui6 libqt6core6 libxcb1 libx11-6

# OpenCV (рантайм)
sudo apt install -y libopencv-core4.* libopencv-imgproc4.* libopencv-highgui4.*
# або повний пакет для розробки:
# sudo apt install -y libopencv-dev

# GStreamer (база та плагіни)
sudo apt install -y libgstreamer1.0-0 libgstreamer-plugins-base1.0-0   gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav

# libcamera (для libcamerasrc)
sudo apt install -y libcamera0 libcamera-tools
```

Перевірка наявності плагінів та утиліт:
```bash
gst-inspect-1.0 libcamerasrc
gst-inspect-1.0 v4l2src
libcamera-hello --version
```

---

## Збірка

Інструменти:
```bash
sudo apt install -y build-essential cmake qtbase5-dev qt6-base-dev libopencv-dev
```

Збірка проєкту:
```bash
git clone https://github.com/<your-username>/DualCam.git
cd DualCam
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Виконуваний файл: `./DualCam`

---

## Запуск (будь‑які комбінації камер)

Параметри задаються окремо для **cam0** та **cam1**.

Ключі (скорочено):
- `--camX-backend=auto|libcamera|v4l2`
- `--camX-id=<N>` (для libcamera)
- `--camX-device=/dev/videoN` (для V4L2)
- `--camX-size=WxH` (наприклад, `1280x960`)
- `--camX-fps=N`
- `--camX-gray=0|1` (примусово моно)
- `--camX-v4l2-fmt=GRAY8|BGR|...` (caps‑формат для `v4l2src`)
- `--camX-pipeline="...gst..."` (повний override GStreamer‑pipeline)
- `--camX-warmup-ms=NNN` (теплий старт)
- `--camX-verbose=0|1`

За замовчуванням: обидві камери `AUTO`, розмір `1280x960@30`, `device` — `/dev/video0` і `/dev/video1`, `v4l2-fmt=GRAY8`, `forceGray=false`.

### Приклади

**1) Обидві — libcamera (кольорові RPi‑модулі)**
```bash
./DualCam --cam0-backend=libcamera --cam0-id=0           --cam1-backend=libcamera --cam1-id=1           --cam0-size=1280x720 --cam1-size=1280x720           --cam0-fps=30 --cam1-fps=30
```

**2) Обидві — V4L2 (наприклад, дві VEYE)**
```bash
./DualCam --cam0-backend=v4l2 --cam0-device=/dev/video0           --cam1-backend=v4l2 --cam1-device=/dev/video1           --cam0-size=1440x1088 --cam1-size=1440x1088           --cam0-v4l2-fmt=GRAY8 --cam1-v4l2-fmt=GRAY8           --cam0-gray=1 --cam1-gray=1
```

**3) Змішано: libcamera + VEYE (V4L2)**
```bash
./DualCam --cam0-backend=libcamera --cam0-id=0 --cam0-size=1280x720 --cam0-fps=30           --cam1-backend=v4l2 --cam1-device=/dev/video0           --cam1-size=1440x1088 --cam1-v4l2-fmt=GRAY8 --cam1-gray=1 --cam1-fps=30
```

**4) Повний GStreamer‑override для камери**
```bash
./DualCam --cam0-pipeline="v4l2src device=/dev/video0 ! video/x-raw,format=GRAY8,width=1440,height=1088,framerate=30/1 ! appsink drop=true max-buffers=1 sync=false"
```

---

## Особливості для VEYE (V4L2)

Для плат ADP‑MV1 / MV‑серії на RPi 5:
1. Встановіть драйвер під вашу версію ядра, підключіть overlay.
2. Після перезавантаження підніміть медіаграф:
   ```bash
   ~/raspberrypi_v4l2/rpi5_scripts/media_setting_rpi5.sh mvcam -fmt RAW8 -w 1440 -h 1088
   ```
3. Запускайте DualCam строго з:
   - `--camX-backend=v4l2`
   - `--camX-device=/dev/videoN`
   - `--camX-size=1440x1088`
   - `--camX-v4l2-fmt=GRAY8`
   - (за потреби) `--camX-gray=1`

У меню зміна роздільної здатності **не застосовується** до VEYE в фіксованому режимі 1440×1088.

---

## Керування та інтерфейс

- **Camera**: вибір 1 або 2 камер, базові пресети роздільної здатності та FPS.
- **Advanced**: перемикачі *Difference* (карта різниці) і *Graph* (графік різкості), та *Freeze history*.
- **Window**: повноекранний режим, перезапуск застосунку.

Відображення:
- Верхній ряд — два потоки (ліворуч cam0, праворуч cam1).
- Середній — diff (якщо увімкнено).
- Нижній — графік різкості (якщо увімкнено).

---

## Діагностика та відомі нюанси

Перевірити пристрої:
```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

Перевірити формати/режими:
```bash
v4l2-ctl -d /dev/video0 --all
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Швидкий попередній перегляд поза DualCam:
```bash
# VEYE (GRAY8)
ffplay -f video4linux2 -video_size 1440x1088 -pixel_format gray -i /dev/video0

# RPi (libcamera)
gst-launch-1.0 libcamerasrc ! videoconvert ! autovideosink
```

Перевірка залежностей застосунку:
```bash
ldd ./DualCam | grep "not found" || echo "OK"
```

Поширені повідомлення:
- `QStandardPaths: wrong permissions on runtime directory /run/user/1000` — виправити: `sudo chmod 700 /run/user/1000` (або перевірте `XDG_RUNTIME_DIR`).
- `QCommandLineOption: Option names cannot start with a '-'` — у коді імена опцій задані без початкового дефіса, але запускати потрібно як звично: `--cam0-backend=...` тощо.

---

## Структура проєкту

```
.
├─ capture_backend.h   # єдиний backend: libcamera/V4L2, пайплайни, теплий старт
├─ MainWindow.h/.cpp   # UI, відмальовка, графік, diff, логіка двох камер
├─ main.cpp            # парсинг аргументів, створення вікна
├─ CMakeLists.txt
└─ README.md
```

---

## Ліцензія

Укажіть тип ліцензії (MIT/BSD/GPL тощо) та додайте файл `LICENSE` за потреби.

---

## Як залити в GitHub

Створіть/оновіть `README.md` у корені репозиторію:
```bash
git add README.md
git commit -m "Add README (uk)"
git push
```
