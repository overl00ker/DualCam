# DualCam — сборка под твою конфигурацию (Windows)

Проверено по факту установленных у тебя версий (15.10.2026).

## Что стоит у тебя
| Компонент | Состояние |
|---|---|
| Qt | **6.11.1** в `D:\Qt` — есть и `mingw_64`, и `msvc2022_64` ✔ |
| Qt add-ons | Charts, DataVisualization, OpenGL, OpenGLWidgets ✔ |
| Инструменты из Qt | MinGW 13.1 (`D:\Qt\Tools\mingw1310_64`), CMake 3.30, Ninja ✔ |
| OpenCV | **5.0**: prebuilt под MSVC в `D:\opencv\build`, исходники в `D:\opencv\sources` ✔ |
| OpenCV под MinGW | собран из исходников в `D:\opencv\build_mingw\install` ✔ |
| Visual Studio / MSVC | ❌ стояла на `G:\MVStudio`; после объединения дисков её нет |

Пока MSVC нет, собирать нужно через MinGW (способ 1).

---

## Способ 1 — MinGW (работает сейчас)

```cmd
cmake --preset mingw-release
cmake --build out\build\mingw-release
```

`out\build\mingw-release\DualCam.exe` сразу запускается: после сборки рядом
автоматически раскладываются Qt (windeployqt), DLL OpenCV и рантайм MinGW.
Пресет сам добавляет MinGW и Ninja в PATH — нужен только `cmake`
(`D:\Qt\Tools\CMake_64\bin`). Тест EXIF: `ctest` в папке сборки.

### OpenCV под MinGW (один раз, уже сделано)
Prebuilt OpenCV собран MSVC, с MinGW он не линкуется. Сборка из исходников
(~10 минут на 16 потоках), из cmd с MinGW и Ninja в PATH:

```cmd
set PATH=D:\Qt\Tools\mingw1310_64\bin;D:\Qt\Tools\Ninja;D:\Qt\Tools\CMake_64\bin;%PATH%
cmake -G Ninja -S D:/opencv/sources -B D:/opencv/build_mingw ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=D:/opencv/build_mingw/install ^
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio,video,features,flann,calib,geometry,stereo,objdetect ^
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF ^
  -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF ^
  -DWITH_IPP=OFF -DWITH_FFMPEG=OFF -DWITH_MSMF=OFF -DWITH_DSHOW=ON -DWITH_OPENCL=OFF ^
  -DWITH_PROTOBUF=OFF -DWITH_TBB=OFF -DWITH_OPENEXR=OFF -DWITH_OBSENSOR=OFF -DCPU_BASELINE=AVX2
cmake --build D:/opencv/build_mingw --target install
```

Камеры на Windows идут через DirectShow (`WITH_DSHOW=ON`) — его достаточно.

---

## Способ 2 — MSVC (если снова поставишь VS 2022 / Build Tools)

В **«x64 Native Tools Command Prompt for VS 2022»**, из папки проекта:

```cmd
cmake --preset x64-release
cmake --build out\build\x64-release
```

Пути по умолчанию для MSVC уже прописаны в CMakeLists.txt:
`D:/Qt/6.11.1/msvc2022_64` и `D:/opencv/build`. DLL раскладываются так же
автоматически.

---

## OpenCV 5.0 vs 4.x (Pi)
Код собирается с обеими версиями. Одно отличие: в OpenCV 5 детектор AKAZE
переехал в `opencv_contrib`, поэтому третий запасной детектор при
автовыравнивании компилируется только с OpenCV 4 (на Pi). SIFT и ORB есть
в обеих версиях.
