cmake_minimum_required(VERSION 3.16)
project(DualCam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(Qt6_DIR    "D:/IF NASU/Addons/Qt/6.9.1/msvc2022_64/lib/cmake/Qt6")
set(OpenCV_DIR "D:/IF NASU/Addons/opencv/build/x64/vc16/lib")


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(QT_REQUIRED_COMPONENTS Widgets)
find_package(Qt6 QUIET COMPONENTS ${QT_REQUIRED_COMPONENTS})
if (Qt6_FOUND)
  message(STATUS "Using Qt6")
  set(QT_LIB Qt6::Widgets)
else()
  find_package(Qt5 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})
  message(STATUS "Using Qt5")
  set(QT_LIB Qt5::Widgets)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)

set(DUALCAM_SOURCES
    src/main.cpp
    src/core.cpp
    src/mainwindow.cpp
)

set(DUALCAM_HEADERS
    src/core.h
    src/mainwindow.h
)

add_executable(${PROJECT_NAME} ${DUALCAM_SOURCES} ${DUALCAM_HEADERS})

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${QT_LIB}
    opencv::opencv 
    Threads::Threads
)

if(UNIX AND NOT APPLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
  )
endif()


