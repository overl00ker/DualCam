cmake_minimum_required(VERSION 3.16)
project(DualCamQt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    main.cpp
    MainWindow.cpp
    MainWindow.h
    capture_backend.h
)

target_link_libraries(${PROJECT_NAME}
    Qt6::Widgets
    ${OpenCV_LIBS}
)
