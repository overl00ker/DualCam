cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project("DualFocus" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(Qt6_DIR "D:/Qt/6.10.0/msvc2022_64/lib/cmake/Qt6")

find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Charts
    OpenGLWidgets
    OpenGL 
)

set(OpenCV_INCLUDE_DIR "D:/opencv/build/include")
set(OpenCV_LIB_DIR "D:/opencv/build/x64/vc16/lib") 
add_definitions(-DOPENCV_WORLD)

add_executable(DualFocus
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

target_include_directories(DualFocus PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

target_link_libraries(DualFocus PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Charts
    Qt6::OpenGLWidgets
    Qt6::OpenGL 

    optimized "${OpenCV_LIB_DIR}/opencv_world4120.lib"
    debug "${OpenCV_LIB_DIR}/opencv_world4120d.lib"
)