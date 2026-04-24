cmake_minimum_required(VERSION 3.16)

project("DualCam" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if(WIN32)
    if(NOT DEFINED CACHE{OpenCV_DIR})
        set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib" CACHE PATH "OpenCV build directory")
    endif()
    if(NOT DEFINED CACHE{Qt6_DIR})
        set(Qt6_DIR "D:/Qt/6.10.0/msvc2022_64/lib/cmake/Qt6" CACHE PATH "Qt6 cmake directory")
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Charts
    OpenGLWidgets
    OpenGL 
)

add_executable(DualCam
    main.cpp
    mainwindow.cpp
    mainwindow.h
)

target_include_directories(DualCam PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(DualCam PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Charts
    Qt6::OpenGLWidgets
    Qt6::OpenGL 
    ${OpenCV_LIBS} 
)