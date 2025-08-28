cmake_minimum_required(VERSION 3.16)
project(DualCamQt LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(OpenCV_DIR "D:/IF NASU/Addons/opencv/build")
set(Qt6_DIR "C:/Qt/6.9.2/msvc2022_64/lib/cmake/Qt6")

find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)


add_executable(DualCam
    main.cpp
    DualCam.cpp
    DualCam.h
)


set_source_files_properties(main.cpp PROPERTIES SKIP_AUTOMOC ON)


if(UNIX AND NOT APPLE)
find_library(V4L2_LIB v4l2)
if(V4L2_LIB)
target_link_libraries(DualCam PRIVATE ${V4L2_LIB})
target_compile_definitions(DualCam PRIVATE HAVE_V4L2=1)
endif()
endif()


target_include_directories(DualCam PRIVATE ${OpenCV_INCLUDE_DIRS})



target_link_libraries(DualCam PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBS})