cmake_minimum_required(VERSION 3.16)

project("DualCam" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

option(DUALCAM_SEPARATE_VIEWER
       "Open the profile / 3-D surface viewers in the separate DualCamViewer process" OFF)

if(WIN32)
    if(MINGW)
        set(_dualcam_qt_kit "mingw_64")
        set(_dualcam_opencv "D:/opencv/build_mingw/install")
    else()
        set(_dualcam_qt_kit "msvc2022_64")
        set(_dualcam_opencv "D:/opencv/build")
    endif()
    if(NOT DEFINED CACHE{OpenCV_DIR})
        set(OpenCV_DIR "${_dualcam_opencv}" CACHE PATH "OpenCV build directory")
    endif()
    if(NOT DEFINED CACHE{Qt6_DIR})
        set(Qt6_DIR "D:/Qt/6.11.1/${_dualcam_qt_kit}/lib/cmake/Qt6" CACHE PATH "Qt6 cmake directory")
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
    DataVisualization
)

set(DUALCAM_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    exif_writer.cpp
    exif_writer.h
    viewer_dialogs.cpp
    viewer_dialogs.h
    resource_monitor.cpp
    resource_monitor.h
    resources.qrc
)

add_executable(DualCam ${DUALCAM_SOURCES})

if(WIN32)
    add_library(DualCamIcon OBJECT appicon.rc)
    set_target_properties(DualCamIcon PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    target_sources(DualCam PRIVATE $<TARGET_OBJECTS:DualCamIcon>)
endif()

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
    Qt6::DataVisualization
    ${OpenCV_LIBS}
)

if(DUALCAM_SEPARATE_VIEWER)
    target_compile_definitions(DualCam PRIVATE DUALCAM_SEPARATE_VIEWER)
endif()

add_executable(DualCamViewer
    viewer_main.cpp
    viewer_dialogs.cpp
    viewer_dialogs.h
)

target_include_directories(DualCamViewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(DualCamViewer PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Charts
    Qt6::OpenGLWidgets
    Qt6::OpenGL
    Qt6::DataVisualization
)

enable_testing()
add_executable(test_exif test_exif.cpp exif_writer.cpp exif_writer.h)
target_include_directories(test_exif PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_exif PRIVATE ${OpenCV_LIBS})
add_test(NAME exif_roundtrip COMMAND test_exif)

if(WIN32)
    if(MINGW)
        set(_dualcam_runtime --compiler-runtime)
    else()
        set(_dualcam_runtime --no-compiler-runtime)
    endif()
    if(TARGET Qt6::windeployqt)
        add_custom_command(TARGET DualCam POST_BUILD
            COMMAND Qt6::windeployqt --no-translations ${_dualcam_runtime} "$<TARGET_FILE:DualCam>"
            VERBATIM)
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        foreach(_dualcam_target DualCam test_exif)
            add_custom_command(TARGET ${_dualcam_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_RUNTIME_DLLS:${_dualcam_target}> "$<TARGET_FILE_DIR:${_dualcam_target}>"
                COMMAND_EXPAND_LISTS)
        endforeach()
    else()
        message(WARNING "CMake >= 3.21 is needed to copy the OpenCV DLLs next to DualCam.exe")
    endif()
endif()
