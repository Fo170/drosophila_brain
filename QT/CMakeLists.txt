cmake_minimum_required(VERSION 3.20)
project(drosophila_brain VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
set(EIGEN_ROOT "/tmp/eigen-3.4.0")

add_executable(drosophila_brain
    src/main.cpp
    src/core/network.cpp
    src/world/world_3d.cpp
    src/render/glwidget.cpp
    src/render/brainchart.cpp
    src/ui/mainwindow.cpp
)

target_link_libraries(drosophila_brain PRIVATE
    Qt6::Core Qt6::Gui Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets
)

target_include_directories(drosophila_brain PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN_ROOT}
)

if(APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework OpenGL")
endif()

install(TARGETS drosophila_brain RUNTIME DESTINATION bin)
