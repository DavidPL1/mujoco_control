cmake_minimum_required(VERSION 3.15...3.27)
project(mujoco_ik_control)

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not set. Defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# set(PYBIND_FINDPYTHON ON)
find_library(GLFW libglfw.so.3)
find_package(Eigen3 REQUIRED)
find_package(mujoco REQUIRED)

pybind11_add_module(opspace_ctrl
    src/opspace_ctrl.cc
)
target_link_libraries(opspace_ctrl PRIVATE
    mujoco::mujoco
    Eigen3::Eigen
    pybind11::module
)
target_compile_definitions(opspace_ctrl
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

install(
  TARGETS
    opspace_ctrl
  DESTINATION
    ${PROJECT_NAME}
)