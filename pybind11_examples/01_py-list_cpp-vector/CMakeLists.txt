cmake_minimum_required(VERSION 3.0)
project(example C CXX)

find_package(OpenCV REQUIRED)
add_subdirectory(pybind11)

add_library(example SHARED example.cpp)
target_link_libraries(example ${OpenCV_LIBS} pybind11::module)


set_target_properties(example PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")

