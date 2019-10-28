cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fast_ans LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)

add_subdirectory(ext/pybind11-2.2.4)

pybind11_add_module(fast_ans
    src/python_interface.cpp
    src/ans.h
    src/gaussian.h
    src/tests.cpp
)
target_include_directories(fast_ans PRIVATE ext/eigen-3.3.7)
target_link_libraries(fast_ans PRIVATE
    OpenMP::OpenMP_CXX
)
