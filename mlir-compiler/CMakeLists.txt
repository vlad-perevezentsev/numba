cmake_minimum_required(VERSION 3.5)

project(mlir_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(plier)
add_subdirectory(mlir-compiler)
