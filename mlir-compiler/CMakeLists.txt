cmake_minimum_required(VERSION 3.5)

project(mlir_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(HandleLLVMOptions)

set(SOURCES_LIST
    lowering.cpp
    module.cpp
    type_parser.cpp
    )
set(HEADERS_LIST
    lowering.hpp
    type_parser.hpp
    )

pybind11_add_module(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /EHsc)
endif ()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    MLIRSupport
    MLIRLLVMIR
    MLIRStandardOps
    MLIRTargetLLVMIR
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ./include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    )
