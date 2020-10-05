cmake_minimum_required(VERSION 3.5)

project(mlir_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(include/plier)

set(SOURCES_LIST
    src/compiler.cpp
    src/dialect.cpp
    src/lowering.cpp
    src/module.cpp
    src/passes/plier_to_std.cpp
    src/type_parser.cpp
    src/utils.cpp
    )
set(HEADERS_LIST
    src/compiler.hpp
    src/lowering.hpp
    src/passes/plier_to_std.hpp
    src/type_parser.hpp
    src/utils.hpp
    include/plier/dialect.hpp
    include/plier/PlierOps.td
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
    MLIRStandard
    MLIRTargetLLVMIR
    MLIRTransforms
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ./include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include
    )

add_dependencies(${PROJECT_NAME} MLIRPlierOpsIncGen)
