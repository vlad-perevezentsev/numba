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
    src/pipelines/base_pipeline.cpp
    src/pipelines/lower_to_llvm.cpp
    src/pipelines/plier_to_linalg.cpp
    src/pipelines/plier_to_std.cpp
    src/rewrites/call_lowering.cpp
    src/rewrites/cast_lowering.cpp
    src/rewrites/type_conversion.cpp
    src/transforms/loop_utils.cpp
    src/transforms/pipeline_utils.cpp
    src/compiler.cpp
    src/dialect.cpp
    src/lowering.cpp
    src/module.cpp
    src/pipeline_registry.cpp
    src/utils.cpp
    )
set(HEADERS_LIST
    include/plier/dialect.hpp
    include/plier/PlierOps.td
    src/pipelines/base_pipeline.hpp
    src/pipelines/lower_to_llvm.hpp
    src/pipelines/plier_to_linalg.hpp
    src/pipelines/plier_to_std.hpp
    src/rewrites/call_lowering.hpp
    src/rewrites/cast_lowering.hpp
    src/rewrites/type_conversion.hpp
    src/transforms/loop_utils.hpp
    src/transforms/pipeline_utils.hpp
    src/compiler.hpp
    src/lowering.hpp
    src/pipeline_registry.hpp
    src/utils.hpp
    )

pybind11_add_module(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /EHsc)
endif ()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMTarget
    MLIRSupport
    MLIRLLVMIR
    MLIRStandard
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRStandardToLLVM
    MLIRLinalgTransforms
    MLIRSCFToStandard
    MLIRTensor
    MLIRTensorTransforms
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ./src
    ./include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include
    )

add_dependencies(${PROJECT_NAME} MLIRPlierOpsIncGen)
