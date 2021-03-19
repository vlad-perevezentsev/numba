cmake_minimum_required(VERSION 3.5)

project(mlir_compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(mlir-compiler)

if(DEFINED DPCOMP_TREE)
    message(STATUS "Out of tree PLIER is used")
    add_subdirectory(${DPCOMP_TREE} plier)
elseif(DEFINED DPCOMP_DIR)
    message(STATUS "PLIER from DPCOMP_DIR is used")
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${DPCOMP_DIR}/include
        )
    target_link_directories(${PROJECT_NAME} PRIVATE
        ${DPCOMP_DIR}
        ${DPCOMP_DIR}/Release
        )
else()
    message(STATUS "PLIER from local directory is used")
    add_subdirectory(plier)
endif()
