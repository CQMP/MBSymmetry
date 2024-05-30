cmake_minimum_required(VERSION 3.19)

project(SymmetryMBPT C CXX)
set(CMAKE_CXX_STANDARD 14)

option(WITH_CUDA "Build cuda programs" ON)
if (WITH_CUDA)
    find_package(CUDAToolkit)
    enable_language("CUDA")
    include(CheckLanguage)
    check_language(CUDA)
    add_compile_definitions(WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

find_package(ALPSCore REQUIRED COMPONENTS hdf5 params gf)
find_package(LAPACK REQUIRED)
find_package(HDF5 COMPONENTS C HL REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS ${EIGEN3_INCLUDE_DIR})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("setting build type to default, i.e. optimized build")
endif()

if(ENABLE_ASAN)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        message(FATAL_ERROR "Address Sanitizer is not supported on Intel Classic compiler.
        Use Intel oneAPI compiler (icpx) instead.")
    endif()
    add_compile_options(
            -fsanitize=address
            -fno-omit-frame-pointer
    )
    add_link_options(
            -fsanitize=address
    )
    # `add_link_options` only affects executables added after.
    #target_link_libraries(${BIN_NAME} -fsanitize=address)
endif()

set(INTEGRALS_PATH "" CACHE STRING "Path to realistic input data")
if(INTEGRALS_PATH STREQUAL "")
    add_definitions(-DINTEGRALS_PATH="${PROJECT_SOURCE_DIR}/data")
else()
    add_definitions(-DINTEGRALS_PATH="${INTEGRALS_PATH}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/libs)

add_subdirectory(src)

#add_executable(gw_solver
#        main.cpp
#        )
#target_link_libraries(gw_solver PUBLIC common-lib cuda-lib solver-lib scmbpt-lib)

# Testing framework
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
    option(TEST_XML_OUTPUT "Enable XML output for tests" ON)
    add_subdirectory(test)
endif(BUILD_TESTS)
