cmake_minimum_required(VERSION 3.19)

project(SymmetryAdaptation C CXX )
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(USE_SPGLIB " Use spglib for symmetry analysis." ON)

find_package(ALPSCore REQUIRED COMPONENTS hdf5 params)
find_package(LAPACK REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 REQUIRED)
message(STATUS ${EIGEN3_INCLUDE_DIR})
find_package(spglib REQUIRED)
include_directories(${SPGLIB_INCLUDE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("setting build type to default, i.e. optimized build")
endif()

include_directories(${CMAKE_SOURCE_DIR}/deps/ndarrays)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(adapt_space src/space_group_adapt.cpp)
target_link_libraries(adapt_space group)

# Testing framework
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
  option(TEST_XML_OUTPUT "Enable XML output for tests" ON)
  add_subdirectory(test)
endif(BUILD_TESTS)

