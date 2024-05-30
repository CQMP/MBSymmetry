project(cpuref)

add_library(symmetry-mbpt-cpu-headers INTERFACE)

target_include_directories(symmetry-mbpt-cpu-headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cpu_ref>
        "$<INSTALL_INTERFACE:include/cpu_ref>")

add_library(cpu-lib dyson.cpp dyson_block.cpp gw.cpp gw_block.cpp)
target_link_libraries(cpu-lib symmetry-mbpt-cpu-headers common-lib
        ${ALPSCore_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_HL_LIBRARIES})
