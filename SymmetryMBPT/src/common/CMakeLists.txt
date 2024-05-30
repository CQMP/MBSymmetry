project(common)

add_library(symmetry-mbpt-common-headers INTERFACE)
target_include_directories(symmetry-mbpt-common-headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/common>
        "$<INSTALL_INTERFACE:include/common>")

add_library(common-lib
        block_utils.cpp
        mpi_utils.cpp
        params_t.cpp
        symmetry_utils.cpp
        transformer_t.cpp
        analysis.cpp
        )

set(t "common-lib")

target_link_libraries(${t} PUBLIC symmetry-mbpt-common-headers
        ${ALPSCore_LIBRARIES}
        ${LAPACK_LIBRARIES} ${HDF5_HL_LIBRARIES})# -fsanitize=address)

if(TARGET Eigen3::Eigen)
    target_link_libraries(${t} PUBLIC Eigen3::Eigen)
else(TARGET Eigen3::Eigen)
    target_compile_definitions(${t} PRIVATE ${EIGEN3_DEFINITIONS})
    target_include_directories(${t} PUBLIC ${EIGEN3_INCLUDE_DIRS})
endif(TARGET Eigen3::Eigen)
