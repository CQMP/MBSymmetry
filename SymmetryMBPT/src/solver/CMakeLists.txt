project(solver)

add_library(symmetry-mbpt-solver-headers INTERFACE)

target_include_directories(symmetry-mbpt-solver-headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/solver>
        "$<INSTALL_INTERFACE:include/solver>")

add_library(solver-lib cuhf_solver_block.cpp cugw_solver_block.cpp)
target_link_libraries(solver-lib symmetry-mbpt-solver-headers common-lib
        ${ALPSCore_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_HL_LIBRARIES})

if(WITH_CUDA)
    target_link_libraries(solver-lib cuda-lib)
endif ()