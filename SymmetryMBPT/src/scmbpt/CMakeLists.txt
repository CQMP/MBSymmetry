project(scmbpt)

add_library(symmetry-mbpt-scmbpt-headers INTERFACE)
target_include_directories(symmetry-mbpt-scmbpt-headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/scmbpt>
        "$<INSTALL_INTERFACE:include/scmbpt>")

add_library(scmbpt-lib
        sc_loop_block_t.cpp
        sc_type_block.cpp
        sc_loop_gf_block.cpp
        )
target_link_libraries(scmbpt-lib symmetry-mbpt-scmbpt-headers common-lib
        ${ALPSCore_LIBRARIES} ${LAPACK_LIBRARIES} ${HDF5_HL_LIBRARIES})

target_link_libraries(scmbpt-lib solver-lib)
