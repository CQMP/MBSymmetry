add_library(symmetry_adaptation-headers INTERFACE)

target_include_directories(symmetry_adaptation-headers INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        "$<INSTALL_INTERFACE:include>")

set(src_files
        basis.cpp
        wigner_d.cpp
        numeric.cpp
        simul_diag.cpp
        dirac_character.cpp
        molecule.cpp
        translation_vector.cpp
        k_mesh.cpp
        k_space_structure.cpp
        diagonalization.cpp
        spglib_anlys.cpp
        space_group.cpp
        space_group_rep.cpp
        )

add_library(group ${src_files})

set(t "group")

target_link_libraries(${t} PUBLIC symmetry_adaptation-headers ${ALPSCore_LIBRARIES} ${LAPACK_LIBRARIES})

if(TARGET Eigen3::Eigen)
    target_link_libraries(${t} PUBLIC Eigen3::Eigen)
else(TARGET Eigen3::Eigen)
    target_compile_definitions(${t} PRIVATE ${EIGEN3_DEFINITIONS})
    target_include_directories(${t} PUBLIC ${EIGEN3_INCLUDE_DIRS})
endif(TARGET Eigen3::Eigen)

target_link_libraries(${t} PUBLIC ${SPGLIB_LIBRARY})
