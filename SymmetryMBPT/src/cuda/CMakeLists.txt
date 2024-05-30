project(cudaroutine)

if(WITH_CUDA)
    add_library(symmetry-mbpt-cuda-headers INTERFACE)
    target_include_directories(symmetry-mbpt-cuda-headers INTERFACE
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuda>
            "$<INSTALL_INTERFACE:include/cuda>")

    add_library(cuda-lib cublas_routines_prec.cu cuda_check.cpp cuda_common.cu
            cugw_qkpt_block.cu cugw_routines_block.cu)
    set_property(TARGET cuda-lib PROPERTY CUDA_ARCHITECTURES 70 72 80 86) #also check the same parameters in solvers
    target_link_libraries(cuda-lib PUBLIC CUDA::cublas CUDA::cusolver)
    target_link_libraries(cuda-lib PUBLIC symmetry-mbpt-cuda-headers common-lib ${ALPSCore_LIBRARIES})
endif()
