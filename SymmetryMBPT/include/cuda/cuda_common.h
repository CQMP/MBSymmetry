#ifndef SYMMETRYMBPT_CUDA_COMMON_H
#define SYMMETRYMBPT_CUDA_COMMON_H

#include <stdexcept>
#include "cuComplex.h"
#include "cuda_types_map.h"

__global__ void acquire_lock(int *lock);
__global__ void release_lock(int *lock);

#endif //SYMMETRYMBPT_CUDA_COMMON_H
