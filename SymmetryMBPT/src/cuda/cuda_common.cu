#include "cuda_common.h"

__global__ void acquire_lock(int *lock){
  while (atomicCAS(lock, 0, 1) != 0)
    ;
}

__global__ void release_lock(int *lock){
  atomicExch(lock, 0);
}
