#ifndef SYMMETRYMBPT_CUBLAS_ROUTINE_PREC_H
#define SYMMETRYMBPT_CUBLAS_ROUTINE_PREC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>


cublasStatus_t ASUM(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
cublasStatus_t ASUM(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);

cublasStatus_t RSCAL(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);
cublasStatus_t RSCAL(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);

cublasStatus_t RAXPY(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);
cublasStatus_t RAXPY(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);

cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n, int k,
                    const cuDoubleComplex *alpha,
                    const cuDoubleComplex *A, int lda,
                    const cuDoubleComplex *B, int ldb,
                    const cuDoubleComplex *beta,
                    cuDoubleComplex *C, int ldc);
cublasStatus_t GEMM(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n, int k,
                    const cuComplex *alpha,
                    const cuComplex *A, int lda,
                    const cuComplex *B, int ldb,
                    const cuComplex *beta,
                    cuComplex *C, int ldc);

cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
                                    cublasOperation_t transa,
                                    cublasOperation_t transb,
                                    int m, int n, int k,
                                    const cuDoubleComplex *alpha,
                                    const cuDoubleComplex *A, int lda,
                                    long long int          strideA,
                                    const cuDoubleComplex *B, int ldb,
                                    long long int          strideB,
                                    const cuDoubleComplex *beta,
                                    cuDoubleComplex       *C, int ldc,
                                    long long int          strideC,
                                    int batchCount);
cublasStatus_t GEMM_STRIDED_BATCHED(cublasHandle_t handle,
                                    cublasOperation_t transa,
                                    cublasOperation_t transb,
                                    int m, int n, int k,
                                    const cuComplex *alpha,
                                    const cuComplex *A, int lda,
                                    long long int          strideA,
                                    const cuComplex *B, int ldb,
                                    long long int          strideB,
                                    const cuComplex *beta,
                                    cuComplex       *C, int ldc,
                                    long long int          strideC,
                                    int batchCount);

cublasStatus_t GEAM(cublasHandle_t handle,
                    cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n,
                    const cuDoubleComplex *alpha,
                    const cuDoubleComplex *A, int lda,
                    const cuDoubleComplex *beta,
                    const cuDoubleComplex *B, int ldb,
                    cuDoubleComplex *C, int ldc);
cublasStatus_t GEAM(cublasHandle_t handle,
                    cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n,
                    const cuComplex *alpha,
                    const cuComplex *A, int lda,
                    const cuComplex *beta,
                    const cuComplex *B, int ldb,
                    cuComplex *C, int ldc);

cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
                               cublasFillMode_t uplo,
                               int n,
                               cuDoubleComplex **Aarray,
                               int lda,
                               int *infoArray,
                               int batchSize);
cusolverStatus_t POTRF_BATCHED(cusolverDnHandle_t handle,
                               cublasFillMode_t uplo,
                               int n,
                               cuComplex **Aarray,
                               int lda,
                               int *infoArray,
                               int batchSize);

cusolverStatus_t POTRS(cusolverDnHandle_t handle,
                       cublasFillMode_t uplo,
                       int n,
                       int nrhs,
                       const cuDoubleComplex *A,
                       int lda,
                       cuDoubleComplex *B,
                       int ldb,
                       int *devInfo);
cusolverStatus_t POTRS(cusolverDnHandle_t handle,
                       cublasFillMode_t uplo,
                       int n,
                       int nrhs,
                       const cuComplex *A,
                       int lda,
                       cuComplex *B,
                       int ldb,
                       int *devInfo);


#endif //SYMMETRYMBPT_CUBLAS_ROUTINE_PREC_H
