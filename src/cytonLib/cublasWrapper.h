/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _CYTONLIB_CUBLASWRAPPER_H_
#define _CYTONLIB_CUBLASWRAPPER_H_

#include "basicHeads.h"

namespace cytonLib
{


inline cublasStatus_t cublasXgemv(cublasHandle_t handle, cublasOperation_t trans,
	 int m, int n, const float *alpha,  const float *A, int lda,
	 const float *x, int incx, const float *beta, float *y, int incy)
{
	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t cublaXgemv(cublasHandle_t handle, cublasOperation_t trans,
	 int m, int n, const double *alpha,  const double *A, int lda,
	 const double *x, int incx, const double *beta, double *y, int incy)
{
	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const float *alpha, const float *A, int lda,
	 const float *B, int ldb, const float *beta, float *C, int ldc)
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const double *alpha, const double *A, int lda,
	 const double *B, int ldb, const double *beta, double *C, int ldc)
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemmBatch(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const float *alpha, const float *A, int lda, int strideA,
	 const float *B, int ldb, int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)
{
	return cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
			alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const float *alpha, const float *A, int lda, int strideA,
	 const float *B, int ldb, int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)
{
	return cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
			alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}


inline cublasStatus_t cublasXgemmBatch(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA,
	 const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)
{
	return cublasDgemmStridedBatched(handle, transa, transb, m, n, k,
			alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA,
	 const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)
{
	return cublasDgemmStridedBatched(handle, transa, transb, m, n, k,
			alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

cublasStatus_t cublasGemmBatchWrapper1(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const Precision *alpha, const Precision *A, int lda, long long int strideA,
	 const Precision *B, int ldb, long long int strideB, const Precision *beta, Precision *C, int ldc, long long int strideC, int batchCount);

inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n, const float *alpha,
		const float *x, int incx, float *y, int incy)
{
	return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n, const double *alpha,
		const double *x, int incx, double *y, int incy)
{
	return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const float *alpha,
		float *x, int incx)
{
	return cublasSscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const double *alpha,
		double *x, int incx)
{
	return cublasDscal(handle, n, alpha, x, incx);
}

inline cublasStatus_t cublasXnrm2(cublasHandle_t handle, int n,
    const float *x, int incx, float  *result)
{
	return cublasSnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t cublasXnrm2(cublasHandle_t handle, int n,
    const double *x, int incx, double  *result)
{
	return cublasDnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t cublasXsyrkx(cublasHandle_t handle,
                            cublasFillMode_t uplo, cublasOperation_t trans,
                            int n, int k,
                            const float           *alpha,
                            const float           *A, int lda,
                            const float           *B, int ldb,
                            const float           *beta,
                            float           *C, int ldc)
{
	return cublasSsyrkx(handle, uplo, trans,
			n, k, alpha, A, lda, B, ldb, beta, C, ldc);

}

inline cublasStatus_t cublasXsyrkx(cublasHandle_t handle,
                            cublasFillMode_t uplo, cublasOperation_t trans,
                            int n, int k,
                            const double          *alpha,
                            const double          *A, int lda,
                            const double          *B, int ldb,
                            const double          *beta,
                            double          *C, int ldc)
{
	return cublasDsyrkx(handle, uplo, trans,
				n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}



inline curandStatus_t curandGenerateUniformX(curandGenerator_t generator, float *outputPtr, size_t num)
{
	return curandGenerateUniform(generator, outputPtr, num);
}

inline curandStatus_t curandGenerateUniformX(curandGenerator_t generator, double *outputPtr, size_t num)
{
	return curandGenerateUniformDouble(generator, outputPtr, num);
}


inline cublasStatus_t cublasGemmBatchWrapper1(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const Precision *alpha, const Precision *A, int lda, long long int strideA,
	 const Precision *B, int ldb, long long int strideB, const Precision *beta, Precision *C, int ldc, long long int strideC, int batchCount)
{

	cublasStatus_t res;
	for(int i=0; i<batchCount;i++)
	{
		checkError(res=cublasXgemm(handle, transa, transb, m, n, k,
					alpha, A+strideA*i, lda, B+strideB*i, ldb, beta, C+strideC*i, ldc));
	}
	return res;
}


inline cublasStatus_t cublasXsyrk(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *beta,
                           float           *C, int ldc)
{
	return cublasSsyrk(handle, uplo, trans,
			n, k, alpha, A, lda, beta, C, ldc);
}

inline cublasStatus_t cublasXsyrk(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *beta,
                           double          *C, int ldc)
{
	return cublasDsyrk(handle, uplo, trans,
				n, k, alpha, A, lda, beta, C, ldc);
}

inline cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double          *alpha,
                           const double          *A, int lda,
                           double          *B, int ldb)
{
	return cublasDtrsm(handle, side, uplo, trans, diag,
			m, n, alpha, A, lda, B, ldb);
}

inline cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float          *alpha,
                           const float          *A, int lda,
                           float  *B, int ldb)
{
	return cublasStrsm(handle, side, uplo, trans, diag,
			m, n, alpha, A, lda, B, ldb);
}






}

#endif /* CUBLASWRAPPER_H_ */
