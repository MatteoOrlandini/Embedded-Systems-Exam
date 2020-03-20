/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include gesvdjbatch_example.cpp 
 *   g++ -o gesvdjbatch_example gesvdjbatch_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %20.16E\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int m = 3; /* 1 <= m <= 32 */
    const int n = 2; /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int batchSize = 2;
    const int minmn = (m < n)? m : n; /* min(m,n) */
/*  
 *        |  1  -1  |
 *   A0 = | -1   2  |
 *        |  0   0  |
 *
 *   A0 = U0 * S0 * V0**T
 *   S0 = diag(2.6180, 0.382) 
 *
 *        |  3   4  |
 *   A1 = |  4   7  |
 *        |  0   0  |
 *
 *   A1 = U1 * S1 * V1**T
 *   S1 = diag(9.4721, 0.5279) 
 */
	    double A[lda*n*batchSize]; /* A = [A0 ; A1] */
    double U[ldu*m*batchSize]; /* U = [U0 ; U1] */
    double V[ldv*n*batchSize]; /* V = [V0 ; V1] */
    double S[minmn*batchSize]; /* S = [S0 ; S1] */
    int info[batchSize];       /* info = [info0 ; info1] */

    double *d_A  = NULL; /* lda-by-n-by-batchSize */
    double *d_U  = NULL; /* ldu-by-m-by-batchSize */
    double *d_V  = NULL; /* ldv-by-n-by-batchSize */
    double *d_S  = NULL; /* minmn-by-batchSizee */
    int* d_info  = NULL; /* batchSize */
    int lwork = 0;       /* size of workspace */
    double *d_work = NULL; /* device workspace for gesvdjBatched */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd  = 0;   /* don't sort singular values */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

/* residual and executed_sweeps are not supported on gesvdjBatched */
    double residual = 0;
    int executed_sweeps = 0;

    double *A0 = A;
    double *A1 = A + lda*n; /* Aj is m-by-n */
/*
 *        |  1  -1  |
 *   A0 = | -1   2  |
 *        |  0   0  |
 *   A0 is column-major
 */
    A0[0 + 0*lda] =  1.0;
    A0[1 + 0*lda] = -1.0;
    A0[2 + 0*lda] =  0.0;

    A0[0 + 1*lda] = -1.0;
    A0[1 + 1*lda] =  2.0;
    A0[2 + 1*lda] =  0.0;

/*
 *        |  3   4  |
 *   A1 = |  4   7  |
 *        |  0   0  |
 *   A1 is column-major
 */
    A1[0 + 0*lda] = 3.0;
    A1[1 + 0*lda] = 4.0;
    A1[2 + 0*lda] = 0.0;

    A1[0 + 1*lda] = 4.0;
    A1[1 + 1*lda] = 7.0;
    A1[2 + 1*lda] = 0.0;

    printf("example of gesvdjBatched \n");
    printf("m = %d, n = %d \n", m, n);
    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);

    printf("A0 = (matlab base-1)\n");
    printMatrix(m, n, A0, lda, "A0");
    printf("A1 = (matlab base-1)\n");
    printMatrix(m, n, A1, lda, "A1");
    printf("=====\n");
/* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* disable sorting */
    status = cusolverDnXgesvdjSetSortEig(
        gesvdj_params,
        sort_svd);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(double)*lda*n*batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_U   , sizeof(double)*ldu*m*batchSize);
    cudaStat3 = cudaMalloc ((void**)&d_V   , sizeof(double)*ldv*n*batchSize);
    cudaStat4 = cudaMalloc ((void**)&d_S   , sizeof(double)*minmn*batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int   )*batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*n*batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    
    /* step 4: query working space of gesvdjBatched */
    status = cusolverDnDgesvdjBatched_bufferSize(
        cusolverH,
        jobz,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        gesvdj_params,
        batchSize
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 5: compute singular values of A0 and A1 */
    status = cusolverDnDgesvdjBatched(
        cusolverH,
        jobz,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        d_work,
        lwork,
        d_info,
        gesvdj_params,
        batchSize
    );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(U    , d_U   , sizeof(double)*ldu*m*batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V    , d_V   , sizeof(double)*ldv*n*batchSize, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(S    , d_S   , sizeof(double)*minmn*batchSize, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int) * batchSize       , cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    for(int i = 0 ; i < batchSize ; i++){
        if ( 0 == info[i] ){
            printf("matrix %d: gesvdj converges \n", i);
        }else if ( 0 > info[i] ){
/* only info[0] shows if some input parameter is wrong.
 * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
 */
            printf("Error: %d-th parameter is wrong \n", -info[i] );
            exit(1);
        }else { /* info = m+1 */
/* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
            printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i] );
        }
    }

/* Step 6: show singular values and singular vectors */
    double *S0 = S;
    double *S1 = S + minmn;
    printf("==== \n");
    for(int i = 0 ; i < minmn ; i++){
        printf("S0(%d) = %20.16E\n", i+1, S0[i]);
    }
    printf("==== \n");
    for(int i = 0 ; i < minmn ; i++){
        printf("S1(%d) = %20.16E\n", i+1, S1[i]);
    }
    printf("==== \n");

    double *U0 = U;
    double *U1 = U + ldu*m; /* Uj is m-by-m */
    printf("U0 = (matlab base-1)\n");
    printMatrix(m, m, U0, ldu, "U0");
    printf("U1 = (matlab base-1)\n");
    printMatrix(m, m, U1, ldu, "U1");

    double *V0 = V;
    double *V1 = V + ldv*n; /* Vj is n-by-n */
    printf("V0 = (matlab base-1)\n");
    printMatrix(n, n, V0, ldv, "V0");
    printf("V1 = (matlab base-1)\n");
    printMatrix(n, n, V1, ldv, "V1");
   /*
 * The folowing two functions do not support batched version.
 * The error CUSOLVER_STATUS_NOT_SUPPORTED is returned. 
 */
    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_U    ) cudaFree(d_U);
    if (d_V    ) cudaFree(d_V);
    if (d_S    ) cudaFree(d_S);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);

    cudaDeviceReset();

    return 0;
}


