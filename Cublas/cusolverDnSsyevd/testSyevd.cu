/* This function computes in single precision all eigenvalues and, optionally,
eigenvectors of a real symmetric matrix A. The second parameter can take
the values CUSOLVER EIG MODE VECTOR or CUSOLVER EIG MODE NOVECTOR
and answers the question whether the eigenvectors are desired. The sym-
metric matrix A can be stored in lower (CUBLAS FILL MODE LOWER) or upper
(CUBLAS FILL MODE UPPER) mode. If the eigenvectors are desired, then on
exit A contains orthonormal eigenvectors. The eigenvalues are stored in an
array W.*/

// nvcc testSyevd.cu -lcusolver

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

__global__ void simmMatrix(float * matrix, int rows, int cols){
	int t=threadIdx.x; //max(t)=2k
	int b=blockIdx.x; //max(b)=2k
	if (t>cols) return;
	if (t < b) *(matrix+rows*b+t)=0;
}

int main ( int argc , char * argv [])
{
	cusolverDnHandle_t cusolverH ;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS ;
	cudaError_t cudaStat = cudaSuccess ;
	int m = 12;
	//float * A ;	// mxm matrix
	float * V ;	// mxm matrix of eigenvectors
	float * W ;	// m - vector of eigenvalues
	// prepare memory on the host
	//A = ( float *) malloc ( m * m * sizeof ( float ));
	V = ( float *) malloc ( m * m * sizeof ( float ));
	W = ( float *) malloc ( m * sizeof ( float ));
	// define random A
	//for ( int i =0; i < m * m ; i ++) 
		//A [ i ] = i ;
	//float A[] = {0, 3, 6, 3, 4, 7, 6, 7, 8};
	//float A[] = {0, 3, 6, 0, 4, 7, 0, 0, 8};
	float A[] = {32.198,-1.5333,-4.7021,1.1095,3.5423,-1.0214,6.2234,9.457,4.761,-0.55268,-1.4301,-6.0476,
	-1.5333,43.969,-3.7997,-1.1536,-2.1299,3.5344,-4.3703,-3.2037,-9.2924,-0.44999,0.81734,
	2.3339,-4.7021,-3.7997,35.111,-12.121,1.6777,4.3666,3.4262,3.3346,7.2731,3.7535,4.0614,-1.1761,
	1.1095,-1.1536,-12.121,27.514,-5.5409,-0.29909,-3.0546,2.9462,-6.6693,3.1638,-5.6013,-9.9002,3.5423,
	-2.1299,1.6777,-5.5409,25.403,6.9837,4.5548,5.8709,1.9852,-4.6966,1.505,7.993,
	-1.0214,3.5344,4.3666,-0.29909,6.9837,42.332,-10.821,4.9534,8.0425,0.83761,0.39079,7.556,
	6.2234,-4.3703,3.4262,-3.0546,4.5548,-10.821,36.472,7.6667,3.3435,-10.844,-2.3241,-9.2912,9.457,
	-3.2037,3.3346,2.9462,5.8709,4.9534,7.6667,25.854,2.2944,0.67835,-4.4471,0.44142,4.761,-9.2924,7.2731,
	-6.6693,1.9852,8.0425,3.3435,2.2944,35.404,-0.41831,4.813,7.5699,-0.55268,-0.44999,3.7535,3.1638,-4.6966,
	0.83761,-10.844,0.67835,-0.41831,19.49,0.24713,3.3647,
	-1.4301,0.81734,4.0614,-5.6013,1.505,0.39079,-2.3241,-4.4471,4.813,0.24713,14.601,-3.3365,
	-6.0476,2.3339,-1.1761,-9.9002,7.993,7.556,-9.2912,0.44142,7.5699,3.3647,-3.3365,46.288};
	// declare arrays on the device
	float * d_A ;	// mxm matrix A on the device
	float * d_W ;	// m - vector of eigenvalues on the device	
	int * devInfo;	// info on the device
	float * d_work ;	// workspace on the device
	int lwork = 0;	// workspace size
	// create cusolver handle
	cusolver_status = cusolverDnCreate (& cusolverH );
	// prepare memory on the device
	cudaStat = cudaMalloc (( void **)& d_A , sizeof ( float )* m * m );
	cudaStat = cudaMalloc (( void **)& d_W , sizeof ( float )* m );
	cudaStat = cudaMalloc (( void **)& devInfo , sizeof ( int ));
	cudaStat = cudaMemcpy ( d_A ,A , sizeof ( float )* m * m ,	cudaMemcpyHostToDevice );	// copy A - > d_A
	
	simmMatrix<<<m,m>>>(d_A,m,m);
	// compute eigenvalues and eigenvectors
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	// use lower left triangle of the matrix
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;
	// compute buffer size and prepare workspace
	cusolver_status = cusolverDnSsyevd_bufferSize ( cusolverH ,	jobz , uplo , m , d_A , m , d_W , & lwork );
	cudaStat = cudaMalloc (( void **)& d_work , sizeof ( float )* lwork );
	//// compute the eigenvalues and eigenvectors for a symmetric ,
	// real mxm matrix ( only the lower left triangle af A is used )
	cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, m, d_A, m, d_W, d_work, lwork, devInfo);
	cudaStat = cudaMemcpy (W , d_W , sizeof ( float )* m , cudaMemcpyDeviceToHost );	// copy d_W - > W
	cudaStat = cudaMemcpy (V , d_A , sizeof ( float )* m * m ,	cudaMemcpyDeviceToHost );	// copy d_A - > V
	printf ( "eigenvalues :\n " );	// print first eigenvalues
	for ( int i = 0 ; i < m ; i ++){
		printf ( "\t%E\n " , W [ i ]);
	}
	//if jobz = CUSOLVER_EIG_MODE_VECTOR A contains the orthonormal eigenvectors of the matrix A. 
	printf ( "eigenvectors :\n " );	// print first eigenvectors
	for ( int i = 0 ; i < m  ; i ++){
		printf ( "\t%E\t%E\t%E\t\n" , V [ i ], V [ i + m], V [ i + 2*m]);
	}
}

