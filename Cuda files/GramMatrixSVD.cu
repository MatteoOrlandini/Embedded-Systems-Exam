#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h> //for chdir
#include <stdbool.h> //for bool type
#include "host_functions.h" //for host functions
#include "cuda_error_check.h" //for error checking
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

//nvcc GramMatrixSVD.cu host_functions.cu svd_one_sided_jacobi_C.cu -o GramMatrixSVD -lcublas -lcusolver 

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

static const float eps = 1e-4;
float * host_B;
float * host_AUX1;
float * cudaB; 

__device__ int sign (float num){
    if (num > 0) return 1;
	if (num < 0) return -1;
	return 0;
}

__global__ void print (float * matrix, int iter,int rows, int k) {
	//int k = threadIdx.x;
	printf("%d; %f %f %f %f %f %f  %f\n",iter,matrix[0],matrix[1],matrix[2],matrix[3],matrix[4],matrix[5],matrix[2*k*2*k-1]);
	/*
	for (int i = 0; i < cols; i++){
		for (int j = 0; j < rows; j++){
			printf("%d:%f ", i*rows+j,matrix[i*rows+j]);
		}
		printf ("\n");
	}
	*/
}

__global__ void printMatrix(float * matrix, int rows, int cols){
	for (int i = 0; i < rows*cols; i++){
			printf ("%d: %f ", i, *(matrix+i));
	}
}

__global__ void simmMatrix(float * matrix, int rows, int cols){
	int t=threadIdx.x; //max(t)=2k
	int b=blockIdx.x; //max(b)=2k
	if (t>cols) return;
	if (t < b) *(matrix+rows*b+t)=*(matrix+rows*t+b);
}

__global__ void append (float *B,float *A_ij,int i,int j,int k,int rows) { 
	int h = threadIdx.x;
	if (h>2*k*rows) return;
	//printf("%d ",h);
	if (h<k*rows) A_ij[h]=*(B+i*rows+h);
	else A_ij[h]=*(B+j*rows+h-k*rows);
}

__global__ void update(float *B, float *A_ij, int i, int j, int k, int rows){
	int h = threadIdx.x;
	if (h>2*k*rows) return;
	if (h<k*rows) *(B+i*rows+h)=A_ij[h];
	else *(B+j*rows+h-k*rows)=A_ij[h];
}

__global__ void computeSingVals (float * B, float * AUX1, int rows, int columns){
	int k = threadIdx.x; //max(k)=rows-1
	int j = blockIdx.x; //max(j)=columns-1
	__shared__ float t;
	if ((j < columns) && (k < rows)){
		float *pj = B + rows * j + k;
		t = 0;
		__syncthreads();
		atomicAdd(&t, *pj * *pj);
		__syncthreads();
		AUX1[j] = sqrtf(t);
	}
}

int main (int argc, char * argv[]) {
    char input [100], fileName[] = {"Matrix/"};
    chdir("../");
    FILE *fp;
    if (argc == 1){
        do
        {
            printf ("Insert matrix name: \n");
            scanf ("%s", input);
            strcat(fileName, input);
        }
        while(openFile(&fp, fileName, "r") == false);
    }
    else if (argc == 2){
        sprintf(fileName, "Matrix/%s", argv[1]);
        if (openFile(&fp, fileName, "r") == false)
        	exit(1);
	} 
    float * matrix;
    int rows, cols;
    fillRowMajorOrderMatrix(&fp, &matrix, &rows, &cols);
    fclose(fp);
    // Column order matrix
    createColumnMajorOrderMatrix(&host_B, matrix, rows, cols);
    // initialize host_AUX1 array to zero
    initializeArray (&host_AUX1, cols);
    // Open new file to store the singular values
    sprintf(fileName, "SingularValues/CudaHost/Singular Values Cuda Host %dx%d.txt", rows, cols);
    openFile(&fp, fileName, "w");
    // cuda events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//compute one sided jacobi
    int iterations = svd_one_sided_jacobi_C(rows, cols);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_host = 0;
	cudaEventElapsedTime(&milliseconds_host, start, stop);
	cudaEventDestroy (start); 
	cudaEventDestroy (stop); 
    printf("Iterations on host: %d \n", iterations);
    descentOrdering(host_AUX1, cols);
    //print array and save on file
	printAndSaveArray (&fp, host_AUX1, cols);
    fclose(fp);
	
	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	free(host_B);
	createColumnMajorOrderMatrix(&host_B, matrix, rows, cols);

	/*
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			printf ("B[%d]:%f\t", i*cols+j, host_B[i*cols+j]);
		}
		printf ("\n");
	}
	*/

	cublasStatus_t stat;
	cublasHandle_t  handle; 
	stat = cublasCreate(&handle); 
	
	cusolverDnHandle_t cusolverH ;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS ;
	cudaError_t cudaStat = cudaSuccess ;
	
	cusolver_status = cusolverDnCreate (&cusolverH);
	
	int * devInfo;	// info on the device
	float * d_work ;	// workspace on the device
	int lwork = 0;	// workspace size
	
	float * dev_AUX1, * dev_B, *A_ij, *G, *d_W;
	float beta = 0.0f;
	float alpha = 1.0f;
	int k = cols/2; 
	bool * dev_exit_flag;

	CudaSafeCall(cudaMalloc((void**)&dev_B, rows*cols*sizeof(float) ));
	CudaSafeCall(cudaMalloc((void**)&dev_exit_flag, sizeof(bool) ));
	CudaSafeCall(cudaMalloc((void**)&dev_AUX1, cols * sizeof(float) ));
	CudaSafeCall(cudaMalloc((void**)&A_ij, 2*k*rows * sizeof(float) ));
	CudaSafeCall(cudaMalloc((void**)&G, 2*k * 2*k * sizeof(float) ));
	cudaStat = cudaMalloc ((void**)&devInfo , sizeof ( int ));
	//cudaStat = cudaMalloc ((void**)&d_work , sizeof ( float )* lwork );
	cudaStat = cudaMalloc ((void**)&d_W , sizeof ( float )* 2*k );
	
	CudaSafeCall(cudaMemcpy( dev_B, host_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	
	//printf ("MATRIX ALLOCATED ON DEVICE \n");
	int iter = 0;
	bool host_exit_flag = false;
	
	// compute eigenvalues and eigenvectors
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR ;
	// use lower left triangle of the matrix
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//while(!host_exit_flag) {
	while(iter<10) { ////////////////////////////////////////////////////////////////////////////<--------------
		++iter;
		host_exit_flag = true;
		CudaSafeCall(cudaMemcpy( dev_exit_flag, &host_exit_flag, sizeof(bool), cudaMemcpyHostToDevice));
		for (int i=0; i<cols-2*k+1; i+=k)
			for (int j=i+k; j<cols-k+1; j+=k){
				append<<<1,2*k*rows>>>(dev_B,A_ij,i,j,k,rows); //A_ij = [A_i   A_j]
				//print<<<1,1>>>(A_ij,iter,rows,k);
				//cudaDeviceSynchronize();
				
				stat = cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, 2*k, rows, &alpha, A_ij, rows, &beta, G, 2*k); //G = A'*A
				//print<<<1,1>>>(G,iter,rows,k);
				
				//simmMatrix<<<2*k,2*k>>>(G,2*k,2*k);
				//printMatrix<<<1,1>>>(G,2*k,2*k);
				//cudaDeviceSynchronize();
				
				// compute buffer size and prepare workspace
				cusolver_status = cusolverDnSsyevd_bufferSize ( cusolverH ,	jobz , uplo , 2*k , G , 2*k , d_W , &lwork );
				
				cudaStat = cudaMalloc (( void **)& d_work , sizeof ( float )* lwork );
								
				// compute the eigenvalues and eigenvectors for a symmetric ,
				// real mxm matrix ( only the lower left triangle af G is used )
				cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, 2*k, G, 2*k, d_W, d_work, lwork, devInfo);
				//cudaDeviceSynchronize();

				//print<<<1,1>>>(G,iter,rows,k);	
				//printMatrix<<<1,1>>>(G,2*k,2*k);
				//printf ("\n \n \n \n \n \n ");		
				//printMatrix<<<1,1>>>(d_W,1,2*k);			
				
				// A_ij = A_ij*U
				stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, rows, 2*k, 2*k, &alpha, A_ij, rows, G, 2*k, &beta, A_ij, rows);
				//cudaDeviceSynchronize();
				
				update<<<1,2*k*rows>>>(dev_B,A_ij,i,j,k,rows);
				cudaDeviceSynchronize();
		}
		CudaSafeCall(cudaMemcpy( &host_exit_flag, dev_exit_flag, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_device = 0;
	cudaEventElapsedTime(&milliseconds_device, start, stop);
	// open new file to store the singular values on device
	sprintf(fileName, "SingularValues/CudaDevice/GramMatrixSVD/Singular Values Cuda Device %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");

	printf ("Iterations on device: %d\n", iter);

	// calculate singular values
	computeSingVals<<<cols, rows>>> (dev_B, dev_AUX1, rows, cols);
	CudaCheckError();
	CudaSafeCall(cudaMemcpy( host_AUX1, dev_AUX1, cols * sizeof(float),  cudaMemcpyDeviceToHost));

	descentOrdering(host_AUX1, cols);
	printAndSaveArray (&fp, host_AUX1, cols);
	fclose(fp);


	//printf("SINGULAR VALUES STORED TO FILE \n");

	// free the memory allocated on the CPU 
	free(host_B);
	free(host_AUX1);

	//printf ("MEMORY ON HOST DEALLOCATED \n");

	// free the memory allocated on the GPU
	cudaFree(dev_B);
	cudaFree(dev_exit_flag);
	cudaFree(dev_AUX1);

	//printf ("MEMORY ON DEVICE DEALLOCATED \n");

	printf ("Time on host: %f ms\n", milliseconds_host);
	printf ("Time on device: %f ms\n", milliseconds_device);

	sprintf(fileName, "Time/CudaHost/Time %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_host);	
	fclose(fp);
	
	sprintf(fileName, "Time/CudaDevice/GramMatrixSVD/Time %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_device);	
	fclose(fp);
	
	cudaDeviceReset ( ); 

	return 0;
}
			

