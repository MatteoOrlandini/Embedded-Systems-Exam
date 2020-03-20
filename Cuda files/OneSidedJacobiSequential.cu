//nvcc OneSidedJacobiSequential.cu svd_one_sided_jacobi_C.cu host_functions.cu -o OneSidedJacobiSequential

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h> //for chdir
#include <stdbool.h> //for bool type
#include "host_functions.h" //for host functions
#include "cuda_error_check.h" //for error checking

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

__global__ void rotate (float * B, int i, int j, int  rows, bool * exit_flag){
	int k = threadIdx.x; 
	__shared__ float alpha, beta, gamm, limit, tao, t, c, s;
	float *pi, *pj;
	if (k < rows) {
		alpha = beta = gamm = 0;
		__syncthreads();
		pi = B + rows * i + k;
		pj = B + rows * j + k;
		atomicAdd(&alpha, *pi * *pi);
		atomicAdd(&beta, *pj * *pj);	
		atomicAdd(&gamm, *pi * *pj);
		__syncthreads();
		if (* exit_flag) {
			//const float limit = fabsf(gamm) / sqrtf(alpha * beta);
			limit = fabsf(gamm) / sqrtf(alpha * beta);
			if (limit > eps) {
				* exit_flag = false;
			}
		}
		//const float tao = (beta - alpha) / (2 * gamm);
		//const float t = sign (tao) / (fabsf(tao) + sqrtf(1 + tao * tao)); 
		//const float c = expf(-0.5f * log1pf(t * t));  // new trick by Giorgio! Better than passing to 64 bits.
		//const float s = c * t;
		tao = (beta - alpha) / (2 * gamm);
		t = sign (tao) / (fabsf(tao) + sqrtf(1 + tao * tao)); 
		c = expf(-0.5f * log1pf(t * t));  // new trick by Giorgio! Better than passing to 64 bits.
		s = c * t;
		const float tmp = *pi;
		*pi = c * tmp - s * *pj;
		*pj = s * tmp + c * *pj;
	}
}

__global__ void computeSingVals (float * B, float * AUX1, int rows, int columns){
	int k = threadIdx.x;
	int j = blockIdx.x;
	__shared__ float t;
	if ((j < columns) && (k < rows)){
		float *pj = B + rows * j + k;
		t = 0;
		atomicAdd(&t, *pj * *pj);
		AUX1[j] = sqrtf(t);
	}
}

int main (int argc, char * argv[])
{
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
    int rows, columns;
    fillRowMajorOrderMatrix(&fp, &matrix, &rows, &columns);
    fclose(fp);
    // Column order matrix
    createColumnMajorOrderMatrix(&host_B, matrix, rows, columns);
    // initialize host_AUX1 array to zero
    initializeArray (&host_AUX1, columns);
    // Open new file to store the singular values
    sprintf(fileName, "SingularValues/CudaHost/Singular Values Cuda Host %dx%d.txt", rows, columns);
    openFile(&fp, fileName, "w");
    // cuda events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    //compute one sided jacobi
    int iterations = svd_one_sided_jacobi_C(rows, columns);
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_host = 0;
	cudaEventElapsedTime(&milliseconds_host, start, stop);
	cudaEventDestroy (start); 
	cudaEventDestroy (stop); 
    printf("Iterations on host: %d \n", iterations);
    descentOrdering(host_AUX1, columns);
    //print array and save on file
	printAndSaveArray (&fp, host_AUX1, columns);
    fclose(fp);
	
	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	free(host_B);
	createColumnMajorOrderMatrix(&host_B, matrix, rows, columns);
	
	float *AUX1, * B;	
	bool * exit_flag;

	CudaSafeCall(cudaMalloc( (void**)&B, rows*columns*sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&exit_flag, sizeof(bool) ));
	CudaSafeCall(cudaMalloc( (void**)&AUX1, columns * sizeof(float) ));

	CudaSafeCall(cudaMemcpy( B, host_B, rows * columns * sizeof(float), cudaMemcpyHostToDevice));

	//printf ("MATRIX ALLOCATED ON DEVICE \n");
	//dim3 block(rows, 0, 0);
	//dim3 grid(1, 0, 0);
	int iter = 0;
	bool host_exit_flag = false;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	while (!host_exit_flag) {
	++iter;
	host_exit_flag = true;
	CudaSafeCall(cudaMemcpy( exit_flag, &host_exit_flag, sizeof(bool), cudaMemcpyHostToDevice));
	for (int j = columns - 1; j >= 1; --j)
		for (int i = j - 1; i >= 0; --i) {	
			rotate<<<1, rows>>> (B, i, j, rows, exit_flag);
			CudaCheckError();
		}
	CudaSafeCall(cudaMemcpy( &host_exit_flag, exit_flag, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_device = 0;
	cudaEventElapsedTime(&milliseconds_device, start, stop);
	cudaEventDestroy (start); 
	cudaEventDestroy (stop);
	//Open new file to store the singular values on device
	sprintf(fileName, "SingularValues/CudaDevice/OneSidedSequential/Singular Values Cuda Device %dx%d.txt", rows, columns);
	openFile(&fp, fileName, "w");

	printf ("Iterations on device: %d\n", iter);

	// calculate singular values
	computeSingVals<<<columns, rows>>> (B, AUX1, rows, columns);
	CudaCheckError();
	CudaSafeCall(cudaMemcpy( host_AUX1, AUX1, columns * sizeof(float),  cudaMemcpyDeviceToHost));

	descentOrdering(host_AUX1, columns);
	printAndSaveArray (&fp, host_AUX1, columns);
	fclose(fp);
	//printf("SINGULAR VALUES STORED TO FILE \n");

	// free the memory allocated on the CPU 
    free(host_B);
	free(host_AUX1);

	//printf ("MEMORY ON HOST DEALLOCATED \n");

	// free the memory allocated on the GPU
	cudaFree(B);
	cudaFree(exit_flag);
	cudaFree(AUX1);

	//printf ("MEMORY ON DEVICE DEALLOCATED \n");
	
	printf ("Time on host: %f ms\n", milliseconds_host);
	printf ("Time on device: %f ms\n", milliseconds_device);

	sprintf(fileName, "Time/CudaHost/Time %dx%d.txt", rows, columns);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_host);	
	fclose(fp);
	
	sprintf(fileName, "Time/CudaDevice/OneSidedSequential/Time %dx%d.txt", rows, columns);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_device);	
	fclose(fp);
		
	cudaDeviceReset ( ); 
	return 0;
}


