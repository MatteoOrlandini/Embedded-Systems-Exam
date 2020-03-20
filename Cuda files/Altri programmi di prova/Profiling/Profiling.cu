#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
	cudaProfilerInitialize ( "config.txt", "output.txt", cudaCSV);
	cudaProfilerStart();
    cuda_hello<<<1,1>>>(); 
    cudaProfilerStop();
    return 0;
}

