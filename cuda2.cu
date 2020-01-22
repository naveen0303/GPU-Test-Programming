//Author: Naveen Milind Chalawadi
// Input Arrays - (CPU)Arrayin		(GPU)GpAin
// Output Array - (CPU)Arrayout		(GPU)GpAout
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


using namespace std;

//Macro for checking errors when using Cuda APIs and printing the error report on screen
#define Handle_Error(err) (HANDLE_ERROR(err))
static void Handle_Error(cudaError_t err) {
	if (err != cudaSuccess) {
		cout << "Error!!!!" << endl;
		cout << cudaGetErrorString(err) << endl;
	}
}

//Function Definition and Declaration
__global__ void reduction_sum(float*,float*, int);
__global__ void reduction_sum(float* GpAout, float* GpAin, int m)
{
	extern __shared__ int shareddata[];
	
	//compute the global_id and local_id for each thread
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;
	
	// each thread loads one element from global memory to shared memory
	shareddata[local_id] = GpAin[global_id];
	__syncthreads();
	
	// do reduction in shared mem
	for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
		if (local_id < k) {
			shareddata[local_id] += shareddata[local_id + k];
		}
		__syncthreads();
	}
		
	// write result for this block to global memory
	if (local_id == 0) GpAout[blockIdx.x] = shareddata[0];
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Error!!! Need an argument - Array Size\n");
		exit(1);
	}
	int m = atoi(argv[1]);
	int datasize1 = m * sizeof(float);
	float* Arrayin = (float*)malloc(datasize1);
	
	//Initialize Input Arrays (I'm just using initial value of 1 assigned to all elements in both the arrays
	for (int i = 0; i < m; i++) {
		Arrayin[i] = 1;
	}
	
	int datasize2 = 1 * sizeof(float);
	float* Arrayout = (float*)malloc(datasize2);
	Arrayout[0] = 0;

	//GPU Implementation starts here
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));	// getting properties of cuda device to get the calculate threads

	dim3 threads(prop.maxThreadsPerBlock);//calculate the gridsize and blocksizze
	dim3 blocks(m / threads.x + 1);
	int sharedBytes = threads.x * sizeof(float);

	float* GpAin, *GpAout;						//pointer to device memory
	HANDLE_ERROR(cudaMalloc(&GpAin, datasize1));		// allocate memory to device pointers
	HANDLE_ERROR(cudaMalloc(&GpAout, datasize2));


	HANDLE_ERROR(cudaMemcpy(GpAin, Arrayin, datasize1, cudaMemcpyHostToDevice));	//copy the values from CPU to GPU
	
	//Assigning clock timers to time the implementations
	cudaEvent_t startTime, stopTime;
	float elapsedTime = 0;
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);

	//Perform the convolution 
	cudaEventRecord(startTime);
	reduction_sum <<<blocks, threads, sharedBytes >>> (GpAout, GpAin, m);
	cudaThreadSynchronize();
	cudaEventRecord(stopTime);

	//Get the elapsed time 
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&elapsedTime, startTime, stopTime);

	HANDLE_ERROR(cudaMemcpy(Arrayout, GpAout, datasize1, cudaMemcpyDeviceToHost));	//copy the calculated values back into the CPU from GPU

	
	cudaFree(GpAin);//Free the allocated memory
	cudaFree(GpAout);
	cout << Arrayout[0] << endl;
	free(Arrayin);						//free the allocated memory for input
	free(Arrayout);

	printf("\nComputation Time: %f ms", elapsedTime);
	//print the time taken for convolution using GPU

	return 0;
}
