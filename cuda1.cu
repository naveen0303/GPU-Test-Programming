//Author: Naveen Milind Chalawadi
//Assumption Size of the 2nd Array is always less than the 1st Array
// Input Arrays - (CPU)Array1,Array2  (GPU)GA1,GA2
// Output Array - (CPU)Arrayout       (GPU)GAout
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
__global__ void Kconvolve(float*, float*, float*, int, int);
__global__ void Kconvolve(float* GAout, float* GA1, float* GA2, int m, int n)
{
	//compute the global id for the thread
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	float temp_sum = 0;

	int startpoint = thread_id - (n / 2); // as x(m)*h(n) = h(n)*x(m-n)
	//here m is the size of 1st Array and n is the size of the 2nd Array
	for (int j = 0; j < n; j++) {
		int k = startpoint + j;
		if (k >= 0 && k < m) {
			temp_sum += GA1[k] * GA2[j];
		}
	}
	GAout[thread_id] = temp_sum;
	__syncthreads();
}


int main(int argc,char **argv)
{
	if (argc != 3)
	{
		printf("Error!!! Need two arguments (number of elements)sizes of Array_1 and Array_2\n");
		exit(1);
	}
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int datasize1 = m * sizeof(float);
	int datasize2 = n * sizeof(float);
	float* Array1 = (float*)malloc(datasize1);
	float* Array2 = (float*)malloc(datasize2);
	
	/*
	memset(Array1, 1, datasize1);
	memset(Array2, 1, datasize2);
	*/
	//Initialize Input Arrays (I'm just using initial value of 1 assigned to all elements in both the arrays
	for (int i = 0; i < m; i++) {
		Array1[i] = 1;
	}
	for (int i = 0; i < n; i++) {
		Array2[i] = 1;
	}

	//int l = m + n - 1;
	//int datasize3 = l * sizeof(float);
	//float* Arrayout = (float*)malloc(datasize3);
	//performing circular convolution
	float* Arrayout = (float*)malloc(datasize1);

	int width = m;
	//Initialize Output Array to 0;
	for (int i = 0; i < m; i++) {
		Arrayout[i] = 0;
	}

	//GPU Implementation starts here
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));	// getting properties of cuda device to get the calculate threads

	dim3 threads(prop.maxThreadsPerBlock);//calculate the gridsize and blocksizze
	dim3 blocks(width / threads.x + 1);

	float* GA1, * GA2, * GAout;						//pointer to device memory
	HANDLE_ERROR(cudaMalloc(&GA1, datasize1));		// allocate memory to device pointers
	HANDLE_ERROR(cudaMalloc(&GA2, datasize2));
	HANDLE_ERROR(cudaMalloc(&GAout, datasize1));

	
	HANDLE_ERROR(cudaMemcpy(GA1, Array1, datasize1, cudaMemcpyHostToDevice));	//copy the values from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(GA2, Array2, datasize2, cudaMemcpyHostToDevice));

	//Assigning clock timers to time the implementations
	cudaEvent_t startTime, stopTime;
	float elapsedTime = 0;
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);

	//Perform the convolution 
	cudaEventRecord(startTime);
	Kconvolve <<<blocks, threads >>> (GAout, GA1, GA2, m, n);
	cudaThreadSynchronize();
	cudaEventRecord(stopTime);

	//Get the elapsed time 
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&elapsedTime, startTime, stopTime);

	HANDLE_ERROR(cudaMemcpy(Arrayout, GAout, datasize1, cudaMemcpyDeviceToHost));	//copy the calculated values back into the CPU from GPU

	for (int i = 0; i < m; ++i) {
		cout << Arrayout[i] << '\t';
	}

	cudaFree(GA1);//Free the allocated memory
	cudaFree(GA2);
	cudaFree(GAout);
	free(Array1);						//free the allocated memory for input
	free(Array2);
	free(Arrayout);

	printf("\nComputation Time: %f ms", elapsedTime);
	//print the time taken for convolution using GPU
	
    return 0;
}
