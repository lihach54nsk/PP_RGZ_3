#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

cudaError_t addWithCuda(int *c, const int *a, unsigned int size);

using namespace std;

__device__ bool Prime(long long n)
{
	for (int i = 2; i <= sqrt((double)n); i++)
		if (n%i == 0)
			return false;
	return true;
}

__global__ void addKernel(int *c, const int *a, int size)
{
	int j = 0;

	for (int k = 0; k < size; k++)
	{
		if (Prime(a[k]) == true) { c[j] = a[k]; j++; }
		else continue;
	}
}

int main()
{
	const int arraySize = 400000;
	int *a = new int[arraySize];
	int *c = new int[arraySize];

	for (int c = 1; c < arraySize; c++)
	{
		a[c - 1] = c;
	}

	// Add vectors in parallel.
	auto begin = chrono::high_resolution_clock::now();
	cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	auto end = chrono::high_resolution_clock::now();

	int i = 0;
	while (c[i] > 0)
	{
		cout << c[i] << endl;
		i++;
	}

	cout << "Work time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
	system("pause");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, unsigned int size)
{
	int *dev_a = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//int threadsPerBlock = 55;
	//int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	// Launch a kernel on the GPU with one thread for each element.

	cudaEventRecord(start, 0);

	addKernel <<< 10, 100 >>> (dev_c, dev_a, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaEventRecord(stop, 0);
	float time = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	cout << "Work time: " << time << endl;
	system("pause");

	return cudaStatus;
}