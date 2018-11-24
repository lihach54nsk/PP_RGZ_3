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
	//int size = 10000;

	for (int k = 0; k < size; k++)
	{
		if (Prime(a[k]) == true) { c[j] = a[k]; j++; }
		else continue;
	}
}

int main()
{
	const int arraySize = 50000;
	int a[arraySize];
	int c[arraySize];

	for (int c = 1; c < arraySize; c++)
	{
		a[c - 1] = c;
	}

	// Add vectors in parallel.

	cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	int i = 0;
	while (c[i] > 0)
	{
		cout << c[i] << endl;
		i++;
	}

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

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	// Launch a kernel on the GPU with one thread for each element.
	auto begin = chrono::high_resolution_clock::now();
	addKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, size);
	auto end = chrono::high_resolution_clock::now();

	cout << "Work time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
	system("pause");

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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}