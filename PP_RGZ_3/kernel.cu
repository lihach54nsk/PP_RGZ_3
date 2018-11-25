#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

cudaError_t addWithCuda(long long value);

using namespace std;

__global__ void addKernel(long long from, long long a, char *output, int cudaCores)
{
	const long long current = threadIdx.x + from + cudaCores * blockIdx.x;

	long long outPos = current - from;

	output[outPos] = 0;

	if (a % current == 0) output[outPos] = -1;
	else output[outPos] = 1;

	/*for (int i = from; i < sqrt((double)a); i++)
	{
		if (a % i == 0) output[outPos] = -1;
		else output[outPos] = 1;
	}*/
}

int main()
{
	long long value;

	cout << "Write value: "; cin >> value;

	// Add vectors in parallel.
	auto begin = chrono::high_resolution_clock::now();
	cudaError_t cudaStatus = addWithCuda(value);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	auto end = chrono::high_resolution_clock::now();

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
cudaError_t addWithCuda(long long value)
{
	int cudaCores = 1000;

	long long from = 2;
	const long long bufferSize = value - from;
	const long long blockCount = (bufferSize / cudaCores) + (bufferSize%cudaCores == 0 ? 0 : 1);

	if (bufferSize < cudaCores)
	{
		cudaCores = bufferSize;
	}

	char *output = new char[bufferSize];
	char *dev_output;

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

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_output, bufferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	// Launch a kernel on the GPU with one thread for each element.

	cudaEventRecord(start, 0);

	addKernel <<< blockCount, cudaCores >>> (from, value, dev_output, cudaCores);

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
	cudaStatus = cudaMemcpy(output, dev_output, 1, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_output);

	cout << (int)output[0] << endl;
	if ((int)output[0] > 0)	cout << "Chislo " << value << " prostoe" << endl;
	else cout << "Chislo " << value << " ne prostoe" << endl;
	
	/*while (y < bufferSize)
	{
		cout << (int)output[y] << endl;
		y++;
	}*/

	cout << "Work time: " << time << endl;
	system("pause");

	return cudaStatus;
}