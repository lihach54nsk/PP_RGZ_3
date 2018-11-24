#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <cstdlib>

using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

/*bool Prime(long long n)
{
	for (int i = 2; i <= sqrt(n); i++)
		if (n%i == 0)
			return false;
	return true;
}*/

__global__ void addKernel(int *input, int *output)
{	
	int j = 0;
	int size = 100000;
	for (int i = 0; i < size; i++)
	{
		for (int i = 2; i <= sqrtf(input[i]); i++)
		{
			if (input[i] % i == 0) continue;
			else
			{
				output[j] = input[i]; j++;
			}
		}
		//if (Prime(input[i]) == true) { output[j] = input[i]; j++; }
		//else continue;
	}
}

int main()
{
	int input[100000];
	int output[sizeof(input) / sizeof(int)];

	//int* testA = new int[1];
	//int* testB = new int[1];

	int* I_a = (int*)malloc(sizeof(input) / sizeof(int)); // ������� ������
	int* O_v = (int*)malloc(sizeof(input) / sizeof(int)); // �������� ������
	int* finale = new int[sizeof(input) / sizeof(int)];

	for (int c = 1; c < 99999; c++)
	{
		input[c - 1] = c;
	}

	I_a = input;
	O_v = output;
	//testA[0] = 5;
	//testB[0] = 0;

	//int* cudaTestA, *cudaTestB;
	//cudaMalloc(&cudaTestA, 1);
	//cudaMalloc(&cudaTestB, 1);

	int* cuda_A;
	cudaMalloc(&cuda_A, sizeof(input) / sizeof(int));
	int* cuda_B;
	cudaMalloc(&cuda_B, sizeof(input) / sizeof(int));

	//cudaMemcpy(cudaTestA, testA, 1, cudaMemcpyHostToDevice);

	cudaMemcpy(cuda_A, I_a, sizeof(input) / sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, O_v, sizeof(input) / sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blockPerGrid = 10;
	fprintf(stderr, "Initialization success!");

	addKernel <<< blockPerGrid, threadsPerBlock >>> (cuda_A, cuda_B);	

	//cudaMemcpy(testB, cudaTestB, 1, cudaMemcpyHostToDevice);
	
	cudaMemcpy(finale, cuda_B, sizeof(input) / sizeof(int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();

	int i = 0;
	//char* cout = "cout";
	while (finale[i] != 5)
	{
		//cout = (char*)O_v[i];
		fprintf(stderr, (char*)finale);
		i++;
	}

	cudaFree(I_a);

	system("pause");
}

    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

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
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
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

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
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

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/