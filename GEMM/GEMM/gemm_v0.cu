#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do{\
	cudaError_t err = call;\
	if (err != cudaSuccess) {\
		fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		exit(1);\
	}\
}while(0)

#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024

#define BLOCK_SIZE 32

__global__ void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < M && col < N)
	{
		float sum = 0.0f;
		for (int k = 0;k < K;k++)
		{
			sum += A[K * row + k] * B[N * k + col];
		}
		C[row * N + col] = sum;
	}
}


void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K)
{
	for (int i = 0;i < M;i++)
	{
		for (int j = 0; j < N;j++)
		{
			float sum = 0.0f;
			for (int k = 0;k < K;k++)
			{
				sum += A[i * K + k] * B[N * k + j];
			}
			C[i * N + j] = sum;
		}
	}
}

int main()
{
	int M = MATRIX_M;
	int N = MATRIX_N;
	int K = MATRIX_K;

	// 分配主机内存
	size_t size_A = M * K * sizeof(float);
	size_t size_B = K * N * sizeof(float);
	size_t size_C = M * N * sizeof(float);

	float* h_A = (float*)malloc(size_A);
	float* h_B = (float*)malloc(size_B);
	float* h_C = (float*)malloc(size_C);
	float* h_C_ref = (float*)malloc(size_C);

	// 初始化数据
	for (int i = 0;i < M * K;i++) h_A[i] = (float)rand() / RAND_MAX;
	for (int i = 0;i < K * N;i++) h_B[i] = (float)rand() / RAND_MAX;

	// 分配设备内存
	float* d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
	CUDA_CHECK(cudaMalloc((void**)&d_B, size_A));
	CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

	// 数据搬运
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

	// 设置Kernel配置
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize(((N + blockSize.x - 1) / blockSize.x), (M + blockSize.y - 1) / blockSize.y);
	printf("Launching kernel with Grid(%d, %d), Block(%d, %d)...\n",
		gridSize.x, gridSize.y, blockSize.x, blockSize.y);

	// warm up
	for (int i = 0;i < 20;i++)
	{
		gemm_v0 << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 计时开始
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// 执行Kernel
	int repeats = 20;
	CUDA_CHECK(cudaEventRecord(start));
	for (int r = 0;r < repeats;r++)
	{
		gemm_v0 << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	// 计时结束
	float milliseconds = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

	// 搬运结果
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

	// 验证正确性
	printf("Verifying results on CPU...\n");
	gemm_cpu(h_A, h_B, h_C_ref, M, N, K);

	bool correct = true;
	for (int i = 0; i < M * N; i++) {
		if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
			correct = false;
			printf("Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], h_C_ref[i]);
			break;
		}
	}

	if (correct) {
		printf("Result verification: PASSED\n");
	}
	else {
		printf("Result verification: FAILED\n");
	}


	// 计算性能指标
	double total_ops = 2.0 * (double)M * N * K;
	double avg_time_ms = milliseconds / repeats;
	double avg_time_sec = avg_time_ms / 1000.0;
	double gflops = (total_ops / avg_time_sec) / 1e9;
	double total_bytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
	double bandwidth_GB_sec = (total_bytes / avg_time_sec) / 1e9;

	printf("\nPerformance Report:\n");
	printf("Matrix Size: %d x %d x %d\n", M, N, K);
	printf("Total Data:  %.2f MB\n", total_bytes / (1024 * 1024));
	printf("Total Ops:   %.2f GFLOPs\n", total_ops / 1e9);
	printf("Time (avg):  %.3f ms\n", avg_time_ms);
	printf("Throughput:  %.2f GFLOPS\n", gflops);
	printf("Bandwidth:   %.2f GB/s\n", bandwidth_GB_sec);

	// 10. 清理资源
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_ref);


	system("pause");
	return 0;
}