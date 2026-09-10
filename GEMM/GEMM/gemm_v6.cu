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

#define BLOCK_SIZE 256

#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])
#define CFLOAT4(ptr) (reinterpret_cast<const float4*>(&(ptr))[0])

//Launching kernel with Grid(8, 8), Block(256)...
//Verifying results on CPU...
//Result verification : PASSED
//
//Performance Report :
//Matrix Size : 1024 x 1024 x 1024
//Total Data : 12.00 MB
//Total Ops : 2.15 GFLOPs
//Time(avg) : 1.926 ms
//Throughput : 1115.06 GFLOPS
//Bandwidth : 6.53 GB / s



template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v6(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
	__shared__ float As[BK][BM + 4];
	__shared__ float Bs[BK][BN];

	int tid = threadIdx.x;
	int warpid = tid / 32;
	int laneid = tid % 32;
	int warp_row = warpid / 2;
	int warp_col = warpid % 2;
	int lane_row = laneid % 2 + (laneid / 16) * 2;
	int lane_col = (laneid % 16) / 2;
	int thread_row = (warp_row * 4 + lane_row) * TM;
	int thread_col = (warp_col * 8 + lane_col) * TN;

	float a_frag[2][TM];
	float b_frag[2][TN];
	float c_frag[TM][TN] = { 0.0f };

	int by = blockIdx.y, bx = blockIdx.x;

	// 加载A、B到shared memery
	// tile A
	const int row_a = tid / (BK / 4);                 // 0..127
	const int col_a = tid % (BK / 4) * 4;                 // 0 或 4
	const float* A_ptr = A + (by * BM + row_a) * K + col_a;

	// tile B
	const int row_b = tid / (BN / 4);                 // 0..7
	const int col_b = tid % (BN / 4) * 4;                 // 0,4...,124
	const float* B_ptr = B + row_b * N + bx * BN + col_b;

	for (int bk = 0;bk < K;bk += BK)
	{
		float ldg_a[4], ldg_b[4];
		FLOAT4(ldg_a[0]) = CFLOAT4(A_ptr[bk]);
		FLOAT4(ldg_b[0]) = CFLOAT4(B_ptr[bk * N]);

		// Load tile A
#pragma unroll
		for (int i = 0;i < 4;i++)
		{
			As[col_a + i][row_a] = ldg_a[i];
		}
		FLOAT4(Bs[row_b][col_b]) = FLOAT4(ldg_b[0]);

		// Load tile B
		__syncthreads();

		// 外积累加
#pragma unroll
		for (int i = 0;i < TM;i += 4) FLOAT4(a_frag[0][i]) = FLOAT4(As[0][thread_row + i]);
#pragma unroll
		for (int j = 0;j < TN;j += 4) FLOAT4(b_frag[0][j]) = FLOAT4(Bs[0][thread_col + j]);
#pragma unroll
		for (int k = 0;k < BK;k++)
		{
			if (k + 1 < BK)
			{
#pragma unroll
				for (int i = 0;i < TM;i += 4) FLOAT4(a_frag[(k + 1) & 1][i]) = FLOAT4(As[k + 1][thread_row + i]);
#pragma unroll
				for (int j = 0;j < TN;j += 4) FLOAT4(b_frag[(k + 1) & 1][j]) = FLOAT4(Bs[k + 1][thread_col + j]);
			}
#pragma unroll
			for (int i = 0;i < TM;i++)
#pragma unroll
				for (int j = 0;j < TN;j++) c_frag[i][j] += a_frag[k & 1][i] * b_frag[k & 1][j];
		}
		__syncthreads();
	}
		// 写回
#pragma unroll
		for (int i = 0;i < TM;i++)
		{
#pragma unroll
			for (int j = 0;j < TN;j += 4)
			{
				FLOAT4(C[(by * BM + thread_row + i) * N + BN * bx + thread_col + j]) = FLOAT4(c_frag[i][j]);
			}
		}

}


void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) C[i * N + j] = 0.0f;   // 先清零
		for (int k = 0; k < K; k++) {
			float a = A[i * K + k];                     // 提到外面
			for (int j = 0; j < N; j++)
				C[i * N + j] += a * B[k * N + j];       // B 和 C 都是连续访问！
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
	float* d_A, * d_B, * d_C;
	CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
	CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
	CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

	// 数据搬运
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

	// 设置Kernel配置
	dim3 blockSize(BLOCK_SIZE);
	dim3 gridSize((N + 128 - 1) / 128, (M + 128 - 1) / 128);
	printf("Launching kernel with Grid(%d, %d), Block(%d)...\n",
		gridSize.x, gridSize.y, blockSize.x);

	// warm up
	for (int i = 0;i < 20;i++)
	{
		gemm_v6<128, 128, 8, 8, 8> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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
		gemm_v6<128, 128, 8, 8, 8> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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


