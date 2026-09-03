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

// THREADS = BM/TM * BN/TN
template <int BM, int BK, int THREADS>
__device__ inline void load_tile_A(const float* A, float (*As)[BK],
	int by, int bk, int tid, int M, int K) {
	constexpr int STRIDE = THREADS / BK;      // 256/8 = 32
	const int row = tid / BK;                 // 0..31
	const int col = tid % BK;                 // 0..7
#pragma unroll
	for (int i = 0; i < BM; i += STRIDE) {    // 4 次，覆盖 128 行
		int gr = by * BM + row + i;
		int gc = bk + col;
		As[row + i][col] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
	}
}

template <int BK, int BN, int THREADS>
__device__ inline void load_tile_B(const float* B, float (*Bs)[BN],
	int bx, int bk, int tid, int K, int N) {
	constexpr int STRIDE = THREADS / BN;      // 256/128 = 2
	const int row = tid / BN;                 // 0..1
	const int col = tid % BN;                 // 0..127
#pragma unroll
	for (int r = 0; r < BK; r += STRIDE) {    // 4 次，覆盖 8 行
		int gr = bk + row + r;
		int gc = bx * BN + col;
		Bs[row + r][col] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
	}
}

template <int BM, int BN, int BK>
__global__ void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K)
{
	__shared__ float As[BM][BK];
	__shared__ float Bs[BK][BN];

	int tid = threadIdx.x;
	int thread_row = (tid / (BN / TN)) * TM;
	int thread_col = (tid % (BN / TN)) * TN;

	float a_frag[TM];
	float b_frag[TN];
	float c_frag[TM][TN];

	int by = blockIdx.y, bx = blockIdx.x;

	for (int bk = 0;bk < K;bk += BK)
	{
		// 加载A、B到shared memery
		constexpr int THREADS = BM / TM * BN / TN;
		load_tile_A<BM, BK, THREADS>(A, As, by, bk, tid, M, K);
		load_tile_B<BK, BN, THREADS>(B, Bs, bx, bk, tid, K, N);
		__syncthreads();

		// 外积累加
		for (int k = 0;k < BK;k++)
		{
			for (int i = 0;i < TM;i++)
			{
				a_frag[i] = As[thread_row + i][k];
			}
			for (int j = 0;j < TN;j++)
			{
				b_frag[j] = BS[k][thread_col + j];
			}

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
		gemm_v2<128, 128, 8, 256> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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
		gemm_v2<128, 128, 8, 256> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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