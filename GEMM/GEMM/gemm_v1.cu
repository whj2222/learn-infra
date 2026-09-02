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


template <int BM, int BN, int BK, int THREADS>
__global__ void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K)
{
	__shared__ float As[BM][BK];
	__shared__ float Bs[BK][BN];

	int r0 = blockIdx.y * BM;
	int c0 = blockIdx.x * BN;
	int tid = threadIdx.x;

	// 加载tileA时的线程重排
	constexpr int A_BLOCK_X = BK;
	constexpr int A_BLOCK_Y = BLOCK_SIZE / BK;
	int a_thread_x = tid % A_BLOCK_X;
	int a_thread_y = tid / A_BLOCK_X;
	
	// 加载tileB时的线程重排
	constexpr int B_BLOCK_X = 32;
	constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
	int b_thread_x = tid % B_BLOCK_X;
	int b_thread_y = tid / B_BLOCK_X;

	// 加载tileC时的线程重排
	constexpr int C_BLOCK_X = 16;
	constexpr int C_BLOCK_Y = 256 / C_BLOCK_X;
	int c_thread_x = tid % C_BLOCK_X;
	int c_thread_y = tid / C_BLOCK_X;

	// 每个线程负责 Tm * Tn 个输出元素
	constexpr int Tm = BM / C_BLOCK_Y;
	constexpr int Tn = BN / C_BLOCK_X;
	float Ct[Tm][Tn] = { 0.0f };

	for (int k = 0;k < K;k += BK)
	{
		// 加载tileA
		#pragma unroll
		for (int i = a_thread_y;i < BM;i += A_BLOCK_Y)
		{
			int r = r0 + i, c = k + a_thread_x;
			As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
		}

		// 加载tileB
		#pragma unroll
		for (int j = b_thread_x; j < BN;j += B_BLOCK_X)
		{
			int r = k + b_thread_y, c = c0 + j;
			Bs[b_thread_y][j] = (r < K && c < N) ? B[N * r + c] : 0.0f;
		}

		__syncthreads();


		// 外积计算
		#pragma unroll
		for (int p = 0; p < BK; p++) {
			for (int i = 0; i < Tm; i++) {
				int row = c_thread_y + i * C_BLOCK_Y;
				for (int j = 0; j < Tn; j++) {
					int col = c_thread_x + j * C_BLOCK_X;
					Ct[i][j] += As[row][p] * Bs[p][col];
				}
			}
		}
		__syncthreads();
	}
	

	for (int i = 0;i < Tm;i++)
	{
		int r = r0 + c_thread_y + i * C_BLOCK_Y;
		for (int j = 0;j < Tn;j++)
		{
			int c = c0 + c_thread_x + j * C_BLOCK_X;
			if (r < M && c < N) C[r * N + c] = Ct[i][j];
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
		gemm_v1<128, 128, 8, 256> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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
		gemm_v1<128, 128, 8, 256> << <gridSize, blockSize >> > (d_A, d_B, d_C, M, N, K);
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