
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#pragma comment(lib, "cublas.lib")

#define CUDA_CHECK(call) do{\
	cudaError_t err = call;\
	if (err != cudaSuccess) {\
		fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		exit(1);\
	}\
}while(0)

#define CUBLAS_CHECK(call) do{\
	cublasStatus_t st = call;\
	if (st != CUBLAS_STATUS_SUCCESS) {\
		fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", __FILE__, __LINE__, (int)st);\
		exit(1);\
	}\
}while(0)

#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024

//Device: NVIDIA T600 Laptop GPU(SM 7.5)
//SM count : 14
//Boost clock : 1.40 GHz
//FP32 peak : 4999.68 GFLOPS(SM x 128 x 2 x clock)
//Regs / SM : 65536
//Threads / SM : 1024
//Smem / SM : 64 KB
//
//Running cuBLAS SGEMM(1024 x 1024 x 1024)...
//Verifying results on CPU...
//Result verification : PASSED
//
//Performance Report(cuBLAS) :
//	Matrix Size : 1024 x 1024 x 1024
//	Total Data : 12.00 MB
//	Total Ops : 2.15 GFLOPs
//	Time(avg) : 2.230 ms
//	Throughput : 962.79 GFLOPS
//	Bandwidth : 5.64 GB / s
//	Efficiency : 19.3 % of FP32 peak
//	请按任意键继续. . .

static void run_cublas(cublasHandle_t handle, const float* dA, const float* dB,
	float* dC, int M, int N, int K)
{
	const float alpha = 1.0f, beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		N, M, K,
		&alpha,
		dB, N,
		dA, K,
		&beta,
		dC, N);
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

	// 打印设备信息，用来判断理论峰值
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

	int clock_khz = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
	double clock_ghz = clock_khz / 1e6;

	double peak_gflops = (double)prop.multiProcessorCount * 128 * 2 * clock_ghz;
	printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
	printf("  SM count      : %d\n", prop.multiProcessorCount);
	printf("  Boost clock   : %.2f GHz\n", clock_ghz);
	printf("  FP32 peak     : %.2f GFLOPS (SM x 128 x 2 x clock)\n", peak_gflops);
	printf("  Regs / SM     : %d\n", prop.regsPerMultiprocessor);
	printf("  Threads / SM  : %d\n", prop.maxThreadsPerMultiProcessor);
	printf("  Smem / SM     : %zu KB\n\n", prop.sharedMemPerMultiprocessor / 1024);

	// 分配主机内存
	size_t size_A = (size_t)M * K * sizeof(float);
	size_t size_B = (size_t)K * N * sizeof(float);
	size_t size_C = (size_t)M * N * sizeof(float);

	float* h_A = (float*)malloc(size_A);
	float* h_B = (float*)malloc(size_B);
	float* h_C = (float*)malloc(size_C);
	float* h_C_ref = (float*)malloc(size_C);

	// 初始化数据
	for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
	for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

	// 分配设备内存
	float* d_A, * d_B, * d_C;
	CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
	CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
	CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

	// 数据搬运
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

	// 创建 cuBLAS handle
	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle));

	// 明确关掉 TF32，保证走纯 FP32 路径，和手写 kernel 公平对比。
	// 想看 TF32 有多快，把这行换成 CUBLAS_TF32_TENSOR_OP_MATH。
	CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

	printf("Running cuBLAS SGEMM (%d x %d x %d)...\n", M, N, K);

	// warm up：次数给足，cuBLAS 前几次会做算法选择和模块加载
	for (int i = 0; i < 50; i++) {
		run_cublas(handle, d_A, d_B, d_C, M, N, K);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 计时开始
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	int repeats = 50;
	CUDA_CHECK(cudaEventRecord(start));
	for (int r = 0; r < repeats; r++) {
		run_cublas(handle, d_A, d_B, d_C, M, N, K);
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

	printf("\nPerformance Report (cuBLAS):\n");
	printf("Matrix Size: %d x %d x %d\n", M, N, K);
	printf("Total Data:  %.2f MB\n", total_bytes / (1024 * 1024));
	printf("Total Ops:   %.2f GFLOPs\n", total_ops / 1e9);
	printf("Time (avg):  %.3f ms\n", avg_time_ms);
	printf("Throughput:  %.2f GFLOPS\n", gflops);
	printf("Bandwidth:   %.2f GB/s\n", bandwidth_GB_sec);
	printf("Efficiency:  %.1f%% of FP32 peak\n", gflops / peak_gflops * 100.0);

	// 清理资源
	cublasDestroy(handle);
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
