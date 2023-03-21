// file: spmm.cu.cc
//
// Using cusparse API to test SpMM performance.
//  author: guyue huang
//  date  : 2021/10/13
// compile: nvcc version >=11.0

#include "../../src/ge-spmm/gespmm.h" // gespmmCsrSpMM()
#include "../util/sp_util.hpp"        // read_mtx
#include <cstdlib>                    // std::rand(), RAND_MAX
#include <cuda_runtime_api.h>         // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h> // cusparseSpMM (>= v11.0) or cusparseScsrmm
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

int main(int argc, const char **argv) {

  /// check command-line argument

  if (argc < 2) {
    printf("Require command-line argument: name of the sparse matrix file in "
           ".mtx format.\n");
    return EXIT_FAILURE;
  }

  //
  // Load sparse matrix
  //

  int M;                              // number of A-rows
  int K;                              // number of A-columns
  int nnz;                            // number of non-zeros in A
  float cutime;
  std::vector<int> csr_indptr_buffer; // buffer for indptr array in CSR format
  std::vector<int>
      csr_indices_buffer; // buffer for indices (column-ids) array in CSR format

  std::vector<int> row_buffer; 
  std::vector<int> col_buffer; 
  GpuTimer gpu_timer1;   
  GpuTimer gpu_timer2;   

  // load sparse matrix from mtx file
  read_mtx_file(argv[1], M, K, nnz, csr_indptr_buffer, csr_indices_buffer);
  read_coomtx_file(argv[1], M, K, nnz, row_buffer, col_buffer);
  printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original "
         "values and use randomly generated values.\n",
         M, K, nnz);//读取数据返回信息
  double std1 =0;
  double std2 = 0;
  int warmup_iter = 100;
  int repeat_iter = 100;

  for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
    if (iter == warmup_iter) {
      gpu_timer1.start();
    }
    std1 = get_rand_coo(nnz,M,K,row_buffer,col_buffer);
  }
  gpu_timer1.stop();
  float kernel_dur_msecs1 = gpu_timer1.elapsed_msecs()/repeat_iter/2;
  printf("\n%f\n",kernel_dur_msecs1);

  for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
    if (iter == warmup_iter) {
      gpu_timer2.start();
    }
    std2 = get_max_std_matrix(nnz,M,K);
  }
  gpu_timer2.stop();
  float kernel_dur_msecs2 = gpu_timer2.elapsed_msecs()/repeat_iter/2;
  printf("\n%f\n",kernel_dur_msecs2);
  std::cout<<std::endl;
  std::cout<<std1<<std::endl;
  std::cout<<std2<<std::endl;
  double std3 = (std2-std1)/(std2);
  std::cout<<std3<<std::endl;
  // std::vector<int>::iterator ite = row_buffer.begin();
  // int i=0;
	// for (; ite != row_buffer.end(); ite++){
	// 	std::cout << *ite;
  //   i++;
  //   if(i==999) break;
	// }
	// std::cout << std::endl;

  

  // Create GPU arrays
  int N = 128; // number of B-columns
  if (argc > 2) {
    N = atoi(argv[2]);
  }
  assert(
      N > 0 &&
      "second command-line argument is number of B columns, should be >0.\n");

  float *B_h = NULL, *C_h = NULL, *csr_values_h = NULL, *C_ref = NULL;
  float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL;

  B_h = (float *)malloc(sizeof(float) * K * N);//输入
  C_h = (float *)malloc(sizeof(float) * M * N);//输出
  C_ref = (float *)malloc(sizeof(float) * M * N);//传入的矩阵CSR格式信息
  csr_values_h = (float *)malloc(sizeof(float) * nnz);
  if (!B_h || !C_h || !C_ref || !csr_values_h) {
    printf("Host allocation failed.\n");
    return EXIT_FAILURE;
  }

  fill_random(csr_values_h, nnz);
  fill_random(B_h, K * N);

  CUDA_CHECK(cudaMalloc((void **)&B_d, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc((void **)&C_d, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc((void **)&csr_values_d, sizeof(float) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1)));
  CUDA_CHECK(cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz));

  CUDA_CHECK(
      cudaMemcpy(B_d, B_h, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * M * N));
  CUDA_CHECK(cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(),
                        sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr_indices_d, csr_indices_buffer.data(),
                        sizeof(int) * nnz, cudaMemcpyHostToDevice));

  //
  // Run Cusparse-SpMM and check result
  //

  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  CUSPARSE_CHECK(cusparseCreate(&handle));

  // creating sparse csr matrix
  CUSPARSE_CHECK(cusparseCreateCsr(
      &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  // creating dense matrices
  CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatInputDescr, K, N, N, B_d, CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW));
  CUSPARSE_CHECK(cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));

  // allocate workspace buffer
  size_t workspace_size;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

  // run SpMM
  CUSPARSE_CHECK(cusparseSpMM(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                              &alpha, csrDescr, dnMatInputDescr, &beta,
                              dnMatOutputDescr, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, workspace));

  CUDA_CHECK(
      cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(),
                                  csr_indices_buffer.data(), csr_values_h, B_h,
                                  C_ref);

  bool correct = check_result<float>(M, N, C_h, C_ref);

  //
  // Benchmark Cusparse-SpMM performance
  //

  if (correct) {

    GpuTimer gpu_timer;
    int warmup_iter = 10;
    int repeat_iter = 100;
    for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
      if (iter == warmup_iter) {
        gpu_timer.start();
      }

      cusparseSpMM(handle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                   CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                   &alpha, csrDescr, dnMatInputDescr, &beta, dnMatOutputDescr,
                   CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
    }
    gpu_timer.stop();

    float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;

    cutime = kernel_dur_msecs;

    float MFlop_count = (float)nnz / 1e6 * N * 2;

    float gflops = MFlop_count / kernel_dur_msecs;

    printf("[Cusparse] Report: spmm A(%d x %d) * B(%d x %d) sparsity %f "
           "(nnz=%d) \n Time %f (ms), Throughput %f (gflops).\n",
           M, K, K, N, (float)nnz / M / K, nnz, kernel_dur_msecs, gflops);
  }

  SpMatCsrDescr_t spmatA{M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d};
  gespmmAlg_t algs[] = {
      GESPMM_ALG_SEQREDUCE_ROWBALANCE,  GESPMM_ALG_PARREDUCE_ROWBALANCE,
      GESPMM_ALG_SEQREDUCE_NNZBALANCE,  GESPMM_ALG_PARREDUCE_NNZBALANCE,
      GESPMM_ALG_ROWCACHING_ROWBALANCE, GESPMM_ALG_ROWCACHING_NNZBALANCE};

  for (auto alg : algs) {

    //
    // Run GE-SpMM and check result
    //

    CUDA_CHECK(cudaMemset(C_d, 0x0, sizeof(float) * M * N));

    gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);

    cudaDeviceSynchronize();
    CUDA_CHECK(
        cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(),
                                    csr_indices_buffer.data(), csr_values_h,
                                    B_h, C_ref);

    bool correct = check_result<float>(M, N, C_h, C_ref);

    if (correct) {

      // benchmark GE-SpMM performance

      GpuTimer gpu_timer;
      int warmup_iter = 10;
      int repeat_iter = 100;
      for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
        if (iter == warmup_iter) {
          gpu_timer.start();
        }

        gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);
      }
      gpu_timer.stop();

      float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;

      float MFlop_count = (float)nnz / 1e6 * N * 2;

      float gflops = MFlop_count / kernel_dur_msecs;

      printf("[GE-SpMM][Alg: %d] Report: spmm A(%d x %d) * B(%d x %d) sparsity "
             "%f (nnz=%d) \n Time %f (CuSPARSE), Throughput %f (gflops).\n",
             alg, M, K, K, N, (float)nnz / M / K, nnz, kernel_dur_msecs / cutime,
             gflops);
    }
  }

  if (N >= 32)    printf("\nusing alg8\n");
  else if (N > 4 && std3<0.3) printf("\nusing alg2\n");
  else if (N > 4 && std3>0.3) printf("\nusing alg0\n");
  else            printf("\nusing alg1\n");

  /// free memory

  if (B_h)
    free(B_h);
  if (C_h)
    free(C_h);
  if (C_ref)
    free(C_ref);
  if (csr_values_h)
    free(csr_values_h);
  if (B_d)
    CUDA_CHECK(cudaFree(B_d));
  if (C_d)
    CUDA_CHECK(cudaFree(C_d));
  if (csr_values_d)
    CUDA_CHECK(cudaFree(csr_values_d));
  if (csr_indptr_d)
    CUDA_CHECK(cudaFree(csr_indptr_d));
  if (csr_indices_d)
    CUDA_CHECK(cudaFree(csr_indices_d));
  if (workspace)
    CUDA_CHECK(cudaFree(workspace));

  // destroy matrix/vector descriptors
  CUSPARSE_CHECK(cusparseDestroyDnMat(dnMatInputDescr));
  CUSPARSE_CHECK(cusparseDestroyDnMat(dnMatOutputDescr));
  CUSPARSE_CHECK(cusparseDestroySpMat(csrDescr));
  CUSPARSE_CHECK(cusparseDestroy(handle));
  return EXIT_SUCCESS;
}
