// file: spmm_util.hpp
//
// Utilities for SpMM example.
// Including: array initialization, timer, and file loader.
//  author: guyue huang
//  date  : 2021/06/29

#include "mmio.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>            // std::rand()
#include <cuda_runtime_api.h> // cudaEvent APIs
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <typeinfo>
#include <vector>
#include <map>
#include <utility>
#include <numeric>
#include <malloc.h>

#define CUDA_CHECK(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CUSPARSE_CHECK(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Fill a host array with random numbers.
void fill_random(float array[], int size) {
  for (int i = 0; i < size; i++) {
    array[i] = (float)(std::rand() % 3) / 10;
  }
}

// Fill a host array with all 0
template <typename DType> void fill_zero(DType array[], int size) {
  memset(array, 0x0, sizeof(array[0]) * size);
}

// Compute spmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void spmm_reference_host(
    int M, // number of A-rows
    int N, // number of B_columns
    int K, // number of A columns
    const Index *csr_indptr, const int *csr_indices,
    const DType *csr_values, // three arrays of A's CSR format
    const DType *B,          // assume row-major
    DType *C_ref)            // assume row-major
{
  fill_zero(C_ref, M * N);
  for (int64_t i = 0; i < M; i++) {
    Index begin = csr_indptr[i];
    Index end = csr_indptr[i + 1];
    for (Index p = begin; p < end; p++) {
      int k = csr_indices[p];
      DType val = csr_values[p];
      for (int64_t j = 0; j < N; j++) {
        C_ref[i * N + j] += val * B[k * N + j];
      }
    }
  }
}

// Compute sddmm correct numbers. All arrays are host memory locations.
template <typename Index, typename DType>
void sddmm_reference_host(
    int M,   // number of S-rows, S is the sparse matrix
    int N,   // number of S_cols
    int K, // number of A columns
    int nnz,  // number of nonzeros in S

    const Index *csr_indptr, const Index *csr_indices,
    const DType *csr_values, // three arrays of the sparse matrix's CSR format
    const DType *A,          // assume row-major
    const DType *B,          // assume row-major, assume transposed
    DType *C_ref)            // assume row-major
{
  for (int i = 0; i < M; i++) {
    Index lb = csr_indptr[i];
    Index hb = csr_indptr[i + 1];
    Index offset1, offset2;
    DType acc = 0;
    for (int ptr = lb; ptr < hb; ptr++) {
      offset1 = i * K;
      offset2 = csr_indices[ptr] * K;
      for (int k = 0; k < K; k++) {
        acc += A[k + offset1] * B[k + offset2];
      }
      C_ref[ptr] = acc * csr_values[ptr];
      acc = 0;
    }
  }
}

// Compare two MxN matrices
template <typename DType>
bool check_result(int M, int N, DType *C, DType *C_ref) {
  bool passed = true;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      DType c = C[i * N + j];
      DType c_ref = C_ref[i * N + j];
      if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
        printf(
            "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
            i, j, c, c_ref);
        passed = false;
      }
    }
  }
  return passed;
}

// Encapsule CUDA timing APIs.
//
// Usage:
//   GpuTimer timer; // create
//   timer.start();  // when you start recording
//   timer.stop();   // when  you stop recording
//   float dur = timer.elapsed_msecs(); // duration in milliseconds

struct GpuTimer {
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  GpuTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
  }

  ~GpuTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }

  void start() { cudaEventRecord(startEvent, 0); }

  void stop() {
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
  }

  float elapsed_msecs() {
    float elapsed;
    cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
    return elapsed;
  }
};

// Load sparse matrix from an mtx file. Only non-zero positions are loaded, and
// values are dropped.
void read_mtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int> &csr_indptr_buffer,
                   std::vector<int> &csr_indices_buffer) {
  FILE *f;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  //Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples

  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
    }
  }

  /// make symmetric

  if (mm_is_symmetric(matcode)) {
    std::vector<std::tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);

      new_coords.push_back(std::make_tuple(i, j));
      new_coords.push_back(std::make_tuple(j, i));
    }
    std::sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    std::sort(coords.begin(), coords.end());
  }

  /// generate csr from coo

  csr_indptr_buffer.clear();
  csr_indices_buffer.clear();

  int curr_pos = 0;
  csr_indptr_buffer.push_back(0);
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && (std::get<0>(coords[curr_pos]) == row)) {
      csr_indices_buffer.push_back(std::get<1>(coords[curr_pos]));
      curr_pos++;
    }
    // assert((std::get<0>(coords[curr_pos]) > row || curr_pos == nnz));
    csr_indptr_buffer.push_back(curr_pos);
  }

  nnz = csr_indices_buffer.size();
}

// Load sparse matrix from an mtx file. Only non-zero positions are loaded, and
// values are dropped.
void read_coomtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int> &coo_row_buffer,
                   std::vector<int> &coo_col_buffer) {
  FILE *f;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  //Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  // read tuples

  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
    }
  }

  /// make symmetric

  if (mm_is_symmetric(matcode)) {
    std::vector<std::tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);

      new_coords.push_back(std::make_tuple(i, j));
      new_coords.push_back(std::make_tuple(j, i));
    }
    std::sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    std::sort(coords.begin(), coords.end());
  }

  /// generate coo

  coo_row_buffer.clear();
  coo_col_buffer.clear();

  for (int64_t curr = 0; curr < nnz; curr++) {
      coo_col_buffer.push_back(std::get<1>(coords[curr]));
      coo_row_buffer.push_back(std::get<0>(coords[curr]));
  }

  nnz = coo_col_buffer.size();
}

int fun(int acc, std::pair<std::pair<int, int>,int> p) {
    return acc + p.second;
}

double count_std(std::map<std::pair<int, int>, int> p,int block ){
  int sumNum = accumulate(p.begin(), p.end(), 0, fun);
	double mean = (double)sumNum / block; //均值
	double accum = 0.0;
	for_each(p.begin(), p.end(), [&](const std::pair<std::pair<int, int>, int> d) {
		accum += (d.second - mean)*(d.second - mean);
	});

  for(int i = block ; i > p.size(); i--){
    accum += mean*mean;
  }

	double var = accum / block; //方差
	double std = sqrt(var); //标准差

	return std;
}


double count_std(int p[],int length){//均值、方差和标准差计算
	// double sumNum = accumulate(p.begin(), p.end(), 0.0);
	// double mean = sumNum / p.size(); //均值
	// double accum = 0.0;
	// for_each(p.begin(), p.end(), [&](const double d) {
	// 	accum += (d - mean)*(d - mean);
	// });
	// double var = accum / p.size(); //方差
	// double std = sqrt(var); //标准差

	// return std;
  double sum = 0;
  for(int loop = 0; loop < length; loop++) {
    sum += p[loop];
  }

  double mean = sum / length;//均值
  // printf("%f",mean);
  double var = 0;
  for (int j = 0; j < length;j++){
      var += pow(p[j]-mean,2);
  }

  var /= length;//求方差
    
  return pow(var,0.5);//求标准差
  
}



int get_min_std_matrix(int nnz,int row,int col){ //最均匀情况，相同nnz把所有非0按稀疏度重组，均匀度=1
    //gap = int(shape[0]*shape[1]/nnz)# 多少个数里面有一个非零
    //location_p = [[0] * int(shape[1]/10)] * int(shape[0]/10)
    return 0;
}

 
double get_max_std_matrix(int nnz,int row,int col){ // 最不均匀情况，相同nnz把所有非0堆在一起,均匀度=0
    if(row<10000){
      row = ((row%100==0)?row:(row+100-row%100));//扩充到整倍数
      col = ((col%100==0)?col:(col+100-col%100));//扩充到整倍数
      int rowfill = nnz/row;//填满多少行
      int rowfill100 = rowfill/100;
      int rowfillnot100 = rowfill%100; //不满100行的部分
      int colfill = nnz/col;//剩余多少列
      int index = row*col/10000;
      int location[index]={0};
      for (int i=0;i< (int)(rowfill100*(col/100));i++)
        location[i]=10000;

      for (int i=(int)(rowfill100*(col/100));i<(int)(rowfill100*(col/100)+(col/100));i++){
        int ap = ((colfill-100>=0)?100:colfill);
        colfill= ((colfill-100>=0)?colfill-100:0);
        location[i]=rowfillnot100*100+ap;
      }

      // for (int i=(int)(rowfill100*(col/100)+(col/100));i<(int)(row*col/10000);i++){
      //   location[i]=0;
      // }

      return count_std(location, index);
      // return 1;
    }else{
      row = ((row%500==0)?row:(row+500-row%500));//扩充到整倍数
      col = ((col%500==0)?col:(col+500-col%500));//扩充到整倍数
      int rowfill = nnz/row;//填满多少行
      int rowfill500 = rowfill/500;
      int rowfillnot500 = rowfill%500; //不满500行的部分
      int colfill = nnz/col;//剩余多少列
      int index = row*col/250000;
      int location[index]={0};
      for (int i=0;i< (int)(rowfill500*(col/500));i++)
        location[i]=250000;

      for (int i=(int)(rowfill500*(col/500));i<(int)(rowfill500*(col/500)+(col/500));i++){
        int ap = ((colfill-500>=0)?500:colfill);
        colfill= ((colfill-500>=0)?colfill-500:0);
        location[i]=rowfillnot500*500+ap;
      }

      // for (int i=(int)(rowfill100*(col/100)+(col/100));i<(int)(row*col/10000);i++){
      //   location[i]=0;
      // }

      return count_std(location, index);
    }

}

double get_rand_coo(int nnz,int row,int col,std::vector<int> &row_buffer, std::vector<int> &col_buffer){//coo 格式 100*10一组

    // std::map<std::pair<int, int>, int> location;
    // for(int i=0;i<nnz;i++){
    //   int rowindex = row_buffer[i]/100;
    //   int colindex = col_buffer[i]/100;
    //   std::pair<int, int> coor (rowindex,colindex);
    //   if(location.find(coor) != location.end()) location[coor]+=1;
    //   else location.insert(std::make_pair(coor,1));
    // }

    // return count_std(location,row/100*(col/100));
    if(row<10000){
      row = ((row%100==0)?row:(row+100-row%100));//扩充到整倍数
      col = ((col%100==0)?col:(col+100-col%100));//扩充到整倍数
      int k= row*col/10000;
      int location[k]={0};
      for(int i=0;i<nnz;i+=20){
        int rowindex = row_buffer[i]/100;
        int colindex = col_buffer[i]/100;
        int index = rowindex*col/100+colindex;
        location[index] += 1;
      }
      return count_std(location,k);
    }else{
      row = ((row%500==0)?row:(row+500-row%500));//扩充到整倍数
      col = ((col%500==0)?col:(col+500-col%500));//扩充到整倍数
      int k= row*col/250000;
      int location[k]={0};
      for(int i=0;i<nnz;i+=100){
        int rowindex = row_buffer[i]/500;
        int colindex = col_buffer[i]/500;
        int index = rowindex*col/500+colindex;
        location[index] += 1;
      }
      return count_std(location,k);
    }
}

double get_stand_std(int nnz,int row,int col,std::vector<int> &row_buffer, std::vector<int> &col_buffer){ // 将均匀程度分布在0,1之间，1表示分布最均匀
    int A_std = get_rand_coo(nnz,row,col,row_buffer,col_buffer);
    int A_max_std = get_max_std_matrix(nnz,row,col);
    int A_min_std = get_min_std_matrix(nnz,row,col);
    if (A_std < A_min_std)
        return 1;
    else
        return max(0,(A_std-A_max_std)/(A_min_std-A_max_std));//均匀分布函数F(x)=x-a/b-a
}