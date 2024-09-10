
#include "deepmd_common.h"
#include "fmt/format.h"
#include <assert.h>
namespace LAMMPS_NS {

#ifdef OPT_CBLAS

void matmul(const int m, const int n, const int k,
  double *A, double* B, double *C, double *D){
  double alpha = 1.;
  double beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++) {
      std::memcpy(D + i * n, C, n * sizeof(double));
    }
  } else {
    beta = 0.;
  }

  // printf("matmul m n k %d %d %d \n", m, n, k); std::fflush(stdout);

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
}

void matmul(const int m, const int n, const int k,
  float *A, float* B, float *C, float *D){
  float alpha = 1.;
  float beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++){
      std::memcpy(D + i * n, C, n * sizeof(float));
    }
  } else {
    beta = 0.;
    memset(D, 0, m * n * sizeof(float));
  }

  #ifdef OPT_CBLAS
  if(k == 240 && n == 240 && m <= 3) {
    matmul_1x240_240x240(m, n, k, A, B, D);
  } 
  else if(k == 240 && n == 2048 && m == 1) {
    matmul_1x240_240x2048(m, n, k, A, B, D);
  } 
  // else if(k == 240 && n == 2048 && m == 2) {
  //   matmul_2x240_240x2048(m, n, k, A, B, D);
  // } 
  // else if(k == 240 && n == 2048 && m == 3) {
  //   matmul_3x240_240x2048(m, n, k, A, B, D);
  // }
  else if(k == 2048 && n == 240 && m == 1) {
    matmul_1x2048_2048x240(m, n, k, A, B, D);
  }
  else if(k == 2048 && n == 240 && m == 2) {
    matmul_2x2048_2048x240(m, n, k, A, B, D);
  }
  else if(k == 2048 && n == 240 && m == 3) {
    matmul_3x2048_2048x240(m, n, k, A, B, D);
  }
  else if(k == 240 && n == 1 && m <= 3) {
    matmul_1x240_240x1(m, n, k, A, B, D);
  }
  else {
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
        m,n,k,
        alpha,A,k,
        B,n,
        beta,D,n);
  }
  #else 
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
        m,n,k,
        alpha,A,k,
        B,n,
        beta,D,n);
  #endif
}


void matmul(const int m, const int n, const int k,
  float *A, float16_t* B, float *C, float *D){
  float alpha = 1.;
  float beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++) {
      std::memcpy(D + i * n, C, n * sizeof(float));
    }
  } else {
    beta = 0.;
    memset(D, 0, m * n * sizeof(float));
  }

  if(k == 2048 && n == 240 && m == 1) {
    matmul_f16_1x2048_2048x240_nn(m, n, k, A, B, D);
  } else if(k == 2048 && n == 240 && m == 2) {
    matmul_f16_2x2048_2048x240_nn(m, n, k, A, B, D);
  } else if(k == 2048 && n == 240 && m == 3) {
    matmul_f16_3x2048_2048x240_nn(m, n, k, A, B, D);
  } else if(k == 240 && n == 2048 && m == 1) {
    matmul_f16_1x240_240x2048_nn(m, n, k, A, B, D);
  } else if(k == 240 && n == 2048 && m == 2) {
    matmul_f16_2x240_240x2048_nn(m, n, k, A, B, D);
  } else if(k == 240 && n == 2048 && m == 3) {
    matmul_f16_3x240_240x2048_nn(m, n, k, A, B, D);
  } else {
    assert(1 == 0);
  }
}

void matmul_3d(const int t, const int m, const int n, const int k,
  double* A, double* B, double* C, bool _transpose_a = false, bool _transpose_b = false){
  double alpha = 1.;
  double beta = 0.;

  int lda=_transpose_a ? m : k;
  int ldb=_transpose_b? k : n;
  int ldc=n;

  CBLAS_TRANSPOSE transpose_a = _transpose_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transpose_b = _transpose_b ? CblasTrans : CblasNoTrans;

  for(int ii = 0; ii < t; ii++) {
    cblas_dgemm(CblasRowMajor,transpose_a,transpose_b,
      m,n,k,
      alpha,A+ii*m*k,lda,
      B+ii*k*n,ldb,
      beta,C+ii*m*n,ldc);
  }
}

void matmul_3d(const int t, const int m, const int n, const int k,
  float* A, float* B, float* C, bool _transpose_a = false, bool _transpose_b = false){
  float alpha = 1.;
  float beta = 0.;

  int lda=_transpose_a ? m : k;
  int ldb=_transpose_b? k : n;
  int ldc=n;

  CBLAS_TRANSPOSE transpose_a = _transpose_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transpose_b = _transpose_b ? CblasTrans : CblasNoTrans;

  #ifdef OPT_CBLAS
  if(k == 4 && n == 16 && m == 128 && _transpose_a == true && _transpose_b == false) {
    for(int ii = 0; ii < t; ii++) {
      matmul_128x4_4x16_tn(m, n, k, 
              A+ii*m*k, B+ii*k*n, C+ii*m*n);
    }
  } 
  else if(k == 16 && n == 128 && m == 4 && _transpose_a == false && _transpose_b == true) {
    for(int ii = 0; ii < t; ii++) {
      matmul_4x16_16x128_nt(m, n, k, 
              A+ii*m*k, B+ii*k*n, C+ii*m*n);
    }
  }
  // else if(k == 128 && n == 16 && m == 4 && _transpose_a == false && _transpose_b == false) {
  //   for(int ii = 0; ii < t; ii++) {
  //     matmul_4x128_128x16_nn(m, n, k, 
  //             A+ii*m*k, B+ii*k*n, C+ii*m*n);
  //   }
  // }
  else {
    // if(t > 3 || _transpose_a == true || _transpose_a == true){
      for(int ii = 0; ii < t; ii++) {
        cblas_sgemm(CblasRowMajor,transpose_a,transpose_b,
          m,n,k,
          alpha,A+ii*m*k,lda,
          B+ii*k*n,ldb,
          beta,C+ii*m*n,ldc);
      }
    // }
    // else{
    //   for(int ii = 0; ii < t; ii++) {
    //     float *A_t = A+ii*m*k, *B_t=B+ii*k*n, *C_t=C+ii*m*n;
    //     for (int mm = 0; mm < m; ++mm) {
    //       for (int nn = 0; nn < n; ++nn) {
    //         for (int kk = 0; kk < k; ++kk) {
    //             C_t[mm*n+nn] += A_t[mm*k+kk] * B_t[kk*n+nn];
    //         }
    //       }
    //     }
    //   }
    // }
  } 
  #else
    for(int ii = 0; ii < t; ii++) {
      cblas_sgemm(CblasRowMajor,transpose_a,transpose_b,
        m,n,k,
        alpha,A+ii*m*k,lda,
        B+ii*k*n,ldb,
        beta,C+ii*m*n,ldc);
    }

  #endif
  
  
}

#else

void matmul(const int m, const int n, const int k,
  double *A, double* B, double *C, double *D){
  double alpha = 1.;
  double beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++) {
      std::memcpy(D + i * n, C, n * sizeof(double));
    }
  } else {
    beta = 0.;
  }

  // printf("matmul m n k %d %d %d \n", m, n, k); std::fflush(stdout);

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
}

void matmul(const int m, const int n, const int k,
  float *A, float* B, float *C, float *D){
  float alpha = 1.;
  float beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++){
      std::memcpy(D + i * n, C, n * sizeof(float));
    }
  } else {
    beta = 0.;
    memset(D, 0, m * n * sizeof(float));
  }

  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n); 

}


void matmul(const int m, const int n, const int k,
  float *A, float16_t* B, float *C, float *D){
  float alpha = 1.;
  float beta = 1.;
  int lda=k;
  int ldb=n;
  int ldc=n;

  if(C != NULL) {
    for(int i = 0; i < m; i++){
      std::memcpy(D + i * n, C, n * sizeof(float));
    }
  } else {
    beta = 0.;
    memset(D, 0, m * n * sizeof(float));
  }

  if(k == 2048 && n == 240 && m == 1) {
    matmul_f16_1x2048_2048x240_nn(m, n, k, A, B, D);
  } else if(k == 2048 && n == 240 && m == 2) {
    matmul_f16_2x2048_2048x240_nn(m, n, k, A, B, D);
  } else if(k == 240 && n == 2048 && m == 1) {
    matmul_f16_1x240_240x2048_nn(m, n, k, A, B, D);
  } else if(k == 240 && n == 2048 && m == 2) {
    matmul_f16_2x240_240x2048_nn(m, n, k, A, B, D);
  } else {
    assert(1 == 0);
  }
}

void matmul_3d(const int t, const int m, const int n, const int k,
  double* A, double* B, double* C, bool _transpose_a = false, bool _transpose_b = false){
  double alpha = 1.;
  double beta = 0.;

  int lda=_transpose_a ? m : k;
  int ldb=_transpose_b? k : n;
  int ldc=n;

  CBLAS_TRANSPOSE transpose_a = _transpose_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transpose_b = _transpose_b ? CblasTrans : CblasNoTrans;

  for(int ii = 0; ii < t; ii++) {
    cblas_dgemm(CblasRowMajor,transpose_a,transpose_b,
      m,n,k,
      alpha,A+ii*m*k,lda,
      B+ii*k*n,ldb,
      beta,C+ii*m*n,ldc);
  }
}

void matmul_3d(const int t, const int m, const int n, const int k,
  float* A, float* B, float* C, bool _transpose_a = false, bool _transpose_b = false){
  float alpha = 1.;
  float beta = 0.;

  int lda=_transpose_a ? m : k;
  int ldb=_transpose_b? k : n;
  int ldc=n;

  CBLAS_TRANSPOSE transpose_a = _transpose_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transpose_b = _transpose_b ? CblasTrans : CblasNoTrans;

  for(int ii = 0; ii < t; ii++) {
    cblas_sgemm(CblasRowMajor,transpose_a,transpose_b,
      m,n,k,
      alpha,A+ii*m*k,lda,
      B+ii*k*n,ldb,
      beta,C+ii*m*n,ldc);
  }
}

#endif
}
