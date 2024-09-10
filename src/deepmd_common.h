#ifndef DEEPMD_COMMON_H
#define DEEPMD_COMMON_H 

// #define WITH_TENSOR_FLOW
#define COMBIN_OMP

#define SPLIT_TYPE_EMBEDDING

#define __ARM_FEATURE_SVE

// #define HIGH_PREC

// #define OPT_CBLAS

// make CC=fcc TARGET=ARMV8SVE  NOFORTRAN=1 -j48 |& tee compile.log

#ifdef HIGH_PREC
typedef double FPTYPE;
typedef double ENERGYTYPE;
#define cblas_xgemm cblas_dgemm
#define TABLE_STEP 16

#else 
typedef float  FPTYPE;
typedef double ENERGYTYPE;
#define cblas_xgemm cblas_sgemm
#define TABLE_STEP 32
#endif

#ifndef HIGH_PREC
#ifdef OPT_CBLAS
#define T_FLOAT_16
#endif
#endif

#include <cblas.h>

// #include "OPENBLAS/include/cblas.h"
// #include "../../CBLAS/openblas/include/cblas.h"

#include <math.h>
#include <string>
#include <vector>
#include <cstring>
#include <arm_sve.h>
#include <stdlib.h>
#include <iostream>
#include "matrix_tool.h"

namespace LAMMPS_NS {

template<typename T>
inline T dot(
    T a[4], 
    T b[4]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 
}

template<typename T>
inline FPTYPE dot3 (const T* r0, const T* r1) {
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
}


template<typename T>
inline T tanh_opt(T in) {
  if (-0.01 < in && in < 0.01){
    T x2 = in * in; T x3 = x2 * in; T x5 = x3 * x2; T x7 = x5 * x2;
    const T c3 = (-1.0 / 3.0); const T c5 = (2.0 / 15.0); const T c7 = (-17.0 / 315.0);
    return c7 * x7 + c5 * x5 + c3 * x3 + in; 
  }
  else { 
    T ep = std::exp(in); T em = 1.0 / ep; return (ep - em) / (ep + em); 
  }
}

template<typename T>
inline void fast_tanh(const int n, T* in, T* out) {
  for (int i = 0; i < n; i++) {
    // out[i] =  std::tanh(in[i]); 
    out[i] =  tanh_opt(in[i]); 
  }
}

//  dx = dy * (1 - tanh(x)^2) = dy * (1 - y^2)
template<typename T>
inline void fast_tanh_grad(const int n, T* y, T* dy, T* dx) {
  for (int i = 0; i < n; i++) {
    dx[i] = dy[i] * (1. - y[i] * y[i]);
  }
}

template<typename T>
inline void idt_mult(const int m, const int n, T* idt, T* in, T *out) {
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      out[i*n+j] =  in[i*n+j] * idt[j]; 
    }
  }
}

template<typename T>
inline void idt_mult_grad(const int m, const int n, T* idt, T* in, T *out) {
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      out[i*n+j] =  in[i*n+j] * idt[j]; 
    }
  }
}

template<typename T>
inline void matrix_add(const int m, const int n, T* A ,T* B) {
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      B[i*n+j] +=  A[i*n+j]; 
    }
  }
}

template<typename T>
inline void matrix_add(const int m, const int n, T* A ,T* B, T*C) {
  for(int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      C[i*n+j] +=  A[i*n+j] + B[i*n+j]; 
    }
  }
}

template<typename T>
inline void print_v(int n, std::string mesg, const T* v) {
  printf("%s :\n", mesg.c_str());
  for(int ii = 0; ii < n; ii ++) {
      // if(std::is_same<double, T>::value) printf("%0.9f ", (double)v[ii]);
      // else if(std::is_same<float, T>::value) printf("%0.9f ", (double)v[ii]);
      // else printf("%d ", v[ii]);
      if(std::is_same<double, T>::value) printf("%0.9f ", v[ii]);
      else if(std::is_same<float, T>::value) printf("%0.9f ", v[ii]);
      else printf("%d ", v[ii]);
      if(ii % 100 == 0 && ii != 0) printf("\n");
  }
  printf("\n"); std::fflush(stdout);
}

inline void print_v(int n, std::string mesg, std::vector<int> v) {
  printf("%s :\n", mesg.c_str());
  for(int ii = 0; ii < n; ii ++) {
      printf("%d ", v[ii]);
      if(ii % 100 == 0 && ii != 0) printf("\n");
  }
  printf("\n"); std::fflush(stdout);
}

// D = AB+C
// A (m, k)
// B (k, n)
// C (   n)
// D (m, n)

inline void cum_sum(
    std::vector<int> & _sec, 
    const std::vector<int> & _n_sel)  {
  _sec.resize (_n_sel.size() + 1);
  _sec[0] = 0;
  for (int ii = 1; ii < _sec.size(); ++ii) {
    _sec[ii] = _sec[ii-1] + _n_sel[ii-1];
  }
}
//////////////////////////////////////////////
///////////////// matmul /////////////////////
//////////////////////////////////////////////

#ifdef __ARM_FEATURE_SVE
inline void matmul_1x240_240x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {
  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);

  float *tmp_a, *tmp_b, *tmp_d;
  int kk = 0;      

  for(int mm = 0; mm < M; mm++) {
    tmp_a = a_fp32 + mm * K;

    tmp_d = d_fp32 + mm * N;

    svld1_vnum_15(ain, tmp_d);

    for(kk = 0; kk < K; kk++) {
      float scala_a = tmp_a[kk];
      tmp_b = b_fp32 + kk * N;

      svld1_vnum_15(bin, tmp_b);
      svmla_z_15(ain, ain, bin, scala_a)
    }

    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_1x240_240x2048(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);
  float *tmp_a, *tmp_b, *tmp_d;
  tmp_d = d_fp32;
  
  int kk = 0;
    
  for(int mm = 0; mm < M; mm++) {

    tmp_a = a_fp32 + mm * K;

    for(int ll = 0; ll < 8; ll++) {

      tmp_d = d_fp32 + mm * N + ll * N / 8;

      svld1_vnum_16(ain, tmp_d);

      for(kk = 0; kk < K; kk++) {
        float scala_a = tmp_a[kk];
        tmp_b = b_fp32 + kk * N + ll * N / 8;

        svld1_vnum_16(bin, tmp_b);
        svmla_z_16(ain, ain, bin, scala_a)
      }

      svst1_vnum_16(tmp_d, ain);
    }    
  }
}

inline void matmul_2x240_240x2048(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain_0);
  init_vec_16(ain_1);
  init_vec_16(bin);

  float *tmp_a_0, *tmp_a_1, *tmp_b, *tmp_d_0, *tmp_d_1;
  // 2048 = 8 x 8 x 8 x 8 x 4    
  int kk = 0;
    
  for(int ll = 0; ll < 16; ll++) {
    tmp_a_0 = a_fp32 + 0 * K;
    tmp_a_1 = a_fp32 + 1 * K;

    tmp_d_0 = d_fp32 + 0 * N + ll * N / 16;
    tmp_d_1 = d_fp32 + 1 * N + ll * N / 16;

    svld1_vnum_8(ain_0, tmp_d_0);
    svld1_vnum_8(ain_1, tmp_d_1);

    for(kk = 0; kk < K; kk++) {
      float scala_a_0 = tmp_a_0[kk];
      float scala_a_1 = tmp_a_1[kk];
      tmp_b = b_fp32 + kk * N + ll * N / 16;

      svld1_vnum_8(bin, tmp_b);
      svmla_z_8(ain_0, ain_0, bin, scala_a_0)
      svmla_z_8(ain_1, ain_1, bin, scala_a_1)
    }

    svst1_vnum_8(tmp_d_0, ain_0);
    svst1_vnum_8(tmp_d_1, ain_1);
  }
}

inline void matmul_3x240_240x2048(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain_0);
  init_vec_16(ain_1);
  init_vec_16(ain_2);
  init_vec_16(bin);

  float *tmp_a_0, *tmp_a_1, *tmp_a_2, *tmp_b;
  float *tmp_d_0, *tmp_d_1, *tmp_d_2;
  // 2048 = 8 x 8 x 8 x 8 x 4    
  int kk = 0;
    
  for(int ll = 0; ll < 16; ll++) {
    tmp_a_0 = a_fp32 + 0 * K;
    tmp_a_1 = a_fp32 + 1 * K;
    tmp_a_2 = a_fp32 + 2 * K;

    tmp_d_0 = d_fp32 + 0 * N + ll * N / 16;
    tmp_d_1 = d_fp32 + 1 * N + ll * N / 16;
    tmp_d_2 = d_fp32 + 2 * N + ll * N / 16;

    svld1_vnum_8(ain_0, tmp_d_0);
    svld1_vnum_8(ain_1, tmp_d_1);
    svld1_vnum_8(ain_2, tmp_d_2);

    for(kk = 0; kk < K; kk++) {
      float scala_a_0 = tmp_a_0[kk];
      float scala_a_1 = tmp_a_1[kk];
      float scala_a_2 = tmp_a_2[kk];
      tmp_b = b_fp32 + kk * N + ll * N / 16;

      svld1_vnum_8(bin, tmp_b);
      svmla_z_8(ain_0, ain_0, bin, scala_a_0)
      svmla_z_8(ain_1, ain_1, bin, scala_a_1)
      svmla_z_8(ain_2, ain_2, bin, scala_a_2)
    }

    svst1_vnum_8(tmp_d_0, ain_0);
    svst1_vnum_8(tmp_d_1, ain_1);
    svst1_vnum_8(tmp_d_2, ain_2);
  }
}

inline void matmul_1x240_240x2048_normal(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain_0);
  init_vec_16(ain_1);
  init_vec_16(bin);

  float *tmp_a_0, *tmp_a_1, *tmp_b, *tmp_d_0, *tmp_d_1;
  // 2048 = 8 x 8 x 8 x 8 x 4    
  int kk = 0;
  
  for(int m = 0; m < M; m+=2) {
    if(m != M - 1) {
      tmp_a_0 = a_fp32 + (m+0) * K;
      tmp_a_1 = a_fp32 + (m+1) * K;
      for(int ll = 0; ll < 16; ll++) {

        tmp_d_0 = d_fp32 + (m+0) * N + ll * N / 16;
        tmp_d_1 = d_fp32 + (m+1) * N + ll * N / 16;

        svld1_vnum_8(ain_0, tmp_d_0);
        svld1_vnum_8(ain_1, tmp_d_1);

        for(kk = 0; kk < K; kk++) {
          float scala_a_0 = tmp_a_0[kk];
          float scala_a_1 = tmp_a_1[kk];
          tmp_b = b_fp32 + kk * N + ll * N / 16;

          svld1_vnum_8(bin, tmp_b);
          svmla_z_8(ain_0, ain_0, bin, scala_a_0)
          svmla_z_8(ain_1, ain_1, bin, scala_a_1)
        }

        svst1_vnum_8(tmp_d_0, ain_0);
        svst1_vnum_8(tmp_d_1, ain_1);
      }
    } else {
      tmp_a_0 = a_fp32 + m * K;

      for(int ll = 0; ll < 8; ll++) {

        tmp_d_0 = d_fp32 + m * N + ll * N / 8;

        svld1_vnum_16(ain_0, tmp_d_0);

        for(kk = 0; kk < K; kk++) {
          float scala_a = tmp_a_0[kk];
          tmp_b = b_fp32 + kk * N + ll * N / 8;

          svld1_vnum_16(bin, tmp_b);
          svmla_z_16(ain_0, ain_0, bin, scala_a);
        }

        svst1_vnum_16(tmp_d_0, ain_0);
      }
    }
  }
}


inline void matmul_1x2048_2048x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);

  float *tmp_a, *tmp_b, *tmp_d;
      
  for(int mm = 0; mm < M; mm++) {
    tmp_d = d_fp32 + mm * N;

    svld1_vnum_15(ain, tmp_d);

    tmp_a = a_fp32 + mm * K;
    for(int kk = 0; kk < K; kk++) {

      float scala_a = tmp_a[kk];
      tmp_b = b_fp32 + kk * N;

      svld1_vnum_15(bin, tmp_b);
      svmla_z_15(ain, ain, bin, scala_a)
    }
    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_2x2048_2048x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain0);
  init_vec_16(ain1);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);

  float *tmp_a0,*tmp_a1, *tmp_b, *tmp_d0, *tmp_d1;
      
  // for(int mm = 0; mm < M; mm++) {
  tmp_d0 = d_fp32 + 0 * N;
  tmp_d1 = d_fp32 + 1 * N;

  svld1_vnum_15(ain0, tmp_d0);
  svld1_vnum_15(ain1, tmp_d1);

  tmp_a0 = a_fp32 + 0 * K;
  tmp_a1 = a_fp32 + 1 * K;
  for(int kk = 0; kk < K; kk++) {
    float scala_a0 = tmp_a0[kk];
    float scala_a1 = tmp_a1[kk];
    tmp_b = b_fp32 + kk * N;

    svld1_vnum_15(bin, tmp_b);
    svmla_z_15(ain0, ain0, bin, scala_a0)
    svmla_z_15(ain1, ain1, bin, scala_a1)
  }
  svst1_vnum_15(tmp_d0, ain0);
  svst1_vnum_15(tmp_d1, ain1);
  // }
}
inline void matmul_3x2048_2048x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain0);
  init_vec_16(ain1);
  init_vec_16(ain2);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);

  float *tmp_a0,*tmp_a1,*tmp_a2, *tmp_b, *tmp_d0, *tmp_d1,*tmp_d2;
      
  // for(int mm = 0; mm < M; mm++) {
  tmp_d0 = d_fp32 + 0 * N;
  tmp_d1 = d_fp32 + 1 * N;
  tmp_d2 = d_fp32 + 2 * N;

  svld1_vnum_15(ain0, tmp_d0);
  svld1_vnum_15(ain1, tmp_d1);
  svld1_vnum_15(ain2, tmp_d2);

  tmp_a0 = a_fp32 + 0 * K;
  tmp_a1 = a_fp32 + 1 * K;
  tmp_a2 = a_fp32 + 2 * K;
  for(int kk = 0; kk < K; kk++) {

    float scala_a0 = tmp_a0[kk];
    float scala_a1 = tmp_a1[kk];
    float scala_a2 = tmp_a2[kk];
    tmp_b = b_fp32 + kk * N;

    svld1_vnum_15(bin, tmp_b);
    svmla_z_15(ain0, ain0, bin, scala_a0)
    svmla_z_15(ain1, ain1, bin, scala_a1)
    svmla_z_15(ain2, ain2, bin, scala_a2)
  }
  svst1_vnum_15(tmp_d0, ain0);
  svst1_vnum_15(tmp_d1, ain1);
  svst1_vnum_15(tmp_d2, ain2);
  // }
}

inline void matmul_4x128_128x16(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain);
  init_vec_16(bin);
  init_vec_16(cin);
  init_vec_16(din);

  float *tmp_a, *tmp_b, *tmp_d;
      
  for(int mm = 0; mm < M; mm++) {
    tmp_d = d_fp32 + mm * N;

    svld1_vnum_15(ain, tmp_d);

    tmp_a = a_fp32 + mm * K;
    for(int kk = 0; kk < K; kk++) {

      float scala_a = tmp_a[kk];
      tmp_b = b_fp32 + kk * N;

      svld1_vnum_15(bin, tmp_b);
      svmla_z_15(ain, ain, bin, scala_a)
    }
    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_128x4_4x16_tn(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_8(ain);
  init_vec_8(bin);

  float *tmp_a, *tmp_b, *tmp_d;

  tmp_b = b_fp32;
  bin_0 = svld1_vnum(ptrue, tmp_b, 0 );
  bin_1 = svld1_vnum(ptrue, tmp_b, 1 );
  bin_2 = svld1_vnum(ptrue, tmp_b, 2 );
  bin_3 = svld1_vnum(ptrue, tmp_b, 3 );

  for(int mm = 0; mm < M; mm += 8) {
    tmp_a = a_fp32 + mm;
    svmul_z_t_8(ain, bin_0, tmp_a);

    tmp_a += M;
    svmla_z_t_8(ain, ain, bin_1, tmp_a);

    tmp_a += M;
    svmla_z_t_8(ain, ain, bin_2, tmp_a);

    tmp_a += M;
    svmla_z_t_8(ain, ain, bin_3, tmp_a);

    tmp_d = d_fp32 + mm * N;
    svst1_vnum_8(tmp_d, ain);
  }
}

inline void matmul_4x16_16x128_nt(const int M, const int N, const int K,
  float *A, float* B, float *C) {

  svbool_t ptrue = svptrue_b32();
  
  float *tmp_a, *tmp_b, *tmp_c;
  tmp_a = A; 
  tmp_b = B;
  tmp_c = C; 
  svfloat32_t ain0;   
  svfloat32_t ain1;   
  svfloat32_t ain2;   
  svfloat32_t ain3;

  svfloat32_t bin0;   
  svfloat32_t bin1;   
  svfloat32_t bin2;   
  svfloat32_t bin3; 

  svfloat32_t cin0;   
  svfloat32_t cin1;   
  svfloat32_t cin2;   
  svfloat32_t cin3; 

  ain0 = svld1_vnum(ptrue, tmp_a, 0);  
  ain1 = svld1_vnum(ptrue, tmp_a, 1);  
  ain2 = svld1_vnum(ptrue, tmp_a, 2);  
  ain3 = svld1_vnum(ptrue, tmp_a, 3);

  for(int i = 0; i <32; i++) {    
    tmp_b = B + i*64 ;
    bin0 = svld1_vnum(ptrue, tmp_b, 0);  
    bin1 = svld1_vnum(ptrue, tmp_b, 1);  
    bin2 = svld1_vnum(ptrue, tmp_b, 2);  
    bin3 = svld1_vnum(ptrue, tmp_b, 3);

    cin0 = svdup_f32(0.); 
    cin1 = svdup_f32(0.); 
    cin2 = svdup_f32(0.); 
    cin3 = svdup_f32(0.); 

    cin0 = svmul_z(ptrue, bin0, ain0) ; 
    tmp_c[i*4] = svaddv(ptrue,cin0 );
    cin1 = svmul_z(ptrue, bin0, ain1) ; 
    tmp_c[i*4+128] = svaddv(ptrue,cin1 );
    cin2 = svmul_z(ptrue, bin0, ain2) ; 
    tmp_c[i*4+256] = svaddv(ptrue,cin2 );
    cin3 = svmul_z(ptrue, bin0, ain3) ; 
    tmp_c[i*4+384] = svaddv(ptrue,cin3 );

    cin0 = svmul_z(ptrue, bin1, ain0) ; 
    tmp_c[i*4+1] = svaddv(ptrue,cin0 );
    cin1 = svmul_z(ptrue, bin1, ain1) ; 
    tmp_c[i*4+128+1] = svaddv(ptrue,cin1 );
    cin2 = svmul_z(ptrue, bin1, ain2) ; 
    tmp_c[i*4+256+1] = svaddv(ptrue,cin2 );
    cin3 = svmul_z(ptrue, bin1, ain3) ; 
    tmp_c[i*4+384+1] = svaddv(ptrue,cin3 );

    cin0 = svmul_z(ptrue, bin2, ain0) ; 
    tmp_c[i*4+2] = svaddv(ptrue,cin0 );
    cin1 = svmul_z(ptrue, bin2, ain1) ; 
    tmp_c[i*4+128+2] = svaddv(ptrue,cin1 );
    cin2 = svmul_z(ptrue, bin2, ain2) ; 
    tmp_c[i*4+256+2] = svaddv(ptrue,cin2 );
    cin3 = svmul_z(ptrue, bin2, ain3) ; 
    tmp_c[i*4+384+2] = svaddv(ptrue,cin3 );

    cin0 = svmul_z(ptrue, bin3, ain0) ; 
    tmp_c[i*4+3] = svaddv(ptrue,cin0 );
    cin1 = svmul_z(ptrue, bin3, ain1) ; 
    tmp_c[i*4+128+3] = svaddv(ptrue,cin1 );
    cin2 = svmul_z(ptrue, bin3, ain2) ; 
    tmp_c[i*4+256+3] = svaddv(ptrue,cin2 );
    cin3 = svmul_z(ptrue, bin3, ain3) ; 
    tmp_c[i*4+384+3] = svaddv(ptrue,cin3 );
  }
}



inline void matmul_4x128_128x16_nn(const int M, const int N, const int K,
  float *A, float* B, float *C) {

    for(int kk = 0; kk < K; kk++) {
      for(int mm = 0; mm < M; mm++) {
        C[mm*16+ 0] = A[mm*K+kk] * B[kk*N + 0];
        C[mm*16+ 1] = A[mm*K+kk] * B[kk*N + 1];
        C[mm*16+ 2] = A[mm*K+kk] * B[kk*N + 2];
        C[mm*16+ 3] = A[mm*K+kk] * B[kk*N + 3];
        C[mm*16+ 4] = A[mm*K+kk] * B[kk*N + 4];
        C[mm*16+ 5] = A[mm*K+kk] * B[kk*N + 5];
        C[mm*16+ 6] = A[mm*K+kk] * B[kk*N + 6];
        C[mm*16+ 7] = A[mm*K+kk] * B[kk*N + 7];
        C[mm*16+ 8] = A[mm*K+kk] * B[kk*N + 8];
        C[mm*16+ 9] = A[mm*K+kk] * B[kk*N + 9];
        C[mm*16+ 10] = A[mm*K+kk] * B[kk*N + 10];
        C[mm*16+ 11] = A[mm*K+kk] * B[kk*N + 11];
        C[mm*16+ 12] = A[mm*K+kk] * B[kk*N + 12];
        C[mm*16+ 13] = A[mm*K+kk] * B[kk*N + 13];
        C[mm*16+ 14] = A[mm*K+kk] * B[kk*N + 14];
        C[mm*16+ 15] = A[mm*K+kk] * B[kk*N + 15];
      }
    }
}


inline void matmul_1x240_240x1(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_16(ain);
  init_vec_16(bin);

  float *tmp_a, *tmp_b, *tmp_d;

  tmp_b = b_fp32;
    
  svld1_vnum_15(bin, tmp_b);
  for(int mm = 0; mm < M; mm++) {
    tmp_a = a_fp32 + mm * K;

    svld1_vnum_15(ain, tmp_a);

    svmul_z_15(ain, ain, bin);

    svaddv_15(d_fp32[mm], ain);
  }
}

inline void matmul_128x4_4x16(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  svbool_t ptrue = svptrue_b32();

  init_vec_8(ain);
  init_vec_8(bin);

  float *tmp_a, *tmp_b, *tmp_d;

  // load B
  tmp_b = b_fp32;
  bin_0 = svld1_vnum(ptrue, tmp_b, 0 );
  bin_1 = svld1_vnum(ptrue, tmp_b, 1 );
  bin_2 = svld1_vnum(ptrue, tmp_b, 2 );
  bin_3 = svld1_vnum(ptrue, tmp_b, 3 );

  for(int mm = 0; mm < M; mm+=2) {
    tmp_a = a_fp32 + mm * K;
    tmp_d = d_fp32 + mm * N;

    ain_0 = svmul_z(ptrue,        bin_0,  tmp_a[0]);
    ain_0 = svmla_z(ptrue, ain_0, bin_1,  tmp_a[1]);
    ain_0 = svmla_z(ptrue, ain_0, bin_2,  tmp_a[2]);
    ain_0 = svmla_z(ptrue, ain_0, bin_3,  tmp_a[3]);

    ain_1 = svmul_z(ptrue,        bin_0,  tmp_a[0+K]);
    ain_1 = svmla_z(ptrue, ain_1, bin_1,  tmp_a[1+K]);
    ain_1 = svmla_z(ptrue, ain_1, bin_2,  tmp_a[2+K]);
    ain_1 = svmla_z(ptrue, ain_1, bin_3,  tmp_a[3+K]);

    svst1_vnum(ptrue, tmp_d, 0, ain_0);
    svst1_vnum(ptrue, tmp_d, 1, ain_1);
  } 
}

inline void matmul_f16_1x2048_2048x240_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
    svbool_t ptrue = svptrue_b32();
  svbool_t _ptrue = svptrue_b16();
  svbool_t _half_ptrue = svwhilelt_b16_u64(0, 16);

  init_vec_16(ain);
  init_vec_16(bin);
  init_vec_16f16(c16in);
  init_vec_16f16(d16in);

  dup_0_16fp16(c16in);

  int m_index;
  float *tmp_a, *tmp_c;
  float16_t *tmp_b;
  tmp_c = C;  
  int kk = 0;
  float16_t a_fp16[2048];
  float16_t c_fp16[240];
  float16_t *tmp_f;
  tmp_f = c_fp16 ;
  for(int j = 0; j <2048; j++) {
    a_fp16[j]=A[j];
  }
    
  tmp_a = A;  
  tmp_c = C;

  for(kk = 0; kk < K; kk++) {
    float16_t scala_a16 = a_fp16[kk];
    tmp_b = B + kk * N ;
    svld1_vnum_fp16_16_240(d16in, tmp_b);
    svmla_z_f16_240(c16in, c16in, d16in, scala_a16 );
  }     

  svst1_vnum_fp16_8_240(c_fp16, c16in);
  for(int j = 0; j < 240; j++) {
      C[j]+= c_fp16[j];
  }
}

inline void matmul_f16_2x2048_2048x240_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
  svbool_t _ptrue = svptrue_b16();
  svbool_t _half_ptrue = svwhilelt_b16_u64(0, 16);

  init_vec_16f16(c16in1);
  init_vec_16f16(d16in);

  dup_0_16fp16(c16in1);
  int m_index;

  float *tmp_a;
  float16_t *tmp_b;
  int kk = 0;
  float16_t a_fp16[4096];
  float16_t c_fp16[480];
  float16_t *tmp_c;
  tmp_c = c_fp16 ;
  for(int j = 0; j <4096; j++) {
    a_fp16[j]=A[j];
  }
  
  for(int mm = 0; mm < M; mm++) {
    tmp_a = A + mm * K; 

    for(kk = 0; kk < K; kk++) {
      float16_t scala_a16_1 = a_fp16[kk+mm*K];
      
      tmp_b = B + kk * N ;

      svld1_vnum_fp16_16_240(d16in, tmp_b);
      svmla_z_f16_240(c16in1, c16in1, d16in, scala_a16_1 );

    }     
    svst1_vnum_fp16_8_240(tmp_c, c16in1);
    tmp_c = tmp_c + N ;
  } 
  for(int j = 0; j < 480; j++) {
    C[j]+= c_fp16[j];
  }
}

inline void matmul_f16_3x2048_2048x240_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
  svbool_t _ptrue = svptrue_b16();
  svbool_t _half_ptrue = svwhilelt_b16_u64(0, 16);

  init_vec_16f16(c16in1);
  init_vec_16f16(d16in);

  dup_0_16fp16(c16in1);
  int m_index;

  float *tmp_a;
  float16_t *tmp_b;
  int kk = 0;
  float16_t a_fp16[3*2048];
  float16_t c_fp16[240*3];
  float16_t *tmp_c;
  tmp_c = c_fp16 ;
  for(int j = 0; j <3*2048; j++) {
    a_fp16[j]=A[j];
  }
  
  for(int mm = 0; mm < M; mm++) {
    tmp_a = A + mm * K; 

    for(kk = 0; kk < K; kk++) {
      float16_t scala_a16_1 = a_fp16[kk+mm*K];
      
      tmp_b = B + kk * N ;

      svld1_vnum_fp16_16_240(d16in, tmp_b);
      svmla_z_f16_240(c16in1, c16in1, d16in, scala_a16_1 );

    }     
    svst1_vnum_fp16_8_240(tmp_c, c16in1);
    tmp_c = tmp_c + N ;
  } 
  for(int j = 0; j < 240*3; j++) {
    C[j]+= c_fp16[j];
  }
}

inline void matmul_f16_1x240_240x2048_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
  svbool_t _ptrue = svptrue_b16();

  init_vec_16f16(c16in);
  init_vec_16f16(d16in);

  float16_t *tmp_b;
  int kk = 0;
  float16_t a_fp16[240];
  float16_t c_fp16[2048] = {0.};
  float16_t *tmp_c;
  for(int j = 0; j <240; j++) {
    a_fp16[j]=A[j];
  }
  
  for(int ll = 0; ll < 4; ll++) {

    tmp_c = c_fp16 + ll * N / 4;
    svld1_vnum_f16_16(c16in, tmp_c);
    for(kk = 0; kk < K; kk++) {
      float16_t scala_a16_1 = a_fp16[kk];
      tmp_b = B + kk * N  + ll * N / 4;

      svld1_vnum_f16_16(d16in, tmp_b);
      
      svmla_z_f16_16(c16in, c16in, d16in, scala_a16_1 );
      
    } 
    svst1_vnum_f16_16(tmp_c, c16in);   
  } 
  
  for(int j = 0; j < 2048; j++) {
    C[j]+= c_fp16[j];
  }
}

inline void matmul_f16_2x240_240x2048_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
  svbool_t _ptrue = svptrue_b16();

  init_vec_16f16(c16in1);
  init_vec_16f16(c16in2);
  init_vec_16f16(d16in);

  int m_index;

  float16_t *tmp_b;
  int kk = 0;
  float16_t a_fp16[480];
  float16_t c_fp16[4096] = {0};
  float16_t *tmp_c1, *tmp_c2;
  for(int j = 0; j <480; j++) {
    a_fp16[j]=A[j];
  }
         
  for(int ll = 0; ll < 8; ll++) {
    tmp_c1 = c_fp16 + ll * N / 8;
    tmp_c2 = c_fp16 + N + ll * N / 8;
    svld1_vnum_f16_8(c16in1, tmp_c1);
    svld1_vnum_f16_8(c16in2, tmp_c2);

    for(kk = 0; kk < K; kk++) {
      float16_t scala_a16_1 = a_fp16[kk];
      float16_t scala_a16_2 = a_fp16[kk+K];
      tmp_b = B + kk * N  + ll * N / 8;

      svld1_vnum_f16_8(d16in, tmp_b);
      svmla_z_f16_8(c16in1, c16in1, d16in, scala_a16_1 );
      svmla_z_f16_8(c16in2, c16in2, d16in, scala_a16_2 );
    } 
    tmp_c1 = c_fp16 + ll * N / 8;
    tmp_c2 = c_fp16 + N + ll * N / 8;
    svst1_vnum_f16_8(tmp_c1, c16in1); 
    svst1_vnum_f16_8(tmp_c2, c16in2);
  }
  
  for(int j = 0; j < 4096; j++) {
    C[j]+= c_fp16[j];
  }
}

inline void matmul_f16_3x240_240x2048_nn(const int M, const int N, const int K,
  float *A, float16_t* B, float *C) {
  svbool_t _ptrue = svptrue_b16();

  init_vec_16f16(c16in1);
  init_vec_16f16(c16in2);
  init_vec_16f16(c16in3);
  init_vec_16f16(d16in);

  int m_index;

  float16_t *tmp_b;
  int kk = 0;
  float16_t a_fp16[3*240];
  float16_t c_fp16[3*2048] = {0};
  float16_t *tmp_c1, *tmp_c2, *tmp_c3;
  for(int j = 0; j <3*240; j++) {
    a_fp16[j]=A[j];
  }
         
  for(int ll = 0; ll < 8; ll++) {
    tmp_c1 = c_fp16 + 0 * N +ll * N / 8;
    tmp_c2 = c_fp16 + 1 * N + ll * N / 8;
    tmp_c3 = c_fp16 + 2 * N + ll * N / 8;
    svld1_vnum_f16_8(c16in1, tmp_c1);
    svld1_vnum_f16_8(c16in2, tmp_c2);
    svld1_vnum_f16_8(c16in3, tmp_c3);

    for(kk = 0; kk < K; kk++) {
      float16_t scala_a16_1 = a_fp16[kk+K*0];
      float16_t scala_a16_2 = a_fp16[kk+K*1];
      float16_t scala_a16_3 = a_fp16[kk+K*2];
      tmp_b = B + kk * N  + ll * N / 8;

      svld1_vnum_f16_8(d16in, tmp_b);
      svmla_z_f16_8(c16in1, c16in1, d16in, scala_a16_1 );
      svmla_z_f16_8(c16in2, c16in2, d16in, scala_a16_2 );
      svmla_z_f16_8(c16in3, c16in3, d16in, scala_a16_3 );
    } 
    tmp_c1 = c_fp16 + 0 * N + ll * N / 8;
    tmp_c2 = c_fp16 + 1 * N + ll * N / 8;
    tmp_c3 = c_fp16 + 2 * N + ll * N / 8;
    svst1_vnum_f16_8(tmp_c1, c16in1); 
    svst1_vnum_f16_8(tmp_c2, c16in2);
    svst1_vnum_f16_8(tmp_c3, c16in3);
  }
  
  for(int j = 0; j < 3*2048; j++) {
    C[j]+= c_fp16[j];
  }
}


#endif


//////////////////////////////////////////////
///////////////// matmul /////////////////////
//////////////////////////////////////////////


void matmul(const int m, const int n, const int k,
  double *A, double* B, double *C, double *D);

void matmul(const int m, const int n, const int k,
  float *A, float* B, float *C, float *D) ;

void matmul(const int m, const int n, const int k,
  float *A, float16_t* B, float *C, float *D) ;

void matmul_3d(const int t, const int m, const int n, const int k,
  double* A, double* B, double* C, bool _transpose_a, bool _transpose_b);

void matmul_3d(const int t, const int m, const int n, const int k,
  float* A, float* B, float* C, bool _transpose_a, bool _transpose_b) ;

}
#endif