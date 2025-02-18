#ifndef DEEPMD_COMMON_H
#define DEEPMD_COMMON_H 

// #define WITH_TENSOR_FLOW
#define COMBIN_OMP

#define SPLIT_TYPE_EMBEDDING

// #define __ARM_FEATURE_SVE

// #define HIGH_PREC

// #define OPT_CBLAS

// make CC=fcc TARGET=ARMV8SVE  NOFORTRAN=1 -j48 |& tee compile.log

#include <immintrin.h>  // AVX-512 intrinsic header


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

// #include <cblas.h>
#include <mkl_cblas.h>

// #include "OPENBLAS/include/cblas.h"
// #include "../../CBLAS/openblas/include/cblas.h"

#include <math.h>
#include <string>
#include <vector>
#include <cstring>
#include <stdlib.h>
#include <iostream>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
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

// #ifdef __ARM_FEATURE_SVE
inline void matmul_1x240_240x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {
  // svbool_t ptrue = svptrue_b32();

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

      __m512 vll2 = _mm512_set1_ps(scala_a);
      svmla_z_15(ain, ain, bin, vll2)
    }

    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_1x240_240x2048(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  // svbool_t ptrue = svptrue_b32();

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

        __m512 vll2 = _mm512_set1_ps(scala_a);        
        svmla_z_16(ain, ain, bin, vll2)
      }

      svst1_vnum_16(tmp_d, ain);
    }    
  }
}


inline void matmul_1x2048_2048x240(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  // svbool_t ptrue = svptrue_b32();

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

      __m512 vll2 = _mm512_set1_ps(scala_a);
      svmla_z_15(ain, ain, bin, vll2)
    }
    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_4x128_128x16(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  // svbool_t ptrue = svptrue_b32();

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
      __m512 vll2 = _mm512_set1_ps(scala_a);        

      svmla_z_15(ain, ain, bin, vll2)
    }
    svst1_vnum_15(tmp_d, ain);
  }
}

inline void matmul_128x4_4x16_tn(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  // svbool_t ptrue = svptrue_b32();

  init_vec_8(ain);
  init_vec_8(bin);

  float *tmp_a, *tmp_b, *tmp_d;

  tmp_b = b_fp32;
  bin_0 = _mm512_load_ps(tmp_b + 16 * 0 );
  bin_1 = _mm512_load_ps(tmp_b + 16 * 1 );
  bin_2 = _mm512_load_ps(tmp_b + 16 * 2 );
  bin_3 = _mm512_load_ps(tmp_b + 16 * 3 );

  __m512 vll0;  __m512 vll1;  __m512 vll2;  __m512 vll3;  __m512 vll4;  __m512 vll5;  __m512 vll6;  __m512 vll7;
  for(int mm = 0; mm < M; mm += 8) {
    tmp_a = a_fp32 + mm;

    vll0 = _mm512_set1_ps(tmp_a[0]);        
    vll1 = _mm512_set1_ps(tmp_a[1]);        
    vll2 = _mm512_set1_ps(tmp_a[2]);        
    vll3 = _mm512_set1_ps(tmp_a[3]);        
    vll4 = _mm512_set1_ps(tmp_a[4]);        
    vll5 = _mm512_set1_ps(tmp_a[5]);        
    vll6 = _mm512_set1_ps(tmp_a[6]);        
    vll7 = _mm512_set1_ps(tmp_a[7]);        

    svmul_z_t_vec_8(ain, bin_0, vll);

    tmp_a += M;
    vll0 = _mm512_set1_ps(tmp_a[0]);        
    vll1 = _mm512_set1_ps(tmp_a[1]);        
    vll2 = _mm512_set1_ps(tmp_a[2]);        
    vll3 = _mm512_set1_ps(tmp_a[3]);        
    vll4 = _mm512_set1_ps(tmp_a[4]);        
    vll5 = _mm512_set1_ps(tmp_a[5]);        
    vll6 = _mm512_set1_ps(tmp_a[6]);        
    vll7 = _mm512_set1_ps(tmp_a[7]);    
    svmla_z_vec_8(ain, ain, bin_1, vll);

    tmp_a += M;
    vll0 = _mm512_set1_ps(tmp_a[0]);        
    vll1 = _mm512_set1_ps(tmp_a[1]);        
    vll2 = _mm512_set1_ps(tmp_a[2]);        
    vll3 = _mm512_set1_ps(tmp_a[3]);        
    vll4 = _mm512_set1_ps(tmp_a[4]);        
    vll5 = _mm512_set1_ps(tmp_a[5]);        
    vll6 = _mm512_set1_ps(tmp_a[6]);        
    vll7 = _mm512_set1_ps(tmp_a[7]);    
    svmla_z_vec_8(ain, ain, bin_2, vll);

    tmp_a += M;
    vll0 = _mm512_set1_ps(tmp_a[0]);        
    vll1 = _mm512_set1_ps(tmp_a[1]);        
    vll2 = _mm512_set1_ps(tmp_a[2]);        
    vll3 = _mm512_set1_ps(tmp_a[3]);        
    vll4 = _mm512_set1_ps(tmp_a[4]);        
    vll5 = _mm512_set1_ps(tmp_a[5]);        
    vll6 = _mm512_set1_ps(tmp_a[6]);        
    vll7 = _mm512_set1_ps(tmp_a[7]);    
    svmla_z_vec_8(ain, ain, bin_3, vll);

    tmp_d = d_fp32 + mm * N;
    svst1_vnum_8(tmp_d, ain);
  }
}

inline void matmul_4x16_16x128_nt(const int M, const int N, const int K,
  float *A, float* B, float *C) {

  // svbool_t ptrue = svptrue_b32();
  
  float *tmp_a, *tmp_b, *tmp_c;
  tmp_a = A; 
  tmp_b = B;
  tmp_c = C; 
  __m512 ain0;   
  __m512 ain1;   
  __m512 ain2;   
  __m512 ain3;

  __m512 bin0;   
  __m512 bin1;   
  __m512 bin2;   
  __m512 bin3; 

  __m512 cin0;   
  __m512 cin1;   
  __m512 cin2;   
  __m512 cin3; 

  ain0 = _mm512_load_ps(tmp_a + 16 * 0);  
  ain1 = _mm512_load_ps(tmp_a + 16 * 1);  
  ain2 = _mm512_load_ps(tmp_a + 16 * 2);  
  ain3 = _mm512_load_ps(tmp_a + 16 * 3);

  for(int i = 0; i <32; i++) {    
    tmp_b = B + i*64 ;
    bin0 = _mm512_load_ps(tmp_b + 16 * 0);  
    bin1 = _mm512_load_ps(tmp_b + 16 * 1);  
    bin2 = _mm512_load_ps(tmp_b + 16 * 2);  
    bin3 = _mm512_load_ps(tmp_b + 16 * 3);

    cin0 = _mm512_setzero_ps(); 
    cin1 = _mm512_setzero_ps(); 
    cin2 = _mm512_setzero_ps(); 
    cin3 = _mm512_setzero_ps(); 

    cin0 = _mm512_mul_ps(bin0, ain0) ; 
    tmp_c[i*4] = _mm512_reduce_add_ps(cin0 );
    cin1 = _mm512_mul_ps(bin0, ain1) ; 
    tmp_c[i*4+128] = _mm512_reduce_add_ps(cin1 );
    cin2 = _mm512_mul_ps(bin0, ain2) ; 
    tmp_c[i*4+256] = _mm512_reduce_add_ps(cin2 );
    cin3 = _mm512_mul_ps(bin0, ain3) ; 
    tmp_c[i*4+384] = _mm512_reduce_add_ps(cin3 );

    cin0 = _mm512_mul_ps(bin1, ain0) ; 
    tmp_c[i*4+1] = _mm512_reduce_add_ps(cin0 );
    cin1 = _mm512_mul_ps(bin1, ain1) ; 
    tmp_c[i*4+128+1] = _mm512_reduce_add_ps(cin1 );
    cin2 = _mm512_mul_ps(bin1, ain2) ; 
    tmp_c[i*4+256+1] = _mm512_reduce_add_ps(cin2 );
    cin3 = _mm512_mul_ps(bin1, ain3) ; 
    tmp_c[i*4+384+1] = _mm512_reduce_add_ps(cin3 );

    cin0 = _mm512_mul_ps(bin2, ain0) ; 
    tmp_c[i*4+2] = _mm512_reduce_add_ps(cin0 );
    cin1 = _mm512_mul_ps(bin2, ain1) ; 
    tmp_c[i*4+128+2] = _mm512_reduce_add_ps(cin1 );
    cin2 = _mm512_mul_ps(bin2, ain2) ; 
    tmp_c[i*4+256+2] = _mm512_reduce_add_ps(cin2 );
    cin3 = _mm512_mul_ps(bin2, ain3) ; 
    tmp_c[i*4+384+2] = _mm512_reduce_add_ps(cin3 );

    cin0 = _mm512_mul_ps(bin3, ain0) ; 
    tmp_c[i*4+3] = _mm512_reduce_add_ps(cin0 );
    cin1 = _mm512_mul_ps(bin3, ain1) ; 
    tmp_c[i*4+128+3] = _mm512_reduce_add_ps(cin1 );
    cin2 = _mm512_mul_ps(bin3, ain2) ; 
    tmp_c[i*4+256+3] = _mm512_reduce_add_ps(cin2 );
    cin3 = _mm512_mul_ps(bin3, ain3) ; 
    tmp_c[i*4+384+3] = _mm512_reduce_add_ps(cin3 );
  }
}



inline void matmul_4x128_128x16_nn(const int M, const int N, const int K,
  float *A, float* B, float *C) {

    __m512 bin_0, cin_0;
    __m512 bin_1, cin_1;
    __m512 bin_2, cin_2;
    __m512 bin_3, cin_3;

    cin_0 = _mm512_setzero_ps(); 
    cin_1 = _mm512_setzero_ps(); 
    cin_2 = _mm512_setzero_ps(); 
    cin_3 = _mm512_setzero_ps(); 

    for(int kk = 0; kk < K; kk++) {
      float a0 = A[0*K+kk];
      float a1 = A[1*K+kk];
      float a2 = A[2*K+kk];
      float a3 = A[3*K+kk];

      bin_0 = _mm512_load_ps(B + 16 * kk);

      __m512 vll0 = _mm512_set1_ps(a0);        
      __m512 vll1 = _mm512_set1_ps(a1);        
      __m512 vll2 = _mm512_set1_ps(a2);        
      __m512 vll3 = _mm512_set1_ps(a3);

      cin_0 = _mm512_fmadd_ps(bin_0,  vll0, cin_0);
      cin_1 = _mm512_fmadd_ps(bin_0,  vll1, cin_1);
      cin_2 = _mm512_fmadd_ps(bin_0,  vll2, cin_2);
      cin_3 = _mm512_fmadd_ps(bin_0,  vll3, cin_3);


    }
    _mm512_store_ps(C + 16 * 0, cin_0);
    _mm512_store_ps(C + 16 * 1, cin_1);
    _mm512_store_ps(C + 16 * 2, cin_2);
    _mm512_store_ps(C + 16 * 3, cin_3);
}


inline void matmul_1x240_240x1(const int M, const int N, const int K,
  float *a_fp32, float* b_fp32, float *d_fp32) {

  // svbool_t ptrue = svptrue_b32();

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

  // svbool_t ptrue = svptrue_b32();

  init_vec_8(ain);
  init_vec_8(bin);

  float *tmp_a, *tmp_b, *tmp_d;

  // load B
  tmp_b = b_fp32;
  bin_0 = _mm512_load_ps(tmp_b + 16 * 0 );
  bin_1 = _mm512_load_ps(tmp_b + 16 * 1 );
  bin_2 = _mm512_load_ps(tmp_b + 16 * 2 );
  bin_3 = _mm512_load_ps(tmp_b + 16 * 3 );

  for(int mm = 0; mm < M; mm+=2) {
    tmp_a = a_fp32 + mm * K;
    tmp_d = d_fp32 + mm * N;

    
      __m512 vll0 = _mm512_set1_ps(tmp_a[0]);        
      __m512 vll1 = _mm512_set1_ps(tmp_a[1]);        
      __m512 vll2 = _mm512_set1_ps(tmp_a[2]);        
      __m512 vll3 = _mm512_set1_ps(tmp_a[3]);  
      __m512 vlr0 = _mm512_set1_ps(tmp_a[0+K]);        
      __m512 vlr1 = _mm512_set1_ps(tmp_a[1+K]);        
      __m512 vlr2 = _mm512_set1_ps(tmp_a[2+K]);        
      __m512 vlr3 = _mm512_set1_ps(tmp_a[3+K]);  

    ain_0 = _mm512_mul_ps(bin_0,  vll0);
    ain_0 = _mm512_fmadd_ps(bin_1,  vll1, ain_0);
    ain_0 = _mm512_fmadd_ps(bin_2,  vll2, ain_0);
    ain_0 = _mm512_fmadd_ps(bin_3,  vll3, ain_0);

    ain_1 = _mm512_mul_ps(bin_0,  vlr0);
    ain_1 = _mm512_fmadd_ps(bin_1,  vlr1, ain_1);
    ain_1 = _mm512_fmadd_ps(bin_2,  vlr2, ain_1);
    ain_1 = _mm512_fmadd_ps(bin_3,  vlr3, ain_1);

    _mm512_store_ps(tmp_d + 16 * 0, ain_0);
    _mm512_store_ps(tmp_d + 16 * 1, ain_1);
  } 
}

// #endif


//////////////////////////////////////////////
///////////////// matmul /////////////////////
//////////////////////////////////////////////


void matmul(const int m, const int n, const int k,
  double *A, double* B, double *C, double *D);

void matmul(const int m, const int n, const int k,
  float *A, float* B, float *C, float *D) ;

// void matmul(const int m, const int n, const int k,
//   float *A, _Float16* B, float *C, float *D) ;

void matmul_3d(const int t, const int m, const int n, const int k,
  double* A, double* B, double* C, bool _transpose_a, bool _transpose_b);

void matmul_3d(const int t, const int m, const int n, const int k,
  float* A, float* B, float* C, bool _transpose_a, bool _transpose_b) ;

}
#endif