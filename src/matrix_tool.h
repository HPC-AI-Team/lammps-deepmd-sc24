#ifndef MATRIX_TOOL_H
#define MATRIX_TOOL_H

#include <immintrin.h>  // AVX-512 intrinsic header


// #ifdef __ARM_FEATURE_SVE
// #include <arm_sve.h>

#define init_vec_32(in) \
    __m512 in##_0;   \
    __m512 in##_1;   \
    __m512 in##_2;   \
    __m512 in##_3;   \
    __m512 in##_4;   \
    __m512 in##_5;   \
    __m512 in##_6;   \
    __m512 in##_7;   \
    __m512 in##_8;   \
    __m512 in##_9;   \
    __m512 in##_10;  \
    __m512 in##_11;  \
    __m512 in##_12;  \
    __m512 in##_13;  \
    __m512 in##_14;  \
    __m512 in##_15;  \
    __m512 in##_16;  \
    __m512 in##_17;  \
    __m512 in##_18;  \
    __m512 in##_19;  \
    __m512 in##_20;  \
    __m512 in##_21;  \
    __m512 in##_22;  \
    __m512 in##_23;  \
    __m512 in##_24;  \
    __m512 in##_25;  \
    __m512 in##_26;  \
    __m512 in##_27;  \
    __m512 in##_28;  \
    __m512 in##_29;  \
    __m512 in##_30;  \
    __m512 in##_31;  


#define init_vec_16(in) \
    __m512 in##_0;   \
    __m512 in##_1;   \
    __m512 in##_2;   \
    __m512 in##_3;   \
    __m512 in##_4;   \
    __m512 in##_5;   \
    __m512 in##_6;   \
    __m512 in##_7;   \
    __m512 in##_8;   \
    __m512 in##_9;   \
    __m512 in##_10;  \
    __m512 in##_11;  \
    __m512 in##_12;  \
    __m512 in##_13;  \
    __m512 in##_14;  \
    __m512 in##_15;  

#define init_vec_8(in) \
    __m512 in##_0;   \
    __m512 in##_1;   \
    __m512 in##_2;   \
    __m512 in##_3;   \
    __m512 in##_4;   \
    __m512 in##_5;   \
    __m512 in##_6;   \
    __m512 in##_7;   

#define init_vec_4(in) \
    __m512 in##_0;   \
    __m512 in##_1;   \
    __m512 in##_2;   \
    __m512 in##_3;   
     
#define svld1_vnum_32(in, vec) \
  in##_0 = _mm512_load_ps(vec + 16 * 0);  \
  in##_1 = _mm512_load_ps(vec + 16 * 1);  \
  in##_2 = _mm512_load_ps(vec + 16 * 2);  \
  in##_3 = _mm512_load_ps(vec + 16 * 3);  \
  in##_4 = _mm512_load_ps(vec + 16 * 4);  \
  in##_5 = _mm512_load_ps(vec + 16 * 5);  \
  in##_6 = _mm512_load_ps(vec + 16 * 6);  \
  in##_7 = _mm512_load_ps(vec + 16 * 7);  \
  in##_8 = _mm512_load_ps(vec + 16 * 8);  \
  in##_9 = _mm512_load_ps(vec + 16 * 9);  \
  in##_10  = _mm512_load_ps(vec + 16 * 10); \
  in##_11  = _mm512_load_ps(vec + 16 * 11); \
  in##_12  = _mm512_load_ps(vec + 16 * 12); \
  in##_13  = _mm512_load_ps(vec + 16 * 13); \
  in##_14  = _mm512_load_ps(vec + 16 * 14); \
  in##_15  = _mm512_load_ps(vec + 16 * 15); \
  in##_16  = _mm512_load_ps(vec + 16 * 16); \
  in##_17  = _mm512_load_ps(vec + 16 * 17); \
  in##_18  = _mm512_load_ps(vec + 16 * 18); \
  in##_19  = _mm512_load_ps(vec + 16 * 19); \
  in##_20  = _mm512_load_ps(vec + 16 * 20); \
  in##_21  = _mm512_load_ps(vec + 16 * 21); \
  in##_22  = _mm512_load_ps(vec + 16 * 22); \
  in##_23  = _mm512_load_ps(vec + 16 * 23); \
  in##_24  = _mm512_load_ps(vec + 16 * 24); \
  in##_25  = _mm512_load_ps(vec + 16 * 25); \
  in##_26  = _mm512_load_ps(vec + 16 * 26); \
  in##_27  = _mm512_load_ps(vec + 16 * 27); \
  in##_28  = _mm512_load_ps(vec + 16 * 28); \
  in##_29  = _mm512_load_ps(vec + 16 * 29); \
  in##_30  = _mm512_load_ps(vec + 16 * 30); \
  in##_31  = _mm512_load_ps(vec + 16 * 31); 

#define svld1_vnum_16(in, vec) \
  in##_0 = _mm512_load_ps(vec + 16 * 0);  \
  in##_1 = _mm512_load_ps(vec + 16 * 1);  \
  in##_2 = _mm512_load_ps(vec + 16 * 2);  \
  in##_3 = _mm512_load_ps(vec + 16 * 3);  \
  in##_4 = _mm512_load_ps(vec + 16 * 4);  \
  in##_5 = _mm512_load_ps(vec + 16 * 5);  \
  in##_6 = _mm512_load_ps(vec + 16 * 6);  \
  in##_7 = _mm512_load_ps(vec + 16 * 7);  \
  in##_8 = _mm512_load_ps(vec + 16 * 8);  \
  in##_9 = _mm512_load_ps(vec + 16 * 9);  \
  in##_10  = _mm512_load_ps(vec + 16 * 10); \
  in##_11  = _mm512_load_ps(vec + 16 * 11); \
  in##_12  = _mm512_load_ps(vec + 16 * 12); \
  in##_13  = _mm512_load_ps(vec + 16 * 13); \
  in##_14  = _mm512_load_ps(vec + 16 * 14); \
  in##_15  = _mm512_load_ps(vec + 16 * 15); 

#define svld1_vnum_15(in, vec) \
  in##_0 = _mm512_load_ps(vec + 16 * 0);  \
  in##_1 = _mm512_load_ps(vec + 16 * 1);  \
  in##_2 = _mm512_load_ps(vec + 16 * 2);  \
  in##_3 = _mm512_load_ps(vec + 16 * 3);  \
  in##_4 = _mm512_load_ps(vec + 16 * 4);  \
  in##_5 = _mm512_load_ps(vec + 16 * 5);  \
  in##_6 = _mm512_load_ps(vec + 16 * 6);  \
  in##_7 = _mm512_load_ps(vec + 16 * 7);  \
  in##_8 = _mm512_load_ps(vec + 16 * 8);  \
  in##_9 = _mm512_load_ps(vec + 16 * 9);  \
  in##_10  = _mm512_load_ps(vec + 16 * 10); \
  in##_11  = _mm512_load_ps(vec + 16 * 11); \
  in##_12  = _mm512_load_ps(vec + 16 * 12); \
  in##_13  = _mm512_load_ps(vec + 16 * 13); \
  in##_14  = _mm512_load_ps(vec + 16 * 14);

#define svld1_vnum_8(in, vec) \
  in##_0 = _mm512_load_ps(vec + 16 * 0);  \
  in##_1 = _mm512_load_ps(vec + 16 * 1);  \
  in##_2 = _mm512_load_ps(vec + 16 * 2);  \
  in##_3 = _mm512_load_ps(vec + 16 * 3);  \
  in##_4 = _mm512_load_ps(vec + 16 * 4);  \
  in##_5 = _mm512_load_ps(vec + 16 * 5);  \
  in##_6 = _mm512_load_ps(vec + 16 * 6);  \
  in##_7 = _mm512_load_ps(vec + 16 * 7);  

#define svld1_vnum_4(in, vec) \
  in##_0 = _mm512_load_ps(vec + 16 * 0);  \
  in##_1 = _mm512_load_ps(vec + 16 * 1);  \
  in##_2 = _mm512_load_ps(vec + 16 * 2);  \
  in##_3 = _mm512_load_ps(vec + 16 * 3);  


#define svmla_z_32(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps( in2##_0, in3, in1##_0);  \
  out##_1 = _mm512_fmadd_ps( in2##_1, in3, in1##_1);  \
  out##_2 = _mm512_fmadd_ps( in2##_2, in3, in1##_2);  \
  out##_3 = _mm512_fmadd_ps( in2##_3, in3, in1##_3);  \
  out##_4 = _mm512_fmadd_ps( in2##_4, in3, in1##_4);  \
  out##_5 = _mm512_fmadd_ps( in2##_5, in3, in1##_5);  \
  out##_6 = _mm512_fmadd_ps( in2##_6, in3, in1##_6);  \
  out##_7 = _mm512_fmadd_ps( in2##_7, in3, in1##_7);  \
  out##_8 = _mm512_fmadd_ps( in2##_8, in3, in1##_8);  \
  out##_9 = _mm512_fmadd_ps( in2##_9, in3, in1##_9);  \
  out##_10  = _mm512_fmadd_ps( in2##_10, in3, in1##_10); \
  out##_11  = _mm512_fmadd_ps( in2##_11, in3, in1##_11); \
  out##_12  = _mm512_fmadd_ps( in2##_12, in3, in1##_12); \
  out##_13  = _mm512_fmadd_ps( in2##_13, in3, in1##_13); \
  out##_14  = _mm512_fmadd_ps( in2##_14, in3, in1##_14); \
  out##_15  = _mm512_fmadd_ps( in2##_15, in3, in1##_15); \
  out##_16 = _mm512_fmadd_ps( in2##_16, in3, in1##_16);  \
  out##_17 = _mm512_fmadd_ps( in2##_17, in3, in1##_17);  \
  out##_18 = _mm512_fmadd_ps( in2##_18, in3, in1##_18);  \
  out##_19 = _mm512_fmadd_ps( in2##_19, in3, in1##_19);  \
  out##_20 = _mm512_fmadd_ps( in2##_20, in3, in1##_20);  \
  out##_21 = _mm512_fmadd_ps( in2##_21, in3, in1##_21);  \
  out##_22 = _mm512_fmadd_ps( in2##_22, in3, in1##_22);  \
  out##_23 = _mm512_fmadd_ps( in2##_23, in3, in1##_23);  \
  out##_24 = _mm512_fmadd_ps( in2##_24, in3, in1##_24);  \
  out##_25 = _mm512_fmadd_ps( in2##_25, in3, in1##_25);  \
  out##_26  = _mm512_fmadd_ps( in2##_26, in3, in1##_26); \
  out##_27  = _mm512_fmadd_ps( in2##_27, in3, in1##_27); \
  out##_28  = _mm512_fmadd_ps( in2##_28, in3, in1##_28); \
  out##_29  = _mm512_fmadd_ps( in2##_29, in3, in1##_29); \
  out##_30  = _mm512_fmadd_ps( in2##_30, in3, in1##_30); \
  out##_31  = _mm512_fmadd_ps( in2##_31, in3, in1##_31); 

#define svmla_z_16(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps( in2##_0, in3, in1##_0);  \
  out##_1 = _mm512_fmadd_ps( in2##_1, in3, in1##_1);  \
  out##_2 = _mm512_fmadd_ps( in2##_2, in3, in1##_2);  \
  out##_3 = _mm512_fmadd_ps( in2##_3, in3, in1##_3);  \
  out##_4 = _mm512_fmadd_ps( in2##_4, in3, in1##_4);  \
  out##_5 = _mm512_fmadd_ps( in2##_5, in3, in1##_5);  \
  out##_6 = _mm512_fmadd_ps( in2##_6, in3, in1##_6);  \
  out##_7 = _mm512_fmadd_ps( in2##_7, in3, in1##_7);  \
  out##_8 = _mm512_fmadd_ps( in2##_8, in3, in1##_8);  \
  out##_9 = _mm512_fmadd_ps( in2##_9, in3, in1##_9);  \
  out##_10  = _mm512_fmadd_ps( in2##_10, in3, in1##_10); \
  out##_11  = _mm512_fmadd_ps( in2##_11, in3, in1##_11); \
  out##_12  = _mm512_fmadd_ps( in2##_12, in3, in1##_12); \
  out##_13  = _mm512_fmadd_ps( in2##_13, in3, in1##_13); \
  out##_14  = _mm512_fmadd_ps( in2##_14, in3, in1##_14); \
  out##_15  = _mm512_fmadd_ps( in2##_15, in3, in1##_15); 

#define svmla_z_15(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps(in2##_0, in3, in1##_0);  \
  out##_1 = _mm512_fmadd_ps(in2##_1, in3, in1##_1);  \
  out##_2 = _mm512_fmadd_ps(in2##_2, in3, in1##_2);  \
  out##_3 = _mm512_fmadd_ps(in2##_3, in3, in1##_3);  \
  out##_4 = _mm512_fmadd_ps(in2##_4, in3, in1##_4);  \
  out##_5 = _mm512_fmadd_ps(in2##_5, in3, in1##_5);  \
  out##_6 = _mm512_fmadd_ps(in2##_6, in3, in1##_6);  \
  out##_7 = _mm512_fmadd_ps(in2##_7, in3, in1##_7);  \
  out##_8 = _mm512_fmadd_ps(in2##_8, in3, in1##_8);  \
  out##_9 = _mm512_fmadd_ps(in2##_9, in3, in1##_9);  \
  out##_10  = _mm512_fmadd_ps(in2##_10, in3, in1##_10); \
  out##_11  = _mm512_fmadd_ps(in2##_11, in3, in1##_11); \
  out##_12  = _mm512_fmadd_ps(in2##_12, in3, in1##_12); \
  out##_13  = _mm512_fmadd_ps(in2##_13, in3, in1##_13); \
  out##_14  = _mm512_fmadd_ps(in2##_14, in3, in1##_14); 

#define svmla_z_8(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps( in2##_0, in3, in1##_0);  \
  out##_1 = _mm512_fmadd_ps( in2##_1, in3, in1##_1);  \
  out##_2 = _mm512_fmadd_ps( in2##_2, in3, in1##_2);  \
  out##_3 = _mm512_fmadd_ps( in2##_3, in3, in1##_3);  \
  out##_4 = _mm512_fmadd_ps( in2##_4, in3, in1##_4);  \
  out##_5 = _mm512_fmadd_ps( in2##_5, in3, in1##_5);  \
  out##_6 = _mm512_fmadd_ps( in2##_6, in3, in1##_6);  \
  out##_7 = _mm512_fmadd_ps( in2##_7, in3, in1##_7);  

#define svmla_z_4(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps( in2##_0, in3, in1##_0);  \
  out##_1 = _mm512_fmadd_ps( in2##_1, in3, in1##_1);  \
  out##_2 = _mm512_fmadd_ps( in2##_2, in3, in1##_2);  \
  out##_3 = _mm512_fmadd_ps( in2##_3, in3, in1##_3);  


#define svst1_vnum_32(out, vec) \
  _mm512_store_ps(out + 16 * 0, vec##_0);  \
  _mm512_store_ps(out + 16 * 1, vec##_1);  \
  _mm512_store_ps(out + 16 * 2, vec##_2);  \
  _mm512_store_ps(out + 16 * 3, vec##_3);  \
  _mm512_store_ps(out + 16 * 4, vec##_4);  \
  _mm512_store_ps(out + 16 * 5, vec##_5);  \
  _mm512_store_ps(out + 16 * 6, vec##_6);  \
  _mm512_store_ps(out + 16 * 7, vec##_7);  \
  _mm512_store_ps(out + 16 * 8, vec##_8);  \
  _mm512_store_ps(out + 16 * 9, vec##_9);  \
  _mm512_store_ps(out + 16 * 10, vec##_10); \
  _mm512_store_ps(out + 16 * 11, vec##_11); \
  _mm512_store_ps(out + 16 * 12, vec##_12); \
  _mm512_store_ps(out + 16 * 13, vec##_13); \
  _mm512_store_ps(out + 16 * 14, vec##_14); \
  _mm512_store_ps(out + 16 * 15, vec##_15); \
  _mm512_store_ps(out + 16 * 16, vec##_16); \
  _mm512_store_ps(out + 16 * 17, vec##_17); \
  _mm512_store_ps(out + 16 * 18, vec##_18); \
  _mm512_store_ps(out + 16 * 19, vec##_19); \
  _mm512_store_ps(out + 16 * 20, vec##_20); \
  _mm512_store_ps(out + 16 * 21, vec##_21); \
  _mm512_store_ps(out + 16 * 22, vec##_22); \
  _mm512_store_ps(out + 16 * 23, vec##_23); \
  _mm512_store_ps(out + 16 * 24, vec##_24); \
  _mm512_store_ps(out + 16 * 25, vec##_25); \
  _mm512_store_ps(out + 16 * 26, vec##_26); \
  _mm512_store_ps(out + 16 * 27, vec##_27); \
  _mm512_store_ps(out + 16 * 28, vec##_28); \
  _mm512_store_ps(out + 16 * 29, vec##_29); \
  _mm512_store_ps(out + 16 * 30, vec##_30); \
  _mm512_store_ps(out + 16 * 31, vec##_31); 


#define svst1_vnum_16(out, vec) \
  _mm512_store_ps(out + 16 * 0, vec##_0);  \
  _mm512_store_ps(out + 16 * 1, vec##_1);  \
  _mm512_store_ps(out + 16 * 2, vec##_2);  \
  _mm512_store_ps(out + 16 * 3, vec##_3);  \
  _mm512_store_ps(out + 16 * 4, vec##_4);  \
  _mm512_store_ps(out + 16 * 5, vec##_5);  \
  _mm512_store_ps(out + 16 * 6, vec##_6);  \
  _mm512_store_ps(out + 16 * 7, vec##_7);  \
  _mm512_store_ps(out + 16 * 8, vec##_8);  \
  _mm512_store_ps(out + 16 * 9, vec##_9);  \
  _mm512_store_ps(out + 16 * 10, vec##_10); \
  _mm512_store_ps(out + 16 * 11, vec##_11); \
  _mm512_store_ps(out + 16 * 12, vec##_12); \
  _mm512_store_ps(out + 16 * 13, vec##_13); \
  _mm512_store_ps(out + 16 * 14, vec##_14); \
  _mm512_store_ps(out + 16 * 15, vec##_15); 

#define svst1_vnum_15(out, vec) \
  _mm512_store_ps(out + 16 * 0, vec##_0);  \
  _mm512_store_ps(out + 16 * 1, vec##_1);  \
  _mm512_store_ps(out + 16 * 2, vec##_2);  \
  _mm512_store_ps(out + 16 * 3, vec##_3);  \
  _mm512_store_ps(out + 16 * 4, vec##_4);  \
  _mm512_store_ps(out + 16 * 5, vec##_5);  \
  _mm512_store_ps(out + 16 * 6, vec##_6);  \
  _mm512_store_ps(out + 16 * 7, vec##_7);  \
  _mm512_store_ps(out + 16 * 8, vec##_8);  \
  _mm512_store_ps(out + 16 * 9, vec##_9);  \
  _mm512_store_ps(out + 16 * 10, vec##_10); \
  _mm512_store_ps(out + 16 * 11, vec##_11); \
  _mm512_store_ps(out + 16 * 12, vec##_12); \
  _mm512_store_ps(out + 16 * 13, vec##_13); \
  _mm512_store_ps(out + 16 * 14, vec##_14); 

#define svst1_vnum_8(out, vec) \
  _mm512_store_ps(out + 16 * 0, vec##_0);  \
  _mm512_store_ps(out + 16 * 1, vec##_1);  \
  _mm512_store_ps(out + 16 * 2, vec##_2);  \
  _mm512_store_ps(out + 16 * 3, vec##_3);  \
  _mm512_store_ps(out + 16 * 4, vec##_4);  \
  _mm512_store_ps(out + 16 * 5, vec##_5);  \
  _mm512_store_ps(out + 16 * 6, vec##_6);  \
  _mm512_store_ps(out + 16 * 7, vec##_7);  

#define svst1_vnum_4(out, vec) \
  _mm512_store_ps(out + 16 * 0, vec##_0);  \
  _mm512_store_ps(out + 16 * 1, vec##_1);  \
  _mm512_store_ps(out + 16 * 2, vec##_2);  \
  _mm512_store_ps(out + 16 * 3, vec##_3);  


#define svmla_z_t_8(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps(in2, in3[0], in1##_0);  \
  out##_1 = _mm512_fmadd_ps(in2, in3[1], in1##_1);  \
  out##_2 = _mm512_fmadd_ps(in2, in3[2], in1##_2);  \
  out##_3 = _mm512_fmadd_ps(in2, in3[3], in1##_3);  \
  out##_4 = _mm512_fmadd_ps(in2, in3[4], in1##_4);  \
  out##_5 = _mm512_fmadd_ps(in2, in3[5], in1##_5);  \
  out##_6 = _mm512_fmadd_ps(in2, in3[6], in1##_6);  \
  out##_7 = _mm512_fmadd_ps(in2, in3[7], in1##_7);  

#define svmla_z_vec_8(out, in1, in2, in3) \
  out##_0 = _mm512_fmadd_ps(in2, in3##0, in1##_0);  \
  out##_1 = _mm512_fmadd_ps(in2, in3##1, in1##_1);  \
  out##_2 = _mm512_fmadd_ps(in2, in3##2, in1##_2);  \
  out##_3 = _mm512_fmadd_ps(in2, in3##3, in1##_3);  \
  out##_4 = _mm512_fmadd_ps(in2, in3##4, in1##_4);  \
  out##_5 = _mm512_fmadd_ps(in2, in3##5, in1##_5);  \
  out##_6 = _mm512_fmadd_ps(in2, in3##6, in1##_6);  \
  out##_7 = _mm512_fmadd_ps(in2, in3##7, in1##_7);  

#define svmul_z_t_8(out, in2, in3) \
  out##_0 = _mm512_mul_ps( in2, in3[0]);  \
  out##_1 = _mm512_mul_ps( in2, in3[1]);  \
  out##_2 = _mm512_mul_ps( in2, in3[2]);  \
  out##_3 = _mm512_mul_ps( in2, in3[3]);  \
  out##_4 = _mm512_mul_ps( in2, in3[4]);  \
  out##_5 = _mm512_mul_ps( in2, in3[5]);  \
  out##_6 = _mm512_mul_ps( in2, in3[6]);  \
  out##_7 = _mm512_mul_ps( in2, in3[7]);  

#define svmul_z_t_vec_8(out, in2, in3) \
  out##_0 = _mm512_mul_ps( in2, in3##0);  \
  out##_1 = _mm512_mul_ps( in2, in3##1);  \
  out##_2 = _mm512_mul_ps( in2, in3##2);  \
  out##_3 = _mm512_mul_ps( in2, in3##3);  \
  out##_4 = _mm512_mul_ps( in2, in3##4);  \
  out##_5 = _mm512_mul_ps( in2, in3##5);  \
  out##_6 = _mm512_mul_ps( in2, in3##6);  \
  out##_7 = _mm512_mul_ps( in2, in3##7);  


#define svmul_z_15(out, in2, in3) \
  out##_0 = _mm512_mul_ps( in2##_0, in3##_0);  \
  out##_1 = _mm512_mul_ps( in2##_1, in3##_1);  \
  out##_2 = _mm512_mul_ps( in2##_2, in3##_2);  \
  out##_3 = _mm512_mul_ps( in2##_3, in3##_3);  \
  out##_4 = _mm512_mul_ps( in2##_4, in3##_4);  \
  out##_5 = _mm512_mul_ps( in2##_5, in3##_5);  \
  out##_6 = _mm512_mul_ps( in2##_6, in3##_6);  \
  out##_7 = _mm512_mul_ps( in2##_7, in3##_7);  \
  out##_8 = _mm512_mul_ps( in2##_8, in3##_8);  \
  out##_9 = _mm512_mul_ps( in2##_9, in3##_9);  \
  out##_10 = _mm512_mul_ps( in2##_10, in3##_10);  \
  out##_11 = _mm512_mul_ps( in2##_11, in3##_11);  \
  out##_12 = _mm512_mul_ps( in2##_12, in3##_12);  \
  out##_13 = _mm512_mul_ps( in2##_13, in3##_13);  \
  out##_14 = _mm512_mul_ps( in2##_14, in3##_14);

#define svaddv_15(out, in2) \
  out += _mm512_reduce_add_ps(in2##_0);  \
  out += _mm512_reduce_add_ps(in2##_1);  \
  out += _mm512_reduce_add_ps(in2##_2);  \
  out += _mm512_reduce_add_ps(in2##_3);  \
  out += _mm512_reduce_add_ps(in2##_4);  \
  out += _mm512_reduce_add_ps(in2##_5);  \
  out += _mm512_reduce_add_ps(in2##_6);  \
  out += _mm512_reduce_add_ps(in2##_7);  \
  out += _mm512_reduce_add_ps(in2##_8);  \
  out += _mm512_reduce_add_ps(in2##_9);  \
  out += _mm512_reduce_add_ps(in2##_10);  \
  out += _mm512_reduce_add_ps(in2##_11);  \
  out += _mm512_reduce_add_ps(in2##_12);  \
  out += _mm512_reduce_add_ps(in2##_13);  \
  out += _mm512_reduce_add_ps(in2##_14); 


#endif