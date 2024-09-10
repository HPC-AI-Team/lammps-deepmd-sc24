#ifndef MATRIX_TOOL_H
#define MATRIX_TOOL_H

#include <arm_sve.h>

#define init_vec_32(in) \
    svfloat32_t in##_0;   \
    svfloat32_t in##_1;   \
    svfloat32_t in##_2;   \
    svfloat32_t in##_3;   \
    svfloat32_t in##_4;   \
    svfloat32_t in##_5;   \
    svfloat32_t in##_6;   \
    svfloat32_t in##_7;   \
    svfloat32_t in##_8;   \
    svfloat32_t in##_9;   \
    svfloat32_t in##_10;  \
    svfloat32_t in##_11;  \
    svfloat32_t in##_12;  \
    svfloat32_t in##_13;  \
    svfloat32_t in##_14;  \
    svfloat32_t in##_15;  \
    svfloat32_t in##_16;  \
    svfloat32_t in##_17;  \
    svfloat32_t in##_18;  \
    svfloat32_t in##_19;  \
    svfloat32_t in##_20;  \
    svfloat32_t in##_21;  \
    svfloat32_t in##_22;  \
    svfloat32_t in##_23;  \
    svfloat32_t in##_24;  \
    svfloat32_t in##_25;  \
    svfloat32_t in##_26;  \
    svfloat32_t in##_27;  \
    svfloat32_t in##_28;  \
    svfloat32_t in##_29;  \
    svfloat32_t in##_30;  \
    svfloat32_t in##_31;  


#define init_vec_16(in) \
    svfloat32_t in##_0;   \
    svfloat32_t in##_1;   \
    svfloat32_t in##_2;   \
    svfloat32_t in##_3;   \
    svfloat32_t in##_4;   \
    svfloat32_t in##_5;   \
    svfloat32_t in##_6;   \
    svfloat32_t in##_7;   \
    svfloat32_t in##_8;   \
    svfloat32_t in##_9;   \
    svfloat32_t in##_10;  \
    svfloat32_t in##_11;  \
    svfloat32_t in##_12;  \
    svfloat32_t in##_13;  \
    svfloat32_t in##_14;  \
    svfloat32_t in##_15;  

#define init_vec_8(in) \
    svfloat32_t in##_0;   \
    svfloat32_t in##_1;   \
    svfloat32_t in##_2;   \
    svfloat32_t in##_3;   \
    svfloat32_t in##_4;   \
    svfloat32_t in##_5;   \
    svfloat32_t in##_6;   \
    svfloat32_t in##_7;   

#define init_vec_4(in) \
    svfloat32_t in##_0;   \
    svfloat32_t in##_1;   \
    svfloat32_t in##_2;   \
    svfloat32_t in##_3;   
     
#define svld1_vnum_32(in, vec) \
  in##_0 = svld1_vnum(ptrue, vec, 0);  \
  in##_1 = svld1_vnum(ptrue, vec, 1);  \
  in##_2 = svld1_vnum(ptrue, vec, 2);  \
  in##_3 = svld1_vnum(ptrue, vec, 3);  \
  in##_4 = svld1_vnum(ptrue, vec, 4);  \
  in##_5 = svld1_vnum(ptrue, vec, 5);  \
  in##_6 = svld1_vnum(ptrue, vec, 6);  \
  in##_7 = svld1_vnum(ptrue, vec, 7);  \
  in##_8 = svld1_vnum(ptrue, vec, 8);  \
  in##_9 = svld1_vnum(ptrue, vec, 9);  \
  in##_10  = svld1_vnum(ptrue, vec, 10); \
  in##_11  = svld1_vnum(ptrue, vec, 11); \
  in##_12  = svld1_vnum(ptrue, vec, 12); \
  in##_13  = svld1_vnum(ptrue, vec, 13); \
  in##_14  = svld1_vnum(ptrue, vec, 14); \
  in##_15  = svld1_vnum(ptrue, vec, 15); \
  in##_16  = svld1_vnum(ptrue, vec, 16); \
  in##_17  = svld1_vnum(ptrue, vec, 17); \
  in##_18  = svld1_vnum(ptrue, vec, 18); \
  in##_19  = svld1_vnum(ptrue, vec, 19); \
  in##_20  = svld1_vnum(ptrue, vec, 20); \
  in##_21  = svld1_vnum(ptrue, vec, 21); \
  in##_22  = svld1_vnum(ptrue, vec, 22); \
  in##_23  = svld1_vnum(ptrue, vec, 23); \
  in##_24  = svld1_vnum(ptrue, vec, 24); \
  in##_25  = svld1_vnum(ptrue, vec, 25); \
  in##_26  = svld1_vnum(ptrue, vec, 26); \
  in##_27  = svld1_vnum(ptrue, vec, 27); \
  in##_28  = svld1_vnum(ptrue, vec, 28); \
  in##_29  = svld1_vnum(ptrue, vec, 29); \
  in##_30  = svld1_vnum(ptrue, vec, 30); \
  in##_31  = svld1_vnum(ptrue, vec, 31); 

#define svld1_vnum_16(in, vec) \
  in##_0 = svld1_vnum(ptrue, vec, 0);  \
  in##_1 = svld1_vnum(ptrue, vec, 1);  \
  in##_2 = svld1_vnum(ptrue, vec, 2);  \
  in##_3 = svld1_vnum(ptrue, vec, 3);  \
  in##_4 = svld1_vnum(ptrue, vec, 4);  \
  in##_5 = svld1_vnum(ptrue, vec, 5);  \
  in##_6 = svld1_vnum(ptrue, vec, 6);  \
  in##_7 = svld1_vnum(ptrue, vec, 7);  \
  in##_8 = svld1_vnum(ptrue, vec, 8);  \
  in##_9 = svld1_vnum(ptrue, vec, 9);  \
  in##_10  = svld1_vnum(ptrue, vec, 10); \
  in##_11  = svld1_vnum(ptrue, vec, 11); \
  in##_12  = svld1_vnum(ptrue, vec, 12); \
  in##_13  = svld1_vnum(ptrue, vec, 13); \
  in##_14  = svld1_vnum(ptrue, vec, 14); \
  in##_15  = svld1_vnum(ptrue, vec, 15); 

#define svld1_vnum_15(in, vec) \
  in##_0 = svld1_vnum(ptrue, vec, 0);  \
  in##_1 = svld1_vnum(ptrue, vec, 1);  \
  in##_2 = svld1_vnum(ptrue, vec, 2);  \
  in##_3 = svld1_vnum(ptrue, vec, 3);  \
  in##_4 = svld1_vnum(ptrue, vec, 4);  \
  in##_5 = svld1_vnum(ptrue, vec, 5);  \
  in##_6 = svld1_vnum(ptrue, vec, 6);  \
  in##_7 = svld1_vnum(ptrue, vec, 7);  \
  in##_8 = svld1_vnum(ptrue, vec, 8);  \
  in##_9 = svld1_vnum(ptrue, vec, 9);  \
  in##_10  = svld1_vnum(ptrue, vec, 10); \
  in##_11  = svld1_vnum(ptrue, vec, 11); \
  in##_12  = svld1_vnum(ptrue, vec, 12); \
  in##_13  = svld1_vnum(ptrue, vec, 13); \
  in##_14  = svld1_vnum(ptrue, vec, 14);

#define svld1_vnum_8(in, vec) \
  in##_0 = svld1_vnum(ptrue, vec, 0);  \
  in##_1 = svld1_vnum(ptrue, vec, 1);  \
  in##_2 = svld1_vnum(ptrue, vec, 2);  \
  in##_3 = svld1_vnum(ptrue, vec, 3);  \
  in##_4 = svld1_vnum(ptrue, vec, 4);  \
  in##_5 = svld1_vnum(ptrue, vec, 5);  \
  in##_6 = svld1_vnum(ptrue, vec, 6);  \
  in##_7 = svld1_vnum(ptrue, vec, 7);  

#define svld1_vnum_4(in, vec) \
  in##_0 = svld1_vnum(ptrue, vec, 0);  \
  in##_1 = svld1_vnum(ptrue, vec, 1);  \
  in##_2 = svld1_vnum(ptrue, vec, 2);  \
  in##_3 = svld1_vnum(ptrue, vec, 3);  


#define svmla_z_32(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(ptrue, in1##_7, in2##_7, in3);  \
  out##_8 = svmla_z(ptrue, in1##_8, in2##_8, in3);  \
  out##_9 = svmla_z(ptrue, in1##_9, in2##_9, in3);  \
  out##_10  = svmla_z(ptrue, in1##_10, in2##_10, in3); \
  out##_11  = svmla_z(ptrue, in1##_11, in2##_11, in3); \
  out##_12  = svmla_z(ptrue, in1##_12, in2##_12, in3); \
  out##_13  = svmla_z(ptrue, in1##_13, in2##_13, in3); \
  out##_14  = svmla_z(ptrue, in1##_14, in2##_14, in3); \
  out##_15  = svmla_z(ptrue, in1##_15, in2##_15, in3); \
  out##_16 = svmla_z(ptrue, in1##_16, in2##_16, in3);  \
  out##_17 = svmla_z(ptrue, in1##_17, in2##_17, in3);  \
  out##_18 = svmla_z(ptrue, in1##_18, in2##_18, in3);  \
  out##_19 = svmla_z(ptrue, in1##_19, in2##_19, in3);  \
  out##_20 = svmla_z(ptrue, in1##_20, in2##_20, in3);  \
  out##_21 = svmla_z(ptrue, in1##_21, in2##_21, in3);  \
  out##_22 = svmla_z(ptrue, in1##_22, in2##_22, in3);  \
  out##_23 = svmla_z(ptrue, in1##_23, in2##_23, in3);  \
  out##_24 = svmla_z(ptrue, in1##_24, in2##_24, in3);  \
  out##_25 = svmla_z(ptrue, in1##_25, in2##_25, in3);  \
  out##_26  = svmla_z(ptrue, in1##_26, in2##_26, in3); \
  out##_27  = svmla_z(ptrue, in1##_27, in2##_27, in3); \
  out##_28  = svmla_z(ptrue, in1##_28, in2##_28, in3); \
  out##_29  = svmla_z(ptrue, in1##_29, in2##_29, in3); \
  out##_30  = svmla_z(ptrue, in1##_30, in2##_30, in3); \
  out##_31  = svmla_z(ptrue, in1##_31, in2##_31, in3); 

#define svmla_z_16(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(ptrue, in1##_7, in2##_7, in3);  \
  out##_8 = svmla_z(ptrue, in1##_8, in2##_8, in3);  \
  out##_9 = svmla_z(ptrue, in1##_9, in2##_9, in3);  \
  out##_10  = svmla_z(ptrue, in1##_10, in2##_10, in3); \
  out##_11  = svmla_z(ptrue, in1##_11, in2##_11, in3); \
  out##_12  = svmla_z(ptrue, in1##_12, in2##_12, in3); \
  out##_13  = svmla_z(ptrue, in1##_13, in2##_13, in3); \
  out##_14  = svmla_z(ptrue, in1##_14, in2##_14, in3); \
  out##_15  = svmla_z(ptrue, in1##_15, in2##_15, in3); 

#define svmla_z_15(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(ptrue, in1##_7, in2##_7, in3);  \
  out##_8 = svmla_z(ptrue, in1##_8, in2##_8, in3);  \
  out##_9 = svmla_z(ptrue, in1##_9, in2##_9, in3);  \
  out##_10  = svmla_z(ptrue, in1##_10, in2##_10, in3); \
  out##_11  = svmla_z(ptrue, in1##_11, in2##_11, in3); \
  out##_12  = svmla_z(ptrue, in1##_12, in2##_12, in3); \
  out##_13  = svmla_z(ptrue, in1##_13, in2##_13, in3); \
  out##_14  = svmla_z(ptrue, in1##_14, in2##_14, in3); 

#define svmla_z_8(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(ptrue, in1##_7, in2##_7, in3);  

#define svmla_z_4(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2##_3, in3);  


#define svst1_vnum_32(out, vec) \
  svst1_vnum(ptrue, out, 0, vec##_0);  \
  svst1_vnum(ptrue, out, 1, vec##_1);  \
  svst1_vnum(ptrue, out, 2, vec##_2);  \
  svst1_vnum(ptrue, out, 3, vec##_3);  \
  svst1_vnum(ptrue, out, 4, vec##_4);  \
  svst1_vnum(ptrue, out, 5, vec##_5);  \
  svst1_vnum(ptrue, out, 6, vec##_6);  \
  svst1_vnum(ptrue, out, 7, vec##_7);  \
  svst1_vnum(ptrue, out, 8, vec##_8);  \
  svst1_vnum(ptrue, out, 9, vec##_9);  \
  svst1_vnum(ptrue, out, 10, vec##_10); \
  svst1_vnum(ptrue, out, 11, vec##_11); \
  svst1_vnum(ptrue, out, 12, vec##_12); \
  svst1_vnum(ptrue, out, 13, vec##_13); \
  svst1_vnum(ptrue, out, 14, vec##_14); \
  svst1_vnum(ptrue, out, 15, vec##_15); \
  svst1_vnum(ptrue, out, 16, vec##_16); \
  svst1_vnum(ptrue, out, 17, vec##_17); \
  svst1_vnum(ptrue, out, 18, vec##_18); \
  svst1_vnum(ptrue, out, 19, vec##_19); \
  svst1_vnum(ptrue, out, 20, vec##_20); \
  svst1_vnum(ptrue, out, 21, vec##_21); \
  svst1_vnum(ptrue, out, 22, vec##_22); \
  svst1_vnum(ptrue, out, 23, vec##_23); \
  svst1_vnum(ptrue, out, 24, vec##_24); \
  svst1_vnum(ptrue, out, 25, vec##_25); \
  svst1_vnum(ptrue, out, 26, vec##_26); \
  svst1_vnum(ptrue, out, 27, vec##_27); \
  svst1_vnum(ptrue, out, 28, vec##_28); \
  svst1_vnum(ptrue, out, 29, vec##_29); \
  svst1_vnum(ptrue, out, 30, vec##_30); \
  svst1_vnum(ptrue, out, 31, vec##_31); 


#define svst1_vnum_16(out, vec) \
  svst1_vnum(ptrue, out, 0, vec##_0);  \
  svst1_vnum(ptrue, out, 1, vec##_1);  \
  svst1_vnum(ptrue, out, 2, vec##_2);  \
  svst1_vnum(ptrue, out, 3, vec##_3);  \
  svst1_vnum(ptrue, out, 4, vec##_4);  \
  svst1_vnum(ptrue, out, 5, vec##_5);  \
  svst1_vnum(ptrue, out, 6, vec##_6);  \
  svst1_vnum(ptrue, out, 7, vec##_7);  \
  svst1_vnum(ptrue, out, 8, vec##_8);  \
  svst1_vnum(ptrue, out, 9, vec##_9);  \
  svst1_vnum(ptrue, out, 10, vec##_10); \
  svst1_vnum(ptrue, out, 11, vec##_11); \
  svst1_vnum(ptrue, out, 12, vec##_12); \
  svst1_vnum(ptrue, out, 13, vec##_13); \
  svst1_vnum(ptrue, out, 14, vec##_14); \
  svst1_vnum(ptrue, out, 15, vec##_15); 

#define svst1_vnum_15(out, vec) \
  svst1_vnum(ptrue, out, 0, vec##_0);  \
  svst1_vnum(ptrue, out, 1, vec##_1);  \
  svst1_vnum(ptrue, out, 2, vec##_2);  \
  svst1_vnum(ptrue, out, 3, vec##_3);  \
  svst1_vnum(ptrue, out, 4, vec##_4);  \
  svst1_vnum(ptrue, out, 5, vec##_5);  \
  svst1_vnum(ptrue, out, 6, vec##_6);  \
  svst1_vnum(ptrue, out, 7, vec##_7);  \
  svst1_vnum(ptrue, out, 8, vec##_8);  \
  svst1_vnum(ptrue, out, 9, vec##_9);  \
  svst1_vnum(ptrue, out, 10, vec##_10); \
  svst1_vnum(ptrue, out, 11, vec##_11); \
  svst1_vnum(ptrue, out, 12, vec##_12); \
  svst1_vnum(ptrue, out, 13, vec##_13); \
  svst1_vnum(ptrue, out, 14, vec##_14); 

#define svst1_vnum_8(out, vec) \
  svst1_vnum(ptrue, out, 0, vec##_0);  \
  svst1_vnum(ptrue, out, 1, vec##_1);  \
  svst1_vnum(ptrue, out, 2, vec##_2);  \
  svst1_vnum(ptrue, out, 3, vec##_3);  \
  svst1_vnum(ptrue, out, 4, vec##_4);  \
  svst1_vnum(ptrue, out, 5, vec##_5);  \
  svst1_vnum(ptrue, out, 6, vec##_6);  \
  svst1_vnum(ptrue, out, 7, vec##_7);  

#define svst1_vnum_4(out, vec) \
  svst1_vnum(ptrue, out, 0, vec##_0);  \
  svst1_vnum(ptrue, out, 1, vec##_1);  \
  svst1_vnum(ptrue, out, 2, vec##_2);  \
  svst1_vnum(ptrue, out, 3, vec##_3);  


#define svmla_z_t_8(out, in1, in2, in3) \
  out##_0 = svmla_z(ptrue, in1##_0, in2, in3[0]);  \
  out##_1 = svmla_z(ptrue, in1##_1, in2, in3[1]);  \
  out##_2 = svmla_z(ptrue, in1##_2, in2, in3[2]);  \
  out##_3 = svmla_z(ptrue, in1##_3, in2, in3[3]);  \
  out##_4 = svmla_z(ptrue, in1##_4, in2, in3[4]);  \
  out##_5 = svmla_z(ptrue, in1##_5, in2, in3[5]);  \
  out##_6 = svmla_z(ptrue, in1##_6, in2, in3[6]);  \
  out##_7 = svmla_z(ptrue, in1##_7, in2, in3[7]);  


#define svmul_z_t_8(out, in2, in3) \
  out##_0 = svmul_z(ptrue, in2, in3[0]);  \
  out##_1 = svmul_z(ptrue, in2, in3[1]);  \
  out##_2 = svmul_z(ptrue, in2, in3[2]);  \
  out##_3 = svmul_z(ptrue, in2, in3[3]);  \
  out##_4 = svmul_z(ptrue, in2, in3[4]);  \
  out##_5 = svmul_z(ptrue, in2, in3[5]);  \
  out##_6 = svmul_z(ptrue, in2, in3[6]);  \
  out##_7 = svmul_z(ptrue, in2, in3[7]);  


#define svmul_z_15(out, in2, in3) \
  out##_0 = svmul_z(ptrue, in2##_0, in3##_0);  \
  out##_1 = svmul_z(ptrue, in2##_1, in3##_1);  \
  out##_2 = svmul_z(ptrue, in2##_2, in3##_2);  \
  out##_3 = svmul_z(ptrue, in2##_3, in3##_3);  \
  out##_4 = svmul_z(ptrue, in2##_4, in3##_4);  \
  out##_5 = svmul_z(ptrue, in2##_5, in3##_5);  \
  out##_6 = svmul_z(ptrue, in2##_6, in3##_6);  \
  out##_7 = svmul_z(ptrue, in2##_7, in3##_7);  \
  out##_8 = svmul_z(ptrue, in2##_8, in3##_8);  \
  out##_9 = svmul_z(ptrue, in2##_9, in3##_9);  \
  out##_10 = svmul_z(ptrue, in2##_10, in3##_10);  \
  out##_11 = svmul_z(ptrue, in2##_11, in3##_11);  \
  out##_12 = svmul_z(ptrue, in2##_12, in3##_12);  \
  out##_13 = svmul_z(ptrue, in2##_13, in3##_13);  \
  out##_14 = svmul_z(ptrue, in2##_14, in3##_14);

#define svaddv_15(out, in2) \
  out += svaddv(ptrue, in2##_0);  \
  out += svaddv(ptrue, in2##_1);  \
  out += svaddv(ptrue, in2##_2);  \
  out += svaddv(ptrue, in2##_3);  \
  out += svaddv(ptrue, in2##_4);  \
  out += svaddv(ptrue, in2##_5);  \
  out += svaddv(ptrue, in2##_6);  \
  out += svaddv(ptrue, in2##_7);  \
  out += svaddv(ptrue, in2##_8);  \
  out += svaddv(ptrue, in2##_9);  \
  out += svaddv(ptrue, in2##_10);  \
  out += svaddv(ptrue, in2##_11);  \
  out += svaddv(ptrue, in2##_12);  \
  out += svaddv(ptrue, in2##_13);  \
  out += svaddv(ptrue, in2##_14); 


#define svmla_z_f16_240(out, in1, in2, in3) \
  out##_0 = svmla_z(_ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(_ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(_ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(_ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(_ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(_ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(_ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(_half_ptrue, in1##_7, in2##_7, in3); 

#define svmla_z_f16_8(out, in1, in2, in3) \
  out##_0 = svmla_z(_ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(_ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(_ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(_ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(_ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(_ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(_ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(_ptrue, in1##_7, in2##_7, in3);  


#define svld1_vnum_fp16_16_240(in, vec) \
  in##_0 = svld1_vnum(_ptrue, vec, 0);  \
  in##_1 = svld1_vnum(_ptrue, vec, 1);  \
  in##_2 = svld1_vnum(_ptrue, vec, 2);  \
  in##_3 = svld1_vnum(_ptrue, vec, 3);  \
  in##_4 = svld1_vnum(_ptrue, vec, 4);  \
  in##_5 = svld1_vnum(_ptrue, vec, 5);  \
  in##_6 = svld1_vnum(_ptrue, vec, 6);  \
  in##_7 = svld1_vnum(_half_ptrue, vec, 7);  


#define svst1_vnum_fp16_8_240(out, vec) \
  svst1_vnum(_ptrue, out, 0, vec##_0);  \
  svst1_vnum(_ptrue, out, 1, vec##_1);  \
  svst1_vnum(_ptrue, out, 2, vec##_2);  \
  svst1_vnum(_ptrue, out, 3, vec##_3);  \
  svst1_vnum(_ptrue, out, 4, vec##_4);  \
  svst1_vnum(_ptrue, out, 5, vec##_5);  \
  svst1_vnum(_ptrue, out, 6, vec##_6);  \
  svst1_vnum(_half_ptrue, out, 7, vec##_7);  

#define init_vec_16f16(in) \
    svfloat16_t in##_0;   \
    svfloat16_t in##_1;   \
    svfloat16_t in##_2;   \
    svfloat16_t in##_3;   \
    svfloat16_t in##_4;   \
    svfloat16_t in##_5;   \
    svfloat16_t in##_6;   \
    svfloat16_t in##_7;   \
    svfloat16_t in##_8;   \
    svfloat16_t in##_9;   \
    svfloat16_t in##_10;  \
    svfloat16_t in##_11;  \
    svfloat16_t in##_12;  \
    svfloat16_t in##_13;  \
    svfloat16_t in##_14;  \
    svfloat16_t in##_15;  

#define dup_0_16fp16(in) \
    in##_0 = svdup_f16(0.);   \
    in##_1 = svdup_f16(0.);   \
    in##_2 = svdup_f16(0.);   \
    in##_3 = svdup_f16(0.);   \
    in##_4 = svdup_f16(0.);   \
    in##_5 = svdup_f16(0.);   \
    in##_6 = svdup_f16(0.);   \
    in##_7 = svdup_f16(0.);   \
    in##_8 = svdup_f16(0.);   \
    in##_9 = svdup_f16(0.);   \
    in##_10 = svdup_f16(0.);  \
    in##_11 = svdup_f16(0.);  \
    in##_12 = svdup_f16(0.);  \
    in##_13 = svdup_f16(0.);  \
    in##_14 = svdup_f16(0.);  \
    in##_15 = svdup_f16(0.);  

#define init_vec_8f16(in) \
    svfloat16_t in##_0;   \
    svfloat16_t in##_1;   \
    svfloat16_t in##_2;   \
    svfloat16_t in##_3;   \
    svfloat16_t in##_4;   \
    svfloat16_t in##_5;   \
    svfloat16_t in##_6;   \
    svfloat16_t in##_7;   

#define svld1_vnum_f16_16(in, vec) \
  in##_0 = svld1_vnum(_ptrue, vec, 0);  \
  in##_1 = svld1_vnum(_ptrue, vec, 1);  \
  in##_2 = svld1_vnum(_ptrue, vec, 2);  \
  in##_3 = svld1_vnum(_ptrue, vec, 3);  \
  in##_4 = svld1_vnum(_ptrue, vec, 4);  \
  in##_5 = svld1_vnum(_ptrue, vec, 5);  \
  in##_6 = svld1_vnum(_ptrue, vec, 6);  \
  in##_7 = svld1_vnum(_ptrue, vec, 7);  \
  in##_8 = svld1_vnum(_ptrue, vec, 8);  \
  in##_9 = svld1_vnum(_ptrue, vec, 9);  \
  in##_10  = svld1_vnum(_ptrue, vec, 10); \
  in##_11  = svld1_vnum(_ptrue, vec, 11); \
  in##_12  = svld1_vnum(_ptrue, vec, 12); \
  in##_13  = svld1_vnum(_ptrue, vec, 13); \
  in##_14  = svld1_vnum(_ptrue, vec, 14); \
  in##_15  = svld1_vnum(_ptrue, vec, 15); 

#define svmla_z_f16_16(out, in1, in2, in3) \
  out##_0 = svmla_z(_ptrue, in1##_0, in2##_0, in3);  \
  out##_1 = svmla_z(_ptrue, in1##_1, in2##_1, in3);  \
  out##_2 = svmla_z(_ptrue, in1##_2, in2##_2, in3);  \
  out##_3 = svmla_z(_ptrue, in1##_3, in2##_3, in3);  \
  out##_4 = svmla_z(_ptrue, in1##_4, in2##_4, in3);  \
  out##_5 = svmla_z(_ptrue, in1##_5, in2##_5, in3);  \
  out##_6 = svmla_z(_ptrue, in1##_6, in2##_6, in3);  \
  out##_7 = svmla_z(_ptrue, in1##_7, in2##_7, in3);  \
  out##_8 = svmla_z(_ptrue, in1##_8, in2##_8, in3);  \
  out##_9 = svmla_z(_ptrue, in1##_9, in2##_9, in3);  \
  out##_10  = svmla_z(_ptrue, in1##_10, in2##_10, in3); \
  out##_11  = svmla_z(_ptrue, in1##_11, in2##_11, in3); \
  out##_12  = svmla_z(_ptrue, in1##_12, in2##_12, in3); \
  out##_13  = svmla_z(_ptrue, in1##_13, in2##_13, in3); \
  out##_14  = svmla_z(_ptrue, in1##_14, in2##_14, in3); \
  out##_15  = svmla_z(_ptrue, in1##_15, in2##_15, in3); 

#define svst1_vnum_f16_16(out, vec) \
  svst1_vnum(_ptrue, out, 0, vec##_0);  \
  svst1_vnum(_ptrue, out, 1, vec##_1);  \
  svst1_vnum(_ptrue, out, 2, vec##_2);  \
  svst1_vnum(_ptrue, out, 3, vec##_3);  \
  svst1_vnum(_ptrue, out, 4, vec##_4);  \
  svst1_vnum(_ptrue, out, 5, vec##_5);  \
  svst1_vnum(_ptrue, out, 6, vec##_6);  \
  svst1_vnum(_ptrue, out, 7, vec##_7);  \
  svst1_vnum(_ptrue, out, 8, vec##_8);  \
  svst1_vnum(_ptrue, out, 9, vec##_9);  \
  svst1_vnum(_ptrue, out, 10, vec##_10); \
  svst1_vnum(_ptrue, out, 11, vec##_11); \
  svst1_vnum(_ptrue, out, 12, vec##_12); \
  svst1_vnum(_ptrue, out, 13, vec##_13); \
  svst1_vnum(_ptrue, out, 14, vec##_14); \
  svst1_vnum(_ptrue, out, 15, vec##_15); 


#define svld1_vnum_f16_8(in, vec) \
  in##_0 = svld1_vnum(_ptrue, vec, 0);  \
  in##_1 = svld1_vnum(_ptrue, vec, 1);  \
  in##_2 = svld1_vnum(_ptrue, vec, 2);  \
  in##_3 = svld1_vnum(_ptrue, vec, 3);  \
  in##_4 = svld1_vnum(_ptrue, vec, 4);  \
  in##_5 = svld1_vnum(_ptrue, vec, 5);  \
  in##_6 = svld1_vnum(_ptrue, vec, 6);  \
  in##_7 = svld1_vnum(_ptrue, vec, 7); 

#define svst1_vnum_f16_8(out, vec) \
  svst1_vnum(_ptrue, out, 0, vec##_0);  \
  svst1_vnum(_ptrue, out, 1, vec##_1);  \
  svst1_vnum(_ptrue, out, 2, vec##_2);  \
  svst1_vnum(_ptrue, out, 3, vec##_3);  \
  svst1_vnum(_ptrue, out, 4, vec##_4);  \
  svst1_vnum(_ptrue, out, 5, vec##_5);  \
  svst1_vnum(_ptrue, out, 6, vec##_6);  \
  svst1_vnum(_ptrue, out, 7, vec##_7);  
  
#endif
