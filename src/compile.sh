# module sw lang/tcsds-1.2.38

(make -j48 omp_float 2>&1) | tee compile.log
mv lmp_omp_float lmp_execute_float
cp lmp_execute_float BIN_coul_long
cp lmp_execute_float BIN_coul_long/step_by_step
# cp lmp_execute_float BIN_coul_long/accuracy
# cp lmp_omp_float BIN_test_fft/lmp_execute_tf

(make -j48 omp_cblas 2>&1) | tee compile.log
# mv lmp_omp_cblas lmp_execute_cblas
# cp lmp_execute_cblas BIN_coul_long
# cp lmp_execute_cblas BIN_coul_long/step_by_step
# cp lmp_execute_cblas BIN_coul_long/accuracy
# cp lmp_execute_cblas BIN_test_fft


(make -j48 omp_double 2>&1) | tee compile.log
mv lmp_omp_double lmp_execute_double
cp lmp_execute_double BIN_coul_long
cp lmp_execute_double BIN_coul_long/step_by_step
# cp lmp_execute_double BIN_coul_long/accuracy
# # cp lmp_execute_tf BIN_test_fft

# (make -j48 omp_tf 2>&1) | tee compile.log
# mv lmp_omp_tf lmp_execute_tf
# cp lmp_execute_tf BIN_coul_long
# cp lmp_execute_tf BIN_coul_long/step_by_step
# cp lmp_execute_tf BIN_coul_long/accuracy
# # cp lmp_execute_tf BIN_test_fft
