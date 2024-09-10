module sw lang/tcsds-1.2.38

(make -j48 omp_double 2>&1) | tee compile.log

mv lmp_omp_double lmp_execute_double
cp lmp_execute_double BIN_deepmd_water
cp lmp_execute_double BIN_deepmd_copper

(make -j48 omp_cblas 2>&1) | tee compile.log

mv lmp_omp_cblas lmp_execute_cblas
cp lmp_execute_cblas BIN_deepmd_water
cp lmp_execute_cblas BIN_deepmd_copper

(make -j48 omp_float 2>&1) | tee compile.log

mv lmp_omp_float lmp_execute_float
cp lmp_execute_float BIN_deepmd_water
cp lmp_execute_float BIN_deepmd_copper

