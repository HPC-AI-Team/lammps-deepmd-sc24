#ifdef PAIR_CLASS

PairStyle(deepmd/omp,PairDeepMD)

#else

#ifndef LMP_PAIR_DEEPMD_H
#define LMP_PAIR_DEEPMD_H

#include "pair.h"
#include <iostream>
#include <fstream>
#include "deepmd_util.h"
#include "thr_omp.h"

#define GIT_SUMM @GIT_SUMM@
#define GIT_HASH @GIT_HASH@
#define GIT_BRANCH @GIT_BRANCH@
#define GIT_DATE @GIT_DATE@
#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif
#define DEEPMD_ROOT @CMAKE_INSTALL_PREFIX@
#define TensorFlow_INCLUDE_DIRS @TensorFlow_INCLUDE_DIRS@
#define TensorFlow_LIBRARY @TensorFlow_LIBRARY@
#define DPMD_CVT_STR(x) #x
#define DPMD_CVT_ASSTR(X) DPMD_CVT_STR(X)
#define STR_GIT_SUMM DPMD_CVT_ASSTR(GIT_SUMM)
#define STR_GIT_HASH DPMD_CVT_ASSTR(GIT_HASH)
#define STR_GIT_BRANCH DPMD_CVT_ASSTR(GIT_BRANCH)
#define STR_GIT_DATE DPMD_CVT_ASSTR(GIT_DATE)
#define STR_FLOAT_PREC DPMD_CVT_ASSTR(FLOAT_PREC)
#define STR_DEEPMD_ROOT DPMD_CVT_ASSTR(DEEPMD_ROOT)
#define STR_TensorFlow_INCLUDE_DIRS DPMD_CVT_ASSTR(TensorFlow_INCLUDE_DIRS)
#define STR_TensorFlow_LIBRARY DPMD_CVT_ASSTR(TensorFlow_LIBRARY)


namespace LAMMPS_NS {


class PairDeepMD : public Pair, public ThrOMP {
 public:
  PairDeepMD(class LAMMPS *);
  virtual ~PairDeepMD();
  virtual void compute(int, int);
  // virtual void compute(int, int, int);
  virtual void *extract(const char *, int &);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  double init_one(int i, int j);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  void print_summary(const std::string pre) const;
  int get_node_rank();
  void create_dcoord(int nall, int tid);
  // std::string get_file_content(const std::string & model);
  // std::vector<std::string> get_file_content(const std::vector<std::string> & models);

 protected:  
  virtual void allocate();
  double **scale;

private:  
  int first_time[T_THREAD] = {0};
  DeepPot *deep_pot;
  DeepPot **deep_pots;
  int num_threads = 1;
  // std::vector<std::vector<int>> backward_index_maps;
  int ** backward_index_maps;
  int * backward_index_size;
  std::vector<InputNlist> lmp_lists;  

  int**  thread_neigh;
  int**  thread_local_ilist;
  int**  thread_local_numneigh;
  int*** thread_firstneigh;
  int **forward_index_map;


  std::string graph_path;

  double cutoff;
  int numb_types;
  std::vector<std::vector<double > > all_force;
  int out_freq;
  std::string out_file;
  int out_each;
  int out_rel;
  int out_rel_v;
  bool is_restart;

  int *test_buffer[T_THREAD];

  int max_nall;
  int max_nloc;
  int max_nnei;

  double* dvirial;
  double** thread_dvirial;
  double* thread_dener;
  FPTYPE** thread_dcoord;
  double** thread_dforce;
  int **thread_dtype;
  double* dcoord;

  FPTYPE eps;
  FPTYPE eps_v;
  FPTYPE rcut, rcut_smth;
  std::vector<int> sel;
  int nnei;
};

}

#endif
#endif
