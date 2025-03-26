#ifndef DEEPMD_UTIL_H
#define DEEPMD_UTIL_H


#include "pointers.h"
#include <iostream>
#include <fstream>
#include <assert.h>
// #include "timer.h"
#include "deepmd_common.h"


#ifdef WITH_TENSOR_FLOW
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#endif

// #define SPLIT_TYPE_EMBEDDING

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif
namespace LAMMPS_NS {

#define last_layer_size 128
#define dim_descrpt 2048
#define fitting_layer_num 3
#define n_axis_neuron 16
#define FITTING_LAYER 4

struct PB_param_type1
{
  double c_matrix_0[2048*240];
  double c_matrix_1[240*240];
  double c_matrix_2[240*240];
  double c_matrix_3[240*1];
  double c_bias_0[240];
  double c_bias_1[240];
  double c_bias_2[240];
  double c_bias_3[1];
  double c_idt_0[240];
  double c_idt_1[240];
  double c_idt_2[240];
  double c_idt_3[240];
  double c_table[1360*768];
  double c_table_info[6];
  double std_ones[2048];
  double avg_zero[2048];
// } ;
} __attribute__ ((aligned(256)));

struct PB_param_type4
{
  float c_matrix_0[2048*240];
  float c_matrix_1[240*240];
  float c_matrix_2[240*240];
  float c_matrix_3[240*1];
  float c_bias_0[240];
  float c_bias_1[240];
  float c_bias_2[240];
  float c_bias_3[1];
  float c_idt_0[240];
  float c_idt_1[240];
  float c_idt_2[240];
  float c_idt_3[240];
  float c_table[136000*256];
  float c_table_info[6];
  float std_ones[2048];
  float avg_zero[2048];
// } ;
} __attribute__ ((aligned(256)));

struct PB_param_type2
{
  double c_matrix_0[2][2048*240];
  double c_matrix_1[2][240*240];
  double c_matrix_2[2][240*240];
  double c_matrix_3[2][240*1];
  double c_bias_0[2][240];
  double c_bias_1[2][240];
  double c_bias_2[2][240];
  double c_bias_3[2][1];
  double c_idt_0[2][240];
  double c_idt_1[2][240];
  double c_idt_2[2][240];
  double c_idt_3[2][240];
  double c_table[2*2][1360*768];
  double c_table_info[6];
  double std_ones[2*552];
  double avg_zero[2*552];
// } ;
} __attribute__ ((aligned(256)));

struct PB_param_type3
{
  float c_matrix_0[2][2048*240];
  float c_matrix_1[2][240*240];
  float c_matrix_2[2][240*240];
  float c_matrix_3[2][240*1];
  float c_bias_0[2][240];
  float c_bias_1[2][240];
  float c_bias_2[2][240];
  float c_bias_3[2][1];
  float c_idt_0[2][240];
  float c_idt_1[2][240];
  float c_idt_2[2][240];
  float c_idt_3[2][240];
  float c_table[2*2][136000*256];
  float c_table_info[6];
  float std_ones[2*552];
  float avg_zero[2*552];
// } ;
} __attribute__ ((aligned(256)));

struct NeighborInfo 
{
  int type;
  FPTYPE dist;
  int index;
  NeighborInfo () 
      : type (0), dist(0), index(0) 
      {
      }
  NeighborInfo (int tt, FPTYPE dd, int ii) 
      : type (tt), dist(dd), index(ii) 
      {
      }
  bool operator < (const NeighborInfo & b) const 
      {
	return (type < b.type || 
		(type == b.type && 
		 (dist < b.dist || 
		  (dist == b.dist && index < b.index) ) ) );
      }
};

struct InputNlist {
  int inum;
  int * ilist;
  int * numneigh;
  int ** firstneigh;
  InputNlist () 
      : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL)
      {};
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {};
  ~InputNlist(){};
};


struct AtomManage {

};

// template <typename FPTYPE>
class AtomMap 
{
public:
  AtomMap();
  void reserve(int _max_atoms);

  void init(int* in_begin, int natoms);

  void init(int* in_begin, int natoms, int real_types);

  // void forward (typename std::vector<FPTYPE >::iterator out,
	// 	const typename std::vector<FPTYPE >::const_iterator in, 
	// 	const int stride = 1) const ;

  template<typename VT>
  void forward (VT* out, const VT* in,  const int stride = 1) const ;

  // void backward (typename std::vector<FPTYPE >::iterator out,
	// 	 const typename std::vector<FPTYPE >::const_iterator in, 
	// 	 const int stride = 1) const ;

  template<typename VT>
  void backward (VT* out, const VT* in, const int stride = 1) const ;

  const int* get_type () const {return atype;}
  const int* get_fwd_map () const {return fwd_idx_map;}  
  const int* get_bkw_map () const {return idx_map;}

  int nloc;
  int real_types;
private:
  int* idx_map;
  int* fwd_idx_map;
  int* atype;
  std::vector<std::pair<int, int > > sorting;
};

// template class AtomMap<float>;
// template class AtomMap<double>;

struct NeighborListData 
{
  int* ilist;
  int** jlist;
  int* numneigh;
  int** firstneigh;  
public:
  int nloc;
  int inum;
  void reserve(int _max_atoms, int _nall, int _nnei);
  void copy_from_nlist(const InputNlist & inlist);
  void shuffle(const int* fwd_map, int natoms);
  void shuffle(const AtomMap & map);
  void make_inlist(InputNlist & inlist);
};

// typedef struct Session_Buf_Struct {
//   FPTYPE** descrptor;
//   FPTYPE** rg_silce;
//   FPTYPE** rg_fusion;
//   FPTYPE** qmat, **qmat_grad;

//   FPTYPE** s_vector_grad;
//   FPTYPE** r_matrix_grid;
//   FPTYPE** r_matrix_grid_3d[3];

//   FPTYPE *descrptor_grad;
//   FPTYPE *layer_0, *layer_1, *layer_2, *layer_final, *layer_final_qmat;
//   FPTYPE *layer_0_tanh, *layer_1_tanh, *layer_2_tanh;
//   FPTYPE *layer_0_grad, *layer_1_grad, *layer_2_grad;
//   FPTYPE *layer_1_grad_reg, *layer_2_grad_reg, *layer_final_grad;

//   __fp16 *gemm_fp16_buf;

//   FPTYPE *rg_fusion_grad, *rg_slice_grad;
//   FPTYPE *buf;
// } Session_Buf;

class DeepPot: public Pointers {
public:
  DeepPot (class LAMMPS *) ;
  // ~DeepPot() ;

public:
  void init(FPTYPE _rcut, 
              FPTYPE _rcut_smth, 
              FPTYPE _ntypes,
              std::vector<int>& _sel,
              std::vector<FPTYPE> &	_box,
              std::string graph_path,
              int _MODEL_TYPE);

  void init(DeepPot *_deep_pot, int _tid);

  void init_value();
  void reserve_buffer(int _max_atoms, int _nall);

  void splite_atom();

  // void swith_model(int _current_model);

  void shuffer_dextf(int *bd_idx, FPTYPE *delef_);

  void load_data_from_dat(std::string graph_path);

  void store_pb_data();
  void table_convert(FPTYPE** &_in_table, int _ntypes);
  
  void compute (double &	ener,
		double* &	force,
		double* &	virial,
		FPTYPE* &	coord,
		int*&	atype,
		const int			nghost_,
		const int			nloc_,
		const InputNlist &		inlist,
		const int&			ago);

  void compute (double *ener,
		double*	force,
		double*	virial);

  void session_run ();

  void prod_env_mat_a();

  void fitting_net_dipole(int type_i);
  void fitting_net_normal(int type_i);
  void embedding_net(int type_i);
  void prod_R_matrix(int type_i);
  void prod_atom_nlist();

  void tabulateFusion(int _loc, int _nnei,
                      FPTYPE* &em_x,
                      FPTYPE* &em,
                      FPTYPE *out,
                      const FPTYPE* _table);

                      
  void tabulateFusion_sve(int _loc, int _nnei,
                      FPTYPE* &em_x,
                      FPTYPE* &em,
                      FPTYPE *out,
                      const FPTYPE* _table);

  void tabulate_fusion_grad_cpu_packing(int _nloc, int _nnei,
                      FPTYPE *dy_dem_x, 
                      FPTYPE *dy_dem,
                      const FPTYPE * _table, 
                      FPTYPE *em_x, 
                      FPTYPE *em, 
                      FPTYPE *dy) ;
  void tabulate_fusion_grad_cpu_packing_sve(int _nloc, int _nnei,
                      FPTYPE *dy_dem_x, 
                      FPTYPE *dy_dem,
                      const FPTYPE * _table, 
                      FPTYPE *em_x, 
                      FPTYPE *em, 
                      FPTYPE *dy) ;

  void tabulateFusion_v1_sve(int _loc, int _nnei,
                      FPTYPE* &em_x,
                      FPTYPE* &em,
                      FPTYPE *out,
                      const FPTYPE* _table);
  void tabulate_fusion_grad_cpu_packing_v1_sve(int _nloc, int _nnei,
                      FPTYPE *dy_dem_x, 
                      FPTYPE *dy_dem,
                      const FPTYPE * _table, 
                      FPTYPE *em_x, 
                      FPTYPE *em, 
                      FPTYPE *dy) ;

  #ifdef WITH_TENSOR_FLOW
  void load_data_and_bcast(std::string graph_path);
  void load_tensorflow_model(int _in_type, std::string prefix, 
    FPTYPE  *&c_table_info_in, FPTYPE **&c_table_in,
    FPTYPE  ***&c_matrix_in, FPTYPE ***&c_bias_in, FPTYPE ***&c_idt_in, FPTYPE  ***&c_matrix_t_in,
    float16_t  ***&c_matrix_fp16_in,  float16_t ***&c_matrix_t_fp16_in,
    FPTYPE *&avg_zero_in, FPTYPE *&std_ones_in,
    FPTYPE** &grad_f_data_in);


  FPTYPE* get_node_attr(std::string node_name);
  FPTYPE* print_all_node_attr();
  void get_vector_from_model(std::vector<int> & vec, const std::string node_name);

  std::vector<tensorflow::Tensor> tensors;
  #endif
 
  void prod_force_a_cpu(
      const FPTYPE * net_deriv, 
      const int ifrom, 
      const int ito, 
      const int g_ifrom,
      const int g_ito) ;
  void prod_force_a_cpu(
      const FPTYPE * net_deriv, 
      const int type_i,
      const int type_i_in );

  void prod_force_a_cpu(
    const FPTYPE * net_deriv,
    const FPTYPE * env_deriv
    ) ;

  void prod_virial_a_cpu(
    const FPTYPE * net_deriv, 
    const int ifrom, 
    const int ito,
    const int g_ifrom,
    const int g_ito);

  void prod_virial_a_cpu(
    const FPTYPE * net_deriv, 
      const int type_i,
      const int type_i_in );

  void prod_virial_a_cpu(
    const FPTYPE * net_deriv,
    const FPTYPE * env_deriv
    );

  double memory_usage() {return 0.0;}; 

  // FPTYPE cutoff () const {assert(inited); return rcut;};
  // int numb_types () const {assert(inited); return ntypes;};
  // void get_type_map (std::string & type_map);

public:
  std::vector<int> sel, sel_a, sel_r;
  int nnei_a, nnei_r, nnei, nem;
  FPTYPE rcut, rcut_smth;
  int ndescrpt_a, ndescrpt_r, ndescrpt;
  int ntypes;

#ifdef WITH_TENSOR_FLOW
  tensorflow::GraphDef graph_def;
#endif
  
  FPTYPE  *c_table_info, **c_table;
  FPTYPE  ***c_matrix, ***c_bias, ***c_idt,  ***c_matrix_t;
  float16_t  ***c_matrix_fp16,  ***c_matrix_t_fp16;
  FPTYPE* avg_zero, *std_ones;
  FPTYPE** grad_f_data;

  int MODEL_TYPE;

  std::vector<int> dipole_sel_type;
  
  std::vector<int> n_neuron;
  std::vector<FPTYPE> box;
  std::vector<int> sec_a;

  int  dipole_flag;
  // int current_model;

  class Timer *t_timer;

  // float *table_fitting;

  int max_nnei;
  int max_nlist;
  int max_nall;
  int max_nloc;
  int max_atoms;

  int*  thread_neigh;
  int*  thread_local_ilist;
  int*  thread_local_numneigh;
  int** thread_firstneigh;
  int*  forward_index_map;
  int*  backward_index_map;
  int   backward_index_size;
  InputNlist lmp_list;

private:
  int num_intra_nthreads, num_inter_nthreads;
  bool inited;
  template<class VT> VT get_scalar(const std::string & name) const;

  int* nlist;

  FPTYPE cell_size;
  std::string model_type;

  int nghost;
  int nall;
  int nloc;

  int tid;

  // PB_param_type1 pb_param_type1;

  std::vector<int> type_natoms, sec_type_atom;

  // copy neighbor list info from host
  bool init_nbor;
  NeighborListData nlist_data;
  InputNlist in_nlist;
  AtomMap atommap;

  int max_nbor_size;

  int **d_nlist_a;
  int *d_nlist_size;

  NeighborInfo *sel_nei;
  int sel_nei_size;

  int *nei_num_v;

  std::string mesg;

  int* datype;
  FPTYPE* dcoord;
  double* dforce;
  double* dvirial;
  FPTYPE* dextf;
  
  FPTYPE*    ori_dipole;
  int*    ori_datype;
  FPTYPE* ori_dcoord;
  double* ori_dforce;
  double 	dener;

  FPTYPE** rij, **r_matrix, **r_matrix_deriv;
  FPTYPE** s_vector;

  FPTYPE** descrptor;
  FPTYPE** rg_silce;
  FPTYPE** rg_fusion;
  FPTYPE** qmat, **qmat_grad;

  FPTYPE** s_vector_grad;
  FPTYPE** r_matrix_grid;
  FPTYPE** r_matrix_grid_3d[3];

  FPTYPE *descrptor_grad;
  FPTYPE *layer_0, *layer_1, *layer_2, *layer_final, *layer_final_qmat;
  FPTYPE *layer_0_tanh, *layer_1_tanh, *layer_2_tanh;
  FPTYPE *layer_0_grad, *layer_1_grad, *layer_2_grad;
  FPTYPE *layer_1_grad_reg, *layer_2_grad_reg, *layer_final_grad;

  __fp16 *gemm_fp16_buf;

  FPTYPE *rg_fusion_grad, *rg_slice_grad;

  FPTYPE *grad_one_matrix;

  // Session_Buf *sess_bufs, *this_sess;
  // void reserve_sessBuf(Session_Buf &_sess_buf, int _max_nloc, int *_n_neuron, int _ntypes, int _max_nnei);
};



}

#endif