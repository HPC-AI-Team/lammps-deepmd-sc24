#include "pointers.h"
#include "deepmd_util.h"
#include "stdlib.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdexcept>
#include "timer.h"
#include "update.h"
#include "output.h"
#include "utils.h"
#include "error.h"
#include "comm.h"
#include <math.h>
#include "memory.h"
#include "atom.h"
#include "neighbor.h"
#include <sstream>
#include <iomanip>
#include "neigh_list.h"
#ifdef __cplusplus
extern "C"
{
#endif
   #include <cblas.h>
#ifdef __cplusplus
}
#endif

using namespace LAMMPS_NS;

inline void task_division(int nthreads, int nall, int tid, int &ifrom, int &ito) {
  int idelta_i = nall / nthreads;
  int idelta_j = nall % nthreads;
  int _bias    = idelta_j == 0 ? 0 : 1;
  
  if(tid >= idelta_j) {
    ifrom = (idelta_i + _bias) * idelta_j + idelta_i * (tid - idelta_j);
    ito = ifrom + idelta_i; 
  } else {
    ifrom = (idelta_i + _bias) * (tid);
    ito = ifrom + idelta_i + _bias; 
  }
  ito = (ito > nall) ? nall : ito; 
 }

inline void locate_xx(
    const FPTYPE& lower, 
    const FPTYPE& upper,
    const FPTYPE& max, 
    const FPTYPE& stride0, 
    const FPTYPE& stride1, 
    FPTYPE& xx, 
    int& table_idx)  {
  if (xx < lower) {
    table_idx = 0;
    xx = 0;
  }
  else if (xx < upper) {
    table_idx = (int)((xx - lower) / stride0);
    xx -= (table_idx * stride0 + lower);
  }
  else if (xx < max) {
    int first_stride = int((upper - lower) / stride0);
    table_idx = first_stride + (int)((xx - upper) / stride1);
    xx -= ((table_idx - first_stride) * stride1 + upper);
  }
  else {
    table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
    xx = 0;
  }
}


inline void spline5_switch (
    FPTYPE & vv,
    FPTYPE & dd,
    const FPTYPE & xx, 
    const float & rmin, 
    const float & rmax) {
  if (xx < rmin) {
    vv = 1;
    dd = 0;
  }
  else if (xx < rmax) {
    FPTYPE uu = (xx - rmin) / (rmax - rmin) ;
    FPTYPE du = 1. / (rmax - rmin) ;
    vv = uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
    dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
  }
  else {
    dd = 0;
    vv = 0;
  }
}

AtomMap::
AtomMap() {}

void AtomMap::
reserve(int _max_nloc) {
  idx_map = new int[_max_nloc];          memset(idx_map, 0, _max_nloc * sizeof(int));
  fwd_idx_map = new int[_max_nloc];      memset(fwd_idx_map, 0, _max_nloc * sizeof(int));
  atype = new int[_max_nloc];            memset(atype, 0, _max_nloc * sizeof(int));

  sorting.resize(_max_nloc);
  nloc = 0;
}


void AtomMap::init(int* in_begin, 
     int natoms)
{
  nloc = natoms;
  int *iter = in_begin;
  for (unsigned ii = 0; ii < natoms; ++ii) {
    sorting[ii] = std::pair<int, int > (*(iter++), ii);
  }
  // 根据type 排序
  sort (sorting.begin(), sorting.begin()+natoms);

  memset(idx_map,     0, natoms * sizeof(int));
  memset(fwd_idx_map, 0, natoms * sizeof(int));
  for (unsigned ii = 0; ii < natoms; ++ii) {
    // 排序后的id
    idx_map[ii] = sorting[ii].second;
    // 排序后对应之前的id
    fwd_idx_map[sorting[ii].second] = ii;
    // 排序后的类型
    atype[ii] = sorting[ii].first;
  }
}

template<typename VT>
void AtomMap::forward (VT* out,
	 const VT* in, 
	 const int stride) const  {
  int natoms = nloc;
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      *(out + ii*stride + dd) = *(in + gro_i*stride + dd);
    }
  }
}

template<typename VT>
void AtomMap::backward (VT* out,
	  const VT* in, 
	  const int stride) const {
  int natoms = nloc;
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      *(out + gro_i*stride + dd) = *(in + ii*stride + dd);
    }
  }
}

void NeighborListData::reserve(int _max_nloc, int _max_nall, int _max_nlist)
{
  ilist = new int[_max_nloc];  memset(ilist, 0, sizeof(int) * _max_nloc);
  numneigh = new int[_max_nloc];  memset(numneigh, 0, sizeof(int) * _max_nloc);
  firstneigh = new int*[_max_nloc];    memset(firstneigh, 0, sizeof(int*) * _max_nloc);

  jlist = new int*[_max_nloc];
  for(int i = 0; i < _max_nloc; i++) {
    jlist[i] = new int[_max_nlist]; memset(jlist[i], 0, sizeof(int) * _max_nlist);
    firstneigh[i] = jlist[i];
  }
}

void NeighborListData::copy_from_nlist(const InputNlist & inlist)
{
  inum = nloc = inlist.inum;  

  memcpy(ilist, inlist.ilist, inum*sizeof(int));
  for(int ii = 0; ii < inum; ++ii){
    int jnum = inlist.numneigh[ii];
    numneigh[ii] = jnum;
    memcpy(jlist[ii], inlist.firstneigh[ii], jnum*sizeof(int));
  }
}

void NeighborListData::shuffle(const AtomMap & map)
{
  const int* fwd_map = map.get_fwd_map();
  shuffle(fwd_map, map.nloc);
}

// 把ilist和jlist变为新的位置
void NeighborListData::shuffle(const int* fwd_map, int _nloc)
{
  for(unsigned ii = 0; ii < inum; ++ii) {
    if(ilist[ii] < _nloc){
      ilist[ii] = fwd_map[ilist[ii]];
    }
  }
  for(unsigned ii = 0; ii < inum; ++ii) {
    for(unsigned jj = 0; jj < numneigh[ii]; ++jj){
      if(jlist[ii][jj] < _nloc) {
	      jlist[ii][jj] = fwd_map[jlist[ii][jj]];
      }
    }
  }
}

void NeighborListData::make_inlist(InputNlist & inlist) {
  inlist.inum = inum;
  inlist.ilist = &ilist[0];
  inlist.numneigh = &numneigh[0];
  inlist.firstneigh = &firstneigh[0];
}


template<typename VT>
inline void select_map(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & idx_map, 
	   const int & stride)
{
  for (int ii = 0; ii < in.size() / stride; ++ii) {
    if (idx_map[ii] >= 0) {
      int to_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	      out[to_ii * stride + dd] = in[ii * stride + dd];
      }
    }
  }
}

inline void select_real_atoms(std::vector<int> & fwd_map,
		  std::vector<int> & bkw_map,
		  int & nghost_real,
		  const std::vector<FPTYPE> & dcoord_, 
		  const std::vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes)
{
  std::vector<int > sel_type;
  for (int ii = 0; ii < ntypes; ++ii){
    sel_type.push_back(ii);
  }
  // select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);

  std::vector<int> sel_type_ (sel_type);
  sort(sel_type_.begin(), sel_type_.end());  
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  int nloc_real = 0;
  nghost_real = 0;
  fwd_map.resize(nall);
  bkw_map.clear();
  bkw_map.reserve(nall);  
  int cc = 0;
  for (int ii = 0; ii < nall; ++ii){
    // exclude virtual sites
    // select the type with id < ntypes
    if (binary_search(sel_type_.begin(), sel_type_.end(), datype_[ii])){
      bkw_map.push_back(ii);
      if (ii < nloc) {
	      nloc_real += 1;
      }
      else {
	      nghost_real += 1;
      }
      fwd_map[ii] = cc;
      cc ++;
    }
    else {
      fwd_map[ii] = -1;
    }
  }
  assert((nloc_real+nghost_real) == bkw_map.size());  
}


DeepPot::DeepPot (LAMMPS *lmp) : Pointers(lmp){
  t_timer = new Timer(lmp);
  t_timer->init();

  c_matrix = new FPTYPE**[FITTING_LAYER];
  c_bias = new FPTYPE**[FITTING_LAYER];
  c_idt = new FPTYPE**[FITTING_LAYER];
  c_matrix_t = new FPTYPE**[FITTING_LAYER];
  c_matrix_fp16 = new float16_t**[FITTING_LAYER];
  c_matrix_t_fp16 = new float16_t**[FITTING_LAYER];

  c_matrix_pair = new FPTYPE**[FITTING_LAYER];
  c_bias_pair = new FPTYPE**[FITTING_LAYER];
  c_idt_pair = new FPTYPE**[FITTING_LAYER];
  c_matrix_t_pair = new FPTYPE**[FITTING_LAYER];
  c_matrix_fp16_pair = new float16_t**[FITTING_LAYER];
  c_matrix_t_fp16_pair = new float16_t**[FITTING_LAYER];

  c_matrix_dipole = new FPTYPE**[FITTING_LAYER];
  c_bias_dipole = new FPTYPE**[FITTING_LAYER];
  c_idt_dipole = new FPTYPE**[FITTING_LAYER];
  c_matrix_t_dipole = new FPTYPE**[FITTING_LAYER];
  c_matrix_fp16_dipole = new float16_t**[FITTING_LAYER];
  c_matrix_t_fp16_dipole = new float16_t**[FITTING_LAYER];
}

void DeepPot::splite_atom(int _current_model) {
  int global_nlocal = atom->nlocal;
  int global_nghost = atom->nghost;
  int global_nall = global_nlocal + global_nghost;
  int nthreads = comm->nthreads;
  int ago = neighbor->ago;
  int ifrom, ito;
  NeighList *list = neighbor->lists[0];

  swith_model(_current_model);

  task_division(nthreads, global_nlocal, tid, ifrom, ito);
  ifrom = 0;
  if(tid == 0) ito = atom->nlocal;
  else ito = 0;

  if (DEBUG_MSG) utils::logmesg(lmp, "splite_atom tid {} ifrom {} ito {} nlocal  {} nghost {} ago {}\n", 
    tid,  ifrom, ito, atom->nlocal, atom->nghost, ago);

  if(ago == 0) {
    backward_index_size = 0;
    for(int global_i_index = ifrom; global_i_index < ito; global_i_index++) {
      backward_index_map[backward_index_size++] = global_i_index;
    }
    int local_nloc = ago == 0 ? backward_index_size : lmp_list.inum;
    int local_nall = local_nloc;

    if (DEBUG_MSG) utils::logmesg(lmp, "splite_atom tid {} local_nloc {}\n", tid, local_nloc);

    if(lmp_list.inum != 0) {
      memset(thread_neigh, 0,          max_nlist * local_nloc * sizeof(int));
      memset(thread_local_ilist, 0,    local_nloc * sizeof(int));
      memset(thread_local_numneigh, 0, local_nloc * sizeof(int));
      memset(thread_firstneigh, 0,     local_nloc * sizeof(int*));
    }

    for(int i = 0; i < global_nall; i++) {
      forward_index_map[i] = -1;
    }

    int total_neigh=0;
    for(int local_i_index = 0; local_i_index < local_nloc; local_i_index++) {
      int global_i_index = backward_index_map[local_i_index];
      thread_local_ilist[local_i_index] = local_i_index;
      thread_local_numneigh[local_i_index] = list->numneigh[global_i_index];
      total_neigh += thread_local_numneigh[local_i_index];
      forward_index_map[global_i_index] = local_i_index;
    }

    if(DEBUG_MSG) utils::logmesg(lmp, "splite_atom tid {} total_neigh {} \n",  tid,  total_neigh);

    int cur_neigh = 0;
    for(int local_i_index = 0; local_i_index < local_nloc;local_i_index++) {
      int global_i_index = backward_index_map[local_i_index];
      thread_firstneigh[local_i_index] = &thread_neigh[cur_neigh];

      if(max_nlist < thread_local_numneigh[local_i_index]) error->one(FLERR, "[ERROR] max_nlist < thread_local_numneigh[local_i_index] {} {} ", max_nlist, thread_local_numneigh[local_i_index]);

      for(int nei_iter = 0; nei_iter < thread_local_numneigh[local_i_index]; nei_iter++ ) {
        int global_j_idx = list->firstneigh[global_i_index][nei_iter];
        int local_j_index = forward_index_map[global_j_idx];
        if(local_j_index == -1){
          local_j_index = local_nall++;
          forward_index_map[global_j_idx] = local_j_index;
          backward_index_map[backward_index_size++] = global_j_idx;
        }
        thread_firstneigh[local_i_index][nei_iter] = local_j_index;
      }
      cur_neigh += thread_local_numneigh[local_i_index];
    }

    if(max_nlist * max_nloc <= cur_neigh)   error->one(FLERR, "[ERROR] max_nlist*max_nloc < cur_neigh   {} {} ", max_nlist*max_nloc, cur_neigh);

    lmp_list.inum = local_nloc;
    lmp_list.ilist = thread_local_ilist;
    lmp_list.numneigh = thread_local_numneigh;
    lmp_list.firstneigh = thread_firstneigh;
    assert(cur_neigh == total_neigh);

    for(int local_index = 0;local_index < local_nall;local_index++) {
      int global_index = backward_index_map[local_index];
      ori_datype[local_index] = atom->type[global_index] - 1;
    }

    nloc = local_nloc;
    nall = local_nall;
    nghost = local_nall - local_nloc;

    if(max_nloc < nloc) error->one(FLERR, "[ERRIR] max_nloc < nloc max_nloc {} nloc {}", max_nloc, nloc);
    if(max_nall < nall) error->one(FLERR, "[ERRIR] max_nall < nall max_nloc {} nall {}", max_nall, nall);

    nlist_data.copy_from_nlist(lmp_list);

    // 初始化 atommap，对local atom的type进行排序
    atommap.init(ori_datype, nloc);

    // 更新列表，将ilst和jlist更新为排序后的顺序
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(in_nlist);

    max_nbor_size = 0;

    for(int ii = 0; ii < in_nlist.inum; ++ii){
      if(in_nlist.numneigh[ii] > max_nbor_size) max_nbor_size = in_nlist.numneigh[ii];
    }

    assert(max_nbor_size < max_nlist);

    memcpy(datype, atommap.get_type(), nloc * sizeof(int));    
    memcpy(datype + nloc, ori_datype + nloc, nghost * sizeof(int));

    // local atom每种类型原子数量
    for(int ii = 0; ii < ntypes; ii++){
      type_natoms[ii] = 0;
    }
    for (unsigned ii = 0; ii < nloc; ++ii) {
      type_natoms[datype[ii]] ++;
    }

    cum_sum(sec_type_atom, type_natoms);

    // if(DEBUG_MSG) print_v(ntypes, "[INFO] type atoms  "   , type_natoms);
    if(DEBUG_MSG) if(tid == 0) print_v(ntypes, "[INFO] type atoms  "   , type_natoms);
    if(DEBUG_MSG) if(tid == 0) print_v(ntypes+1, "[INFO] sec_type_atom  ", sec_type_atom);

    if(DEBUG_MSG) if(tid == 0) print_v(ntypes+1, "[INFO] sec_a  ", sec_a);

    // if(DEBUG_MSG) print_v(nloc, "[INFO] ilist  ", in_nlist.ilist);
    // if(DEBUG_MSG) print_v(nloc, "[INFO] numneigh  ", in_nlist.numneigh);

    for (unsigned ii = 0; ii < nloc; ++ii) {
      int i_idx = in_nlist.ilist[ii];
      d_nlist_size[i_idx] = 0;
      for(unsigned jj = 0; jj < in_nlist.numneigh[ii]; ++jj) {
        int j_idx = in_nlist.firstneigh[ii][jj];
        d_nlist_a[i_idx][d_nlist_size[i_idx]++] = j_idx ;
      }
    }

    if(DEBUG_MSG) if(tid == 0) print_v(d_nlist_size[0], "[INFO] d_nlist_a  ", d_nlist_a[0]);
    if(DEBUG_MSG) print_v(local_nloc, fmt::format("lmp_list : {}", lmp_list.inum), lmp_list.numneigh);
    if(DEBUG_MSG) utils::logmesg(lmp, "split tid {} build local_nall local_nloc {} local_nall {} \n",  tid,  local_nloc, local_nall);
  }

  #pragma omp barrier


  double *_coord = atom->x[0];
  for(int local_index = 0; local_index < nall; local_index++) {
    int global_index = backward_index_map[local_index];
    ori_dcoord[local_index*3+0] = _coord[global_index*3+0];
    ori_dcoord[local_index*3+1] = _coord[global_index*3+1];
    ori_dcoord[local_index*3+2] = _coord[global_index*3+2];
  }
  
  // 将local atom原子位置排序 
  memcpy(dcoord, ori_dcoord, nall*3*sizeof(FPTYPE));
  atommap.forward (dcoord, ori_dcoord, 3);

  if(DEBUG_MSG) if(tid == 0) print_v(nall * 1, fmt::format("datype   {} {} :: ", nloc, nall), datype);
  if(DEBUG_MSG) if(tid == 0) print_v(nall * 3, fmt::format("dcoord   {} :: ", nall * 3), dcoord);
  // 每种类型原子的邻居数量
  t_timer->stamp();
  // prod_env_mat_a();
  prod_atom_nlist();
  t_timer->stamp(Timer::PROD_ENV);

  // int nall = backward_index_size;
  // int nghost = nall - local_nloc;
  // nghost = nall - local_nloc;
}


#ifdef WITH_TENSOR_FLOW

void DeepPot::get_vector_from_model (std::vector<int> & vec, const std::string node_name) {
  int node_size = graph_def.node_size();
  bool _find = false;
  for(int ii = 0; ii < node_size; ii++) {
    tensorflow::NodeDef node_def = graph_def.node(ii);
    if(node_def.name() == node_name) {
      google::protobuf::Map<std::string, tensorflow::AttrValue > _map = node_def.attr();
      for(auto it = _map.begin(); it != _map.end(); ++it) {
        if(it->second.has_tensor()){
          tensorflow::TensorProto tensor_proto = it->second.tensor();

          tensorflow::Tensor tensor;
          bool status = tensor.FromProto(tensor_proto);
          if (!status) {
            error->one(FLERR, fmt::format("[INFO] {} transfer failed \n", node_name));
          }

          _find = true;

          int size = 1;
          for(int jj = 0; jj < tensor.dims(); jj++) {
            size *= tensor.dim_size(jj);
          }

          vec.resize(size);
          std::string tensor_debug = tensor.DebugString().c_str();
          
          if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] {} {} size {}  type {}\n", node_name, tensor_debug, size, (int)tensor.dtype()));

          for(int jj = 0; jj < size; jj++) {
            vec[jj] = tensor.flat<int>().data()[jj];
          }
          return;
        }
      }
    }
  }
}



FPTYPE*  DeepPot::print_all_node_attr() {
  int node_size = graph_def.node_size();

  utils::logmesg(lmp, fmt::format("[INFO] node_size {}  \n", node_size));

  for (const auto& node : graph_def.node()) {
    utils::logmesg(lmp, fmt::format("[INFO] Node name: {} {}\n", node.name(), node.op()));

    google::protobuf::Map<std::string, tensorflow::AttrValue > _map = node.attr();
    for(auto it = _map.begin(); it != _map.end(); ++it) {
      if(it->second.has_tensor()){
        tensorflow::TensorProto tensor_proto = it->second.tensor();

        tensorflow::Tensor tensor;
        bool status = tensor.FromProto(tensor_proto);
        if (!status) {
            // utils::logmesg(lmp, fmt::format("[INFO] {} transfer failed \n", node_name));
            return NULL;
        }
        std::string tensor_debug = tensor.DebugString().c_str();
        utils::logmesg(lmp, fmt::format("     {} type {}\n", tensor_debug, (int)tensor.dtype()));

        // #endif
      }
    }
  }
  return NULL;
}

FPTYPE*  DeepPot::get_node_attr(std::string node_name) {
  int node_size = graph_def.node_size();
  bool _find = false;
  for(int ii = 0; ii < node_size; ii++) {
    tensorflow::NodeDef node_def = graph_def.node(ii);
    if(node_def.name() == node_name) {
      google::protobuf::Map<std::string, tensorflow::AttrValue > _map = node_def.attr();
      for(auto it = _map.begin(); it != _map.end(); ++it) {
        if(it->second.has_tensor()){
          tensorflow::TensorProto tensor_proto = it->second.tensor();

          tensorflow::Tensor tensor;
          bool status = tensor.FromProto(tensor_proto);
          if (!status) {
              utils::logmesg(lmp, fmt::format("[INFO] {} transfer failed \n", node_name));
              return NULL;
          }

          _find = true;

          int size = 1;
          for(int jj = 0; jj < tensor.dims(); jj++) {
            size *= tensor.dim_size(jj);
          }
          FPTYPE *tmp_data = new FPTYPE[size];

          std::string tensor_debug = tensor.DebugString().c_str();
          
          if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] {} {} size {}  type {}\n", node_name, tensor_debug, size, (int)tensor.dtype()));

          for(int jj = 0; jj < size; jj++) {
            if((int)tensor.dtype() == 1)
              tmp_data[jj] = tensor.flat<float>().data()[jj];
            else 
              tmp_data[jj] = tensor.flat<double>().data()[jj];
          }
          return tmp_data;
        }
      }      
    }
  }
  // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] {} not find \n", node_name));
  return NULL;
}

void DeepPot::load_tensorflow_model(int _in_type, std::string prefix, 
    FPTYPE  *&c_table_info_in, FPTYPE **&c_table_in,
    FPTYPE  ***&c_matrix_in, FPTYPE ***&c_bias_in, FPTYPE ***&c_idt_in, FPTYPE  ***&c_matrix_t_in,
    float16_t  ***&c_matrix_fp16_in,  float16_t ***&c_matrix_t_fp16_in,
    FPTYPE *&avg_zero_in, FPTYPE *&std_ones_in,
    FPTYPE** &grad_f_data_in
  ) {
  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] load_tensorflow_model \n"));

  // print_all_node_attr(graph_def);


  c_table_info_in = get_node_attr(prefix+"filter_type_0/TabulateFusionSeA/table_info");
  // c_table_info_in = get_node_attr(prefix+"filter_type_0/TabulateFusion/table_info");

  if (c_table_info_in == NULL) {
    error->all(FLERR,"c_table_info_in is NULL \n");
  }

  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] c_table_info_in {} {} {} {} {} \n", 
      c_table_info_in[0], c_table_info_in[1], c_table_info_in[2], c_table_info_in[3], c_table_info_in[4]));

  for(int ii = 0; ii < _in_type; ii++) {
    std::string _table_name, _matrix_name, _bias_name, _idt_name;

    for(int jj = 0; jj < ntypes; jj++) {
      _table_name = jj == 0 ? (prefix + "filter_type_" + std::to_string(ii)  + "/Cast/x") 
                : (prefix + "filter_type_" + std::to_string(ii)  + "/Cast_" + std::to_string(jj) + "/x");
      c_table_in[ii*ntypes+jj]   = get_node_attr(_table_name);

      assert(c_table_in[ii*ntypes+jj] != NULL);
    }

    for(int kk = 0; kk < fitting_layer_num; kk++ ) {
      _matrix_name  = prefix + "layer_" + std::to_string(kk)  + "_type_" + std::to_string(ii) + "/matrix";
      _bias_name    = prefix + "layer_" + std::to_string(kk)  + "_type_" + std::to_string(ii) + "/bias";
      _idt_name     = prefix + "layer_" + std::to_string(kk)  + "_type_" + std::to_string(ii) + "/idt";

      c_matrix_in[kk][ii]  = get_node_attr(_matrix_name);
      c_bias_in[kk][ii]    = get_node_attr(_bias_name);
      c_idt_in[kk][ii]     = get_node_attr(_idt_name);
    }

    _matrix_name  = prefix + "final_layer_type_" + std::to_string(ii) + "/matrix";
    _bias_name    = prefix + "final_layer_type_" + std::to_string(ii) + "/bias";

    c_matrix_in[fitting_layer_num][ii]  = get_node_attr(_matrix_name);
    c_bias_in[fitting_layer_num][ii]    = get_node_attr(_bias_name);

    std_ones_in    = get_node_attr(prefix + "descrpt_attr/t_std");
    avg_zero_in    = get_node_attr(prefix + "descrpt_attr/t_avg");
  }

  int matrix_size[4][2] = {{dim_descrpt, n_neuron[0]},
                              {n_neuron[0], n_neuron[1]},
                              {n_neuron[1], n_neuron[2]},
                              {n_neuron[2], 1}};

  if(prefix == "dipole_charge/") matrix_size[3][1] = last_layer_size;

  for(int ii = 0; ii < FITTING_LAYER; ii++) {
    c_matrix_t_in[ii] = new FPTYPE*[_in_type];
    c_matrix_fp16_in[ii] = new float16_t*[_in_type];
    c_matrix_t_fp16_in[ii] = new float16_t*[_in_type];

    for(int type_i = 0;  type_i < _in_type; type_i++) {
      c_matrix_t_in[ii][type_i] = new FPTYPE[matrix_size[ii][0] * matrix_size[ii][1]];
      c_matrix_t_fp16_in[ii][type_i] = new float16_t[matrix_size[ii][0] * matrix_size[ii][1]];
      c_matrix_fp16_in[ii][type_i] = new float16_t[matrix_size[ii][0] * matrix_size[ii][1]];
      for(int mm = 0; mm < matrix_size[ii][0]; mm++) {
        for(int nn = 0; nn < matrix_size[ii][1]; nn++) {
          c_matrix_t_in[ii][type_i][nn*matrix_size[ii][0]+mm] = c_matrix_in[ii][type_i][mm*matrix_size[ii][1]+nn];
          c_matrix_t_fp16_in[ii][type_i][nn*matrix_size[ii][0]+mm] = c_matrix_in[ii][type_i][mm*matrix_size[ii][1]+nn];
          c_matrix_fp16_in[ii][type_i][mm*matrix_size[ii][1]+nn] = c_matrix_in[ii][type_i][mm*matrix_size[ii][1]+nn];
        }
      }
    }
  }

  grad_f_data_in = new FPTYPE*[_in_type];
  for(int type_i = 0; type_i < _in_type; type_i++) {
    FPTYPE _one_matrix[1] = {1.0};
    grad_f_data_in[type_i] = new FPTYPE[240];
    memset(grad_f_data_in[type_i], 0, 240 * sizeof(FPTYPE));

    matmul(1, n_neuron[2], 1, _one_matrix, c_matrix_t_in[3][type_i], NULL, grad_f_data_in[type_i]);
  }

  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] finish load table \n"));


  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state c_matrix[0][0]:"),  c_matrix[0][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[1][0]:"),  c_matrix[1][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[2][0]:"),  c_matrix[2][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[3][0]:"),  c_matrix[3][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[0][0]  :"),  c_bias[0][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[1][0]  :"),  c_bias[1][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[2][0]  :"),  c_bias[2][0]  );
  // if(DEBUG_MSG) print_v(1    , fmt::format("pb state c_bias[3][0]  :"),  c_bias[3][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[0][0]   :"),  c_idt[0][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[1][0]   :"),  c_idt[1][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[2][0]   :"),  c_idt[2][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[3][0]   :"),  c_idt[3][0]   );
  // if(DEBUG_MSG) print_v(136  , fmt::format("pb state c_table[0]    :"),  c_table[0]    );
  // if(DEBUG_MSG) print_v(6    , fmt::format("pb state c_table_info  :"),  c_table_info  );
  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state std_ones      :"),  std_ones      );
  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state avg_zero      :"),  avg_zero      );

}

void DeepPot::load_data_and_bcast(std::string graph_path) {
  std::string file_content;
  if(comm->me == 0) {
    // tensorflow::Status status = tensorflow::ReadFileToString(tensorflow::Env::Default(), graph_path, &file_content);
    std::ifstream file(graph_path, std::ios::binary);
    if (!file) {
      error->all(FLERR, "Failed to open graph file: {} \n", graph_path);
    }
    if (!graph_def.ParseFromIstream(&file)) {
        error->all(FLERR, "Failed to parse graph file: {} \n", graph_path);
    }
    std::string serialized_graph;
    graph_def.SerializeToString(&serialized_graph);
    int size = serialized_graph.size();

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)serialized_graph.c_str(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
  } else {
    int size;
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> buffer(size);
    MPI_Bcast(buffer.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    graph_def.ParseFromArray(buffer.data(), size);
  }
}

#endif

inline void convert_H_W(FPTYPE* _in_table, int N, int C, int H, int W) {
  FPTYPE* _table = new FPTYPE[N * C * H * W];

  memcpy(_table, _in_table, N * C * H * W * sizeof(FPTYPE));

  for(int n = 0; n < N; n++) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // 原索引：input[n, c, h, w]
          int old_idx = ((n * C + c) * H + h) * W + w;
          // 目标索引：output[n, c, w, h]
          int new_idx = ((n * C + c) * W + w) * H + h;
          _in_table[new_idx] = _table[old_idx];
        }
      }
    }
  }

  delete[] _table;
}

void DeepPot::table_convert(FPTYPE** &_in_table, int _ntypes) {
  FPTYPE * _table; 

  if(comm->tabulate_flag == 5) {
    _table = new FPTYPE[1360 * 768];
  } else {
    _table = new FPTYPE[136000 * 256];
  }

  for(int ptr = 0; ptr < _ntypes * _ntypes; ptr++) {
    if(comm->tabulate_flag == 5) {
      FPTYPE const lower   = c_table_info[0];
      FPTYPE const upper   = c_table_info[1];
      FPTYPE const _max    = c_table_info[2];
      FPTYPE const stride0 = c_table_info[3];
      FPTYPE const stride1 = c_table_info[4];

      int total = (upper - lower) / stride0 + (_max - upper) / stride1;

      memcpy(_table, _in_table[ptr], 1360 * 768 * sizeof(FPTYPE));

      int N = total, C = 8, H = 6, W = 16;
      for(int n = 0; n < N; n++) {
        for (int c = 0; c < C; ++c) {
          for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
              // 原索引：input[n, c, h, w]
              int old_idx = ((n * C + c) * H + h) * W + w;
              // 目标索引：output[n, c, w, h]
              int new_idx = ((n * C + c) * W + w) * H + h;
              _in_table[ptr][new_idx] = _table[old_idx];
            }
          }
        }
      }
      memcpy(_table, _in_table[ptr], 1360 * 768 * sizeof(FPTYPE));
      N = total, C = 4, H = 32, W = 6;
      for(int n = 0; n < N; n++) {
        for (int c = 0; c < C; ++c) {
          for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
              // 原索引：input[n, c, h, w]
              int old_idx = ((n * C + c) * H + h) * W + w;
              // 目标索引：output[n, c, w, h]
              int new_idx = ((n * C + c) * W + w) * H + h;
              _in_table[ptr][new_idx] = _table[old_idx];
            }
          }
        }
      }

    } else {
      FPTYPE const lower   = c_table_info[0];
      FPTYPE const upper   = c_table_info[1];
      FPTYPE const _max    = c_table_info[2];
      FPTYPE const stride0 = c_table_info[3];
      FPTYPE const stride1 = c_table_info[4];

      int total = (upper - lower) / stride0 + (_max - upper) / stride1;

      memcpy(_table, _in_table[ptr], total * 256 * sizeof(FPTYPE));

      int N = total, C = 4, H = 32, W = 2;
      for(int n = 0; n < N; n++) {
        for (int c = 0; c < C; ++c) {
          for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
              // 原索引：input[n, c, h, w]
              int old_idx = ((n * C + c) * H + h) * W + w;
              // 目标索引：output[n, c, w, h]
              int new_idx = ((n * C + c) * W + w) * H + h;
              _in_table[ptr][new_idx] = _table[old_idx];
            }
          }
        }
      }
    }
  }

  delete[] _table;
}

void DeepPot::init_value() {
  

  // for(int ii = 0; ii < 4; ii++) {
  //   c_matrix_fp32[ii] = new float*[ntypes];
  //   c_matrix_t_fp32[ii] = new float*[ntypes];
  //   c_bias_fp32[ii] = new float*[ntypes];

  //   if(ii >= 1 && ii <= 2) {
  //     c_idt_fp32[ii] = new float*[ntypes];
  //   }

  //   for(int type_i = 0;  type_i < ntypes; type_i++) {
  //     c_matrix_fp32[ii][type_i] = new float[matrix_size[ii][0] * matrix_size[ii][1]];      
  //     c_matrix_t_fp32[ii][type_i] = new float[matrix_size[ii][0] * matrix_size[ii][1]];      

  //     for(int mm = 0; mm < matrix_size[ii][0] * matrix_size[ii][1]; mm++) {
  //       c_matrix_fp32[ii][type_i][mm] = c_matrix[ii][type_i][mm];
  //       c_matrix_t_fp32[ii][type_i][mm] = c_matrix_t[ii][type_i][mm];
  //     }

  //     c_bias_fp32[ii][type_i] = new float[n_neuron[ii]];

  //     for(int mm = 0; mm < n_neuron[ii]; mm++) {
  //       c_bias_fp32[ii][type_i][mm] = c_bias[ii][type_i][mm];
  //     }

  //     if(ii >= 1 && ii <= 2) {
  //       c_idt_fp32[ii][type_i] = new float[n_neuron[ii]];

  //       for(int mm = 0; mm < n_neuron[ii]; mm++) {
  //         c_idt_fp32[ii][type_i][mm] = c_idt[ii][type_i][mm];
  //       }
  //     }
  //   }
  // }
}

void DeepPot::reserve_buffer(int _max_nloc, int _max_nall) {
  max_nloc  = _max_nloc;
  max_nall  = _max_nall;

  if(DEBUG_MSG) utils::logmesg(lmp, fmt::format("[info] reserve_buffer tid {} max_nnei {} _max_nloc {} _max_nall {} max_nlist {} \n", 
                tid, max_nnei, _max_nloc, _max_nall, max_nlist));

  thread_neigh = new int[_max_nloc*max_nlist];
  thread_local_ilist = new int[_max_nloc];
  thread_local_numneigh = new int[_max_nloc];
  thread_firstneigh = new int*[_max_nloc];
  thread_firstneigh[0] = thread_neigh;

  forward_index_map = new int[max_nall];
  backward_index_map = new int[max_nall];
  

  dforce = new double[_max_nall * 3];           memset(dforce, 0, _max_nall * 3 * sizeof(double));
  dvirial = new double[9];                  memset(dvirial, 0, 9 * sizeof(double));
  dcoord = new FPTYPE[_max_nall * 3];           memset(dcoord, 0, _max_nall * 3 * sizeof(FPTYPE));
  datype = new int[_max_nall];                  memset(datype, 0, _max_nall     * sizeof(int));
  
  ori_dforce = new double[_max_nall * 3];           memset(ori_dforce, 0, _max_nall * 3 * sizeof(double));
  ori_dcoord = new FPTYPE[_max_nall * 3];           memset(ori_dcoord, 0, _max_nall * 3 * sizeof(FPTYPE));
  ori_datype = new int[_max_nall];                  memset(ori_datype, 0, _max_nall     * sizeof(int));

  rij = new FPTYPE*[ntypes*ntypes];
  r_matrix = new FPTYPE*[ntypes*ntypes];
  r_matrix_deriv = new FPTYPE*[ntypes*ntypes];
  s_vector = new FPTYPE*[ntypes*ntypes];

  for(int i = 0; i < ntypes*ntypes; i++) {
    rij[i] = new FPTYPE[_max_nloc * nnei * 3] ;     memset(rij[i], 0, _max_nloc * nnei * 3 * sizeof(FPTYPE));
    r_matrix[i] = new FPTYPE[_max_nloc * ndescrpt] ;     memset(r_matrix[i], 0, _max_nloc * ndescrpt * sizeof(FPTYPE));
    r_matrix_deriv[i] = new FPTYPE[_max_nloc * ndescrpt * 3] ;     memset(r_matrix_deriv[i], 0, _max_nloc * ndescrpt * 3 * sizeof(FPTYPE));
    s_vector[i] = new FPTYPE[_max_nloc * max_nnei];  memset(s_vector[i], 0, _max_nloc*max_nnei*sizeof(FPTYPE)); 
  }

  grad_one_matrix = new FPTYPE[_max_nloc * 3]; for(int i = 0; i < _max_nloc * 3; i++) grad_one_matrix[i] = 1;

  nlist = new int[_max_nloc * nnei] ;     memset(nlist, 0, _max_nloc * nnei * sizeof(int));
  
  d_nlist_a = new int*[_max_nloc];
  for(int i = 0; i < _max_nloc; i++) {
    d_nlist_a[i] = new int[max_nlist];   memset(d_nlist_a[i], 0, max_nlist * sizeof(int)); 
  }
  d_nlist_size = new int[_max_nloc];   memset(d_nlist_size, 0, _max_nloc * sizeof(int)); 

  // sel_nei = new uint64_t[max_nlist];     memset(sel_nei, 0, max_nlist * sizeof(uint64_t)); 
  sel_nei_size = 0;
  sel_nei = new NeighborInfo[max_nlist]; memset(sel_nei, 0, max_nlist*sizeof(NeighborInfo));

  nei_num_v = new int[ntypes+1];        memset(nei_num_v, 0, (ntypes+1) * sizeof(int)); 

  sess_bufs = new Session_Buf[dipole_flag ? 2 : 1];
  reserve_sessBuf(sess_bufs[0], _max_nloc, n_neuron.data(), ntypes, max_nnei);
  if(dipole_flag) reserve_sessBuf(sess_bufs[1], _max_nloc, n_neuron.data(), ntypes, max_nnei);
  
  atommap.reserve(_max_nloc);
  nlist_data.reserve(_max_nloc, _max_nall, max_nlist);
}

void DeepPot::reserve_sessBuf(Session_Buf &_sess_buf, int _max_nloc, int *_n_neuron, int _ntypes, int _max_nnei) {
  _sess_buf.s_vector_grad = new FPTYPE*[_ntypes*_ntypes];
  _sess_buf.r_matrix_grid = new FPTYPE*[_ntypes*_ntypes];
  for(int dim = 0; dim < 3; dim++) _sess_buf.r_matrix_grid_3d[dim] = new FPTYPE*[_ntypes*_ntypes];
  for(int i = 0; i < _ntypes*_ntypes; i++) {
  
    _sess_buf.s_vector_grad[i] = new FPTYPE[_max_nloc * _max_nnei];  memset(_sess_buf.s_vector_grad[i], 0, _max_nloc*_max_nnei*sizeof(FPTYPE)); 
    _sess_buf.r_matrix_grid[i] = new FPTYPE[_max_nloc * _max_nnei * 4];  memset(_sess_buf.r_matrix_grid[i], 0, _max_nloc*_max_nnei*4*sizeof(FPTYPE)); 
    for(int dim = 0; dim < 3; dim++) {_sess_buf.r_matrix_grid_3d[dim][i] = new FPTYPE[_max_nloc * _max_nnei * 4];  memset(_sess_buf.r_matrix_grid_3d[dim][i], 0, _max_nloc*_max_nnei*4*sizeof(FPTYPE)); }
  }
  
  _sess_buf.rg_fusion             = new FPTYPE*[_ntypes];
  _sess_buf.rg_silce             = new FPTYPE*[_ntypes];
  _sess_buf.descrptor             = new FPTYPE*[_ntypes];
  _sess_buf.qmat                      = new FPTYPE*[_ntypes];
  _sess_buf.qmat_grad                      = new FPTYPE*[_ntypes];
  for(int i = 0; i < _ntypes; i++) {
    _sess_buf.rg_fusion[i] = new FPTYPE[_max_nloc*4*last_layer_size]; memset(_sess_buf.rg_fusion[i], 0, _max_nloc*4*last_layer_size*sizeof(FPTYPE)); 
    _sess_buf.rg_silce[i] = new FPTYPE[_max_nloc*4*n_axis_neuron];   memset(_sess_buf.rg_silce[i], 0, _max_nloc*4*n_axis_neuron*sizeof(FPTYPE)); 
    _sess_buf.descrptor[i] = new FPTYPE[_max_nloc*last_layer_size*n_axis_neuron];   memset(_sess_buf.descrptor[i], 0, _max_nloc*last_layer_size*n_axis_neuron*sizeof(FPTYPE)); 
    
    if(dipole_flag) {
      _sess_buf.qmat[i]               =  new FPTYPE[_max_nloc*3*last_layer_size];   memset(_sess_buf.qmat[i], 0, _max_nloc*3*last_layer_size*sizeof(FPTYPE)); 
      _sess_buf.qmat_grad[i]          =  new FPTYPE[_max_nloc*3*last_layer_size];   memset(_sess_buf.qmat_grad[i], 0, _max_nloc*3*last_layer_size*sizeof(FPTYPE)); 

    }
  }
  
  _sess_buf.descrptor_grad = new FPTYPE[_max_nloc * dim_descrpt];     memset(_sess_buf.descrptor_grad, 0, sizeof(FPTYPE)* _max_nloc * dim_descrpt   );
  _sess_buf.layer_0 = new FPTYPE[_max_nloc * n_neuron[0]];           memset(_sess_buf.layer_0, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[0]         );
  _sess_buf.layer_1 = new FPTYPE[_max_nloc * n_neuron[1]];           memset(_sess_buf.layer_1, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[1]         );
  _sess_buf.layer_2 = new FPTYPE[_max_nloc * n_neuron[2]];           memset(_sess_buf.layer_2, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[2]         );
  _sess_buf.layer_final = new FPTYPE[_max_nloc * last_layer_size];   memset(_sess_buf.layer_final, 0, sizeof(FPTYPE)* _max_nloc * last_layer_size                   );
  _sess_buf.layer_final_qmat = new FPTYPE[_max_nloc*1*3];            memset(_sess_buf.layer_final_qmat, 0, sizeof(FPTYPE)* _max_nloc*1*3                   );
  _sess_buf.layer_0_tanh = new FPTYPE[_max_nloc * n_neuron[0]];      memset(_sess_buf.layer_0_tanh, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[0]    );
  _sess_buf.layer_1_tanh = new FPTYPE[_max_nloc * n_neuron[1]];      memset(_sess_buf.layer_1_tanh, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[1]    );
  _sess_buf.layer_2_tanh = new FPTYPE[_max_nloc * n_neuron[2]];      memset(_sess_buf.layer_2_tanh, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[2]    );
  _sess_buf.layer_0_grad = new FPTYPE[_max_nloc * n_neuron[0]];      memset(_sess_buf.layer_0_grad, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[0]    );
  _sess_buf.layer_1_grad = new FPTYPE[_max_nloc * n_neuron[1]];      memset(_sess_buf.layer_1_grad, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[1]    );
  _sess_buf.layer_2_grad = new FPTYPE[_max_nloc * n_neuron[2]];      memset(_sess_buf.layer_2_grad, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[2]    );
  _sess_buf.layer_1_grad_reg = new FPTYPE[_max_nloc * n_neuron[1]];  memset(_sess_buf.layer_1_grad_reg, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[1]);
  _sess_buf.layer_2_grad_reg = new FPTYPE[_max_nloc * n_neuron[2]];  memset(_sess_buf.layer_2_grad_reg, 0, sizeof(FPTYPE)* _max_nloc * n_neuron[2]);
  _sess_buf.layer_final_grad = new FPTYPE[_max_nloc * last_layer_size];  memset(_sess_buf.layer_final_grad, 0, sizeof(FPTYPE)* _max_nloc * last_layer_size);
  
  _sess_buf.gemm_fp16_buf = new float16_t[_max_nloc * (n_neuron[0] + dim_descrpt)];  memset(_sess_buf.gemm_fp16_buf, 0, sizeof(float16_t)* _max_nloc * (n_neuron[0] + dim_descrpt));
  
  _sess_buf.rg_fusion_grad = new FPTYPE[_max_nloc * last_layer_size * 4];  memset(_sess_buf.rg_fusion_grad, 0, sizeof(FPTYPE)* _max_nloc * last_layer_size * 4);
  _sess_buf.rg_slice_grad = new FPTYPE[_max_nloc * n_axis_neuron * 4];    memset(_sess_buf.rg_slice_grad, 0, sizeof(FPTYPE)* _max_nloc * n_axis_neuron * 4);
  
}

void DeepPot::init(DeepPot *_deep_pot, int _tid) {
  tid = _tid;
  sel = _deep_pot->sel;
  sel_a = _deep_pot->sel_a;
  sel_r = _deep_pot->sel_r;
  dipole_flag = _deep_pot->dipole_flag;

  nnei_a = _deep_pot->nnei_a;
  nnei_r = _deep_pot->nnei_r;
  nnei = _deep_pot->nnei;
  nem = _deep_pot->nem;

  rcut = _deep_pot->rcut;
  rcut_smth = _deep_pot->rcut_smth;

  ndescrpt_a = _deep_pot->ndescrpt_a;
  ndescrpt_r = _deep_pot->ndescrpt_r;
  ndescrpt = _deep_pot->ndescrpt;

  ntypes = _deep_pot->ntypes;

  n_neuron = _deep_pot->n_neuron;

  box = _deep_pot->box;

  sec_a.resize(ntypes+1, 0);
  cum_sum(sec_a, sel_a);

  sec_type_atom.resize(ntypes+1, 0);
  type_natoms.resize(ntypes);

  c_table = new FPTYPE*[ntypes * ntypes];

  c_table_info_pair = _deep_pot->c_table_info_pair;
  c_table_pair = new FPTYPE*[ntypes * ntypes];
  for(int ii = 0; ii < ntypes * ntypes; ii++){
    c_table_pair[ii] = _deep_pot->c_table_pair[ii];
  }
  avg_zero_pair = _deep_pot->avg_zero_pair;
  std_ones_pair = _deep_pot->std_ones_pair;

  for(int ii = 0; ii < FITTING_LAYER; ii++) {
    c_matrix_pair[ii] = new FPTYPE*[ntypes];
    c_bias_pair[ii] = new FPTYPE*[ntypes];
    c_idt_pair[ii] = new FPTYPE*[ntypes];
    c_matrix_t_pair[ii] = new FPTYPE*[ntypes];
    c_matrix_fp16_pair[ii] = new float16_t*[ntypes];
    c_matrix_t_fp16_pair[ii] = new float16_t*[ntypes];
    for(int type_i = 0; type_i < ntypes; type_i++) {
      c_matrix_pair[ii][type_i] = _deep_pot->c_matrix_pair[ii][type_i];
      c_bias_pair[ii][type_i] = _deep_pot->c_bias_pair[ii][type_i];
      c_idt_pair[ii][type_i] = _deep_pot->c_idt_pair[ii][type_i];
      c_matrix_t_pair[ii][type_i] = _deep_pot->c_matrix_t_pair[ii][type_i];
      c_matrix_fp16_pair[ii][type_i] = _deep_pot->c_matrix_fp16_pair[ii][type_i];
      c_matrix_t_fp16_pair[ii][type_i] = _deep_pot->c_matrix_t_fp16_pair[ii][type_i];
    }
  }

  grad_f_data_pair = _deep_pot->grad_f_data_pair;
  
  if(dipole_flag) {
    dipole_sel_type = _deep_pot->dipole_sel_type;

    c_table_info_dipole = _deep_pot->c_table_info_dipole;
    c_table_dipole = new FPTYPE*[ntypes * ntypes];
    for(int ii = 0; ii < ntypes * ntypes; ii++){
      c_table_dipole[ii] = _deep_pot->c_table_dipole[ii];
    }
    avg_zero_dipole = _deep_pot->avg_zero_dipole;
    std_ones_dipole = _deep_pot->std_ones_dipole;

  
    for(int ii = 0; ii < 4; ii++) {
      c_matrix_dipole[ii] = new FPTYPE*[ntypes];
      c_bias_dipole[ii] = new FPTYPE*[ntypes];
      c_idt_dipole[ii] = new FPTYPE*[ntypes];
      c_matrix_t_dipole[ii] = new FPTYPE*[ntypes];
      c_matrix_fp16_dipole[ii] = new float16_t*[ntypes];
      c_matrix_t_fp16_dipole[ii] = new float16_t*[ntypes];
      for(int type_i = 0; type_i < ntypes; type_i++) {
        c_matrix_dipole[ii][type_i] = _deep_pot->c_matrix_dipole[ii][type_i];
        c_bias_dipole[ii][type_i] = _deep_pot->c_bias_dipole[ii][type_i];
        c_idt_dipole[ii][type_i] = _deep_pot->c_idt_dipole[ii][type_i];
        c_matrix_t_dipole[ii][type_i] = _deep_pot->c_matrix_t_dipole[ii][type_i];
        c_matrix_fp16_dipole[ii][type_i] = _deep_pot->c_matrix_fp16_dipole[ii][type_i];
        c_matrix_t_fp16_dipole[ii][type_i] = _deep_pot->c_matrix_t_fp16_dipole[ii][type_i];
      }
    }
    grad_f_data_dipole = _deep_pot->grad_f_data_dipole;
  }

  max_nnei = _deep_pot->max_nnei;
  max_nlist = _deep_pot->max_nlist;

  // init_value();
}

void DeepPot::store_pb_data() {
  c_table_info = c_table_info_pair; 
  c_table = c_table_pair;
  std_ones = std_ones_pair;
  avg_zero = avg_zero_pair;
  for(int i = 0; i < 4; i++) {
    c_matrix[i] = c_matrix_pair[i];
    c_bias[i] = c_bias_pair[i];
    c_idt[i] = c_idt_pair[i];
    c_matrix_t[i] = c_matrix_t_pair[i];
    c_matrix_fp16[i] = c_matrix_fp16_pair[i];
  }

  if(ntypes == 1 && comm->tabulate_flag == 5) {
    PB_param_type1 pb_param_type1;

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin store_pb_data\n"));

    if(c_matrix[0][0]) memcpy(pb_param_type1.c_matrix_0,   c_matrix[0][0],     2048*240 * sizeof(double));
    if(c_matrix[1][0]) memcpy(pb_param_type1.c_matrix_1,   c_matrix[1][0],     240*240 * sizeof(double));
    if(c_matrix[2][0]) memcpy(pb_param_type1.c_matrix_2,   c_matrix[2][0],     240*240 * sizeof(double));
    if(c_matrix[3][0]) memcpy(pb_param_type1.c_matrix_3,   c_matrix[3][0],     240*1 * sizeof(double));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_matrix\n"));
    if(c_bias[0][0]  ) memcpy(pb_param_type1.c_bias_0,     c_bias[0][0],       240 * sizeof(double));
    if(c_bias[1][0]  ) memcpy(pb_param_type1.c_bias_1,     c_bias[1][0],       240 * sizeof(double));
    if(c_bias[2][0]  ) memcpy(pb_param_type1.c_bias_2,     c_bias[2][0],       240 * sizeof(double));
    if(c_bias[3][0]  ) memcpy(pb_param_type1.c_bias_3,     c_bias[3][0],       1 * sizeof(double));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_bias\n"));
    // if(c_idt[0][0]   ) memcpy(pb_param_type1.c_idt_0,      c_idt[0][0],        240 * sizeof(double));
    if(c_idt[1][0]   ) memcpy(pb_param_type1.c_idt_1,      c_idt[1][0],        240 * sizeof(double));
    if(c_idt[2][0]   ) memcpy(pb_param_type1.c_idt_2,      c_idt[2][0],        240 * sizeof(double));
    // if(c_idt[3][0]   ) memcpy(pb_param_type1.c_idt_3,      c_idt[3][0],        240 * sizeof(double));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_idt\n"));

    if(c_table[0]    ) memcpy(pb_param_type1.c_table,      c_table[0],         1360*768 * sizeof(double));
    if(c_table_info  ) memcpy(pb_param_type1.c_table_info, c_table_info,       6 * sizeof(double));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_table_info\n"));
    if(std_ones      ) memcpy(pb_param_type1.std_ones,     std_ones,           2048 * sizeof(double));
    if(avg_zero      ) memcpy(pb_param_type1.avg_zero,     avg_zero,           2048 * sizeof(double));

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy store_pb_data\n"));

    FILE * fp;
    if((fp = fopen ("pb_data_copper.dat_t","wb"))==NULL)  {
      error->all(FLERR, "fp open fail \n");
    }
  
    if(fwrite(&pb_param_type1,sizeof(pb_param_type1),1,fp)!=1) {
      error->all(FLERR, "file write error \n");
    }

    fclose(fp);
  } else if(ntypes == 1 && comm->tabulate_flag == 1) {
    PB_param_type4 pb_param_type4;

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin store_pb_data\n"));

    if(c_matrix[0][0]) memcpy(pb_param_type4.c_matrix_0,   c_matrix[0][0],     2048*240 * sizeof(float));
    if(c_matrix[1][0]) memcpy(pb_param_type4.c_matrix_1,   c_matrix[1][0],     240*240 * sizeof(float));
    if(c_matrix[2][0]) memcpy(pb_param_type4.c_matrix_2,   c_matrix[2][0],     240*240 * sizeof(float));
    if(c_matrix[3][0]) memcpy(pb_param_type4.c_matrix_3,   c_matrix[3][0],     240*1 * sizeof(float));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_matrix\n"));
    if(c_bias[0][0]  ) memcpy(pb_param_type4.c_bias_0,     c_bias[0][0],       240 * sizeof(float));
    if(c_bias[1][0]  ) memcpy(pb_param_type4.c_bias_1,     c_bias[1][0],       240 * sizeof(float));
    if(c_bias[2][0]  ) memcpy(pb_param_type4.c_bias_2,     c_bias[2][0],       240 * sizeof(float));
    if(c_bias[3][0]  ) memcpy(pb_param_type4.c_bias_3,     c_bias[3][0],       1 * sizeof(float));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_bias\n"));
    // if(c_idt[0][0]   ) memcpy(pb_param_type4.c_idt_0,      c_idt[0][0],        240 * sizeof(float));
    if(c_idt[1][0]   ) memcpy(pb_param_type4.c_idt_1,      c_idt[1][0],        240 * sizeof(float));
    if(c_idt[2][0]   ) memcpy(pb_param_type4.c_idt_2,      c_idt[2][0],        240 * sizeof(float));
    // if(c_idt[3][0]   ) memcpy(pb_param_type4.c_idt_3,      c_idt[3][0],        240 * sizeof(float));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_idt\n"));

    if(c_table[0]    ) memcpy(pb_param_type4.c_table,      c_table[0],         136000*256 * sizeof(float));
    if(c_table_info  ) memcpy(pb_param_type4.c_table_info, c_table_info,       6 * sizeof(float));
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_table_info\n"));
    if(std_ones      ) memcpy(pb_param_type4.std_ones,     std_ones,           2048 * sizeof(float));
    if(avg_zero      ) memcpy(pb_param_type4.avg_zero,     avg_zero,           2048 * sizeof(float));

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy store_pb_data\n"));

    FILE * fp;
    if((fp = fopen ("pb_data_copper_v0001.dat","wb"))==NULL)  {
      error->all(FLERR, "fp open fail \n");
    }
  
    if(fwrite(&pb_param_type4,sizeof(pb_param_type4),1,fp)!=1) {
      error->all(FLERR, "file write error \n");
    }

    fclose(fp);
  } else if(ntypes == 2 && comm->tabulate_flag == 5) {
    PB_param_type2 pb_param_type2;

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin store_pb_data\n"));
    for(int ii = 0; ii < ntypes; ii++) {
      if(c_matrix[0][ii]) memcpy(pb_param_type2.c_matrix_0[ii],   c_matrix[0][ii],     2048*240 * sizeof(double));
      if(c_matrix[1][ii]) memcpy(pb_param_type2.c_matrix_1[ii],   c_matrix[1][ii],     240*240 * sizeof(double));
      if(c_matrix[2][ii]) memcpy(pb_param_type2.c_matrix_2[ii],   c_matrix[2][ii],     240*240 * sizeof(double));
      if(c_matrix[3][ii]) memcpy(pb_param_type2.c_matrix_3[ii],   c_matrix[3][ii],     240*1 * sizeof(double));
      // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_matrix\n"));
      if(c_bias[0][ii]  ) memcpy(pb_param_type2.c_bias_0[ii],     c_bias[0][ii],       240 * sizeof(double));
      if(c_bias[1][ii]  ) memcpy(pb_param_type2.c_bias_1[ii],     c_bias[1][ii],       240 * sizeof(double));
      if(c_bias[2][ii]  ) memcpy(pb_param_type2.c_bias_2[ii],     c_bias[2][ii],       240 * sizeof(double));
      if(c_bias[3][ii]  ) memcpy(pb_param_type2.c_bias_3[ii],     c_bias[3][ii],       1 * sizeof(double));
      // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_bias\n"));
      // if(c_idt[0][ii]   ) memcpy(pb_param_type2.c_idt_0[ii],      c_idt[0][ii],        240 * sizeof(double));
      if(c_idt[1][ii]   ) memcpy(pb_param_type2.c_idt_1[ii],      c_idt[1][ii],        240 * sizeof(double));
      if(c_idt[2][ii]   ) memcpy(pb_param_type2.c_idt_2[ii],      c_idt[2][ii],        240 * sizeof(double));
      // if(c_idt[3][ii]   ) memcpy(pb_param_type2.c_idt_3[ii],      c_idt[3][ii],        240 * sizeof(double));
    }
    if(std_ones      ) memcpy(pb_param_type2.std_ones,     std_ones,           2*552 * sizeof(double));
    if(avg_zero      ) memcpy(pb_param_type2.avg_zero,     avg_zero,           2*552 * sizeof(double));
    // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_idt\n"));
    if(c_table_info  ) memcpy(pb_param_type2.c_table_info, c_table_info,       6 * sizeof(double));

    for(int ii = 0; ii < ntypes*ntypes; ii++) {
      if(c_table[ii]    ) memcpy(pb_param_type2.c_table[ii],      c_table[ii],         1360*768 * sizeof(double));
    }
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_table_info\n"));

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy store_pb_data\n"));

    FILE * fp;
    if((fp = fopen ("pb_data_water.dat","wb"))==NULL)  {
      error->all(FLERR, "fp open fail \n");
    }
  
    if(fwrite(&pb_param_type2,sizeof(pb_param_type2),1,fp)!=1) {
      error->all(FLERR, "file write error \n");
    }

    fclose(fp);
  } else if(ntypes == 2 && comm->tabulate_flag == 1){
    PB_param_type3 pb_param_type3;

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin store_pb_data\n"));
    for(int ii = 0; ii < ntypes; ii++) {
      if(c_matrix[0][ii]) memcpy(pb_param_type3.c_matrix_0[ii],   c_matrix[0][ii],     2048*240 * sizeof(float));
      if(c_matrix[1][ii]) memcpy(pb_param_type3.c_matrix_1[ii],   c_matrix[1][ii],     240*240 * sizeof(float));
      if(c_matrix[2][ii]) memcpy(pb_param_type3.c_matrix_2[ii],   c_matrix[2][ii],     240*240 * sizeof(float));
      if(c_matrix[3][ii]) memcpy(pb_param_type3.c_matrix_3[ii],   c_matrix[3][ii],     240*1 * sizeof(float));
      // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_matrix\n"));
      if(c_bias[0][ii]  ) memcpy(pb_param_type3.c_bias_0[ii],     c_bias[0][ii],       240 * sizeof(float));
      if(c_bias[1][ii]  ) memcpy(pb_param_type3.c_bias_1[ii],     c_bias[1][ii],       240 * sizeof(float));
      if(c_bias[2][ii]  ) memcpy(pb_param_type3.c_bias_2[ii],     c_bias[2][ii],       240 * sizeof(float));
      if(c_bias[3][ii]  ) memcpy(pb_param_type3.c_bias_3[ii],     c_bias[3][ii],       1 * sizeof(float));
      // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_bias\n"));
      // if(c_idt[0][ii]   ) memcpy(pb_param_type3.c_idt_0[ii],      c_idt[0][ii],        240 * sizeof(float));
      if(c_idt[1][ii]   ) memcpy(pb_param_type3.c_idt_1[ii],      c_idt[1][ii],        240 * sizeof(float));
      if(c_idt[2][ii]   ) memcpy(pb_param_type3.c_idt_2[ii],      c_idt[2][ii],        240 * sizeof(float));
      // if(c_idt[3][ii]   ) memcpy(pb_param_type3.c_idt_3[ii],      c_idt[3][ii],        240 * sizeof(float));
    }
    if(std_ones      ) memcpy(pb_param_type3.std_ones,     std_ones,           2*552 * sizeof(float));
    if(avg_zero      ) memcpy(pb_param_type3.avg_zero,     avg_zero,           2*552 * sizeof(float));
    // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_idt\n"));
    if(c_table_info  ) memcpy(pb_param_type3.c_table_info, c_table_info,       6 * sizeof(float));

    for(int ii = 0; ii < ntypes*ntypes; ii++) {
      if(c_table[ii]    ) memcpy(pb_param_type3.c_table[ii],      c_table[ii],         136000*256 * sizeof(float));
    }
    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy c_table_info\n"));

    if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] finish memcpy store_pb_data\n"));

    FILE * fp;
    if((fp = fopen ("pb_data_water_v0001.dat","wb"))==NULL)  {
      error->all(FLERR, "fp open fail \n");
    }
  
    if(fwrite(&pb_param_type3,sizeof(pb_param_type3),1,fp)!=1) {
      error->all(FLERR, "file write error \n");
    }

    fclose(fp);

  }  
}

void DeepPot::load_data_from_dat(std::string graph_path) {
  c_table_info = c_table_info_pair; c_table = c_table_pair;
  std_ones = std_ones_pair;
  avg_zero = avg_zero_pair;
  for(int i = 0; i < 4; i++) {
    c_matrix[i] = c_matrix_pair[i];
    c_bias[i] = c_bias_pair[i];
    c_idt[i] = c_idt_pair[i];
    c_matrix_t[i] = c_matrix_t_pair[i];
    c_matrix_fp16[i] = c_matrix_fp16_pair[i];
  }

  if(ntypes == 1 && comm->tabulate_flag == 5) {
    PB_param_type1 pb_param_type1;

    if(comm->me == 0) {
      // std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      ifs.read((char*)&pb_param_type1 , sizeof(PB_param_type1));
      ifs.close();

      // FILE *fid;
      // fid = fopen(graph_path.c_str(),"rb");
      // size_t count = fread((char*)&pb_param_type1,sizeof(PB_param_type1),1,fid);
      // fclose(fid);
      if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] load_data_from_dat graph_path {} \n", graph_path);

      MPI_Bcast((char*)&pb_param_type1, sizeof(PB_param_type1), MPI_CHAR, 0, world);
    } else {
      MPI_Bcast((char*)&pb_param_type1, sizeof(PB_param_type1), MPI_CHAR, 0, world);
    }


    c_matrix[0][0] = new FPTYPE[2048*240];    for(int i = 0; i < 2048*240; i++) c_matrix[0][0][i] = (FPTYPE)pb_param_type1.c_matrix_0[i];
    c_matrix[1][0] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[1][0][i] = (FPTYPE)pb_param_type1.c_matrix_1[i];
    c_matrix[2][0] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[2][0][i] = (FPTYPE)pb_param_type1.c_matrix_2[i];
    c_matrix[3][0] = new FPTYPE[240*1]   ;    for(int i = 0; i < 240*1; i++)    c_matrix[3][0][i] = (FPTYPE)pb_param_type1.c_matrix_3[i];
    c_bias[0][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[0][0][i]   = (FPTYPE)pb_param_type1.c_bias_0[i];
    c_bias[1][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[1][0][i]   = (FPTYPE)pb_param_type1.c_bias_1[i];
    c_bias[2][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[2][0][i]   = (FPTYPE)pb_param_type1.c_bias_2[i];
    c_bias[3][0]   = new FPTYPE[1]       ;    for(int i = 0; i < 1; i++)        c_bias[3][0][i]   = (FPTYPE)pb_param_type1.c_bias_3[i];
    c_idt[0][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[0][0][i]    = (FPTYPE)pb_param_type1.c_idt_0[i];
    c_idt[1][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[1][0][i]    = (FPTYPE)pb_param_type1.c_idt_1[i];
    c_idt[2][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[2][0][i]    = (FPTYPE)pb_param_type1.c_idt_2[i];
    c_idt[3][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[3][0][i]    = (FPTYPE)pb_param_type1.c_idt_3[i];
    c_table[0]     = new FPTYPE[1360*768];    for(int i = 0; i < 1360*768; i++) c_table[0][i]     = (FPTYPE)pb_param_type1.c_table[i];
    c_table_info   = new FPTYPE[6]       ;    for(int i = 0; i < 6; i++)        c_table_info[i]   = (FPTYPE)pb_param_type1.c_table_info[i];
    std_ones       = new FPTYPE[2048]    ;    for(int i = 0; i < 2048; i++)     std_ones[i]       = (FPTYPE)pb_param_type1.std_ones[i];
    avg_zero       = new FPTYPE[2048]    ;    for(int i = 0; i < 2048; i++)     avg_zero[i]       = (FPTYPE)pb_param_type1.avg_zero[i];
    
  } else if (ntypes == 1 && comm->tabulate_flag == 1) {
    PB_param_type4 *pb_param_type4 = new PB_param_type4;

    if(comm->me == 0) {
      // std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      ifs.read((char*)pb_param_type4 , sizeof(PB_param_type4));
      ifs.close();

      if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] load_data_from_dat graph_path {} \n", graph_path);

      MPI_Bcast((char*)pb_param_type4, sizeof(PB_param_type4), MPI_CHAR, 0, world);
    } else {
      MPI_Bcast((char*)pb_param_type4, sizeof(PB_param_type4), MPI_CHAR, 0, world);
    }

    c_matrix[0][0] = new FPTYPE[2048*240];    for(int i = 0; i < 2048*240; i++) c_matrix[0][0][i] = (FPTYPE)pb_param_type4->c_matrix_0[i];
    c_matrix[1][0] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[1][0][i] = (FPTYPE)pb_param_type4->c_matrix_1[i];
    c_matrix[2][0] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[2][0][i] = (FPTYPE)pb_param_type4->c_matrix_2[i];
    c_matrix[3][0] = new FPTYPE[240*1]   ;    for(int i = 0; i < 240*1; i++)    c_matrix[3][0][i] = (FPTYPE)pb_param_type4->c_matrix_3[i];
    c_bias[0][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[0][0][i]   = (FPTYPE)pb_param_type4->c_bias_0[i];
    c_bias[1][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[1][0][i]   = (FPTYPE)pb_param_type4->c_bias_1[i];
    c_bias[2][0]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[2][0][i]   = (FPTYPE)pb_param_type4->c_bias_2[i];
    c_bias[3][0]   = new FPTYPE[1]       ;    for(int i = 0; i < 1; i++)        c_bias[3][0][i]   = (FPTYPE)pb_param_type4->c_bias_3[i];
    c_idt[0][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[0][0][i]    = (FPTYPE)pb_param_type4->c_idt_0[i];
    c_idt[1][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[1][0][i]    = (FPTYPE)pb_param_type4->c_idt_1[i];
    c_idt[2][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[2][0][i]    = (FPTYPE)pb_param_type4->c_idt_2[i];
    c_idt[3][0]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[3][0][i]    = (FPTYPE)pb_param_type4->c_idt_3[i];
    c_table[0]     = new FPTYPE[136000*256];  for(int i = 0; i < 136000*256; i++) c_table[0][i]   = (FPTYPE)pb_param_type4->c_table[i];
    // c_table[0]     = new FPTYPE[136000*256];  for(int i = 0; i < 136000*256; i++) c_table[0][i]   = (FPTYPE) i * 0.01;
    c_table_info   = new FPTYPE[6]       ;    for(int i = 0; i < 6; i++)        c_table_info[i]   = (FPTYPE)pb_param_type4->c_table_info[i];
    std_ones       = new FPTYPE[2048]    ;    for(int i = 0; i < 2048; i++)     std_ones[i]       = (FPTYPE)pb_param_type4->std_ones[i];
    avg_zero       = new FPTYPE[2048]    ;    for(int i = 0; i < 2048; i++)     avg_zero[i]       = (FPTYPE)pb_param_type4->avg_zero[i];

    delete pb_param_type4;

  } else if (ntypes == 2 && comm->tabulate_flag == 5) {
    PB_param_type2 pb_param_type2;

    if(comm->me == 0) {
      // std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      ifs.read((char*)&pb_param_type2 , sizeof(PB_param_type2));
      ifs.close();

      // FILE *fid;
      // fid = fopen(graph_path.c_str(),"rb");
      // size_t count = fread((char*)&pb_param_type2,sizeof(PB_param_type2),1,fid);
      // fclose(fid);
      if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] load_data_from_dat graph_path {} \n", graph_path);

      MPI_Bcast((char*)&pb_param_type2, sizeof(PB_param_type2), MPI_CHAR, 0, world);
    } else {
      MPI_Bcast((char*)&pb_param_type2, sizeof(PB_param_type2), MPI_CHAR, 0, world);
    }

    for(int type_i = 0; type_i < ntypes; type_i++) {
      c_matrix[0][type_i] = new FPTYPE[2048*240];    for(int i = 0; i < 2048*240; i++) c_matrix[0][type_i][i] = (FPTYPE)pb_param_type2.c_matrix_0[type_i][i];
      c_matrix[1][type_i] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[1][type_i][i] = (FPTYPE)pb_param_type2.c_matrix_1[type_i][i];
      c_matrix[2][type_i] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[2][type_i][i] = (FPTYPE)pb_param_type2.c_matrix_2[type_i][i];
      c_matrix[3][type_i] = new FPTYPE[240*1]   ;    for(int i = 0; i < 240*1; i++)    c_matrix[3][type_i][i] = (FPTYPE)pb_param_type2.c_matrix_3[type_i][i];
      c_bias[0][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[0][type_i][i]   = (FPTYPE)pb_param_type2.c_bias_0[type_i][i];
      c_bias[1][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[1][type_i][i]   = (FPTYPE)pb_param_type2.c_bias_1[type_i][i];
      c_bias[2][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[2][type_i][i]   = (FPTYPE)pb_param_type2.c_bias_2[type_i][i];
      c_bias[3][type_i]   = new FPTYPE[1]       ;    for(int i = 0; i < 1; i++)        c_bias[3][type_i][i]   = (FPTYPE)pb_param_type2.c_bias_3[type_i][i];
      c_idt[0][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[0][type_i][i]    = (FPTYPE)pb_param_type2.c_idt_0[type_i][i];
      c_idt[1][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[1][type_i][i]    = (FPTYPE)pb_param_type2.c_idt_1[type_i][i];
      c_idt[2][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[2][type_i][i]    = (FPTYPE)pb_param_type2.c_idt_2[type_i][i];
      c_idt[3][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[3][type_i][i]    = (FPTYPE)pb_param_type2.c_idt_3[type_i][i];
    }

    std_ones       = new FPTYPE[2*552]   ;    for(int i = 0; i < 2*552; i++)     std_ones[i]       = (FPTYPE)pb_param_type2.std_ones[i];
    avg_zero       = new FPTYPE[2*552]   ;    for(int i = 0; i < 2*552; i++)     avg_zero[i]       = (FPTYPE)pb_param_type2.avg_zero[i];
    c_table_info   = new FPTYPE[6]       ;    for(int i = 0; i < 6; i++)        c_table_info[i]   = (FPTYPE)pb_param_type2.c_table_info[i];

    for(int type_i = 0; type_i < ntypes*ntypes; type_i++) {
      if(comm->tabulate_flag == 1)  {c_table[type_i]     = new FPTYPE[13600*256];    for(int i = 0; i < 13600*256; i++) c_table[type_i][i]     = i*i*(type_i + 1);}
      else                          {c_table[type_i]     = new FPTYPE[1360*768];    for(int i = 0; i < 1360*768; i++) c_table[type_i][i]     = (FPTYPE)pb_param_type2.c_table[type_i][i];}
    }
  } else if (ntypes == 2 && comm->tabulate_flag == 1) {
    PB_param_type3 *pb_param_type3 = new PB_param_type3;

    if(comm->me == 0) {
      // std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      std::ifstream ifs(graph_path, std::ios::in | std::ios::binary);
      ifs.read((char*)pb_param_type3 , sizeof(PB_param_type3));
      ifs.close();

      if(DEBUG_MSG) utils::logmesg(lmp, "[NUMA] load_data_from_dat graph_path {} \n", graph_path);

      MPI_Bcast((char*)pb_param_type3, sizeof(PB_param_type3), MPI_CHAR, 0, world);
    } else {
      MPI_Bcast((char*)pb_param_type3, sizeof(PB_param_type3), MPI_CHAR, 0, world);
    }

    for(int type_i = 0; type_i < ntypes; type_i++) {
      c_matrix[0][type_i] = new FPTYPE[2048*240];    for(int i = 0; i < 2048*240; i++) c_matrix[0][type_i][i] = (FPTYPE)pb_param_type3->c_matrix_0[type_i][i];
      c_matrix[1][type_i] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[1][type_i][i] = (FPTYPE)pb_param_type3->c_matrix_1[type_i][i];
      c_matrix[2][type_i] = new FPTYPE[240*240] ;    for(int i = 0; i < 240*240; i++)  c_matrix[2][type_i][i] = (FPTYPE)pb_param_type3->c_matrix_2[type_i][i];
      c_matrix[3][type_i] = new FPTYPE[240*1]   ;    for(int i = 0; i < 240*1; i++)    c_matrix[3][type_i][i] = (FPTYPE)pb_param_type3->c_matrix_3[type_i][i];
      c_bias[0][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[0][type_i][i]   = (FPTYPE)pb_param_type3->c_bias_0[type_i][i];
      c_bias[1][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[1][type_i][i]   = (FPTYPE)pb_param_type3->c_bias_1[type_i][i];
      c_bias[2][type_i]   = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_bias[2][type_i][i]   = (FPTYPE)pb_param_type3->c_bias_2[type_i][i];
      c_bias[3][type_i]   = new FPTYPE[1]       ;    for(int i = 0; i < 1; i++)        c_bias[3][type_i][i]   = (FPTYPE)pb_param_type3->c_bias_3[type_i][i];
      c_idt[0][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[0][type_i][i]    = (FPTYPE)pb_param_type3->c_idt_0[type_i][i];
      c_idt[1][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[1][type_i][i]    = (FPTYPE)pb_param_type3->c_idt_1[type_i][i];
      c_idt[2][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[2][type_i][i]    = (FPTYPE)pb_param_type3->c_idt_2[type_i][i];
      c_idt[3][type_i]    = new FPTYPE[240]     ;    for(int i = 0; i < 240; i++)      c_idt[3][type_i][i]    = (FPTYPE)pb_param_type3->c_idt_3[type_i][i];
    }

    std_ones       = new FPTYPE[2*552]   ;    for(int i = 0; i < 2*552; i++)     std_ones[i]       = (FPTYPE)pb_param_type3->std_ones[i];
    avg_zero       = new FPTYPE[2*552]   ;    for(int i = 0; i < 2*552; i++)     avg_zero[i]       = (FPTYPE)pb_param_type3->avg_zero[i];
    c_table_info   = new FPTYPE[6]       ;    for(int i = 0; i < 6; i++)         c_table_info[i]   = (FPTYPE)pb_param_type3->c_table_info[i];

    for(int type_i = 0; type_i < ntypes*ntypes; type_i++) {
      c_table[type_i]     = new FPTYPE[136000*256];    for(int i = 0; i < 136000*256; i++) c_table[type_i][i]     = (FPTYPE)pb_param_type3->c_table[type_i][i];
    }
    delete pb_param_type3;
  }

  int matrix_size[4][2] = {{dim_descrpt, n_neuron[0]},
                              {n_neuron[0], n_neuron[1]},
                              {n_neuron[1], n_neuron[2]},
                              {n_neuron[2], 1}};

  for(int ii = 0; ii < 4; ii++) {
    c_matrix_t[ii] = new FPTYPE*[ntypes];
    c_matrix_fp16[ii] = new float16_t*[ntypes];
    c_matrix_t_fp16[ii] = new float16_t*[ntypes];
    for(int type_i = 0;  type_i < ntypes; type_i++) {
      c_matrix_t[ii][type_i] = new FPTYPE[matrix_size[ii][0] * matrix_size[ii][1]];
      c_matrix_t_fp16[ii][type_i] = new float16_t[matrix_size[ii][0] * matrix_size[ii][1]];
      c_matrix_fp16[ii][type_i] = new float16_t[matrix_size[ii][0] * matrix_size[ii][1]];
      for(int mm = 0; mm < matrix_size[ii][0]; mm++) {
        for(int nn = 0; nn < matrix_size[ii][1]; nn++) {
          c_matrix_t[ii][type_i][nn*matrix_size[ii][0]+mm] = c_matrix[ii][type_i][mm*matrix_size[ii][1]+nn];
          c_matrix_t_fp16[ii][type_i][nn*matrix_size[ii][0]+mm] = c_matrix[ii][type_i][mm*matrix_size[ii][1]+nn];
          c_matrix_fp16[ii][type_i][mm*matrix_size[ii][1]+nn] = c_matrix[ii][type_i][mm*matrix_size[ii][1]+nn];
        }
      }
    }
  }

  grad_f_data = new FPTYPE*[ntypes];
  for(int type_i = 0; type_i < ntypes; type_i++) {
    FPTYPE _one_matrix[1] = {1.0};
    grad_f_data[type_i] = new FPTYPE[240];
    memset(grad_f_data[type_i], 0, 240 * sizeof(FPTYPE));

    matmul(1, n_neuron[2], 1, _one_matrix, c_matrix_t[3][type_i], NULL, grad_f_data[type_i]);
  }

  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state c_matrix[0][0]:"),  c_matrix[0][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[1][0]:"),  c_matrix[1][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[2][0]:"),  c_matrix[2][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_matrix[3][0]:"),  c_matrix[3][0]);
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[0][0]  :"),  c_bias[0][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[1][0]  :"),  c_bias[1][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_bias[2][0]  :"),  c_bias[2][0]  );
  // if(DEBUG_MSG) print_v(1    , fmt::format("pb state c_bias[3][0]  :"),  c_bias[3][0]  );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[0][0]   :"),  c_idt[0][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[1][0]   :"),  c_idt[1][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[2][0]   :"),  c_idt[2][0]   );
  // if(DEBUG_MSG) print_v(240  , fmt::format("pb state c_idt[3][0]   :"),  c_idt[3][0]   );
  // if(DEBUG_MSG) print_v(136  , fmt::format("pb state c_table[0]    :"),  c_table[0]    );
  // if(DEBUG_MSG) print_v(6    , fmt::format("pb state c_table_info  :"),  c_table_info  );
  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state std_ones      :"),  std_ones      );
  // if(DEBUG_MSG) print_v(204  , fmt::format("pb state avg_zero      :"),  avg_zero      );
}

void DeepPot::init(FPTYPE _rcut, FPTYPE _rcut_smth, 
        FPTYPE _ntypes, std::vector<int>& _sel,
        std::vector<FPTYPE> &_box,
        std::string graph_path,
      int _dipole_flag){
  rcut = _rcut;
  rcut_smth = _rcut_smth;
  ntypes = _ntypes;
  sel = _sel;
  sel_a = sel;
  dipole_flag = _dipole_flag;

  box = _box;

  nnei_a = 0; nnei_r = 0;
  for(auto &i : sel_a) nnei_a += i;
  nnei = nnei_a + nnei_r;

  nem = nnei * 4;

  ndescrpt_a = nnei_a * 4;
  ndescrpt_r = nnei_r * 1;
  ndescrpt = ndescrpt_a + ndescrpt_r;

  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] nnei {} ndescrpt {} \n", nnei, ndescrpt));

  sec_a.resize(ntypes+1, 0);
  cum_sum(sec_a, sel_a);

  sec_type_atom.resize(ntypes+1, 0);
  type_natoms.resize(ntypes);

  n_neuron.assign({240, 240, 240, 1});

  c_table   = new FPTYPE*[ntypes * ntypes];

  c_table_pair   = new FPTYPE*[ntypes * ntypes];
  
  for(int ii = 0; ii < FITTING_LAYER; ii++) {
    c_matrix_pair[ii] = new FPTYPE*[ntypes];
    c_bias_pair[ii] = new FPTYPE*[ntypes];
    c_idt_pair[ii] = new FPTYPE*[ntypes];
  }
  
  if(dipole_flag) {
    c_table_dipole = new FPTYPE*[ntypes * ntypes];
    for(int ii = 0; ii < FITTING_LAYER; ii++) {
      c_matrix_dipole[ii] = new FPTYPE*[ntypes];
      c_bias_dipole[ii] = new FPTYPE*[ntypes];
      c_idt_dipole[ii] = new FPTYPE*[ntypes];
    }
  }

  std::string suffix = graph_path.substr(graph_path.find_last_of(".") + 1, graph_path.length());
  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] graph_path  : {} {} \n", graph_path, suffix));

  if(suffix == "pb") {
    #ifdef WITH_TENSOR_FLOW
      load_data_and_bcast(graph_path);
      
      load_tensorflow_model(ntypes, "", 
        c_table_info_pair, c_table_pair,
        c_matrix_pair, c_bias_pair, c_idt_pair, c_matrix_t_pair,
        c_matrix_fp16_pair,   c_matrix_t_fp16_pair,
        avg_zero_pair, std_ones_pair,
        grad_f_data_pair);
      if(dipole_flag) {
        load_tensorflow_model(1, "dipole_charge/", 
          c_table_info_dipole, c_table_dipole,
          c_matrix_dipole, c_bias_dipole, c_idt_dipole, c_matrix_t_dipole,
          c_matrix_fp16_dipole,   c_matrix_t_fp16_dipole,
          avg_zero_dipole, std_ones_dipole,
          grad_f_data_dipole);

        get_vector_from_model(dipole_sel_type, "dipole_charge/model_attr/sel_type");
      }
    #else
      error->all(FLERR,"Illegal graph_path");
    #endif
  } else if(suffix == "dat" ) {
    load_data_from_dat(graph_path);
  } else {
    error->all(FLERR,"Illegal graph_path");
  }
 
  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] load data success \n"));
  
  // if(comm->me == 0) store_pb_data();
  // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] save data success \n"));

  max_nnei = 0;
  for(auto i : sel) if(i > max_nnei) max_nnei = i;
  max_nlist = 3 * nnei;

  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[INFO] max_nlist {} max_nnei {} nnei {} \n", max_nlist, max_nnei, nnei));

  // if(comm->me == 0) utils::logmesg_arry(lmp, fmt::format("table:begore"), c_table_pair[0], 768, 1);

  for(int i = 0; i < ntypes * ntypes; i++) {
    int N = (c_table_info_pair[1] - c_table_info_pair[0]) / c_table_info_pair[3] + 
                          (c_table_info_pair[2] - c_table_info_pair[1]) / c_table_info_pair[4];
    #ifdef HIGH_PREC
      convert_H_W(c_table_pair[i], 1360, 8, 16, 6);
    #else 
      convert_H_W(c_table_pair[i], 1360, 4, 32, 6);
    #endif
  }
  for(int i = 0; i < ntypes * nem; i++) {
    std_ones_pair[i] = 1./std_ones_pair[i];
  }
  
  if(dipole_flag) {
    for(int i = 0; i < 1 * ntypes; i++) {
      int N = (c_table_info_dipole[1] - c_table_info_dipole[0]) / c_table_info_dipole[3] + 
               (c_table_info_dipole[2] - c_table_info_dipole[1]) / c_table_info_dipole[4];
      #ifdef HIGH_PREC
        convert_H_W(c_table_dipole[i], 1360, 8, 16, 6);
      #else 
        convert_H_W(c_table_dipole[i], 1360, 4, 32, 6);
      #endif
    } 
    for(int i = 0; i < ntypes * nem; i++) {
      std_ones_dipole[i] = 1./std_ones_dipole[i];
    }
  }

  if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] table_info  : {} {} {} {} {} {} \n", c_table_info_pair[0], c_table_info_pair[1],
        c_table_info_pair[2], c_table_info_pair[3],c_table_info_pair[4], c_table_info_pair[5]));

  // if(comm->me == 0) utils::logmesg_arry(lmp, fmt::format("table:after"), c_table_pair[0], 768, 1);

  // MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Finalize();
}

void DeepPot::swith_model(int _current_model) {
  if(DEBUG_MSG) utils::logmesg(lmp, "[INFO] swith_model current_mode {}\n", _current_model);

  if(_current_model == 0) {
    c_table_info = c_table_info_pair;  c_table = c_table_pair;
    std_ones = std_ones_pair;  avg_zero = avg_zero_pair;
    grad_f_data = grad_f_data_pair;
    for(int i = 0; i < FITTING_LAYER; i++) {
      c_matrix[i] = c_matrix_pair[i];
      c_bias[i] = c_bias_pair[i];
      c_idt[i] = c_idt_pair[i];
      c_matrix_t[i] = c_matrix_t_pair[i];
      c_matrix_fp16[i] = c_matrix_fp16_pair[i];
      c_matrix_t_fp16[i] = c_matrix_t_fp16_pair[i];
    }
    this_sess = &sess_bufs[0];
  } else {
    c_table_info = c_table_info_dipole;  c_table = c_table_dipole;
    std_ones = std_ones_dipole; avg_zero = avg_zero_dipole;
    grad_f_data = grad_f_data_dipole;
    for(int i = 0; i < FITTING_LAYER; i++) {
      c_matrix[i] = c_matrix_dipole[i];
      c_bias[i] = c_bias_dipole[i];
      c_idt[i] = c_idt_dipole[i];
      c_matrix_t[i] = c_matrix_t_dipole[i];
      c_matrix_fp16[i] = c_matrix_fp16_dipole[i];
      c_matrix_t_fp16[i] = c_matrix_t_fp16_dipole[i];
    }
    this_sess = &sess_bufs[1];
  }

  descrptor = this_sess->descrptor; 
  rg_silce = this_sess->rg_silce; 
  rg_fusion = this_sess->rg_fusion; 
  qmat          = this_sess->qmat;
  qmat_grad     = this_sess->qmat_grad;

  s_vector_grad = this_sess->s_vector_grad;
  r_matrix_grid = this_sess->r_matrix_grid;
  for(int dim = 0; dim < 3; dim++) r_matrix_grid_3d[dim] = this_sess->r_matrix_grid_3d[dim];

  descrptor_grad = this_sess->descrptor_grad;
  layer_0 = this_sess->layer_0;
  layer_1 = this_sess->layer_1;
  layer_2 = this_sess->layer_2;
  layer_final = this_sess->layer_final;
  layer_final_qmat = this_sess->layer_final_qmat;
  layer_0_tanh = this_sess->layer_0_tanh;
  layer_1_tanh = this_sess->layer_1_tanh;
  layer_2_tanh = this_sess->layer_2_tanh;
  layer_0_grad = this_sess->layer_0_grad;
  layer_1_grad = this_sess->layer_1_grad;
  layer_2_grad = this_sess->layer_2_grad;
  layer_1_grad_reg = this_sess->layer_1_grad_reg;
  layer_2_grad_reg = this_sess->layer_2_grad_reg;
  layer_final_grad = this_sess->layer_final_grad;

  gemm_fp16_buf = this_sess->gemm_fp16_buf;
  rg_fusion_grad = this_sess->rg_fusion_grad;
  rg_slice_grad = this_sess->rg_slice_grad;
}

// dforce_为 parallel_force
void DeepPot::compute (ENERGYTYPE *			dener_,
  double*	dforce_,
  double*	dvirial_,	 
  int _current_model) {
 
  dener = 0;

  current_model = _current_model;
  
  //  切换模式
 swith_model(_current_model);

  if(DEBUG_MSG) utils::logmesg(lmp, "[INFO] deepmd compute tid {} _current_model \n", tid, _current_model);


  if (nloc == 0) {
    *dener_ = 0;
    memset(dvirial_, 0, 9 * sizeof(double));
    return;
  }

  memset(dforce, 0, sizeof(double) * 3 * nall);
  memset(dvirial, 0, sizeof(double) * 9);

  // fwd_map，存放 real atom在新的表里的位置
  // bkw_map，存放 所有的 real atom
  session_run();

  *dener_ = dener;
  memcpy(ori_dforce, dforce, nall * 3 * sizeof(double));
  memcpy(dvirial_, dvirial, 9 * sizeof(double));
  atommap.backward (ori_dforce, dforce, 3);
  
  // backward会原本的排布
  for(int local_index = 0; local_index < nall; local_index++) {
    int global_index = backward_index_map[local_index];
    dforce_[global_index * 3 + 0] = ori_dforce[local_index * 3 + 0];
    dforce_[global_index * 3 + 1] = ori_dforce[local_index * 3 + 1];
    dforce_[global_index * 3 + 2] = ori_dforce[local_index * 3 + 2];
  }
}

void DeepPot::compute (ENERGYTYPE &			dener_,
	 double* &	dforce_,
	 double* &	dvirial_,
	 FPTYPE* &	dcoord_,
	 int* &	datype_,
	 const int			nghost_,
   const int      nloc_,
	 const InputNlist &		lmp_list,
	 const int&			ago,
   int _current_model) {

  nghost = nghost_;
  nloc =   nloc_;
  nall = nghost + nloc;
  
  dener = 0;

  current_model = _current_model;

  // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin deeppot compute _current_model {}\n", _current_model));

  swith_model(_current_model);

  if (nloc == 0) {
    dener_ = 0;
    memset(dvirial_, 0, 9 * sizeof(double));
    return;
  }

  memset(dforce, 0, sizeof(double) * 3 * nall);
  memset(dvirial, 0, sizeof(double) * 9);

  if(max_nloc < nloc) error->one(FLERR, "[ERRIR] max_nloc < nloc max_nloc {} nloc {}", max_nloc, nloc);
  if(max_nall < nall) error->one(FLERR, "[ERRIR] max_nall < nall max_nloc {} nall {}", max_nall, nall);

  // if(comm->me == 0) utils::logmesg(lmp, fmt::format("[info] begin deeppot compute\n"));

  // fwd_map，存放 real atom在新的表里的位置
  // bkw_map，存放 所有的 real atom

  // t_timer->stamp();
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);

    // 初始化 atommap，对local atom的type进行排序
    atommap.init(datype_, nloc);

    // 更新列表，将ilst和jlist更新为排序后的顺序
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(in_nlist);

    max_nbor_size = 0;

    for(int ii = 0; ii < in_nlist.inum; ++ii){
      if(in_nlist.numneigh[ii] > max_nbor_size) max_nbor_size = in_nlist.numneigh[ii];
    }

    assert(max_nbor_size < max_nlist);

    // if(DEBUG_MSG) if(tid == 0)  
    // if(DEBUG_MSG) utils::logmesg(lmp, fmt::format("[info] max_nbor_size {} inum {} nloc {} nnei {} \n", max_nbor_size, in_nlist.inum, nloc, nnei));

    memcpy(datype, atommap.get_type(), nloc * sizeof(int));

    memcpy(datype + nloc, datype_ + nloc, nghost * sizeof(int));

    // local atom每种类型原子数量
    for(int ii = 0; ii < ntypes; ii++){
      type_natoms[ii] = 0;
    }
    for (unsigned ii = 0; ii < nloc; ++ii) {
      type_natoms[datype[ii]] ++;
    }

    cum_sum(sec_type_atom, type_natoms);

    // if(DEBUG_MSG) print_v(ntypes, "[INFO] type atoms  "   , type_natoms);
    if(DEBUG_MSG) if(tid == 0) print_v(ntypes, "[INFO] type atoms  "   , type_natoms);
    if(DEBUG_MSG) if(tid == 0) print_v(ntypes+1, "[INFO] sec_type_atom  ", sec_type_atom);

    if(DEBUG_MSG) if(tid == 0) print_v(ntypes+1, "[INFO] sec_a  ", sec_a);

    // if(DEBUG_MSG) print_v(nloc, "[INFO] ilist  ", in_nlist.ilist);
    // if(DEBUG_MSG) print_v(nloc, "[INFO] numneigh  ", in_nlist.numneigh);

    for (unsigned ii = 0; ii < nloc; ++ii) {
      int i_idx = in_nlist.ilist[ii];
      d_nlist_size[i_idx] = 0;
      for(unsigned jj = 0; jj < in_nlist.numneigh[ii]; ++jj) {
        int j_idx = in_nlist.firstneigh[ii][jj];
        d_nlist_a[i_idx][d_nlist_size[i_idx]++] = j_idx ;
      }
    }

    if(DEBUG_MSG) if(tid == 0) print_v(d_nlist_size[0], "[INFO] d_nlist_a  ", d_nlist_a[0]);
  }

  // 将local atom原子位置排序 
  memcpy(dcoord, dcoord_, nall*3*sizeof(FPTYPE));
  atommap.forward (dcoord, dcoord_, 3);

  if(DEBUG_MSG) if(tid == 0) print_v(nall * 1, fmt::format("datype   {} {} :: ", nloc, nall), datype);
  if(DEBUG_MSG) if(tid == 0) print_v(nall * 3, fmt::format("dcoord   {} :: ", nall * 3), dcoord);
  
  // return;
  session_run();

  dener_ = dener;
  memcpy(dforce_, dforce, nall * 3 * sizeof(double));
  memcpy(dvirial_, dvirial, 9 * sizeof(double));
  atommap.backward (dforce_, dforce, 3);
  
  // bkw map
  // dforce_.resize(fwd_map.size() * 3);
  // select_map<FPTYPE>(dforce_, dforce, bkw_map, 3);
}


void DeepPot::session_run () {
  // get Descriptor

  int sess_ntypes = current_model ? dipole_sel_type.size() : ntypes;

  for(int type_i = 0; type_i < sess_ntypes; type_i++) {
    prod_R_matrix(type_i);
  }  
  for(int type_i = 0; type_i < sess_ntypes; type_i++) {
    embedding_net(type_i);
  }  

  for(int type_i = 0; type_i < sess_ntypes; type_i++) {
    if(current_model)
      fitting_net_dipole(type_i);
    else
      fitting_net_normal(type_i);
  }

  if(DEBUG_MSG) if(tid == 0)  print_v(nloc * 3, fmt::format("prod_force_a_cpu dforce \n"), dforce);
  if(DEBUG_MSG) if(tid == 0)  print_v(9, fmt::format("prod_force_a_cpu dvirial \n"), dvirial);
  if(DEBUG_MSG) if(tid == 0)  print_v(nloc * 3, fmt::format("dforce \n"), dforce);
  return;
}

void DeepPot::embedding_net(int type_i) {
  if(type_natoms[type_i] == 0) return;

  int start_index = sec_type_atom[type_i];

  memset(rg_fusion[type_i], 0, type_natoms[type_i] * 4 * last_layer_size * sizeof(FPTYPE));

  for(int type_i_in = 0; type_i_in < ntypes; type_i_in++) {
    int t_ptr = type_i * ntypes + type_i_in;

    if(comm->tabulate_flag == 5) {
      tabulateFusion_sve(type_natoms[type_i], sel[type_i_in], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion[type_i], c_table[t_ptr]);
      // tabulateFusion(type_natoms[type_i], sel[type_i_in], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion[type_i], c_table[t_ptr]);
    } else if(comm->tabulate_flag == 1) {
      tabulateFusion_v1_sve(type_natoms[type_i], sel[type_i_in], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion[type_i], c_table[t_ptr]);
    }

    if(DEBUG_MSG)  if(tid == 0) print_v(sel[type_i_in], fmt::format(" tabluate out s_vector[t_ptr] type_i {} type_i_in {}", type_i, type_i_in), s_vector[t_ptr]);
    if(DEBUG_MSG)  if(tid == 0) print_v(sel[type_i_in] * 4, fmt::format(" tabluate out r_matrix[t_ptr] type_i {} type_i_in {}", type_i, type_i_in), r_matrix[t_ptr]);
    //  if(DEBUG_MSG)  if(tid == 0) print_v(768, fmt::format(" tabluate out c_table[t_ptr] type_i {} type_i_in {}", type_i, type_i_in), c_table[t_ptr]);
    if(DEBUG_MSG)  if(tid == 0) print_v(last_layer_size * 4, fmt::format(" tabluate out rg_fusion[type_i] type_i {} type_i_in {}", type_i, type_i_in), rg_fusion[type_i]);
    if(DEBUG_MSG)  if(tid == 0) printf("\n\n");
  }
  
  for(int _i = 0; _i < type_natoms[type_i] * 4 * last_layer_size; _i++) {
    rg_fusion[type_i][_i] *= 4.0 / ndescrpt;
  }

  t_timer->stamp(Timer::TABULATE);

  if(DEBUG_MSG)  if(tid == 0) print_v(last_layer_size * 4, fmt::format("rg_fusion[type_i] type_i {} ", type_i), rg_fusion[type_i]);

  for(int ii = 0, kk = 0; ii < type_natoms[type_i] * 4 * last_layer_size; ii+=last_layer_size, kk+=n_axis_neuron) {
    for(int jj = 0; jj < n_axis_neuron; jj++){
      rg_silce[type_i][kk+jj] = rg_fusion[type_i][ii+jj];
    }
  }
  
  if(current_model == 1) {
    int ll = 0;
    for(int ii = 0 ; ii < type_natoms[type_i]; ii++) 
      for(int jj = 1 ; jj < 4; jj++) 
        for(int kk = 0 ; kk < last_layer_size; kk++) {
          int _x = ll % last_layer_size; 
          int _y = (ll / last_layer_size) % 3;
          int _z = ll / last_layer_size / 3;
          int new_ptr = (_z * last_layer_size + _x) * 3 + _y;
          qmat[type_i][new_ptr] = rg_fusion[type_i][(ii * 4 + jj) * last_layer_size + kk];
          ll++;
        }
  }

  memset(descrptor[type_i], 0, sizeof(FPTYPE) * type_natoms[type_i] * last_layer_size * n_axis_neuron);

  matmul_3d(type_natoms[type_i], last_layer_size, n_axis_neuron, 4, rg_fusion[type_i], rg_silce[type_i], descrptor[type_i], true, false);
  t_timer->stamp(Timer::EM_MUT_3D);

  if(DEBUG_MSG) if(tid == 0) print_v(last_layer_size * n_axis_neuron, fmt::format("descrptor[type_i] type_i {}\n",type_i ), descrptor[type_i]);
  if(DEBUG_MSG) if(tid == 0) print_v(last_layer_size * 3, fmt::format("qmat[type_i]          type_i {}\n",type_i ), qmat[type_i]);
}


void DeepPot::fitting_net_dipole(int type_i) {
  if(type_natoms[type_i] == 0) return;

  t_timer->stamp();

  if(comm->fp16_flag) {
    matmul(type_natoms[type_i], n_neuron[0], dim_descrpt,  descrptor[type_i], c_matrix_fp16[0][type_i], c_bias[0][type_i], layer_0, gemm_fp16_buf);
  } else {
    matmul(type_natoms[type_i], n_neuron[0], dim_descrpt,  descrptor[type_i], c_matrix[0][type_i], c_bias[0][type_i], layer_0);
  }

  t_timer->stamp(Timer::MATMUL_ADD_0);

  fast_tanh(type_natoms[type_i] * n_neuron[0], layer_0, layer_0_tanh);
  t_timer->stamp(Timer::FAST_TANH);

  // layer_1
  matmul(type_natoms[type_i], n_neuron[1], n_neuron[0],  layer_0_tanh,    c_matrix[1][type_i], c_bias[1][type_i], layer_1);
  t_timer->stamp(Timer::MATMUL_ADD_1);
  // print_v(n_neuron[1], fmt::format("matmul_add layer1 type_i {}: ", type_i), layer_1);
  fast_tanh(type_natoms[type_i] * n_neuron[1], layer_1, layer_1_tanh);
  t_timer->stamp(Timer::FAST_TANH);
  // print_v(n_neuron[1], fmt::format("fast_tanh layer1 type_i {}: ", type_i), layer_1_tanh);
  idt_mult(type_natoms[type_i], n_neuron[1], c_idt[1][type_i], layer_1_tanh, layer_1);    
  t_timer->stamp(Timer::IDT_MULT);
  // print_v(n_neuron[1], fmt::format("idt_mult layer1 type_i {}: ", type_i), layer_1);
  matrix_add(type_natoms[type_i], n_neuron[1], layer_0_tanh, layer_1);
  t_timer->stamp(Timer::MATRIX_ADD);
  // print_v(n_neuron[1], fmt::format("matrix_add layer1 type_i {}: ", type_i), layer_1);

  // layer_2
  matmul(type_natoms[type_i], n_neuron[2], n_neuron[1],  layer_1,    c_matrix[2][type_i], c_bias[2][type_i], layer_2);
  t_timer->stamp(Timer::MATMUL_ADD_2);
  // print_v(n_neuron[1], fmt::format("matmul_add layer2 type_i {}: ", type_i), layer_2);
  fast_tanh(type_natoms[type_i] * n_neuron[2], layer_2, layer_2_tanh);
  t_timer->stamp(Timer::FAST_TANH);


  // print_v(n_neuron[1], fmt::format("fast_tanh layer2 type_i {}: ", type_i), layer_2_tanh);
  idt_mult(type_natoms[type_i], n_neuron[2], c_idt[2][type_i], layer_2_tanh, layer_2);
  t_timer->stamp(Timer::IDT_MULT);
  // print_v(n_neuron[1], fmt::format("idt_mult layer2 type_i {}: ", type_i), layer_2);

  matrix_add(type_natoms[type_i], n_neuron[1], layer_1, layer_2);
  t_timer->stamp(Timer::MATRIX_ADD);
  // print_v(n_neuron[1], fmt::format("matrix_add layer1 type_i {}: ", type_i), layer_2);

  matmul(type_natoms[type_i], last_layer_size,   n_neuron[2],  layer_2,    c_matrix[3][type_i], c_bias[3][type_i], layer_final);
  if(DEBUG_MSG) if(tid == 0) print_v(last_layer_size, fmt::format("layer_final type_{}: ", type_i), layer_final);

  matmul_3d(type_natoms[type_i], 1, 3, last_layer_size, layer_final, qmat[type_i], layer_final_qmat, false, false);

  if(DEBUG_MSG) if(tid == 0) print_v(type_natoms[type_i] * 3, fmt::format("dipole layer_final_qmat type_{}: ", type_i), layer_final_qmat);

  for(int dim = 0; dim < 3; dim++) {

    for(int i = 0; i < type_natoms[type_i]; i++ ) {
      grad_one_matrix[i * 3 + 0] = 0; grad_one_matrix[i * 3 + 1] = 0; grad_one_matrix[i * 3 + 2] = 0; 
      grad_one_matrix[i * 3 + dim] = 1;
    }
  
    matmul_3d(type_natoms[type_i], 1, last_layer_size, 3, grad_one_matrix, qmat[type_i], layer_final_grad, false, true);
    if(DEBUG_MSG) if(tid == 0) print_v(last_layer_size, fmt::format("layer_final_grad type_{}: ", type_i), layer_final_grad);
  
    matmul_3d(type_natoms[type_i], last_layer_size, 3, 1, layer_final, grad_one_matrix, qmat_grad[type_i], true, false);
    if(DEBUG_MSG) if(tid == 0) print_v(last_layer_size*3, fmt::format("qmat_grad type_{}: ", type_i), qmat_grad[type_i]);
    
    matmul(type_natoms[type_i], n_neuron[2], last_layer_size, layer_final_grad, c_matrix_t[3][type_i], NULL, layer_2_grad_reg);
    if(DEBUG_MSG) if(tid == 0) print_v(n_neuron[2], fmt::format("layer_2_grad_reg type_{}: ", type_i), layer_2_grad_reg);
  
    // layer_2_grad
    idt_mult_grad(type_natoms[type_i], n_neuron[2], c_idt[2][type_i], layer_2_grad_reg, layer_2_grad);
    t_timer->stamp(Timer::IDT_MULT_GRAD);
    // print_v(n_neuron[2], fmt::format("final idt_mult_grad type_i {}: ", type_i), layer_2_grad);
    fast_tanh_grad(type_natoms[type_i] * n_neuron[2], layer_2_tanh,layer_2_grad, layer_2_grad);
    t_timer->stamp(Timer::FAST_TANH_GRAD);
    // print_v(n_neuron[2], fmt::format("fast_tanh_grad_2 type_i {}: ", type_i), layer_2_grad);
    matmul(type_natoms[type_i], n_neuron[1], n_neuron[2], layer_2_grad, c_matrix_t[2][type_i], NULL, layer_1_grad_reg);
    t_timer->stamp(Timer::MATMUL_2D_1);
    // print_v(n_neuron[2], fmt::format("layer_1_grad_reg matmul_2d type_i {}: ", type_i), layer_1_grad_reg);
      
    matrix_add(type_natoms[type_i], n_neuron[1], layer_2_grad_reg, layer_1_grad_reg);
    t_timer->stamp(Timer::MATRIX_ADD);
    // print_v(n_neuron[2], fmt::format("layer_1_grad_reg matrix_add type_i {}: ", type_i), layer_1_grad);
    
    // layer_1_grad
    idt_mult_grad(type_natoms[type_i], n_neuron[1], c_idt[1][type_i], layer_1_grad_reg, layer_1_grad);
    t_timer->stamp(Timer::IDT_MULT_GRAD);
    // print_v(n_neuron[2], fmt::format("final idt_mult_grad_1 type_i {}: ", type_i), layer_1_grad);
    fast_tanh_grad(type_natoms[type_i] * n_neuron[1], layer_1_tanh,layer_1_grad, layer_1_grad);
    t_timer->stamp(Timer::FAST_TANH_GRAD);
    // print_v(n_neuron[2], fmt::format("fast_tanh_grad_1 type_i {}: ", type_i), layer_1_grad);
    
    matmul(type_natoms[type_i], n_neuron[0], n_neuron[1], layer_1_grad, c_matrix_t[1][type_i], NULL, layer_0_grad);
    t_timer->stamp(Timer::MATMUL_2D_2);
    // print_v(n_neuron[2], fmt::format("layer_0_grad type_i {}: ", type_i), layer_0_grad);
    
    matrix_add(type_natoms[type_i], n_neuron[1], layer_1_grad_reg, layer_0_grad);
    t_timer->stamp(Timer::MATRIX_ADD);
    // print_v(n_neuron[2], fmt::format("layer_0_grad matrix_add type_i {}: ", type_i), layer_0_grad);
    
    // layer_0_grad
    fast_tanh_grad(type_natoms[type_i] * n_neuron[1], layer_0_tanh,layer_0_grad, layer_0_grad);
    t_timer->stamp(Timer::FAST_TANH_GRAD);
    // print_v(n_neuron[2], fmt::format("fast_tanh_grad_0 type_i {}: ", type_i), layer_0_grad);
    
    if(comm->fp16_flag){
      matmul(type_natoms[type_i], dim_descrpt, n_neuron[0], layer_0_grad, c_matrix_t_fp16[0][type_i], NULL, descrptor_grad, gemm_fp16_buf);
    } else {
      matmul(type_natoms[type_i], dim_descrpt, n_neuron[0], layer_0_grad, c_matrix_t[0][type_i], NULL, descrptor_grad);
    }
    
    if(DEBUG_MSG) if(tid == 0) print_v(n_axis_neuron * last_layer_size, fmt::format("dipole descriptor_grad [n * 2048] type_{}: ", type_i), descrptor_grad);
    
    t_timer->stamp(Timer::MATMUL_2D_3);
    
    memset(rg_fusion_grad, 0, type_natoms[type_i] * last_layer_size * 4);
    memset(rg_slice_grad, 0, type_natoms[type_i] * n_axis_neuron * 4);
    
    matmul_3d(type_natoms[type_i], 4, last_layer_size, n_axis_neuron, rg_silce[type_i], descrptor_grad, rg_fusion_grad, false, true);
    matmul_3d(type_natoms[type_i], 4, n_axis_neuron,   last_layer_size, rg_fusion[type_i], descrptor_grad, rg_slice_grad, false, false);
    
    t_timer->stamp(Timer::MATMUL_3D);
    
    if(DEBUG_MSG) if(tid == 0) print_v(4 * n_axis_neuron  , fmt::format("dipole rg_slice_grad type_{}: ", type_i), rg_slice_grad);
    
    for(int ii = 0; ii < type_natoms[type_i] * 4; ii++) {
      for(int jj = 0; jj < n_axis_neuron; jj++) {
        rg_fusion_grad[ii*last_layer_size+jj] += rg_slice_grad[ii*n_axis_neuron+jj];
      }
    }
    // t_timer->stamp(Timer::FIT_SLICE);

    int ll = 0;
    for(int ii = 0 ; ii < type_natoms[type_i]; ii++) 
      for(int jj = 1 ; jj < 4; jj++) 
        for(int kk = 0 ; kk < last_layer_size; kk++) {
          int _x = ll % last_layer_size; 
          int _y = (ll / last_layer_size) % 3;
          int _z = ll / last_layer_size / 3;
          int new_ptr = (_z * last_layer_size + _x) * 3 + _y;
          rg_fusion_grad[(ii * 4 + jj) * last_layer_size + kk] += qmat_grad[type_i][new_ptr];
          ll++;
        }
    
    if(DEBUG_MSG) if(tid == 0) print_v(4 * last_layer_size, fmt::format("dipole rg_fusion_grad  type_{}: ", type_i), rg_fusion_grad);
    
    for(int ii = 0; ii < type_natoms[type_i] * 4 * last_layer_size; ii++) {
      rg_fusion_grad[ii] *= 4.0 / ndescrpt;
    }
    // t_timer->stamp(Timer::FITTING_NET);
    
    if(DEBUG_MSG) if(tid == 0) print_v(4 * last_layer_size, fmt::format("dipole rg_fusion_grad after / type_{}: ", type_i), rg_fusion_grad);

    t_timer->stamp();
    int t_ptr = type_i * ntypes + type_i;
  
    if(comm->tabulate_flag == 5) {
      tabulate_fusion_grad_cpu_packing_sve(type_natoms[type_i], sel[type_i], s_vector_grad[t_ptr], r_matrix_grid_3d[dim][t_ptr], 
            c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
    } else if(comm->tabulate_flag == 1) {
      tabulate_fusion_grad_cpu_packing_v1_sve(type_natoms[type_i], sel[type_i], s_vector_grad[t_ptr], r_matrix_grid_3d[dim][t_ptr], 
            c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
    }
  
    for(int ii = 0; ii < type_natoms[type_i] * sel[type_i]; ii++) {
        r_matrix_grid_3d[dim][t_ptr][ii*4] += s_vector_grad[t_ptr][ii];
    }
  
    t_timer->stamp(Timer::TABULATE_GRAD);
  
    if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i] * 1, fmt::format("dipole s_vector_grad {} {}:", type_i, type_i), s_vector_grad[t_ptr]);
    if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i] * 4, fmt::format("dipole r_matrix_grid_3d[{}] {} {}:", dim, type_i, type_i), r_matrix_grid_3d[dim][t_ptr]);
  
    t_timer->stamp(Timer::PROD_FV);
  }
  
  // for(int type_i_in = 0; type_i_in < ntypes; type_i_in++) {
  //   t_timer->stamp();
  //   int t_ptr = type_i * ntypes + type_i_in;
  
  //   if(comm->tabulate_flag == 5) {
  //     tabulate_fusion_grad_cpu_packing_sve(type_natoms[type_i], sel[type_i_in], s_vector_grad[t_ptr], r_matrix_grid[t_ptr], 
  //           c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
  //   } else if(comm->tabulate_flag == 1) {
  //     tabulate_fusion_grad_cpu_packing_v1_sve(type_natoms[type_i], sel[type_i_in], s_vector_grad[t_ptr], r_matrix_grid[t_ptr], 
  //           c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
  //   }
  
  //   for(int ii = 0; ii < type_natoms[type_i] * sel[type_i_in]; ii++) {
  //       r_matrix_grid[t_ptr][ii*4] += s_vector_grad[t_ptr][ii];
  //   }
  
  //   t_timer->stamp(Timer::TABULATE_GRAD);
  
  //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in] * 1, fmt::format("s_vector_grad {} {}:", type_i, type_i_in), s_vector_grad[t_ptr]);
  //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in] * 4, fmt::format("r_matrix_grid {} {}:", type_i, type_i_in), r_matrix_grid[t_ptr]);
  
  //   prod_force_a_cpu(r_matrix_grid[t_ptr],
  //             type_i, type_i_in);
  //   if(update->ntimestep == output->next || update->ntimestep == 0) {
  //     prod_virial_a_cpu(r_matrix_grid[t_ptr],
  //               type_i, type_i_in);
  //   } 
  
  //   t_timer->stamp(Timer::PROD_FV);
  // }
}

void DeepPot::fitting_net_normal(int type_i) {
  if(type_natoms[type_i] == 0) return;

  t_timer->stamp();

  if(comm->fp16_flag) {
    matmul(type_natoms[type_i], n_neuron[0], dim_descrpt,  descrptor[type_i], c_matrix_fp16[0][type_i], c_bias[0][type_i], layer_0, gemm_fp16_buf);
  } else {
    matmul(type_natoms[type_i], n_neuron[0], dim_descrpt,  descrptor[type_i], c_matrix[0][type_i], c_bias[0][type_i], layer_0);
  }

  t_timer->stamp(Timer::MATMUL_ADD_0);

  fast_tanh(type_natoms[type_i] * n_neuron[0], layer_0, layer_0_tanh);
  t_timer->stamp(Timer::FAST_TANH);

  // layer_1
  matmul(type_natoms[type_i], n_neuron[1], n_neuron[0],  layer_0_tanh,    c_matrix[1][type_i], c_bias[1][type_i], layer_1);
  t_timer->stamp(Timer::MATMUL_ADD_1);
  // print_v(n_neuron[1], fmt::format("matmul_add layer1 type_i {}: ", type_i), layer_1);
  fast_tanh(type_natoms[type_i] * n_neuron[1], layer_1, layer_1_tanh);
  t_timer->stamp(Timer::FAST_TANH);
  // print_v(n_neuron[1], fmt::format("fast_tanh layer1 type_i {}: ", type_i), layer_1_tanh);
  idt_mult(type_natoms[type_i], n_neuron[1], c_idt[1][type_i], layer_1_tanh, layer_1);    
  t_timer->stamp(Timer::IDT_MULT);
  // print_v(n_neuron[1], fmt::format("idt_mult layer1 type_i {}: ", type_i), layer_1);
  matrix_add(type_natoms[type_i], n_neuron[1], layer_0_tanh, layer_1);
  t_timer->stamp(Timer::MATRIX_ADD);
  // print_v(n_neuron[1], fmt::format("matrix_add layer1 type_i {}: ", type_i), layer_1);

  // layer_2
  matmul(type_natoms[type_i], n_neuron[2], n_neuron[1],  layer_1,    c_matrix[2][type_i], c_bias[2][type_i], layer_2);
  t_timer->stamp(Timer::MATMUL_ADD_2);
  // print_v(n_neuron[1], fmt::format("matmul_add layer2 type_i {}: ", type_i), layer_2);
  fast_tanh(type_natoms[type_i] * n_neuron[2], layer_2, layer_2_tanh);
  t_timer->stamp(Timer::FAST_TANH);




  print_v(n_neuron[2], fmt::format("final grad type_i {}: ", type_i), layer_2_grad_reg);

  if(update->ntimestep == output->next || update->ntimestep == 0) {
    // print_v(n_neuron[1], fmt::format("fast_tanh layer2 type_i {}: ", type_i), layer_2_tanh);
    idt_mult(type_natoms[type_i], n_neuron[2], c_idt[2][type_i], layer_2_tanh, layer_2);
    t_timer->stamp(Timer::IDT_MULT);
    // print_v(n_neuron[1], fmt::format("idt_mult layer2 type_i {}: ", type_i), layer_2);

    matrix_add(type_natoms[type_i], n_neuron[1], layer_1, layer_2);
    t_timer->stamp(Timer::MATRIX_ADD);
    // print_v(n_neuron[1], fmt::format("matrix_add layer1 type_i {}: ", type_i), layer_2);

    // layer_3
    matmul(type_natoms[type_i], 1,            n_neuron[2],  layer_2,    c_matrix[3][type_i], c_bias[3][type_i], layer_final);

    if(DEBUG_MSG) if(tid == 0) print_v(type_natoms[type_i], fmt::format("layer_final type_{}: ", type_i), layer_final);

    // printf("enner %0.6f\n", layer_final[0]); std::fflush(stdout);

    for(int ii = 0; ii < type_natoms[type_i]; ii++) {
      dener += layer_final[ii];
    }
    t_timer->stamp(Timer::MATMUL_ADD_3);
  }

  // layer_3_grad
  // matmul(type_natoms[type_i], n_neuron[2], 1, grad_f_data.data(), c_matrix_t[3][type_i], NULL, layer_2_grad_reg);
  for(int ii = 0; ii < type_natoms[type_i]; ii++) {
    memcpy(layer_2_grad_reg + ii * n_neuron[2], grad_f_data[type_i],  n_neuron[2]*sizeof(FPTYPE));
  }
  t_timer->stamp(Timer::MATMUL_2D_1);

  // layer_2_grad
  idt_mult_grad(type_natoms[type_i], n_neuron[2], c_idt[2][type_i], layer_2_grad_reg, layer_2_grad);
  t_timer->stamp(Timer::IDT_MULT_GRAD);
  // print_v(n_neuron[2], fmt::format("final idt_mult_grad type_i {}: ", type_i), layer_2_grad);
  fast_tanh_grad(type_natoms[type_i] * n_neuron[2], layer_2_tanh,layer_2_grad, layer_2_grad);
  t_timer->stamp(Timer::FAST_TANH_GRAD);
  // print_v(n_neuron[2], fmt::format("fast_tanh_grad_2 type_i {}: ", type_i), layer_2_grad);
  matmul(type_natoms[type_i], n_neuron[1], n_neuron[2], layer_2_grad, c_matrix_t[2][type_i], NULL, layer_1_grad_reg);
  t_timer->stamp(Timer::MATMUL_2D_1);
  // print_v(n_neuron[2], fmt::format("layer_1_grad_reg matmul_2d type_i {}: ", type_i), layer_1_grad_reg);
    
  matrix_add(type_natoms[type_i], n_neuron[1], layer_2_grad_reg, layer_1_grad_reg);
  t_timer->stamp(Timer::MATRIX_ADD);
  // print_v(n_neuron[2], fmt::format("layer_1_grad_reg matrix_add type_i {}: ", type_i), layer_1_grad);
  
  // layer_1_grad
  idt_mult_grad(type_natoms[type_i], n_neuron[1], c_idt[1][type_i], layer_1_grad_reg, layer_1_grad);
  t_timer->stamp(Timer::IDT_MULT_GRAD);
  // print_v(n_neuron[2], fmt::format("final idt_mult_grad_1 type_i {}: ", type_i), layer_1_grad);
  fast_tanh_grad(type_natoms[type_i] * n_neuron[1], layer_1_tanh,layer_1_grad, layer_1_grad);
  t_timer->stamp(Timer::FAST_TANH_GRAD);
  // print_v(n_neuron[2], fmt::format("fast_tanh_grad_1 type_i {}: ", type_i), layer_1_grad);
  
  matmul(type_natoms[type_i], n_neuron[0], n_neuron[1], layer_1_grad, c_matrix_t[1][type_i], NULL, layer_0_grad);
  t_timer->stamp(Timer::MATMUL_2D_2);
  // print_v(n_neuron[2], fmt::format("layer_0_grad type_i {}: ", type_i), layer_0_grad);
  
  matrix_add(type_natoms[type_i], n_neuron[1], layer_1_grad_reg, layer_0_grad);
  t_timer->stamp(Timer::MATRIX_ADD);
  // print_v(n_neuron[2], fmt::format("layer_0_grad matrix_add type_i {}: ", type_i), layer_0_grad);
  
  // layer_0_grad
  fast_tanh_grad(type_natoms[type_i] * n_neuron[1], layer_0_tanh,layer_0_grad, layer_0_grad);
  t_timer->stamp(Timer::FAST_TANH_GRAD);
  // print_v(n_neuron[2], fmt::format("fast_tanh_grad_0 type_i {}: ", type_i), layer_0_grad);
  
  if(comm->fp16_flag){
    matmul(type_natoms[type_i], dim_descrpt, n_neuron[0], layer_0_grad, c_matrix_t_fp16[0][type_i], NULL, descrptor_grad, gemm_fp16_buf);
  } else {
    matmul(type_natoms[type_i], dim_descrpt, n_neuron[0], layer_0_grad, c_matrix_t[0][type_i], NULL, descrptor_grad);
  }
  
  if(DEBUG_MSG) if(tid == 0) print_v(n_axis_neuron * last_layer_size, fmt::format("descriptor_grad [n * 2048] type_{}: ", type_i), descrptor_grad);
  
  
  t_timer->stamp(Timer::MATMUL_2D_3);
  
  memset(rg_fusion_grad, 0, type_natoms[type_i] * last_layer_size * 4);
  memset(rg_slice_grad, 0, type_natoms[type_i] * n_axis_neuron * 4);
  
  matmul_3d(type_natoms[type_i], 4, last_layer_size, n_axis_neuron, rg_silce[type_i], descrptor_grad, rg_fusion_grad, false, true);
  matmul_3d(type_natoms[type_i], 4, n_axis_neuron,   last_layer_size, rg_fusion[type_i], descrptor_grad, rg_slice_grad, false, false);
  
  t_timer->stamp(Timer::MATMUL_3D);
  
  if(DEBUG_MSG) if(tid == 0) print_v(4 * n_axis_neuron  , fmt::format("rg_slice_grad type_{}: ", type_i), rg_slice_grad);
  
  for(int ii = 0; ii < type_natoms[type_i] * 4; ii++) {
    for(int jj = 0; jj < n_axis_neuron; jj++) {
      rg_fusion_grad[ii*last_layer_size+jj] += rg_slice_grad[ii*n_axis_neuron+jj];
    }
  }
  // t_timer->stamp(Timer::FIT_SLICE);

  
  if(DEBUG_MSG) if(tid == 0) print_v(4 * last_layer_size, fmt::format("rg_fusion_grad  type_{}: ", type_i), rg_fusion_grad);
  
  for(int ii = 0; ii < type_natoms[type_i] * 4 * last_layer_size; ii++) {
    rg_fusion_grad[ii] *= 4.0 / ndescrpt;
  }
  // t_timer->stamp(Timer::FITTING_NET);
  
  if(DEBUG_MSG) if(tid == 0) print_v(4 * last_layer_size, fmt::format("rg_fusion_grad after / type_{}: ", type_i), rg_fusion_grad);
  
  for(int type_i_in = 0; type_i_in < ntypes; type_i_in++) {
    t_timer->stamp();
    int t_ptr = type_i * ntypes + type_i_in;
  
    if(comm->tabulate_flag == 5) {
      tabulate_fusion_grad_cpu_packing_sve(type_natoms[type_i], sel[type_i_in], s_vector_grad[t_ptr], r_matrix_grid[t_ptr], 
            c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
    } else if(comm->tabulate_flag == 1) {
      tabulate_fusion_grad_cpu_packing_v1_sve(type_natoms[type_i], sel[type_i_in], s_vector_grad[t_ptr], r_matrix_grid[t_ptr], 
            c_table[t_ptr], s_vector[t_ptr], r_matrix[t_ptr], rg_fusion_grad);
    }
  
    for(int ii = 0; ii < type_natoms[type_i] * sel[type_i_in]; ii++) {
        r_matrix_grid[t_ptr][ii*4] += s_vector_grad[t_ptr][ii];
    }
  
    t_timer->stamp(Timer::TABULATE_GRAD);
  
    if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in] * 1, fmt::format("s_vector_grad {} {}:", type_i, type_i_in), s_vector_grad[t_ptr]);
    if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in] * 4, fmt::format("r_matrix_grid {} {}:", type_i, type_i_in), r_matrix_grid[t_ptr]);
  
    prod_force_a_cpu(r_matrix_grid[t_ptr],
              type_i, type_i_in);
    if(update->ntimestep == output->next || update->ntimestep == 0) {
      prod_virial_a_cpu(r_matrix_grid[t_ptr],
                type_i, type_i_in);
    } 
  
    t_timer->stamp(Timer::PROD_FV);
  }
}

void DeepPot::prod_atom_nlist() {
  if(DEBUG_MSG) utils::logmesg(lmp, "[INFO] prod_atom_nlist tid {} \n", tid);

  for(int type_i = 0; type_i < ntypes; type_i++) {
    for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
      int t_ptr = type_i * ntypes + type_i_in;
      memset(rij[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 3);
    }

    for (int ii = sec_type_atom[type_i],  _ii = 0; ii < sec_type_atom[type_i+1]; ++ii, ++_ii) {
      int*    fmt_nlist_a = nlist + ii * nnei;
      
      // format_nlist_i_cpu_opt
      for(int i = 0;i<  sec_a.back() ; i++) {
        fmt_nlist_a[i] = -1;
      }
    
      memset(sel_nei, 0, d_nlist_size[ii]*sizeof(NeighborInfo));
      sel_nei_size = 0;
  
      FPTYPE ix = dcoord[ii * 3 + 0];
      FPTYPE iy = dcoord[ii * 3 + 1];
      FPTYPE iz = dcoord[ii * 3 + 2];
  
      for (unsigned kk = 0; kk < d_nlist_size[ii]; ++kk) {
        FPTYPE diff[3];
        const int & j_idx = d_nlist_a[ii][kk];
        diff[0] = dcoord[j_idx * 3 + 0] - ix;
        diff[1] = dcoord[j_idx * 3 + 1] - iy;
        diff[2] = dcoord[j_idx * 3 + 2] - iz;
        FPTYPE rr = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];    
        if (rr <= rcut * rcut) {
          sel_nei[sel_nei_size].type = datype[j_idx];
          sel_nei[sel_nei_size].dist = rr;
          sel_nei[sel_nei_size].index = j_idx;
          sel_nei_size++;
          assert(sel_nei_size <= max_nlist);
        }
      }
  
    
      std::sort(sel_nei, sel_nei+sel_nei_size);
  
      for(int kk = 0; kk < ntypes+1; kk++) {
        nei_num_v[kk] = sec_a[kk];
      }
      int overflowed = -1;
      for (unsigned kk = 0; kk < sel_nei_size; ++kk) {
        const int & nei_type = sel_nei[kk].type;
        int index = sel_nei[kk].index;
        if (nei_num_v[nei_type] < sec_a[nei_type+1]) {
            fmt_nlist_a[nei_num_v[nei_type]++]  = index;
        }
        else {
          break;
          overflowed = nei_type;
          // assert(1 == 0);
        }
      }

      for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
        int t_ptr = type_i * ntypes + type_i_in;

        FPTYPE* d_rij_a = rij[t_ptr] + _ii * sel[type_i_in] * 3;

        for (int nei_iter = 0; nei_iter < sel[type_i_in]; ++nei_iter) {
          // 无效邻居为 -1
          if (fmt_nlist_a[nei_iter+sec_a[type_i_in]] < 0) break;
          const int & j_idx = fmt_nlist_a[nei_iter+sec_a[type_i_in]];
          for (int dd = 0; dd < 3; ++dd) {
            d_rij_a[nei_iter * 3 + dd] = dcoord[j_idx * 3 + dd] - dcoord[ii * 3 + dd];
          }
        }
      }
    }
  }
}

void DeepPot::prod_R_matrix(int type_i) {
  for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
    int t_ptr = type_i * ntypes + type_i_in;
    memset(r_matrix[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 4);
    memset(s_vector[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in]);
    memset(r_matrix_deriv[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 4 * 3);
  }

  if(DEBUG_MSG) if(tid == 0) utils::logmesg(lmp, fmt::format("[info] clear memory \n"));

  for (int ii = sec_type_atom[type_i],  _ii = 0; ii < sec_type_atom[type_i+1]; ++ii, ++_ii) {
    int*    fmt_nlist_a = nlist + ii * nnei;
    
    FPTYPE* AVG = &avg_zero[datype[ii] * nem];
    FPTYPE* STD = &std_ones[datype[ii] * nem];

    // 对每一类邻居进行for循环
    for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
      int t_ptr = type_i * ntypes + type_i_in;
      FPTYPE* d_em_a        = r_matrix[t_ptr] + _ii * sel[type_i_in] * 4;
      FPTYPE* d_xyz_scatter = s_vector[t_ptr] + _ii * sel[type_i_in];
      FPTYPE* d_em_a_deriv = r_matrix_deriv[t_ptr] + _ii * sel[type_i_in] * 4 * 3;
      FPTYPE* d_rij_a = rij[t_ptr] + _ii * sel[type_i_in] * 3;

      for (int jj = 0; jj < sel[type_i_in] * 4; ++jj) {
        d_em_a[jj] = - AVG[jj] * STD[jj];
      }
      for (int jj = 0; jj < sel[type_i_in]; ++jj) {
        d_xyz_scatter[jj] = d_em_a[jj*4];
      }

      // 对每一类邻居的有效原子进行for循环
      // for (int nei_iter = sec_a[type_i_in]; nei_iter < sec_a[type_i_in + 1]; ++nei_iter) {
      for (int nei_iter = 0; nei_iter < sel[type_i_in]; ++nei_iter) {
        // 无效邻居为 -1
        if (fmt_nlist_a[nei_iter+sec_a[type_i_in]] < 0) break;

        const FPTYPE * rr = &d_rij_a[nei_iter * 3];
        FPTYPE nr2 = dot3(rr, rr);
        FPTYPE inr = 1./sqrt(nr2);
        FPTYPE nr = nr2 * inr;
        FPTYPE inr2 = inr * inr;
        FPTYPE inr4 = inr2 * inr2;
        FPTYPE inr3 = inr4 * nr;
        FPTYPE sw, dsw;
        spline5_switch(sw, dsw, nr, rcut_smth, rcut);
        
        int idx_deriv = nei_iter * 4 * 3;	// 4 components time 3 directions
        int idx_value = nei_iter * 4;	// 4 components
        
        // 4 value components
        d_em_a[idx_value + 0] = 1./nr;
        d_em_a[idx_value + 1] = rr[0] / nr2;
        d_em_a[idx_value + 2] = rr[1] / nr2;
        d_em_a[idx_value + 3] = rr[2] / nr2;
        // deriv of component 1/r
        d_em_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[0] * inr;
        d_em_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[1] * inr;
        d_em_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[2] * inr;
        // deriv of component x/r2
        d_em_a_deriv[idx_deriv + 3] = (2. * rr[0] * rr[0] * inr4 - inr2) * sw - d_em_a[idx_value + 1] * dsw * rr[0] * inr;
        d_em_a_deriv[idx_deriv + 4] = (2. * rr[0] * rr[1] * inr4	) * sw - d_em_a[idx_value + 1] * dsw * rr[1] * inr;
        d_em_a_deriv[idx_deriv + 5] = (2. * rr[0] * rr[2] * inr4	) * sw - d_em_a[idx_value + 1] * dsw * rr[2] * inr;
        // deriv of component y/r2
        d_em_a_deriv[idx_deriv + 6] = (2. * rr[1] * rr[0] * inr4	) * sw - d_em_a[idx_value + 2] * dsw * rr[0] * inr;
        d_em_a_deriv[idx_deriv + 7] = (2. * rr[1] * rr[1] * inr4 - inr2) * sw - d_em_a[idx_value + 2] * dsw * rr[1] * inr;
        d_em_a_deriv[idx_deriv + 8] = (2. * rr[1] * rr[2] * inr4	) * sw - d_em_a[idx_value + 2] * dsw * rr[2] * inr;
        // deriv of component z/r2
        d_em_a_deriv[idx_deriv + 9] = (2. * rr[2] * rr[0] * inr4	) * sw - d_em_a[idx_value + 3] * dsw * rr[0] * inr;
        d_em_a_deriv[idx_deriv +10] = (2. * rr[2] * rr[1] * inr4	) * sw - d_em_a[idx_value + 3] * dsw * rr[1] * inr;
        d_em_a_deriv[idx_deriv +11] = (2. * rr[2] * rr[2] * inr4 - inr2) * sw - d_em_a[idx_value + 3] * dsw * rr[2] * inr;
        // 4 value components
        d_em_a[idx_value + 0] *= sw;
        d_em_a[idx_value + 1] *= sw;
        d_em_a[idx_value + 2] *= sw;
        d_em_a[idx_value + 3] *= sw;

        d_em_a[idx_value + 0] = (d_em_a[idx_value + 0] - AVG[idx_value + 0]) * STD[idx_value + 0];
        d_em_a[idx_value + 1] = (d_em_a[idx_value + 1] - AVG[idx_value + 1]) * STD[idx_value + 1];
        d_em_a[idx_value + 2] = (d_em_a[idx_value + 2] - AVG[idx_value + 2]) * STD[idx_value + 2];
        d_em_a[idx_value + 3] = (d_em_a[idx_value + 3] - AVG[idx_value + 3]) * STD[idx_value + 3];

        d_xyz_scatter[nei_iter] = d_em_a[idx_value + 0];

        d_em_a_deriv[idx_deriv + 0] *=  STD[idx_value + 0];
        d_em_a_deriv[idx_deriv + 1] *=  STD[idx_value + 0];
        d_em_a_deriv[idx_deriv + 2] *=  STD[idx_value + 0];

        d_em_a_deriv[idx_deriv + 3] *=  STD[idx_value + 1];
        d_em_a_deriv[idx_deriv + 4] *=  STD[idx_value + 1];
        d_em_a_deriv[idx_deriv + 5] *=  STD[idx_value + 1];

        d_em_a_deriv[idx_deriv + 6] *=  STD[idx_value + 2];
        d_em_a_deriv[idx_deriv + 7] *=  STD[idx_value + 2];
        d_em_a_deriv[idx_deriv + 8] *=  STD[idx_value + 2];

        d_em_a_deriv[idx_deriv + 9] *=  STD[idx_value + 3];
        d_em_a_deriv[idx_deriv + 10] *=  STD[idx_value + 3];
        d_em_a_deriv[idx_deriv + 11] *=  STD[idx_value + 3];

        // if(idx_value == 0 && ii == 0 ) {
        //   printf("d_rij_a %.9f %.9f %.9f  ",  d_rij_a[0], d_rij_a[1], d_rij_a[2]);
        //   printf("d_em_a %.9f %.9f %.9f %.9f ",  d_em_a[0], d_em_a[1], d_em_a[2], d_em_a[3] );
        //   printf("nr2 %.9f inr  %.9f nr %.9f sw %.9f \n",  nr2 ,inr  ,nr ,sw);
        // }
      }
      
      if(_ii == 0) {
        if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4, fmt::format("prod_env_mat atom 0 r_matrix : type_i {} type_i_in {}", type_i, type_i_in), r_matrix[t_ptr]);
        if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in], fmt::format("prod_env_mat atom 0 s_vector : type_i {} type_i_in {}", type_i, type_i_in), s_vector[t_ptr]);
        if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4 *3, fmt::format("prod_env_mat atom 0 r_matrix_deriv : type_i {} type_i_in {}", type_i, type_i_in), r_matrix_deriv[t_ptr]);
        if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*3, fmt::format("prod_env_mat atom 0 rij : type_i {} type_i_in {}", type_i, type_i_in), rij[t_ptr]);
        // if(DEBUG_MSG) if(tid == 0) print_v(nnei, fmt::format("prod_env_mat atom 0 nlist : type_i {} type_i_in {}", type_i, type_i_in), nlist);
      }

      // if(_ii == 5) {
      //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4,     fmt::format("prod_env_mat atom 5 r_matrix : type_i {} type_i_in {}", type_i, type_i_in), r_matrix[t_ptr]+5*sel[type_i_in]*4);
      //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in],    fmt::format("prod_env_mat atom 5 s_vector : type_i {} type_i_in {}", type_i, type_i_in), s_vector[t_ptr]+5*sel[type_i_in]);
      //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4 *3, fmt::format("prod_env_mat atom 5 r_matrix_deriv : type_i {} type_i_in {}", type_i, type_i_in), r_matrix_deriv[t_ptr]+5*sel[type_i_in]*4 *3);
      //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*3,     fmt::format("prod_env_mat atom 5 rij : type_i {} type_i_in {}", type_i, type_i_in), rij[t_ptr]+5*sel[type_i_in]*3);
      // }
    }
  }
}

void DeepPot::prod_env_mat_a() {
  for(int type_i = 0; type_i < ntypes; type_i++) {
    for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
      int t_ptr = type_i * ntypes + type_i_in;
      memset(r_matrix[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 4);
      memset(s_vector[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in]);
      memset(r_matrix_deriv[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 4 * 3);
      memset(rij[t_ptr], 0, sizeof(FPTYPE) *  type_natoms[type_i] * sel[type_i_in] * 3);
    }
  }

  if(DEBUG_MSG) if(tid == 0) utils::logmesg(lmp, fmt::format("[info] clear memory \n"));

  for(int type_i = 0; type_i < ntypes; type_i++) {
    for (int ii = sec_type_atom[type_i],  _ii = 0; ii < sec_type_atom[type_i+1]; ++ii, ++_ii) {
      int*    fmt_nlist_a = nlist + ii * nnei;
      
      // format_nlist_i_cpu_opt
      for(int i = 0;i< sec_a.back();i++) {
        fmt_nlist_a[i] = -1;
      }
    
      memset(sel_nei, 0, d_nlist_size[ii]*sizeof(NeighborInfo));
      sel_nei_size = 0;

      FPTYPE ix = dcoord[ii * 3 + 0];
      FPTYPE iy = dcoord[ii * 3 + 1];
      FPTYPE iz = dcoord[ii * 3 + 2];

      for (unsigned kk = 0; kk < d_nlist_size[ii]; ++kk) {
        FPTYPE diff[3];
        const int & j_idx = d_nlist_a[ii][kk];
        diff[0] = dcoord[j_idx * 3 + 0] - ix;
        diff[1] = dcoord[j_idx * 3 + 1] - iy;
        diff[2] = dcoord[j_idx * 3 + 2] - iz;
        FPTYPE rr = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];    
        if (rr <= rcut * rcut) {
          sel_nei[sel_nei_size].type = datype[j_idx];
          sel_nei[sel_nei_size].dist = rr;
          sel_nei[sel_nei_size].index = j_idx;
          sel_nei_size++;
          assert(sel_nei_size <= max_nlist);
        }
      }

    
      std::sort(sel_nei, sel_nei+sel_nei_size);

      for(int kk = 0; kk < ntypes+1; kk++) {
        nei_num_v[kk] = sec_a[kk];
      }
      int overflowed = -1;
      for (unsigned kk = 0; kk < sel_nei_size; ++kk) {
        const int & nei_type = sel_nei[kk].type;
        int index = sel_nei[kk].index;
        if (nei_num_v[nei_type] < sec_a[nei_type+1]) {
            fmt_nlist_a[nei_num_v[nei_type]++]  = index;
        }
        else {
          break;
          overflowed = nei_type;
          // assert(1 == 0);
        }
      }

      FPTYPE* AVG = &avg_zero[datype[ii] * nem];
      FPTYPE* STD = &std_ones[datype[ii] * nem];

      // 对每一类邻居进行for循环
      for (int type_i_in = 0; type_i_in < ntypes; ++type_i_in) {
        int t_ptr = type_i * ntypes + type_i_in;
        FPTYPE* d_em_a        = r_matrix[t_ptr] + _ii * sel[type_i_in] * 4;
        FPTYPE* d_xyz_scatter = s_vector[t_ptr] + _ii * sel[type_i_in];
        FPTYPE* d_em_a_deriv = r_matrix_deriv[t_ptr] + _ii * sel[type_i_in] * 4 * 3;
        FPTYPE* d_rij_a = rij[t_ptr] + _ii * sel[type_i_in] * 3;

        for (int jj = 0; jj < sel[type_i_in] * 4; ++jj) {
          d_em_a[jj] = - AVG[jj] * STD[jj];
        }
        for (int jj = 0; jj < sel[type_i_in]; ++jj) {
          d_xyz_scatter[jj] = d_em_a[jj*4];
        }

        // 对每一类邻居的有效原子进行for循环
        // for (int nei_iter = sec_a[type_i_in]; nei_iter < sec_a[type_i_in + 1]; ++nei_iter) {
        for (int nei_iter = 0; nei_iter < sel[type_i_in]; ++nei_iter) {
          // 无效邻居为 -1
          if (fmt_nlist_a[nei_iter+sec_a[type_i_in]] < 0) break;
          const int & j_idx = fmt_nlist_a[nei_iter+sec_a[type_i_in]];
          for (int dd = 0; dd < 3; ++dd) {
            d_rij_a[nei_iter * 3 + dd] = dcoord[j_idx * 3 + dd] - dcoord[ii * 3 + dd];
          }

          const FPTYPE * rr = &d_rij_a[nei_iter * 3];
          FPTYPE nr2 = dot3(rr, rr);
          FPTYPE inr = 1./sqrt(nr2);
          FPTYPE nr = nr2 * inr;
          FPTYPE inr2 = inr * inr;
          FPTYPE inr4 = inr2 * inr2;
          FPTYPE inr3 = inr4 * nr;
          FPTYPE sw, dsw;
          spline5_switch(sw, dsw, nr, rcut_smth, rcut);
          
          int idx_deriv = nei_iter * 4 * 3;	// 4 components time 3 directions
          int idx_value = nei_iter * 4;	// 4 components
          
          // 4 value components
          d_em_a[idx_value + 0] = 1./nr;
          d_em_a[idx_value + 1] = rr[0] / nr2;
          d_em_a[idx_value + 2] = rr[1] / nr2;
          d_em_a[idx_value + 3] = rr[2] / nr2;
          // deriv of component 1/r
          d_em_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[0] * inr;
          d_em_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[1] * inr;
          d_em_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - d_em_a[idx_value + 0] * dsw * rr[2] * inr;
          // deriv of component x/r2
          d_em_a_deriv[idx_deriv + 3] = (2. * rr[0] * rr[0] * inr4 - inr2) * sw - d_em_a[idx_value + 1] * dsw * rr[0] * inr;
          d_em_a_deriv[idx_deriv + 4] = (2. * rr[0] * rr[1] * inr4	) * sw - d_em_a[idx_value + 1] * dsw * rr[1] * inr;
          d_em_a_deriv[idx_deriv + 5] = (2. * rr[0] * rr[2] * inr4	) * sw - d_em_a[idx_value + 1] * dsw * rr[2] * inr;
          // deriv of component y/r2
          d_em_a_deriv[idx_deriv + 6] = (2. * rr[1] * rr[0] * inr4	) * sw - d_em_a[idx_value + 2] * dsw * rr[0] * inr;
          d_em_a_deriv[idx_deriv + 7] = (2. * rr[1] * rr[1] * inr4 - inr2) * sw - d_em_a[idx_value + 2] * dsw * rr[1] * inr;
          d_em_a_deriv[idx_deriv + 8] = (2. * rr[1] * rr[2] * inr4	) * sw - d_em_a[idx_value + 2] * dsw * rr[2] * inr;
          // deriv of component z/r2
          d_em_a_deriv[idx_deriv + 9] = (2. * rr[2] * rr[0] * inr4	) * sw - d_em_a[idx_value + 3] * dsw * rr[0] * inr;
          d_em_a_deriv[idx_deriv +10] = (2. * rr[2] * rr[1] * inr4	) * sw - d_em_a[idx_value + 3] * dsw * rr[1] * inr;
          d_em_a_deriv[idx_deriv +11] = (2. * rr[2] * rr[2] * inr4 - inr2) * sw - d_em_a[idx_value + 3] * dsw * rr[2] * inr;
          // 4 value components
          d_em_a[idx_value + 0] *= sw;
          d_em_a[idx_value + 1] *= sw;
          d_em_a[idx_value + 2] *= sw;
          d_em_a[idx_value + 3] *= sw;

          d_em_a[idx_value + 0] = (d_em_a[idx_value + 0] - AVG[idx_value + 0]) * STD[idx_value + 0];
          d_em_a[idx_value + 1] = (d_em_a[idx_value + 1] - AVG[idx_value + 1]) * STD[idx_value + 1];
          d_em_a[idx_value + 2] = (d_em_a[idx_value + 2] - AVG[idx_value + 2]) * STD[idx_value + 2];
          d_em_a[idx_value + 3] = (d_em_a[idx_value + 3] - AVG[idx_value + 3]) * STD[idx_value + 3];

          d_xyz_scatter[nei_iter] = d_em_a[idx_value + 0];

          d_em_a_deriv[idx_deriv + 0] *=  STD[idx_value + 0];
          d_em_a_deriv[idx_deriv + 1] *=  STD[idx_value + 0];
          d_em_a_deriv[idx_deriv + 2] *=  STD[idx_value + 0];

          d_em_a_deriv[idx_deriv + 3] *=  STD[idx_value + 1];
          d_em_a_deriv[idx_deriv + 4] *=  STD[idx_value + 1];
          d_em_a_deriv[idx_deriv + 5] *=  STD[idx_value + 1];

          d_em_a_deriv[idx_deriv + 6] *=  STD[idx_value + 2];
          d_em_a_deriv[idx_deriv + 7] *=  STD[idx_value + 2];
          d_em_a_deriv[idx_deriv + 8] *=  STD[idx_value + 2];

          d_em_a_deriv[idx_deriv + 9] *=  STD[idx_value + 3];
          d_em_a_deriv[idx_deriv + 10] *=  STD[idx_value + 3];
          d_em_a_deriv[idx_deriv + 11] *=  STD[idx_value + 3];

          // if(idx_value == 0 && ii == 0 ) {
          //   printf("d_rij_a %.9f %.9f %.9f  ",  d_rij_a[0], d_rij_a[1], d_rij_a[2]);
          //   printf("d_em_a %.9f %.9f %.9f %.9f ",  d_em_a[0], d_em_a[1], d_em_a[2], d_em_a[3] );
          //   printf("nr2 %.9f inr  %.9f nr %.9f sw %.9f \n",  nr2 ,inr  ,nr ,sw);
          // }
        }
        
        if(_ii == 0) {
          if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4, fmt::format("prod_env_mat atom 0 r_matrix : type_i {} type_i_in {}", type_i, type_i_in), r_matrix[t_ptr]);
          if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in], fmt::format("prod_env_mat atom 0 s_vector : type_i {} type_i_in {}", type_i, type_i_in), s_vector[t_ptr]);
          if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4 *3, fmt::format("prod_env_mat atom 0 r_matrix_deriv : type_i {} type_i_in {}", type_i, type_i_in), r_matrix_deriv[t_ptr]);
          if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*3, fmt::format("prod_env_mat atom 0 rij : type_i {} type_i_in {}", type_i, type_i_in), rij[t_ptr]);
          // if(DEBUG_MSG) if(tid == 0) print_v(nnei, fmt::format("prod_env_mat atom 0 nlist : type_i {} type_i_in {}", type_i, type_i_in), nlist);
        }

        // if(_ii == 5) {
        //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4,     fmt::format("prod_env_mat atom 5 r_matrix : type_i {} type_i_in {}", type_i, type_i_in), r_matrix[t_ptr]+5*sel[type_i_in]*4);
        //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in],    fmt::format("prod_env_mat atom 5 s_vector : type_i {} type_i_in {}", type_i, type_i_in), s_vector[t_ptr]+5*sel[type_i_in]);
        //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*4 *3, fmt::format("prod_env_mat atom 5 r_matrix_deriv : type_i {} type_i_in {}", type_i, type_i_in), r_matrix_deriv[t_ptr]+5*sel[type_i_in]*4 *3);
        //   if(DEBUG_MSG) if(tid == 0) print_v(sel[type_i_in]*3,     fmt::format("prod_env_mat atom 5 rij : type_i {} type_i_in {}", type_i, type_i_in), rij[t_ptr]+5*sel[type_i_in]*3);
        // }
      }
    }
  }

  // if(DEBUG_MSG) if(tid == 0) print_v(2*552, fmt::format("prod_env_mat atom 0 AVG : "), avg_zero);
  // if(DEBUG_MSG) if(tid == 0) print_v(2*552, fmt::format("prod_env_mat atom 0 STD : "), std_ones);

  // print_v(ndescrpt, fmt::format("prod_env_mat atom 1 r_matrix : "), r_matrix + 64 * ndescrpt);
  // print_v(ndescrpt * 3, fmt::format("prod_env_mat atom 1 r_matrix_deriv : "), r_matrix_deriv + 64 * ndescrpt * 3);
  // print_v(nnei, fmt::format("prod_env_mat atom 1 nlist : "), nlist + 64 * nnei);
}

void DeepPot::tabulateFusion(int _loc,
  int _nnei,
  FPTYPE* &em_x, // sij
  FPTYPE* &em, // Ri
  FPTYPE *out,
  const FPTYPE* _table) {

  // 此类local atom的数量
  const FPTYPE lower   = c_table_info[0];
  const FPTYPE upper   = c_table_info[1];
  const FPTYPE _max    = c_table_info[2];
  const FPTYPE stride0 = c_table_info[3];
  const FPTYPE stride1 = c_table_info[4];

  // printf("_nnei %d \n", _nnei);
  // printf("lower %f \n",  lower  );  
  // printf(" upper %f \n",  upper  );  
  // printf(" _max %f \n",  _max   );   
  // printf(" stride0 %f \n",  stride0);
  // printf(" stride1 %f \n",  stride1);

  // for every atom, execute a small manual gemm ~
  // double * res = new double[4 * last_layer_size];
  // #pragma omp parallel for

  for (int ii = 0; ii < _loc; ii++) { // 对loc atom 遍历
    FPTYPE ll[4] = {0};
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1]; // 拿到最后一个邻居
    bool unloop = false; 

    FPTYPE* out0 = out + ii * last_layer_size * 4 + 0 * last_layer_size;
    FPTYPE* out1 = out + ii * last_layer_size * 4 + 1 * last_layer_size;
    FPTYPE* out2 = out + ii * last_layer_size * 4 + 2 * last_layer_size;
    FPTYPE* out3 = out + ii * last_layer_size * 4 + 3 * last_layer_size; // 输出的指针

    for (int jj = 0; jj < _nnei; jj++) {  // 对所有邻居遍历
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj];  // sij
      if (ago == xx) { // 是空
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      for (int kbs = 0; kbs < last_layer_size; kbs+=TABLE_STEP) {
        int kbe = kbs + TABLE_STEP;
        const FPTYPE *table0 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 0];
        const FPTYPE *table1 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 1];
        const FPTYPE *table2 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 2];
        const FPTYPE *table3 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 3];
        const FPTYPE *table4 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 4];
        const FPTYPE *table5 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          FPTYPE a0  = table0[kk-kbs]; 
          FPTYPE a1  = table1[kk-kbs]; 
          FPTYPE a2  = table2[kk-kbs]; 
          FPTYPE a3  = table3[kk-kbs];
          FPTYPE a4  = table4[kk-kbs];
          FPTYPE a5  = table5[kk-kbs];
          FPTYPE var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx; // 128 次 多项式拟合

          if (unloop) {
            out0[kk] += (_nnei - jj) * var * ll[0];
            out1[kk] += (_nnei - jj) * var * ll[1];
            out2[kk] += (_nnei - jj) * var * ll[2];
            out3[kk] += (_nnei - jj) * var * ll[3];
          }
          else {
            out0[kk] += var * ll[0];
            out1[kk] += var * ll[1];
            out2[kk] += var * ll[2];
            out3[kk] += var * ll[3];
          }
        }
      }

      if (unloop) break;
    }
  }
}

// tabulateFusion_sve
#ifdef HIGH_PREC
void DeepPot::tabulateFusion_sve(int _loc,
  int _nnei,
  FPTYPE* &em_x,
  FPTYPE* &em,
  FPTYPE *out,
  const FPTYPE* _table) {

  #ifdef __ARM_FEATURE_SVE

  const double lower   = c_table_info[0];
  const double upper   = c_table_info[1];
  const double _max    = c_table_info[2];
  const double stride0 = c_table_info[3];
  const double stride1 = c_table_info[4];

  // std::cout << "(_loc,_nnei,last_layer_size)" << " : " << "(" << _loc << "," << _nnei << "," << last_layer_size << ")" << std::endl;

  // for every atom, execute a small manual gemm ~
  // double * res = new double[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < _loc; ii++) {
    double ll[4] = {0};
    double ago = em_x[ii * _nnei + _nnei - 1];
    bool unloop = false; 

    double * out0 = &out[ii * last_layer_size * 4 + 0 * last_layer_size];
    double * out1 = &out[ii * last_layer_size * 4 + 1 * last_layer_size];
    double * out2 = &out[ii * last_layer_size * 4 + 2 * last_layer_size];
    double * out3 = &out[ii * last_layer_size * 4 + 3 * last_layer_size];

    for (int jj = 0; jj < _nnei; jj++) { 
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      double xx = em_x[ii * _nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      assert(last_layer_size % svcntd() == 0);

      svbool_t ptrue = svptrue_b64();
      svfloat64_t vnei_sub_jj = svdup_f64((double(_nnei - jj)));
      svfloat64_t vxx = svdup_f64(xx);
      svfloat64_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat64_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat64_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat64_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat64_t vll0 = svdup_f64(ll[0]);
      svfloat64_t vll1 = svdup_f64(ll[1]);
      svfloat64_t vll2 = svdup_f64(ll[2]);
      svfloat64_t vll3 = svdup_f64(ll[3]);
      svfloat64_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat64_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat64_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat64_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);

      for(int kk = 0; kk < last_layer_size; kk += svcntd() * 2){
        const double* TABLE = &_table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat64_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat64_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat64_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat64_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat64_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat64_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat64_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat64_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat64_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat64_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat64_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat64_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        svfloat64_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat64_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat64_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat64_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat64_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat64_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat64_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat64_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat64_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat64_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat64_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat64_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat64_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat64_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat64_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat64_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat64_t vvar_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat64_t vvar_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat64_t vout0_0 = svld1(ptrue, out0 + kk);
        svfloat64_t vout0_1 = svld1(ptrue, out0 + kk + svcntd());
        svfloat64_t vout1_0 = svld1(ptrue, out1 + kk);
        svfloat64_t vout1_1 = svld1(ptrue, out1 + kk + svcntd());
        svfloat64_t vout2_0 = svld1(ptrue, out2 + kk);
        svfloat64_t vout2_1 = svld1(ptrue, out2 + kk + svcntd());
        svfloat64_t vout3_0 = svld1(ptrue, out3 + kk);
        svfloat64_t vout3_1 = svld1(ptrue, out3 + kk + svcntd());

        if(unloop){
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0_);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0_);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1_);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1_);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2_);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2_);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3_);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3_);
        } else {
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3);
        }
        svst1(ptrue, out0 + kk, vout0_0);
        svst1(ptrue, out0 + kk + svcntd(), vout0_1);
        svst1(ptrue, out1 + kk, vout1_0);
        svst1(ptrue, out1 + kk + svcntd(), vout1_1);
        svst1(ptrue, out2 + kk, vout2_0);
        svst1(ptrue, out2 + kk + svcntd(), vout2_1);
        svst1(ptrue, out3 + kk, vout3_0);
        svst1(ptrue, out3 + kk + svcntd(), vout3_1);
      }

      if (unloop) break;
    }
  }

  #endif
}
#else 
void DeepPot::tabulateFusion_sve(int _loc,
  int _nnei,
  FPTYPE* &em_x,
  FPTYPE* &em,
  FPTYPE *out,
  const FPTYPE* _table) {

  const FPTYPE lower   = c_table_info[0];
  const FPTYPE upper   = c_table_info[1];
  const FPTYPE _max    = c_table_info[2];
  const FPTYPE stride0 = c_table_info[3];
  const FPTYPE stride1 = c_table_info[4];

  // for every atom, execute a small manual gemm ~
  // FPTYPE * res = new FPTYPE[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < _loc; ii++) {
    svbool_t ptrue = svptrue_b32();

    FPTYPE ll[4] = {0};
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1];
    bool unloop = false; 

    FPTYPE* out0 = out + ii * last_layer_size * 4 + 0 * last_layer_size;
    FPTYPE* out1 = out + ii * last_layer_size * 4 + 1 * last_layer_size;
    FPTYPE* out2 = out + ii * last_layer_size * 4 + 2 * last_layer_size;
    FPTYPE* out3 = out + ii * last_layer_size * 4 + 3 * last_layer_size;

    // int do_prefetch = prefetch_flag;
    // if(do_prefetch) {
    //   for(int jj = 0; jj < PREFETCH_SIZE; jj++) {
    //     FPTYPE xx = em_x[ii * _nnei + jj]; 
    //     int table_idx = 0;
    //     locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);        
    //     const float* TABLE = &_table[table_idx * last_layer_size * 6];
    //     for(int kk = 0; kk < last_layer_size * 6 / svcntw(); kk++){
    //       svprfb_vnum(ptrue, TABLE, kk, SV_PLDL2STRM) ;
    //     }
    //   }
    // }
    // void svprfb_vnum(svbool_t pg, const void *base, int64_t vnum, svprfop op) ;

    for (int jj = 0; jj < _nnei; jj++) { 
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj]; 
      FPTYPE xx_next;
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      assert(last_layer_size % svcntw() == 0);

      svfloat32_t vnei_sub_jj = svdup_f32((double(_nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);
      svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);

      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2){
        const float* TABLE = &_table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11); 

        svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat32_t vvar_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat32_t vvar_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat32_t vout0_0 = svld1(ptrue, out0 + kk);
        svfloat32_t vout0_1 = svld1(ptrue, out0 + kk + svcntw());
        svfloat32_t vout1_0 = svld1(ptrue, out1 + kk);
        svfloat32_t vout1_1 = svld1(ptrue, out1 + kk + svcntw());
        svfloat32_t vout2_0 = svld1(ptrue, out2 + kk);
        svfloat32_t vout2_1 = svld1(ptrue, out2 + kk + svcntw());
        svfloat32_t vout3_0 = svld1(ptrue, out3 + kk);
        svfloat32_t vout3_1 = svld1(ptrue, out3 + kk + svcntw());

        if(unloop){
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0_);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0_);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1_);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1_);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2_);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2_);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3_);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3_);
        }else{
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3);
        }
        svst1(ptrue, out0 + kk, vout0_0);
        svst1(ptrue, out0 + kk + svcntw(), vout0_1);
        svst1(ptrue, out1 + kk, vout1_0);
        svst1(ptrue, out1 + kk + svcntw(), vout1_1);
        svst1(ptrue, out2 + kk, vout2_0);
        svst1(ptrue, out2 + kk + svcntw(), vout2_1);
        svst1(ptrue, out3 + kk, vout3_0);
        svst1(ptrue, out3 + kk + svcntw(), vout3_1);
      }
      if (unloop) break;
    }
  }
}
#endif


void DeepPot::tabulate_fusion_grad_cpu_packing(
    int _loc, int _nnei,
    FPTYPE *dy_dem_x, 
    FPTYPE *dy_dem,
    const FPTYPE * _table, 
    FPTYPE *em_x, 
    FPTYPE *em, 
    FPTYPE *dy) {
  
  memset(dy_dem_x, 0.0, sizeof(FPTYPE) * _loc * _nnei);
  memset(dy_dem, 0.0, sizeof(FPTYPE) * _loc * _nnei * 4);
  FPTYPE const lower   = c_table_info[0];
  FPTYPE const upper   = c_table_info[1];
  FPTYPE const _max    = c_table_info[2];
  FPTYPE const stride0 = c_table_info[3];
  FPTYPE const stride1 = c_table_info[4];

  for (int ii = 0; ii < _loc; ii++) {
    FPTYPE ll[4];
    FPTYPE rr[4];
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1];
    const FPTYPE* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const FPTYPE* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const FPTYPE* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const FPTYPE* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;
    for (int jj = 0; jj < _nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      FPTYPE* dy_dem_tmp = &dy_dem[ii * _nnei * 4 + jj * 4];

      FPTYPE grad = 0.0;
      FPTYPE dy_dem_0 = 0.0;
      FPTYPE dy_dem_1 = 0.0;
      FPTYPE dy_dem_2 = 0.0;
      FPTYPE dy_dem_3 = 0.0;

      for (int kbs = 0; kbs < last_layer_size; kbs += TABLE_STEP){
        int kbe = kbs + TABLE_STEP;
        const FPTYPE* table0 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 0];
        const FPTYPE* table1 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 1];
        const FPTYPE* table2 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 2];
        const FPTYPE* table3 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 3];
        const FPTYPE* table4 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 4];
        const FPTYPE* table5 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          rr[0] = dy0[kk];
          rr[1] = dy1[kk];
          rr[2] = dy2[kk];
          rr[3] = dy3[kk];
          FPTYPE a0  = table0[kk-kbs]; 
          FPTYPE a1  = table1[kk-kbs]; 
          FPTYPE a2  = table2[kk-kbs]; 
          FPTYPE a3  = table3[kk-kbs];
          FPTYPE a4  = table4[kk-kbs];
          FPTYPE a5  = table5[kk-kbs];
          FPTYPE res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

          if (unloop) {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (_nnei - jj);
            dy_dem_0 += res * rr[0] * (_nnei - jj);
            dy_dem_1 += res * rr[1] * (_nnei - jj);
            dy_dem_2 += res * rr[2] * (_nnei - jj);
            dy_dem_3 += res * rr[3] * (_nnei - jj);
          }
          else {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
            dy_dem_0 += res * rr[0];
            dy_dem_1 += res * rr[1];
            dy_dem_2 += res * rr[2];
            dy_dem_3 += res * rr[3];
          }
        }
      }

      dy_dem_x[ii * _nnei + jj] = grad;
      dy_dem_tmp[0] = dy_dem_0;
      dy_dem_tmp[1] = dy_dem_1;
      dy_dem_tmp[2] = dy_dem_2;
      dy_dem_tmp[3] = dy_dem_3;

      if (unloop) break;
    }
  }
  // printf("_nnei %d \n", _nnei);
  // printf("lower %f \n",  lower  );  
  // printf(" upper %f \n",  upper  );  
  // printf(" _max %f \n",  _max   );   
  // printf(" stride0 %f \n",  stride0);
  // printf(" stride1 %f \n",  stride1);


  // print_v(_nnei*4, fmt::format("table:"), _table);
  // print_v(_nnei, fmt::format("em_x:"), em_x);
  // print_v(_nnei*4, fmt::format("em:"), em);
  // print_v(4 * last_layer_size    , fmt::format("dy:"), dy);
  // print_v(_nnei, fmt::format("dy_dem_x:"), dy_dem_x);
  // print_v(_nnei * 4, fmt::format("dy_dem:"), dy_dem);
}


#ifdef HIGH_PREC
void DeepPot::tabulate_fusion_grad_cpu_packing_sve(
    int _loc, int _nnei,
    FPTYPE *dy_dem_x, 
    FPTYPE *dy_dem,
    const FPTYPE * _table, 
    FPTYPE *em_x, 
    FPTYPE *em, 
    FPTYPE *dy) {
  #ifdef __ARM_FEATURE_SVE

  memset(dy_dem_x, 0, sizeof(double) * _loc * _nnei);
  memset(dy_dem, 0, sizeof(double) * _loc * _nnei * 4);
  double const lower   = c_table_info[0];
  double const upper   = c_table_info[1];
  double const _max    = c_table_info[2];
  double const stride0 = c_table_info[3];
  double const stride1 = c_table_info[4];
  // for every atom, execute a small gemm~
  // double * res = new double[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < _loc; ii++) {
    double ll[4];
    double rr[4];
    double ago = em_x[ii * _nnei + _nnei - 1];
    const double* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const double* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const double* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const double* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;
    for (int jj = 0; jj < _nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      double xx = em_x[ii * _nnei + jj]; 
      if (ago == xx) {
      unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      double* dy_dem_tmp = &dy_dem[ii * _nnei * 4 + jj * 4];

      svfloat64_t vgard = svdup_f64(0.);
      svfloat64_t vdy_dem_0 = svdup_f64(0.);
      svfloat64_t vdy_dem_1 = svdup_f64(0.);
      svfloat64_t vdy_dem_2 = svdup_f64(0.);
      svfloat64_t vdy_dem_3 = svdup_f64(0.);

      assert(last_layer_size % svcntd() == 0);
      svfloat64_t vtwo = svdup_f64(2.);
      svfloat64_t vthree = svdup_f64(3.);
      svfloat64_t vfour = svdup_f64(4.);
      svfloat64_t vfive = svdup_f64(5.);

      svbool_t ptrue = svptrue_b64();
      svfloat64_t vnei_sub_jj = svdup_f64((double(_nnei - jj)));
      svfloat64_t vxx = svdup_f64(xx);

      svfloat64_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat64_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat64_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat64_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat64_t v2xx1 = svmul_z(ptrue, vtwo, vxx);
      svfloat64_t v3xx2 = svmul_z(ptrue, vthree, vxx2);
      svfloat64_t v4xx3 = svmul_z(ptrue, vfour, vxx3);
      svfloat64_t v5xx4 = svmul_z(ptrue, vfive, vxx4);
      svfloat64_t vll0 = svdup_f64(ll[0]);
      svfloat64_t vll1 = svdup_f64(ll[1]);
      svfloat64_t vll2 = svdup_f64(ll[2]);
      svfloat64_t vll3 = svdup_f64(ll[3]);
      svfloat64_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat64_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat64_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat64_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);
      for(int kk = 0; kk < last_layer_size; kk += svcntd() * 2){
        svfloat64_t vrr0_0 = svld1(ptrue, dy0 + kk);
        svfloat64_t vrr0_1 = svld1(ptrue, dy0 + kk + svcntd());
        svfloat64_t vrr1_0 = svld1(ptrue, dy1 + kk);
        svfloat64_t vrr1_1 = svld1(ptrue, dy1 + kk + svcntd());
        svfloat64_t vrr2_0 = svld1(ptrue, dy2 + kk);
        svfloat64_t vrr2_1 = svld1(ptrue, dy2 + kk + svcntd());
        svfloat64_t vrr3_0 = svld1(ptrue, dy3 + kk);
        svfloat64_t vrr3_1 = svld1(ptrue, dy3 + kk + svcntd());

        const double* TABLE = &_table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat64_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat64_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat64_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat64_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat64_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat64_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat64_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat64_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat64_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat64_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat64_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat64_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        // double res = a0 + a1 * xx + a2 * xx2 + a3 * xx3 + a4 * xx4 + a5 * xx5;
        svfloat64_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat64_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat64_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat64_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat64_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat64_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat64_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat64_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat64_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat64_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat64_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat64_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat64_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat64_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat64_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat64_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat64_t vres_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat64_t vres_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        // a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4
        svfloat64_t tmp9_0 = svmla_z(ptrue, va1_0, va2_0, v2xx1);
        svfloat64_t tmp9_1 = svmla_z(ptrue, va1_1, va2_1, v2xx1);
        svfloat64_t tmp10_0 = svmul_z(ptrue, va3_0, v3xx2);
        svfloat64_t tmp10_1 = svmul_z(ptrue, va3_1, v3xx2);
        svfloat64_t tmp11_0 = svmul_z(ptrue, va4_0, v4xx3);
        svfloat64_t tmp11_1 = svmul_z(ptrue, va4_1, v4xx3);
        svfloat64_t tmp12_0 = svmul_z(ptrue, va5_0, v5xx4);
        svfloat64_t tmp12_1 = svmul_z(ptrue, va5_1, v5xx4);
        svfloat64_t tmp13_0 = svadd_z(ptrue, tmp9_0, tmp10_0);
        svfloat64_t tmp13_1 = svadd_z(ptrue, tmp9_1, tmp10_1);
        svfloat64_t tmp14_0 = svadd_z(ptrue, tmp11_0, tmp12_0);
        svfloat64_t tmp14_1 = svadd_z(ptrue, tmp11_1, tmp12_1);
        svfloat64_t tmp15_0 = svadd_z(ptrue, tmp13_0, tmp14_0); 
        svfloat64_t tmp15_1 = svadd_z(ptrue, tmp13_1, tmp14_1); 

        // dot(ll, rr);
        svfloat64_t tmp16_0 = svmul_z(ptrue, vll0, vrr0_0);
        svfloat64_t tmp16_1 = svmul_z(ptrue, vll0, vrr0_1);
        svfloat64_t tmp17_0 = svmul_z(ptrue, vll1, vrr1_0);
        svfloat64_t tmp17_1 = svmul_z(ptrue, vll1, vrr1_1);
        svfloat64_t tmp18_0 = svmul_z(ptrue, vll2, vrr2_0);
        svfloat64_t tmp18_1 = svmul_z(ptrue, vll2, vrr2_1);
        svfloat64_t tmp19_0 = svmul_z(ptrue, vll3, vrr3_0);
        svfloat64_t tmp19_1 = svmul_z(ptrue, vll3, vrr3_1);
        svfloat64_t tmp20_0 = svadd_z(ptrue, tmp16_0, tmp17_0);
        svfloat64_t tmp20_1 = svadd_z(ptrue, tmp16_1, tmp17_1);
        svfloat64_t tmp21_0 = svadd_z(ptrue, tmp18_0, tmp19_0);
        svfloat64_t tmp21_1 = svadd_z(ptrue, tmp18_1, tmp19_1);
        svfloat64_t tmp22_0 = svadd_z(ptrue, tmp20_0, tmp21_0);
        svfloat64_t tmp22_1 = svadd_z(ptrue, tmp20_1, tmp21_1);

        // grad = (a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4 ) * dot(ll, rr);
        svfloat64_t vgard_0 = svmul_z(ptrue, tmp15_0, tmp22_0);
        svfloat64_t vgard_1 = svmul_z(ptrue, tmp15_1, tmp22_1);

        svfloat64_t vres0_0 = svmul_z(ptrue, vres_0, vrr0_0);
        svfloat64_t vres0_1 = svmul_z(ptrue, vres_1, vrr0_1);
        svfloat64_t vres1_0 = svmul_z(ptrue, vres_0, vrr1_0);
        svfloat64_t vres1_1 = svmul_z(ptrue, vres_1, vrr1_1);
        svfloat64_t vres2_0 = svmul_z(ptrue, vres_0, vrr2_0);
        svfloat64_t vres2_1 = svmul_z(ptrue, vres_1, vrr2_1);
        svfloat64_t vres3_0 = svmul_z(ptrue, vres_0, vrr3_0);
        svfloat64_t vres3_1 = svmul_z(ptrue, vres_1, vrr3_1);
        if(unloop){
          vgard_0 = svmul_z(ptrue, vgard_0, vnei_sub_jj);
          vgard_1 = svmul_z(ptrue, vgard_1, vnei_sub_jj);
          vres0_0 = svmul_z(ptrue, vres0_0, vnei_sub_jj);
          vres0_1 = svmul_z(ptrue, vres0_1, vnei_sub_jj);
          vres1_0 = svmul_z(ptrue, vres1_0, vnei_sub_jj);
          vres1_1 = svmul_z(ptrue, vres1_1, vnei_sub_jj);
          vres2_0 = svmul_z(ptrue, vres2_0, vnei_sub_jj);
          vres2_1 = svmul_z(ptrue, vres2_1, vnei_sub_jj);
          vres3_0 = svmul_z(ptrue, vres3_0, vnei_sub_jj);
          vres3_1 = svmul_z(ptrue, vres3_1, vnei_sub_jj);
        }
        vgard = svadd_z(ptrue, vgard, vgard_0);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_0);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_0);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_0);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_0);
        vgard = svadd_z(ptrue, vgard, vgard_1);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_1);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_1);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_1);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_1);
      }

      dy_dem_x[ii * _nnei + jj] = svaddv(ptrue, vgard);
      dy_dem_tmp[0] = svaddv(ptrue, vdy_dem_0);
      dy_dem_tmp[1] = svaddv(ptrue, vdy_dem_1);
      dy_dem_tmp[2] = svaddv(ptrue, vdy_dem_2);
      dy_dem_tmp[3] = svaddv(ptrue, vdy_dem_3);  

      if (unloop) break;
    }
  }

  if((getenv("COMM_DEBUG_FLAG") != nullptr && atoi(getenv("COMM_DEBUG_FLAG")) == 1)){
    std::stringstream ss;

    ss << "tabulate_grad  "<< std::endl;
    ss << "em  "<< std::endl;
    for(int _i = 0; _i < _nnei * 4; _i++) {
      ss << std::fixed << std::setprecision(9) << em[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "em_x  "<< std::endl;
    for(int _i = 0; _i < _nnei; _i++) {
      ss << std::fixed << std::setprecision(9) << em_x[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "dy_dem  "<< std::endl;
    for(int _i = 0; _i < _nnei; _i++) {
      ss << std::fixed << std::setprecision(9) << dy_dem[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "dy_dem_x  "<< std::endl;
    for(int _i = 0; _i < _nnei * 4; _i++) {
      ss << std::fixed << std::setprecision(9) << dy_dem_x[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "dy  "<< std::endl;
    for(int _i = 0; _i < 4* last_layer_size; _i++) {
      ss << std::fixed << std::setprecision(9) << dy[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "table  "<< std::endl;
    for(int _i = 0; _i < 128; _i++) {
      ss << std::fixed << std::setprecision(9) << _table[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    std::cout << ss.str() << std::endl;
    fflush(stdout);
  }

  #endif
}

#else
void DeepPot::tabulate_fusion_grad_cpu_packing_sve(
    int _loc, int _nnei,
    FPTYPE *dy_dem_x, 
    FPTYPE *dy_dem,
    const FPTYPE * _table, 
    FPTYPE *em_x, 
    FPTYPE *em, 
    FPTYPE *dy) {

  memset(dy_dem_x, 0.0, sizeof(float) * _loc * _nnei);
  memset(dy_dem, 0.0, sizeof(float) * _loc * _nnei * 4);
  float const lower   = c_table_info[0];
  float const upper   = c_table_info[1];
  float const _max    = c_table_info[2];
  float const stride0 = c_table_info[3];
  float const stride1 = c_table_info[4];
  // for every atom, execute a small gemm~
  // float * res = new float[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < _loc; ii++) {
    float ll[4];
    float rr[4];
    float ago = em_x[ii * _nnei + _nnei - 1];
    const float* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const float* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const float* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const float* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;

    svbool_t ptrue = svptrue_b32();


    // int do_prefetch = prefetch_flag;
    // if(do_prefetch) {
    //   for(int jj = 0; jj < PREFETCH_SIZE; jj++) {
    //     FPTYPE xx = em_x[ii * _nnei + jj]; 
    //     int n_table_idx = 0;
    //     locate_xx(lower, upper, _max, stride0, stride1, xx, n_table_idx);        
    //     const float* TABLE = &_table[n_table_idx * last_layer_size * 6];
    //     for(int kk = 0; kk < last_layer_size * 6 / svcntw(); kk++){
    //       svprfb_vnum(ptrue, TABLE, kk, SV_PLDL2STRM) ;
    //     }
    //   }
    // }

    for (int jj = 0; jj < _nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      float xx = em_x[ii * _nnei + jj]; 
      float xx_next;
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      // if(do_prefetch) {
      //   if(jj + PREFETCH_SIZE >= _nnei) do_prefetch = false;
      //   xx_next = em_x[ii * _nnei + jj + PREFETCH_SIZE];
      //   if(xx_next == ago) do_prefetch = false;
      // }
      // if(do_prefetch) {
      //   int n_table_idx = 0;
      //   locate_xx(lower, upper, _max, stride0, stride1, xx_next, n_table_idx);        
      //   const float* TABLE = &_table[n_table_idx * last_layer_size * 6];
      //   for(int kk = 0; kk < last_layer_size * 6 / svcntw(); kk++){
      //     svprfb_vnum(ptrue, TABLE, kk, SV_PLDL2STRM) ;
      //   }
      // }
      
      float* dy_dem_tmp = &dy_dem[ii * _nnei * 4 + jj * 4];

      svfloat32_t vgard = svdup_f32(0.f);
      svfloat32_t vdy_dem_0 = svdup_f32(0.f);
      svfloat32_t vdy_dem_1 = svdup_f32(0.f);
      svfloat32_t vdy_dem_2 = svdup_f32(0.f);
      svfloat32_t vdy_dem_3 = svdup_f32(0.f);

      assert(last_layer_size % svcntw() == 0);

      svfloat32_t vtwo = svdup_f32(2.f);
      svfloat32_t vthree = svdup_f32(3.f);
      svfloat32_t vfour = svdup_f32(4.f);
      svfloat32_t vfive = svdup_f32(5.f);

      svfloat32_t vnei_sub_jj = svdup_f32((double(_nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);

      svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat32_t v2xx1 = svmul_z(ptrue, vtwo, vxx);
      svfloat32_t v3xx2 = svmul_z(ptrue, vthree, vxx2);
      svfloat32_t v4xx3 = svmul_z(ptrue, vfour, vxx3);
      svfloat32_t v5xx4 = svmul_z(ptrue, vfive, vxx4);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);
      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2){
        svfloat32_t vrr0_0 = svld1(ptrue, dy0 + kk);
        svfloat32_t vrr0_1 = svld1(ptrue, dy0 + kk + svcntw());
        svfloat32_t vrr1_0 = svld1(ptrue, dy1 + kk);
        svfloat32_t vrr1_1 = svld1(ptrue, dy1 + kk + svcntw());
        svfloat32_t vrr2_0 = svld1(ptrue, dy2 + kk);
        svfloat32_t vrr2_1 = svld1(ptrue, dy2 + kk + svcntw());
        svfloat32_t vrr3_0 = svld1(ptrue, dy3 + kk);
        svfloat32_t vrr3_1 = svld1(ptrue, dy3 + kk + svcntw());

        const float* TABLE = &_table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        // double res = a0 + a1 * xx + a2 * xx2 + a3 * xx3 + a4 * xx4 + a5 * xx5;
        svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);

        svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);

        svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);

        svfloat32_t vres_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat32_t vres_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        // a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4
        svfloat32_t tmp9_0 = svmla_z(ptrue, va1_0, va2_0, v2xx1);
        svfloat32_t tmp9_1 = svmla_z(ptrue, va1_1, va2_1, v2xx1);

        svfloat32_t tmp10_0 = svmul_z(ptrue, va3_0, v3xx2);
        svfloat32_t tmp10_1 = svmul_z(ptrue, va3_1, v3xx2);
        svfloat32_t tmp11_0 = svmul_z(ptrue, va4_0, v4xx3);
        svfloat32_t tmp11_1 = svmul_z(ptrue, va4_1, v4xx3);
        svfloat32_t tmp12_0 = svmul_z(ptrue, va5_0, v5xx4);
        svfloat32_t tmp12_1 = svmul_z(ptrue, va5_1, v5xx4);

        svfloat32_t tmp13_0 = svadd_z(ptrue, tmp9_0, tmp10_0);
        svfloat32_t tmp13_1 = svadd_z(ptrue, tmp9_1, tmp10_1);
        svfloat32_t tmp14_0 = svadd_z(ptrue, tmp11_0, tmp12_0);
        svfloat32_t tmp14_1 = svadd_z(ptrue, tmp11_1, tmp12_1);
        svfloat32_t tmp15_0 = svadd_z(ptrue, tmp13_0, tmp14_0); 
        svfloat32_t tmp15_1 = svadd_z(ptrue, tmp13_1, tmp14_1); 

        // dot(ll, rr);
        svfloat32_t tmp16_0 = svmul_z(ptrue, vll0, vrr0_0);
        svfloat32_t tmp16_1 = svmul_z(ptrue, vll0, vrr0_1);
        svfloat32_t tmp17_0 = svmul_z(ptrue, vll1, vrr1_0);
        svfloat32_t tmp17_1 = svmul_z(ptrue, vll1, vrr1_1);
        svfloat32_t tmp18_0 = svmul_z(ptrue, vll2, vrr2_0);
        svfloat32_t tmp18_1 = svmul_z(ptrue, vll2, vrr2_1);
        svfloat32_t tmp19_0 = svmul_z(ptrue, vll3, vrr3_0);
        svfloat32_t tmp19_1 = svmul_z(ptrue, vll3, vrr3_1);
        
        svfloat32_t tmp20_0 = svadd_z(ptrue, tmp16_0, tmp17_0);
        svfloat32_t tmp20_1 = svadd_z(ptrue, tmp16_1, tmp17_1);
        svfloat32_t tmp21_0 = svadd_z(ptrue, tmp18_0, tmp19_0);
        svfloat32_t tmp21_1 = svadd_z(ptrue, tmp18_1, tmp19_1);
        svfloat32_t tmp22_0 = svadd_z(ptrue, tmp20_0, tmp21_0);
        svfloat32_t tmp22_1 = svadd_z(ptrue, tmp20_1, tmp21_1);

        // grad = (a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4 ) * dot(ll, rr);
        svfloat32_t vgard_0 = svmul_z(ptrue, tmp15_0, tmp22_0);
        svfloat32_t vgard_1 = svmul_z(ptrue, tmp15_1, tmp22_1);

        svfloat32_t vres0_0 = svmul_z(ptrue, vres_0, vrr0_0);
        svfloat32_t vres0_1 = svmul_z(ptrue, vres_1, vrr0_1);
        svfloat32_t vres1_0 = svmul_z(ptrue, vres_0, vrr1_0);
        svfloat32_t vres1_1 = svmul_z(ptrue, vres_1, vrr1_1);
        svfloat32_t vres2_0 = svmul_z(ptrue, vres_0, vrr2_0);
        svfloat32_t vres2_1 = svmul_z(ptrue, vres_1, vrr2_1);
        svfloat32_t vres3_0 = svmul_z(ptrue, vres_0, vrr3_0);
        svfloat32_t vres3_1 = svmul_z(ptrue, vres_1, vrr3_1);
        if(unloop){
          vgard_0 = svmul_z(ptrue, vgard_0, vnei_sub_jj);
          vgard_1 = svmul_z(ptrue, vgard_1, vnei_sub_jj);
          vres0_0 = svmul_z(ptrue, vres0_0, vnei_sub_jj);
          vres0_1 = svmul_z(ptrue, vres0_1, vnei_sub_jj);
          vres1_0 = svmul_z(ptrue, vres1_0, vnei_sub_jj);
          vres1_1 = svmul_z(ptrue, vres1_1, vnei_sub_jj);
          vres2_0 = svmul_z(ptrue, vres2_0, vnei_sub_jj);
          vres2_1 = svmul_z(ptrue, vres2_1, vnei_sub_jj);
          vres3_0 = svmul_z(ptrue, vres3_0, vnei_sub_jj);
          vres3_1 = svmul_z(ptrue, vres3_1, vnei_sub_jj);
        }
        vgard = svadd_z(ptrue, vgard, vgard_0);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_0);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_0);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_0);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_0);
        vgard = svadd_z(ptrue, vgard, vgard_1);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_1);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_1);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_1);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_1);
      }

      dy_dem_x[ii * _nnei + jj] = svaddv(ptrue, vgard);
      dy_dem_tmp[0] = svaddv(ptrue, vdy_dem_0);
      dy_dem_tmp[1] = svaddv(ptrue, vdy_dem_1);
      dy_dem_tmp[2] = svaddv(ptrue, vdy_dem_2);
      dy_dem_tmp[3] = svaddv(ptrue, vdy_dem_3);  

      if (unloop) break;
    }
  }   
}
#endif

inline void make_index_range (
    int & idx_start,
    int & idx_end,
    const int & nei_idx, 
    const int & _nnei) 
{
  if (nei_idx < _nnei) {
    idx_start = nei_idx * 4;
    idx_end   = nei_idx * 4 + 4;
  }
  else {
    throw std::runtime_error("should no reach here");
  }
}
#ifdef SPLIT_TYPE_EMBEDDING
void DeepPot::prod_force_a_cpu(
  const FPTYPE * net_deriv, 
  const int type_i,
  const int type_i_in)  {

  const int _ndescrpt = 4 * sel[type_i_in];
  int t_ptr = type_i * ntypes + type_i_in;

  const FPTYPE * env_deriv = r_matrix_deriv[t_ptr]; 

  // compute force of a frame
  for (int i_idx = sec_type_atom[type_i], ii = 0; i_idx < sec_type_atom[type_i+1]; ++i_idx, ++ii) {
    // deriv wrt center atom
    for (int aa = 0; aa < _ndescrpt; ++aa) {
      dforce[i_idx * 3 + 0] -= net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 0];
      dforce[i_idx * 3 + 1] -= net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 1];
      dforce[i_idx * 3 + 2] -= net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 2];
    }
    // deriv wrt neighbors
    for (int jj = sec_a[type_i_in], jj_in = 0; jj < sec_a[type_i_in+1]; ++jj, ++jj_in) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int aa = jj_in * 4; aa < jj_in * 4 + 4; ++aa) {
        dforce[j_idx * 3 + 0] += net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 0];
        dforce[j_idx * 3 + 1] += net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 1];
        dforce[j_idx * 3 + 2] += net_deriv[ii * _ndescrpt + aa] * env_deriv[ii * _ndescrpt * 3 + aa * 3 + 2];
      }
    }
  }

  if((getenv("COMM_DEBUG_FLAG") != nullptr && atoi(getenv("COMM_DEBUG_FLAG")) == 1)){
    std::stringstream ss;

    
    ss << "prod_force_a_cpu  "<< std::endl;
    ss << "net_deriv  "<< std::endl;
    for(int _i = 0; _i < 4 * sel[type_i_in]; _i++) {
      ss << std::fixed << std::setprecision(9) << net_deriv[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    ss << std::endl << "env_deriv  "<< std::endl;
    for(int _i = 0; _i < 4 * sel[type_i_in] * 3; _i++) {
      ss << std::fixed << std::setprecision(9) << env_deriv[_i] << " ";
      if(_i % 100 == 0 && _i != 0) ss << std::endl;
    }
    std::cout << ss.str() << std::endl;
    fflush(stdout);
  
  }

  

  // printf("prod_force_a_cpu from %d to %d \n", sec_type_atom[type_i], sec_type_atom[type_i+1]);
}
#else
void DeepPot::prod_force_a_cpu(
  const FPTYPE * net_deriv, 
  const int type_i,
  const int type_i_in)  {

  const int _ndescrpt = 4 * sel[type_i_in];
  int t_ptr = type_i * ntypes + type_i_in;

  const FPTYPE * env_deriv = r_matrix_deriv; 

  // compute force of a frame
  for (int i_idx = sec_type_atom[type_i], ii = 0; i_idx < sec_type_atom[type_i+1]; ++i_idx, ++ii) {
    // deriv wrt center atom
    for (int aa_in = 0, aa = sec_a[type_i_in]; aa_in < _ndescrpt; ++aa_in, ++aa) {
      dforce[i_idx * 3 + 0] -= net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
      dforce[i_idx * 3 + 1] -= net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
      dforce[i_idx * 3 + 2] -= net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
    }
    // deriv wrt neighbors
    for (int jj = sec_a[type_i_in], jj_in = 0; jj < sec_a[type_i_in+1]; ++jj, ++jj_in) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int aa_in = jj_in * 4, aa = jj*4; aa_in < jj_in * 4 + 4; ++aa_in, ++aa) {
        dforce[j_idx * 3 + 0] += net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
        dforce[j_idx * 3 + 1] += net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
        dforce[j_idx * 3 + 2] += net_deriv[ii * _ndescrpt + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
      }
    }
  }
  // printf("prod_force_a_cpu from %d to %d \n", sec_type_atom[type_i], sec_type_atom[type_i+1]);
}

#endif


void DeepPot::prod_force_a_cpu(
    const FPTYPE * net_deriv, 
    const int ifrom, 
    const int ito,
    const int g_ifrom,
    const int g_ito)  {

  // int _nnei_t = g_ito - g_ifrom;

  // const FPTYPE * env_deriv = r_matrix_deriv;

  // // compute force of a frame
  // for (int i_idx = ifrom, ii = 0; i_idx < ito; ++i_idx, ++ii) {
  //   // deriv wrt center atom
  //   for (int aa_out = g_ifrom * 4, aa_in = 0; aa_out < g_ito * 4; ++aa_out, ++aa_in) {
  //     dforce[i_idx * 3 + 0] -= net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 0];
  //     dforce[i_idx * 3 + 1] -= net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 1];
  //     dforce[i_idx * 3 + 2] -= net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 2];
  //   }

  //   int aa_start, aa_end;
  //   // deriv wrt neighbors
  //   for (int jj_out = g_ifrom, jj_in = 0; jj_out < g_ito; ++jj_out, ++jj_in) {
  //     int j_idx = nlist[i_idx * nnei + jj_out];
  //     if (j_idx < 0) continue;

  //     assert(jj_out < nnei);

  //     for (int aa_out = jj_out * 4, aa_in = jj_in * 4; aa_out < jj_out * 4 + 4; ++aa_out, ++aa_in) {
  //       dforce[j_idx * 3 + 0] += net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 0];
  //       dforce[j_idx * 3 + 1] += net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 1];
  //       dforce[j_idx * 3 + 2] += net_deriv[ii * _nnei_t * 4 + aa_in] * env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + 2];
  //     }
  //   }
  // }
}

void DeepPot::prod_force_a_cpu(
    const FPTYPE * net_deriv,
    const FPTYPE * env_deriv
  )  {


  std::vector<FPTYPE> __force;
  __force.resize(nall * 3, 0);

  memset(__force.data(), 0, sizeof(FPTYPE)* nall * 3);

  // compute force of a frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt center atom
    for (int aa = 0; aa < ndescrpt; ++aa) {
      __force[i_idx * 3 + 0] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
      __force[i_idx * 3 + 1] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
      __force[i_idx * 3 + 2] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
    }
    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int aa_start, aa_end;
      make_index_range (aa_start, aa_end, jj, nnei);
      for (int aa = aa_start; aa < aa_end; ++aa) {
        __force[j_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
        __force[j_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
        __force[j_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
      }
    }
  }

  // print_v(ndescrpt, fmt::format("prod_force net_deriv atom 0"), net_deriv);
  // print_v(ndescrpt * 3, fmt::format("prod_force env_deriv atom 0"), env_deriv);
  // print_v(nnei, fmt::format("prod_force env_deriv atom 0"), nlist);

  // print_v(ndescrpt, fmt::format("prod_force net_deriv atom 1"), net_deriv + 64 * ndescrpt);
  // print_v(ndescrpt * 3, fmt::format("prod_force env_deriv atom 1"), env_deriv + 64 * ndescrpt * 3);
  // print_v(nnei, fmt::format("prod_force env_deriv atom 1"), nlist+64*nnei);


  print_v(nloc * 3, fmt::format("prod_force __force \nn"), __force.data());

}

void DeepPot::prod_virial_a_cpu(
    const FPTYPE * net_deriv, 
    const int ifrom, 
    const int ito,
    const int g_ifrom,
    const int g_ito) {

  // const FPTYPE * env_deriv = r_matrix_deriv; 

  // int _nnei_t = g_ito - g_ifrom;

  // // compute virial of a frame
  // for (int i_idx = ifrom, ii = 0; i_idx < ito; ++i_idx, ii++) {

  //   // deriv wrt neighbors
  //   for (int jj_out = g_ifrom, jj_in = 0; jj_out < g_ito; ++jj_out, ++jj_in) {
  //     int j_idx = nlist[i_idx * nnei + jj_out];
  //     if (j_idx < 0) continue;
  //     assert(jj_out < nnei);

  //     for (int aa_out = jj_out * 4, aa_in = jj_in * 4; aa_out < jj_out * 4 + 4; ++aa_out, ++aa_in) {
  //       FPTYPE pref = -1.0 * net_deriv[ii * _nnei_t * 4 + aa_in];
        
  //       for (int dd0 = 0; dd0 < 3; ++dd0){
  //         for (int dd1 = 0; dd1 < 3; ++dd1){
  //           FPTYPE tmp_v = pref * rij[i_idx * nnei * 3 + jj_out * 3 + dd1] *  env_deriv[i_idx * ndescrpt * 3 + aa_out * 3 + dd0];
  //           dvirial[dd0 * 3 + dd1] -= tmp_v;
  //           // atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
  //         }
  //       }
  //     }
  //   }
  // }  
}

void DeepPot::prod_virial_a_cpu(
    const FPTYPE * net_deriv,
    const FPTYPE * env_deriv
  ) {

  // const FPTYPE * env_deriv = r_matrix_deriv; 

  // for (int ii = 0; ii < nloc; ++ii){
  //   int i_idx = ii;

  //   // deriv wrt neighbors
  //   for (int jj = 0; jj < nnei; ++jj){
  //     int j_idx = nlist[i_idx * nnei + jj];
  //     if (j_idx < 0) continue;
  //     int aa_start, aa_end;
  //     make_index_range (aa_start, aa_end, jj, nnei);
  //     for (int aa = aa_start; aa < aa_end; ++aa) {
  //       double pref = -1.0 * net_deriv[i_idx * ndescrpt + aa];
        
  //       for (int dd0 = 0; dd0 < 3; ++dd0){
  //         for (int dd1 = 0; dd1 < 3; ++dd1){
  //           double tmp_v = pref * rij[i_idx * nnei * 3 + jj * 3 + dd1] *  env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
  //           dvirial[dd0 * 3 + dd1] -= tmp_v;
  //         }
  //       }
  //     }
  //   }
  // }
}

#ifdef SPLIT_TYPE_EMBEDDING
void DeepPot::prod_virial_a_cpu(
  const FPTYPE * net_deriv,
  const int type_i,
  const int type_i_in ) {

  const int _ndescrpt = 4 * sel[type_i_in];
  int t_ptr = type_i * ntypes + type_i_in;

  const FPTYPE * env_deriv = r_matrix_deriv[t_ptr]; 

  // compute force of a frame
  for (int i_idx = sec_type_atom[type_i], ii = 0; i_idx < sec_type_atom[type_i+1]; ++i_idx, ++ii) {
    // deriv wrt neighbors
    for (int jj = sec_a[type_i_in], jj_in = 0; jj < sec_a[type_i_in+1]; ++jj, ++jj_in) {      
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int aa = jj_in*4; aa < jj_in*4+4; ++aa) {
        double pref = -1.0 * net_deriv[ii * _ndescrpt + aa];
        
        for (int dd0 = 0; dd0 < 3; ++dd0){
          for (int dd1 = 0; dd1 < 3; ++dd1) {
            double tmp_v = pref * rij[t_ptr][ii * sel[type_i_in] * 3 + jj_in * 3 + dd1] *  env_deriv[ii * _ndescrpt * 3 + aa * 3 + dd0];
            dvirial[dd0 * 3 + dd1] -= tmp_v;
          }
        }
      }
    }
  }
}
#else
void DeepPot::prod_virial_a_cpu(
  const FPTYPE * net_deriv,
  const int type_i,
  const int type_i_in ) {

  const int _ndescrpt = 4 * sel[type_i_in];
  int t_ptr = type_i * ntypes + type_i_in;

  const FPTYPE * env_deriv = r_matrix_deriv; 

  // compute force of a frame
  for (int i_idx = sec_type_atom[type_i], ii = 0; i_idx < sec_type_atom[type_i+1]; ++i_idx, ++ii) {
    // deriv wrt neighbors
    for (int jj = sec_a[type_i_in], jj_in = 0; jj < sec_a[type_i_in+1]; ++jj, ++jj_in) {      
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int aa_in = jj_in*4, aa=jj*4; aa_in < jj_in*4+4; ++aa_in, ++aa) {
        double pref = -1.0 * net_deriv[ii * _ndescrpt + aa_in];
        
        for (int dd0 = 0; dd0 < 3; ++dd0){
          for (int dd1 = 0; dd1 < 3; ++dd1) {
            double tmp_v = pref * rij[ii * nnei * 3 + jj * 3 + dd1] *  env_deriv[i_idx * ndescrpt * 3 + aa * 3 + dd0];
            dvirial[dd0 * 3 + dd1] -= tmp_v;
          }
        }
      }
    }
  }
}

#endif

void DeepPot::tabulateFusion_v1_sve(int _loc,
  int _nnei,
  FPTYPE* &em_x,
  FPTYPE* &em,
  FPTYPE *out,
  const FPTYPE* _table) {

  const FPTYPE lower   = c_table_info[0];
  const FPTYPE upper   = c_table_info[1];
  const FPTYPE _max    = c_table_info[2];
  const FPTYPE stride0 = c_table_info[3];
  const FPTYPE stride1 = c_table_info[4];

#ifndef HIGH_PREC
  // for every atom, execute a small manual gemm ~
  // FPTYPE * res = new FPTYPE[4 * last_layer_size];
  // #pragma omp parallel for

  #if 0
  for (int ii = 0; ii < _loc; ii++) { // 对loc atom 遍历
    FPTYPE ll[4] = {0};
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1]; // 拿到最后一个邻居
    bool unloop = false; 

    FPTYPE* out0 = out + ii * last_layer_size * 4 + 0 * last_layer_size;
    FPTYPE* out1 = out + ii * last_layer_size * 4 + 1 * last_layer_size;
    FPTYPE* out2 = out + ii * last_layer_size * 4 + 2 * last_layer_size;
    FPTYPE* out3 = out + ii * last_layer_size * 4 + 3 * last_layer_size; // 输出的指针

    for (int jj = 0; jj < _nnei; jj++) {  // 对所有邻居遍历
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj];  // sij
      if (ago == xx) { // 是空
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      // assert(table_idx < 136000);

      for (int kbs = 0; kbs < last_layer_size; kbs+=TABLE_STEP) {
        int kbe = kbs + TABLE_STEP;
        const FPTYPE *table0 = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kbs * TABLE_STRIDE_V1 + TABLE_STEP * 0];
        const FPTYPE *table1 = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kbs * TABLE_STRIDE_V1 + TABLE_STEP * 1];
        // const FPTYPE *table2 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 2];
        // const FPTYPE *table3 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 3];
        // const FPTYPE *table4 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 4];
        // const FPTYPE *table5 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          FPTYPE a0  = table0[kk-kbs]; 
          FPTYPE a1  = table1[kk-kbs]; 
          // FPTYPE a2  = table2[kk-kbs]; 
          // FPTYPE a3  = table3[kk-kbs];
          // FPTYPE a4  = table4[kk-kbs];
          // FPTYPE a5  = table5[kk-kbs];
          // FPTYPE var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx; // 128 次 多项式拟合
          FPTYPE var = a0 + a1 * xx;

          if (unloop) {
            out0[kk] += (_nnei - jj) * var * ll[0];
            out1[kk] += (_nnei - jj) * var * ll[1];
            out2[kk] += (_nnei - jj) * var * ll[2];
            out3[kk] += (_nnei - jj) * var * ll[3];
          }
          else {
            out0[kk] += var * ll[0];
            out1[kk] += var * ll[1];
            out2[kk] += var * ll[2];
            out3[kk] += var * ll[3];
          }
        }
      }

      if (unloop) break;
    }
  }

  #else

  for (int ii = 0; ii < _loc; ii++) {
    svbool_t ptrue = svptrue_b32();

    FPTYPE ll[4] = {0};
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1];
    bool unloop = false; 

    FPTYPE* out0 = out + ii * last_layer_size * 4 + 0 * last_layer_size;
    FPTYPE* out1 = out + ii * last_layer_size * 4 + 1 * last_layer_size;
    FPTYPE* out2 = out + ii * last_layer_size * 4 + 2 * last_layer_size;
    FPTYPE* out3 = out + ii * last_layer_size * 4 + 3 * last_layer_size;

    for (int jj = 0; jj < _nnei; jj++) { 
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];

      FPTYPE xx = em_x[ii * _nnei + jj]; 
      FPTYPE xx_next;
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      assert(last_layer_size % svcntw() == 0);

      svfloat32_t vnei_sub_jj = svdup_f32((float(_nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);
      // svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      // svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      // svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      // svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);

      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2) {
        const FPTYPE* TABLE = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kk * TABLE_STRIDE_V1];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        // svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        // svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        // svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        // svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        // svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        // svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        // svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        // svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11); 

        // svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        // svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        // svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        // svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        // svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        // svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        // svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        // svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        // svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        // svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        // svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        // svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        // svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        // svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        // svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        // svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        // svfloat32_t vvar_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        // svfloat32_t vvar_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat32_t vvar_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t vvar_1 = svmla_z(ptrue, va0_1, va1_1, vxx);

        svfloat32_t vout0_0 = svld1(ptrue, out0 + kk);
        svfloat32_t vout0_1 = svld1(ptrue, out0 + kk + svcntw());
        svfloat32_t vout1_0 = svld1(ptrue, out1 + kk);
        svfloat32_t vout1_1 = svld1(ptrue, out1 + kk + svcntw());
        svfloat32_t vout2_0 = svld1(ptrue, out2 + kk);
        svfloat32_t vout2_1 = svld1(ptrue, out2 + kk + svcntw());
        svfloat32_t vout3_0 = svld1(ptrue, out3 + kk);
        svfloat32_t vout3_1 = svld1(ptrue, out3 + kk + svcntw());

        if(unloop){
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0_);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0_);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1_);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1_);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2_);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2_);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3_);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3_);
        }else{
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3);
        }
        svst1(ptrue, out0 + kk, vout0_0);
        svst1(ptrue, out0 + kk + svcntw(), vout0_1);
        svst1(ptrue, out1 + kk, vout1_0);
        svst1(ptrue, out1 + kk + svcntw(), vout1_1);
        svst1(ptrue, out2 + kk, vout2_0);
        svst1(ptrue, out2 + kk + svcntw(), vout2_1);
        svst1(ptrue, out3 + kk, vout3_0);
        svst1(ptrue, out3 + kk + svcntw(), vout3_1);
      }
      if (unloop) break;
    }
  }

  #endif
  #endif

}


void DeepPot::tabulate_fusion_grad_cpu_packing_v1_sve(
    int _loc, int _nnei,
    FPTYPE *dy_dem_x, 
    FPTYPE *dy_dem,
    const FPTYPE * _table, 
    FPTYPE *em_x, 
    FPTYPE *em, 
    FPTYPE *dy) {

#ifndef HIGH_PREC
  memset(dy_dem_x, 0.0, sizeof(FPTYPE) * _loc * _nnei);
  memset(dy_dem, 0.0, sizeof(FPTYPE) * _loc * _nnei * 4);
  FPTYPE const lower   = c_table_info[0];
  FPTYPE const upper   = c_table_info[1];
  FPTYPE const _max    = c_table_info[2];
  FPTYPE const stride0 = c_table_info[3];
  FPTYPE const stride1 = c_table_info[4];


  #if 0 

  for (int ii = 0; ii < _loc; ii++) {
    FPTYPE ll[4];
    FPTYPE rr[4];
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1];
    const FPTYPE* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const FPTYPE* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const FPTYPE* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const FPTYPE* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;
    for (int jj = 0; jj < _nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      assert(table_idx < 136000);

      FPTYPE* dy_dem_tmp = &dy_dem[ii * _nnei * 4 + jj * 4];

      FPTYPE grad = 0.0;
      FPTYPE dy_dem_0 = 0.0;
      FPTYPE dy_dem_1 = 0.0;
      FPTYPE dy_dem_2 = 0.0;
      FPTYPE dy_dem_3 = 0.0;

      for (int kbs = 0; kbs < last_layer_size; kbs += TABLE_STEP){
        int kbe = kbs + TABLE_STEP;
        const FPTYPE *table0 = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kbs * TABLE_STRIDE_V1 + TABLE_STEP * 0];
        const FPTYPE *table1 = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kbs * TABLE_STRIDE_V1 + TABLE_STEP * 1];
        // const FPTYPE* table2 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 2];
        // const FPTYPE* table3 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 3];
        // const FPTYPE* table4 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 4];
        // const FPTYPE* table5 = &_table[table_idx * last_layer_size * 6 + kbs * 6 + TABLE_STEP * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          rr[0] = dy0[kk];
          rr[1] = dy1[kk];
          rr[2] = dy2[kk];
          rr[3] = dy3[kk];
          FPTYPE a0  = table0[kk-kbs]; 
          FPTYPE a1  = table1[kk-kbs]; 
          // FPTYPE a2  = table2[kk-kbs]; 
          // FPTYPE a3  = table3[kk-kbs];
          // FPTYPE a4  = table4[kk-kbs];
          // FPTYPE a5  = table5[kk-kbs];
          // FPTYPE res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
          FPTYPE res = a0 + a1  * xx;

          if (unloop) {
            grad += (a1) * dot(ll, rr) * (_nnei - jj);
            dy_dem_0 += res * rr[0] * (_nnei - jj);
            dy_dem_1 += res * rr[1] * (_nnei - jj);
            dy_dem_2 += res * rr[2] * (_nnei - jj);
            dy_dem_3 += res * rr[3] * (_nnei - jj);
          }
          else {
            grad += (a1 ) * dot(ll, rr);
            dy_dem_0 += res * rr[0];
            dy_dem_1 += res * rr[1];
            dy_dem_2 += res * rr[2];
            dy_dem_3 += res * rr[3];
          }
        }
      }

      dy_dem_x[ii * _nnei + jj] = grad;
      dy_dem_tmp[0] = dy_dem_0;
      dy_dem_tmp[1] = dy_dem_1;
      dy_dem_tmp[2] = dy_dem_2;
      dy_dem_tmp[3] = dy_dem_3;

      if (unloop) break;
    }
  }

  #else 
  // for every atom, execute a small gemm~
  // float * res = new float[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < _loc; ii++) {
    FPTYPE ll[4];
    FPTYPE rr[4];
    FPTYPE ago = em_x[ii * _nnei + _nnei - 1];
    const FPTYPE* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const FPTYPE* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const FPTYPE* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const FPTYPE* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;

    svbool_t ptrue = svptrue_b32();

    for (int jj = 0; jj < _nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * _nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * _nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * _nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * _nnei * 4 + jj * 4 + 3];
      FPTYPE xx = em_x[ii * _nnei + jj]; 
      FPTYPE xx_next;
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      
      FPTYPE* dy_dem_tmp = &dy_dem[ii * _nnei * 4 + jj * 4];

      svfloat32_t vgard = svdup_f32(0.f);
      svfloat32_t vdy_dem_0 = svdup_f32(0.f);
      svfloat32_t vdy_dem_1 = svdup_f32(0.f);
      svfloat32_t vdy_dem_2 = svdup_f32(0.f);
      svfloat32_t vdy_dem_3 = svdup_f32(0.f);

      // assert(last_layer_size % svcntw() == 0);

      // svfloat32_t vtwo = svdup_f32(2.f);
      // svfloat32_t vthree = svdup_f32(3.f);
      // svfloat32_t vfour = svdup_f32(4.f);
      // svfloat32_t vfive = svdup_f32(5.f);

      svfloat32_t vnei_sub_jj = svdup_f32((double(_nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);

      // svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      // svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      // svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      // svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      // svfloat32_t v2xx1 = svmul_z(ptrue, vtwo, vxx);
      // svfloat32_t v3xx2 = svmul_z(ptrue, vthree, vxx2);
      // svfloat32_t v4xx3 = svmul_z(ptrue, vfour, vxx3);
      // svfloat32_t v5xx4 = svmul_z(ptrue, vfive, vxx4);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);
      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2) {
        svfloat32_t vrr0_0 = svld1(ptrue, dy0 + kk);
        svfloat32_t vrr0_1 = svld1(ptrue, dy0 + kk + svcntw());
        svfloat32_t vrr1_0 = svld1(ptrue, dy1 + kk);
        svfloat32_t vrr1_1 = svld1(ptrue, dy1 + kk + svcntw());
        svfloat32_t vrr2_0 = svld1(ptrue, dy2 + kk);
        svfloat32_t vrr2_1 = svld1(ptrue, dy2 + kk + svcntw());
        svfloat32_t vrr3_0 = svld1(ptrue, dy3 + kk);
        svfloat32_t vrr3_1 = svld1(ptrue, dy3 + kk + svcntw());

        const FPTYPE* TABLE = &_table[table_idx * last_layer_size * TABLE_STRIDE_V1 + kk * TABLE_STRIDE_V1];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        // svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        // svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        // svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        // svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        // svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        // svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        // svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        // svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        // double res = a0 + a1 * xx + a2 * xx2 + a3 * xx3 + a4 * xx4 + a5 * xx5;
        // svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        // svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        // svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        // svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        // svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        // svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        // svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        // svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        // svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        // svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        // svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        // svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        // svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        // svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        // svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        // svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        // svfloat32_t vres_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        // svfloat32_t vres_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat32_t vres_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t vres_1 = svmla_z(ptrue, va0_1, va1_1, vxx);

        // a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4
        // svfloat32_t tmp9_0 = svmla_z(ptrue, va1_0, va2_0, v2xx1);
        // svfloat32_t tmp9_1 = svmla_z(ptrue, va1_1, va2_1, v2xx1);
        // svfloat32_t tmp10_0 = svmul_z(ptrue, va3_0, v3xx2);
        // svfloat32_t tmp10_1 = svmul_z(ptrue, va3_1, v3xx2);
        // svfloat32_t tmp11_0 = svmul_z(ptrue, va4_0, v4xx3);
        // svfloat32_t tmp11_1 = svmul_z(ptrue, va4_1, v4xx3);
        // svfloat32_t tmp12_0 = svmul_z(ptrue, va5_0, v5xx4);
        // svfloat32_t tmp12_1 = svmul_z(ptrue, va5_1, v5xx4);
        // svfloat32_t tmp13_0 = svadd_z(ptrue, tmp9_0, tmp10_0);
        // svfloat32_t tmp13_1 = svadd_z(ptrue, tmp9_1, tmp10_1);
        // svfloat32_t tmp14_0 = svadd_z(ptrue, tmp11_0, tmp12_0);
        // svfloat32_t tmp14_1 = svadd_z(ptrue, tmp11_1, tmp12_1);
        // svfloat32_t tmp15_0 = svadd_z(ptrue, tmp13_0, tmp14_0); 
        // svfloat32_t tmp15_1 = svadd_z(ptrue, tmp13_1, tmp14_1); 

        // dot(ll, rr);
        svfloat32_t tmp16_0 = svmul_z(ptrue, vll0, vrr0_0);
        svfloat32_t tmp16_1 = svmul_z(ptrue, vll0, vrr0_1);
        svfloat32_t tmp17_0 = svmul_z(ptrue, vll1, vrr1_0);
        svfloat32_t tmp17_1 = svmul_z(ptrue, vll1, vrr1_1);
        svfloat32_t tmp18_0 = svmul_z(ptrue, vll2, vrr2_0);
        svfloat32_t tmp18_1 = svmul_z(ptrue, vll2, vrr2_1);
        svfloat32_t tmp19_0 = svmul_z(ptrue, vll3, vrr3_0);
        svfloat32_t tmp19_1 = svmul_z(ptrue, vll3, vrr3_1);
        svfloat32_t tmp20_0 = svadd_z(ptrue, tmp16_0, tmp17_0);
        svfloat32_t tmp20_1 = svadd_z(ptrue, tmp16_1, tmp17_1);
        svfloat32_t tmp21_0 = svadd_z(ptrue, tmp18_0, tmp19_0);
        svfloat32_t tmp21_1 = svadd_z(ptrue, tmp18_1, tmp19_1);
        svfloat32_t tmp22_0 = svadd_z(ptrue, tmp20_0, tmp21_0);
        svfloat32_t tmp22_1 = svadd_z(ptrue, tmp20_1, tmp21_1);

        // grad = (a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4 ) * dot(ll, rr);
        // svfloat32_t vgard_0 = svmul_z(ptrue, tmp15_0, tmp22_0);
        // svfloat32_t vgard_1 = svmul_z(ptrue, tmp15_1, tmp22_1);
        svfloat32_t vgard_0 = svmul_z(ptrue, va1_0, tmp22_0);
        svfloat32_t vgard_1 = svmul_z(ptrue, va1_1, tmp22_1);

        svfloat32_t vres0_0 = svmul_z(ptrue, vres_0, vrr0_0);
        svfloat32_t vres0_1 = svmul_z(ptrue, vres_1, vrr0_1);
        svfloat32_t vres1_0 = svmul_z(ptrue, vres_0, vrr1_0);
        svfloat32_t vres1_1 = svmul_z(ptrue, vres_1, vrr1_1);
        svfloat32_t vres2_0 = svmul_z(ptrue, vres_0, vrr2_0);
        svfloat32_t vres2_1 = svmul_z(ptrue, vres_1, vrr2_1);
        svfloat32_t vres3_0 = svmul_z(ptrue, vres_0, vrr3_0);
        svfloat32_t vres3_1 = svmul_z(ptrue, vres_1, vrr3_1);
        if(unloop){
          vgard_0 = svmul_z(ptrue, vgard_0, vnei_sub_jj);
          vgard_1 = svmul_z(ptrue, vgard_1, vnei_sub_jj);
          vres0_0 = svmul_z(ptrue, vres0_0, vnei_sub_jj);
          vres0_1 = svmul_z(ptrue, vres0_1, vnei_sub_jj);
          vres1_0 = svmul_z(ptrue, vres1_0, vnei_sub_jj);
          vres1_1 = svmul_z(ptrue, vres1_1, vnei_sub_jj);
          vres2_0 = svmul_z(ptrue, vres2_0, vnei_sub_jj);
          vres2_1 = svmul_z(ptrue, vres2_1, vnei_sub_jj);
          vres3_0 = svmul_z(ptrue, vres3_0, vnei_sub_jj);
          vres3_1 = svmul_z(ptrue, vres3_1, vnei_sub_jj);
        }
        vgard = svadd_z(ptrue, vgard, vgard_0);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_0);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_0);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_0);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_0);
        vgard = svadd_z(ptrue, vgard, vgard_1);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_1);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_1);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_1);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_1);
      }

      dy_dem_x[ii * _nnei + jj] = svaddv(ptrue, vgard);
      dy_dem_tmp[0] = svaddv(ptrue, vdy_dem_0);
      dy_dem_tmp[1] = svaddv(ptrue, vdy_dem_1);
      dy_dem_tmp[2] = svaddv(ptrue, vdy_dem_2);
      dy_dem_tmp[3] = svaddv(ptrue, vdy_dem_3);  

      if (unloop) break;
    }
  }   
  #endif
  #endif
}
