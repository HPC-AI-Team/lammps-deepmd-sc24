#include <iostream>
#include <string.h>
#include <iomanip>
#include <limits>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "output.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "modify.h"
#include "fix.h"
#include "citeme.h"
#include <assert.h>
#include <omp.h>
#include "utils.h" 
#include "suffix.h"

#include "pair_deepmd.h"
#include <omp.h>
#include <iostream>
#include <vector>

using namespace LAMMPS_NS;
using namespace std;

static const char cite_user_deepmd_package[] =
	"USER-DEEPMD package:\n\n"
    "@article{Wang_ComputPhysCommun_2018_v228_p178,\n"
    "  author = {Wang, Han and Zhang, Linfeng and Han, Jiequn and E, Weinan},\n"
    "  doi = {10.1016/j.cpc.2018.03.016},\n"
    "  url = {https://doi.org/10.1016/j.cpc.2018.03.016},\n"
    "  year = 2018,\n"
    "  month = {jul},\n"
    "  publisher = {Elsevier {BV}},\n"
    "  volume = 228,\n"
    "  journal = {Comput. Phys. Commun.},\n"
    "  title = {{DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics}},\n"
    "  pages = {178--184}\n"
	"}\n\n";


static int stringCmp(const void *a, const void* b)
{
    char* m = (char*)a;
    char* n = (char*)b;
    int i, sum = 0;

    for(i = 0; i < MPI_MAX_PROCESSOR_NAME; i++)
        if (m[i] == n[i])
            continue;
        else
        {
            sum = m[i] - n[i];
            break;
        }
    return sum;
}

int PairDeepMD::get_node_rank() {
#ifdef _FUGAKU
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank % (48/num_threads);
#else
    char host_name[MPI_MAX_PROCESSOR_NAME];
    memset(host_name, '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    char (*host_names)[MPI_MAX_PROCESSOR_NAME];
    int n, namelen, color, rank, nprocs, myrank;
    size_t bytes;
    MPI_Comm nodeComm;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Get_processor_name(host_name,&namelen);

    bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
    for (int ii = 0; ii < nprocs; ii++) {
        memset(host_names[ii], '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    }
    
    strcpy(host_names[rank], host_name);

    for (n=0; n<nprocs; n++)
        MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
    qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

    color = 0;
    for (n=0; n<nprocs-1; n++)
    {
        if(strcmp(host_name, host_names[n]) == 0)
        {
            break;
        }
        if(strcmp(host_names[n], host_names[n+1]))
        {
            color++;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
    MPI_Comm_rank(nodeComm, &myrank);

    MPI_Barrier(MPI_COMM_WORLD);
    int looprank=myrank;
    // printf (" Assigning device %d  to process on node %s rank %d, OK\n",looprank,  host_name, rank );
    free(host_names);
    return looprank;
#endif
}

static void 
ana_st (double & max, 
	double & min, 
	double & sum, 
	const vector<double> & vec, 
	const int & nloc) 
{
  if (nloc == 0) return;
  max = vec[0];
  min = vec[0];
  sum = vec[0];
  for (unsigned ii = 1; ii < nloc; ++ii){
    if (vec[ii] > max) max = vec[ii];
    if (vec[ii] < min) min = vec[ii];
    sum += vec[ii];
  }
}

PairDeepMD::PairDeepMD(LAMMPS *lmp) 
    : Pair(lmp), ThrOMP(lmp, THR_PAIR)
{
  // if (Pair::lmp->citeme) Pair::lmp->citeme->add(cite_user_deepmd_package);
  if (strcmp(update->unit_style,"metal") != 0) {
    error->all(FLERR,"Pair deepmd requires metal unit, please set it by \"units metal\"");
  }

  restartinfo = 1;
  pppmflag = 1;
  respa_enable = 0;
  writedata = 0;
  cutoff = 0.;
  numb_types = 0;
  out_freq = 0;
  out_each = 0;
  out_rel = 0;
  out_rel_v = 0;
  eps = 0.;
  eps_v = 0.;
  scale = NULL;
  is_restart = false;
  // set comm size needed by this Pair
  comm_reverse = 1;

  num_threads = comm->nthreads;

  lmp_lists.resize(num_threads);

  suffix_flag |= Suffix::OMP;
}

void
PairDeepMD::print_summary(const string pre) const
{
  if (comm->me == 0){
    cout << "Summary of lammps deepmd module ..." << endl;
    cout << pre << ">>> Info of deepmd-kit:" << endl;
    // deep_pot.print_summary(pre);
    cout << pre << ">>> Info of lammps module:" << endl;
    cout << pre << "use deepmd-kit at:  " << STR_DEEPMD_ROOT << endl;
    cout << pre << "source:             " << STR_GIT_SUMM << endl;
    cout << pre << "source branch:      " << STR_GIT_BRANCH << endl;
    cout << pre << "source commit:      " << STR_GIT_HASH << endl;
    cout << pre << "source commit at:   " << STR_GIT_DATE << endl;
    cout << pre << "build float prec:   " << STR_FLOAT_PREC << endl;
    cout << pre << "build with tf inc:  " << STR_TensorFlow_INCLUDE_DIRS << endl;
    cout << pre << "build with tf lib:  " << STR_TensorFlow_LIBRARY << endl;
  }
}

PairDeepMD::~PairDeepMD() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairDeepMD::compute(int eflag, int vflag) {
    ev_init(eflag, vflag);
    #pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
    {

      int tid = omp_get_thread_num();
      if (DEBUG_MSG) utils::logmesg(Pair::lmp,"[info] PairDeepMD::compute start tid {} comm->nthreads {} \n ", tid, comm->nthreads);

      double **x = atom->x;
      double **f = atom->f;
      int *type = atom->type;
      int nlocal = atom->nlocal;
      int nghost = atom->nghost;
      int nall = nlocal + nghost;
      const int inum = list->inum;
      const int nthreads = comm->nthreads;

      int ifrom, ito;

      // double *parallel_dforce = f[0]+nall*tid*3;

      ThrData *thr = fix->get_thr(tid);
      double *parallel_dforce = thr->get_f()[0];

      if(first_time[tid] == 0) {
        if(tid == 0) {

          int _thread_atom_num = (atom->natoms / comm->nprocs) * 4 / nthreads;
          if(_thread_atom_num < 10) _thread_atom_num = 16;

          max_nloc = _thread_atom_num;
          max_nall = nall * 3;
          max_nnei = nnei * 3;

          if(comm->me == 0) utils::logmesg(Pair::lmp, "PairDeepMD param max_nloc {} max_nall {} max_nnei {} \n",  max_nloc,  max_nall, max_nnei);

          for(int _tid = 0; _tid < num_threads; _tid++){
            deep_pots[_tid]->reserve_buffer(max_nloc, max_nall);
          }

          memory->create(dcoord,   atom->nmax * 3,"pair_deepmd:dcoord");
          memory->create(dvirial,   9,"pair_deepmd:dvirial");
          memory->create(thread_dvirial,        nthreads, 9,"pair_deepmd:thread_dvirial");
          memory->create(thread_dforce,         nthreads, max_nall * 3, "pair_deepmd::thread_dforce");
          memory->create(thread_dcoord,         nthreads, max_nall * 3, "pair_deepmd::thread_dcoord");
          memory->create(thread_dtype,          nthreads, max_nall, "pair_deepmd::thread_dtype");
          memory->create(forward_index_map,     nthreads, max_nall, "pair_deepmd::forward_index_map");
          memory->create(backward_index_maps,   nthreads, max_nall, "pair_deepmd::backward_index_maps");
          memory->create(backward_index_size,   nthreads, "pair_deepmd::backward_index_size");      
          memory->create(thread_dener,          nthreads, "pair_deepmd::thread_dener");

          // if(comm->me == 0) utils::logmesg(Pair::lmp, "[INFO] thread_atom_num {} \n", max_nloc);
          memory->create(thread_neigh,          nthreads, max_nnei * max_nloc,"pair_deepmd:thread_neigh");
          memory->create(thread_local_ilist,    nthreads, max_nloc, "pair_deepmd::thread_local_ilist");
          memory->create(thread_local_numneigh, nthreads, max_nloc, "pair_deepmd::thread_local_numneigh");

          memset(thread_neigh[0], 0,          nthreads * max_nnei * max_nloc * sizeof(int));
          memset(thread_local_ilist[0], 0,    nthreads * max_nloc * sizeof(int));
          memset(thread_local_numneigh[0], 0, nthreads * max_nloc * sizeof(int));

          thread_firstneigh  = new int**[nthreads];

          for(int _tid = 0; _tid < num_threads; _tid++) {
            thread_firstneigh[_tid]  = new int*[max_nloc];
            thread_firstneigh[_tid][0] =  thread_neigh[_tid];
          }

          MPI_Barrier(world);
        }
        first_time[tid] = 1;

        #pragma omp barrier
      }

      // if(tid == 11) {
      //   for(int ii = 0; ii < nall; ii++) {
      //     for (int dd = 0; dd < 3; ++dd) {
      //       dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
      //     }
      //   }
      // }

      create_dcoord(nall, tid);

      #pragma omp barrier

      // #pragma omp parallel  
      {
        int idelta_i = nlocal / nthreads;
        int idelta_j = nlocal % nthreads;
        int _bias    = idelta_j == 0 ? 0 : 1;
        
        if(tid >= idelta_j) {
          ifrom = (idelta_i + _bias) * idelta_j + idelta_i * (tid - idelta_j);
          ito = ifrom + idelta_i; 
        } else {
          ifrom = (idelta_i + _bias) * (tid);
          ito = ifrom + idelta_i + _bias; 
        }
        ito = (ito > nlocal) ? nlocal : ito; 

        double dener (0);
      
        // get coord
        int ago = neighbor->ago;

        if (DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD::compute tid {} ifrom {} ito {} nlocal  {} nghost {} ago {}\n", 
                tid,  ifrom, ito, nlocal, nghost, ago);
      
        // thread inner part 
        {
          if(ago == 0) {
            backward_index_size[tid] = 0;

            for(int global_i_index = ifrom; global_i_index < ito; global_i_index++) {
              backward_index_maps[tid][backward_index_size[tid]++] = global_i_index;
            }
          }
            
          int* local_backward_index_map = backward_index_maps[tid];
          InputNlist&  local_lmp_list = lmp_lists[tid];

          int local_nlocal = ago == 0 ? backward_index_size[tid] : local_lmp_list.inum;

          if(ago == 0) {
            if(local_lmp_list.inum != 0) {
              memset(thread_neigh[tid], 0,          512 * local_nlocal * sizeof(int));
              memset(thread_local_ilist[tid], 0,    local_nlocal * sizeof(int));
              memset(thread_local_numneigh[tid], 0, local_nlocal * sizeof(int));
              memset(thread_firstneigh[tid], 0,     local_nlocal * sizeof(int*));
            }

            // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} delte local_lmp_list local_nlocal  {}\n",  tid,  local_nlocal);

            for(int i = 0; i < nall; i++){
              forward_index_map[tid][i] = -1;
            }

            int* local_ilist    = thread_local_ilist[tid];
            int* local_numneigh = thread_local_numneigh[tid];
            int** firstneigh    = thread_firstneigh[tid];

            int total_neigh=0;
            for(int local_i_index = 0; local_i_index < local_nlocal; local_i_index++) {
              int global_i_index = local_backward_index_map[local_i_index];
              local_ilist[local_i_index] = local_i_index;
              local_numneigh[local_i_index] = list->numneigh[global_i_index];
              total_neigh += local_numneigh[local_i_index];
              forward_index_map[tid][global_i_index] = local_i_index;
            }
            if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} total_neigh {} \n",  tid,  total_neigh);
            int* neigh = thread_neigh[tid];

            int local_nall=local_nlocal;
            int cur_neigh = 0;
            for(int local_i_index = 0; local_i_index < local_nlocal;local_i_index++) {
              int global_i_index = local_backward_index_map[local_i_index];
              firstneigh[local_i_index] = &neigh[cur_neigh];

              if(max_nnei < local_numneigh[local_i_index]) error->one(FLERR, "[ERROR] max_nnei < local_numneigh[local_i_index] {} {} ", max_nnei, local_numneigh[local_i_index]);

              for(int nei_iter = 0; nei_iter < local_numneigh[local_i_index];nei_iter++ ) {
                int global_j_idx = list->firstneigh[global_i_index][nei_iter];
                int local_j_index = forward_index_map[tid][global_j_idx];
                if(local_j_index == -1){
                  local_j_index = local_nall++;
                  forward_index_map[tid][global_j_idx] = local_j_index;
                  local_backward_index_map[backward_index_size[tid]++] = global_j_idx;
                }
                firstneigh[local_i_index][nei_iter] = local_j_index;
              }
              cur_neigh += local_numneigh[local_i_index];
            }

            if(max_nnei*max_nloc <= cur_neigh)   error->one(FLERR, "[ERROR] max_nnei*max_nloc < cur_neigh   {} {} ", max_nnei*max_nloc, cur_neigh);

            local_lmp_list.inum = local_nlocal;
            local_lmp_list.ilist = local_ilist;
            local_lmp_list.numneigh = local_numneigh;
            local_lmp_list.firstneigh = firstneigh;
            assert(cur_neigh == total_neigh);
          }

          // if(DEBUG_MSG) print_v(local_nlocal, fmt::format("local_lmp_list : "), local_lmp_list.ilist);
          if(DEBUG_MSG) print_v(local_nlocal, fmt::format("local_lmp_list : {}", local_lmp_list.inum), local_lmp_list.numneigh);

          int local_nall = backward_index_size[tid];
          int local_nghost = local_nall - local_nlocal;

          if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} build local_nall local_nlocal {} local_nall {} \n",  tid,  local_nlocal, local_nall);

          // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} finish preprae thread_dtype {} {} {}\n", 
          //       tid, local_nlocal, local_nghost, local_nall);

          // Pair::lmp->parral_barrier(12, tid);  

          if(ago == 0) {
            for(int local_index = 0;local_index < local_nall;local_index++) {
              int global_index = local_backward_index_map[local_index];
              thread_dtype[tid][local_index] = type[global_index] - 1;
            }
          }

          // Pair::lmp->parral_barrier(12, tid);      

          for(int local_index = 0;local_index < local_nall;local_index++) {
            int global_index = local_backward_index_map[local_index];
            thread_dcoord[tid][local_index*3+0] = dcoord[global_index*3+0];
            thread_dcoord[tid][local_index*3+1] = dcoord[global_index*3+1];
            thread_dcoord[tid][local_index*3+2] = dcoord[global_index*3+2];
          }

          // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} finish preprae thread_dcoord {} {} {}\n", 
          //       tid, local_nlocal, local_nghost, local_nall);

          // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} begin deep_pots[tid]->compute\n", tid);

          // if(DEBUG_MSG) if(tid == 0) print_v(local_nlocal * 3, fmt::format("parallel_nlocal : "), thread_dcoord[tid]);

          if(max_nall <= local_nall) error->one(FLERR, "[ERROR] max_nall < local_nall {} {} ", max_nall, local_nall);
          if(max_nloc <= local_nlocal)   error->one(FLERR, "[ERROR] max_nloc < local_nlocal   {} {} ", max_nloc, local_nlocal);

          #pragma omp barrier

          // print_v(local_nall * 3, fmt::format("parallel_nlocal : "), thread_dcoord[tid]);


          deep_pots[tid]->compute (thread_dener[tid], thread_dforce[tid], thread_dvirial[tid], thread_dcoord[tid], thread_dtype[tid], local_nghost, local_nlocal, local_lmp_list, ago);

      // #ifdef HIGH_PREC
      //     deep_pots[tid]->compute (thread_dener[tid], thread_dforce[tid], thread_dvirial[tid], thread_dcoord[tid], thread_dtype[tid], local_nghost, local_nlocal, local_lmp_list, ago);
      // #else 
      //     deep_pots[tid]->compute (thread_dener[tid], thread_dforce[tid], thread_dvirial[tid], thread_dcoord[tid], thread_dtype[tid], local_nghost, local_nlocal, local_lmp_list, ago);
      // #endif


          // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} finish deep_pots[tid]->compute\n", tid);
          // Pair::lmp->parral_barrier(12, tid);  
        
          for(int local_index = 0; local_index < local_nall; local_index++) {
            int global_index = local_backward_index_map[local_index];
            parallel_dforce[global_index * 3 + 0] = thread_dforce[tid][local_index * 3 + 0];
            parallel_dforce[global_index * 3 + 1] = thread_dforce[tid][local_index * 3 + 1];
            parallel_dforce[global_index * 3 + 2] = thread_dforce[tid][local_index * 3 + 2];
          }

          // if(tid == 2) print_v(local_nall, fmt::format("parallel_dforce type_: "), thread_dforce[2].data());

          // if(DEBUG_MSG) utils::logmesg(Pair::lmp, "PairDeepMD tid {} finish force convert \n", tid);
        }

        data_reduce_thr_threadpool_param(&(f[0][0]), nall, nthreads, 3, tid, scale[1][1], Pair::lmp);
        
        #pragma omp barrier

        if(tid == 0) {
          // for(int ii = 0; ii < 12; ii++)
          //   if(DEBUG_MSG) print_v(nall, fmt::format("parallel_dforce {} : ", ii), parallel_dforce[ii].data());

          // for(int _t = 1; _t < nthreads; _t++) {
          //   for(int ii = 0 ; ii < nall; ii++) {
          //     f[ii][0] += f[_t*nall+ii][0];
          //     f[ii][1] += f[_t*nall+ii][1];
          //     f[ii][2] += f[_t*nall+ii][2];
          //   }
          // }

          // if(DEBUG_MSG) print_v(nlocal, fmt::format("parallel_dforce : "), f[0]);

          // get force
          // for (int ii = 0; ii < nall; ++ii) {
          //   for (int dd = 0; dd < 3; ++dd) {
          //     f[ii][dd] *= scale[1][1];
          //   }
          // }

          // print_v(nall, fmt::format("parallel_dforce scale: "), f[0]);
          
          // // accumulate energy and virial

          if (eflag) {
            for(int i = 0;i<nthreads;i++){
              dener += thread_dener[i];
            }
            eng_vdwl += scale[1][1] * dener;
          }

          memset(dvirial, 0, sizeof(double) * 9);
          if (vflag) {
            for(int i = 0;i<nthreads;i++){
              dvirial[0] += thread_dvirial[i][0];
              dvirial[1] += thread_dvirial[i][1];
              dvirial[2] += thread_dvirial[i][2];
              dvirial[3] += thread_dvirial[i][3];
              dvirial[4] += thread_dvirial[i][4];
              dvirial[5] += thread_dvirial[i][5];
              dvirial[6] += thread_dvirial[i][6];
              dvirial[7] += thread_dvirial[i][7];
              dvirial[8] += thread_dvirial[i][8];
            }

            virial[0] += 1.0 * dvirial[0] * scale[1][1];
            virial[1] += 1.0 * dvirial[4] * scale[1][1];
            virial[2] += 1.0 * dvirial[8] * scale[1][1];
            virial[3] += 1.0 * dvirial[3] * scale[1][1];
            virial[4] += 1.0 * dvirial[6] * scale[1][1];
            virial[5] += 1.0 * dvirial[7] * scale[1][1];

            // print_v(6, fmt::format("virial type_: "), virial);
          }

          // memset(f[0], 0, nlocal * sizeof(double) * 3);
          // memset(atom->v[0], 0, nlocal * sizeof(double) * 3);
          // memset(dvirial, 0, sizeof(double) * 9);
        }
      } // end omp

      // for(int tid = 0; tid < nthreads; tid++){
      //   print_v(6, fmt::format("virial type_: "), virial);
      // }

    }


}

// void PairDeepMD::compute(int eflag, int vflag) {
  
// }


void PairDeepMD::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(scale,n+1,n+1,"pair:scale");

  for (int i = 1; i <= n; i++){
    for (int j = i; j <= n; j++){
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types; ++i) {
    if (i > n) continue;
    for (int j = i; j <= numb_types; ++j) {
      if (j > n) continue;
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}


static bool 
is_key (const string& input) 
{
  vector<string> keys ;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("ttm");
  keys.push_back("atomic");
  keys.push_back("relative");
  keys.push_back("relative_v");

  for (int ii = 0; ii < keys.size(); ++ii){
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}


void PairDeepMD::settings(int narg, char **arg)
{
  if (comm->me == 0) utils::logmesg(Pair::lmp,"[info] airDeepMD::settings start \n ");


  #ifdef __ARM_FEATURE_SVE
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] __ARM_FEATURE_SVE on \n");
  #else 
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] __ARM_FEATURE_SVE off \n");
  #endif
  #ifdef HIGH_PREC
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] HIGH_PREC on \n");
  #else 
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] HIGH_PREC off \n");
  #endif
  #ifdef T_FLOAT_16
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] T_FLOAT_16 on \n");
  #else 
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] T_FLOAT_16 off \n");
  #endif
    #ifdef OPT_CBLAS
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] OPT_CBLAS on \n");
  #else 
      if (comm->me == 0) utils::logmesg(Pair::lmp, "[info] OPT_CBLAS off \n");
  #endif

  // if (narg != 2) error->all(FLERR, "Illegal pair_style command");

  int iarg = 0;

  cutoff = rcut = utils::numeric(FLERR, arg[iarg++], false, Pair::lmp);
  rcut_smth = utils::numeric(FLERR, arg[iarg++], false, Pair::lmp);
  numb_types = utils::numeric(FLERR, arg[iarg++], false, Pair::lmp);

  for(int i = 0; i < numb_types; i++) {
    sel.push_back(utils::numeric(FLERR, arg[iarg++], false, Pair::lmp));
  }

  nnei = 0;
  for(auto &i : sel) nnei += i;

  graph_path = arg[iarg++];

  if (comm->me == 0) utils::logmesg(Pair::lmp, fmt::format("[info] graph_path  : {} \n", graph_path.c_str()));


  vector<FPTYPE > dbox (9, 0) ;

  // get box
  dbox[0] = domain->h[0];	// xx
  dbox[4] = domain->h[1];	// yy
  dbox[8] = domain->h[2];	// zz
  dbox[7] = domain->h[3];	// zy
  dbox[6] = domain->h[4];	// zx
  dbox[3] = domain->h[5];	// yx

  deep_pot = new DeepPot(Pair::lmp);
  deep_pot->init (rcut, rcut_smth, numb_types, sel, dbox, graph_path);

  deep_pots = new DeepPot*[num_threads];
  for(int i = 0; i < num_threads; i++){
    deep_pots[i] =  new DeepPot(Pair::lmp);
    deep_pots[i]->init(deep_pot, i);
  }

  // Pair::lmp->deep_pots = deep_pots;

  if (comm->me == 0) utils::logmesg(Pair::lmp, fmt::format("[info] num_threads : {} \n", num_threads));
  if (comm->me == 0) utils::logmesg(Pair::lmp, fmt::format("[info] numb_types  : {} \n", numb_types));
  if (comm->me == 0) utils::logmesg(Pair::lmp, fmt::format("[info] narg        : {} \n", narg));

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  
  if (comm->me == 0){
    string pre = "  ";
    cout << pre << ">>> Info of model(s):" << endl << pre << "using " << setw(3) << 1 << " model(s): ";
    if (narg == 1) {
      cout << arg[0] << " ";
    }
    else {
      // for (int ii = 0; ii < models.size(); ++ii){
      // 	cout << models[ii] << " ";
      // }
    }
    cout << endl
        << pre << "rcut in model:      " << cutoff << endl
        << pre << "ntypes in model:    " << numb_types << endl;
  }
  
  comm_reverse = 1 * 3;
  all_force.resize(1);
}

void PairDeepMD::read_restart(FILE *)
{
  is_restart = true;
}

void PairDeepMD::write_restart(FILE *)
{
  // pass
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMD::coeff(int narg, char **arg)
{
  if (!allocated) {
    allocate();
  }

  int n = atom->ntypes;
  int ilo,ihi,jlo,jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  if (narg == 2) {
    utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
    utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
    if (ilo != 1 || jlo != 1 || ihi != n || jhi != n) {
      error->all(FLERR,"deepmd requires that the scale should be set to all atom types, i.e. pair_coeff * *.");
    }
  }  
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
      scale[i][j] = 1.0;
      if (i > numb_types || j > numb_types) {
        char warning_msg[1024];
        sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
        error->warning(FLERR, warning_msg);
      }
    }
  }
}


void PairDeepMD::init_style()
{
  if (out_each == 1){
    int ntotal = atom->natoms;
    int nprocs = comm->nprocs;
    // memory->create(counts, nprocs, "deepmd:counts");
    // memory->create(displacements, nprocs, "deepmd:displacements");
    // memory->create(stdfsend,ntotal,"deepmd:stdfsendall");
    // memory->create(stdfrecv,ntotal,"deepmd:stdfrecvall");
    // memory->create(tagsend,ntotal,"deepmd:tagsendall");
    // memory->create(tagrecv,ntotal,"deepmd:tagrecvall");
  }

  neighbor->add_request(this,NeighConst::REQ_FULL);
  // comm->full_flag = 1;
  // neighbor->add_request(this,0);
}


double PairDeepMD::init_one(int i, int j)
{
  if (i > numb_types || j > numb_types) {
    char warning_msg[1024];
    sprintf(warning_msg, "Interaction between types %d and %d is set with deepmd, but will be ignored.\n Deepmd model has only %d types, it only computes the mulitbody interaction of types: 1-%d.", i, j, numb_types, numb_types);
    error->warning(FLERR, warning_msg);
  }

  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];

  return cutoff;
}


/* ---------------------------------------------------------------------- */

int PairDeepMD::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = all_force[0][3*i+0];
    buf[m++] = all_force[0][3*i+1];
    buf[m++] = all_force[0][3*i+2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int dd = 0; dd < 1; ++dd){
      all_force[dd][3*j+0] += buf[m++];
      all_force[dd][3*j+1] += buf[m++];
      all_force[dd][3*j+2] += buf[m++];
    }
  }

}

void *PairDeepMD::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cutoff;
  }
  if (strcmp(str,"scale") == 0) {
    dim = 2;
    return (void *) scale;
  }
  return NULL;
}


void PairDeepMD::create_dcoord(int nall, int tid) {

  const int nvals = nall;
  const int idelta = nvals / num_threads + 1;
  const int ifrom = tid * idelta;
  const int ito = ((ifrom + idelta) > nvals) ? nvals : (ifrom + idelta);

  double **x = atom->x;

  for(int ii = ifrom; ii < ito; ii++) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }
}
