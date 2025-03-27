#include <iostream>
#include <iomanip>
#include <limits>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "fix.h"
#include "fix_dplr.h"
#include "memory.h"
// #include "pppm_dplr.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

static bool 
is_key (const string& input) 
{
  vector<string> keys ;
  keys.push_back("type_associate");
  keys.push_back("bond_type");
  keys.push_back("efield");
  for (int ii = 0; ii < keys.size(); ++ii){
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}


FixDPLR::FixDPLR(LAMMPS *lmp, int narg, char **arg) 
    :Fix(lmp, narg, arg), 
     efield_force_flag(0)
{
  // lammps/lammps#2560
  energy_global_flag = 1;
  virial_global_flag = 1;

  efield[0] = efield[1] = efield[2] = 0;

  if (strcmp(update->unit_style,"metal") != 0) {
    error->all(FLERR,"Pair deepmd requires metal unit, please set it by \"units metal\"");
  }
  
  int iarg = 3;
  vector<int> map_vec;
  bond_type.clear();
  while (iarg < narg) {
    if (! is_key(arg[iarg])) {
      error->all(FLERR,"Illegal pair_style command\nwrong number of parameters\n");
    }
    else if (string(arg[iarg]) == string("efield")) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix adapt command, efield should be provided 3 float numbers");
      efield[0] = atof(arg[iarg+1]);
      efield[1] = atof(arg[iarg+2]);
      efield[2] = atof(arg[iarg+3]);
      iarg += 4;
    }
    else if (string(arg[iarg]) == string("type_associate")) {
      int iend = iarg+1;
      while (iend < narg && (! is_key(arg[iend]) )) {
        map_vec.push_back(atoi(arg[iend])-1);
        iend ++;
      }
      iarg = iend;
    }
    else if (string(arg[iarg]) == string("bond_type")) {
      int iend = iarg+1;
      while (iend < narg && (! is_key(arg[iend]) )) {
        bond_type.push_back(atoi(arg[iend])-1);
        iend ++;
      }
      sort(bond_type.begin(), bond_type.end());
      iarg = iend;
    }
    else {
      break;
    }
  }
  assert(map_vec.size() % 2 == 0), "number of ints provided by type_associate should be even";
  for (int ii = 0; ii < map_vec.size()/2; ++ii){
    type_asso[map_vec[ii*2+0]] = map_vec[ii*2+1];
    bk_type_asso[map_vec[ii*2+1]] = map_vec[ii*2+0];
  }

  pair_deepmd = (PairDeepMD *) force->pair_match("deepmd/omp",1);
  if (!pair_deepmd) {
    error->all(FLERR,"pair_style deepmd should be set before this fix\n");
  }
  pppm_dplr = (PPPMDPLR*) force->kspace_match("pppm/dplr", 1);
  if (!pppm_dplr) {
    error->all(FLERR,"kspace_style pppm/dplr should be set before this fix\n");
  }

  deep_pots_dipole = pair_deepmd->deep_pots_dipole;

  // dpt.init(model, 0, "dipole_charge");
  // dtm.init(model, 0, "dipole_charge");
  
  dipole_sel_type = deep_pots_dipole[0]->dipole_sel_type;
  ntypes = pair_deepmd->numb_types;

  if(comm->me == 0) utils::logmesg_arry(lmp, "[INFO] fix_dplr map_vec", map_vec.data(), map_vec.size(), 1);
  if(comm->me == 0) utils::logmesg_arry(lmp, "[INFO] fix_dplr dipole_sel_type", dipole_sel_type.data(), dipole_sel_type.size(), 1);
  
  sort(dipole_sel_type.begin(), dipole_sel_type.end());
  dpl_type.clear();
  for (int ii = 0; ii < dipole_sel_type.size(); ++ii){
    dpl_type.push_back(type_asso[dipole_sel_type[ii]]);
  }
  if(comm->me == 0) utils::logmesg_arry(lmp, "[INFO] fix_dplr dpl_type", dpl_type.data(), dpl_type.size(), 1);
  // set comm size needed by this fix
  comm_reverse = 3;
}

int FixDPLR::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

void FixDPLR::init()
{
}

void FixDPLR::setup(int vflag)
{
  if (vflag) {
    v_setup(vflag);
  }
  else {
    evflag = 0;
  }
}

void FixDPLR::setup_pre_force(int vflag){
  int max_nloc, max_nall;
  atom->setMaxNum(max_nloc, max_nall);

  if(DEBUG_MSG) utils::logmesg(lmp, "[INFO] setup_pre_force param max_nloc {} max_nall {}\n",  max_nloc,  max_nall);

  memory->create(dvirial,               9,"fix_dplr:dvirial");
  memory->create(thread_dvirial,        comm->nthreads, 9,"fix_dplr:thread_dvirial");
  memory->create(dipole_recd,           max_nloc * 3, "fix_dplr::thread_dener");
  memory->create(thread_dipole_recd,    comm->nthreads, max_nloc * 3 , "fix_dplr::thread_dipole_recd");
  
  bd_pairs.resize(max_nloc);
  memory->create(bd_idx,    max_nall, "fix_dplr::bd_idx");
  pre_force(vflag);
};


void
FixDPLR::init_valid_pairs()
{  
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  nbd_pairs = 0;

  for (int ii = 0; ii < nbondlist; ++ii) {
    int idx0=-1, idx1=-1;
    if ( ! binary_search(bond_type.begin(), bond_type.end(), bondlist[ii][2] - 1) ){
      continue;
    }
    if (binary_search(dipole_sel_type.begin(), dipole_sel_type.end(), atom->type[bondlist[ii][0]]-1) && 
          binary_search(dpl_type.begin(), dpl_type.end(), atom->type[bondlist[ii][1]]-1)
	  ){
      idx0 = bondlist[ii][0];
      idx1 = bondlist[ii][1];
    }
    else if (binary_search(dipole_sel_type.begin(), dipole_sel_type.end(), atom->type[bondlist[ii][1]]-1)  &&
	     binary_search(dpl_type.begin(), dpl_type.end(), atom->type[bondlist[ii][0]]-1)
	  ){
      idx0 = bondlist[ii][1];
      idx1 = bondlist[ii][0];
    }
    else {
      error->all(FLERR, "find a bonded pair the types of which are not associated");
    }
    if ( ! (idx0 < nlocal && idx1 < nlocal) ){
      error->all(FLERR, "find a bonded pair that is not on the same processor, something should not happen");
    }
    bd_pairs[nbd_pairs].first = idx0;
    bd_pairs[nbd_pairs].second = idx1;
    nbd_pairs++;
  }
  if(DEBUG_MSG) {
    std::string tmp;
     tmp += "[info] bd_pairs ";
     for(int i = 0; i < nbd_pairs ;i++) {
        tmp += fmt::format("  {}:{}", bd_pairs[i].first, bd_pairs[i].second);
        if(i != 0 && (i % 100 == 0)) tmp += "\n      ";
     } 
     utils::logmesg(lmp, "{} \n", tmp);
  }

  for(int i = 0; i < nall; i++) {
    bd_idx[i] = -1;
  }
  for (int i = 0; i < nbd_pairs; ++i){
    bd_idx[bd_pairs[i].first] = bd_pairs[i].second;
  }
}

void FixDPLR::post_integrate()
{
  // double **x = atom->x;
  double **v = atom->v;
  // int *type = atom->type;
  // int nlocal = atom->nlocal;
  // int nghost = atom->nghost;
  // int nall = nlocal + nghost;

  // vector<pair<int,int> > validbd_pairs;
  // get_validbd_pairs(validbd_pairs);  

  for (int ii = 0; ii < nbd_pairs; ++ii){
    int idx0 = bd_pairs[ii].first;
    int idx1 = bd_pairs[ii].second;
    for (int dd = 0; dd < 3; ++dd){
      v[idx1][dd] = v[idx0][dd] ;
    }
  }
}

void FixDPLR::pre_force(int vflag)
{

    // double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  if(neighbor->ago == 0) {
    init_valid_pairs();
    atom->nlocal_real = 0;
    for(int i = 0; i < nlocal; i++) {
      if(atom->type[i] <= ntypes)  {
        atom->nlocal_real++;
      }
    }
  }
}


void FixDPLR::post_force(int vflag)
{
  if(DEBUG_MSG) utils::logmesg(lmp, "\n\n*************** post_force ntimestep {} *****************\n", update->ntimestep);

  if (vflag) {
    v_setup(vflag);
  }
  else {
    evflag = 0;
  }
  if (vflag_atom) {
    error->all(FLERR,"atomic virial calculation is not supported by this fix\n");
  }

  // printf("FixDPLR vflag %d \n", vflag); fflush(stdout);
  // printf("FixDPLR evflag %d \n", evflag); fflush(stdout);

  FPTYPE *fele = pppm_dplr->fele;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // revise force and virial according to efield
  double * q = atom->q;

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force fele after \n"),fele, 3*nlocal, 1 );

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();

    // deep_pots_dipole[tid]->splite_atom();

    // #pragma omp barrier

    memset(thread_dipole_recd[tid], 0, sizeof(double) * nlocal * 3);

    deep_pots_dipole[tid]->shuffer_dextf(bd_idx, fele);
    double *parallel_dforce = pppm_dplr->f_lr[0] + tid * nall * 3;
    memset(parallel_dforce, 0, sizeof(double) * nall * 3);
    deep_pots_dipole[tid]->compute_dipole(thread_dipole_recd[tid], parallel_dforce, thread_dvirial[tid]);

    pair_deepmd->force_reduce(&(pppm_dplr->f_lr[0][0]), nall, comm->nthreads, 3, tid, 1.);

    #pragma omp barrier

  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force f \n"), pppm_dplr->f_lr[0], 3*nlocal, 1 );
  
  // self correction of bonded force
  for (int ii = 0; ii < nbd_pairs; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      pppm_dplr->f_lr[bd_pairs[ii].first][dd] += fele[bd_pairs[ii].second*3+dd];
    }    
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force add bonded \n"), pppm_dplr->f_lr[0], 3*nlocal, 1 );
  
  
  for (int ii = 0; ii < nlocal; ++ii) {
    if(atom->type[ii] > ntypes) continue;           
    for (int dd = 0; dd < 3; ++dd){
      pppm_dplr->f_lr[ii][dd] += fele[ii*3+dd];
    }    
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force add fele \n"), pppm_dplr->f_lr[0], 3*nlocal, 1 );

  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] post_force dfcorr \n"), pppm_dplr->f_lr[0], atom->nlocal * 3, 1);
  
  comm->reverse_comm(this, 3);
  
  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] post_force dfcorr2 \n"), pppm_dplr->f_lr[0], atom->nlocal * 3, 1);

  double ** f = atom->f;
  for (int ii = 0; ii < nlocal; ++ii){
    for(int dd = 0; dd < 3; ++dd){
      f[ii][dd] += pppm_dplr->f_lr[ii][dd];
    }
  }

  
  if (vflag) {
    memset(dipole_recd, 0, sizeof(double) * nlocal * 3);

    for(int ii = 0; ii < comm->nthreads; ii++){
      for(int jj = 0; jj < 3 * nlocal; jj++) 
        dipole_recd[jj] += thread_dipole_recd[ii][jj];
    }  

    // for (int ii = 0; ii < nbd_pairs; ++ii){
    //   int idx0 = bd_pairs[ii].first;
    //   int idx1 = bd_pairs[ii].second;
    //   for (int dd = 0; dd < 3; ++dd) {
    //     dipole_recd[idx0*3+dd] = thread_dipole_recd[0][idx0*3+dd];
    //   }
    // }

    for(int ii = 0; ii < comm->nthreads; ii++){
      if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force thread_dipole_recd tid {} ", ii),thread_dipole_recd[ii], 3*nlocal, 1 );
    }

    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force dipole_recd \n"),dipole_recd, 3*nlocal, 1 );

    memset(dvirial, 0, sizeof(double) * 9);
    
    for(int ii = 0;ii < comm->nthreads; ii++){
      for(int jj = 0; jj < 9; jj++) 
        dvirial[jj] += thread_dvirial[ii][jj];
    }  
    if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("fix post_force dvirial \n"),dvirial,9, 1 );

    for (int ii = 0; ii < nbd_pairs; ++ii){
      int idx0 = bd_pairs[ii].first;
      int idx1 = bd_pairs[ii].second;
      for (int dd0 = 0; dd0 < 3; ++dd0){
        for (int dd1 = 0; dd1 < 3; ++dd1){
          dvirial[dd0*3+dd1] -= fele[idx1*3+dd0] * dipole_recd[idx0*3+dd1];
        }
      }
    }

    if(DEBUG_MSG) utils::logmesg_arry(lmp,fmt::format("[info] post_force dvirial 2 \n"), dvirial, 9, 1);

    double vv[6] = {0.0};
    vv[0] += dvirial[0];
    vv[1] += dvirial[4];
    vv[2] += dvirial[8];
    vv[3] += dvirial[3];
    vv[4] += dvirial[6];
    vv[5] += dvirial[7];
    v_tally(0, vv);  
    // print_v(6, fmt::format("virial type_: "), virial);
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp,fmt::format("[info] post_force fix  virial finial \n"),virial, 6, 1);

  if(DEBUG_MSG) utils::logmesg(lmp, "\n********************************\n");
}


int FixDPLR::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = pppm_dplr->f_lr[0][3*i+0];
    buf[m++] = pppm_dplr->f_lr[0][3*i+1];
    buf[m++] = pppm_dplr->f_lr[0][3*i+2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixDPLR::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    pppm_dplr->f_lr[0][3*j+0] += buf[m++];
    pppm_dplr->f_lr[0][3*j+1] += buf[m++];
    pppm_dplr->f_lr[0][3*j+2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   return energy added by fix
------------------------------------------------------------------------- */

double FixDPLR::compute_scalar(void)
{
  if (efield_force_flag == 0) {
    MPI_Allreduce(&efield_fsum[0],&efield_fsum_all[0],4,MPI_DOUBLE,MPI_SUM,world);
    efield_force_flag = 1;
  }
  return efield_fsum_all[0];
}

/* ----------------------------------------------------------------------
   return total extra force due to fix
------------------------------------------------------------------------- */

double FixDPLR::compute_vector(int n)
{
  if (efield_force_flag == 0) {
    MPI_Allreduce(&efield_fsum[0],&efield_fsum_all[0],4,MPI_DOUBLE,MPI_SUM,world);
    efield_force_flag = 1;
  }
  return efield_fsum_all[n+1];
}
