// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "verlet.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "timer.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

Verlet::Verlet(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) {}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void Verlet::init()
{
  Integrate::init();

  // warn if no fixes doing time integration

  bool do_time_integrate = false;
  for (const auto &fix : modify->get_fix_list())
    if (fix->time_integrate) do_time_integrate = true;

  if (!do_time_integrate && (comm->me == 0))
    error->warning(FLERR,"No fixes with time integration, atoms won't move");

  // virial_style:
  // VIRIAL_PAIR if computed explicitly in pair via sum over pair interactions
  // VIRIAL_FDOTR if computed implicitly in pair by
  //   virial_fdotr_compute() via sum over ghosts

  if (force->newton_pair) virial_style = VIRIAL_FDOTR;
  else virial_style = VIRIAL_PAIR;

  // setup lists of computes for global and per-atom PE and pressure

  ev_setup();

  // detect if fix omp is present for clearing force arrays

  if (modify->get_fix_by_id("package_omp")) external_force_clear = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void Verlet::setup(int flag)
{

  if(DEBUG_MSG) utils::logmesg(lmp, "steup .........\n");

  if (comm->me == 0 && screen) {
    fputs("Setting up Verlet run ...\n",screen);
    if (flag) {
      fmt::print(screen,"  Unit style    : {}\n"
                        "  Current step  : {}\n"
                        "  Time step     : {}\n",
                 update->unit_style,update->ntimestep,update->dt);
      timer->print_timeout(screen);
    }
  }

  if (lmp->kokkos)
    error->all(FLERR,"KOKKOS package requires run_style verlet/kk");

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] atom x lmp->execute(LAMMPS::PAIR_COMPUTE) \n"), atom->x[0], atom->nlocal * 3, 1);
  comm->exchange();
  // if (atom->sortfreq > 0) atom->sort();
  // atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build(1);
  modify->setup_post_neighbor();
  neighbor->ncalls = 0;

  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] atom x nall \n"), atom->x[0], (atom->nlocal+atom->nghost) * 3, 1);


  // compute all forces

  force->setup();
  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] after pair lmp->execute(LAMMPS::PAIR_COMPUTE) \n"), atom->f[0], atom->nlocal * 3, 1);
  if(DEBUG_MSG) utils::logmesg_arry(lmp,fmt::format("[info] after pair virial \n"), force->pair->virial, 6, 1);

  if(DEBUG_MSG) MPI_Barrier(MPI_COMM_WORLD);


  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] after reverse lmp->execute(LAMMPS::PAIR_COMPUTE) \n"), atom->f[0], atom->nlocal * 3, 1);

  // utils::logmesg_arry_x(lmp,fmt::format("[info] after reverse lmp->execute(LAMMPS::PAIR_COMPUTE) \n"), atom->f[0], atom->nlocal * 3, 1);


  modify->setup(vflag);
  output->setup(flag);
  update->setupflag = 0;

  if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] after modify atom->v \n"), atom->v[0], atom->nlocal * 3, 1);


  if(DEBUG_MSG) MPI_Barrier(MPI_COMM_WORLD);


  // MPI_Barrier(world);  
  // MPI_Finalize();
  // exit(0);
}



/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void Verlet::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_post_neighbor = modify->n_post_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_pre_reverse = modify->n_pre_reverse;
  int n_post_force_any = modify->n_post_force_any;
  int n_end_of_step = modify->n_end_of_step;

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  for (int i = 0; i < n; i++) {
    if (timer->check_timeout(i)) {
      update->nsteps = i;
      break;
    }

    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);

    // initial time integration

    timer->stamp();
    modify->initial_integrate(vflag);
    if (n_post_integrate) modify->post_integrate();
    timer->stamp(Timer::MODIFY);

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) {
        timer->stamp();
        modify->pre_exchange();
        timer->stamp(Timer::MODIFY);
      }
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();

      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();
      comm->exchange();
      // if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) {
        modify->pre_neighbor();
        timer->stamp(Timer::MODIFY);
      }
      neighbor->build(1);
      timer->stamp(Timer::NEIGH);
      if (n_post_neighbor) {
        modify->post_neighbor();
        timer->stamp(Timer::MODIFY);
      }
    }

    if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] atom x nall \n"), atom->x[0], (atom->nlocal+atom->nghost) * 3, 1);


    // force computations
    // important for pair to come before bonded contributions
    // since some bonded potentials tally pairwise energy/virial
    // and Pair:ev_tally() needs to be called before any tallying

    force_clear();

    // utils::logmesg_arry_x(lmp,fmt::format("[info] atom x after force_clear \n"), atom->x[0], (atom->nlocal+atom->nghost) * 3, 1);

    // self_timer->stamp();
    // MPI_Barrier(world);
    // self_timer->stamp(Timer::PROD_ENV);

    timer->stamp();

    if (n_pre_force) {
      modify->pre_force(vflag);
      timer->stamp(Timer::MODIFY);
    }

    if (pair_compute_flag) {
      force->pair->compute(eflag,vflag);
      timer->stamp(Timer::PAIR);
    }

    if(DEBUG_MSG) MPI_Barrier(MPI_COMM_WORLD);


    // self_timer->stamp();
    // MPI_Barrier(world);
    // self_timer->stamp(Timer::PREPARE);

    if (atom->molecular != Atom::ATOMIC) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
      timer->stamp(Timer::BOND);
    }

    if(DEBUG_MSG) utils::logmesg_arry(lmp,fmt::format("[info] after pair virial \n"), force->pair->virial, 6, 1);

    // utils::logmesg_arry_x(lmp,fmt::format("[info] atom x before kspace \n"), atom->x[0], (atom->nlocal) * 3, 1);



    if (kspace_compute_flag) {
      force->kspace->compute(eflag,vflag);
      timer->stamp(Timer::KSPACE);
    }
    if(DEBUG_MSG) MPI_Barrier(MPI_COMM_WORLD);


    if (n_pre_reverse) {
      modify->pre_reverse(eflag,vflag);
      timer->stamp(Timer::MODIFY);
    }

    // reverse communication of forces

    timer->stamp();

    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }

    if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] after reverse lmp->execute(LAMMPS::PAIR_COMPUTE) \n"), atom->f[0], atom->nlocal * 3, 1);

    // utils::logmesg_arry_x(lmp,fmt::format("[info] before postforce ntimestep {} \n", ntimestep), atom->f[0], atom->nlocal * 3, 1);

    // force modifications, final time integration, diagnostics

    if (n_post_force_any) modify->post_force(vflag);

    if(DEBUG_MSG) utils::logmesg_arry(lmp,fmt::format("[info] after post_force virial \n"), force->pair->virial, 6, 1);


    modify->final_integrate();
    if (n_end_of_step) modify->end_of_step();
    timer->stamp(Timer::MODIFY);

    if(DEBUG_MSG) utils::logmesg_arry_x(lmp,fmt::format("[info] after modify atom->v \n"), atom->v[0], atom->nlocal * 3, 1);


    // all output

    if (ntimestep == output->next) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }


    // accuracy test
    {

      // std::string mesg = "_f = [";
      // for(int i = 0; i < 384; i++) {
      //   mesg += fmt::format(" {}, {}, {}", atom->f[i][0], atom->f[i][1], atom->f[i][2]);
      //   if(i != atom->nlocal - 1) mesg += ", ";
      // }
      // mesg += "]\n";
      // utils::logmesg(lmp,mesg);
  
      // utils::logmesg_arry(lmp,fmt::format("[info ]tag \n"), atom->tag, atom->nlocal, 1);
  
      // double all_f[512*3];
      // std::vector<int> all_tag(512);
      // std::vector<int> all_type(512);
      // int nlocal_nodes[48];
      // int  recvcounts[48];
      // int   displs[48];
      // MPI_Allgather(&atom->nlocal,1,MPI_INT,nlocal_nodes,1,MPI_INT,MPI_COMM_WORLD);
      
      // for(int i = 0; i < 48; i++) recvcounts[i] = nlocal_nodes[i] * 3;
      // displs[0] = 0;
      // for (int i = 1; i < 48; i++) displs[i] = displs[i - 1] + recvcounts[i - 1];
      // MPI_Gatherv(atom->f[0], atom->nlocal * 3, MPI_DOUBLE, all_f, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
      // for(int i = 0; i < 48; i++) recvcounts[i] = nlocal_nodes[i];
      // displs[0] = 0;
      // for (int i = 1; i < 48; i++) displs[i] = displs[i - 1] + recvcounts[i - 1];
      // MPI_Gatherv(atom->tag, atom->nlocal, MPI_INT, all_tag.data(), recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
      // MPI_Gatherv(atom->type, atom->nlocal, MPI_INT, all_type.data(), recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
  
      // std::string mesg = "_f = [";
      // for(int i = 1; i < 385; i++) {
      //   for(int j = 0; j < 512; j++) {
      //     if(all_tag[j] == i) {
      //       mesg += fmt::format(" {}, {}, {}", all_f[j*3+0], all_f[j*3+1], all_f[j*3+2]);
      //       if(i != 384) mesg += ", ";
      //       continue;
      //     }
      //   }
      // }
      // mesg += "]\n";
      // utils::logmesg(lmp,mesg);
  
      // utils::logmesg_arry(lmp,fmt::format("[info ]tag \n"), all_tag.data(), 512, 1);
      // utils::logmesg_arry(lmp,fmt::format("[info ]all_type \n"), all_type.data(), 512, 1);
    }

  }

  if (n_post_integrate) modify->post_integrate();

}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void Verlet::setup_minimal(int flag)
{
  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build(1);
    modify->setup_post_neighbor();
    neighbor->ncalls = 0;
  }

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if (force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) {
    force->kspace->setup();
    if (kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }

  modify->setup_pre_reverse(eflag,vflag);
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag);
  update->setupflag = 0;

  
}

void Verlet::cleanup()
{
  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void Verlet::force_clear()
{
  size_t nbytes;

  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}
