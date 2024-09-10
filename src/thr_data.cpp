/* -------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
   per-thread data management for LAMMPS
------------------------------------------------------------------------- */

#include <cstdio>
#include <cstring>

#include "thr_data.h"

#include "memory.h"
#include "timer.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ThrData::ThrData(int tid, Timer *t) :
    _f(nullptr), _torque(nullptr), _erforce(nullptr), _de(nullptr), _drho(nullptr), _mu(nullptr),
    _lambda(nullptr), _rhoB(nullptr), _D_values(nullptr), _rho(nullptr), _fp(nullptr),
    _rho1d(nullptr), _drho1d(nullptr), _rho1d_6(nullptr), _drho1d_6(nullptr), _tid(tid), _timer(t)
{
  _timer_active = 0;
}

/* ---------------------------------------------------------------------- */

void ThrData::check_tid(int tid)
{
  if (tid != _tid)
    fprintf(stderr, "WARNING: external and internal tid mismatch %d != %d\n", tid, _tid);
}

/* ---------------------------------------------------------------------- */

void ThrData::_stamp(enum Timer::ttype flag)
{
  // do nothing until it gets set to 0 in ::setup()
  if (_timer_active < 0) return;

  if (flag == Timer::START) { _timer_active = 1; }

  if (_timer_active) _timer->stamp(flag);
}

/* ---------------------------------------------------------------------- */

double ThrData::get_time(enum Timer::ttype flag)
{
  if (_timer)
    return _timer->get_wall(flag);
  else
    return 0.0;
}

/* ---------------------------------------------------------------------- */

void ThrData::init_force(int nall, double **f, double **torque, double *erforce, double *de,
                         double *drho)
{
  eng_vdwl = eng_coul = eng_bond = eng_angle = eng_dihed = eng_imprp = eng_kspce = 0.0;
  memset(virial_pair, 0, 6 * sizeof(double));
  memset(virial_bond, 0, 6 * sizeof(double));
  memset(virial_angle, 0, 6 * sizeof(double));
  memset(virial_dihed, 0, 6 * sizeof(double));
  memset(virial_imprp, 0, 6 * sizeof(double));
  memset(virial_kspce, 0, 6 * sizeof(double));

  eatom_pair = eatom_bond = eatom_angle = eatom_dihed = eatom_imprp = eatom_kspce = nullptr;
  vatom_pair = vatom_bond = vatom_angle = vatom_dihed = vatom_imprp = vatom_kspce = nullptr;

  if (nall >= 0 && f) {
    _f = f + _tid * nall;
    memset(&(_f[0][0]), 0, nall * 3 * sizeof(double));
    // printf("[info] pre_force init_force tid %d\n",_tid);
  } else
    _f = nullptr;

  if (nall >= 0 && torque) {
    _torque = torque + _tid * nall;
    memset(&(_torque[0][0]), 0, nall * 3 * sizeof(double));
  } else
    _torque = nullptr;

  if (nall >= 0 && erforce) {
    _erforce = erforce + _tid * nall;
    memset(&(_erforce[0]), 0, nall * sizeof(double));
  } else
    _erforce = nullptr;

  if (nall >= 0 && de) {
    _de = de + _tid * nall;
    memset(&(_de[0]), 0, nall * sizeof(double));
  } else
    _de = nullptr;

  if (nall >= 0 && drho) {
    _drho = drho + _tid * nall;
    memset(&(_drho[0]), 0, nall * sizeof(double));
  } else
    _drho = nullptr;
}

/* ----------------------------------------------------------------------
   set up and clear out locally managed per atom arrays
------------------------------------------------------------------------- */

void ThrData::init_eam(int nall, double *rho)
{
  if (nall >= 0 && rho) {
    _rho = rho + _tid * nall;
    memset(_rho, 0, nall * sizeof(double));
  }
}

/* ---------------------------------------------------------------------- */

void ThrData::init_adp(int nall, double *rho, double **mu, double **lambda)
{
  init_eam(nall, rho);

  if (nall >= 0 && mu && lambda) {
    _mu = mu + _tid * nall;
    _lambda = lambda + _tid * nall;
    memset(&(_mu[0][0]), 0, nall * 3 * sizeof(double));
    memset(&(_lambda[0][0]), 0, nall * 6 * sizeof(double));
  }
}

/* ---------------------------------------------------------------------- */

void ThrData::init_eim(int nall, double *rho, double *fp)
{
  init_eam(nall, rho);

  if (nall >= 0 && fp) {
    _fp = fp + _tid * nall;
    memset(_fp, 0, nall * sizeof(double));
  }
}

/* ----------------------------------------------------------------------
   if order > 0 : set up per thread storage for PPPM
   if order < 0 : free per thread storage for PPPM
------------------------------------------------------------------------- */
#if defined(FFT_SINGLE)
typedef float FFT_SCALAR;
#else
typedef double FFT_SCALAR;
#endif

void ThrData::init_pppm(int order, Memory *memory)
{
  FFT_SCALAR **rho1d, **drho1d;
  if (order > 0) {
    rho1d = static_cast<FFT_SCALAR **>(_rho1d);
    drho1d = static_cast<FFT_SCALAR **>(_drho1d);
    if (rho1d) memory->destroy2d_offset(rho1d, -order / 2);
    if (drho1d) memory->destroy2d_offset(drho1d, -order / 2);
    memory->create2d_offset(rho1d, 3, -order / 2, order / 2, "thr_data:rho1d");
    memory->create2d_offset(drho1d, 3, -order / 2, order / 2, "thr_data:drho1d");
    _rho1d = static_cast<void *>(rho1d);
    _drho1d = static_cast<void *>(drho1d);
  } else {
    order = -order;
    rho1d = static_cast<FFT_SCALAR **>(_rho1d);
    drho1d = static_cast<FFT_SCALAR **>(_drho1d);
    if (rho1d) memory->destroy2d_offset(rho1d, -order / 2);
    if (drho1d) memory->destroy2d_offset(drho1d, -order / 2);
    _rho1d = nullptr;
    _drho1d = nullptr;
  }
}

/* ----------------------------------------------------------------------
   if order > 0 : set up per thread storage for PPPM
   if order < 0 : free per thread storage for PPPM
------------------------------------------------------------------------- */
#if defined(FFT_SINGLE)
typedef float FFT_SCALAR;
#else
typedef double FFT_SCALAR;
#endif

void ThrData::init_pppm_disp(int order_6, Memory *memory)
{
  FFT_SCALAR **rho1d_6, **drho1d_6;
  if (order_6 > 0) {
    rho1d_6 = static_cast<FFT_SCALAR **>(_rho1d_6);
    drho1d_6 = static_cast<FFT_SCALAR **>(_drho1d_6);
    if (rho1d_6) memory->destroy2d_offset(rho1d_6, -order_6 / 2);
    if (drho1d_6) memory->destroy2d_offset(drho1d_6, -order_6 / 2);
    memory->create2d_offset(rho1d_6, 3, -order_6 / 2, order_6 / 2, "thr_data:rho1d_6");
    memory->create2d_offset(drho1d_6, 3, -order_6 / 2, order_6 / 2, "thr_data:drho1d_6");
    _rho1d_6 = static_cast<void *>(rho1d_6);
    _drho1d_6 = static_cast<void *>(drho1d_6);
  } else {
    order_6 = -order_6;
    rho1d_6 = static_cast<FFT_SCALAR **>(_rho1d_6);
    drho1d_6 = static_cast<FFT_SCALAR **>(_drho1d_6);
    if (rho1d_6) memory->destroy2d_offset(rho1d_6, -order_6 / 2);
    if (drho1d_6) memory->destroy2d_offset(drho1d_6, -order_6 / 2);
  }
}

/* ----------------------------------------------------------------------
   compute global pair virial via summing F dot r over own & ghost atoms
   at this point, only pairwise forces have been accumulated in atom->f
------------------------------------------------------------------------- */

void ThrData::virial_fdotr_compute(double **x, int nlocal, int nghost, int nfirst)
{

  // sum over force on all particles including ghosts

  if (nfirst < 0) {
    int nall = nlocal + nghost;
    for (int i = 0; i < nall; i++) {
      virial_pair[0] += _f[i][0] * x[i][0];
      virial_pair[1] += _f[i][1] * x[i][1];
      virial_pair[2] += _f[i][2] * x[i][2];
      virial_pair[3] += _f[i][1] * x[i][0];
      virial_pair[4] += _f[i][2] * x[i][0];
      virial_pair[5] += _f[i][2] * x[i][1];
    }

    // neighbor includegroup flag is set
    // sum over force on initial nfirst particles and ghosts

  } else {
    int nall = nfirst;
    for (int i = 0; i < nall; i++) {
      virial_pair[0] += _f[i][0] * x[i][0];
      virial_pair[1] += _f[i][1] * x[i][1];
      virial_pair[2] += _f[i][2] * x[i][2];
      virial_pair[3] += _f[i][1] * x[i][0];
      virial_pair[4] += _f[i][2] * x[i][0];
      virial_pair[5] += _f[i][2] * x[i][1];
    }
    nall = nlocal + nghost;
    for (int i = nlocal; i < nall; i++) {
      virial_pair[0] += _f[i][0] * x[i][0];
      virial_pair[1] += _f[i][1] * x[i][1];
      virial_pair[2] += _f[i][2] * x[i][2];
      virial_pair[3] += _f[i][1] * x[i][0];
      virial_pair[4] += _f[i][2] * x[i][0];
      virial_pair[5] += _f[i][2] * x[i][1];
    }
  }
}

/* ---------------------------------------------------------------------- */

double ThrData::memory_usage()
{
  double bytes = (7 + 6 * 6) * sizeof(double);
  bytes += (double) 2 * sizeof(double *);
  bytes += (double) 4 * sizeof(int);

  return bytes;
}

/* additional helper functions */

// reduce per thread data into the first part of the data
// array that is used for the non-threaded parts and reset
// the temporary storage to 0.0. this routine depends on
// multi-dimensional arrays like force stored in this order
// x1,y1,z1,x2,y2,z2,...
// we need to post a barrier to wait until all threads are done
// with writing to the array .
void LAMMPS_NS::data_reduce_thr(double *dall, int nall, int nthreads, int ndim, int tid)
{
#if defined(_OPENMP)
  // NOOP in single-threaded execution.
  if (nthreads == 1) return;
#pragma omp barrier
  {
    const int nvals = ndim * nall;
    const int idelta = nvals / nthreads + 1;
    const int ifrom = tid * idelta;
    const int ito = ((ifrom + idelta) > nvals) ? nvals : (ifrom + idelta);

#if defined(USER_OMP_NO_UNROLL)
    if (ifrom < nvals) {
      int m = 0;

      for (m = ifrom; m < ito; ++m) {
        for (int n = 1; n < nthreads; ++n) {
          dall[m] += dall[n * nvals + m];
          dall[n * nvals + m] = 0.0;
        }
      }
    }
#else
    // this if protects against having more threads than atoms
    if (ifrom < nvals) {
      int m = 0;

      // for architectures that have L1 D-cache line sizes of 64 bytes
      // (8 doubles) wide, explicitly unroll this loop to  compute 8
      // contiguous values in the array at a time
      // -- modify this code based on the size of the cache line
      double t0, t1, t2, t3, t4, t5, t6, t7;
      for (m = ifrom; m < (ito - 7); m += 8) {
        t0 = dall[m];
        t1 = dall[m + 1];
        t2 = dall[m + 2];
        t3 = dall[m + 3];
        t4 = dall[m + 4];
        t5 = dall[m + 5];
        t6 = dall[m + 6];
        t7 = dall[m + 7];
        for (int n = 1; n < nthreads; ++n) {
          t0 += dall[n * nvals + m];
          t1 += dall[n * nvals + m + 1];
          t2 += dall[n * nvals + m + 2];
          t3 += dall[n * nvals + m + 3];
          t4 += dall[n * nvals + m + 4];
          t5 += dall[n * nvals + m + 5];
          t6 += dall[n * nvals + m + 6];
          t7 += dall[n * nvals + m + 7];
          dall[n * nvals + m] = 0.0;
          dall[n * nvals + m + 1] = 0.0;
          dall[n * nvals + m + 2] = 0.0;
          dall[n * nvals + m + 3] = 0.0;
          dall[n * nvals + m + 4] = 0.0;
          dall[n * nvals + m + 5] = 0.0;
          dall[n * nvals + m + 6] = 0.0;
          dall[n * nvals + m + 7] = 0.0;
        }
        dall[m] = t0;
        dall[m + 1] = t1;
        dall[m + 2] = t2;
        dall[m + 3] = t3;
        dall[m + 4] = t4;
        dall[m + 5] = t5;
        dall[m + 6] = t6;
        dall[m + 7] = t7;
      }
      // do the last < 8 values
      for (; m < ito; m++) {
        for (int n = 1; n < nthreads; ++n) {
          dall[m] += dall[n * nvals + m];
          dall[n * nvals + m] = 0.0;
        }
      }
    }
#endif
  }
#else
  // NOOP in non-threaded execution.
  return;
#endif
}




void LAMMPS_NS::data_reduce_thr_threadpool(double *dall, int nall, int nthreads, int ndim, int tid, LAMMPS* lmp) {
  
#if defined(_OPENMP)
  // NOOP in single-threaded execution.
  if (nthreads == 1) return;
#pragma omp barrier

  const int nvals = ndim * nall;
  const int idelta = nvals / nthreads + 1;
  const int ifrom = tid * idelta;
  const int ito = ((ifrom + idelta) > nvals) ? nvals : (ifrom + idelta);

  // if(lmp->comm->debug_flag) utils::logmesg(lmp,"data_reduce_thr_threadpool tid {} ifrom {} ito {} nall {} nlocal {} \n", tid, ifrom, ito, nall, lmp->atom->nlocal);

  if (ifrom < nvals) {
    int m = 0;

    // for architectures that have L1 D-cache line sizes of 64 bytes
    // (8 doubles) wide, explicitly unroll this loop to  compute 8
    // contiguous values in the array at a time
    // -- modify this code based on the size of the cache line

    double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31;

    for (m = ifrom; m < (ito - 31); m += 32) {
      t0 = dall[m + 0];
      t1 = dall[m + 1];
      t2 = dall[m + 2];
      t3 = dall[m + 3];
      t4 = dall[m + 4];
      t5 = dall[m + 5];
      t6 = dall[m + 6];
      t7 = dall[m + 7];
      t8 = dall[m + 8];
      t9 = dall[m + 9];
      t10 = dall[m + 10];
      t11 = dall[m + 11];
      t12 = dall[m + 12];
      t13 = dall[m + 13];
      t14 = dall[m + 14];
      t15 = dall[m + 15];
      t16 = dall[m + 16];
      t17 = dall[m + 17];
      t18 = dall[m + 18];
      t19 = dall[m + 19];
      t20 = dall[m + 20];
      t21 = dall[m + 21];
      t22 = dall[m + 22];
      t23 = dall[m + 23];
      t24 = dall[m + 24];
      t25 = dall[m + 25];
      t26 = dall[m + 26];
      t27 = dall[m + 27];
      t28 = dall[m + 28];
      t29 = dall[m + 29];
      t30 = dall[m + 30];
      t31 = dall[m + 31];
      for (int n = 1; n < nthreads; ++n) {
        t0 += dall[n * nvals + m + 0];
        t1 += dall[n * nvals + m + 1];
        t2 += dall[n * nvals + m + 2];
        t3 += dall[n * nvals + m + 3];
        t4 += dall[n * nvals + m + 4];
        t5 += dall[n * nvals + m + 5];
        t6 += dall[n * nvals + m + 6];
        t7 += dall[n * nvals + m + 7];
        t8 += dall[n * nvals + m + 8];
        t9 += dall[n * nvals + m + 9];
        t10 += dall[n * nvals + m + 10];
        t11 += dall[n * nvals + m + 11];
        t12 += dall[n * nvals + m + 12];
        t13 += dall[n * nvals + m + 13];
        t14 += dall[n * nvals + m + 14];
        t15 += dall[n * nvals + m + 15];
        t16 += dall[n * nvals + m + 16];
        t17 += dall[n * nvals + m + 17];
        t18 += dall[n * nvals + m + 18];
        t19 += dall[n * nvals + m + 19];
        t20 += dall[n * nvals + m + 20];
        t21 += dall[n * nvals + m + 21];
        t22 += dall[n * nvals + m + 22];
        t23 += dall[n * nvals + m + 23];
        t24 += dall[n * nvals + m + 24];
        t25 += dall[n * nvals + m + 25];
        t26 += dall[n * nvals + m + 26];
        t27 += dall[n * nvals + m + 27];
        t28 += dall[n * nvals + m + 28];
        t29 += dall[n * nvals + m + 29];
        t30 += dall[n * nvals + m + 30];
        t31 += dall[n * nvals + m + 31];

        dall[n * nvals + m + 0] = 0.0;
        dall[n * nvals + m + 1] = 0.0;
        dall[n * nvals + m + 2] = 0.0;
        dall[n * nvals + m + 3] = 0.0;
        dall[n * nvals + m + 4] = 0.0;
        dall[n * nvals + m + 5] = 0.0;
        dall[n * nvals + m + 6] = 0.0;
        dall[n * nvals + m + 7] = 0.0;
        dall[n * nvals + m + 8] = 0.0;
        dall[n * nvals + m + 9] = 0.0;
        dall[n * nvals + m + 10] = 0.0;
        dall[n * nvals + m + 11] = 0.0;
        dall[n * nvals + m + 12] = 0.0;
        dall[n * nvals + m + 13] = 0.0;
        dall[n * nvals + m + 14] = 0.0;
        dall[n * nvals + m + 15] = 0.0;
        dall[n * nvals + m + 16] = 0.0;
        dall[n * nvals + m + 17] = 0.0;
        dall[n * nvals + m + 18] = 0.0;
        dall[n * nvals + m + 19] = 0.0;
        dall[n * nvals + m + 20] = 0.0;
        dall[n * nvals + m + 21] = 0.0;
        dall[n * nvals + m + 22] = 0.0;
        dall[n * nvals + m + 23] = 0.0;
        dall[n * nvals + m + 24] = 0.0;
        dall[n * nvals + m + 25] = 0.0;
        dall[n * nvals + m + 26] = 0.0;
        dall[n * nvals + m + 27] = 0.0;
        dall[n * nvals + m + 28] = 0.0;
        dall[n * nvals + m + 29] = 0.0;
        dall[n * nvals + m + 30] = 0.0;
        dall[n * nvals + m + 31] = 0.0;
      }
      dall[m + 0] = t0;
      dall[m + 1] = t1;
      dall[m + 2] = t2;
      dall[m + 3] = t3;
      dall[m + 4] = t4;
      dall[m + 5] = t5;
      dall[m + 6] = t6;
      dall[m + 7] = t7;
      dall[m + 8] = t8;
      dall[m + 9] = t9;
      dall[m + 10] = t10;
      dall[m + 11] = t11;
      dall[m + 12] = t12;
      dall[m + 13] = t13;
      dall[m + 14] = t14;
      dall[m + 15] = t15;
      dall[m + 16] = t16;
      dall[m + 17] = t17;
      dall[m + 18] = t18;
      dall[m + 19] = t19;
      dall[m + 20] = t20;
      dall[m + 21] = t21;
      dall[m + 22] = t22;
      dall[m + 23] = t23;
      dall[m + 24] = t24;
      dall[m + 25] = t25;
      dall[m + 26] = t26;
      dall[m + 27] = t27;
      dall[m + 28] = t28;
      dall[m + 29] = t29;
      dall[m + 30] = t30;
      dall[m + 31] = t31;
    }
    // do the last < 32 values
    for (; m < ito; m++) {
      for (int n = 1; n < nthreads; ++n) {
        dall[m] += dall[n * nvals + m];
        dall[n * nvals + m] = 0.0;
      }
    }
  }

#else
  // NOOP in non-threaded execution.
  return;
#endif
}

void LAMMPS_NS::data_reduce_thr_threadpool_param(double *dall, int nall, int nthreads, int ndim, int tid, int scale, LAMMPS* lmp) {

#if defined(_OPENMP)
  // NOOP in single-threaded execution.
  if (nthreads == 1) return;
#pragma omp barrier

  const int nvals = ndim * nall;
  const int idelta = nvals / nthreads + 1;
  const int ifrom = tid * idelta;
  const int ito = ((ifrom + idelta) > nvals) ? nvals : (ifrom + idelta);

  // if(lmp->comm->debug_flag) utils::logmesg(lmp,"data_reduce_thr_threadpool tid {} ifrom {} ito {} nall {} nlocal {} \n", tid, ifrom, ito, nall, lmp->atom->nlocal);

  if (ifrom < nvals) {
    int m = 0;

    // for architectures that have L1 D-cache line sizes of 64 bytes
    // (8 doubles) wide, explicitly unroll this loop to  compute 8
    // contiguous values in the array at a time
    // -- modify this code based on the size of the cache line

    double t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31;

    for (m = ifrom; m < (ito - 31); m += 32) {
      t0 = dall[m + 0];
      t1 = dall[m + 1];
      t2 = dall[m + 2];
      t3 = dall[m + 3];
      t4 = dall[m + 4];
      t5 = dall[m + 5];
      t6 = dall[m + 6];
      t7 = dall[m + 7];
      t8 = dall[m + 8];
      t9 = dall[m + 9];
      t10 = dall[m + 10];
      t11 = dall[m + 11];
      t12 = dall[m + 12];
      t13 = dall[m + 13];
      t14 = dall[m + 14];
      t15 = dall[m + 15];
      t16 = dall[m + 16];
      t17 = dall[m + 17];
      t18 = dall[m + 18];
      t19 = dall[m + 19];
      t20 = dall[m + 20];
      t21 = dall[m + 21];
      t22 = dall[m + 22];
      t23 = dall[m + 23];
      t24 = dall[m + 24];
      t25 = dall[m + 25];
      t26 = dall[m + 26];
      t27 = dall[m + 27];
      t28 = dall[m + 28];
      t29 = dall[m + 29];
      t30 = dall[m + 30];
      t31 = dall[m + 31];
      for (int n = 1; n < nthreads; ++n) {
        t0 += dall[n * nvals + m + 0];
        t1 += dall[n * nvals + m + 1];
        t2 += dall[n * nvals + m + 2];
        t3 += dall[n * nvals + m + 3];
        t4 += dall[n * nvals + m + 4];
        t5 += dall[n * nvals + m + 5];
        t6 += dall[n * nvals + m + 6];
        t7 += dall[n * nvals + m + 7];
        t8 += dall[n * nvals + m + 8];
        t9 += dall[n * nvals + m + 9];
        t10 += dall[n * nvals + m + 10];
        t11 += dall[n * nvals + m + 11];
        t12 += dall[n * nvals + m + 12];
        t13 += dall[n * nvals + m + 13];
        t14 += dall[n * nvals + m + 14];
        t15 += dall[n * nvals + m + 15];
        t16 += dall[n * nvals + m + 16];
        t17 += dall[n * nvals + m + 17];
        t18 += dall[n * nvals + m + 18];
        t19 += dall[n * nvals + m + 19];
        t20 += dall[n * nvals + m + 20];
        t21 += dall[n * nvals + m + 21];
        t22 += dall[n * nvals + m + 22];
        t23 += dall[n * nvals + m + 23];
        t24 += dall[n * nvals + m + 24];
        t25 += dall[n * nvals + m + 25];
        t26 += dall[n * nvals + m + 26];
        t27 += dall[n * nvals + m + 27];
        t28 += dall[n * nvals + m + 28];
        t29 += dall[n * nvals + m + 29];
        t30 += dall[n * nvals + m + 30];
        t31 += dall[n * nvals + m + 31];

        dall[n * nvals + m + 0] = 0.0;
        dall[n * nvals + m + 1] = 0.0;
        dall[n * nvals + m + 2] = 0.0;
        dall[n * nvals + m + 3] = 0.0;
        dall[n * nvals + m + 4] = 0.0;
        dall[n * nvals + m + 5] = 0.0;
        dall[n * nvals + m + 6] = 0.0;
        dall[n * nvals + m + 7] = 0.0;
        dall[n * nvals + m + 8] = 0.0;
        dall[n * nvals + m + 9] = 0.0;
        dall[n * nvals + m + 10] = 0.0;
        dall[n * nvals + m + 11] = 0.0;
        dall[n * nvals + m + 12] = 0.0;
        dall[n * nvals + m + 13] = 0.0;
        dall[n * nvals + m + 14] = 0.0;
        dall[n * nvals + m + 15] = 0.0;
        dall[n * nvals + m + 16] = 0.0;
        dall[n * nvals + m + 17] = 0.0;
        dall[n * nvals + m + 18] = 0.0;
        dall[n * nvals + m + 19] = 0.0;
        dall[n * nvals + m + 20] = 0.0;
        dall[n * nvals + m + 21] = 0.0;
        dall[n * nvals + m + 22] = 0.0;
        dall[n * nvals + m + 23] = 0.0;
        dall[n * nvals + m + 24] = 0.0;
        dall[n * nvals + m + 25] = 0.0;
        dall[n * nvals + m + 26] = 0.0;
        dall[n * nvals + m + 27] = 0.0;
        dall[n * nvals + m + 28] = 0.0;
        dall[n * nvals + m + 29] = 0.0;
        dall[n * nvals + m + 30] = 0.0;
        dall[n * nvals + m + 31] = 0.0;
      }
      dall[m + 0] = t0 * scale;
      dall[m + 1] = t1 * scale;
      dall[m + 2] = t2 * scale;
      dall[m + 3] = t3 * scale;
      dall[m + 4] = t4 * scale;
      dall[m + 5] = t5 * scale;
      dall[m + 6] = t6 * scale;
      dall[m + 7] = t7 * scale;
      dall[m + 8] = t8 * scale;
      dall[m + 9] = t9 * scale;
      dall[m + 10] = t10 * scale;
      dall[m + 11] = t11 * scale;
      dall[m + 12] = t12 * scale;
      dall[m + 13] = t13 * scale;
      dall[m + 14] = t14 * scale;
      dall[m + 15] = t15 * scale;
      dall[m + 16] = t16 * scale;
      dall[m + 17] = t17 * scale;
      dall[m + 18] = t18 * scale;
      dall[m + 19] = t19 * scale;
      dall[m + 20] = t20 * scale;
      dall[m + 21] = t21 * scale;
      dall[m + 22] = t22 * scale;
      dall[m + 23] = t23 * scale;
      dall[m + 24] = t24 * scale;
      dall[m + 25] = t25 * scale;
      dall[m + 26] = t26 * scale;
      dall[m + 27] = t27 * scale;
      dall[m + 28] = t28 * scale;
      dall[m + 29] = t29 * scale;
      dall[m + 30] = t30 * scale;
      dall[m + 31] = t31 * scale;
    }
    // do the last < 32 values
    for (; m < ito; m++) {
      for (int n = 1; n < nthreads; ++n) {
        dall[m] += dall[n * nvals + m];
        dall[n * nvals + m] = 0.0;
      }
      dall[m] *= scale;
    }
  }
#else
  // NOOP in non-threaded execution.
  return;
#endif
}