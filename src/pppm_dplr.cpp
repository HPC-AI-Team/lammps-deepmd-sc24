#include <math.h>
#include "pppm_dplr.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"
#include "pppm.h"
#include "grid3d.h"
#include "comm.h"
#include "angle.h"
#include "bond.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "grid3d.h"
#include "math_extra.h"
#include "math_special.h"
#include "neighbor.h"
#include "pair.h"
#include "remap_wrap.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER 7
#define OFFSET 16384
#define LARGE 10000.0
#define SMALL 0.00001
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#define OFFSET 16384

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif


/* ---------------------------------------------------------------------- */


PPPMDPLR::PPPMDPLR(LAMMPS *lmp) :
  PPPM(lmp)
{
  triclinic_support = 1;
}

/* ---------------------------------------------------------------------- */

void PPPMDPLR::init()
{
  // DPLR PPPM requires newton on, b/c it computes forces on ghost atoms

  if (force->newton == 0)
    error->all(FLERR,"Kspace style pppm/dplr requires newton on");

  PPPM::init();

  box_pos = new heffte::box3d<>({{nxlo_in, nylo_in, nzlo_in},
              { nxhi_in, nyhi_in, nzhi_in}});

  heffte_wrapper = new heffte::fft3d<heffte::backend::fftw>(*box_pos,
                        *box_pos,MPI_COMM_WORLD);
  heffte_indata = new std::complex<FFT_SCALAR>[heffte_wrapper->size_inbox()];
  heffte_outdata = new std::complex<FFT_SCALAR>[heffte_wrapper->size_outbox()];

  utils::logmesg(lmp, "[INFO] PPPMDPLR data size {} {} \n", 
            heffte_wrapper->size_inbox(),heffte_wrapper->size_outbox());

  
  // cout << " ninit pppm/dplr ---------------------- " << nlocal << endl;
  // fele.resize(nlocal*3);
  // fill(fele.begin(), fele.end(), 0.0);
}

/* ----------------------------------------------------------------------
   compute the PPPM long-range force, energy, virial
------------------------------------------------------------------------- */

void PPPMDPLR::compute(int eflag, int vflag)
{

  // if (me == 0) utils::logmesg(lmp,"[INFO] into PPPMDPLR::compute \n");
  // if (me == 0) utils::logmesg(lmp,"[INFO] differentiation_flag {}\n", differentiation_flag);
  // if (me == 0) utils::logmesg(lmp,"[INFO] triclinic {}\n", domain->triclinic);
  // if (me == 0) utils::logmesg(lmp,"[INFO] slabflag {}\n", slabflag);
  // if (me == 0) utils::logmesg(lmp,"[INFO] atom->q_flag {}\n", atom->q_flag);

  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag,vflag);

  // utils::logmesg(lmp,"[INFO] eflag_atom  {} vflag_atom {}\n", eflag_atom, vflag_atom);
  // utils::logmesg(lmp,"[INFO] evflag_atom  {} peratom_allocate_flag {}\n", evflag_atom, peratom_allocate_flag);


  if (evflag_atom && !peratom_allocate_flag) allocate_peratom();

  // if atom count has changed, update qsum and qsqsum
  // if (me == 0) utils::logmesg(lmp,"[INFO] natoms_original {}\n", natoms_original);
  // if (me == 0) utils::logmesg(lmp,"[INFO] atom->natoms {}  {} \n", atom->natoms, natoms_original);

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  // convert atoms from box to lamda coords

  boxlo = domain->boxlo;

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"pppm:part2grid");
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid

  // utils::logmesg_arry_x(lmp,fmt::format("[info] before particle map atom x  {}\n", atom->nlocal), atom->x[0], atom->nlocal * 3, 1);

  particle_map();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                   gc_buf1, gc_buf2, MPI_FFT_SCALAR);
                          
  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  // if (differentiation_flag == 1)
  //   gc->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),FORWARD_AD,
  //                           gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  // else
  gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK, 3, sizeof(FFT_SCALAR),
                     gc_buf1, gc_buf2, MPI_FFT_SCALAR);


  // extra per-atom energy/virial communication

  if (evflag_atom) {
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK_PERATOM, 7,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);

  }

  // calculate the force on my particles

  fieldforce();

  // extra per-atom energy/virial communication

  if (evflag_atom) fieldforce_peratom();

  // sum global energy across procs and add in volume-dependent term

  const double qscale = qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy = energy_all;

    energy *= 0.5*volume;
    // do not add self-term, for neutral systems qsum == 0
    // energy -= g_ewald*qsqsum/MY_PIS +
    //   MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // ntotal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    double *q = atom->q;
    int nlocal = atom->nlocal;
    int ntotal = nlocal;
    // if (tip4pflag) ntotal += atom->nghost;

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] *= 0.5;
        eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) eatom[i] *= 0.5*qscale;
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5*qscale;
    }
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();

  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);
}



#ifdef SELF_HEFFTE
/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void PPPMDPLR::poisson_ik()
{
  if(comm->fft_type_flag == 0) poisson_ik_normal();
  else if(comm->fft_type_flag == 1) poisson_ik_heffte();
  // int i,j,k,n;
  // double eng;
  // int _nfft;
  // int xlo, xhi, ylo, yhi, zlo, zhi;

  // if(comm->fft_type_flag == 0) {
  //   xlo = nxlo_fft; xhi = nxhi_fft; ylo = nylo_fft; yhi = nyhi_fft; zlo = nzlo_fft; zhi = nzhi_fft; 
  //   _nfft = nfft;
  // } else {
  //   xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 
  //   _nfft = nfft_brick;
  // }

  // // transform charge density (r -> k)

  // n = 0;

  // if(comm->fft_type_flag == 0) {
  //   for (i = 0; i < _nfft; i++) {
  //     work1[n++] = density_fft[i];
  //     work1[n++] = ZEROF;
  //   }
  //   fft1->compute(work1,work1,FFT3d::FORWARD);

  // } if(comm->fft_type_flag == 1) {
  //   for (int i = 0; i < _nfft; i++) {
  //     heffte_indata[i].real(density_fft[i]);
  //     heffte_indata[i].imag(ZEROF);
  //   }

  //   run_forward();
  //   n = 0;
  //   for (int i = 0; i < _nfft; i++) {
  //     work1[n++] = heffte_outdata[i].real();
  //     work1[n++] = heffte_outdata[i].imag();
  //   }
  // }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 \n"),work1, _nfft*2, 1 );
  

  // // global energy and virial contribution

  // double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  // double s2 = scaleinv*scaleinv;

  // if (eflag_global || vflag_global) {
  //   if (vflag_global) {
  //     n = 0;
  //     for (i = 0; i < _nfft; i++) {
  //       eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
  //       for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
  //       if (eflag_global) energy += eng;
  //       n += 2;
  //     }
  //   } else {
  //     n = 0;
  //     for (i = 0; i < _nfft; i++) {
  //       energy +=
  //         s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
  //       n += 2;
  //     }
  //   }
  // }

  // // scale by 1/total-grid-pts to get rho(k)
  // // multiply by Green's function to get V(k)

  // n = 0;
  // for (i = 0; i < _nfft; i++) {
  //   work1[n++] *= scaleinv * greensfn[i];
  //   work1[n++] *= scaleinv * greensfn[i];
  // }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 greensfn evflag_atom {} \n", evflag_atom),work1, _nfft*2, 1 );


  // // extra FFTs for per-atom energy/virial

  // if (evflag_atom) poisson_peratom();

  // // triclinic system

  // if (triclinic) {
  //   poisson_ik_triclinic();
  //   return;
  // }

  // // compute gradients of V(r) in each of 3 dims by transforming ik*V(k)
  // // FFT leaves data in 3d brick decomposition
  // // copy it into inner portion of vdx,vdy,vdz arrays

  // // x direction gradient

  
  // if(comm->fft_type_flag == 0) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         work2[n] = -fkx[i]*work1[n+1];
  //         work2[n+1] = fkx[i]*work1[n];
  //         n += 2;
  //       }
  
  //   fft2->compute(work2,work2,FFT3d::BACKWARD);

  // } if(comm->fft_type_flag == 1) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         heffte_indata[n].real(-fkx[i]*work1[2*n+1]);
  //         heffte_indata[n].imag(fkx[i] *work1[2*n]);
  //         n += 1;
  //       }

  //   run_backward();
  //   n = 0;
  //   for (int i = 0; i < _nfft; i++) {
  //     work2[n++] = heffte_outdata[i].real();
  //     work2[n++] = heffte_outdata[i].imag();
  //   }
  // }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 0 back output \n"),work2, _nfft*2, 1 );

  // n = 0;
  // for (k = nzlo_in; k <= nzhi_in; k++)
  //   for (j = nylo_in; j <= nyhi_in; j++)
  //     for (i = nxlo_in; i <= nxhi_in; i++) {
  //       vdx_brick[k][j][i] = work2[n];
  //       n += 2;
  //     }

  // // y direction gradient

  // if(comm->fft_type_flag == 0) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         work2[n] = -fky[j]*work1[n+1];
  //         work2[n+1] = fky[j]*work1[n];
  //         n += 2;
  //       }
  
  //   fft2->compute(work2,work2,FFT3d::BACKWARD);

  // } if(comm->fft_type_flag == 1) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         heffte_indata[n].real(-fky[j]*work1[2*n+1]);
  //         heffte_indata[n].imag(fky[j] *work1[2*n]);
  //         n += 1;
  //       }

  //   run_backward();
  //   n = 0;
  //   for (int i = 0; i < _nfft; i++) {
  //     work2[n++] = heffte_outdata[i].real();
  //     work2[n++] = heffte_outdata[i].imag();
  //   }
  // }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 1 back \n"),work2, _nfft*2, 1 );

  

  // n = 0;
  // for (k = nzlo_in; k <= nzhi_in; k++)
  //   for (j = nylo_in; j <= nyhi_in; j++)
  //     for (i = nxlo_in; i <= nxhi_in; i++) {
  //       vdy_brick[k][j][i] = work2[n];
  //       n += 2;
  //     }

  // // z direction gradient

  // if(comm->fft_type_flag == 0) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         work2[n] = -fkz[k]*work1[n+1];
  //         work2[n+1] = fkz[k]*work1[n];
  //         n += 2;
  //       }
  
  //   fft2->compute(work2,work2,FFT3d::BACKWARD);

  // } if(comm->fft_type_flag == 1) {
  //   n = 0;
  //   for (k = zlo; k <= zhi; k++)
  //     for (j = ylo; j <= yhi; j++)
  //       for (i = xlo; i <= xhi; i++) {
  //         heffte_indata[n].real(-fkz[k]*work1[2*n+1]);
  //         heffte_indata[n].imag(fkz[k] *work1[2*n]);
  //         n += 1;
  //       }

  //   run_backward();
  //   n = 0;
  //   for (int i = 0; i < _nfft; i++) {
  //     work2[n++] = heffte_outdata[i].real();
  //     work2[n++] = heffte_outdata[i].imag();
  //   }
  // }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 2 back \n"),work2, _nfft*2, 1 );

  // n = 0;
  // for (k = nzlo_in; k <= nzhi_in; k++)
  //   for (j = nylo_in; j <= nyhi_in; j++)
  //     for (i = nxlo_in; i <= nxhi_in; i++) {
  //       vdz_brick[k][j][i] = work2[n];
  //       n += 2;
  //     }
}

void PPPMDPLR::poisson_ik_utofubg() {

}

void PPPMDPLR::poisson_ik_heffte()
{
  int i,j,k,n;
  double eng;
  int _nfft;
  int xlo, xhi, ylo, yhi, zlo, zhi;

  xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 
  _nfft = nfft_brick;

  // transform charge density (r -> k)

  n = 0;

  for (int i = 0; i < _nfft; i++) {
    heffte_indata[i].real(density_fft[i]);
    heffte_indata[i].imag(ZEROF);
  }

  run_forward();
  n = 0;
  for (int i = 0; i < _nfft; i++) {
    work1[n++] = heffte_outdata[i].real();
    work1[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 \n"),work1, _nfft*2, 1 );
  

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < _nfft; i++) {
        eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < _nfft; i++) {
        energy +=
          s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < _nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 greensfn evflag_atom {} \n", evflag_atom),work1, _nfft*2, 1 );


  // extra FFTs for per-atom energy/virial

  if (evflag_atom) poisson_peratom();

  // triclinic system

  if (triclinic) {
    poisson_ik_triclinic();
    return;
  }

  // compute gradients of V(r) in each of 3 dims by transforming ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        heffte_indata[n].real(-fkx[i]*work1[2*n+1]);
        heffte_indata[n].imag(fkx[i] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < _nfft; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 0 back output \n"),work2, _nfft*2, 1 );

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  // y direction gradient

  
  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        heffte_indata[n].real(-fky[j]*work1[2*n+1]);
        heffte_indata[n].imag(fky[j] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < _nfft; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 1 back \n"),work2, _nfft*2, 1 );

  

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        heffte_indata[n].real(-fkz[k]*work1[2*n+1]);
        heffte_indata[n].imag(fkz[k] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < _nfft; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 2 back \n"),work2, _nfft*2, 1 );

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

void PPPMDPLR::poisson_ik_normal()
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,FFT3d::FORWARD);

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPM work1 \n"),work1, nfft*2, 1 );


  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
        eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        energy +=
          s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPM work1 greensn \n"),work1, nfft*2, 1 );


  // extra FFTs for per-atom energy/virial

  if (evflag_atom) poisson_peratom();

  // triclinic system

  if (triclinic) {
    poisson_ik_triclinic();
    return;
  }

  // compute gradients of V(r) in each of 3 dims by transforming ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkx[i]*work1[n+1];
        work2[n+1] = fkx[i]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,FFT3d::BACKWARD);
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPM 0 back \n"),work2, nfft_brick*2, 1 );


  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  // y direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fky[j]*work1[n+1];
        work2[n+1] = fky[j]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,FFT3d::BACKWARD);

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPM 1 back \n"),work2, nfft_brick*2, 1 );


  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkz[k]*work1[n+1];
        work2[n+1] = fkz[k]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,FFT3d::BACKWARD);
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPM 2 back \n"),work2, nfft_brick*2, 1 );


  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void PPPMDPLR::brick2fft()
{
  int n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
        density_fft[n++] = density_brick[iz][iy][ix];

  if(comm->fft_type_flag == 0) {
    remap->perform(density_fft,density_fft,work1);
  } 
}

void PPPMDPLR::compute_gf_ik()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order;

  int xlo, xhi, ylo, yhi, zlo, zhi;

  if(comm->fft_type_flag == 0) {
    xlo = nxlo_fft; xhi = nxhi_fft; ylo = nylo_fft; yhi = nyhi_fft; zlo = nzlo_fft; zhi = nzhi_fft; 
  } else {
    xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 
  }

  n = 0;
  for (m = zlo; m <= zhi; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

    for (l = ylo; l <= yhi; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      for (k = xlo; k <= xhi; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);

                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}


void PPPMDPLR::setup()
{
  if (triclinic) {
    setup_triclinic();
    return;
  }

  // perform some checks to avoid illegal boundaries with read_data

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with PPPM");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab PPPM");
  }

  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM
  // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;

  delvolinv = delxinv*delyinv*delzinv;

  double unitkx = (MY_2PI/xprd);
  double unitky = (MY_2PI/yprd);
  double unitkz = (MY_2PI/zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  double per;

  int xlo, xhi, ylo, yhi, zlo, zhi;

  if(comm->fft_type_flag == 0) {
    xlo = nxlo_fft; xhi = nxhi_fft; ylo = nylo_fft; yhi = nyhi_fft; zlo = nzlo_fft; zhi = nzhi_fft; 
  } else {
    xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 
  }

  for (i = xlo; i <= xhi; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx[i] = unitkx*per;
  }

  for (i = ylo; i <= yhi; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky[i] = unitky*per;
  }

  for (i = zlo; i <= zhi; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = zlo; k <= zhi; k++) {
    for (j = ylo; j <= yhi; j++) {
      for (i = xlo; i <= xhi; i++) {
        sqk = fkx[i]*fkx[i] + fky[j]*fky[j] + fkz[k]*fkz[k];
        if (sqk == 0.0) {
          vg[n][0] = 0.0;
          vg[n][1] = 0.0;
          vg[n][2] = 0.0;
          vg[n][3] = 0.0;
          vg[n][4] = 0.0;
          vg[n][5] = 0.0;
        } else {
          vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
          vg[n][0] = 1.0 + vterm*fkx[i]*fkx[i];
          vg[n][1] = 1.0 + vterm*fky[j]*fky[j];
          vg[n][2] = 1.0 + vterm*fkz[k]*fkz[k];
          vg[n][3] = vterm*fkx[i]*fky[j];
          vg[n][4] = vterm*fkx[i]*fkz[k];
          vg[n][5] = vterm*fky[j]*fkz[k];
        }
        n++;
      }
    }
  }

  if (differentiation_flag == 1) compute_gf_ad();
  else compute_gf_ik();
}

#endif
/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void PPPMDPLR::fieldforce_ik()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  // double **f = atom->f;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // fele.resize(nlocal*3);
  // fill(fele.begin(), fele.end(), 0.0);
  memset(fele, 0, nlocal*3*sizeof(double));

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
    dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
    dz = nz+shiftone - (x[i][2]-boxlo[2])*delzinv;

    compute_rho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        y0 = z0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          x0 = y0*rho1d[0][l];
          ekx -= x0*vdx_brick[mz][my][mx];
          eky -= x0*vdy_brick[mz][my][mx];
          ekz -= x0*vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = qqrd2e * scale * q[i];
    fele[i*3+0] += qfactor*ekx;
    fele[i*3+1] += qfactor*eky;
    if (slabflag != 2) fele[i*3+2] += qfactor*ekz;
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void PPPMDPLR::fieldforce_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz;
  FFT_SCALAR ekx,eky,ekz;
  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  // double **f = atom->f;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // fele.resize(nlocal*3);
  // fill(fele.begin(), fele.end(), 0.0);
  memset(fele, 0, nlocal*3*sizeof(double));


  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
    dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
    dz = nz+shiftone - (x[i][2]-boxlo[2])*delzinv;

    compute_rho1d(dx,dy,dz);
    compute_drho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and subtract self forces

    const double qfactor = qqrd2e * scale;

    s1 = x[i][0]*hx_inv;
    s2 = x[i][1]*hy_inv;
    s3 = x[i][2]*hz_inv;
    sf = sf_coeff[0]*sin(2*MY_PI*s1);
    sf += sf_coeff[1]*sin(4*MY_PI*s1);
    sf *= 2*q[i]*q[i];
    fele[i*3+0] += qfactor*(ekx*q[i] - sf);

    sf = sf_coeff[2]*sin(2*MY_PI*s2);
    sf += sf_coeff[3]*sin(4*MY_PI*s2);
    sf *= 2*q[i]*q[i];
    fele[i*3+1] += qfactor*(eky*q[i] - sf);


    sf = sf_coeff[4]*sin(2*MY_PI*s3);
    sf += sf_coeff[5]*sin(4*MY_PI*s3);
    sf *= 2*q[i]*q[i];
    if (slabflag != 2) fele[i*3+2] += qfactor*(ekz*q[i] - sf);
  }
}

