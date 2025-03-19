#ifdef KSPACE_CLASS

KSpaceStyle(pppm/dplr,PPPMDPLR)

#else

#ifndef LMP_PPPM_DPLR_H
#define LMP_PPPM_DPLR_H

#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif

#define SELF_HEFFTE

#include "pppm.h"
#include <iostream>
#include <vector>
#include "deepmd_common.h"
#include <fftw3-mpi.h>
#include <heffte.h>

namespace LAMMPS_NS {

  class PPPMDPLR : public PPPM {
public:
    PPPMDPLR(class LAMMPS *);
    virtual ~PPPMDPLR () {};
    void init() override;
    void setup() override;
    FPTYPE *fele;
    double **f_lr;

    heffte::box3d<> *box_pos;
    heffte::fft3d<heffte::backend::fftw> *heffte_wrapper;
    std::complex<FFT_SCALAR> *heffte_indata;
    std::complex<FFT_SCALAR> *heffte_outdata;

    void run_forward() {
      heffte_wrapper->forward(heffte_indata, heffte_outdata);
    };
    void run_backward() {
      heffte_wrapper->backward(heffte_indata, heffte_outdata);
    };


    // const std::vector<double > & get_fele() const {return fele;};
protected:
    virtual void compute(int, int) override;
    virtual void fieldforce_ik() override;
    virtual void fieldforce_ad() override;

    #ifdef SELF_HEFFTE
    virtual void poisson_ik() override;
    virtual void particle_map() override;
    virtual void make_rho() override;
    virtual void brick2fft() override;
    virtual void compute_gf_ik() override;
    void poisson_ik_heffte();
    void poisson_ik_utofubg();
    void particle_map_node();
    void make_rho_node();
    void init_heffte_fft();
    void init_node_fft();

    void reverse_node();
    void forward_node();

    struct Swap {
      int sendproc;       // proc to send to for forward comm
      int recvproc;       // proc to recv from for forward comm
      int npack;          // # of datums to pack
      int nunpack;        // # of datums to unpack
      int *packlist;      // 3d array offsets to pack
      int *unpacklist;    // 3d array offsets to unpack
      FFT_SCALAR *send_buf;
      FFT_SCALAR *recv_buf;
    };
  
    int nswap, maxswap;
    Swap *swap;
    #endif
private:
    // std::vector<double > fele;
    int nxlo_node_in, nylo_node_in, nzlo_node_in, nxhi_node_in, nyhi_node_in, nzhi_node_in;
    int nxlo_node_out, nylo_node_out, nzlo_node_out, nxhi_node_out, nyhi_node_out, nzhi_node_out;
    int nfft_node_brick, ngrid_node, nfft_node_both;

    double **vg_brick;
    double *fkx_brick, *fky_brick, *fkz_brick;
    double *greensfn_brick;


    double **x_node;
    double *q_node;
    int nlocal_nodes[NUMA_NUM];
    int nlocal_node;
    int **part2grid_node;
    FFT_SCALAR ***density_brick_node;
    FFT_SCALAR *work1_node, *work2_node;
    FFT_SCALAR *density_fft_node;


    const int con_direction[62][3] = {
      {0, 0, 1},    {0, 1, 0},    {1, 0, 0},
      {0, 1, 1},    {0, -1, 1},    {1, 0, 1},
      {-1, 0, 1},    {1, 1, 0},    {-1, 1, 0},
      {1, 1, 1},    {1, -1, 1},    {-1, 1, 1},
      {-1, -1, 1},  // 13 
  
      {0, 0, 2},    {0, 2, 0},    {2, 0, 0},  // 3
  
      {-2, -2, 2}, {-1, -2, 2}, {0, -2, 2}, {1, -2, 2}, {2, -2, 2}, 
      {-2, -1, 2}, {-1, -1, 2}, {0, -1, 2}, {1, -1, 2}, {2, -1, 2},  // 10
      {-2, 0, 2}, {-1, 0, 2},  {1, 0, 2}, {2, 0, 2},  // 4
      {-2, 1, 2}, {-1, 1, 2}, {0, 1, 2}, {1, 1, 2}, {2, 1, 2}, 
      {-2, 2, 2}, {-1, 2, 2}, {0, 2, 2}, {1, 2, 2}, {2, 2, 2},
  
      {-2, 2, 1}, {-1, 2, 1}, {0, 2, 1}, {1, 2, 1}, {2, 2, 1}, 
      {-2, -2, 1}, {-1, -2, 1}, {0, -2, 1}, {1, -2, 1}, {2, -2, 1},  // 20
  
      {-2, -1, 1}, {2, -1, 1}, 
      {-2, 0, 1}, {2, 0, 1}, 
      {-2, 1, 1}, {2, 1, 1},  
  
      {-2, 2, 0}, {-1, 2, 0},  {1, 2, 0}, {2, 2, 0},   
      {-2, 1, 0}, {2, 1, 0}   // 12
    };


  };

}

#endif
#endif

