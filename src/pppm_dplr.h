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
#include <utofu.h>

namespace LAMMPS_NS {

class FFT_UTOFU_BG : public Pointers{
public:
  FFT_UTOFU_BG (LAMMPS *lmp) : Pointers(lmp){};
  int fft_size[3];
  int nfft;
  int nfft_brick;
  int *nfft_bricks[3];
  int *nfft_bricks_offset[3];
  int max_nfft_brick[3];
  int lcl_size[3];
  int nblocks[3];
  int dgemm_size[3];
  int maxGemmsize;
  int *nodegrid;
  int *nodeloc;
  double *reduce_data;
  FFT_SCALAR *Wsin[3], *Wcos[3], *Wsin_i[3], *Wcos_i[3];
  FFT_SCALAR *calcu_buf;

  int rc;
  utofu_vbg_id_t lcl_vbg_ids[TNI_NUM][MAX_RING][2];
  utofu_vbg_id_t rmt_vbg_ids[TNI_NUM][MAX_RING][MAX_RING][2];
  struct utofu_vbg_setting vbg_settings[TNI_NUM][MAX_RING][2];

  void init(int nx_pppm, int ny_pppm, int nz_pppm,
    int nxlo_in, int nylo_in, int nzlo_in, int nxhi_in, int nyhi_in, int nzhi_in);

  void init_utofu_bg();

  void compute_fft3D_forward(FFT_SCALAR *in_data, int FFT_DIR);
    
};

class PPPMDPLR : public PPPM {
public:
    PPPMDPLR(class LAMMPS *);
    virtual ~PPPMDPLR () {};
    void init() override;
    void setup() override;
    void setup_brick();
    void setup_node();
    FPTYPE *fele;
    FPTYPE *fele_node;
    double **f_lr;

    heffte::box3d<> *box_pos;
    heffte::fft3d<heffte::backend::fftw> *heffte_wrapper;
    std::complex<FFT_SCALAR> *heffte_indata;
    std::complex<FFT_SCALAR> *heffte_outdata;

    FFT_UTOFU_BG *fft_utofu;

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
    void compute_gf_ik_brick();
    void compute_gf_ik_node();
    void poisson_ik_heffte();
    void poisson_ik_utofubg();
    void particle_map_node();
    void fieldforce_ik_brick();
    void fieldforce_ik_node();
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

    double **vg_node;
    double *fkx_node, *fky_node, *fkz_node;
    double *greensfn_node;


    double **x_node;
    double *q_node;
    int nlocal_nodes[NUMA_NUM];
    int nlocal_node;
    int **part2grid_node;
    FFT_SCALAR ***density_brick_node;
    FFT_SCALAR *work1_node, *work2_node;
    FFT_SCALAR *density_fft_node;
    FFT_SCALAR ***vdx_node, ***vdy_node, ***vdz_node;



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

