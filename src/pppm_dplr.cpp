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

#define PI 3.14159265358979323846

inline void init_local_W(FFT_SCALAR *Wsin[3], FFT_SCALAR *Wcos[3], FFT_SCALAR *Wsin_i[3], FFT_SCALAR *Wcos_i[3], 
      int *fft_size, int *lcl_size, int *loc) {

  for(int dir = 0; dir < 3; dir++) {
    for (int m = 0; m < fft_size[dir]; m++) {
      for (int n = 0, j = lcl_size[dir] * loc[dir]; n < lcl_size[dir]; n++, j++) {
        int ptr = m * lcl_size[dir] + n;
        // int ptr = n * fft_size[dir] + m;
        double theta;
        theta = -2.0 * PI * m * j / fft_size[dir];
        Wcos[dir][ptr] = cos(theta);
        Wsin[dir][ptr] = sin(theta);
        theta = 2.0 * PI * m * j / fft_size[dir];
        Wcos_i[dir][ptr] = cos(theta);
        Wsin_i[dir][ptr] = sin(theta);
      }
    }
  }
}

void FFT_UTOFU_BG::init(int nx_pppm, int ny_pppm, int nz_pppm,
  int nxlo_in, int nylo_in, int nzlo_in, int nxhi_in, int nyhi_in, int nzhi_in) {

  nodegrid = comm->nodegrid;
  nodeloc = comm->nodeloc;
  fft_size[0] = nx_pppm;
  fft_size[1] = ny_pppm;
  fft_size[2] = nz_pppm; 

  lcl_size[0] = nxhi_in - nxlo_in + 1;
  lcl_size[1] = nyhi_in - nylo_in + 1;
  lcl_size[2] = nzhi_in - nzlo_in + 1;

  nfft = fft_size[0] * fft_size[1] * fft_size[2];
  nfft_brick = lcl_size[0] * lcl_size[1] * lcl_size[2];

  nblocks[0] = lcl_size[1] * lcl_size[2];
  nblocks[1] = lcl_size[0] * lcl_size[2];
  nblocks[2] = lcl_size[1] * lcl_size[0];

  dgemm_size[0] = (int)fft_size[0] * nblocks[0];
  dgemm_size[1] = (int)fft_size[1] * nblocks[1];
  dgemm_size[2] = (int)fft_size[2] * nblocks[2]; 

  utils::logmesg(lmp, "[INFO] FFT_UTOFU npppm {} {} {} \n", nx_pppm,ny_pppm,nz_pppm);
  utils::logmesg(lmp, "[INFO] FFT_UTOFU fft_out  {}-{} {}-{} {}-{} \n", nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in);
  utils::logmesg_arry(lmp,fmt::format("[info] FFT_UTOFU fft_size \n"), fft_size, 3, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] FFT_UTOFU lcl_size \n"), lcl_size, 3, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] FFT_UTOFU nblocks \n"), nblocks, 3, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] FFT_UTOFU dgemm_size \n"), dgemm_size, 3, 1);

  maxGemmsize = MAX(MAX(dgemm_size[0], dgemm_size[1]), dgemm_size[2]);

  for(int dir = 0; dir < 3; dir++) {
    memory->create(Wcos[dir],fft_size[dir] * lcl_size[dir],"pppm:Wcos");
    memory->create(Wsin[dir],fft_size[dir] * lcl_size[dir],"pppm:Wsin");
    memory->create(Wcos_i[dir],fft_size[dir] * lcl_size[dir],"pppm:Wcos_i");
    memory->create(Wsin_i[dir],fft_size[dir] * lcl_size[dir],"pppm:Wsin_i");
  }
  memory->create(calcu_buf, maxGemmsize * 4,"pppm:calcu_buf");
  memory->create(reduce_data, comm->nnode * 2,"pppm:calcu_buf");
  
  
  for(int dir = 0; dir < 3; dir++) {
    memory->create(nfft_bricks[dir],        comm->comm1D_size[dir],"pppm:nfft_bricks[dir]");
    memory->create(nfft_bricks_offset[dir], comm->comm1D_size[dir],"pppm:nfft_bricks_offset[dir]");
  }
  
  for(int dir = 0; dir < 3; dir++) {
    MPI_Allgather(&nfft_brick, 1, MPI_INT, nfft_bricks[dir],        1, MPI_INT, comm->comm1D[dir]);
    max_nfft_brick[dir] = 0;
    for(int i = 0; i < comm->comm1D_size[dir]; i++) 
      if(max_nfft_brick[dir] < nfft_bricks[dir][i]) 
        max_nfft_brick[dir] = nfft_bricks[dir][i];

    utils::logmesg_arry(lmp, fmt::format("[INFO] nfft_bricks dir {}", dir), nfft_bricks[dir], comm->comm1D_size[dir], 1);
  }

  nfft_bricks_offset[0][0] = nfft_bricks_offset[1][0] = nfft_bricks_offset[2][0] = 0;
  for(int dir = 0; dir < 3; dir++) {
    for(int i = 1; i < comm->comm1D_size[dir]; i++) {
      nfft_bricks_offset[dir][i] = nfft_bricks_offset[dir][i-1] + nfft_bricks[dir][i-1];
    }
    utils::logmesg_arry(lmp, fmt::format("[INFO] nfft_bricks_offset dir {}", dir), nfft_bricks_offset[dir], comm->comm1D_size[dir], 1);
  }

  init_local_W(Wsin, Wcos, Wsin_i, Wcos_i, fft_size, lcl_size, nodeloc);

  for(int dir = 0; dir < 3; dir++) {
    utils::logmesg_arry(lmp, fmt::format("[INFO] Wcos dir {}", dir), Wcos[dir], fft_size[dir]*lcl_size[dir], 1);
    utils::logmesg_arry(lmp, fmt::format("[INFO] Wsin dir {}", dir), Wsin[dir], fft_size[dir]*lcl_size[dir], 1);
  }

}

void FFT_UTOFU_BG::init_utofu_bg() {
  int pre[3];
  int nxt[3];

  for(int dir = 0; dir < 3; dir++) {
    pre[dir] = (comm->me3d[dir] + comm->comm1D_size[dir] - 1) % comm->comm1D_size[dir];
    nxt[dir] = (comm->me3d[dir] + 1) % comm->comm1D_size[dir];
  }

  for(int dir = 0; dir < 3; dir++) {
    for(int tni = dir*2; tni < (dir+1)*2; tni++) {
      for(int r = 0; r < comm->comm1D_size[dir]; r++) {
        // rc = utofu_alloc_vbg(tni, 2, 0, lcl_vbg_ids[tni][r]);
        // if (rc != UTOFU_SUCCESS) {
        //   error->one(FLERR,"utofu_alloc_vbg failed  tni {} r {} rc {} \n", tni, r, rc);
        // }
        lcl_vbg_ids[tni][r][0] = comm->lcl_vbg_ids[tni][r][0];
        lcl_vbg_ids[tni][r][1] = comm->lcl_vbg_ids[tni][r][1];
        MPI_Allgather(lcl_vbg_ids[tni][r], 2, MPI_UINT64_T, rmt_vbg_ids[tni][r][0], 2, MPI_UINT64_T, comm->comm1D[dir]);
      }
    }
  }

  for(int dir = 0; dir < 3; dir++) {
    for(int tni = dir*2; tni < (dir+1)*2; tni++) {
      for(int r = 0; r < comm->comm1D_size[dir]; r++) {
        vbg_settings[tni][r][0].vbg_id = lcl_vbg_ids[tni][r][0];
        vbg_settings[tni][r][0].src_lcl_vbg_id = lcl_vbg_ids[tni][r][1];
        vbg_settings[tni][r][0].src_rmt_vbg_id = UTOFU_VBG_ID_NULL;
        vbg_settings[tni][r][0].dst_lcl_vbg_id = lcl_vbg_ids[tni][r][1];
        vbg_settings[tni][r][0].dst_rmt_vbg_id = UTOFU_VBG_ID_NULL;
        vbg_settings[tni][r][0].dst_path_coords[0] = UTOFU_PATH_COORD_NULL;

        vbg_settings[tni][r][1].vbg_id = lcl_vbg_ids[tni][r][1];
        vbg_settings[tni][r][1].src_lcl_vbg_id = lcl_vbg_ids[tni][r][0];
        vbg_settings[tni][r][1].src_rmt_vbg_id = rmt_vbg_ids[tni][r][pre[dir]][1];
        vbg_settings[tni][r][1].dst_lcl_vbg_id = lcl_vbg_ids[tni][r][0];
        vbg_settings[tni][r][1].dst_rmt_vbg_id = rmt_vbg_ids[tni][r][nxt[dir]][1];
        vbg_settings[tni][r][1].dst_path_coords[0] = UTOFU_PATH_COORD_NULL;

        if(r == comm->me3d[dir]) {
          vbg_settings[tni][r][0].src_lcl_vbg_id = UTOFU_VBG_ID_NULL;
          vbg_settings[tni][r][0].src_rmt_vbg_id = rmt_vbg_ids[tni][r][pre[dir]][1];

          vbg_settings[tni][r][1].src_rmt_vbg_id = UTOFU_VBG_ID_NULL;
          vbg_settings[tni][r][1].dst_lcl_vbg_id = UTOFU_VBG_ID_NULL;
        } else if(r == nxt[dir]) {
          vbg_settings[tni][r][1].dst_rmt_vbg_id = rmt_vbg_ids[tni][r][nxt[dir]][0];
        }

        rc = utofu_set_vbg(vbg_settings[tni][r], 2);
        if (rc != UTOFU_SUCCESS) {
          error->one(FLERR,"utofu_set_vbg failed rc {} \n",rc);
        }
        MPI_Barrier(comm->comm1D[dir]);
        utils::logmesg(lmp, "[INFO] utofu_set_vbg success dir {} tni {} r {} \n", dir, tni, r);
      }
    }
  }
  MPI_Barrier(comm->numa_comm);
}

inline void print_array_2d(double *buf, int M, int N, std::string mesg, LAMMPS *lmp) {
  std::string tmp;
  tmp += fmt::format("[INFO] print_array_2d  {}\n", mesg.c_str());

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      tmp += fmt::format(" {:<3.9f},", buf[i*N+j]);
    }
    tmp += "\n";
  }
  utils::logmesg(lmp, tmp);
}

inline void gather_inner_data_printf(int nfft_brick, int nnode, MPI_Comm numa_comm,
  FFT_SCALAR *in_data, int *nodegrid, int *lcl_size, int * fft_size, int N, int nfft, LAMMPS *lmp, std::string mesg) {
  std::vector<FFT_SCALAR> recv_data(nfft_brick * nnode);
  std::vector<FFT_SCALAR> all_data(nfft_brick * nnode);
  MPI_Gather(in_data, nfft_brick, MPI_DOUBLE, recv_data.data(), nfft_brick, MPI_DOUBLE, 0, numa_comm);

  for (int pzIdx = 0; pzIdx < nodegrid[2]; ++pzIdx) {
    for (int pyIdx = 0; pyIdx < nodegrid[1]; ++pyIdx) {
      for (int pxIdx = 0; pxIdx < nodegrid[0]; ++pxIdx) {
            int procRank = pzIdx * (nodegrid[1] * nodegrid[0]) + pyIdx * nodegrid[0] + pxIdx;
            int baseX = pxIdx * lcl_size[0];
            int baseY = pyIdx * lcl_size[1];
            int baseZ = pzIdx * lcl_size[2];
            int offset = nfft_brick * procRank;

            // 重新映射到全局网格
            for (int k = 0; k < lcl_size[2]; ++k) {
              for (int j = 0; j < lcl_size[1]; ++j) {
                for (int i = 0; i < lcl_size[0]; ++i) {
                      int ptr = (baseZ + k) * fft_size[1] * fft_size[0] + (baseY + j) * fft_size[0] + (baseX + i);
                      all_data[ptr]  = recv_data[offset++];
                    }
                }
            }
        }
    }
  }

  // std::string tmp;
  // tmp += fmt::format("[INFO] PPPMDPLR recv_data {}\n", mesg.c_str());
  // tmp += "{";
  // for(int i = 0; i < nfft ;i += 1) {
  //    tmp += fmt::format(" {:<3.9f},", recv_data[i]);
  // }
  // tmp += "}\n";
  // utils::logmesg(lmp, tmp);

  // tmp = "";
  // tmp += fmt::format("[INFO] PPPMDPLR all_data {}\n", mesg.c_str());
  // tmp += "{";
  // for(int i = 0; i < nfft ;i += 1) {
  //    tmp += fmt::format(" {:<3.9f},", all_data[i]);
  // }
  // tmp += "}\n";
  // utils::logmesg(lmp, tmp);

  // utils::logmesg_arry(lmp, fmt::format("PPPMDPLR in_data dir \n"),in_data,2*nfft_brick, 1 );
  // utils::logmesg_arry(lmp, fmt::format("\nPPPMDPLR all_data {}\n", mesg.c_str()),all_data.data(), nfft, 1 );
  print_array_2d(all_data.data(), nfft_brick*nnode/N, N , fmt::format("\nPPPMDPLR all_data {}\n", mesg.c_str()), lmp);

}

void FFT_UTOFU_BG::compute_fft3D_forward(FFT_SCALAR *in_data, int FFT_DIR) {
  FFT_SCALAR *_Wcos[3], *_Wsin[3];

  if(FFT_DIR == 0) {
    for(int dir = 0; dir < 3; dir++) _Wcos[dir] = Wcos[dir];
    for(int dir = 0; dir < 3; dir++) _Wsin[dir] = Wsin[dir];
  } else {
    for(int dir = 0; dir < 3; dir++) _Wcos[dir] = Wcos_i[dir];
    for(int dir = 0; dir < 3; dir++) _Wsin[dir] = Wsin_i[dir];

  }


  // gather_inner_data_printf(nfft_brick, comm->nnode, comm->numa_comm,
  //   in_data, nodegrid, lcl_size, fft_size , fft_size[0], nfft, lmp, "input data" );

  FFT_SCALAR *cos_out[2], *sin_out[2];
  for(int dir = 0; dir < 3; dir++) {
    cos_out[0] = calcu_buf;
    sin_out[0] = cos_out[0] + dgemm_size[dir];
    cos_out[1] = sin_out[0] + dgemm_size[dir];
    sin_out[1] = cos_out[1] + dgemm_size[dir];

    FFT_SCALAR *in_data_real = in_data;
    FFT_SCALAR *in_data_imag = in_data + nfft_brick;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, fft_size[dir], nblocks[dir], lcl_size[dir],
      1.0, _Wcos[dir], lcl_size[dir],
      in_data_real, lcl_size[dir],
      0.0, cos_out[0], nblocks[dir]
    );
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, fft_size[dir], nblocks[dir], lcl_size[dir],
      1.0, _Wsin[dir], lcl_size[dir],
      in_data_imag, lcl_size[dir],
      0.0, cos_out[1], nblocks[dir]
    );
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, fft_size[dir], nblocks[dir], lcl_size[dir],
      1.0, _Wsin[dir], lcl_size[dir],
      in_data_real, lcl_size[dir], 
      0.0, sin_out[0], nblocks[dir]
    );
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, fft_size[dir], nblocks[dir], lcl_size[dir],
      1.0, _Wcos[dir], lcl_size[dir],
      in_data_imag, lcl_size[dir], 
      0.0, sin_out[1], nblocks[dir]
    );

    for(int i = 0; i < dgemm_size[dir]; i++) {
      cos_out[0][i] -= cos_out[1][i];
      sin_out[0][i] += sin_out[1][i];
    }

    // utils::logmesg_arry(lmp, fmt::format("PPPMDPLR cos_out dir {} \n", dir),calcu_buf,2*dgemm_size[dir], 1 );


    memset(in_data, 0, nfft_brick * sizeof(FFT_SCALAR) *2);

    // MPI_Request requests[1000];
    // int nrequst = 0;

    // for(int r = 0; r < comm->comm1D_size[dir]; r++) {
    //   MPI_Ireduce(&cos_out[0][nfft_bricks_offset[dir][r]],  in_data,                     nfft_bricks[dir][r], MPI_DOUBLE, MPI_SUM, r, comm->comm1D[dir], &requests[nrequst++]);
    //   MPI_Ireduce(&sin_out[0][nfft_bricks_offset[dir][r]],  in_data+nfft_bricks[dir][r], nfft_bricks[dir][r], MPI_DOUBLE, MPI_SUM, r, comm->comm1D[dir], &requests[nrequst++]);
    // }
    // MPI_Waitall(nrequst, requests, MPI_STATUS_IGNORE);

    // gather_inner_data_printf(nfft_brick, comm->nnode, comm->numa_comm,
    //   in_data, nodegrid, lcl_size, fft_size ,  fft_size[(dir+1)%3 ] ,nfft, lmp, fmt::format("out dir {}", dir) );

    MPI_Barrier(comm->comm1D[dir]);
    
    for(int itype = 0; itype < 2; itype++) {
      double *_in_buf = in_data + itype * nfft_brick;
      for(int p = 0; p < max_nfft_brick[dir]; p+=2) {
        for(int r = 0; r < comm->comm1D_size[dir]; r++) {
          int size = nfft_bricks[dir][r] - p;
          if(size <= 0) continue;
          if(size > 2) size = 2;
          // for(int i = 0; i < size; i++) 
            // reduce_data[r*2+i] = calcu_buf[nfft_bricks_offset[dir][r] * 2 + p + i];
          // utofu_reduce_double(lcl_vbg_ids[dir*2+0][r][0], UTOFU_REDUCE_OP_BFPSUM, &reduce_data[r*2], size, 0);
          double *_out_buf = itype == 0 ? &cos_out[0][nfft_bricks_offset[dir][r]] : &sin_out[0][nfft_bricks_offset[dir][r]];
          utofu_reduce_double(lcl_vbg_ids[dir*2+0][r][0], UTOFU_REDUCE_OP_BFPSUM, 
                        &_out_buf[p], size, 0);
        }

        for(int r = 0; r < comm->comm1D_size[dir]; r++) {
          int size = nfft_bricks[dir][r] - p;
          if(size <= 0) continue;
          if(size > 2) size = 2;
          int nerr = 0;
          do {
            rc = utofu_poll_reduce_double(lcl_vbg_ids[dir*2][r][0], 0, &reduce_data[r*2]);
          } while (rc == UTOFU_ERR_NOT_COMPLETED);
          // printf("utofu_reduce_double recv ring %d \n", r); fflush(stdout);
          if(rc != UTOFU_SUCCESS) error->one(FLERR,"utofu_poll_reduce_double fail {} ", rc);

          if(r == comm->me3d[dir]) {
            for(int i = 0; i < size; i++) _in_buf[p+i] = reduce_data[r*2+i]; 
          };
        }
        // MPI_Barrier(comm->comm1D[dir]);
      }
    }
    // utils::logmesg_arry(lmp, fmt::format("PPPMDPLR result dir {} \n", dir),in_data,2*nfft_brick, 1 );

  }

  gather_inner_data_printf(nfft_brick, comm->nnode, comm->numa_comm,
    in_data, nodegrid, lcl_size, fft_size ,  fft_size[0] ,nfft, lmp, fmt::format("forward out dir real {}", 0) );
  // gather_inner_data_printf(nfft_brick, comm->nnode, comm->numa_comm,
  //   in_data+nfft_brick, nodegrid, lcl_size, fft_size ,  fft_size[0] ,nfft, lmp, fmt::format("forward out dir imag {}", 0) );
}

/* ---------------------------------------------------------------------- */

PPPMDPLR::PPPMDPLR(LAMMPS *lmp) :
  PPPM(lmp)
{
  triclinic_support = 1;
  x_node = nullptr;
  part2grid_node = nullptr;
  vg_brick = nullptr;
  fkx_brick = nullptr; fky_brick = nullptr; fkz_brick = nullptr;
  work1_node = work2_node = nullptr;

  fft_utofu = new FFT_UTOFU_BG(lmp);
}

/* ---------------------------------------------------------------------- */

void PPPMDPLR::init()
{
  // DPLR PPPM requires newton on, b/c it computes forces on ghost atoms

  if (force->newton == 0)
    error->all(FLERR,"Kspace style pppm/dplr requires newton on");

  PPPM::init();
  init_heffte_fft();
  init_node_fft();

  MPI_Barrier(MPI_COMM_WORLD);

  utils::logmesg(lmp, "[INFO] finish init_node_fft \n");

  if(comm->numa_id == NUMA_NUM - 1) {
    fft_utofu->init(nx_pppm, ny_pppm, nz_pppm,
      nxlo_node_in, nylo_node_in, nzlo_node_in, nxhi_node_in, nyhi_node_in, nzhi_node_in);

    utils::logmesg(lmp, "[INFO] finish fft_utofu->init \n");
    
    fft_utofu->init_utofu_bg();
    utils::logmesg(lmp, "[INFO] finish fft_utofu->init_utofu_bg \n");
  }
}

void PPPMDPLR::init_heffte_fft() {

  box_pos = new heffte::box3d<>({{nxlo_in, nylo_in, nzlo_in},
    { nxhi_in, nyhi_in, nzhi_in}});

  heffte_wrapper = new heffte::fft3d<heffte::backend::fftw>(*box_pos,
                *box_pos,MPI_COMM_WORLD);
  heffte_indata = new std::complex<FFT_SCALAR>[heffte_wrapper->size_inbox()];
  heffte_outdata = new std::complex<FFT_SCALAR>[heffte_wrapper->size_outbox()];

  utils::logmesg(lmp, "[INFO] PPPMDPLR data size {} {} \n", 
    heffte_wrapper->size_inbox(),heffte_wrapper->size_outbox());
}

void PPPMDPLR::init_node_fft() {
  MPI_Allreduce(&nxlo_out, &nxlo_node_out, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nxhi_out, &nxhi_node_out, 1, MPI_INT, MPI_MAX, comm->node_comm);
  MPI_Allreduce(&nylo_out, &nylo_node_out, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nyhi_out, &nyhi_node_out, 1, MPI_INT, MPI_MAX, comm->node_comm);
  MPI_Allreduce(&nzlo_out, &nzlo_node_out, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nzhi_out, &nzhi_node_out, 1, MPI_INT, MPI_MAX, comm->node_comm);

  MPI_Allreduce(&nxlo_in, &nxlo_node_in, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nxhi_in, &nxhi_node_in, 1, MPI_INT, MPI_MAX, comm->node_comm);
  MPI_Allreduce(&nylo_in, &nylo_node_in, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nyhi_in, &nyhi_node_in, 1, MPI_INT, MPI_MAX, comm->node_comm);
  MPI_Allreduce(&nzlo_in, &nzlo_node_in, 1, MPI_INT, MPI_MIN, comm->node_comm);
  MPI_Allreduce(&nzhi_in, &nzhi_node_in, 1, MPI_INT, MPI_MAX, comm->node_comm);

  utils::logmesg(lmp, "[INFO] PPPM nxlo_node_out  {}-{} {}-{} {}-{} \n", nxlo_node_out,nxhi_node_out,nylo_node_out,nyhi_node_out,nzlo_node_out,nzhi_node_out);
  utils::logmesg(lmp, "[INFO] PPPM nxlo_node_out  {}-{} {}-{} {}-{} \n", nxlo_node_in,nxhi_node_in,nylo_node_in,nyhi_node_in,nzlo_node_in,nzhi_node_in);

  ngrid_node = (nxhi_node_out-nxlo_node_out+1) * (nyhi_node_out-nylo_node_out+1) *
    (nzhi_node_out-nzlo_node_out+1);

  nfft_node_brick = (nxhi_node_in-nxlo_node_in+1) * (nyhi_node_in-nylo_node_in+1) *
    (nzhi_node_in-nzlo_node_in+1);
  
  memory->create3d_offset(density_brick_node,nzlo_node_out,nzhi_node_out,nylo_node_out,nyhi_node_out,
      nxlo_node_out,nxhi_node_out,"pppm:density_brick_node");

  utils::logmesg(lmp, "[INFO] ngrid_node {} nfft_node_brick {}\n", ngrid_node, nfft_node_brick);

  memory->create(density_fft_node,ngrid_node,"pppm:density_fft_node");

  int *nodeloc = comm->nodeloc;
  int *nodegrid = comm->nodegrid;
  maxswap = 26;
  swap = new Swap[maxswap];

  int neiproc[26][2];
  int neipbc[26][3];
  int neidirec[26][3];

  int _max_buf_size = 0;


  int iswap = 0;
  nswap = 26; 
  if(comm->numa_id == NUMA_NUM - 1) {
    int **neigh_fft_prd_out, **neigh_fft_prd_in;
    memory->create(neigh_fft_prd_out,comm->nnode,6,"comm:neigh_fft_prd_out");
    memory->create(neigh_fft_prd_in,comm->nnode,6,"comm:neigh_fft_prd_in");

    int lcl_fft_prd_in[6]  = {nxlo_node_in, nxhi_node_in, nylo_node_in, nyhi_node_in, nzlo_node_in, nzhi_node_in};
    int lcl_fft_prd_out[6] = {nxlo_node_out, nxhi_node_out, nylo_node_out, nyhi_node_out, nzlo_node_out, nzhi_node_out};
    MPI_Allgather(lcl_fft_prd_out,6,MPI_INT,neigh_fft_prd_out[0],6,MPI_INT,comm->numa_comm);
    MPI_Allgather(lcl_fft_prd_in, 6,MPI_INT,neigh_fft_prd_in[0], 6,MPI_INT,comm->numa_comm);

    utils::logmesg(lmp, "[INFO] finish allgather fft_prd\n");

    for(int dim = 0; dim < 13; dim++) {
      for(int ineed = 0; ineed < 2; ineed ++) {
        int neiloc[3];
        for(int j = 0; j < 3; j++) {
          if(ineed % 2 == 0)
            neiloc[j] = (nodeloc[j] - con_direction[dim][j] + nodegrid[j]) % nodegrid[j];
          else
            neiloc[j] = (nodeloc[j] + con_direction[dim][j] + nodegrid[j]) % nodegrid[j];
        };

        int neinode = (neiloc[2] * nodegrid[1] + neiloc[1]) * nodegrid[0] + neiloc[0];

        neiproc[dim][ineed] = neinode * NUMA_NUM + comm->numa_id;
      }
    }


    for(int dim = 0; dim < 13; dim++) {
      for(int ineed = 0; ineed < 2; ineed++) {
        if(ineed %2 == 0){
          swap[iswap].sendproc = neiproc[dim][0];
          swap[iswap].recvproc = neiproc[dim][1];
          neidirec[iswap][0] = con_direction[dim][0]; 
          neidirec[iswap][1] = con_direction[dim][1];
          neidirec[iswap][2] = con_direction[dim][2];
        } else {
          swap[iswap].sendproc = neiproc[dim][1];
          swap[iswap].recvproc = neiproc[dim][0];
          neidirec[iswap][0] = -1 * con_direction[dim][0]; 
          neidirec[iswap][1] = -1 * con_direction[dim][1];
          neidirec[iswap][2] = -1 * con_direction[dim][2];
        }
        iswap++;
      }

    }

    for(int i = 0; i < 3; i++) {
      for(int iswap = 0; iswap < 26; iswap++) {
        if(nodeloc[i] - neidirec[iswap][i] < 0) 
          neipbc[iswap][i]   = 1;
        else if(nodeloc[i] - neidirec[iswap][i] >= nodegrid[i]) 
          neipbc[iswap][i]   = -1;
        else
          neipbc[iswap][i]   = 0;
      }
    }

    utils::logmesg(lmp, "[INFO] PPPM send pack list \n");
    utils::logmesg(lmp, "[INFO] PPPM iswap prd   {}-{} {}-{} {}-{} \n", 
              lcl_fft_prd_out[0], lcl_fft_prd_out[1], lcl_fft_prd_out[2], lcl_fft_prd_out[3], lcl_fft_prd_out[4], lcl_fft_prd_out[5]);
    for(int iswap = 0; iswap < 26; iswap++) {
      int sendnode = swap[iswap].sendproc / NUMA_NUM;
      int nei_prd_in[6] = { neigh_fft_prd_in[sendnode][0] - neipbc[iswap][0] * nx_pppm, 
                            neigh_fft_prd_in[sendnode][1] - neipbc[iswap][0] * nx_pppm, 
                            neigh_fft_prd_in[sendnode][2] - neipbc[iswap][1] * ny_pppm, 
                            neigh_fft_prd_in[sendnode][3] - neipbc[iswap][1] * ny_pppm, 
                            neigh_fft_prd_in[sendnode][4] - neipbc[iswap][2] * nz_pppm, 
                            neigh_fft_prd_in[sendnode][5] - neipbc[iswap][2] * nz_pppm
                          };
      double x_min_pack = std::max(lcl_fft_prd_out[0], nei_prd_in[0]);
      double x_max_pack = std::min(lcl_fft_prd_out[1], nei_prd_in[1]);
      double y_min_pack = std::max(lcl_fft_prd_out[2], nei_prd_in[2]);
      double y_max_pack = std::min(lcl_fft_prd_out[3], nei_prd_in[3]);
      double z_min_pack = std::max(lcl_fft_prd_out[4], nei_prd_in[4]);
      double z_max_pack = std::min(lcl_fft_prd_out[5], nei_prd_in[5]);

      if (x_min_pack > x_max_pack || y_min_pack > y_max_pack || z_min_pack > z_max_pack) {
        swap[iswap].npack = 0;
        utils::logmesg(lmp, "[INFO] PPPM iswap prd in    {} {}   {}-{} {}-{} {}-{} npack {}\n", 
            iswap, swap[iswap].sendproc, nei_prd_in[0], nei_prd_in[1], nei_prd_in[2], nei_prd_in[3], nei_prd_in[4], nei_prd_in[5], swap[iswap].npack);
        continue;
      }

      swap[iswap].packlist   = new int[ngrid_node];
      // swap[iswap].send_buf   = new FFT_SCALAR[ngrid_node];
      
      int n = 0;
      int ix,iy,iz;
      int nx = (nxhi_node_out-nxlo_node_out+1);
      int ny = (nyhi_node_out-nylo_node_out+1);
      for (iz = z_min_pack; iz <= z_max_pack; iz++)
        for (iy = y_min_pack; iy <= y_max_pack; iy++)
          for (ix = x_min_pack; ix <= x_max_pack; ix++)
            swap[iswap].packlist[n++] = (iz-nzlo_node_out)*ny*nx + (iy-nylo_node_out)*nx + (ix-nxlo_node_out);
      
      swap[iswap].npack    = n;

      if(n > _max_buf_size) _max_buf_size = n;
      // swap[iswap].send_buf = new FFT_SCALAR[n*3];

      utils::logmesg(lmp, "[INFO] PPPM iswap prd in    {} {}   {}-{} {}-{} {}-{} npack {}\n", 
              iswap, swap[iswap].sendproc, nei_prd_in[0], nei_prd_in[1], nei_prd_in[2], nei_prd_in[3], nei_prd_in[4], nei_prd_in[5], n);
    }
    //   utils::logmesg(lmp, "[INFO] PPPM iswap prd in     {}    {}-{} {}-{} {}-{} \n", 
    //           nswap, nei_prd_in[0], nei_prd_in[1], nei_prd_in[2], nei_prd_in[3], nei_prd_in[4], nei_prd_in[5]);
    //   utils::logmesg(lmp, "[INFO] PPPM iswap pack       {}    {}-{} {}-{} {}-{} \n", 
    //           nswap, x_min_pack,x_max_pack,y_min_pack,y_max_pack,z_min_pack,z_max_pack);
    
    utils::logmesg(lmp, "[INFO] PPPM recv unpack list \n");
    utils::logmesg(lmp, "[INFO] PPPM iswap prd   {}-{} {}-{} {}-{} \n", 
        lcl_fft_prd_in[0], lcl_fft_prd_in[1], lcl_fft_prd_in[2], lcl_fft_prd_in[3], lcl_fft_prd_in[4], lcl_fft_prd_in[5]);
    for(int iswap = 0; iswap < 26; iswap++) {
      int rswap = (iswap % 2 == 0) ? iswap + 1 : iswap - 1;

      int recvnode = swap[iswap].recvproc / NUMA_NUM;
      int nei_prd_out[6] = {neigh_fft_prd_out[recvnode][0] - neipbc[rswap][0] * nx_pppm, 
                            neigh_fft_prd_out[recvnode][1] - neipbc[rswap][0] * nx_pppm, 
                            neigh_fft_prd_out[recvnode][2] - neipbc[rswap][1] * ny_pppm, 
                            neigh_fft_prd_out[recvnode][3] - neipbc[rswap][1] * ny_pppm, 
                            neigh_fft_prd_out[recvnode][4] - neipbc[rswap][2] * nz_pppm, 
                            neigh_fft_prd_out[recvnode][5] - neipbc[rswap][2] * nz_pppm
                          };
      double x_min_unpack = std::max(lcl_fft_prd_in[0], nei_prd_out[0]);
      double x_max_unpack = std::min(lcl_fft_prd_in[1], nei_prd_out[1]);
      double y_min_unpack = std::max(lcl_fft_prd_in[2], nei_prd_out[2]);
      double y_max_unpack = std::min(lcl_fft_prd_in[3], nei_prd_out[3]);
      double z_min_unpack = std::max(lcl_fft_prd_in[4], nei_prd_out[4]);
      double z_max_unpack = std::min(lcl_fft_prd_in[5], nei_prd_out[5]);

      if (x_min_unpack > x_max_unpack || y_min_unpack > y_max_unpack || z_min_unpack > z_max_unpack) {
        swap[iswap].nunpack = 0;
        utils::logmesg(lmp, "[INFO] PPPM iswap prd out    {} {}   {}-{} {}-{} {}-{} unnpack {}\n", 
          iswap, swap[iswap].recvproc, nei_prd_out[0], nei_prd_out[1], nei_prd_out[2], nei_prd_out[3], nei_prd_out[4], nei_prd_out[5], swap[iswap].nunpack);
        continue;
      }

      swap[iswap].unpacklist   = new int[ngrid_node];
      
      int n = 0;
      int ix,iy,iz;
      int nx = (nxhi_node_out-nxlo_node_out+1);
      int ny = (nyhi_node_out-nylo_node_out+1);
      for (iz = z_min_unpack; iz <= z_max_unpack; iz++)
        for (iy = y_min_unpack; iy <= y_max_unpack; iy++)
          for (ix = x_min_unpack; ix <= x_max_unpack; ix++)
            swap[iswap].unpacklist[n++] = (iz-nzlo_node_out)*ny*nx + (iy-nylo_node_out)*nx + (ix-nxlo_node_out);
      
      swap[iswap].nunpack    = n;
      if(n > _max_buf_size) _max_buf_size = n;
      // swap[iswap].recv_buf   = new FFT_SCALAR[n*3];

      utils::logmesg(lmp, "[INFO] PPPM iswap prd out    {} {}   {}-{} {}-{} {}-{} unnpack {}\n", 
        iswap, swap[iswap].recvproc, nei_prd_out[0], nei_prd_out[1], nei_prd_out[2], nei_prd_out[3], nei_prd_out[4], nei_prd_out[5], n);
    }
  }

  for(int iswap = 0; iswap < 26; iswap++) {
    swap[iswap].send_buf = new FFT_SCALAR[3*_max_buf_size];
    swap[iswap].recv_buf = new FFT_SCALAR[3*_max_buf_size];
  }

  // for(int iswap = 0; iswap < nswap; iswap++) {
  //   utils::logmesg(lmp, "[INFO] iswap {} sendproc {} npack {}  recvproc {} nunpack {} \n", iswap, 
  //     swap[iswap].sendproc, swap[iswap].npack, swap[iswap].recvproc, swap[iswap].nunpack);
  // }
  memory->create(work1_node, nfft_node_brick * 2, "PPPM::work1_node");
  memory->create(work2_node, nfft_node_brick * 2, "PPPM::work2_node");

  memory->create3d_offset(vdx_node,nzlo_node_out,nzhi_node_out,nylo_node_out,nyhi_node_out,
    nxlo_node_out,nxhi_node_out,"pppm:vdx_node");
  memory->create3d_offset(vdy_node,nzlo_node_out,nzhi_node_out,nylo_node_out,nyhi_node_out,
    nxlo_node_out,nxhi_node_out,"pppm:vdy_node");
  memory->create3d_offset(vdz_node,nzlo_node_out,nzhi_node_out,nylo_node_out,nyhi_node_out,
    nxlo_node_out,nxhi_node_out,"pppm:vdz_node");

  MPI_Barrier(MPI_COMM_WORLD);

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

  utils::logmesg(lmp,"[INFO] begin PPPMDPLR::compute \n"); MPI_Barrier(MPI_COMM_WORLD);


  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag,vflag);

  // utils::logmesg(lmp,"[INFO] eflag_atom  {} vflag_atom {}\n", eflag_atom, vflag_atom);
  utils::logmesg(lmp,"[INFO] evflag_atom  {} peratom_allocate_flag {}\n", evflag_atom, peratom_allocate_flag);


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
  // if(comm->numa_id == NUMA_NUM - 1) {
  memory->destroy(part2grid_node);
  memory->destroy(x_node);
  memory->create(part2grid_node,nmax,3,"pppm:part2grid_node");
  memory->create(x_node, nmax, 3, "pppm:x_node");
  memory->create(q_node, nmax, "pppm:q_node");
  // }

  utils::logmesg(lmp,"[INFO] finish q_node create {} \n", nmax); MPI_Barrier(MPI_COMM_WORLD);

  nlocal_node = 0;
  MPI_Allgather(&atom->nlocal,1,MPI_INT,nlocal_nodes,1,MPI_INT,comm->node_comm);
  for(int i = 0; i < NUMA_NUM; i++) nlocal_node += nlocal_nodes[i];

  utils::logmesg_arry(lmp,fmt::format("[info] before particle map nlocal_nodes  {}\n", nlocal_node), nlocal_nodes, NUMA_NUM, 1);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // gather node x
  int displs[NUMA_NUM], recvcounts[NUMA_NUM];
  for(int i = 0; i < NUMA_NUM; i++) recvcounts[i] = nlocal_nodes[i] * 3;
  displs[0] = 0;
  for (int i = 1; i < NUMA_NUM; i++) displs[i] = displs[i - 1] + recvcounts[i - 1];
  MPI_Gatherv(atom->x[0], atom->nlocal * 3, MPI_DOUBLE, x_node[0], recvcounts, displs, MPI_DOUBLE, (NUMA_NUM - 1), comm->node_comm);

  // gather q_node
  for(int i = 0; i < NUMA_NUM; i++) recvcounts[i] = nlocal_nodes[i];
  displs[0] = 0;
  for (int i = 1; i < NUMA_NUM; i++) displs[i] = displs[i - 1] + recvcounts[i - 1];
  MPI_Gatherv(atom->q, atom->nlocal, MPI_DOUBLE, q_node, recvcounts, displs, MPI_DOUBLE, (NUMA_NUM - 1), comm->node_comm);
  

  utils::logmesg_arry(lmp,fmt::format("[info] before particle map recvcounts   nmax {}\n", nmax), recvcounts,      NUMA_NUM, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] before particle map displs        {}\n", nlocal_node), displs,      NUMA_NUM, 1);
    // utils::logmesg_arry_x(lmp,fmt::format("[info] before particle map atom x  {}\n", atom->nlocal), atom->x[0], atom->nlocal * 3, 1);
  
  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid
  utils::logmesg_arry_x(lmp,fmt::format("[info] before particle map atom x  {}\n", atom->nlocal), atom->x[0], atom->nlocal * 3, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] before particle map atom q  {}\n", atom->nlocal), atom->q, atom->nlocal, 1);
  if(comm->numa_id == NUMA_NUM - 1) {
    utils::logmesg_arry_x(lmp,fmt::format("[info] before particle map atom x  {}\n", nlocal_node), x_node[0], nlocal_node * 3, 1);
    utils::logmesg_arry(lmp,fmt::format("[info] before particle map atom q  {}\n", nlocal_node), q_node, nlocal_node, 1);
  }

  
  particle_map();

  utils::logmesg(lmp,"[INFO] finish particle_map \n"); MPI_Barrier(MPI_COMM_WORLD);
  
  make_rho();
  utils::logmesg(lmp,"[INFO] finish make_rho \n"); MPI_Barrier(MPI_COMM_WORLD);
  
  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition
  
  gc->reverse_comm(Grid3d::KSPACE, this, REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                          gc_buf1, gc_buf2, MPI_FFT_SCALAR);
  
  reverse_node();
  utils::logmesg(lmp,"[INFO] finish reverse comm \n"); MPI_Barrier(MPI_COMM_WORLD);
  

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

  forward_node();
  // extra per-atom energy/virial communication

  if (evflag_atom) {
      gc->forward_comm(Grid3d::KSPACE, this, FORWARD_IK_PERATOM, 7,
                       sizeof(FFT_SCALAR), gc_buf1, gc_buf2, MPI_FFT_SCALAR);

  }

  // calculate the force on my particles
  int n; 
  std::string tmp;
  n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdz_brick \n");
  tmp += "{";
  for (int k = nzlo_in; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdz_brick[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);

   n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdz_node \n");
  tmp += "{";
  for (int k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (int j = nylo_node_in; j <= nyhi_node_in; j++)
      for (int i = nxlo_node_in; i <= nxhi_node_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdz_node[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);
  n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdy_brick \n");
  tmp += "{";
  for (int k = nzlo_in; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdy_brick[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);

   n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdy_node \n");
  tmp += "{";
  for (int k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (int j = nylo_node_in; j <= nyhi_node_in; j++)
      for (int i = nxlo_node_in; i <= nxhi_node_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdy_node[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);
  n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdx_brick \n");
  tmp += "{";
  for (int k = nzlo_in; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdx_brick[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);

   n = 0;
  tmp = fmt::format("[INFO] poisson_ik vdx_node \n");
  tmp += "{";
  for (int k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (int j = nylo_node_in; j <= nyhi_node_in; j++)
      for (int i = nxlo_node_in; i <= nxhi_node_in; i++) {
        tmp += fmt::format(" {:<13.9f},", vdx_node[k][j][i]);
        n += 1;
      }
  tmp += "}\n";
  utils::logmesg(lmp, tmp);

  fieldforce_ik();

  utils::logmesg_arry(lmp,fmt::format("[info] fieldforce fele \n"), fele, atom->nlocal * 3, 1);
  utils::logmesg_arry(lmp,fmt::format("[info] fieldforce fele_node \n"), fele_node, nlocal_node * 3, 1);


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


void PPPMDPLR::reverse_node(){

  if(comm->numa_id == NUMA_NUM - 1) {
    MPI_Request *send_requests = new MPI_Request[nswap];
    MPI_Request *recv_requests = new MPI_Request[nswap];
    
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto recv_buf = swap[iswap].recv_buf;
      auto nunpack = swap[iswap].nunpack;
      auto recvproc = swap[iswap].recvproc;
      MPI_Irecv(recv_buf, nunpack, MPI_DOUBLE, recvproc, 0, MPI_COMM_WORLD, &recv_requests[iswap]);
    }
  
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto send_buf = swap[iswap].send_buf;
      auto list = swap[iswap].packlist;
      auto npack = swap[iswap].npack;
      auto sendproc = swap[iswap].sendproc;
  
  
      FFT_SCALAR *src = &density_brick_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      for (int i = 0; i < npack; i++)
        send_buf[i] = src[list[i]];
      MPI_Isend(send_buf, npack, MPI_DOUBLE, sendproc, 0, MPI_COMM_WORLD, &send_requests[iswap]);
    }
  
    MPI_Waitall(26, send_requests, MPI_STATUS_IGNORE);
    MPI_Waitall(26, recv_requests, MPI_STATUS_IGNORE);
  
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto recv_buf = swap[iswap].recv_buf;
      auto nunpack = swap[iswap].nunpack;
      auto list = swap[iswap].unpacklist;
      FFT_SCALAR *src = &density_brick_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      for (int i = 0; i < nunpack; i++)
        src[list[i]] += recv_buf[i];
    }
    utils::logmesg(lmp, "[INFO] PPPM finish reverse_node \n");
  }
}

void PPPMDPLR::forward_node(){

  if(comm->numa_id == NUMA_NUM - 1) {
    MPI_Request *send_requests = new MPI_Request[nswap];
    MPI_Request *recv_requests = new MPI_Request[nswap];
    
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto recv_buf = swap[iswap].recv_buf;
      auto nunpack = swap[iswap].npack;
      auto recvproc = swap[iswap].sendproc;
      MPI_Irecv(recv_buf, 3 * nunpack, MPI_DOUBLE, recvproc, 0, MPI_COMM_WORLD, &recv_requests[iswap]);
    }
  
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto send_buf = swap[iswap].send_buf;
      auto list = swap[iswap].unpacklist;
      auto npack = swap[iswap].nunpack;
      auto sendproc = swap[iswap].recvproc;
  
      int n = 0;
      FFT_SCALAR *xsrc = &vdx_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      FFT_SCALAR *ysrc = &vdy_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      FFT_SCALAR *zsrc = &vdz_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      for (int i = 0; i < npack; i++){
        send_buf[n++] = xsrc[list[i]];
        send_buf[n++] = ysrc[list[i]];
        send_buf[n++] = zsrc[list[i]];
      }
      MPI_Isend(send_buf, 3 * npack, MPI_DOUBLE, sendproc, 0, MPI_COMM_WORLD, &send_requests[iswap]);
    }
  
    MPI_Waitall(26, send_requests, MPI_STATUS_IGNORE);
    MPI_Waitall(26, recv_requests, MPI_STATUS_IGNORE);
  
    for(int iswap = 0; iswap < nswap; iswap++) {
      auto recv_buf = swap[iswap].recv_buf;
      auto nunpack = swap[iswap].npack;
      auto list = swap[iswap].packlist;
      FFT_SCALAR *xsrc = &vdx_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      FFT_SCALAR *ysrc = &vdy_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      FFT_SCALAR *zsrc = &vdz_node[nzlo_node_out][nylo_node_out][nxlo_node_out];
      int n = 0;
      for (int i = 0; i < nunpack; i++) {
        xsrc[list[i]] = recv_buf[n++];
        ysrc[list[i]] = recv_buf[n++];
        zsrc[list[i]] = recv_buf[n++];
      }
    }
    utils::logmesg(lmp, "[INFO] PPPM finish forward_node \n");
  }
}



#ifdef SELF_HEFFTE
/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void PPPMDPLR::poisson_ik()
{
  if(comm->fft_type_flag == 0) PPPM::poisson_ik();
  else if(comm->fft_type_flag == 1) poisson_ik_heffte();

  if(comm->numa_id == NUMA_NUM - 1) 
    poisson_ik_utofubg();
}

void PPPMDPLR::poisson_ik_utofubg() {

  int i,j,k,n;
  double eng;
  int xlo, xhi, ylo, yhi, zlo, zhi;

  xlo = nxlo_node_in; xhi = nxhi_node_in; ylo = nylo_node_in; yhi = nyhi_node_in; zlo = nzlo_node_in; zhi = nzhi_node_in; 

  // transform charge density (r -> k)

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR density_fft_node \n"),density_fft_node, nfft_node_brick, 1 );

  FFT_SCALAR *work1_node_real = work1_node;
  FFT_SCALAR *work1_node_imag = work1_node+nfft_node_brick;
  
  n = 0;
  for (i = 0; i < nfft_node_brick; i++) {
    work1_node_real[i] = density_fft_node[i];
    work1_node_imag[i] = ZEROF;
  }

  fft_utofu->compute_fft3D_forward(work1_node, 0);

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 \n"),work1, nfft_node_brick*2, 1 );

  std::string tmp;
  tmp = fmt::format("[INFO] PPPMDPLR work1_node \n");
  for(int i = 0; i < nfft_node_brick ;i += 1) {
     tmp += fmt::format("  {}:{:<3.9f} {}:{:<3.9f}", 2*i, work1_node_real[i], 2*i+1, work1_node_imag[i]);
     if(i != 0 && (i % 50 == 0)) tmp += "\n      ";
  }
  tmp += "\n";
  utils::logmesg(lmp, tmp);
  

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  
  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft_node_brick; i++) {
        eng = s2 * greensfn_node[i] * (work1_node_real[n]*work1_node_real[n] + work1_node_imag[n]*work1_node_imag[n]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg_node[i][j];
        if (eflag_global) energy += eng;
        n += 1;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft_node_brick; i++) {
        energy +=
          s2 * greensfn_node[i] * (work1_node_real[n]*work1_node_real[n] + work1_node_imag[n]*work1_node_imag[n]);
        n += 1;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft_node_brick; i++) {
    work1_node_real[i] *= scaleinv * greensfn_node[i];
    work1_node_imag[i] *= scaleinv * greensfn_node[i];
  }

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 greensfn evflag_atom {} \n", evflag_atom),work1, nfft_brick*2, 1 );
  tmp = fmt::format("PPPMDPLR work1 greensfn evflag_atom {} \n", evflag_atom);
  for(int i = 0; i < nfft_node_brick ;i += 1) {
     tmp += fmt::format("  {}:{:<3.9f} {}:{:<3.9f}", 2*i, work1_node_real[i], 2*i+1, work1_node_imag[i]);
     if(i != 0 && (i % 50 == 0)) tmp += "\n      ";
  }
  tmp += "\n";
  utils::logmesg(lmp, tmp);


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

  FFT_SCALAR *work2_node_real = work2_node;
  FFT_SCALAR *work2_node_imag = work2_node+nfft_node_brick;

  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        work2_node_real[n] = -fkx_node[i]*work1_node_imag[n];
        work2_node_imag[n] = fkx_node[i] *work1_node_real[n];
        n += 1;
      }
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 0 back input \n"),work2_node_real, nfft_node_brick*2, 1 );


  fft_utofu->compute_fft3D_forward(work2_node, 1);

  // n = 0;
  // for (int i = 0; i < nfft_brick; i++) {
  //   work2[n++] = heffte_outdata[i].real();
  //   work2[n++] = heffte_outdata[i].imag();
  // }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 0 back output \n"),work2_node, nfft_node_brick, 1 );

  
  n = 0;
  for (k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (j = nylo_node_in; j <= nyhi_node_in; j++)
      for (i = nxlo_node_in; i <= nxhi_node_in; i++) {
        vdx_node[k][j][i] = work2_node_real[n];
        n += 1;
      }

  // y direction gradient

  
  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        work2_node_real[n] = -fky_node[j]*work1_node_imag[n];
        work2_node_imag[n] = fky_node[j] *work1_node_real[n];
        n += 1;
      }
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 1 back input \n"),work2_node_real, nfft_node_brick*2, 1 );


  fft_utofu->compute_fft3D_forward(work2_node, 1);

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 1 back output \n"),work2_node, nfft_node_brick*2, 1 );

  n = 0;
  for (k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (j = nylo_node_in; j <= nyhi_node_in; j++)
      for (i = nxlo_node_in; i <= nxhi_node_in; i++) {
        vdy_node[k][j][i] = work2_node_real[n];
        n += 1;
      }



  // z direction gradient

  n = 0;
  for (k = zlo; k <= zhi; k++)
    for (j = ylo; j <= yhi; j++)
      for (i = xlo; i <= xhi; i++) {
        work2_node_real[n] = -fkz_node[k]*work1_node_imag[n];
        work2_node_imag[n] = fkz_node[k] *work1_node_real[n];
        n += 1;
      }
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 2 back input \n"),work2_node_real, nfft_node_brick*2, 1 );


  fft_utofu->compute_fft3D_forward(work2_node, 1);

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 2 back output \n"),work2_node, nfft_node_brick*2, 1 );

  n = 0;
  for (k = nzlo_node_in; k <= nzhi_node_in; k++)
    for (j = nylo_node_in; j <= nyhi_node_in; j++)
      for (i = nxlo_node_in; i <= nxhi_node_in; i++) {
        vdz_node[k][j][i] = work2_node_real[n];
        n += 1;
      }
}

void PPPMDPLR::poisson_ik_heffte()
{
  int i,j,k,n;
  double eng;
  int xlo, xhi, ylo, yhi, zlo, zhi;

  xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 

  // transform charge density (r -> k)

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR density_fft \n"),density_fft, nfft_brick, 1 );

  n = 0;

  for (int i = 0; i < nfft_brick; i++) {
    heffte_indata[i].real(density_fft[i]);
    heffte_indata[i].imag(ZEROF);
  }

  run_forward();
  n = 0;
  for (int i = 0; i < nfft_brick; i++) {
    work1[n++] = heffte_outdata[i].real();
    work1[n++] = heffte_outdata[i].imag();
  }

  
  // std::vector<FFT_SCALAR> in_data(nfft_brick);
  // std::vector<FFT_SCALAR> recv_data(nfft_brick * comm->nprocs);
  // std::vector<FFT_SCALAR> all_data(nfft_brick * comm->nprocs);

  // for (int i = 0; i < nfft_brick; i++) {
  //   in_data[i] = heffte_outdata[i].real();
  // }

  // int lcl_size[3], fft_size[3];
  // lcl_size[0] = nxhi_in - nxlo_in + 1;
  // lcl_size[1] = nyhi_in - nylo_in + 1;
  // lcl_size[2] = nzhi_in - nzlo_in + 1;
  // fft_size[0] = nx_pppm;
  // fft_size[1] = ny_pppm;
  // fft_size[2] = nz_pppm; 

  // // MPI_Gather(in_data.data(), nfft_brick, MPI_DOUBLE, recv_data.data(), nfft_brick, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // n = -1;
  // for (int pzIdx = 0; pzIdx < comm->procgrid[2]; ++pzIdx) {
  //   for (int pyIdx = 0; pyIdx < comm->procgrid[1]; ++pyIdx) {
  //     for (int pxIdx = 0; pxIdx < comm->procgrid[0]; ++pxIdx) {
  //       n++;
  //       // int procRank = pzIdx * (comm->procgrid[1] * comm->procgrid[0]) + pyIdx * comm->procgrid[0] + pxIdx;
  //       int procid = comm->grid2proc[pxIdx][pyIdx][pzIdx];
  //       if(procid == 0) {
  //         memcpy(recv_data.data(), in_data.data(), nfft_brick*sizeof(double));
  //         continue;
  //       }

  //       if(comm->me == procid) {
  //         MPI_Send(in_data.data(), nfft_brick, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  //       }
  //       if(comm->me == 0) {
  //         MPI_Recv(recv_data.data()+n*nfft_brick, nfft_brick, MPI_DOUBLE, procid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //       }
  //     }
  //   }
  // }

  // for (int pzIdx = 0; pzIdx < comm->procgrid[2]; ++pzIdx) {
  //   for (int pyIdx = 0; pyIdx < comm->procgrid[1]; ++pyIdx) {
  //     for (int pxIdx = 0; pxIdx < comm->procgrid[0]; ++pxIdx) {
  //           int procRank = pzIdx * (comm->procgrid[1] * comm->procgrid[0]) + pyIdx * comm->procgrid[0] + pxIdx;
  //           int baseX = pxIdx * lcl_size[0];
  //           int baseY = pyIdx * lcl_size[1];
  //           int baseZ = pzIdx * lcl_size[2];
  //           int offset = nfft_brick * procRank;

  //           // 重新映射到全局网格
  //           for (int k = 0; k < lcl_size[2]; ++k) {
  //             for (int j = 0; j < lcl_size[1]; ++j) {
  //               for (int i = 0; i < lcl_size[0]; ++i) {
  //                     int ptr = (baseZ + k) * fft_size[1] * fft_size[0] + (baseY + j) * fft_size[0] + (baseX + i);
  //                     all_data[ptr]  = recv_data[offset++];
  //                   }
  //               }
  //           }
  //       }
  //   }
  // }

  // std::string tmp;
  // tmp += fmt::format("[INFO] poisson_ik_heffte all_data \n");
  // tmp += "{";
  // for(int i = 0; i < nx_pppm * ny_pppm * nz_pppm ;i += 1) {
  //    tmp += fmt::format(" {:<13.9f},", all_data[i]);
  // }
  // tmp += "}\n";
  // utils::logmesg(lmp, tmp);

  // if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR all_data \n"),all_data.data(), nfft_brick*2, 1 );
  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 \n"),work1, nfft_brick*2, 1 );
  

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft_brick; i++) {
        eng = s2 * greensfn_brick[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg_brick[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft_brick; i++) {
        energy +=
          s2 * greensfn_brick[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft_brick; i++) {
    work1[n++] *= scaleinv * greensfn_brick[i];
    work1[n++] *= scaleinv * greensfn_brick[i];
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR work1 greensfn evflag_atom {} \n", evflag_atom),work1, nfft_brick*2, 1 );


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
        heffte_indata[n].real(-fkx_brick[i]*work1[2*n+1]);
        heffte_indata[n].imag(fkx_brick[i] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < nfft_brick; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 0 back output \n"),work2, nfft_brick*2, 2 );

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
        heffte_indata[n].real(-fky_brick[j]*work1[2*n+1]);
        heffte_indata[n].imag(fky_brick[j] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < nfft_brick; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 1 back \n"),work2, nfft_brick*2, 2 );

  

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
        heffte_indata[n].real(-fkz_brick[k]*work1[2*n+1]);
        heffte_indata[n].imag(fkz_brick[k] *work1[2*n]);
        n += 1;
      }

  run_backward();
  n = 0;
  for (int i = 0; i < nfft_brick; i++) {
    work2[n++] = heffte_outdata[i].real();
    work2[n++] = heffte_outdata[i].imag();
  }

  if(DEBUG_MSG) utils::logmesg_arry(lmp, fmt::format("PPPMDPLR 2 back \n"),work2, nfft_brick*2, 2 );

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}


/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
------------------------------------------------------------------------- */

void PPPMDPLR::particle_map()
{
  if(comm->fft_type_flag < 2) {
    PPPM::particle_map();
  } else {
    particle_map_node();
  }

  if(comm->numa_id == NUMA_NUM - 1)
    particle_map_node();
}

void PPPMDPLR::particle_map_node() {
  int nx,ny,nz;

  int flag = 0;

  if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
    error->one(FLERR,"particle_map_node Non-numeric box dimensions - simulation unstable");

  for (int i = 0; i < nlocal_node; i++) {

    // order = even:
    //   (nx,ny,nz) = global index of grid pt to "lower left" of charge
    // order = odd:
    //   (nx,ny,nz) = global index of grid pt closest to charge due to shift
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    nx = static_cast<int> ((x_node[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
    ny = static_cast<int> ((x_node[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
    nz = static_cast<int> ((x_node[i][2]-boxlo[2])*delzinv+shift) - OFFSET;

    part2grid_node[i][0] = nx;
    part2grid_node[i][1] = ny;
    part2grid_node[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

    if (nx+nlower < nxlo_node_out || nx+nupper > nxhi_node_out ||
        ny+nlower < nylo_node_out || ny+nupper > nyhi_node_out ||
        nz+nlower < nzlo_node_out || nz+nupper > nzhi_node_out)
      flag = 1;
  }

  if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM");
}



/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid
------------------------------------------------------------------------- */

void PPPMDPLR::make_rho() {
  if(comm->fft_type_flag < 2) {
    PPPM::make_rho();
  } else {
    make_rho_node();
  }
  if(comm->numa_id == NUMA_NUM - 1)
    make_rho_node();
}


void PPPMDPLR::make_rho_node()
{
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(density_brick_node[nzlo_node_out][nylo_node_out][nxlo_node_out]),0,
         ngrid_node*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global indices of moving stencil pt

  // double *q_node = atom->q_node;
  // double **x_node = atom->x_node;
  // int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal_node; i++) {

    nx = part2grid_node[i][0];
    ny = part2grid_node[i][1];
    nz = part2grid_node[i][2];
    dx = nx+shiftone - (x_node[i][0]-boxlo[0])*delxinv;
    dy = ny+shiftone - (x_node[i][1]-boxlo[1])*delyinv;
    dz = nz+shiftone - (x_node[i][2]-boxlo[2])*delzinv;

    compute_rho1d(dx,dy,dz);

    z0 = delvolinv * q_node[i];
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      y0 = z0*rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        x0 = y0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          density_brick_node[mz][my][mx] += x0*rho1d[0][l];
        }
      }
    }
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

  if(comm->fft_type_flag < 2) {
    n = 0;
    for (iz = nzlo_in; iz <= nzhi_in; iz++)
      for (iy = nylo_in; iy <= nyhi_in; iy++)
        for (ix = nxlo_in; ix <= nxhi_in; ix++)
          density_fft[n++] = density_brick[iz][iy][ix];
  } else {
    n = 0;
    for (iz = nzlo_node_in; iz <= nzhi_node_in; iz++)
      for (iy = nylo_node_in; iy <= nyhi_node_in; iy++)
        for (ix = nxlo_node_in; ix <= nxhi_node_in; ix++)
        density_fft_node[n++] = density_brick_node[iz][iy][ix];
      }


  utils::logmesg_arry(lmp, fmt::format("[INFO] density_fft n {}", n), density_fft, n, 1);
  if(comm->numa_id == NUMA_NUM - 1) {
    n = 0;
    for (iz = nzlo_node_in; iz <= nzhi_node_in; iz++)
      for (iy = nylo_node_in; iy <= nyhi_node_in; iy++)
        for (ix = nxlo_node_in; ix <= nxhi_node_in; ix++)
        density_fft_node[n++] = density_brick_node[iz][iy][ix];
    
    utils::logmesg_arry(lmp, fmt::format("[INFO] density_fft_node n {}", n), density_fft_node, n, 1);
  }

  if(comm->fft_type_flag == 0) {
    remap->perform(density_fft,density_fft,work1);
  } 
}

void PPPMDPLR::compute_gf_ik_brick() {

  memory->create(greensfn_brick,nfft_both,"pppm:greensfn_brick");

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

  xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 
  
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
          greensfn_brick[n++] = numerator*sum1/denominator;
        } else greensfn_brick[n++] = 0.0;
      }
    }
  }
}

void PPPMDPLR::compute_gf_ik_node() {

  memory->create(greensfn_node,nfft_node_brick,"pppm:nfft_node_brick");

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

  xlo = nxlo_node_in; xhi = nxhi_node_in; ylo = nylo_node_in; yhi = nyhi_node_in; zlo = nzlo_node_in; zhi = nzhi_node_in; 
  
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
          greensfn_node[n++] = numerator*sum1/denominator;
        } else greensfn_node[n++] = 0.0;
      }
    }
  }
}


void PPPMDPLR::setup()
{
  PPPM::setup();
  setup_brick();
  setup_node();
}
void PPPMDPLR::setup_brick()
{
  memory->create1d_offset(fkx_brick,nxlo_in,nxhi_in,"pppm:fkx_brick");
  memory->create1d_offset(fky_brick,nylo_in,nyhi_in,"pppm:fky_brick");
  memory->create1d_offset(fkz_brick,nzlo_in,nzhi_in,"pppm:fkz_brick");

  memory->create(vg_brick,nfft_both,6,"pppm:vg");

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

  xlo = nxlo_in; xhi = nxhi_in; ylo = nylo_in; yhi = nyhi_in; zlo = nzlo_in; zhi = nzhi_in; 

  for (i = xlo; i <= xhi; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx_brick[i] = unitkx*per;
  }

  for (i = ylo; i <= yhi; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky_brick[i] = unitky*per;
  }

  for (i = zlo; i <= zhi; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz_brick[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = zlo; k <= zhi; k++) {
    for (j = ylo; j <= yhi; j++) {
      for (i = xlo; i <= xhi; i++) {
        sqk = fkx_brick[i]*fkx_brick[i] + fky_brick[j]*fky_brick[j] + fkz_brick[k]*fkz_brick[k];
        if (sqk == 0.0) {
          vg_brick[n][0] = 0.0;
          vg_brick[n][1] = 0.0;
          vg_brick[n][2] = 0.0;
          vg_brick[n][3] = 0.0;
          vg_brick[n][4] = 0.0;
          vg_brick[n][5] = 0.0;
        } else {
          vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
          vg_brick[n][0] = 1.0 + vterm*fkx_brick[i]*fkx_brick[i];
          vg_brick[n][1] = 1.0 + vterm*fky_brick[j]*fky_brick[j];
          vg_brick[n][2] = 1.0 + vterm*fkz_brick[k]*fkz_brick[k];
          vg_brick[n][3] = vterm*fkx_brick[i]*fky_brick[j];
          vg_brick[n][4] = vterm*fkx_brick[i]*fkz_brick[k];
          vg_brick[n][5] = vterm*fky_brick[j]*fkz_brick[k];
        }
        n++;
      }
    }
  }

  compute_gf_ik_brick();
}

void PPPMDPLR::setup_node()
{
  memory->create1d_offset(fkx_node,nxlo_node_in,nxhi_node_in,"pppm:fkx_node");
  memory->create1d_offset(fky_node,nylo_node_in,nyhi_node_in,"pppm:fky_node");
  memory->create1d_offset(fkz_node,nzlo_node_in,nzhi_node_in,"pppm:fkz_node");

  memory->create(vg_node,nfft_node_brick, 6,"pppm:vg_node");

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

  xlo = nxlo_node_in; xhi = nxhi_node_in; ylo = nylo_node_in; yhi = nyhi_node_in; zlo = nzlo_node_in; zhi = nzhi_node_in; 

  for (i = xlo; i <= xhi; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx_node[i] = unitkx*per;
  }

  for (i = ylo; i <= yhi; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky_node[i] = unitky*per;
  }

  for (i = zlo; i <= zhi; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz_node[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = zlo; k <= zhi; k++) {
    for (j = ylo; j <= yhi; j++) {
      for (i = xlo; i <= xhi; i++) {
        sqk = fkx_node[i]*fkx_node[i] + fky_node[j]*fky_node[j] + fkz_node[k]*fkz_node[k];
        if (sqk == 0.0) {
          vg_node[n][0] = 0.0;
          vg_node[n][1] = 0.0;
          vg_node[n][2] = 0.0;
          vg_node[n][3] = 0.0;
          vg_node[n][4] = 0.0;
          vg_node[n][5] = 0.0;
        } else {
          vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
          vg_node[n][0] = 1.0 + vterm*fkx_node[i]*fkx_node[i];
          vg_node[n][1] = 1.0 + vterm*fky_node[j]*fky_node[j];
          vg_node[n][2] = 1.0 + vterm*fkz_node[k]*fkz_node[k];
          vg_node[n][3] = vterm*fkx_node[i]*fky_node[j];
          vg_node[n][4] = vterm*fkx_node[i]*fkz_node[k];
          vg_node[n][5] = vterm*fky_node[j]*fkz_node[k];
        }
        n++;
      }
    }
  }

  compute_gf_ik_node();
}

#endif
/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void PPPMDPLR::fieldforce_ik(){
  if(comm->fft_type_flag < 2) fieldforce_ik_brick();
  else fieldforce_ik_node();

  if(comm->numa_id == NUMA_NUM - 1) 
    fieldforce_ik_node();

}
void PPPMDPLR::fieldforce_ik_brick()
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

  int nlocal = atom->nlocal;

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
void PPPMDPLR::fieldforce_ik_node()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  memset(fele_node, 0, nlocal_node*3*sizeof(FPTYPE));

  for (i = 0; i < nlocal_node; i++) {
    nx = part2grid_node[i][0];
    ny = part2grid_node[i][1];
    nz = part2grid_node[i][2];
    dx = nx+shiftone - (x_node[i][0]-boxlo[0])*delxinv;
    dy = ny+shiftone - (x_node[i][1]-boxlo[1])*delyinv;
    dz = nz+shiftone - (x_node[i][2]-boxlo[2])*delzinv;

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
          ekx -= x0*vdx_node[mz][my][mx];
          eky -= x0*vdy_node[mz][my][mx];
          ekz -= x0*vdz_node[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = qqrd2e * scale * q_node[i];
    fele_node[i*3+0] += qfactor*ekx;
    fele_node[i*3+1] += qfactor*eky;
    if (slabflag != 2) fele_node[i*3+2] += qfactor*ekz;
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

