#ifdef FIX_CLASS

FixStyle(dplr,FixDPLR)

#else

#ifndef LMP_FIX_DPLR_H
#define LMP_FIX_DPLR_H

#include <stdio.h>
#include "fix.h"
#include "pair_deepmd.h"
#include "deepmd_util.h"
#include "pppm_dplr.h"
namespace LAMMPS_NS {
  class FixDPLR : public Fix {
public:
    FixDPLR(class LAMMPS *, int, char **);
    virtual ~FixDPLR() {};
    int setmask() override;
    void init() override;
    void setup(int) override;
    void post_integrate() override;
    void pre_force(int) override;
    void post_force(int) override;
    void setup_pre_force(int) override;
    int pack_reverse_comm(int, int, double *) override;
    void unpack_reverse_comm(int, int *, double *) override;
    double compute_scalar(void) override;
    double compute_vector(int) override;
private:
    PairDeepMD * pair_deepmd;
    PPPMDPLR * pppm_dplr;
    // deepmd::DeepTensor dpt;
    // deepmd::DipoleChargeModifier dtm;
    DeepPot **deep_pots_dipole;

    double* dvirial;
    double* dipole_recd;
    double** thread_dvirial;
    double** thread_dipole_recd;
    int ntypes;
    std::vector<int > dipole_sel_type;
    std::vector<int > dpl_type;
    std::vector<int > bond_type;
    std::map<int,int > type_asso;
    std::map<int,int > bk_type_asso;
    // std::vector<FPTYPE> dipole_recd;
    std::vector<double> dfcorr_buff;
    double efield[3];
    double efield_fsum[4], efield_fsum_all[4];
    int efield_force_flag;
    std::vector<std::pair<int,int>> bd_pairs;
    int nbd_pairs;
    int* bd_idx;
    void init_valid_pairs();
  };
}

#endif // LMP_FIX_DPLR_H
#endif // FIX_CLASS
