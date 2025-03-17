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
    virtual void brick2fft() override;
    virtual void compute_gf_ik() override;
    void poisson_ik_normal();
    void poisson_ik_heffte();
    void poisson_ik_utofubg();

    #endif
private:
    // std::vector<double > fele;

  };

}

#endif
#endif

