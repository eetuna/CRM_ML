#pragma once
#include "CRMDYN.hpp"

namespace CRMCatheterModel {

	void fdjac1_dyn(int n, double x[], double fvec[], double fjac[], int ldfjac, int& iflag, int ml, int mu, double epsfcn, double wa1[], double wa2[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]);

	void hybrd_dyn(int n, double x[], double fvec[], double xtol, int maxfev, int ml, int mu, double epsfcn, double diag[], int mode, double factor, int nprint, int& info, int& nfev, double fjac[], int ldfjac, double r[], int lr, double qtf[], double wa1[], double wa2[], double wa3[], double wa4[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]);

	void TrustRegionDogleg_dyn(int n, double x[], double fvec[], double tol, int& info, double wa[], int lwa, DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]);
}