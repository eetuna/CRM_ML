
/* Include libraries. */
#include "mex.h"

#include <Eigen/Dense>

#include <cmath>
#include "CRMDYN.hpp"
#include "CRM.hpp"
#include <fstream>
using namespace std::chrono;

#define EPS 1e-5
#define NUM_DYN_STATE (3+3+9+3+9 +NUM_STATES ) // (v, w, u, n, m, p, R) 27 + 15
//#define NUM_OUTPUT_STATE (3+3) // p, R_z

/*   Copyright 2005-2015 The MathWorks, Inc. */
/*   Written by Peter Lindskog. */


/* Specify the number of outputs here. */
#define NY (NUM_ACT_SET * 3)

#define Ncoilstate (NUM_ACT_SET * (6+6+3+9))

using namespace CRMCatheterModel;

template <typename T>
void printMatrix(T *p, int D1, int D2, const char *text) {
    std::cout << text << " --" << std::endl;
    for (int i=0; i<D1; i++) {
        for (int j=0; j<D2; j++) {
            std::cout << *(p+i*D2+j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----" << std::endl;
}

/* State equations. */
void compute_dx(double *dx, double t, double *x, double *u, double **p,
                const mxArray *auxvar)
{

    /** Retrieve x. **/
    double v_L_pre[NUM_ACT_SET][3], w_L_pre[NUM_ACT_SET][3], nL_initialguess[NUM_ACT_SET][3], mL_initialguess[NUM_ACT_SET][3], pL_pre[NUM_ACT_SET][3], RL_pre[NUM_ACT_SET][9], xf_pre[NUM_STATES];
    // Define initial guesses to be used when solving boundary value problem
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = x[i + j*3];
            w_L_pre[j][i] = x[NUM_ACT_SET*3 + i + j*3];
            mL_initialguess[j][i] = x[NUM_ACT_SET*6 + i + j*3];
            nL_initialguess[j][i] = x[NUM_ACT_SET*9 + i + j*3];
            pL_pre[j][i] = x[NUM_ACT_SET* 12 + i + j*3];
        }
        for (int i = 0; i < 9; ++i) {
            RL_pre[j][i] = x[NUM_ACT_SET *15 +i + j*9];
        }
    }

    for (int i = 0; i < NUM_STATES; ++i) {
        xf_pre[i] = x[i+Ncoilstate];
    }


    // *** Load parameters from file
    // This step would typically needs to be executed only once
    //   Physical Description of the Catheter
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("data/catheter_params/CatheterParameterSet_1_new.txt");
    //   Catheter Configuration in spatial coordinates
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("data/catheter_params/CatheterSpatialConfiguration_1.txt");

    size_t ix, jx;

    auto & OuterRadius = CathParams.OuterRadius;
    auto & InnerRadius = CathParams.InnerRadius;
    auto & YoungsModulus = CathParams.YoungsModulus;
    auto & ShearModulus = CathParams.ShearModulus;
    auto & CoilAlignmentAngles = CathParams.CoilAlignmentAngles;
    auto & CoilTurnAreaMat = CathParams.CoilTurnAreaMat;
    auto & ActMass = CathParams.ActMass;

    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        OuterRadius[ix] = p[2][0];  //parameter only has one value here, assumes isomorph
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        InnerRadius[ix] = p[2][1];
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        YoungsModulus[ix] = p[3][0];
    }
    for (ix = 0; ix < CathParams.no_flex_seg; ix++){
        ShearModulus[ix] = p[3][1];
    }
    for (ix = 0; ix < CathParams.no_act_set; ix++) {
        for (jx = 0; jx < 2; jx++) {
            CoilAlignmentAngles[ix][jx] = p[4][jx+2*ix];
        }
    }
    for (ix = 0; ix < CathParams.no_act_set; ix++) {
        CoilTurnAreaMat[ix][0] = p[5][0 + ix*3];
        CoilTurnAreaMat[ix][4] = p[5][1 + ix*3];
        CoilTurnAreaMat[ix][8] = p[5][2 + ix*3];

    }
    for (ix = 0; ix < CathParams.no_act_set; ix++){
        ActMass[ix] = p[6][ix];
    }

    // *** Other External variables
    // specify if catheter is in free space or if the catheter tip is constrained to a contact point
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    // External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0)  - unit: ??
    //   (this will be used when ContactMode == ContactModeType::FREE_TIP)
    double TipForce[3] = { 0.0, 0.0, 0.0 };
    // The spatial coordinates of the point where the catheter tip is constrained to be
    //   (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double TipConstraintPoint[3] = { 0.0, 0.0, 0.0 };


    /** Retrieve u. **/
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int i = 0; i < NUM_ACT_SET; i++)	for (int j = 0; j < 3; j++)	ActuationCurrents[i][j] = u[i * 3 + j];
//    std::cout << "ActuationCurrents: " << ActuationCurrents[0][0] << " " << ActuationCurrents[0][1] << " " << ActuationCurrents[0][2] <<  std::endl;

//    double InsertedLength = u[NUM_ACT_SET*3];
    /** Retrieve model.parameters. **/
    double damping[NUM_ACT_SET][6];

    for (int i = 0; i < NUM_ACT_SET; ++i) {
        damping[i][0] = damping[i][1] = p[0][0 + i*4];
        damping[i][2] = p[0][1 + i*4];
        damping[i][3] = damping[i][4] = p[0][2 + i*4];
        damping[i][5] = p[0][3 + i*4];
    }

//    std::cout <<"damping_ w: " << damping_[3] << " " << damping_[4] << " " << damping_[5] << std::endl;

    double DELTA_T = p[1][0];

    double InsertedLength =0;
    for (int i = 0; i < 3; ++i) {
        CathParams.SegLengths[i] = p[7][i];
        InsertedLength += p[7][i];
    }


    // *** Numerical Computation Params
    // Stepsize used in numerical integration along the length of the catheter during IVP - unit: mm
    double IntegrationStepSize = 0.2;

    // *** Storage for storing localization marker positions and actuation coil orientations
    double (*ReportedMarkerPos)[3] = new double [CathParams.no_locmarkers][3];
//    double (*ReportedCoilOrient)[9] = new double[CathParams.no_act_set][9];


    // Define initial guesses to be used when solving boundary value problem
    // initial guess for the delta_curvature at the catheter base ( u0 = deltau0 + ustar0 )
    // initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double ftip_initialguess[3] = { 0.0, 0.0, 0.0 };

    // numerical nonlinear equation solver diagnostic outputs
    int localmin;


    /**
     * These are hard coded, need to revise these later
     */
    // we are adding the tubing mass of the coil section to the total mass of actuator unit(kg * mm^2)
    double ActInertia[NUM_ACT_SET][9];
    for (int i = 0; i < NUM_ACT_SET; ++i)
    {
        double I_zz = 0.5 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] + CathParams.InnerRadius[0] * CathParams.InnerRadius[0]);
        double I_xx = 0.25 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] + CathParams.InnerRadius[0] * CathParams.InnerRadius[0]) + 1.0 / 12 * (CathParams.ActMass[i]) * CathParams.SegLengths[2*i+1] * CathParams.SegLengths[2*i+1];
        ActInertia[i][0] = I_xx; ActInertia[i][1] = 0.0; ActInertia[i][2] = 0.0;
        ActInertia[i][3] = 0.0; ActInertia[i][4] = I_xx; ActInertia[i][5] = 0.0;
        ActInertia[i][6] = 0.0; ActInertia[i][7] = 0.0; ActInertia[i][8] = I_zz;
    }

    CRMShootingMethodParams BVPParams =  CRMDYNConstructShootingMethodParamSet(CathParams, CathConfig, InsertedLength, ActuationCurrents, ContactMode,
                                                                               TipConstraintPoint, TipForce, IntegrationStepSize, ActInertia,
                                                                               v_L_pre, w_L_pre, pL_pre,  RL_pre, damping, DELTA_T);


    double out_u0[3], out_nL[NUM_ACT_SET][3], out_mL[NUM_ACT_SET][3], ftip_calc[3];
    double out_tau[NUM_ACT_SET][3];

    DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess, ftip_initialguess,
                out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

//    std::cout << "out_u0: " << out_u0[0] << " " << out_u0[1] << " " << out_u0[2] <<  std::endl;
//    for (int i = 0; i < NUM_ACT_SET; ++i) {
//        std::cout << "out_mL: " << out_mL[i][0] << " " << out_mL[i][1] << " " << out_mL[i][2] <<  std::endl;
//        std::cout << "out_nL: " << out_nL[i][0] << " " << out_nL[i][1] << " " << out_nL[i][2] <<  std::endl;
//    }

    double xf[NUM_STATES], x_coil[NUM_ACT_SET][NUM_COIL_STATES];

    DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                 true, xf, x_coil,ReportedMarkerPos);

    double v_L_update[NUM_ACT_SET][3], w_L_update[NUM_ACT_SET][3], pL_update[NUM_ACT_SET][3], RL_update[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_update[j][i] = x_coil[j][i];
            w_L_update[j][i] = x_coil[j][i+3];
            pL_update[j][i] = x_coil[j][i+6];
        }

        for (int i = 0; i < 9; ++i) {
            RL_update[j][i] = x_coil[j][i+9];
        }
    }

    /** Output report **/
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            dx[i + j*3] = x_coil[j][i]; //vL
            dx[NUM_ACT_SET*3 + i + j*3]= x_coil[j][i+3]; //wL
            dx[NUM_ACT_SET*6 + i + j*3] = out_mL[j][i]; //mL
            dx[NUM_ACT_SET*9 + i + j*3] = out_nL[j][i]; //nL
            dx[NUM_ACT_SET* 12 + i + j*3] = x_coil[j][i+6];
        }
        for (int i = 0; i < 9; ++i) {
            dx[NUM_ACT_SET *15 +i + j*9] = x_coil[j][i+9];
        }
    }

    for (int i = 0; i < NUM_STATES; ++i) {
        dx[i+Ncoilstate] = xf[i]; //tip position
    }
}


/** Output equation. **/
void compute_y(double *y, double t, double *x, double *u, double **p,
               const mxArray *auxvar)
{

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            y[i +j*3] = x[NUM_ACT_SET * 12 +i + j*3]; //p
        }
    }

//    for (int i = 0; i < 3; ++i) {
//        y[i+3] = x[i + 24]; //Rz
//    }

}


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Declaration of input and output arguments. */
    double *x, *u, **p, *dx, *y, *t;
    int     i, np;
    size_t  nu, nx;
    const mxArray *auxvar = NULL; /* Cell array of additional data. */

    if (nrhs < 3) {
        mexErrMsgIdAndTxt("IDNLGREY:ODE_FILE:InvalidSyntax",
                          "At least 3 inputs expected (t, u, x).");
    }

    /* Determine if auxiliary variables were passed as last input.  */
    if ((nrhs > 3) && (mxIsCell(prhs[nrhs-1]))) {
        /* Auxiliary variables were passed as input. */
        auxvar = prhs[nrhs-1];
        np = nrhs - 4; /* Number of parameters (could be 0). */
    } else {
        /* Auxiliary variables were not passed. */
        np = nrhs - 3; /* Number of parameters. */
    }

    /* Determine number of inputs and states. */
    nx = mxGetNumberOfElements(prhs[1]); /* Number of states. */
    nu = mxGetNumberOfElements(prhs[2]); /* Number of inputs. */

    /* Obtain double data pointers from mxArrays. */
    t = mxGetPr(prhs[0]);  /* Current time value (scalar). */
    x = mxGetPr(prhs[1]);  /* States at time t. */
    u = mxGetPr(prhs[2]);  /* Inputs at time t. */

    p = static_cast<double **>(mxCalloc(np, sizeof(double*)));
    for (i = 0; i < np; i++) {
        p[i] = mxGetPr(prhs[3+i]); /* Parameter arrays. */
    }

    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(nx, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NY, 1, mxREAL);
    dx      = mxGetPr(plhs[0]); /* State derivative values. */
    y       = mxGetPr(plhs[1]); /* Output values. */

    /*
      Call the state and output update functions.

      Note: You may also pass other inputs that you might need,
      such as number of states (nx) and number of parameters (np).
      You may also omit unused inputs (such as auxvar).

      For example, you may want to use orders nx and nu, but not time (t)
      or auxiliary data (auxvar). You may write these functions as:
          compute_dx(dx, nx, nu, x, u, p);
          compute_y(y, nx, nu, x, u, p);
    */

    /* Call function for state derivative update. */
    compute_dx(dx, t[0], x, u, p, auxvar);

    /* Call function for output update. */
    compute_y(y, t[0], x, u, p, auxvar);

    /* Clean up. */
    mxFree(p);
}

////test before run on Matlab
//int main(int argc, char** argv){
//
//    double v_L_pre[3] = { 0.0, 0.0, 0.0 };
//    double w_L_pre[3] = { 0.0, 0.0, 0.0 };
//    // Define initial guesses to be used when solving boundary value problem
//    double u0_initialguess[3] = { 0.0, 0.0, 0.0};
//    // initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
//    double nL_initialguess[3] = { 0.0, 0.0, 0.0 };
//    double mL_initialguess[3] = { 0.0, 0.0, 0.0 };
//
//
//    double p_0[3] = {-0.407795, -0.301721, 69.4976};
//    double R_0[9] = {0.999946, -3.98697e-05, -0.0103811,
//                     -3.98697e-05, 0.999971, -0.00768085,
//                    0.0103811, 0.00768085, 0.999917};
//
//
//    double x[NUM_DYN_STATE],  dx[NUM_DYN_STATE], u[NUM_CONTROL];
//    double **p;
//
//    p = (double **) malloc(1 * sizeof(double)); //just damping gmat for now
//    p[0] = (double *) malloc(6*sizeof(double));
//
//    for (int i = 0; i < NUM_DYN_STATE; ++i) {
//        if(i < 3) x[i] = v_L_pre[i];
//        else if (i < 6) x[i] = w_L_pre[i-3];
//        else if (i < 9) x[i] = u0_initialguess[i-6];
//        else if (i < 12) x[i] = nL_initialguess[i-9];
//        else if (i < 15) x[i] = mL_initialguess[i-12];
//        else if (i < 18) x[i] = p_0[i-15];
//        else x[i] = R_0[i-18];
//    }
//
//    for (int i = 0; i < 3; ++i) {
//        u[i] = 0.0;
//    }
//    u[NUM_CONTROL-1] = 98.5;
//
//    double t = 0;
//    for (int i = 0; i < 3; ++i) {
//        p[0][i] = 1.0; //v
//    }
//    for (int i = 0; i < 3; ++i) {
//        p[0][i+3] = 0.05; //w
//    }
//    const mxArray *auxvar = NULL;
//
//    compute_dx(dx, t, x, u, static_cast<double **>(p), auxvar);
//
//    std::cout << "pL: " << dx[15] << " " << dx[16] << " " << dx[17] <<  std::endl;
//
//    std::cout << "RL: " << std::endl;
//    std::cout <<  dx[18] << " " << dx[19] << " " << dx[20] <<  std::endl;
//    std::cout <<  dx[21] << " " << dx[22] << " " << dx[23] <<  std::endl;
//    std::cout <<  dx[24] << " " << dx[25] << " " << dx[26] <<  std::endl;
//
//
//
//
//}