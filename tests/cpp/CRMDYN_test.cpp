#include <iostream>
#include <cmath>
#include <chrono> 

#include "CRM.hpp"		// This is the only header that needs to be included to use the standard CRM Kinematics APIs
using namespace CRMCatheterModel;
#include "CRMTest.h"	// The CRMTest specific stuff is here
#include "CRMDYN.hpp"

using namespace std::chrono;

#define EPS 1e-5


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

template <int D1, int D2>
bool MatrixEqual(double A[D1][D2], double B[D1][D2], double eps) {
	for (int i=0; i<D1;i++)
		for (int j=0; j<D2; j++)
			if (fabs(A[i][j]-B[i][j])<eps) {} else return false;
	return true;
}

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int RunDynamicsExample(void);


int main(int argc, char** argv) {

    RunDynamicsExample();

}

int RunDynamicsExample(void) {

    std::cout << "### RunFKExample() --- CRM Forward Kinematics Examples... " << std::endl;
    std::cout << std::endl << "Free Space Deflection Example: " << std::endl << std::endl;

    // *** Load parameters from file
    // This step would typically needs to be executed only once
    //   Physical Description of the Catheter
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../data/catheter_params/CatheterParameterSet_1_dyn.txt");
    //   Catheter Configuration in spatial coordinates
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../data/catheter_params/CatheterSpatialConfiguration_1.txt");

    // *** Other External variables
    // specify if catheter is in free space or if the catheter tip is constrained to a contact point
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    // External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0)  - unit: ??
    //   (this will be used when ContactMode == ContactModeType::FREE_TIP)
    double TipForce[3] = {0.0, 0.0, 0.0};
    // The spatial coordinates of the point where the catheter tip is constrained to be
    //   (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double TipConstraintPoint[3] = {0.0, 0.0, 0.0};

    // *** Numerical Computation Params
    // Stepsize used in numerical integration along the length of the catheter during IVP - unit: mm
    double IntegrationStepSize = 0.2;

    // *** Storage for storing localization marker positions and actuation coil orientations
    double (*ReportedMarkerPos)[3] = new double[CathParams.no_locmarkers][3];
    double (*ReportedCoilOrient)[9] = new double[CathParams.no_act_set][9];


    // Define initial guesses to be used when solving boundary value problem
    // initial guess for the delta_curvature at the catheter base ( u0 = deltau0 + ustar0 )
    double deltau0_initialguess[3] = {0.0, 0.0, 0.0};
    // initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double ftip_initialguess[3] = {0.0, 0.0, 0.0};

    // numerical nonlinear equation solver diagnostic outputs
    int localmin;
//
////    double InsertedLength = 80;
////    // Actuation currents for each of the coils for each of the coil sets - unit: A
////    double ActuationCurrents[NUM_ACT_SET][3] = {{0.0, 0.0, 0.119}};
//
    double InsertedLength  = 94.3;

    // Actuation currents for each of the coils for each of the coil sets - unit: A
    double ActuationCurrents[NUM_ACT_SET][3] = {{0.0,-0.0,0.1}};
//
////    //
////    // New Forward Kinematics Function
////    //
////    // The Forward Kinematics Function uses a different parameter structure than BVP
////    CRMForwardKinematicsData FKParams;
////    FKParams.CathConfig = &CathConfig;
////    FKParams.CathParams = &CathParams;
////    FKParams.ContactMode = ContactMode;
////    FKParams.FinalValueOnly = false;  // we want the localization coil locations, too
////    FKParams.ReportedMarkerPos = ReportedMarkerPos;
////    FKParams.ReportedCoilOrient = ReportedCoilOrient;
////    for (int i = 0; i < 3; i++) FKParams.TipConstraintPoint[i] = TipConstraintPoint[i];
////    for (int i = 0; i < 3; i++) FKParams.TipForce[i] = TipForce[i];
////    for (int i = 0; i < 3; i++) FKParams.deltau0_initialguess[i] = deltau0_initialguess[i];
////    for (int i = 0; i < 3; i++) FKParams.ftip_initialguess[i] = ftip_initialguess[i];
////    FKParams.IntegrationStepSize = IntegrationStepSize;
////    // inputs
////    VectorXd control_inputs(NUM_ACT_SET * 3 + 1); // actuator currents (distal to proximal) followed by inserted length
////    for (int i = 0; i < NUM_ACT_SET; i++)  for (int j = 0; j < 3; j++) control_inputs(i * 3 + j) = ActuationCurrents[i][j];
////    control_inputs(NUM_ACT_SET * 3) = InsertedLength;
////    // and outputs
////    int outputdim;
////    if (ContactMode == ContactModeType::FREE_TIP) outputdim = 15; else outputdim = 18;
////    VectorXd FKsolution(outputdim); // p[0..2],R[0..8],deltau0[0..2](,ftip[0..2])  R: in row major order
////    double   PotentialEnergy;
////
////    FKsolution = CRM_ForwardKinematics(control_inputs, FKParams, PotentialEnergy, localmin);
////
////    std::cout << "FKsolution " << FKsolution << std::endl;
////
//
//
//
//    double damping[NUM_ACT_SET][6] = {{120.1761626666366,120.1761626666366,
//                                              284.429938756989,
//                                              0.304776127617393,0.304776127617393,
//                                              0.00502712804532508}, {120.1761626666366,120.1761626666366,
//                                              284.429938756989,
//                                              0.304776127617393,0.304776127617393,
//                                              0.00502712804532508}};
//    double DELTA_T = 0.05;
//
//    double xf_pre[NUM_STATES] = {-0.250257,
//            21.6759,
//            76.3461 ,
//            0.999992,
//            0.00335144,
//            -0.00237778,
//            -0.00222794,
//            0.928397,
//            0.371584,
//            0.00345286,
//            -0.371576,
//            0.928396,
//            4.86377e-06,
//            9.43139e-05,
//            -3.78645e-08};
//    double pL[NUM_ACT_SET][3] = {{-0.0182097, 1.36845 ,14.866}, { -0.119139 ,9.11636, 44.9773}};
//    double RL[NUM_ACT_SET][9] = {{0.999998 ,0.00138342 ,-0.0015975,
//    -0.00118122, 0.992739, 0.120285,
//    0.00175231, -0.120283, 0.992738},{0.999983 ,0.00335188 ,-0.00480385,
//                                            -0.00132564 ,0.928318 ,0.371784,
//        0.00570568, -0.371772, 0.928307}};
//    double v_L_pre[NUM_ACT_SET][3] = {{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 }};
//    double w_L_pre[NUM_ACT_SET][3] = {{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 }};
//    // Define initial guesses to be used when solving boundary value problem
//    double nL_initialguess[NUM_ACT_SET][3] = {{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 }};
//    double mL_initialguess[NUM_ACT_SET][3] = {{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 }};
//
////    double nL_initialguess[NUM_ACT_SET][3] = {{ 2.10841e-05, -0.0131396 ,-0.127776 }, {8.64775e-05, -0.0591842 ,-0.153273}};
////    double mL_initialguess[NUM_ACT_SET][3] = {{ -0.442178, -0.000670959 ,-0.000497768 },{ -0.885441 ,-0.00146746, 0.00025844}};
////



    double damping[NUM_ACT_SET][6] = {{12.1761626666366, 12.1761626666366,
                                       284.429938756989,
                                       0.0304776127617393, 0.0304776127617393,
                                       0.00502712804532508}};
    double DELTA_T = 0.05;

    double xf_pre[NUM_STATES] = {-0.458414144062750,
                                 34.411241976876518,
                                 70.457561147732264,
                                 0.999932718178103,
                                 0.009921777042635,
                                 -0.006009780134551,
                                 -0.004651734390922,
                                 0.817579117734723,
                                 0.575797488368325,
                                 0.010626405041486,
                                 -0.575730791763330,
                                 0.817570262993625,
                                 -0.015378744286498,
                                 0.000001280646594,
                                 -0.000349413951059
    };
    double pL[NUM_ACT_SET][3] = {-0.248418562587657, 17.707660318406560, 46.752162601547091};
    double RL[NUM_ACT_SET][9] = {0.999919687839427, 0.009924211584043, -0.007882125064742,
                                 -0.003571217614502, 0.817374079004311, 0.576096225796181,
                                 0.012159945552960, -0.576021809479719, 0.817343875445250};
    double v_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    double w_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    // Define initial guesses to be used when solving boundary value problem
    double nL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    double mL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};


    /**
     * These are hard coded, need to revise these later
     */
    // we are adding the tubing mass of the coil section to the total mass of actuator unit(kg * mm^2)
    double ActInertia[NUM_ACT_SET][9];
    for (int i = 0; i < NUM_ACT_SET; ++i) {
        double I_zz = 0.5 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] +
                                                       CathParams.InnerRadius[0] * CathParams.InnerRadius[0]);
        double I_xx = 0.25 * (CathParams.ActMass[i]) * (CathParams.OuterRadius[0] * CathParams.OuterRadius[0] +
                                                        CathParams.InnerRadius[0] * CathParams.InnerRadius[0]) +
                      1.0 / 12 * (CathParams.ActMass[i]) * CathParams.SegLengths[2 * i + 1] *
                      CathParams.SegLengths[2 * i + 1];
        ActInertia[i][0] = I_xx;
        ActInertia[i][1] = 0.0;
        ActInertia[i][2] = 0.0;
        ActInertia[i][3] = 0.0;
        ActInertia[i][4] = I_xx;
        ActInertia[i][5] = 0.0;
        ActInertia[i][6] = 0.0;
        ActInertia[i][7] = 0.0;
        ActInertia[i][8] = I_zz;
    }

//    /**
//     * TEST DYN
//     */
//    // Actuation currents for each of the coils for each of the coil sets - unit: A
////    ActuationCurrents[0][2] = 0.1;     //ActuationCurrents[0][1] = 0.1;   ActuationCurrents[0][0] = 0.1;
//
//    for (int i = 0; i < NUM_ACT_SET; ++i) {
//        std::cout << "pL: " << pL[i][0] << " " << pL[i][1] << " " << pL[i][2] <<  std::endl;
//        std::cout << "RL: " << std::endl;
//        std::cout <<  RL[i][0] << " " << RL[i][1] << " " << RL[i][2] <<  std::endl;
//        std::cout <<  RL[i][3] << " " << RL[i][4] << " " << RL[i][5] <<  std::endl;
//        std::cout <<  RL[i][6] << " " << RL[i][7] << " " << RL[i][8] <<  std::endl;
//    }
//
//    std::cout << "xf: " << std::endl;
//
//    for (int i = 0; i < 15; ++i) {
//        std::cout << xf[i] << std::endl;
//    }
    double out_u0[3], out_nL[NUM_ACT_SET][3], out_mL[NUM_ACT_SET][3], ftip_calc[3];
    double x_coil[NUM_ACT_SET][NUM_COIL_STATES], xf[NUM_STATES];

    double out_tau[NUM_ACT_SET][3];

    for (int k = 0; k < 4; ++k) {
        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(CathParams, CathConfig, InsertedLength,
                                                                                  ActuationCurrents, ContactMode,
                                                                                  TipConstraintPoint, TipForce,
                                                                                  IntegrationStepSize, ActInertia,
                                                                                  v_L_pre, w_L_pre, pL, RL, damping,
                                                                                  DELTA_T);


        DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess, ftip_initialguess,
                    out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

    //    std::cout << "out_u0: " << out_u0[0] << " " << out_u0[1] << " " << out_u0[2] <<  std::endl;
    //    for (int i = 0; i < NUM_ACT_SET; ++i) {
    //        std::cout << "out_mL: " << out_mL[i][0] << " " << out_mL[i][1] << " " << out_mL[i][2] <<  std::endl;
    //        std::cout << "out_nL: " << out_nL[i][0] << " " << out_nL[i][1] << " " << out_nL[i][2] <<  std::endl;
    //    }


    int REPS = 100;
    // Get starting timepoint 
    auto start = high_resolution_clock::now();

    for (int cnt = 0; cnt < REPS; cnt++)
        DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                     true, xf, x_coil, ReportedMarkerPos);

    // Get ending timepoint 
    auto stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to  
    // get duration. To cast it to proper unit 
    // use duration cast method 
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << std::endl << "Average time taken by FK Solution in " << REPS << " repetitions: " << duration.count() / REPS << " microseconds" << std::endl;



    //    double v_L_update[NUM_ACT_SET][3], w_L_update[NUM_ACT_SET][3], pL_update[NUM_ACT_SET][3], RL_update[NUM_ACT_SET][9];
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                v_L_pre[j][i] = x_coil[j][i];
                w_L_pre[j][i] = x_coil[j][i + 3];
                pL[j][i] = x_coil[j][i + 6];
            }

            for (int i = 0; i < 9; ++i) {
                RL[j][i] = x_coil[j][i + 9];
            }

            for (int i = 0; i < 3; ++i) {
                mL_initialguess[j][i] = out_mL[j][i];
                nL_initialguess[j][i] = out_nL[j][i];
            }

        }

        for (int i = 0; i < NUM_STATES; ++i) {
            xf_pre[i] = xf[i];
        }

        // Print outputs
        std::cout << " *********TEST DYN Configuration********* " << std::endl;
        printMatrix(out_u0, 1, 3, "Calculated curvature at base");
        printMatrix(out_nL[0], 1, 3, "Calculated internal force at L");
        printMatrix(out_mL[0], 1, 3, "Calculated internal moment force at L");

        printMatrix(ftip_calc, 1, 3, "Calculated Tip Force");
        printMatrix(&(xf[0]), 1, 3, "Catheter Tip Position");
        printMatrix(&(xf[3]), 1, 9, "Catheter Tip Orientation");

    //    printMatrix(&(xf[15]), 1, 3, "v at tip");
    //    printMatrix(&(xf[18]), 1, 3, "w at tip");
    //
        for (int i = 0; i < NUM_ACT_SET; ++i) {
            printMatrix(v_L_pre[i], 1, 3, "v at Coil");
            printMatrix(w_L_pre[i], 1, 3, "w at Coil");
            printMatrix(pL[i], 1, 3, "p at Coil");
            printMatrix(RL[i], 1, 9, "R at Coil");
        }
        std::cout << "--localmin--" << localmin << std::endl;

    }

	return (localmin);
}

