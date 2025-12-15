#include <iostream>
#include <vector>
#include <array>
#include "CRM.hpp"
#include "CRMDYN.hpp"

using namespace CRMCatheterModel;

struct SweepResult {
    std::array<double, 3> currents;
    double inserted_length;
    bool converged;
};

int main() {
    // Load catheter parameters/config used in CRMDYNTest
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../catheterdata/CatheterParameterSet_1_dyn.txt");
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../catheterdata/CatheterSpatialConfiguration_1.txt");

    // Contact mode and numerics
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    double TipForce[3] = {0.0, 0.0, 0.0};
    double TipConstraintPoint[3] = {0.0, 0.0, 0.0};
    double IntegrationStepSize = 0.01;  // smaller step to help convergence
    double DELTA_T = 0.05;

    // Damping same as CRMDYN_test.cpp
    double damping[NUM_ACT_SET][6] = {{12.1761626666366, 12.1761626666366,
                                       284.429938756989,
                                       0.0304776127617393, 0.0304776127617393,
                                       0.00502712804532508}};

    // Build actuator inertia
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

    // Grids to test
    std::vector<double> current_vals = {-0.2, -0.1, 0.0, 0.1, 0.2};
    // Use the insertion length from CRMDYNTest only
    std::vector<double> insertions = {94.3};

    std::vector<SweepResult> results;
    for (double ins : insertions) {
        for (double c1 : current_vals) {
            for (double c2 : current_vals) {
                for (double c3 : current_vals) {
                    // If channel 3 is zero, keep channel 1/2 at zero as well
                    if (c3 == 0.0 && (c1 != 0.0 || c2 != 0.0)) {
                        continue;
                    }
                    double ActuationCurrents[NUM_ACT_SET][3] = {{c1, c2, c3}};

                    // --- FIX: Initialize dynamics from forward kinematics ---
                    // Setup FK parameters
                    CRMForwardKinematicsData FKParams;
                    FKParams.CathParams = &CathParams;
                    FKParams.CathConfig = &CathConfig;
                    FKParams.ContactMode = ContactModeType::FREE_TIP;
                    FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
                    FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
                    FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
                    FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
                    FKParams.IntegrationStepSize = IntegrationStepSize;
                    FKParams.FinalValueOnly = false; // We need intermediate coil states

                    // Allocate storage for FK results
                    double markerPosData[CathParams.no_locmarkers * 3];
                    FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData);
                    double coilOrientData[CathParams.no_act_set * 9];
                    FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData);
                    double coilPosData[CathParams.no_act_set * 3];
                    FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData);
                    
                    // Build FK input vector
                    int x_dim = NUM_ACT_SET * 3 + 1;
                    double in_x[x_dim];
                    for (int i = 0; i < NUM_ACT_SET * 3; ++i) {
                        in_x[i] = ActuationCurrents[0][i];
                    }
                    in_x[NUM_ACT_SET * 3] = ins;

                    // Output vector
                    int y_dim = 3 + 9 + 3;
                    double out_y[y_dim];
                    double potentialEnergy;

                    // Call FK to get a good initial state
                    int fk_localmin = CRM_ForwardKinematics(in_x, out_y, potentialEnergy, FKParams);

                    if (fk_localmin != 0) {
                         SweepResult sr = {{c1, c2, c3}, ins, false};
                         results.push_back(sr);
                         std::cout << "  curr=(" << c1 << "," << c2 << "," << c3 << "), ins=" << ins
                                   << " failed to initialize from FK." << std::endl;
                         continue;
                    }

                    // --- Use FK results to seed the dynamics ---
                    double xf_pre[NUM_STATES];
                    for(int i = 0; i < 15; ++i) xf_pre[i] = out_y[i];

                    double pL[NUM_ACT_SET][3];
                    double RL[NUM_ACT_SET][9];
                     for (int j = 0; j < NUM_ACT_SET; j++) {
                        for (int i = 0; i < 3; i++) pL[j][i] = FKParams.ReportedCoilPos[j][i];
                        for (int i = 0; i < 9; i++) RL[j][i] = FKParams.ReportedCoilOrient[j][i];
                    }

                    // Initial guesses for dynamics (zero velocity)
                    double v_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
                    double w_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
                    double nL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
                    double mL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
                    double ftip_initialguess[3] = {0.0, 0.0, 0.0};

                    CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
                        CathParams, CathConfig, ins, ActuationCurrents, ContactMode,
                        TipConstraintPoint, TipForce, IntegrationStepSize, ActInertia,
                        v_L_pre, w_L_pre, pL, RL, damping, DELTA_T);

                    double out_u0[3], out_nL[NUM_ACT_SET][3], out_mL[NUM_ACT_SET][3], ftip_calc[3], out_tau[NUM_ACT_SET][3];
                    double xf[NUM_STATES];
                    int localmin = 0;

                    // Solve BVP
                    DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess,
                                ftip_initialguess, out_u0, out_mL, out_nL, out_tau,
                                ftip_calc, localmin);
                    bool converged = (localmin == 0);
                    if (converged) {
                        // Integrate IVP once to mirror runtime step
                        double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
                        DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc, true, xf, x_coil, nullptr);
                    }

                    SweepResult sr = {{c1, c2, c3}, ins, converged && localmin == 0};
                    results.push_back(sr);
                    if (!converged) {
                        std::cout << "  curr=(" << c1 << "," << c2 << "," << c3 << "), ins=" << ins
                                  << " failed with localmin=" << localmin << std::endl;
                    }
                }
            }
        }
    }

    int failures = 0;
    for (const auto &r : results) {
        if (!r.converged) failures++;
    }

    std::cout << "Tested " << results.size() << " cases\n";
    std::cout << "Failures: " << failures << "\n";
    int printed = 0;
    for (const auto &r : results) {
        if (!r.converged && printed < 20) {
            std::cout << "  curr=(" << r.currents[0] << "," << r.currents[1] << "," << r.currents[2]
                      << "), ins=" << r.inserted_length << " failed\n";
            printed++;
        }
    }
    if (failures > printed) {
        std::cout << "  ... " << (failures - printed) << " more\n";
    }
    return failures;
}
