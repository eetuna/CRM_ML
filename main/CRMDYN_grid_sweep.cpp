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

    // Initial guesses
    double deltau0_initialguess[3] = {0.0, 0.0, 0.0};
    double ftip_initialguess[3] = {0.0, 0.0, 0.0};
    double nL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    double mL_initialguess[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    double v_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};
    double w_L_pre[NUM_ACT_SET][3] = {{0.0, 0.0, 0.0}};

    // Seed tip state from CRMDYN_test.cpp
    double xf_pre[NUM_STATES] = {
        -0.458414144062750, 34.411241976876518, 70.457561147732264,
        0.999932718178103, 0.009921777042635, -0.006009780134551,
        -0.004651734390922, 0.817579117734723, 0.575797488368325,
        0.010626405041486, -0.575730791763330, 0.817570262993625,
        -0.015378744286498, 0.000001280646594, -0.000349413951059};
    double pL[NUM_ACT_SET][3] = {{-0.248418562587657, 17.707660318406560, 46.752162601547091}};
    double RL[NUM_ACT_SET][9] = {{
        0.999919687839427, 0.009924211584043, -0.007882125064742,
        -0.003571217614502, 0.817374079004311, 0.576096225796181,
        0.012159945552960, -0.576021809479719, 0.817343875445250}};

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
                    if (!converged && results.size() < 10) {
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
