#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <string>
#include <sstream>
#include <random>
#include <unordered_map>
#include <iomanip>
#include "CRM.hpp"
#include "CRMDYN.hpp"

using namespace CRMCatheterModel;

struct SweepResult {
    std::array<double, 3> currents;
    double inserted_length;
    bool converged;
    int localmin;
};

static bool seed_from_fk(
    CRMCatheterModelParams* CathParams,
    CatheterConfiguration* CathConfig,
    const std::array<double, 3>& currents,
    double inserted_length,
    double integration_step_size,
    double xf_out[NUM_STATES],
    double pL_out[NUM_ACT_SET][3],
    double RL_out[NUM_ACT_SET][9]
) {
    CRMForwardKinematicsData FKParams;
    FKParams.CathParams = CathParams;
    FKParams.CathConfig = CathConfig;
    FKParams.ContactMode = ContactModeType::FREE_TIP;
    FKParams.TipForce[0] = FKParams.TipForce[1] = FKParams.TipForce[2] = 0.0;
    FKParams.TipConstraintPoint[0] = FKParams.TipConstraintPoint[1] = FKParams.TipConstraintPoint[2] = 0.0;
    FKParams.deltau0_initialguess[0] = FKParams.deltau0_initialguess[1] = FKParams.deltau0_initialguess[2] = 0.0;
    FKParams.ftip_initialguess[0] = FKParams.ftip_initialguess[1] = FKParams.ftip_initialguess[2] = 0.0;
    FKParams.IntegrationStepSize = integration_step_size;
    FKParams.FinalValueOnly = false;  // Need intermediate coil state outputs

    std::vector<double> markerPosData(CathParams->no_locmarkers * 3, 0.0);
    FKParams.ReportedMarkerPos = reinterpret_cast<double(*)[3]>(markerPosData.data());
    std::vector<double> coilOrientData(CathParams->no_act_set * 9, 0.0);
    FKParams.ReportedCoilOrient = reinterpret_cast<double(*)[9]>(coilOrientData.data());
    std::vector<double> coilPosData(CathParams->no_act_set * 3, 0.0);
    FKParams.ReportedCoilPos = reinterpret_cast<double(*)[3]>(coilPosData.data());

    const int x_dim = NUM_ACT_SET * 3 + 1;
    std::vector<double> in_x(x_dim, 0.0);
    for (int set = 0; set < NUM_ACT_SET; ++set) {
        for (int k = 0; k < 3; ++k) {
            in_x[set * 3 + k] = (set == 0) ? currents[k] : 0.0;
        }
    }
    in_x[NUM_ACT_SET * 3] = inserted_length;

    double out_y[NUM_STATES]{};
    double potentialEnergy = 0.0;
    const int fk_localmin = CRM_ForwardKinematics(in_x.data(), out_y, potentialEnergy, FKParams);
    if (fk_localmin != 0) return false;

    for (int i = 0; i < NUM_STATES; ++i) xf_out[i] = out_y[i];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) pL_out[j][i] = FKParams.ReportedCoilPos[j][i];
        for (int i = 0; i < 9; ++i) RL_out[j][i] = FKParams.ReportedCoilOrient[j][i];
    }
    return true;
}

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> out;
    if (n <= 0) return out;
    if (n == 1) {
        out.push_back(lo);
        return out;
    }
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        out.push_back(lo + (hi - lo) * t);
    }
    return out;
}

static std::vector<double> parse_csv_doubles(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stod(tok));
    }
    return out;
}

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  Grid sweep (default):\n"
        << "    --grid N              Use N points per axis (overrides --vals)\n"
        << "    --min X --max Y       Range for --grid (default -0.2..0.2)\n"
        << "    --vals a,b,c,...      Explicit values per axis (default -0.2,-0.1,0,0.1,0.2)\n"
        << "    --ins a,b,c,...       Insertion lengths (default 94.3)\n"
        << "    --dt T                Dynamics timestep/DELTA_T (default 0.05)\n"
        << "    --step S              Spatial integration step size (default 0.01)\n"
        << "    --no-skip-c3zero       Include c3=0 plane (default skips c1/c2!=0 when c3=0)\n"
        << "\n"
        << "  Random rollout (warm-start):\n"
        << "    --random STEPS         Run sequential random steps instead of grid\n"
        << "    --amp A                Uniform current amplitude bound (default 0.2)\n"
        << "    --seed SEED            RNG seed (default 123)\n";
}

int main(int argc, char** argv) {
    // Load catheter parameters/config used in CRMDYNTest
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../catheterdata/CatheterParameterSet_1_dyn.txt");
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../catheterdata/CatheterSpatialConfiguration_1.txt");

    // Contact mode and numerics
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    double TipForce[3] = {0.0, 0.0, 0.0};
    double TipConstraintPoint[3] = {0.0, 0.0, 0.0};
    double IntegrationStepSize = 0.01;  // mm
    double DELTA_T = 0.05;              // s

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

    // Defaults: match historical sweep.
    std::vector<double> current_vals = {-0.2, -0.1, 0.0, 0.1, 0.2};
    std::vector<double> insertions = {94.3};
    bool skip_c3_zero_plane = true;
    int grid_n = 0;
    double grid_min = -0.2;
    double grid_max = 0.2;

    int random_steps = 0;
    double random_amp = 0.2;
    uint32_t random_seed = 123;

    // Parse args (simple flag parser).
    // Note: avoid introducing additional dependencies; keep it minimal.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                print_usage(argv[0]);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--grid") {
            grid_n = std::stoi(need_value("--grid"));
        } else if (arg == "--min") {
            grid_min = std::stod(need_value("--min"));
        } else if (arg == "--max") {
            grid_max = std::stod(need_value("--max"));
        } else if (arg == "--vals") {
            current_vals = parse_csv_doubles(need_value("--vals"));
        } else if (arg == "--ins") {
            insertions = parse_csv_doubles(need_value("--ins"));
        } else if (arg == "--dt") {
            DELTA_T = std::stod(need_value("--dt"));
        } else if (arg == "--step") {
            IntegrationStepSize = std::stod(need_value("--step"));
        } else if (arg == "--no-skip-c3zero") {
            skip_c3_zero_plane = false;
        } else if (arg == "--random") {
            random_steps = std::stoi(need_value("--random"));
        } else if (arg == "--amp") {
            random_amp = std::stod(need_value("--amp"));
        } else if (arg == "--seed") {
            random_seed = static_cast<uint32_t>(std::stoul(need_value("--seed")));
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (grid_n > 0) {
        current_vals = linspace(grid_min, grid_max, grid_n);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Config: step=" << IntegrationStepSize << " dt=" << DELTA_T
              << " vals=" << current_vals.size() << " ins=" << insertions.size()
              << " skip_c3zero=" << (skip_c3_zero_plane ? "true" : "false") << "\n";
    if (random_steps > 0) {
        std::cout << "Mode: random steps=" << random_steps << " amp=" << random_amp << " seed=" << random_seed << "\n";
    } else {
        std::cout << "Mode: grid sweep\n";
    }

    std::vector<SweepResult> results;
    for (double ins : insertions) {
        // Warm-start state carried across the sweep (continuation), like main/CRMDYN_test.cpp.
        bool have_seed = false;
        double xf_seed[NUM_STATES]{};
        double pL_seed[NUM_ACT_SET][3]{};
        double RL_seed[NUM_ACT_SET][9]{};
        double v_L_seed[NUM_ACT_SET][3]{};
        double w_L_seed[NUM_ACT_SET][3]{};
        double nL_guess[NUM_ACT_SET][3]{};
        double mL_guess[NUM_ACT_SET][3]{};

        auto reset_dynamic_guesses = [&]() {
            std::memset(v_L_seed, 0, sizeof(v_L_seed));
            std::memset(w_L_seed, 0, sizeof(w_L_seed));
            std::memset(nL_guess, 0, sizeof(nL_guess));
            std::memset(mL_guess, 0, sizeof(mL_guess));
        };

        // Build a sweep order that changes currents gradually (snake pattern) to help continuation.
        std::vector<std::array<double, 3>> sweep_currents;
        if (random_steps > 0) {
            // Random rollout: sequential warm-started steps.
            std::mt19937 rng(random_seed);
            std::uniform_real_distribution<double> unif(-random_amp, random_amp);
            sweep_currents.reserve(static_cast<size_t>(random_steps));
            for (int k = 0; k < random_steps; ++k) {
                sweep_currents.push_back({unif(rng), unif(rng), unif(rng)});
            }
        } else {
            // Grid: snake pattern to keep adjacent steps close.
            for (size_t iz = 0; iz < current_vals.size(); ++iz) {
                const double c3 = current_vals[iz];
                const bool reverse_c2 = (iz % 2 == 1);
                for (size_t iy = 0; iy < current_vals.size(); ++iy) {
                    const double c2 = reverse_c2 ? current_vals[current_vals.size() - 1 - iy] : current_vals[iy];
                    const bool reverse_c1 = ((iz + iy) % 2 == 1);
                    for (size_t ix = 0; ix < current_vals.size(); ++ix) {
                        const double c1 = reverse_c1 ? current_vals[current_vals.size() - 1 - ix] : current_vals[ix];
                        if (skip_c3_zero_plane) {
                            // Historical behavior: if channel 3 is zero, keep channel 1/2 at zero as well.
                            if (c3 == 0.0 && (c1 != 0.0 || c2 != 0.0)) continue;
                        }
                        sweep_currents.push_back({c1, c2, c3});
                    }
                }
            }
        }

        for (const auto& c : sweep_currents) {
            double ActuationCurrents[NUM_ACT_SET][3]{};
            for (int set = 0; set < NUM_ACT_SET; ++set) {
                for (int k = 0; k < 3; ++k) ActuationCurrents[set][k] = (set == 0) ? c[k] : 0.0;
            }

            // Bootstrap the continuation from FK the first time.
            if (!have_seed) {
                if (!seed_from_fk(&CathParams, &CathConfig, c, ins, IntegrationStepSize, xf_seed, pL_seed, RL_seed)) {
                    results.push_back({c, ins, false, -100});
                    std::cout << "  curr=(" << c[0] << "," << c[1] << "," << c[2] << "), ins=" << ins
                              << " failed to initialize from FK (bootstrap)." << std::endl;
                    continue;
                }
                reset_dynamic_guesses();
                have_seed = true;
            }

            double xf_backup[NUM_STATES];
            double pL_backup[NUM_ACT_SET][3];
            double RL_backup[NUM_ACT_SET][9];
            double v_L_backup[NUM_ACT_SET][3];
            double w_L_backup[NUM_ACT_SET][3];
            double nL_backup[NUM_ACT_SET][3];
            double mL_backup[NUM_ACT_SET][3];
            std::memcpy(xf_backup, xf_seed, sizeof(xf_backup));
            std::memcpy(pL_backup, pL_seed, sizeof(pL_backup));
            std::memcpy(RL_backup, RL_seed, sizeof(RL_backup));
            std::memcpy(v_L_backup, v_L_seed, sizeof(v_L_backup));
            std::memcpy(w_L_backup, w_L_seed, sizeof(w_L_backup));
            std::memcpy(nL_backup, nL_guess, sizeof(nL_backup));
            std::memcpy(mL_backup, mL_guess, sizeof(mL_backup));

            auto run_one = [&](int& localmin_out,
                               double out_u0[3],
                               double out_mL[NUM_ACT_SET][3],
                               double out_nL[NUM_ACT_SET][3],
                               double out_tau[NUM_ACT_SET][3],
                               double ftip_calc[3]) -> bool {
                CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
                    CathParams, CathConfig, ins, ActuationCurrents, ContactMode,
                    TipConstraintPoint, TipForce, IntegrationStepSize, ActInertia,
                    v_L_seed, w_L_seed, pL_seed, RL_seed, damping, DELTA_T);

                double ftip_initialguess[3] = {0.0, 0.0, 0.0};
                DynamicsBVP(BVPParams, xf_seed, mL_guess, nL_guess, ftip_initialguess,
                            out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin_out);

                if (localmin_out != 0) return false;

                // Integrate IVP once, then update warm-start state for the next step.
                double x_coil[NUM_ACT_SET][NUM_COIL_STATES]{};
                double xf_next[NUM_STATES]{};
                DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc, true, xf_next, x_coil, nullptr);

                for (int j = 0; j < NUM_ACT_SET; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        v_L_seed[j][i] = x_coil[j][i];
                        w_L_seed[j][i] = x_coil[j][i + 3];
                        pL_seed[j][i] = x_coil[j][i + 6];
                        mL_guess[j][i] = out_mL[j][i];
                        nL_guess[j][i] = out_nL[j][i];
                    }
                    for (int i = 0; i < 9; ++i) RL_seed[j][i] = x_coil[j][i + 9];
                }
                std::memcpy(xf_seed, xf_next, sizeof(xf_seed));
                return true;
            };

            int localmin = 0;
            double out_u0[3]{};
            double out_mL[NUM_ACT_SET][3]{};
            double out_nL[NUM_ACT_SET][3]{};
            double out_tau[NUM_ACT_SET][3]{};
            double ftip_calc[3]{};

            bool converged = run_one(localmin, out_u0, out_mL, out_nL, out_tau, ftip_calc);

            // Recovery attempt: reseed from FK if warm-start fails (mirrors Python wrapper behavior).
            if (!converged) {
                double xf_fk[NUM_STATES]{};
                double pL_fk[NUM_ACT_SET][3]{};
                double RL_fk[NUM_ACT_SET][9]{};
                if (seed_from_fk(&CathParams, &CathConfig, c, ins, IntegrationStepSize, xf_fk, pL_fk, RL_fk)) {
                    std::memcpy(xf_seed, xf_fk, sizeof(xf_seed));
                    std::memcpy(pL_seed, pL_fk, sizeof(pL_seed));
                    std::memcpy(RL_seed, RL_fk, sizeof(RL_seed));
                    reset_dynamic_guesses();
                    converged = run_one(localmin, out_u0, out_mL, out_nL, out_tau, ftip_calc);
                }
            }

            if (!converged) {
                // Keep the previous converged state for continuation to the next point.
                std::memcpy(xf_seed, xf_backup, sizeof(xf_backup));
                std::memcpy(pL_seed, pL_backup, sizeof(pL_backup));
                std::memcpy(RL_seed, RL_backup, sizeof(RL_backup));
                std::memcpy(v_L_seed, v_L_backup, sizeof(v_L_backup));
                std::memcpy(w_L_seed, w_L_backup, sizeof(w_L_backup));
                std::memcpy(nL_guess, nL_backup, sizeof(nL_backup));
                std::memcpy(mL_guess, mL_backup, sizeof(mL_backup));
            }

            results.push_back({c, ins, converged, localmin});
            if (!converged && random_steps <= 0) {
                std::cout << "  curr=(" << c[0] << "," << c[1] << "," << c[2] << "), ins=" << ins
                          << " failed with localmin=" << localmin << std::endl;
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
                      << "), ins=" << r.inserted_length << " failed (localmin=" << r.localmin << ")\n";
            printed++;
        }
    }
    if (failures > printed) {
        std::cout << "  ... " << (failures - printed) << " more\n";
    }
    return failures;
}
