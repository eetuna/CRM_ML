// CRM Torch Extension - Parameter Management
// Option C Phase 4: Integration with actual dynamics

#pragma once

#include <memory>
#include <string>
#include <stdexcept>
#include "CRM.hpp"
#include "CRMDYN.hpp"

namespace crm_torch {

using namespace CRMCatheterModel;

/**
 * @brief Global parameter manager for CRM extension
 *
 * This singleton manages catheter parameters that are loaded once
 * and reused across forward/backward passes.
 */
class CRMParams {
public:
    // Get singleton instance
    static CRMParams& getInstance() {
        static CRMParams instance;
        return instance;
    }

    // Load parameters from files
    bool loadFromFiles(const std::string& param_file, const std::string& config_file) {
        try {
            params_ = std::make_unique<CRMCatheterModelParams>(
                Load_CRMCatheterModelParams(param_file.c_str())
            );
            config_ = Load_CatheterConfiguration(config_file.c_str());

            // Compute actuation inertia from loaded parameters (same as Python bindings)
            for (int i = 0; i < params_->no_act_set && i < NUM_ACT_SET; i++) {
                double mass = params_->ActMass[i];
                double r_out = params_->OuterRadius[0];
                double r_in = params_->InnerRadius[0];
                double seg_len = params_->SegLengths[2 * i + 1];

                double I_zz = 0.5 * mass * (r_out * r_out + r_in * r_in);
                double I_xx = 0.25 * mass * (r_out * r_out + r_in * r_in) +
                              (1.0 / 12.0) * mass * seg_len * seg_len;

                act_inertia_[i][0] = I_xx;
                act_inertia_[i][4] = I_xx;
                act_inertia_[i][8] = I_zz;
            }

            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load CRM parameters: " + std::string(e.what()));
        }
    }

    // Check if initialized
    bool isInitialized() const { return initialized_; }

    // Get parameters (const access)
    const CRMCatheterModelParams* getParams() const {
        if (!initialized_) {
            throw std::runtime_error("CRM parameters not loaded. Call initialize_params() first.");
        }
        return params_.get();
    }

    // Get config (const access)
    const CatheterConfiguration& getConfig() const {
        if (!initialized_) {
            throw std::runtime_error("CRM parameters not loaded. Call initialize_params() first.");
        }
        return config_;
    }

    // Get timestep
    double getDt() const { return dt_; }
    void setDt(double dt) { dt_ = dt; }

    // Get integration step size
    double getIntegrationStepSize() const { return integration_step_size_; }
    void setIntegrationStepSize(double step_size) { integration_step_size_ = step_size; }

    // Get integrator type
    IntegratorType getIntegratorType() const { return integrator_type_; }
    void setIntegratorType(IntegratorType type) { integrator_type_ = type; }

    // Get default damping values
    void getDamping(double damping_out[NUM_ACT_SET][6]) const {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 6; i++) {
                damping_out[j][i] = damping_[j][i];
            }
        }
    }

    void setDamping(const double damping_in[NUM_ACT_SET][6]) {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 6; i++) {
                damping_[j][i] = damping_in[j][i];
            }
        }
    }

    // Get default actuation inertia
    void getActInertia(double inertia_out[NUM_ACT_SET][9]) const {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 9; i++) {
                inertia_out[j][i] = act_inertia_[j][i];
            }
        }
    }

    void setActInertia(const double inertia_in[NUM_ACT_SET][9]) {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 9; i++) {
                act_inertia_[j][i] = inertia_in[j][i];
            }
        }
    }

private:
    CRMParams() {
        // Initialize default damping values
        for (int j = 0; j < NUM_ACT_SET; j++) {
            damping_[j][0] = 12.1761626666366;
            damping_[j][1] = 12.1761626666366;
            damping_[j][2] = 284.429938756989;
            damping_[j][3] = 0.0304776127617393;
            damping_[j][4] = 0.0304776127617393;
            damping_[j][5] = 0.00502712804532508;

            for (int i = 0; i < 9; i++) {
                act_inertia_[j][i] = 0.0;
            }
        }
    }

    // Delete copy/move constructors
    CRMParams(const CRMParams&) = delete;
    CRMParams& operator=(const CRMParams&) = delete;

    std::unique_ptr<CRMCatheterModelParams> params_;
    CatheterConfiguration config_;
    bool initialized_ = false;

    // Dynamics parameters
    double dt_ = 0.02;  // Default 50Hz
    double integration_step_size_ = 0.2;  // mm
    IntegratorType integrator_type_ = IntegratorType::ABM4;

    // Default damping and inertia
    double damping_[NUM_ACT_SET][6];
    double act_inertia_[NUM_ACT_SET][9];
};

} // namespace crm_torch
