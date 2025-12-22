#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRMTest.h"
#include <iostream>
#include <chrono>

using namespace CRMCatheterModel;

int RunFKExample() {
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../data/catheter_params/CatheterParameterSet_1.txt");
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../data/catheter_params/CatheterSpatialConfiguration_1.txt");

    CRMForwardKinematicsData FKParams;
    FKParams.CathParams = &CathParams;
    FKParams.CathConfig = &CathConfig;
    FKParams.ContactMode = ContactModeType::FREE_TIP;
    FKParams.TipForce.setZero();
    FKParams.deltau0_initialguess.setZero();
    FKParams.IntegrationStepSize = 0.1;
    FKParams.FinalValueOnly = true;

    double ActuationCurrents[NUM_ACT_SET][3] = { {0.02, -0.01, 0.01} };
    double InsertedLength = 50.0;

    Eigen::VectorXd control_inputs(NUM_ACT_SET * 3 + 1);
    for (int i = 0; i < NUM_ACT_SET; i++) for (int j = 0; j < 3; j++) control_inputs(i * 3 + j) = ActuationCurrents[i][j];
    control_inputs(NUM_ACT_SET * 3) = InsertedLength;

    double PotentialEnergy;
    int localmin;
    Eigen::VectorXd FKsolution = CRM_ForwardKinematics(control_inputs, FKParams, PotentialEnergy, localmin);

    std::cout << "FK Solution p: " << FKsolution.head<3>().transpose() << std::endl;
    return localmin;
}

int main() {
    return RunFKExample();
}