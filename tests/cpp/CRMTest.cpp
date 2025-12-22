#include "CRM.hpp"		// This is the only header that needs to be included to use the standard CRM Kinematics APIs
using namespace CRMCatheterModel;
#include "CRMTest.h"	// The CRMTest specific stuff is here


template <typename T>
void printMatrix(T *p, int D1, int D2, const char *text) {
	std::cout << text << " --" << std::endl;
	for (int i=0; i<D1; i++) {
		for (int j=0; j<D2; j++) {
			std::cout << *(p+i*D2+j) << " ";
		}
		std::cout << std::endl;
	}
	//std::cout << "----" << std::endl;
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


int RunFKExample(void);

int main(int argc, char** argv) {

	int FailFlag = 0;

	//RunExample();

	std::cout << std::endl << std::endl << "### Running Examples ..." << std::endl << std::endl;

	RunFKExample();
//
//	std::cout << std::endl << std::endl << "### Running Tests ..." << std::endl << std::endl;
//
//	if (FailFlag == 0) {
//		std::cout << std::endl << std::endl << "###" << std::endl;
//		std::cout << "ALL TESTS PASSED!..." << std::endl;
//		std::cout << "###" << std::endl;
//	}
//	else {
//		std::cout << std::endl << std::endl << "###" << std::endl;
//		std::cout << FailFlag << " TEST(S) FAILED!..." << std::endl;
//		std::cout << "###" << std::endl;
//	}
	return (FailFlag);

}

// Example showing how to use the CRM_ForwardKinematics functions 
//		to calculate the catheter forward kinematics
// This is the preferred method
//
int RunFKExample(void) {

    std::cout << "### RunFKExample() --- CRM Forward Kinematics Examples... " << std::endl;
    std::cout << std::endl << "Free Space Deflection Example: " << std::endl << std::endl;

    // *** Load parameters from file
    // This step would typically needs to be executed only once
    //   Physical Description of the Catheter
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../data/catheter_params/CatheterParameterSet_1_new.txt");
    //   Catheter Configuration in spatial coordinates
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../data/catheter_params/CatheterSpatialConfiguration_1.txt");

    // *** Other External variables
    // specify if catheter is in free space or if the catheter tip is constrained to a contact point
    ContactModeType ContactMode = ContactModeType::FREE_TIP;
    // External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0)  - unit: ??
    //   (this will be used when ContactMode == ContactModeType::FREE_TIP)
    double TipForce[3] = { 0.0, 0.0, 0.0 };
    // The spatial coordinates of the point where the catheter tip is constrained to be
    //   (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double TipConstraintPoint[3] = { 0.0, 0.0, 0.0 };

    // *** Control Inputs
    // Inserted Length of the catheter (length of the catheter that is inside the heart chamber) - unit: mm
    double InsertedLength = 94.0; //40.0;
    // Actuation currents for each of the coils for each of the coil sets - unit: A
    double ActuationCurrents[NUM_ACT_SET][3] = { {0, 0.0, 0.0}};

    // *** Numerical Computation Params
    // Stepsize used in numerical integration along the length of the catheter during IVP - unit: mm
    double IntegrationStepSize = 0.2;

    // *** Storage for storing localization marker positions and actuation coil orientations
    double (*ReportedMarkerPos)[3] = new double [CathParams.no_locmarkers][3];
    double (*ReportedCoilPos)[3] = new double[CathParams.no_act_set][3];
    double (*ReportedCoilOrient)[9] = new double[CathParams.no_act_set][9];


    // Define initial guesses to be used when solving boundary value problem
    // initial guess for the delta_curvature at the catheter base ( u0 = deltau0 + ustar0 )
    double deltau0_initialguess[3] = { 0.0, 0.0, 0.0 };
    // initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
    double ftip_initialguess[3] = { 0.0, 0.0, 0.0 };

    // numerical nonlinear equation solver diagnostic outputs
    int localmin;

    //
    // New Forward Kinematics Function
    //

    // The Forward Kinematics Function uses a different parameter structure than BVP
    CRMForwardKinematicsData FKParams;
    FKParams.CathConfig = &CathConfig;
    FKParams.CathParams = &CathParams;
    FKParams.ContactMode = ContactMode;
    FKParams.FinalValueOnly = false;  // we want the localization coil locations, too
    FKParams.ReportedMarkerPos = ReportedMarkerPos;
    FKParams.ReportedCoilPos = ReportedCoilPos;
    FKParams.ReportedCoilOrient = ReportedCoilOrient;
    for (int i = 0; i < 3; i++) FKParams.TipConstraintPoint[i] = TipConstraintPoint[i];
    for (int i = 0; i < 3; i++) FKParams.TipForce[i] = TipForce[i];
    for (int i = 0; i < 3; i++) FKParams.deltau0_initialguess[i] = deltau0_initialguess[i];
    for (int i = 0; i < 3; i++) FKParams.ftip_initialguess[i] = ftip_initialguess[i];
    FKParams.IntegrationStepSize = IntegrationStepSize;
    // inputs
    VectorXd control_inputs(NUM_ACT_SET * 3 + 1); // actuator currents (distal to proximal) followed by inserted length
    for (int i = 0; i < NUM_ACT_SET; i++)  for (int j = 0; j < 3; j++) control_inputs(i * 3 + j) = ActuationCurrents[i][j];
    control_inputs(NUM_ACT_SET * 3) = InsertedLength;
    // and outputs
    int outputdim;
    if (ContactMode == ContactModeType::FREE_TIP) outputdim = 15; else outputdim = 18;
    VectorXd FKsolution(outputdim); // p[0..2],R[0..8],deltau0[0..2](,ftip[0..2])  R: in row major order
    double   PotentialEnergy;

    // Multiple repetitions to more reliably measure time

    // Cosserat Rod Model - Solve the Forward Kinematics
    FKsolution = CRM_ForwardKinematics(control_inputs, FKParams, PotentialEnergy, localmin);

    std::cout << "FKParams.COILPOS:" << FKParams.ReportedCoilPos[0][0] << " " << FKParams.ReportedCoilPos[0][1] << " " << FKParams.ReportedCoilPos[0][2]   << std::endl;

    std::cout << "FKParams.COILROT:" << FKParams.ReportedCoilOrient[0][0] << " " << FKParams.ReportedCoilOrient[0][1] << " " << FKParams.ReportedCoilOrient[0][2]   << std::endl;
    std::cout << "FKParams.COILROT:" << FKParams.ReportedCoilOrient[0][3] << " " << FKParams.ReportedCoilOrient[0][4] << " " << FKParams.ReportedCoilOrient[0][5]   << std::endl;
    std::cout << "FKParams.COILROT:" << FKParams.ReportedCoilOrient[0][6] << " " << FKParams.ReportedCoilOrient[0][7] << " " << FKParams.ReportedCoilOrient[0][8]   << std::endl;


    //display the results,
    std::cout << "FK Output -- p,R,deltau0: \n" << FKsolution.transpose() << "\n";
    std::cout << "PE:" << PotentialEnergy << std::endl;
    std::cout << "localmin:" << localmin << std::endl;
    std::cout << "----" << std::endl;

    return 0;
}
