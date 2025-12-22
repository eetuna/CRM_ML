mex COMPFLAGS='-std:c++17' Load_CRMCatheterModelParams_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../..

mex COMPFLAGS='-std:c++17' Load_CatheterConfiguration_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../..

mex -R2018a COMPFLAGS='-std:c++17' CRM_ForwardKinematics_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../..

mex -R2018a COMPFLAGS='-std:c++17' CRM_FKJacobian_Analytical_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../..

mex -R2018a COMPFLAGS='-std:c++17' CRM_CalculateCatheterEnergy_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../..
