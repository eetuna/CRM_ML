mex CXXFLAGS='$CXXFLAGS $COMPFLAGS -std=c++17' Load_CRMCatheterModelParams_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../.. -I/usr/local/include/eigen3/

mex CXXFLAGS='$CXXFLAGS $COMPFLAGS -std=c++17' Load_CatheterConfiguration_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../.. -I/usr/local/include/eigen3/

mex -R2018a CXXFLAGS='$CXXFLAGS $COMPFLAGS -std=c++17' CRM_ForwardKinematics_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../.. -I/usr/local/include/eigen3/

mex -R2018a CXXFLAGS='$CXXFLAGS $COMPFLAGS -std=c++17' CRM_FKJacobian_Analytical_matlab.cpp "../src/*.cpp" "../src/numerical/*.cpp" -I../src/ -I../src/numerical/ -I../.. -I/usr/local/include/eigen3/
