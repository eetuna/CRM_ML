function obj = load_parameters( youngModulus,shearModulus,Alignment_Params)

%This file introduces all of the geometric parameters of catheter 
%and calculates the weights of the catheter and coils.
%Additionally, it also defines the magnetic field B_0, currents of coils, I_current

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%magnetic field

Bx = 0; %N/(A*mm) unit
By = 3e-3;%%N/(A*mm) unit  
Bz = 0; %*N/(A*mm)unit  

obj.B_0 = [Bx By Bz]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% prototype #6.2

obj.marker_length_base = 6.40;%5.50
obj.marker_length_coil = 19.10; %18.3;
obj.marker_length_tip = 6.00; %5.62


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% prototype #6.2
%Define the segments of the catheter
NumSegments = 8;
obj.L = 82.65;  %catheter length (mm)  82.00
Delta = obj.L/NumSegments;%(mm)

% This segment carries nothing.
NumSegments_tip = 1;
obj.L_A = 18.70;%19.0

NumSegments_total = NumSegments + NumSegments_tip;

% Define the dimensions of the catheter
r1= 0.125*25.4/2;%outer radius (mm)
r2= 0.078*25.4/2;%inner radius (mm)

% Calculate the moment of inertia
I_x = pi/4*(r1^4-r2^4);%(mm^4)
I_y = I_x;
I_xy = 0;
J = pi/2*(r1^4-r2^4);%(mm^4)  polar moment of inertia of area

obj.I_cons = [I_y I_xy 0;
          I_xy I_x 0;
          0 0 1];%(mm^4)


%%material properties of the catheter
rho_cath = 1119.6; %catheter density (kg/m^3)
E = youngModulus; %%Young's modulus(N/mm^2)
% v = 0.47;
% G = E/(2*(1+v));%%shear modulus(N/mm^2)
G = shearModulus;

C = 1/Delta*[E*(I_x*I_y-I_xy^2) 0 0;
             0 E*(I_x*I_y-I_xy^2) 0;
             0 0 G*J];%(N*mm^5) &(N*mm)
C_lastSeg = 1/obj.L_A*[E*(I_x*I_y-I_xy^2) 0 0;
                   0 E*(I_x*I_y-I_xy^2) 0;
                   0 0 G*J];%(N*mm^5) &(N*mm)
               
%%radius of the coil and density of the coil
rho_coil = 8.940e3;%copper density (kg/m^3)
r_coil = 0.0635/2; % mm
         
% Density of the blood
rho_blood = 1060; %blood density (kg/m^3)
% rho_blood = 0;

% Calculate the weights of the segment and the coils
% g =  0;%9.81;%
g = 9.81;
W_x = pi*(r1^2-r2^2)*Delta*(rho_cath-rho_blood)*g*1.0e-9; %Unit length weight (N)
obj.Delta_W = [0 0 W_x]';

W_A_x = pi*(r1^2-r2^2)*obj.L_A*(rho_cath-rho_blood)*g*1.0e-9; %Unit length weight (N)
obj.W_A = [0 0 W_A_x]';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Introduction of coils
%Nums of winding turns for each coil
N_1 = 30;%number of turns of coil 1
N_2 = 30;%%number of turns of coil 2
N_3 = 100;%number of turns of coil 3

%dimension of the side coil (square side)
lengthSideCoil_1 = 18.30;%18.25; % the length of the side coil (bottom)
lengthSideCoil_2 = 18.30;%18.25; % the length of the side coil (top)

widthSideCoil_1 = 5.10; %%% 2.63 is the width of the side coil; 18.25 is the length of the side coil
widthSideCoil_2 = 5.10;

%diameter of the axial coil
OutDia_AxialCoil = 1.8925*2; %%%4.45 is the outer diameter of the axial coil

%area of cross-section (mm^2)
% as the shape of the side coil is a diamond shape, the effective area is
% as follows 0.5*width*height
A_1 = 0.5*(widthSideCoil_1)*(lengthSideCoil_1);
A_2 = 0.5*(widthSideCoil_2)*(lengthSideCoil_2);
A_3 = pi*(OutDia_AxialCoil/2)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define the mu matrix for magnetic moments

%%% without a matrix to compensate the alignment of the magnetic moment
%%% directions, the three coils are assumed to align with XYZ axes.
% Alignment_Matrix = [1 0 0;
%     0 1 0;
%     0 0 -1];
% MuMatrix = Alignment_Matrix*[N_1*A_1 0 0;
%     0 N_2*A_2 0;
%     0 0 N_3*A_3];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta_1_a = Alignment_Params(1);
theta_2_a = Alignment_Params(2);

theta_1_b = Alignment_Params(3);
theta_2_b = Alignment_Params(4);

alpha_1_a = Alignment_Params(5);
alpha_2_a = Alignment_Params(6);

alpha_1_b = Alignment_Params(7);
alpha_2_b = Alignment_Params(8);

beta_1 = Alignment_Params(9);
beta_2 = Alignment_Params(10);

obj.MuMatrix = zeros(3,3);
obj.MuMatrix(1,1) = 0.5*N_1*A_1*(cos(theta_1_a)*cos(theta_2_a) + cos(theta_1_b)*cos(theta_2_b));
obj.MuMatrix(2,1) = 0.5*N_1*A_1*(sin(theta_2_a) + sin(theta_2_b));
obj.MuMatrix(3,1) = 0.5*N_1*A_1*(-sin(theta_1_a)*cos(theta_2_a) - sin(theta_1_b)*cos(theta_2_b));

obj.MuMatrix(1,2) = -0.5*N_2*A_2*(sin(alpha_2_a) + sin(alpha_2_b));
obj.MuMatrix(2,2) = 0.5*N_2*A_2*(cos(alpha_1_a)*cos(alpha_2_a) + cos(alpha_1_b)*cos(alpha_2_b));
obj.MuMatrix(3,2) = 0.5*N_2*A_2*(sin(alpha_1_a)*cos(alpha_2_a) + sin(alpha_1_b)*cos(alpha_2_b));

obj.MuMatrix(1,3) = N_3*A_3*(sin(beta_2));
obj.MuMatrix(2,3) = -N_3*A_3*(sin(beta_1)*cos(beta_2));
obj.MuMatrix(3,3) = N_3*A_3*(cos(beta_1)*cos(beta_2));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%weight of coils
F_x = (rho_coil-rho_blood)*...
    (pi*r_coil^2*(2*pi*(r1))*g*1.0e-9*N_3 +...
    pi*r_coil^2*(2*(widthSideCoil_1 + lengthSideCoil_1))*g*1.0e-9*N_1 + ...
    pi*r_coil^2*(2*(widthSideCoil_2 + lengthSideCoil_2))*g*1.0e-9*N_2);% 20; %(Newton)
obj.F_C = [0 0 F_x]';
     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ParameterOptions = struct('W_A',W_A,'F_C',F_C,'Delta_W',Delta_W,'B_0',B_0,...
%     'NumSegments_total',NumSegments_total,'NumSegments',NumSegments,'Delta',Delta,...
%     'NumSegments_tip',NumSegments_tip,'I_cons',I_cons,'C',C,'MuMatrix',MuMatrix,'L_A',L_A,'C_lastSeg',C_lastSeg,...
%     'marker_length_base',marker_length_base,'marker_length_coil',marker_length_coil,'marker_length_tip',marker_length_tip,'E',E);
% 

end
