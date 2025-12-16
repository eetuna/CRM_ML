close all
clear all;

load('../catheterdata/Catheter_TipPosition_Circle_r12.mat');
disp('### CRM Forward Kinematics Calculate and Render Examples... ');
disp('Free Space Deflection Example: ');

% *** Load parameters from file
% This step would typically needs to be executed only once
%  Physical Description of the Catheter
FKParams.CathParams=Load_CRMCatheterModelParams_matlab('../catheterdata/ParameterEst_2.txt')
%  Catheter Configuration in spatial coordinates
FKParams.CathConfig=Load_CatheterConfiguration_matlab('../catheterdata/CatheterSpatialConfiguration_1.txt')

% *** Other External variables
% specify if catheter is in free space or if the catheter tip is constrained to a contact point
FKParams.ContactMode = int8(0);  % 0: FREE_TIP;   1: FIXED_TIP
% External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0)  - unit: ??
%   (this will be used when ContactMode == 0, FREE_TIP)
FKParams.TipForce = [ 0.0, 0.0, 0.0 ];
% The spatial coordinates of the point where the catheter tip is constrained to be 
%   (this will be used when ContactMode == ContactModeType::FIXED_TIP)
FKParams.TipConstraintPoint = [ 0.0, 0.0, 0.0 ];
% We want the localization coil locations, too
FKParams.FinalValueOnly = false; 

% *** Numerical Computation Parameters
% Stepsize used in numerical integration along the length of the catheter during IVP - unit: mm
FKParams.IntegrationStepSize = 0.2;
% Define initial guesses to be used when solving boundary value problem
% initial guess for the curvature at the catheter base
FKParams.deltau0_initialguess = [ 0.0, 0.0, 0.0 ];
% initial guess for the contstraint force at the catheter tip (this will be used when ContactMode == ContactModeType::FIXED_TIP)
FKParams.ftip_initialguess = [ 0.0, 0.0, 0.0 ];

% *** Control Inputs
% Inserted Length of the catheter (length of the catheter that is inside the heart chamber) - unit: mm
InsertedLength = 148.0;
% InsertedLength = 100.0;
% Actuation currents for each of the coils for each of the coil sets - unit: A
ActuationCurrents = [ 0.4, 0.0, 0.0 ];  % [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00];   
% pack them together into a single array
control_inputs = [ ActuationCurrents InsertedLength ];


%
% Cosserat Rod Model - Solve the Forward Kinematics 
%
[FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);
% FKsolution: p[0..2],R[0..8],deltau0[0..2](,ftip[0..2])  R: in row major order
% ReportedMarkerPos .. by 3 array of coordinates of localization marker locations 
% localmin: numerical nonlinear equation solver diagnostic output

% Cosserat Rod Model - calculate the Jacobian using the solution of the Forward Kinematics
Ja = CRM_FKJacobian_Analytical_matlab(control_inputs, FKsolution, FKParams); 

% Render the Catheter (this function also performs the forward kinematics for rendering)
DrawCRMCatheterModel(FKParams, control_inputs);
%%
hold on;
scatter3(CatheterTip_Trajectory(1,:),CatheterTip_Trajectory(2,:),CatheterTip_Trajectory(3,:))
hold off
%% % Calculate and save current trajectory.
numPaddings = 20;
threshold = 2;
stepSize = 1e-2;
%%

initialControl = [0.4, 0, 0];
control_inputs = [ ActuationCurrents InsertedLength ];
[FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);
initialTipPosition = FKsolution(1:3,1);
%% 
% Go to the starting point.
counter = 0;
itrmax = 100;
tipPositionIncrement = CatheterTip_Trajectory(:, 1) - initialTipPosition;
% tipPositionIncrement = [tipPositionIncrement;0];
disp('Calculating initial control...');
control_inputs = [initialControl InsertedLength ];
while true
    counter = counter + 1;
    initialStateOld = FKsolution;
    initialControlOld = control_inputs;
    
    % Calculate new control.
    % Cosserat Rod Model - calculate the Jacobian using the solution of the Forward Kinematics
     Ja = CRM_FKJacobian_Analytical_matlab(control_inputs, FKsolution, FKParams); 
    Ja_z_p = Ja(1:3,1:4);
    Jinv = pinv(Ja_z_p);
    controlIncrement = Jinv * tipPositionIncrement;
    initialControl = initialControlOld + ...
        transpose(stepSize * controlIncrement / norm(controlIncrement));
    control_inputs = initialControl;
    % Calculate new configuration.
   [FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);

    
    % Recalculate tip position increment.
    initialTipPosition = FKsolution(1:3,1);
    tipPositionIncrement = CatheterTip_Trajectory(:, 1) - initialTipPosition;

    if norm(tipPositionIncrement) < threshold
        disp('Found closest point');
        disp('Tip position error = ');
        disp(norm(tipPositionIncrement));
        break
    elseif counter > itrmax
        disp('Iterations exeed limit');
        disp('Tip position Error = ');
        disp(norm(tipPositionIncrement));
        break
    end

    disp('[iter, distance] = ');
    disp([counter, norm(tipPositionIncrement)]);

end
    %%
numSteps = size(CatheterTip_Trajectory,2);
% Calculate control trajectory.
states = zeros(size(FKsolution,1), numSteps);
controls = zeros(4, numSteps);
states(:, 1) = FKsolution;
controls(:, 1) = control_inputs';


disp('Calculating control trajectory...');
threshold = 1;
for i = 1 : 1 : numSteps - 1
    if (i==70)
        pause(0.01);
    end
    % Calculate tip position direction
    tipPositionIncrement = CatheterTip_Trajectory(:, i + 1) - ...
        states(1:3, i);
    control_inputs = controls(:, i)';
    counter = 0;

    while true
        FKsolution =  states(:, i);
    counter = counter + 1;
    initialStateOld = states(:, i);
    initialControlOld = control_inputs;
    
    % Calculate new control.
    % Cosserat Rod Model - calculate the Jacobian using the solution of the Forward Kinematics
     Ja = CRM_FKJacobian_Analytical_matlab(control_inputs, FKsolution, FKParams); 
    Ja_z_p = Ja(1:3,1:4);
    Jinv = pinv(Ja_z_p);
    controlIncrement = Jinv * tipPositionIncrement;
    initialControl = initialControlOld + ...
        transpose(stepSize * controlIncrement / norm(controlIncrement));
    control_inputs = initialControl;
    % Calculate new configuration.
   [FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);

    
    % Recalculate tip position increment.
    initialTipPosition = FKsolution(1:3,1);
    tipPositionIncrement = CatheterTip_Trajectory(:, i + 1) - initialTipPosition;

    if norm(tipPositionIncrement) < threshold
        disp('Found closest point');
        disp('Tip position error = ');
        disp(norm(tipPositionIncrement));
        break
    elseif counter > itrmax
        disp('Iterations exeed limit');
        disp('Tip position Error = ');
        disp(norm(tipPositionIncrement));
        break
    end

    disp('[iter, distance] = ');
    disp([counter, norm(tipPositionIncrement)]);

end

    controls(:, i + 1) = initialControl';
    control_inputs = controls(:, i + 1)'  ;

    [FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);
    states(:, i+1) = FKsolution;


%     % Calculate control
%     Ja = CRM_FKJacobian_Analytical_matlab(control_inputs, states(:, i), FKParams); 
%     Ja_z_p = Ja(1:3,1:3);
%     Jinv = pinv(Ja_z_p);
%     controlIncrement = Jinv * tipPositionIncrement;
%     controls(:, i + 1) = controls(:, i) + controlIncrement;
%     control_inputs = [controls(:, i + 1)' InsertedLength ];
% 
%     [FKsolution, PotentialEnergy, ReportedMarkerPos, ReportedCoilOrient, localmin] = CRM_ForwardKinematics_matlab(control_inputs, FKParams);
%     states(:, i+1) = FKsolution;


end
    test_controls = controls(:,1:70);
    for itr = 1:70
        control_inputs = [test_controls(:,itr)' InsertedLength]';
        DrawCRMCatheterModel(FKParams, control_inputs);
    end