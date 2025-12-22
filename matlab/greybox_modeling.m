close all;
clear;

%% input output
path_to_output = 'data/experimental/output_trajectories/lemniscate01_01.txt';
path_to_input = 'data/experimental/input_currents/lemniscateCurrents.mat';
Ts = 0.001; %s
Ts_camera = 0.0167; %s
% read_input_output;
[currents, coil_position_mat, tip_position_mat, data_length] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
% 
% old data
% pL= [-0.531599; -1.71615; 68.9192]; %[-0.524853 ;-1.83144; 68.4151];% [-0.753845; -1.6483; 68.4184];
% RL =[0.999907; -0.000299857; -0.0136261;
% -0.000299857; 0.999032; -0.0439888;
% 0.0136261; 0.0439888 ;0.998939];
% 
% xf = [-0.876827; -2.89352; 97.8925;
%     0.999936; -0.000285689; -0.011342; -0.000201858; 0.999077; -0.0429615; 0.0113438 ;0.042961; 0.999012;
%      1.59055e-05; 9.41417e-05; 1.54048e-09;];


pL= [-3.39973; -10.3573; 65.1541]; %[-0.524853 ;-1.83144; 68.4151];% [-0.753845; -1.6483; 68.4184];
RL =[0.995847; -0.012664; -0.0901545;
-0.0126385; 0.961458; -0.274661;
0.0901581; 0.27466; 0.957305];

xf = [-3.99785 ; -17.94 ; 91.6699;0.993333; -0.0126627; 0.114586;  0.0435856 ;...
    0.961424; -0.271595 ;-0.106726   ;0.274778;  0.955566 ;-0.000254688;   8.2154e-05; -1.40022e-11;];
    


vL = [0;0;0];
wL = [0;0;0];
nL = [0;0;0];
mL = [0;0;0];


% % continuum parameter 50
% damping = [10;200;0.1;0.01]; 
% radius_ = [1.5875;0.9906]; 
% E_ = [5.5471;1.7747];% [5.9232;1.9114]; 
% Coil_align =[-0.1631;-3.0989] ;%[-0.0653;-3.0378];
% Coil_turnarea=[1.8037;2.7781;1.7728];%[1.4;1.9005;1.2914];% [1.44;1.3851;1.60];
% mass_ =[8.0293e-6] ;%[5.3516e-5];
% SegmentLengths = [19.85;  18.3; 59.40];

% load("param_4.mat");
% damping = nlgr_model.Params(1).Value;
% radius_ = [1.5875;0.9906];
% E_ = nlgr_model.Params(4).Value;
% Coil_align = nlgr_model.Params(5).Value; %[-0.0871, -0.3934]
% Coil_turnarea = nlgr_model.Params(6).Value;%[1.3851;1.44;1.60];% [1.44;1.3851;1.60];
% mass_ =nlgr_model.Params(7).Value;% [5.7736e-5]; 
% SegmentLengths =nlgr_model.Params(8).Value;% [5.7736e-5]; 

%%%%%%% for new data
damping = [100;200;0.1;0.01]; 
radius_ = [1.5875;0.9906]; 
E_ = [5.3948;2.3881];% [5.9232;1.9114]; 
Coil_align =[-1.1631;1.0989] ;%[-0.0653;-3.0378];
Coil_turnarea=[1.8037;2.7781;1.7728];%[1.4;1.9005;1.2914];% [1.44;1.3851;1.60];
mass_ =[6.0293e-6] ;%[5.3516e-5];
SegmentLengths = [18.5;  18.3; 57.50];


damping = [100;200;0.1;0.01]; 
radius_ = [1.5875;0.9906]; 
E_ = [5.3948;2.3881];% [5.9232;1.9114]; 
Coil_align =[3.1631;-3.0989] ;%[-0.0653;-3.0378];
Coil_turnarea=[3.8037;2.7781;1.7728];%[1.4;1.9005;1.2914];% [1.44;1.3851;1.60];
mass_ =[7.0293e-6] ;%[5.3516e-5];
SegmentLengths = [18.5;  18.3; 57.50];

% InitialStates = [vL;wL;mL;nL;pL;RL; xf];

load("model_3.mat");
damping = nlgr_model.Params(1).Value;
radius_ = [1.5875;0.9906];
E_ = nlgr_model.Params(4).Value;
Coil_align = nlgr_model.Params(5).Value; %[-0.0871, -0.3934]
Coil_turnarea = nlgr_model.Params(6).Value;%[1.3851;1.44;1.60];% [1.44;1.3851;1.60];
mass_ =nlgr_model.Params(7).Value;% [5.7736e-5]; 
SegmentLengths =nlgr_model.Params(8).Value;% [5.7736e-5]; 

% % %test with varying freq data

% % % % %%% see data_stich.m
% load('coil_position_new.mat');
% load('currents_new.mat');
% currents = currents_mat;
% load('init_new_data.mat');
% % %for greybox
% InitialStates = [vL,wL,mL,nL,pL,RL, xf]';

% for old data
% load('output_currents.mat');
% load('output_coil_traj.mat');
% load('output_tip_traj.mat');
% currents = output_currents;
% coil_position_mat = output_coil_traj;
% tip_position_mat = output_tip_traj;
% load('init_bk_new.mat');% at 621



test_length = data_length ;%1800;% data_length;
p_mat = [];
% num_p = []; %numerical verification
% R_pre = [RL(1:3); RL(4:6);RL(7:9)];
% p_pre = pL';


for i = 1:test_length

    u = [currents(:,i)];
    u(3) = -u(3);
%     u(2) = -u(2);

    u
%     p_dot = R_pre * vL';
%     p = p_pre + Ts * p_dot;
%     p_pre = p;

    [vL, wL, mL, nL, pL, RL, xf] = CRMDYN_c_mex(vL, wL, mL, nL, pL, RL, xf, u, damping, Ts, radius_, E_,...
        Coil_align, Coil_turnarea, mass_, SegmentLengths);
       
    p_mat = [p_mat,pL'];

    i

end


p_mat_01 = p_mat;
coil_position_mat_01 = coil_position_mat;
save('p_mat_01.mat', 'p_mat_01');
save('coil_position_mat_01.mat', 'coil_position_mat_01');

% save('init_new_data.mat','vL','wL','mL','nL','pL','RL', 'xf');

figure(2);

plot3(p_mat(1,:), p_mat(2,:), p_mat(3,:), 'r-');
hold on;
plot3(coil_position_mat(1,:), coil_position_mat(2,:), coil_position_mat(3,:), 'k*');

figure(1);

title('Positions');
ind_start = 1; %142; % 171;500;
subplot(3,1,1);
x = [1:test_length];
plot(x,p_mat(1, 1:test_length), 'k');
hold on;
x = [1:test_length] ;
plot(x,coil_position_mat(1,ind_start:ind_start + test_length-1), 'r');
subplot(3,1,2);
plot(x,p_mat(2, 1:test_length), 'k');
hold on;
plot(x,coil_position_mat(2,ind_start:ind_start + test_length-1), 'r');
subplot(3,1,3);
plot(x,p_mat(3, 1:test_length), 'k');
hold on;
plot(x,coil_position_mat(3,ind_start:ind_start + test_length-1), 'r');

% figure(2);
% x = 1:test_length;
% 
% title('Positions');
% ind_start = 1; %142; % 171;500;
% subplot(3,1,1);
% x = [1:test_length];
% plot(x,num_p(1, 1:test_length), 'k');
% hold on;
% x = [1:test_length] ;
% plot(x,coil_position_mat(1,ind_start:ind_start + test_length-1), 'r');
% subplot(3,1,2);
% plot(x,num_p(2, 1:test_length), 'k');
% hold on;
% plot(x,coil_position_mat(2,ind_start:ind_start + test_length-1), 'r');
% subplot(3,1,3);
% plot(x,num_p(3, 1:test_length), 'k');
% hold on;
% plot(x,coil_position_mat(3,ind_start:ind_start + test_length-1), 'r');


% pt_r = p_mat(:,1);
% pt_d = coil_position_mat(:,ind_start);
% for i = 2: test_length
%     pt_cur_r = p_mat(:,i);
%     pt_cur_d = coil_position_mat(:,i);
% 
%     pr_ = [pt_cur_r,pt_r];
%     pd_ = [pt_cur_d,pt_d];
%     plot3(pr_(1,:), pr_(2,:), pr_(3,:), 'k-');
%     hold on;
%     plot3(pd_(1,:), pd_(2,:), pd_(3,:), 'r-');
%     hold on;
%     pt_r = pt_cur_r;
%     pt_d = pt_cur_d;
% 
%     xlabel('x');
%     ylabel('y');
%     zlabel('z');
% 
% end

%% greybox

Order = [3, 3, 39]; %[Ny, Nu, Nx == dim(v, w, m, n, p, R, xf)]

Params = {damping; Ts;radius_;E_;Coil_align;Coil_turnarea;mass_; SegmentLengths};


nlgr = idnlgrey('CRMDYN_c', Order, Params, InitialStates, Ts);
nlgr.Params(1).Fixed = false;
nlgr.Params(2).Fixed = true;
nlgr.Params(3).Fixed = true;
nlgr.Params(4).Fixed = false;
nlgr.Params(5).Fixed = false;
nlgr.Params(6).Fixed = false;
nlgr.Params(7).Fixed = false;
nlgr.Params(8).Fixed = false;

opt = nlgreyestOptions;
opt.Display = 'on';
opt.SearchOptions.MaxIterations = 20;
% opt.Advanced.ErrorThreshold= 10;
% opt.GradientOptions.Type = 'Basic';
opt.SearchOptions.FunctionTolerance = 0.00001;
opt.SearchMethod = 'lsqnonlin';
nlgr.TimeUnit = 's';
opt.GradientOptions.DifferencingScheme =   'Backward approximation'; 
% nlgr.SimulationOptions.Solver = 'ode45';
ind_y = 1; 

raw_y = [coil_position_mat(:,ind_y:ind_y + test_length-1)];%, normal_mat(:,ind_start:ind_start + test_length-1)'];
dy_init = coil_position_mat(:,ind_y) - p_mat(:,1);
raw_y = raw_y - repmat(dy_init, [1 test_length]);
y = raw_y';

figure(3);
x = 1:test_length;

title('Positions');
subplot(3,1,1);
plot(x,p_mat(1, 1:test_length), 'k');
hold on;
plot(x,raw_y(1,1:1 + test_length-1), 'r');
subplot(3,1,2);
plot(x,p_mat(2, 1:test_length), 'k');
hold on;
plot(x,raw_y(2,1:1 + test_length-1), 'r');
subplot(3,1,3);
plot(x,p_mat(3, 1:test_length), 'k');
hold on;
plot(x,raw_y(3,1:1 + test_length-1), 'r');

% figure(4);
% pt_r = p_mat(:,1);
% pt_d = raw_y(:,ind_y);
% for i = 2: test_length
%     pt_cur_r = p_mat(:,i);
%     pt_cur_d = raw_y(:,i+ind_y);
% 
%     pr_ = [pt_cur_r,pt_r];
%     pd_ = [pt_cur_d,pt_d];
%     plot3(pr_(1,:), pr_(2,:), pr_(3,:), 'k-');
%     hold on;
%     plot3(pd_(1,:), pd_(2,:), pd_(3,:), 'r-');
% hold on;
%     pt_r = pt_cur_r;
%     pt_d = pt_cur_d;
%     pause(0.01);
% 
%     xlabel('x');
%     ylabel('y');
%     zlabel('z');
% 
% end

ind_u = 1; % 622;
u_c= currents(:,ind_u:ind_u+test_length-1)';
u_c(:,3) = -u_c(:,3);
% u_c(:,2) = -u_c(:,2);

% u = [u_c, ones(test_length, 1) * insertedLength];
u = u_c;
z = iddata(y, u, Ts);

nlgr_model = nlgreyest(z, nlgr, opt);

% save('optimized_parameter.mat',"nlgr_model");
%% testing

set(0, 'defaultfigurecolor','w');
figure(7);
compare(z, nlgr_model, compareOptions('InitialCondition', 'e') );