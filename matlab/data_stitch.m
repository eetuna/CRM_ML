close all;
clear;


%% input output
Ts_camera = 0.0167; %s



path_to_output = 'data/experimental/output_trajectories/circle100_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.1; %s
[currents_100_raw, coil_position_mat_100_raw, tip_position_mat_100_raw, data_length_100] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);

coil_position_mat_100 = coil_position_mat_100_raw(:,86:485);
tip_position_mat_100 = tip_position_mat_100_raw(:,86:485);
% currents_100 = currents_100_raw(:, 102:501);
time_100 = [0:1:size(coil_position_mat_100,2)-1]*Ts;
time_ = time_100;

path_to_output = 'data/experimental/output_trajectories/circle50_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.05; %s
[currents_50_raw, coil_position_mat_50_raw, tip_position_mat_50_raw, data_length_50] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);

coil_position_mat_50 = coil_position_mat_50_raw(:,154:553);
tip_position_mat_50 = tip_position_mat_50_raw(:,154:553);

% currents_50 = currents_50_raw(:, 169:568);
t_50 = time_(end) + [1:1:size(coil_position_mat_50,2)]*Ts;
time_ = [time_, t_50];

path_to_output = 'data/experimental/output_trajectories/circle25_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.025; %s
[currents_25, coil_position_mat_25_raw, tip_position_mat_25_raw, data_length_25] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
coil_position_mat_25 = coil_position_mat_25_raw(:,303:702);
tip_position_mat_25 = tip_position_mat_25_raw(:,303:702);

% currents_25 = currents_25(:, 320:720);
t_25 = time_(end) + [1:1:size(coil_position_mat_25,2)]*Ts;
time_ = [time_, t_25];
% 
path_to_output = 'data/experimental/output_trajectories/circle20_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.02; %s
[currents_20, coil_position_mat_20_raw, tip_position_mat_20_raw, data_length_20] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
coil_position_mat_20 = coil_position_mat_20_raw(:,416:815);
tip_position_mat_20 = tip_position_mat_20_raw(:,416:815);
% currents_20 = currents_20(:,421:821);
t_20 = time_(end) + [1:1:size(coil_position_mat_20,2)]*Ts;
time_ = [time_, t_20];


path_to_output = 'data/experimental/output_trajectories/circle18_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.018; %s
[currents_18, coil_position_mat_18_raw, tip_position_mat_18_raw, data_length_18] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
coil_position_mat_18 = coil_position_mat_18_raw(:,442:841);
tip_position_mat_18 = tip_position_mat_18_raw(:,442:841);

% currents_18 = currents_18(:,458:858);
t_18 = time_(end) + [1:1:size(coil_position_mat_18,2)]*Ts;
time_ = [time_, t_18];
% 
path_to_output = 'data/experimental/output_trajectories/circle15_01.txt';
path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
Ts = 0.015; %s
[currents_15, coil_position_mat_15_raw, tip_position_mat_15_raw, data_length_15] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
coil_position_mat_15 = coil_position_mat_15_raw(:,519:918);
tip_position_mat_15 = tip_position_mat_15_raw(:,519:918);
% currents_15 = currents_15(:,532:932);
t_15 = time_(end) + [1:1:size(coil_position_mat_15,2)]*Ts;
time_ = [time_, t_15];
% % 
% path_to_output = 'data/experimental/output_trajectories/circle12_01.txt';
% path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
% Ts = 0.012; %s
% [currents_12, coil_position_mat_12_raw, data_length_12] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
% coil_position_mat_12 = coil_position_mat_12_raw(:,740:940);
% % currents_12 = currents_12(:,740:940);
% t_12 = time_(end) + [1:1:size(coil_position_mat_12,2)]*Ts;
% time_ = [time_, t_12];

% 
% path_to_output = 'data/experimental/output_trajectories/circle10_01.txt';
% path_to_input = 'data/experimental/input_currents/circleCurrents.mat';
% Ts = 0.01; %s
% [currents_10, coil_position_mat_10, data_length_10] = read_input_output(Ts, Ts_camera, path_to_input, path_to_output);
% coil_position_mat_10 = coil_position_mat_10(:,817:1017);
% % currents_10 = currents_10(:,817:1017);
% t_10 = time_(end) + [1:1:size(coil_position_mat_10,2)]*Ts;
% time_ = [time_, t_10];
% 
% 

% coil_position_collect = [ coil_position_mat_100, coil_position_mat_50, coil_position_mat_25, coil_position_mat_20, coil_position_mat_18, coil_position_mat_15, coil_position_mat_12, coil_position_mat_10]; 
% currents_collect = [ currents_100, currents_50, currents_25, currents_20, currents_18,currents_15, currents_12, currents_10];

coil_position_collect = [ coil_position_mat_100, coil_position_mat_50, coil_position_mat_25,  ...
    coil_position_mat_20,coil_position_mat_18, coil_position_mat_15]; 

tip_position_collect = [ tip_position_mat_100, tip_position_mat_50, tip_position_mat_25,  ...
    tip_position_mat_20,tip_position_mat_18, tip_position_mat_15]; 

load('data/experimental/input_currents/circleCurrents.mat');
currents_collect = repmat(expectedCurrents_traj, [1 12]) ;



Tq = [0:0.05:time_(end)];

coil_position_mat = zeros(3, length(Tq));
tip_position_mat = zeros(3, length(Tq));

currents_mat = zeros(3, length(Tq));

for i =1: 3
	coil_position_mat(i,:) = interp1(time_,coil_position_collect(i,:),Tq) ;
    tip_position_mat(i,:) = interp1(time_,tip_position_collect(i,:),Tq) ;
	currents_mat(i,:) = interp1(time_,currents_collect(i,:),Tq) ;
end



figure(1);

title('Positions');
x = [1:size(tip_position_mat, 2)];
subplot(3,1,1);
plot(x,tip_position_mat(1,:), 'r');
subplot(3,1,2);
plot(x,tip_position_mat(2,:), 'r');
subplot(3,1,3);
plot(x,tip_position_mat(3, :), 'r');
hold on;

figure(3);

title('Positions');
x = [1:size(coil_position_mat_100_raw, 2)];
subplot(3,1,1);
plot(x,coil_position_mat_100_raw(1,:), 'r');
subplot(3,1,2);
plot(x,coil_position_mat_100_raw(2,:), 'r');
subplot(3,1,3);
plot(x,coil_position_mat_100_raw(3, :), 'r');
hold on;

figure(4);

title('Positions');
x = [1:size(coil_position_mat_50_raw, 2)];
subplot(3,1,1);
plot(x,coil_position_mat_50_raw(1,:), 'r');
subplot(3,1,2);
plot(x,coil_position_mat_50_raw(2,:), 'r');
subplot(3,1,3);
plot(x,coil_position_mat_50_raw(3, :), 'r');
hold on;

figure(5);

title('Positions');
x = [1:size(coil_position_mat_18_raw, 2)];
subplot(3,1,1);
plot(x,coil_position_mat_18_raw(1,:), 'r');
subplot(3,1,2);
plot(x,coil_position_mat_18_raw(2,:), 'r');
subplot(3,1,3);
plot(x,coil_position_mat_18_raw(3, :), 'r');
hold on;

figure(6);

title('Positions');
x = [1:size(coil_position_mat_15_raw, 2)];
subplot(3,1,1);
plot(x,coil_position_mat_15_raw(1,:), 'r');
subplot(3,1,2);
plot(x,coil_position_mat_15_raw(2,:), 'r');
subplot(3,1,3);
plot(x,coil_position_mat_15_raw(3, :), 'r');
hold on;

figure(2);

plot3(tip_position_mat(1,:), tip_position_mat(2,:), tip_position_mat(3,:), 'k*');
% 
save('data/output/tip_position_new.mat','tip_position_mat');
save('data/output/currents_new.mat','currents_mat');