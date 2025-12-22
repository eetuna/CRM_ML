%%%%%%%%%%%%%%%%%%%%%
%%% This code loads the result data from a txt file procesed by camera tracking system
%%% The crucial element is to separate the individual deflection
%%% corresponding to each current value
%%% Right now the separation is based on the index of each tracking frame.
%%%
%%% Taoming Liu
%%% 03/31/2016
%%%%%%%%%%%%%%%%%%%%%%
%%% This file is based on results obtained from experiment conducted on
%%% 05/19/2016.
%%%%%%%%%%%%%%%%%%%%%%
if ~exist('filePath', 'var')
    error('Variable filePath is not defined. Please specify the input file path.');
end

fileID = fopen(filePath);
C = textscan(fileID,'%f %n %c %f %f %f %c %c %f %f %f %c %c %f %f %f %c %c %f %f %f %c', 'Delimiter', ',','HeaderLines',2);
fclose(fileID);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Separate the deflections based on the binary number on the second
%%% column
time_stamp = C{1};
tick_index = C{2};
num = 0;
change_tick_set = [];

for k = 1:1:length(tick_index)-1
    
    % if the value on the second column changes, it means that the ardruino
    % board receives a new current value. Find out the tick_index when the
    % value changes
    if tick_index(k) ~= tick_index(k+1)
        num = num + 1;
        
        change_tick_set = [change_tick_set k];
        
    end
    
end
change_tick_set = [change_tick_set length(tick_index)];
%%%% the reason of change_tick_set equalling to 205 is that the ardruino
%%%% board will add two additional index in the beginning and one index
%%%% change at the end.

%%%% starting from change_tick_set(2)

% change_tick_set_POI = change_tick_set(2):1:change_tick_set(length(change_tisck_set)-1);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% change the distance unit from meter to mm
basePos = 1000*[C{4} C{5} C{6}];
coilPos = 1000*[C{9} C{10} C{11}];
% coil_normal = [C{14} C{15} C{16}];
tipPos = 1000*[C{19} C{20} C{21}];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% basePos_cam = 1000*[C{4} C{5} C{6}];
% coilPos_cam = 1000*[C{9} C{10} C{11}];
% % coil_normal = [C{14} C{15} C{16}];
% tipPos_cam = 1000*[C{19} C{20} C{21}];
% 
% 
% basePos  = zeros(length(basePos_cam),3);
% coilPos  = zeros(length(basePos_cam),3);
% tipPos  = zeros(length(basePos_cam),3);
% 
% for ip = 1:length(basePos_cam)
%     
%     basePos(ip,:) = (RotMat_cam_mri'*basePos_cam(ip,:)')';
%     coilPos(ip,:) = (RotMat_cam_mri'*coilPos_cam(ip,:)')';
%     tipPos(ip,:) = (RotMat_cam_mri'*tipPos_cam(ip,:)')';
%     
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set up a new coordinate frame which is originated on the base position of
% the catheter prototype. This new origin is the average of all base
% positions at all of the tracking frames.
% basePos_avg = [mean(basePos((change_tick_set(2)+1):(change_tick_set(3)),1)) mean(basePos((change_tick_set(2)+1):(change_tick_set(3)),2)) mean(basePos((change_tick_set(2)+1):(change_tick_set(3)),3))];
basePos_avg = [mean(basePos((change_tick_set(1)+1):(change_tick_set(length(change_tick_set))),1)) mean(basePos((change_tick_set(1)+1):(change_tick_set(length(change_tick_set))),2)) mean(basePos((change_tick_set(1)+1):(change_tick_set(length(change_tick_set))),3))];

numSteps = length(basePos);

% tanslate the coordinate frame originated at the camera origin to the base
% location of the catheter
% As the center of the tracking circle is on the center of the marker, the
% real location of the base is above this center.
% (ParameterOptions.marker_length_base)
% g_cathBase_cam = [eye(3) [0 0 ParameterOptions.marker_length_base/2]'; 0 0 0 1]*[eye(3)' -eye(3)'*basePos_avg'; 0 0 0 1 ];
g_cathBase_cam = [eye(3) [0 0 0]'; 0 0 0 1]*[eye(3)' -eye(3)'*basePos_avg'; 0 0 0 1 ];
g_cathBase_cam2 = [eye(3) -[basePos_avg]';[0 0 0 1]]

basePos_cathBase_group = zeros(numSteps,3);
coilPos_cathBase_group = zeros(numSteps,3);
tipPos_cathBase_group = zeros(numSteps,3);

for i = 1:1:numSteps
    basePos_cathBase_temp = g_cathBase_cam*[basePos(i,:)'; 1];
    basePos_cathBase_group(i,:) = basePos_cathBase_temp(1:3)';
    
    coilPos_cathBase_temp = g_cathBase_cam*[coilPos(i,:)'; 1];
    coilPos_cathBase_group(i,:) = coilPos_cathBase_temp(1:3)';
    
    tipPos_cathBase_temp = g_cathBase_cam*[tipPos(i,:)'; 1];
    tipPos_cathBase_group(i,:) = tipPos_cathBase_temp(1:3)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% for this specific experiment data collected on 2016/03/21, we need to delete the first two steps and the last step.
%%% then the first step of the index starts from 426 and ends at 461.
%%% Average the base position, coil position, tip position at the middle range to obtain the
%%% representation of these position at this step
total_steps = length(change_tick_set)-3;
% total_steps = length(change_tick_set)-5; %for rectangle trajectory on 2016-06-02

basePos_cathBase_avg_set_traj = zeros(total_steps,3);
coilPos_cathBase_avg_set_traj = zeros(total_steps,3);
tipPos_cathBase_avg_set_traj = zeros(total_steps,3);

for m = 2:1:length(change_tick_set)-2
% for m = 2:1:length(change_tick_set)-4%for rectangle trajectory on 2016-06-02
    
    % For each step, only select half amount of deflections as the average
    % of the deflection.
    % _middle_pts_ is the middle point in this range
    % _usable_pts_range_ is the range from this middle point which is
    % selected to average the positions
    m
    m+1
    change_tick_set(m)
    change_tick_set(m+1)
    middle_pts = ceil(0.5*(change_tick_set(m) + 1 + change_tick_set(m+1)));
    %     usable_pts_range = floor(0.25*(change_tick_set(m+1) - (change_tick_set(m) + 1) + 1));
    
    % define the range
    %     index_starting = middle_pts + usable_pts_range;
    index_starting = middle_pts;
    %     index_starting = change_tick_set(m)+1;
    index_end = change_tick_set(m+1);
    
    
    basePosX_cathBase_avg = mean(basePos_cathBase_group(index_starting:index_end,1));
    basePosY_cathBase_avg = mean(basePos_cathBase_group(index_starting:index_end,2));
    basePosZ_cathBase_avg = mean(basePos_cathBase_group(index_starting:index_end,3));
    basePos_cathBase_avg_set_traj(m-1,:) = [basePosX_cathBase_avg basePosY_cathBase_avg basePosZ_cathBase_avg]';

    coilPosX_cathBase_avg = mean(coilPos_cathBase_group(index_starting:index_end,1));
    coilPosY_cathBase_avg = mean(coilPos_cathBase_group(index_starting:index_end,2));
    coilPosZ_cathBase_avg = mean(coilPos_cathBase_group(index_starting:index_end,3));
    coilPos_cathBase_avg_set_traj(m-1,:) = [coilPosX_cathBase_avg coilPosY_cathBase_avg coilPosZ_cathBase_avg]';

    tipPosX_cathBase_avg = mean(tipPos_cathBase_group(index_starting:index_end,1));
    tipPosY_cathBase_avg = mean(tipPos_cathBase_group(index_starting:index_end,2));
    tipPosZ_cathBase_avg = mean(tipPos_cathBase_group(index_starting:index_end,3));
    tipPos_cathBase_avg_set_traj(m-1,:) = [tipPosX_cathBase_avg tipPosY_cathBase_avg tipPosZ_cathBase_avg]';

    
end
