%change the mat table to txt
% new_action_dis2 = load('/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_action_discrepancy_[1,2]x[-2,0]_1LDC.mat');
% new_action_dis2 = new_action_dis2.quantile_all(1:10, 1:20);
% new_action_dis2 = round(new_action_dis2, 4);
% new_action_dis1 = load('/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_action_discrepancy_[0,1]x[-2,0]_1LDC.mat');
% new_action_dis1 = new_action_dis1.quantile_all(1:10, 1:20);
% new_action_dis1 = round(new_action_dis1, 4);
% new_action_dis_table = [new_action_dis1;new_action_dis2];
% writematrix(new_action_dis_table);

clear all
clear global input3
% % Now train multiple controllers
% sample_size = 300;
% state_region_1 = [0, 0.5; -2,0];
% state_region_2 = [0.5, 1; -2,0];
% state_region_3 = [1, 1.5; -2,0];
% state_region_4 = [1.5, 2; -2,0];
% path_HDC = '/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat';
% 
% loaded_HDC = Load_HDC('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat');
% 
% path_save_LDC1 = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC1_theta0-0.5_dot-2-0.txt';
% path_save_LDC2 = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC2_theta0.5-1_dot-2-0.txt';
% path_save_LDC3 = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC3_theta1-1.5_dot-2-0.txt';
% path_save_LDC4 = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC4_theta1.5-2_dot-2-0.txt';
% 
% training_data_1 = gather_training_date(loaded_HDC, state_region_1, sample_size);
% clear global input3
% training_data_2 = gather_training_date(loaded_HDC, state_region_2, sample_size);
% clear global input3
% training_data_3 = gather_training_date(loaded_HDC, state_region_3, sample_size);
% clear global input3
% training_data_4 = gather_training_date(loaded_HDC, state_region_4, sample_size);
% 
% weights_bias_LD1 = Train_LDC(training_data_1, path_save_LDC1);
% weights_bias_LD2 = Train_LDC(training_data_2, path_save_LDC2);
% weights_bias_LD3 = Train_LDC(training_data_3, path_save_LDC3);
% weights_bias_LD4 = Train_LDC(training_data_4, path_save_LDC4);
% 
% split_parameter = 0.1;
% conformal_table_1 = action_based_CP(path_save_LDC1, path_HDC, state_region_1, split_parameter);
% conformal_table_2 = action_based_CP(path_save_LDC2, path_HDC, state_region_2, split_parameter);
% conformal_table_3 = action_based_CP(path_save_LDC3, path_HDC, state_region_3, split_parameter);
% conformal_table_4 = action_based_CP(path_save_LDC4, path_HDC, state_region_4, split_parameter);
% 
% conformal_table_1 = conformal_table_1(1:5, 1:20);
% conformal_table_2 = conformal_table_2(1:5, 1:20);
% conformal_table_3 = conformal_table_3(1:5, 1:20);
% conformal_table_4 = conformal_table_4(1:5, 1:20);
% 
% all_CP_table = [conformal_table_1;conformal_table_2;conformal_table_3;conformal_table_4];
% writematrix(all_CP_table);
% 

%this is for the one LDC
sample_size = 500;
state_region = [0, 2; -2, 0]; %total region will be [0,2]x[-2, 0]
loaded_HDC = Load_HDC('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat');
% path_save_LDC = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/whole_LDC_theta0-2_dot-2-0.txt';
% training_data_500 = gather_training_date(loaded_HDC, state_region, sample_size);
% weights_bias_LD = Train_LDC(training_data_500, path_save_LDC);
% state_region = [0,2;-2,0];
split_parameter = 0.5;

path_LDC = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC_theta0-2_dot-2-0.txt';
path_HDC = '/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat';
loaded_HDC = load(path_HDC);
loaded_HDC = loaded_HDC.net;
[conformal_table_1LDC_traj,conformal_table_1LDC_act]  = action_based_CP(path_LDC, path_HDC, state_region, split_parameter);

function [conformal_traj, conformal_table] = action_based_CP(path_LDC, path_HDC, state_region, split_parameter)

global ini_theta_high
global ini_thetadot_high
global input3
loaded_HDC = load(path_HDC);
loaded_HDC = loaded_HDC.net;
% keep same to the Pure confromal prediction
theta_interv = state_region(1,1) : split_parameter : state_region(1,2);
dot_interv = state_region(2,1) : split_parameter : state_region(2,2);
index_theta = numel(theta_interv);
index_dot = numel(dot_interv);
quantile_all = zeros(index_theta, index_dot);

numPoints = 60;
ctrl_step  = 30;

env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");

for g =1:index_theta-1
intervalStart = theta_interv(g);
intervalEnd = theta_interv(g+1);
sample_theta = intervalStart + (intervalEnd - intervalStart) * rand(numPoints, 1);

%Simulation for both LDC and HDC.
 for k = 1:index_dot-1
     DotStart = dot_interv(k);
     DotEnd = dot_interv(k+1);
     sample_dot = DotStart + (DotEnd - DotStart) * rand(numPoints, 1);
     max_diff_array = zeros(1, numPoints);
     max_action_array = zeros(1, numPoints);
     for l = 1:numPoints
         init_theta = sample_theta(l);
         init_dot = sample_dot(l);

         ini_theta_high = init_theta;
         ini_thetadot_high = init_dot;

    % Simulation for high_dimensional controller
        HDC_actions = [];
        counter = 0;
        observation_test = reset(env);
        done = false;
        while counter < ctrl_step
    %   dlObservation = dlarray(observation, 'SSCB');
        new_input_images = observation_test{1,1};
        new_input_scalars = observation_test{1,2};
        dsX1New = arrayDatastore(new_input_images, 'IterationDimension', 4);
        dsX2New = arrayDatastore(new_input_scalars');
        dsNewData = combine(dsX1New, dsX2New);
        action = predict(loaded_HDC, dsNewData); 
        HDC_actions = [HDC_actions,action];
        [observation_test, reward, done, info] = step(env, action);
        counter = counter + 1;
        end
        theta_high = input3(end - ctrl_step+1:end);
        % change from -pi to pi to 0-2pi
%         for m = 1:ctrl_step
%             if theta_high(1,m) < 0
%                 theta_high(1,m) = 2*pi + theta_high(1,m);
%             end
%         end


    % simulation for low_dimensional controller
    LDC_actions = [];
    states_low = zeros(2,ctrl_step+1);
    states_low(1,1) = init_theta;
    states_low(2,1) = init_dot;
    ts = 0.05;
    est_torque_overall = zeros(ctrl_step,1);

other_model = load(path_LDC);
num_input = 2;
num_ouput = 1;
neurons_1st = 25;
neurons_2nd = 25;
weights1_other = zeros(neurons_1st, num_input);
weights2_other = zeros(neurons_2nd, neurons_1st);
weights3_other = zeros(num_ouput,neurons_2nd);
bias1_other = zeros(neurons_1st, 1);
bias2_other = zeros(neurons_2nd, 1);
bias3_other = zeros(num_ouput,1);
z = 1;
for i = 1:neurons_1st %25 is the neuron in the first layer
    for j = 1:num_input %2 is the size of input
        weights1_other(i, j) = other_model(z);
        z = z + 1;
    end
    bias1_other(i) = other_model(z);
    z = z + 1;
end
%Input the second layer
for i = 1:neurons_2nd %25 is the neuron in the second layer. 25 is the neuron in the second layer
    for j = 1:neurons_1st %2 is the size of neurons in previous layer. 20 is the neuron in the previous layer
        weights2_other(i, j) = other_model(z);
        z = z + 1;
    end
    bias2_other(i) = other_model(z);
    z = z + 1;
end
%input the third layer
for i = 1:num_ouput % the neuron in the third layer 
    for j = 1:neurons_2nd  %the size of neuron in the previous layer
        weights3_other(i, j) = other_model(z);
        z = z+1;
    end
    bias3_other(i) = other_model(z);
    z=z+1;
end 
    for L = 1:ctrl_step
    x = [(states_low(1,L)); states_low(2,L)];
    %est_torque_low = sim(net, x)*2;
    states_1st = weights1_other * x + bias1_other;
    %states_1st2 = 1 ./ (1 + exp(-states_1st)); %sigmoid
    states_1st2 = max(0, states_1st); %relu

    states_2nd = weights2_other * states_1st2 + bias2_other;
    %states_2nd2 = 1 ./ (1 + exp(-states_2nd)); %Sigmoid
    states_2nd2 = max(0, states_2nd);

    states_3rd = weights3_other * states_2nd2 + bias3_other; %linear
%   states_3rd2 = tanh(states_3rd);
    states_3rd2 = states_3rd;

    %est_torque_low = states_3rd2 * 2;
    est_torque_low = states_3rd2;

    torque_low = est_torque_low;
    LDC_actions = [LDC_actions, est_torque_low];
    dx1 = dynamic1(x, est_torque_low);
    dx2 = dynamic1(x + ts * dx1, est_torque_low);
    states_low(:,L+1) = x + ts * 0.5 * (dx1+dx2);

    %LDC needs all the theta to be within [0, 2*pi]?
%     if states_low(1,L+1) < 0
%         states_low(1,L+1) = 2*pi - states_low(1,L+1);
%     end

    end

    

    every_max_diff_array = abs(theta_high(1,1:ctrl_step) - states_low(1, 2:ctrl_step+1));
    max_diff = max(every_max_diff_array);
    max_diff_array(1,l) = max_diff;
    
    eve_max_diff_array_action = abs(HDC_actions(1,1:ctrl_step)- LDC_actions(1,1:ctrl_step));
    max_action = max(eve_max_diff_array_action);
    max_action_array(1,l) = max_action;
     end
 %trajectory-based approach
%      CI95 = quantile(max_diff_array, 0.96);
%      quantile_all(g, k) = CI95;
%      disp('finished:');
%      fprintf('Finished: (%d, %d)\n', g, k);

     CI95 = quantile(max_action_array, 0.96);
     quantile_all(g, k) = CI95;

     CI95_traj = quantile(max_diff_array, 0.96);
     quantile_all_traj(g, k) = CI95_traj;

     disp('finished:');
     fprintf('Finished: (%d, %d)\n', g, k);
     conformal_table = quantile_all;
     conformal_traj = quantile_all_traj;


 end
 
 end


 %% Draw a table to make it clear
 rowNames = [];
 columnNames = [];
for y = 1:index_theta-1
    lower_bound = theta_interv(y);
    upper_bound = theta_interv(y+1);
    rowNames{y} = sprintf('(%g,%g)', lower_bound, upper_bound);
end

for y = 1:index_dot-1
    lower_bound_dot = dot_interv(y);
    upper_bound_dot = dot_interv(y+1);
    columnNames{y} = sprintf('(%g,%g)', lower_bound_dot, upper_bound_dot);
end

fig = uifigure;
uit = uitable(fig,"Data",quantile_all(1:index_theta-1, 1:index_dot-1),'ColumnName',columnNames, 'RowName',rowNames);

end

function training_data_HDC = gather_training_date(loaded_HDC, state_region, sample_size)
numP_area = sample_size;
intervalStart = state_region(1,1);% intervalStart = 0; 
intervalEnd = state_region(1,2);% intervalEnd = 2*pi; 
random_theta = intervalStart + (intervalEnd - intervalStart) * rand(numP_area, 1);
number_theta = numel(random_theta);
dotstart = state_region(2,1);% dotstart = -4;
dotend = state_region(2,2);% dotend = -2;


random_dot = dotstart + (dotend - dotstart) * rand(numP_area, 1);
number_dot = numel(random_dot);
global ini_thetadot_high
global ini_theta_high
global input3

training_data_HDC = [];
env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");
HDC_theta = []; HDC_dot = [];HDC_actions= [];
ctrl_step  = 30;
for i = 1: number_theta
    %for k = 1:number_dot
        ini_theta_high = random_theta(i);
        ini_thetadot_high = random_dot(i);
        HDC_dot = [HDC_dot, random_dot(i)];
        HDC_theta = [HDC_theta,random_theta(i)];
        
        counter = 0;
        observation_test = reset(env);
        plot(env)
        done = false;
        while counter < ctrl_step
    %   dlObservation = dlarray(observation, 'SSCB');
        new_input_images = observation_test{1,1};
        new_input_scalars = observation_test{1,2};
        dsX1New = arrayDatastore(new_input_images, 'IterationDimension', 4);
        dsX2New = arrayDatastore(new_input_scalars');
        dsNewData = combine(dsX1New, dsX2New);
        action = predict(loaded_HDC, dsNewData); 
        HDC_actions = [HDC_actions,action];
        [observation_test, reward, done, info] = step(env, action);
        %After taking the action, we store the next state. 
        HDC_dot = [HDC_dot, observation_test{1,2}];
        HDC_theta = [HDC_theta,input3(end)];
        
        counter = counter + 1;
        end
        HDC_dot = HDC_dot(1:end-1);
        HDC_theta = HDC_theta(1:end-1);

        theta_high = input3(end - ctrl_step+1:end);
        % change from -pi to pi to 0-2pi
        for m = 1:ctrl_step
            if theta_high(1,m) < 0
                theta_high(1,m) = 2*pi + theta_high(1,m);
            end
        end
%we need to delete the last element in the theta and dot.
%     HDC_theta_all = [HDC_theta_all, HDC_theta(1:ctrl_step)];
%     HDC_dot_all = [HDC_dot_all, HDC_dot(1:ctrl_step)];
%     HDC_actions_all = [HDC_actions_all, HDC_actions(1:ctrl_step)];
%     oneloop_training_data_HDC = [HDC_theta_all; HDC_dot_all; HDC_actions_all];
%     training_data_HDC = [training_data_HDC, oneloop_training_data_HDC];
       
end
training_data_HDC = [HDC_theta; HDC_dot; HDC_actions];
value = numel(training_data_HDC);

end

function weights_bias_LD = Train_LDC(training_data_HDC, path_save_LDC)

inputs = [training_data_HDC(1, :); training_data_HDC(2, :)]; % check the first row is the theta and second row is dot
output = training_data_HDC(3,:);

NN_Size = [25,25];  
net = feedforwardnet(NN_Size, 'trainlm');
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'purelin';

 
net.inputs{1}.processFcns = {}; %delete preprocessing procedure.
net.outputs{3}.processFcns = {};

[net,tr] = train(net,inputs, output);
LDC = net;
% directory = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs';
% modelname = 'LDC_theta0-1_dot-2-0.mat';
% fullmodel = fullfile(directory, modelname);
% save(fullmodel, 'net');
% net = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/Multi_LDCs/LDC3_theta0-6.28_dot-2-4.mat');% 20x20 sigmoid, sigmoid, tanh
% net = net.net;
weights1 = net.IW{1};
weights2 = net.LW{2,1};
weights3 = net.LW{3,2};
biases = net.b;
biases1 = biases{1};
biases2 = biases{2};
biases3 = biases{3};
k = 1;
weights_bias_LD = zeros(751,1); %501 neurons dor 20x20 and 751 neurons for 25x25
num_input = 2;
num_ouput = 1;
neurons_1st = 25;
neurons_2nd = 25;
%Input the first layer
for i = 1:neurons_1st %10 is the neuron in the first layer
    for j = 1:num_input %2 is the size of input
        weights_bias_LD(k,1) = weights1(i, j);
        k = k+1;
    end
    weights_bias_LD(k) = biases1(i);
    k = k+1;
end

%Input the second layer
for i = 1:neurons_2nd %5 is the neuron in the second layer. 20 is the neuron in the second layer
    for j = 1:neurons_1st %2 is the size of neurons in previous layer. 20 is the neuron in the previou layer
        weights_bias_LD(k,1) = weights2(i, j);
        k = k+1;
    end
    weights_bias_LD(k) = biases2(i);
    k = k+1;
end
%Input the third layer
for i = 1:num_ouput % the neuron in the third layer 
    for j = 1:neurons_2nd  %the size of neuron in the previous layer
        weights_bias_LD(k,1) = weights3(i, j);
        k = k+1;
    end
    weights_bias_LD(k) = biases3(i);
    k = k+1;
end

dlmwrite(path_save_LDC, weights_bias_LD); %dlmwrite('/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC_theta0-1_dot-1-0.txt', weights_bias_LD);

end

function HDC = Load_HDC(path_of_HDC)
    %loaded_HDC = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat');
    HDC = load(path_of_HDC);
    HDC = HDC.net;
end

function [dx,tau] = dynamic1(x,tau) %at time t to get dtheta, ddtheta
theta  = x(1);
dtheta = x(2);
dx(1,1) = dtheta;
dx(2,1) = (8*tau + 2*sin(theta));

end

% function test_one_trajectory(path_test_LDC, low_state)
% other_model = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/LDC1_theta0-6.28_dot0-8_wholestate.txt');
% %other_model = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/controller_single_pendulum_POLAR');
% 
% 
% num_input = 2;
% num_ouput = 1;
% neurons_1st = 25;
% neurons_2nd = 25;
% 
% weights1_other = zeros(neurons_1st, num_input);
% weights2_other = zeros(neurons_2nd, neurons_1st);
% weights3_other = zeros(num_ouput,neurons_2nd);
% 
% bias1_other = zeros(neurons_1st, 1);
% bias2_other = zeros(neurons_2nd, 1);
% bias3_other = zeros(num_ouput,1);
% z = 1;
% for i = 1:neurons_1st %10 is the neuron in the first layer
%     for j = 1:num_input %2 is the size of input
%         weights1_other(i, j) = other_model(z);
%         z = z+1;
%     end
%     bias1_other(i) = other_model(z);
%     z = z+1;
% end
% %Input the second layer
% for i = 1:neurons_2nd %5 is the neuron in the second layer. 20 is the neuron in the second layer
%     for j = 1:neurons_1st %2 is the size of neurons in previous layer. 20 is the neuron in the previou layer
%         weights2_other(i, j) = other_model(z);
%         z = z+1;
%     end
%     bias2_other(i) = other_model(z);
%     z = z + 1;
% end
% %input the third layer
% for i = 1:num_ouput % the neuron in the third layer 
%     for j = 1:neurons_2nd  %the size of neuron in the previous layer
%         weights3_other(i, j) = other_model(z);
%         z = z+1;
%     end
%     bias3_other(i) = other_model(z);
%     z=z+1;
% end
% end



