clc;
clear all;
close all;
%% combine all the training dataset
torque_1 = load("data_inverted_pendulum/torque_continuous.mat");
%torque_1 = torque_1.torque(1:300);
torque_1 = torque_1.torque;
torque_2 = load("data_inverted_pendulum/torque_1_200_left.mat"); %initial state is the -pi/2
torque_2 = torque_2.torque;
torque_3 = load("data_inverted_pendulum/torque_2_200_right.mat"); %intitial state is the pi/2
torque_3 = torque_3.torque;

theta_1 = load("data_inverted_pendulum/theta_continuous.mat");
theta_1 = theta_1.theta_value;
theta_1 = [pi,theta_1(1:499)];
%theta_1 = [pi,theta_1(1:299)];


theta_2 = load("data_inverted_pendulum/theta200_1_left.mat"); %intial state -pi/2
theta_2 = theta_2.theta200_1_left;
theta_3 = load("data_inverted_pendulum/theta200_2_right.mat");
theta_3 = theta_3.theta200_2_right;

angular_velocity_1 = load("data_inverted_pendulum/angular_velocity_continuous.mat");
angular_velocity_1 = angular_velocity_1.angular_velocity(1:500);
%angular_velocity_1 = angular_velocity_1.angular_velocity(1:300);

angular_velocity_2 = load("data_inverted_pendulum/angular_velocity_1_200_left.mat");
angular_velocity_2 = angular_velocity_2.angular_velocity; %0	-0.390915062904358	-0.790530956348439
angular_velocity_3 = load("data_inverted_pendulum/angular_velocity_2_200_right.mat");
angular_velocity_3 = angular_velocity_3.angular_velocity;

input1 = [theta_1,theta_2,theta_3];
input2 = [angular_velocity_1,angular_velocity_2,angular_velocity_3];

output = [torque_1,torque_2,torque_3];
for k=1:900
inputs(1,k) = input1(1,k);
inputs(2,k) = input2(1,k);
end

%swith the theta from (-pi,pi) to (0,2pi)
for k = 1:900
    if inputs(1,k) < 0
        inputs(1,k) = 2*pi + inputs(1,k);
    end
end
%switch output from (-2,2) to (-1,1)
output  = output/2;

% dlmwrite('input1_700.txt',inputs,'precision','%10.15f')
% dlmwrite('output1_700.txt',output,'precision','%10.15f')



%Train 2nd LDC for the area cannot be verified well.
angular_velocity_2nd = load('/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_data_low_theta/dot_2nd_3300.mat');
angular_velocity_2nd = angular_velocity_2nd.angular_velocity_all;
theta_2nd = load('/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_data_low_theta/theta_2nd_3300.mat');
theta_2nd = theta_2nd.theta_value_all;
torque_2nd = load('/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_data_low_theta/torque_2nd_3300.mat');
torque_2nd = torque_2nd.torque_all;

for k = 1:numel(theta_2nd)
    if theta_2nd(k) < 0
        theta_2nd(k) = 2*pi + theta_2nd(k);
    end
end

inputs = [theta_2nd; angular_velocity_2nd];
output = torque_2nd/2;

% Train the 3rd for specific range of data data.
training_3rd = load('/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_low_theta/2nd_training_data_15000_LDC1.mat');
training_3rd = training_3rd.training_high_theta;
inputs = [training_3rd(1, :); training_3rd(2, :)]; % check the first row is the theta and second row is dot
output = training_3rd(3,:)/2;

%the LDC uses the theta between [0, 2pi] if theta is smaller than 0
%inputs(1,:) = inputs(1,:) + 2*pi;

scatter3(inputs(1,:), inputs(2, :), output(1, :));
xlabel('theta'); ylabel('dot'); zlabel('torque');
hold on;
scatter3(inputs(1,:), inputs(2, :), output(1, :));
xlabel('theta'); ylabel('dot'); zlabel('torque');

%% construct and train regression NN 
NN_Size = [20,20];  % 2 layers, neuron 10-5
net = feedforwardnet(NN_Size, 'trainlm');
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'tansig';
% net.layers{1}.transferFcn = 'poslin';
% net.layers{2}.transferFcn = 'poslin';
 
net.inputs{1}.processFcns = {}; %delete preprocessing procedure.
net.outputs{3}.processFcns = {};

% net.outputs{3}.ProcessParams{1}.ymin = -2;
% net.outputs{3}.ProcessParams{1}.ymax = 2;
% net = configure(net, inputs, output);
% Apply custom scaling factors after configuring the network
% net.outputs{end}.processSettings{1}.gain = diff(output_scaling_factors) / diff(net.outputs{end}.processSettings{1}.xrange);
% net.outputs{end}.processSettings{1}.xoffset = net.outputs{end}.processSettings{1}.xrange(1);
% net.outputs{end}.processSettings{1}.yoffset = output_scaling_factors(1);
% net.performParam.regularization = 0.5; 
% net.trainParam.goal = 1e-5;

[net,tr] = train(net,inputs, output);

directory = '/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_low_theta/make_CP_wide';
modelname = 'LDC6_theta3-4_dot0-3_10000_20x20_sigmoid_tanh.mat';
fullmodel = fullfile(directory, modelname);
save(fullmodel, 'net');

%%well-trained first LDC
%save('NN_20x20_sigmoid_tanh_nopreprocessed_02pi.mat', 'net');  

%% Test the training model
%1.Put all the inputs into the neural network model
training_torque = net(inputs);
x = 1:numel(output);
figure(1)
plot(output, 'r','LineWidth',1.5);
hold on
plot(training_torque,'b-.','LineWidth',1.5)
legend('real_torque','estimated_torque')
xlabel('steps from 1 to 500')
ylabel("value of torque")

errors = output - training_torque;
figure(2)
plot(errors,'b','LineWidth',1.5)
legend('errors between real and estimation');
xlabel('steps from 1 to 500')
ylabel("difference between trained and real torques")
figure(3)
histogram(errors)

% 2.Pipeline test: given one input and put the output into the dynamic system
states_test = zeros(2,501);
states_test(1,1) = input1(1,1);
states_test(2,1) = input2(1,1);
ts = 0.05;
for i = 1:500
    x = [(states_test(1,i)); states_test(2,i)];
    est_torque_low = sim(net, x);
    dx1 = dynamic1(x, est_torque_low);
    dx2 = dynamic1(x+ts*dx1, est_torque_low);
    states_test(:,i+1) = x + ts*0.5*(dx1+dx2);
    states_test(1,i+1) =  wrapToPi(states_test(1,i+1));

end
errors_new = states_low - states_high;
%change to -pi to pi
% error_lowhigh_pi = wrapToPi(errors_new);

theta_test = wrapToPi(states_test(1, 1:501));
figure
subplot(2,2,1)
plot(theta_test(1:300), 'LineWidth',2)
title('test_theta')

subplot(2,2,2)
plot(input1(1:300),'LineWidth',2)
title('real_theta')

subplot(2,2,3)
errors_theta = theta_test(1:500) - input1;
plot(errors_theta, 'LineWidth',2)
title('errors')
subplot(2,2,4)
plot(est_torque_overall,'LineWidth',2);
title('torque')
%% 3.calculate the output from NN and from the hands
weights1 = net.IW{1};
weights2 = net.LW{2,1};
weights3 = net.LW{3,2};
weights = net.LW;
biases = net.b;
biases1 = biases{1};
biases2 = biases{2};
biases3 = biases{3};

states_int = [pi/2; 0];
states_1st = weights1*states_int + biases1;
states_1st2 = 1 ./ (1 + exp(-states_1st));

states_2nd = weights2 * states_1st2 + biases2;
states_2nd2 = 1 ./ (1 + exp(-states_2nd));

output_from_hand = weights3 * states_2nd2 + biases3;
output_from_hand2 = tanh(output_from_hand)
output_from_net = net(states_int)

%% 4.read the weights and biases into txt file.
weights1 = net.IW{1};
weights2 = net.LW{2,1};
weights3 = net.LW{3,2};
biases = net.b;
biases1 = biases{1};
biases2 = biases{2};
biases3 = biases{3};
k = 1;
% weights_bias_LD = zeros(501,1);
weights_bias_LD = zeros(501,1);
num_input = 2;
num_ouput = 1;
neurons_1st = 20;
neurons_2nd = 20;
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

%input the third layer
for i = 1:num_ouput % the neuron in the third layer 
    for j = 1:neurons_2nd  %the size of neuron in the previous layer
        weights_bias_LD(k,1) = weights3(i, j);
        k = k+1;
    end
    weights_bias_LD(k) = biases3(i);
    k = k+1;
end
%dlmwrite('Sigmoid_20x20_wb.txt', weights_bias_LD);
dlmwrite('/Users/yuang/Documents/MATLAB/training_LDC/2nd_training_low_theta/wb_2nd_20x20sigmoid_tanh.txt',weights_bias_LD,'precision','%10.15f')


%% function of dynamic system
function [dx,tau] = dynamic1(x,tau) %at time t to get dtheta, ddtheta
g = 9.81;
L = 1;
m = 1; 
theta  = x(1);
dtheta = x(2);
dx(1,1) = dtheta;
dx(2,1) = 1/(m*L^2) * (tau + m*g*L*sin(theta));
end

