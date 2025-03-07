
%% rectangle data
numPoints = 25000;
intervalStart = 0;%0 
intervalEnd = 2*pi; %3
random_theta = intervalStart + (intervalEnd - intervalStart) * rand(numPoints, 1);
number_theta = numel(random_theta);
dotstart = -6;
dotend = 6;

random_dot = dotstart + (dotend - dotstart) * rand(numPoints, 1);
number_dot = numel(random_dot);

sample_theta = random_theta;
sample_thetadot = random_dot;

%% simulation results for POLAR_IP_LDC
ctrl_step  = 1;
% 2nd way to load the LDC by weights and biases matrix
other_model = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/controller_single_pendulum_POLAR');

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
for i = 1:neurons_1st %10 is the neuron in the first layer
    for j = 1:num_input %2 is the size of input
        weights1_other(i, j) = other_model(z);
        z = z+1;
    end
    bias1_other(i) = other_model(z);
    z = z+1;
end
%Input the second layer
for i = 1:neurons_2nd %5 is the neuron in the second layer. 20 is the neuron in the second layer
    for j = 1:neurons_1st %2 is the size of neurons in previous layer. 20 is the neuron in the previou layer
        weights2_other(i, j) = other_model(z);
        z = z+1;
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



global ini_theta_high
global ini_thetadot_high
global input3
env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");
observation = [];

for j = 1:numPoints
    int_theta = sample_theta(j);
    int_thetadot = sample_thetadot(j);
    ini_theta_high = int_theta;   %Double check DDPG's theta range from [-pi, pi]
    ini_thetadot_high = int_thetadot;
    observation_one_time = reset(env);
    plot(env)
    observation = [observation; observation_one_time];
    % simulation for low_dimensional controller.
    
    states_low = zeros(2,ctrl_step+1);
    states_low(1,1) = int_theta;  %LDC's theta ranges from [0, 2*pi]
    states_low(2,1) = int_thetadot;
    ts = 0.05;

for L = 1:ctrl_step
    x = [(states_low(1,L)); states_low(2,L)];

    % test LDC model
    states_1st = weights1_other * x + bias1_other;
    %states_1st2 = 1 ./ (1 + exp(-states_1st)); %sigmoid
    states_1st2 = max(0, states_1st);           %Relu
    states_2nd = weights2_other * states_1st2 + bias2_other;
    %states_2nd2 = 1 ./ (1 + exp(-states_2nd)); %Sigmoid
    states_2nd2 = max(0, states_2nd);           %Relu
    states_3rd = weights3_other * states_2nd2 + bias3_other; %linear
    %states_3rd2 = tanh(states_3rd);   %tanh
    est_torque_low = states_3rd;
    torque_low = est_torque_low;

    est_torque_overall_other(j) = est_torque_low;
    dx1 = dynamic1(x, est_torque_low);
    dx2 = dynamic1(x + ts * dx1, est_torque_low);
    states_low(:,L+1) = x + ts * 0.5 * (dx1+dx2);
    %states_test_other(1,i+1) =  wrapToPi(states_test_other(1,i+1));

end
%     disp(['finsihed: %d', j]);
    fprintf('Finished number of sample: %d\n', j)

end


% Pre-allocate cell array
training_data = cell(numPoints, 3);

% HDC data for training
for i = 1:numPoints
    % For the observation: Assuming you have a way to generate or load your 50x50 observation
    training_data{i, 1} = observation{i,1};  %  random 50x50 matrix, replace with actual data
    
    training_data{i, 2} = observation{i,2};

    training_data{i, 3} = est_torque_overall_other(i);
end
save('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/training_data_25000_theta(0-2pi)(-6-6)_4HDC.mat', 'training_data');




%% LDC groud truth
% numPoints = 20000;
% intervalStart = 0;%0 
% intervalEnd = 2*pi; %3
% random_theta = intervalStart + (intervalEnd - intervalStart) * rand(numPoints, 1);
% number_theta = numel(random_theta);
% dotstart = -6;
% dotend = 0;
% random_dot = dotstart + (dotend - dotstart) * rand(numPoints, 1);
% number_dot = numel(random_dot);
% 
% sample_theta = random_theta;
% sample_thetadot = random_dot;
% ctrl_test = 1;
%     states_low = zeros(2,ctrl_test+1);
% 
%     ts = 0.05;
% for j = 1:numPoints
% 
%     states_low(1,1) = random_theta(j);  %LDC's theta ranges from [0, 2*pi]
%     states_low(2,1) = random_dot(j);
%     x = [random_theta(j); random_dot(j)];
% 
%     % test LDC model
%     states_1st = weights1_other * x + bias1_other;
%     %states_1st2 = 1 ./ (1 + exp(-states_1st)); %sigmoid
%     states_1st2 = max(0, states_1st);           %Relu
%     states_2nd = weights2_other * states_1st2 + bias2_other;
%     %states_2nd2 = 1 ./ (1 + exp(-states_2nd)); %Sigmoid
%     states_2nd2 = max(0, states_2nd);           %Relu
%     states_3rd = weights3_other * states_2nd2 + bias3_other; %linear
%     %states_3rd2 = tanh(states_3rd);   %tanh
%     est_torque_low = states_3rd;
%     torque_low = est_torque_low;
% 
%     est_torque_overall_other(j) = est_torque_low;
%     disp(['Finish points:', num2str(j)]);
% 
% 
% 
% end
% LDC_ground_truth= zeros(numPoints, 3);
% 
% for i = 1:numPoints
%     % For the observation: Assuming you have a way to generate or load your 50x50 observation
%     LDC_ground_truth(i, 1) = random_theta(i);  % Example: random 50x50 matrix, replace with actual data
%     LDC_ground_truth(i, 2) = random_dot(i);  
%     LDC_ground_truth(i, 3) = est_torque_overall_other(i);
% end
% 
% %save('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/Ground_truth_LDC_20000.mat', 'LDC_ground_truth');
% 
% figure
% scatter3(LDC_ground_truth(:, 1), LDC_ground_truth(:, 2), LDC_ground_truth(:, 3)); 
% title('3D Scatter Plot of LDC ground truth');
% xlabel('theta-Axis');
% ylabel('dot-Axis');
% zlabel('torque-Axis');




function [dx,tau] = dynamic1(x,tau) %at time t to get dtheta, ddtheta
%g = 9.81;
L = 1;
m = 1; 
theta  = x(1);
dtheta = x(2);
dx(1,1) = dtheta;
dx(2,1) = 1/(m*L^2) * (8*tau + 2*sin(theta));
end
