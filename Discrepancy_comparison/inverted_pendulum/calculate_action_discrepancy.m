%% 1.1 sample the IP states uniformly, propotioanl to the area.
data = load('convex_hulls_IP.mat');  % Replace with the correct path if necessary

% Assuming the convex hulls are stored in a cell array named 'convex_hulls'
convex_hulls = data.all_convex_hulls;  % Adjust if the variable name is different

num_hulls = numel(convex_hulls);
areas = zeros(1, num_hulls);
total_samples = 1500000;  % Total number of points to sample across all hulls
sampled_points = cell(1, num_hulls);  % To store sampled points for each hull

% Calculate the area of each convex hull and the total area
for i = 1:num_hulls
    hull_points = convex_hulls{i};
    areas(i) = polyarea(hull_points(:,1), hull_points(:,2));
end
total_area = sum(areas);

% Calculate the number of samples for each convex hull proportional to its area
num_samples_per_hull = round((areas / total_area) * total_samples);

% Sample points within each convex hull
for i = 1:num_hulls
    % Extract the points of the i-th convex hull
    hull_points = convex_hulls{i};
    
    % Triangulate the convex hull
    DT = delaunayTriangulation(hull_points);
    [num_triangles, ~] = size(DT.ConnectivityList);
    
    % Calculate the area of each triangle in the triangulation
    triangle_areas = zeros(num_triangles, 1);
    for j = 1:num_triangles
        vertices = hull_points(DT.ConnectivityList(j, :), :);
        triangle_areas(j) = polyarea(vertices(:,1), vertices(:,2));
    end
    
    % Total area of the hull from triangulation (sanity check)
    hull_area = sum(triangle_areas);
    
    % Define the number of points to sample for this convex hull
    num_samples = num_samples_per_hull(i);
    points = zeros(num_samples, 2);
    
    % Sample points based on the area-weighted triangles
    for k = 1:num_samples
        % Select a triangle based on area weighting
        chosen_triangle = randsample(1:num_triangles, 1, true, triangle_areas / hull_area);
        
        % Get the vertices of the chosen triangle
        vertices = hull_points(DT.ConnectivityList(chosen_triangle, :), :);
        
        % Sample uniformly within the triangle using barycentric coordinates
        r1 = sqrt(rand);
        r2 = rand;
        sampled_point = (1 - r1) * vertices(1, :) + r1 * (1 - r2) * vertices(2, :) + r1 * r2 * vertices(3, :);
        points(k, :) = sampled_point;
    end
    
    % Store the sampled points for the current hull
    sampled_points{i} = points;
end

% Display the areas and number of points sampled in each convex hull
disp('Areas of each convex hull:');
disp(areas);
disp('Number of sampled points per convex hull:');
disp(num_samples_per_hull);


all_sampled_points = vertcat(sampled_points{:});
% Display the size of the merged matrix
disp('Size of the merged sampled points matrix:');
disp(size(all_sampled_points));


%% 1.2 Sample points following Gaussian distribution
% 
% mu_prime = [0.5122, -0.5104];
% 
% % Updated covariance matrix for more sparsity along y-axis
% Sigma_prime = [0.2519, -0.1576; -0.1576, 0.2564];
% 
% % Number of points to sample
% num_samples = 1500000;
% % Generate samples from the updated Gaussian
% Gaussian_samples = mvnrnd(mu_prime, Sigma_prime, num_samples);





%% 2. calculate the one step control action discrepancy
path_LDC = '/Users/yuang/Documents/MATLAB/New_action_discrepancy/new_multiple_LDCs/LDC_theta0-2_dot-2-0.txt';
path_HDC = '/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat';
loaded_HDC = load(path_HDC);
loaded_HDC = loaded_HDC.net;
variance = 0;

one_step_conformal = one_action_CP(path_LDC, path_HDC, variance, all_sampled_points);
%one_step_conformal = one_action_CP(path_LDC, path_HDC, variance, Gaussian_samples);


% load the txt file and combine two files into one
% postion_test = load('/Users/yuang/Documents/MATLAB/New_action_discrepancy/HSCC_nonconformity/sample_pos.txt');
% vel_test = load('/Users/yuang/Documents/MATLAB/New_action_discrepancy/HSCC_nonconformity/sample_vel.txt');
% sample_state = [postion_test;vel_test];
% dlmwrite('/Users/yuang/Documents/MATLAB/New_action_discrepancy/HSCC_nonconformity/sample_states.txt', sample_state');


function one_step_conformal = one_action_CP(path_LDC, path_HDC, variance, all_sampled_points)

global ini_theta_high
global ini_thetadot_high
global input3

loaded_HDC = load(path_HDC);
loaded_HDC = loaded_HDC.net;

% import the sampled states
sample_theta = all_sampled_points(:,1);
sample_dot = all_sampled_points(:,2);
sample_state = all_sampled_points;

num_states = length(all_sampled_points);

env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");

act_diff_list = [];
for l = 1:num_states
     init_theta = sample_theta(l);
     init_dot = sample_dot(l);

     ini_theta_high = init_theta;
     ini_thetadot_high = init_dot;

% Simulation for high_dimensional controller
    observation_test = reset(env);

    new_input_images = observation_test{1,1};
    new_input_scalars = observation_test{1,2};
    dsX1New = arrayDatastore(new_input_images, 'IterationDimension', 4);
    dsX2New = arrayDatastore(new_input_scalars');
    dsNewData = combine(dsX1New, dsX2New);
    HDC_action = predict(loaded_HDC, dsNewData); 

    % simulation for low_dimensional controller
    states_low(1,1) = init_theta;
    states_low(2,1) = init_dot;

    other_model = load(path_LDC);
    num_input = 2;num_ouput = 1; neurons_1st = 25; neurons_2nd = 25;
    weights1_other = zeros(neurons_1st, num_input);
    weights2_other = zeros(neurons_2nd, neurons_1st);
    weights3_other = zeros(num_ouput,neurons_2nd);
    bias1_other = zeros(neurons_1st, 1); bias2_other = zeros(neurons_2nd, 1);
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
    
    x = [init_theta + sqrt(variance)*rand(1,1) ; init_dot + sqrt(variance)*rand(1,1)];

    states_1st = weights1_other * x + bias1_other;
    %states_1st2 = 1 ./ (1 + exp(-states_1st)); %sigmoid
    states_1st2 = max(0, states_1st); %relu

    states_2nd = weights2_other * states_1st2 + bias2_other;
    %states_2nd2 = 1 ./ (1 + exp(-states_2nd)); %Sigmoid
    states_2nd2 = max(0, states_2nd);

    states_3rd = weights3_other * states_2nd2 + bias3_other; %linear
    %states_3rd2 = tanh(states_3rd);
    states_3rd2 = states_3rd;

    %est_torque_low = states_3rd2 * 2;
    est_torque_low = states_3rd2;

    LDC_action = est_torque_low;

    action_diff = abs(HDC_action - LDC_action);
    act_diff_list = [act_diff_list, action_diff];

end

one_step_conformal = act_diff_list;
dlmwrite('/Users/yuang/Documents/MATLAB/Journal_extension/control_action_discrepancy/Gaussian_1500000action_diff_one.txt',act_diff_list);
dlmwrite('/Users/yuang/Documents/MATLAB/Journal_extension/control_action_discrepancy/Gaussian_1500000sample_states.txt', all_sampled_points);



%dlmwrite('/Users/yuang/Documents/MATLAB/New_action_discrepancy/HSCC_nonconformity/1000000action_diff_one.txt', act_diff_list);
%dlmwrite('/Users/yuang/Documents/MATLAB/New_action_discrepancy/HSCC_nonconformity/1000000sample_states.txt', tranpos_state);

end





