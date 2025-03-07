% Load your training data

training_data = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/training_data_25000_theta(0-2pi)(-6-6)_4HDC.mat');
training_data = training_data.training_data;


% Extract and reformat data
num_samples = size(training_data, 1);

input_images = zeros(50, 50, 1, num_samples);
input_scalars = zeros(1, num_samples);
target_data = zeros(1, num_samples);

for i = 1:num_samples
    input_images(:, :,1, i) = training_data{i, 1};
    input_scalars(i) = training_data{i, 2};
    target_data(i) = training_data{i, 3};
end
dsX1Train = arrayDatastore(input_images,IterationDimension=4);
dsX2Train = arrayDatastore(input_scalars');
dsTTrain = arrayDatastore(target_data');
dsTrain = combine(dsX1Train,dsX2Train,dsTTrain);
% 
% [X1Train,TTrain,X2Train] = digitTrain4DArrayData;
% size(input_images)


%% build our own HDC model
% Image input layers
% Image Input Layers
imgInput = imageInputLayer([50 50 1], 'Name', 'imgInput');
convLayer = convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv');
myReluLayer = reluLayer('Name', 'relu');
flattenLayer = flattenLayer('Name', 'flatten');

% Scalar Input Layer
scalarInput = featureInputLayer(1, 'Normalization', 'none', 'Name', 'scalarInput');

% Fully Connected Layer before Concatenation
fcLayer1 = fullyConnectedLayer(20, 'Name', 'fc1'); % adjust the size as needed

% Concatenation Layer
concatLayer = concatenationLayer(1, 2, 'Name', 'concat');

% Further Layers
fcLayer2 = fullyConnectedLayer(1, 'Name', 'fc2'); % adjust the size as needed
regressionLayer = regressionLayer('Name', 'regression');

% Create Layer Graph
layers = layerGraph();
layers = addLayers(layers, [imgInput; convLayer; myReluLayer; flattenLayer; fcLayer1]);
layers = addLayers(layers, scalarInput);
layers = addLayers(layers, [concatLayer; fcLayer2; regressionLayer]);

% Connect Layers
layers = connectLayers(layers, 'fc1', 'concat/in1');
layers = connectLayers(layers, 'scalarInput', 'concat/in2');

plot(layers)

%% start training
options = trainingOptions('sgdm');  % Define your training options
% net = trainNetwork({input_images, input_scalars}, target_data', layers, options);
net = trainNetwork(dsTrain, layers, options);

save('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR_3rd(-6,6).mat', 'net');


%% simulation for HDC
env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");
maxSteps = 30;
counter = 0;
global ini_theta_high
global ini_thetadot_high
global input3

sample_theta = 1.8 : 0.001 : 1.81;
sample_dot = -1.99 : 0.001 : 2;
numPoints = length(sample_theta);
theta_high_all = [];
for l = 1:numPoints
     init_theta = sample_theta(l);
     init_dot = sample_dot(l);
    
     ini_theta_high = init_theta;
     ini_thetadot_high = init_dot;
% ini_theta_high = 1.4;
% ini_thetadot_high = -1.75;
observation_test = reset(env);
%plot(env)
action_items = [];
loaded_HDC = load('/Users/yuang/Documents/MATLAB/IP_POLAR_LDC2HDC/HDC_POLAR.mat');
loaded_HDC = loaded_HDC.net;
while counter < maxSteps
     % Convert observation to dlarray
%     dlObservation = dlarray(observation, 'SSCB');
    new_input_images = observation_test{1,1};
    new_input_scalars = observation_test{1,2};
    dsX1New = arrayDatastore(new_input_images, 'IterationDimension', 4);
    dsX2New = arrayDatastore(new_input_scalars');
    dsNewData = combine(dsX1New, dsX2New);
    action = predict(loaded_HDC, dsNewData); 
    action_items = [action_items,action];
    [observation_test, reward, done, info] = step(env, action);
    counter = counter + 1;
end
 theta_high = input3(end - maxSteps+1:end);
 theta_high = [init_theta, theta_high];
 theta_high_all = [theta_high_all; theta_high];
end
 figure
 plot(theta_high)

%%  build only image-based CNN
% % Image input layers
% % Image Input Layers
% imgInput = imageInputLayer([50 50 1], 'Name', 'imgInput');
% convLayer = convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv');
% myReluLayer = reluLayer('Name', 'relu');
% flattenLayer = flattenLayer('Name', 'flatten');
% 
% % Scalar Input Layer
% %scalarInput = featureInputLayer(1, 'Normalization', 'none', 'Name', 'scalarInput');
% 
% % Fully Connected Layer before Concatenation
% fcLayer1 = fullyConnectedLayer(20, 'Name', 'fc1'); % adjust the size as needed
% 
% % Concatenation Layer
% concatLayer = concatenationLayer(1, 2, 'Name', 'concat');
% 
% % Further Layers
% fcLayer2 = fullyConnectedLayer(1, 'Name', 'fc2'); % adjust the size as needed
% regressionLayer = regressionLayer('Name', 'regression');
% 
% % Create Layer Graph
% layers = layerGraph();
% layers = addLayers(layers, [imgInput; convLayer; myReluLayer; flattenLayer; fcLayer1;fcLayer2;regressionLayer]);
% %layers = addLayers(layers, scalarInput);
% %layers = addLayers(layers, [concatLayer; fcLayer2; regressionLayer]);
% 
% % Connect Layers
% %layers = connectLayers(layers, 'fc1', 'concat/in1');
% %layers = connectLayers(layers, 'scalarInput', 'concat/in2');
% 
% plot(layers)
% 
% %% start training
% options = trainingOptions('sgdm');  % Define your training options
% % net = trainNetwork({input_images, input_scalars}, target_data, layers, options);
% net = trainNetwork(input_images, target_data', layers, options);

