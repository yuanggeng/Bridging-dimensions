import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# pytorch network to yml

class Control_NN(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16):
        super(Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def dump_model_dict(yml_filename, network: Control_NN):
    model_dict = {}
    model_dict['activations'] = {}
    model_dict['activations'][1] = 'Sigmoid'
    model_dict['activations'][2] = 'Sigmoid'
    model_dict['activations'][3] = 'Tanh'
    model_dict['weights'] = {}
    model_dict['offsets'] = {}
    for layer in [1, 2, 3]:
        model_dict['weights'][layer] = network.state_dict()[f'fc{layer}.weight'].tolist()
        model_dict['offsets'][layer] = network.state_dict()[f'fc{layer}.bias'].tolist()
    with open(yml_filename, 'w') as f:
        yaml.dump(model_dict, f)
    return


# load training data
# X_train1 = torch.randn(100, 2)  # 100 samples, 2 features
# Y_train1 = torch.randn(100, 1)  # 100 samples, 1 output
#this training data is for the whole state space
# X_train_np = np.load('1LDC_inputs_states_10000whole.npy')
# Y_train_np = np.load('1LDC_output_action_10000whole.npy')
#this training data is for the LDC2
# X_train_np = np.load('2ndLDC_inputs_(0,0.07)_10000.npy')
# Y_train_np = np.load('2ndLDC_output_(0,0.07)_action_10000.npy')
#this training data is for the LDC3
X_train_np = np.load('3rdLDC_inputs_(0,0.07)(-0.6,0.6)_10000.npy')
Y_train_np = np.load('3rdLDC_output_(0,0.07)(-0.6,0.6)_action_10000.npy')

X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32)
# Hyperparameters
learning_rate = 0.001
epochs = 500
batch_size = 16

# Initialize model, loss, and optimizer
model = Control_NN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # Get mini-batch
        inputs = X_train[i:i + batch_size]
        labels = Y_train[i:i + batch_size]
        # Forward pass
        outputs = model(inputs)
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

dump_model_dict('LDC3_1st_(-0.6,0.6)(0,0.07).yml', model)

torch.save(model, 'LDC3_1st_(-0.6,0.6)(0,0.07).pth')
