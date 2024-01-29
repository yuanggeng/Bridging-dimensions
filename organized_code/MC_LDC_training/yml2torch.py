import torch
import torch.nn as nn
import yaml


# PyTorch model for network - need to hard code this based on yml network
class Control_NN(nn.Module):

    def __init__(self, layer_1_size=32, layer_2_size=32):
        super(Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# yml dict to pytorch network
def load_model_dict(yml_filename, network: Control_NN):
    with open(yml_filename, 'rb') as f:
        model_dict = yaml.safe_load(f)
    state_dict = {}
    state_dict['fc1.weight'] = torch.FloatTensor(model_dict['weights'][1])
    state_dict['fc1.bias'] = torch.FloatTensor(model_dict['offsets'][1])
    state_dict['fc2.weight'] = torch.FloatTensor(model_dict['weights'][2])
    state_dict['fc2.bias'] = torch.FloatTensor(model_dict['offsets'][2])
    state_dict['fc3.weight'] = torch.FloatTensor(model_dict['weights'][3])
    state_dict['fc3.bias'] = torch.FloatTensor(model_dict['offsets'][3])
    network.load_state_dict(state_dict)


# pytorch network to yml
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