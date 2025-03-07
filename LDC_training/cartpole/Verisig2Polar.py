import yaml

#network = 'sig200x200'
#network = 'test_LDC1'
# network = 'MC_LDCs/LDC2_(0,0.07)'

# yml_name = 'switch_60steps_all/LDCs_60steps/LDC60_({0}, {1})-({2}, {3}).yml'.format(
#     round(pos_splited[i], 3), round(pos_splited[i + 1], 3), round(vel_splited[j], 3),
#     round(vel_splited[j + 1], 3))

network = 'Cart_LDC1_3layers_new'
nn_verisig = network + '.yml'

# network = 'MC_LDCs/LDC3_1st_(-0.6,0.6)(0,0.07)'
# nn_verisig = network + '.yml'

dnn_dict = {}
with open(nn_verisig, 'r') as f:
    dnn_dict = yaml.load(f)

layers = len(dnn_dict['activations'])
input_size = len(dnn_dict['weights'][1][0])
output_size = len(dnn_dict['weights'][layers])

for i in dnn_dict['activations']:
    if dnn_dict['activations'][i] == 'Tanh':
        dnn_dict['activations'][i] = 'tanh'
    elif dnn_dict['activations'][i] == 'Sigmoid':
        dnn_dict['activations'][i] = 'sigmoid'
    elif dnn_dict['activations'][i] == 'Linear':
        dnn_dict['activations'][i] = 'Affine'

with open(network, 'w') as nnfile:
    nnfile.write(str(input_size)+"\n")
    nnfile.write(str(output_size)+"\n")
    nnfile.write(str(layers-1)+"\n")    # number of hidden layers
    for i in range(layers-1):           # output size of each hidden layer
        nnfile.write(str(len(dnn_dict['weights'][i+1]))+"\n")
    for i in range(layers):
        nnfile.write(str(dnn_dict['activations'][i+1])+"\n")
    for i in range(layers):
        for j in range(len(dnn_dict['weights'][i+1])):
            for k in range(len(dnn_dict['weights'][i+1][j])):
                nnfile.write(str(dnn_dict['weights'][i+1][j][k])+"\n")
            nnfile.write(str(dnn_dict['offsets'][i+1][j])+"\n")
    nnfile.write(str(0)+"\n")           # output offset
    nnfile.write(str(1)+"\n")           # output scaling factor