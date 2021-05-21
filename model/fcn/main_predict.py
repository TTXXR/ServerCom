import os
import pandas as pd
import numpy as np

import torch
from net import Model
from utils.utils import get_norm
from torch.autograd import Variable


def predict(net, input, label, seq_flag=False):
    if not seq_flag:
        return net.model(input), label
    else:
        out = []
        for i in range(len(input)):
            x = net.model(input[i])
            out.append(x)
            print(x)
        return torch.Tensor(out), label


if __name__ == '__main__':
    root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"
    loss_func = torch.nn.MSELoss()
    num_inputs, num_outputs, num_hiddens = 926, 618, 256
    batch_size, num_epochs = 64, 40

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = Model(num_inputs, num_hiddens, num_outputs, batch_size)
    net.model.load_state_dict(
        torch.load("models/fcn_0.1lr_OutScale_60.pth", map_location=torch.device("cuda:0")))
    net.optimizer.load_state_dict(
        torch.load("models/fcn_0.1lr_OutScale_opt_60.pth", map_location=torch.device("cuda:0")))
    net.model.eval()

    input_data = pd.read_csv(root_path + "Input/" + "1.txt", sep=' ', header=None, dtype=float)
    label_data = pd.read_csv(root_path + "Label/" + "1.txt", sep=' ', header=None, dtype=float)

    # scale 标准化
    # scale = StandardScaler()
    # scale = scale.fit(input_data)
    # input_data = torch.Tensor(scale.transform(input_data))

    # 手动标准化
    input_mean, input_std = get_norm("data/InputNorm.txt")
    output_mean, output_std = get_norm("data/OutputNorm.txt")

    input_mean, input_std = input_mean[0:926], input_std[0:926]
    input_data = torch.Tensor((np.array(input_data).astype('float32') - input_mean) / input_std)

    label_data = torch.Tensor(np.array(label_data))
    input_data = Variable(input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
    label_data = Variable(label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

    # single test
    index = -64
    # pred, target = predict(net, input_data[index:], label_data[index:])
    # loss = loss_func(pred, target).sum()
    # print(loss)

    # sequence test
    pred, target = predict(net, input_data[index:], label_data[index:], seq_flag=True)
    pred = pred * output_std + output_mean
    loss = loss_func(pred, target).sum()
    print(loss)
