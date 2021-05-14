import os

import numpy as np
import torch
import csv

from utils.utils import get_norm
from Server.initialization import initialization


class Server(object):
    def __init__(self, model_path):
        self.model = initialization(100, 1, None, model_path, model_path, train=False, unity=True)

        self.model.model.load_state_dict(torch.load(os.path.join(model_path, 'fcn/models/fcn_0.1lr_OutScale_60.pth')))
        self.model.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'fcn/models/fcn_0.1lr_OutScale_opt_60.pth')))

        # self.data = torch.empty(0, 5307)
        self.data = torch.empty(0, 926)

        self.full = False
        self.input_mean, self.input_std = get_norm(os.path.join(model_path, "fcn/data/InputNorm.txt"))
        self.input_mean, self.input_std = self.input_mean[0:926], self.input_std[0:926]
        self.output_mean, self.output_std = get_norm(os.path.join(model_path, "fcn/data/OutputNorm.txt"))

        self.csv_writer = csv.writer(open('test.csv', 'w', newline=""))

    def forward(self, x):
        x = np.array(x)
        x = (x - self.input_mean) / self.input_std
        x = torch.FloatTensor([x])
        self.data = torch.cat((self.data, x), 0)
        if self.full is True:
            self.data = self.data[1:]
            data_length = 100
        else:
            data_length = self.data.size(0)
            if data_length >= 100:
                self.full = True

        data = self.model.model.forward(self.data.unsqueeze(0).to(torch.device("cuda:0")))
        print(data.shape)

        data = data[0][-1].cpu().detach().numpy()
        data = data * self.output_std + self.output_mean
        self.csv_writer.writerow(data)
        return data.tolist()
