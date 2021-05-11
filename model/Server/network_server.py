import os
from datetime import datetime

import numpy as np
import torch
import csv

from model.utils.data_preprocess import get_norm
from model.utils.initialization import initialization
from sklearn.preprocessing import StandardScaler


class Server(object):
    def __init__(self, model_path):
        self.model = initialization(100, 1, None, model_path, model_path, train=False, unity=True)
        # self.model.load_param()
        # self.model.encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder_1.pth')))
        # self.model.enc_opt.load_state_dict(torch.load(os.path.join(model_path, 'encoder_opt_1.pth')))
        # self.model.decoder.load_state_dict(torch.load(os.path.join(model_path, 'decoder_1.pth')))
        # self.model.dec_opt.load_state_dict(torch.load(os.path.join(model_path, 'decoder_opt_1.pth')))

        self.model.model.load_state_dict(torch.load(os.path.join(model_path, 'fcn_fixedScale_30.pth')))
        self.model.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'fcn_fixedScale_opt_30.pth')))

        # self.data = torch.empty(0, 5307)
        self.data = torch.empty(0, 926)

        self.full = False
        # self.input_mean, self.input_std = get_norm("/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/InputNorm.txt")
        self.input_mean, self.input_std = get_norm("/home/rr/Downloads/nsm_data/utils/inputNorm.txt")
        self.output_mean, self.output_std = get_norm("/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/OutputNorm.txt")

        self.input_mean, self.input_std = self.input_mean[0:926], self.input_std[0:926]
        self.output_mean, self.output_std = self.output_mean[0:926], self.output_std[0:926]

        self.csv_writer = csv.writer(open('test.csv', 'w', newline=""))

        self.scale = StandardScaler()

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
        # t1=datetime.now()

        data = self.model.model.forward(self.data.unsqueeze(0).to(torch.device("cuda:0")))
        print(data.shape)

        # t2=datetime.now()
        # print(t2-t1)
        data = data[0][-1].cpu().detach().numpy()
        # print(data.shape)
        # data = data * self.output_std + self.output_mean
        self.csv_writer.writerow(data)
        return data.tolist()
