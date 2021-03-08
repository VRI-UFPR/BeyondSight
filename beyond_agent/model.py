import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

# from utils.distributions import Categorical, DiagGaussian
# from utils.model import get_grid, ChannelPool, Flatten, NNBase

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L82
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs

# Global Policy model code



class Global_Policy(nn.Module):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1, is_mp3d=False):
        super(Global_Policy, self).__init__()

        self.is_mp3d = is_mp3d
        out_size = int(input_shape[1] / 64. * input_shape[2] / 64.)

        ##
        '''
        ReLU
        POINTWISE 1x1 no bias
        DEPTHWISE 3x3 no bias, stride 2
        BatchNormalize2D
        '''
        ##
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_shape[0], input_shape[0]*2, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0]*2, input_shape[0]*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]*2),

            nn.ReLU(),
            nn.Conv2d(input_shape[0]*2, input_shape[0]*4, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0]*4, input_shape[0]*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]*4),

            nn.ReLU(),
            nn.Conv2d(input_shape[0]*4, input_shape[0]*8, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0]*8, input_shape[0]*8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]*8),


            nn.ReLU(),
            nn.Conv2d(input_shape[0]*8, input_shape[0]*4, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0]*4, input_shape[0]*4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]*4),

            nn.ReLU(),
            nn.Conv2d(input_shape[0]*4, input_shape[0]*2, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0]*2, input_shape[0]*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]*2),

            nn.ReLU(),
            nn.Conv2d(input_shape[0]*2, input_shape[0], 1, stride=1, padding=0, bias=False),
            nn.Conv2d(input_shape[0], input_shape[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_shape[0]),

            Flatten()
        )

        #512x512 depth wise
        self.linear1 = nn.Linear(out_size * (input_shape[0]) + 16, hidden_size)
        self.out = nn.Linear(hidden_size, 2)

        if(self.is_mp3d):
            self.object_goal_emb = nn.Embedding(32, 8)
        else:
            self.object_goal_emb = nn.Embedding(8, 8)
        self.orientation_emb = nn.Linear(1, 8)

        # self.train()


    def forward(self, inputs, orientation, object_goal):
        x = self.main(inputs)

        orientation_emb = nn.ReLU()(self.orientation_emb(orientation))
        object_goal_emb = self.object_goal_emb(object_goal).squeeze(1)
        # object_goal_emb = self.object_goal_emb(object_goal)

        object_goal_emb=object_goal_emb.squeeze(1)
        x = torch.cat((x, orientation_emb, object_goal_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        x = nn.Sigmoid()(self.out(x))

        return x
