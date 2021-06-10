import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from  rnn_state_encoder import  build_rnn_state_encoder



#https://github.com/devendrachaplot/Neural-SLAM/blob/e833e8f20c5a8d71552a5d22fb78ad6980cfc80c/utils/model.py
class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, h, w)

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Global Policy model code
class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Goal_prediction_network(nn.Module):

    def __init__(self, input_shape=(25,256,256), hidden_size=512, embedding_size=8):
        super(Goal_prediction_network, self).__init__()


        '''
        Based on Global Policy of Neural SLAM
        2 linear,

        linear 1 (240/16)*(240/16)*(32)+8 = 7208 to 512
        linear 2 512 to 256

        critic/value_net is a linear from 256 to 1
        '''
        # out_size = c x h x w

        # self.feature_out_size = int(np.floor(hidden_size/2))
        self.feature_out_size = hidden_size

        # last_channel_size = 32
        # out_size = int(input_shape[1] / 16. * input_shape[2] / 16. * last_channel_size )
        #as of right now a map of 256,256 will create a 2^13 features = 8192
        #8192 can be too much for a fully connected layers

        """
        VERSION FROM NEURAL SLAM
        """
        # self.main = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(8, 32, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     Flatten()
        # )

        """
        MY VERSION V2 IT HAS LINEAR LAYERS TRANSFORMED INTO CONV LAYERS
        """
        map_channels = input_shape[0]
        # self.main = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(map_channels, 32, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(32, 8192, 16, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(8192, 4096, 1, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(4096, 2048, 1, stride=1, padding=0),
        #     nn.ReLU(),
        #
        #     Flatten()
        # )

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(map_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),

            Flatten()
        )

        self.linear0 = nn.Linear( 8192, 4096)
        self.linear1 = nn.Linear( 4096, 2048)

        '''
        Let us put some conv layers that are equivalent to linear layers to further reduce
        2^13 to at least 2^11
        '''

        ##21 classes + 1
        self.object_goal_emb = nn.Embedding(21+1, embedding_size)

        # 5 in 5 degrees
        self.orientation_emb = nn.Embedding(73+1, embedding_size)

        self.rnn_input_size = 2048 + embedding_size + embedding_size
        self.rnn_layers = 2
        # self.rnn_layers = 1
        # self.linear1 = nn.Linear( 2048 + embedding_size + embedding_size, hidden_size)
        # self.linear2 = nn.Linear( hidden_size,self.feature_out_size)


        self.state_encoder = build_rnn_state_encoder(
            input_size=self.rnn_input_size,
            hidden_size=self.feature_out_size,
            rnn_type='lstm',
            num_layers=1,
        )

        self.linear2 = nn.Linear( self.rnn_input_size, self.feature_out_size)


    def forward(self, input, object_goal, orientation, rnn_hidden_states, masks):

        x = self.main(input)
        x = nn.ReLU()(self.linear0(x))
        x = nn.ReLU()(self.linear1(x))
        object_goal_emb = self.object_goal_emb(object_goal + 1).squeeze(1)
        orientation_emb = self.orientation_emb(orientation + 1).squeeze(1)

        x_out = torch.cat((x, object_goal_emb, orientation_emb), -1)

        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks
        )

        # x_out = nn.ReLU()(self.linear2(x_out))

        return x_out, rnn_hidden_states
