import gym
import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    def __init__(self, cnn_out_features, v_out_features, q_out_feature) -> None:
        super().__init__(nn.Module)
        self.cnn_out_feature = cnn_out_features
        self.v_out_features = v_out_features
        self.q_out_feature = q_out_feature
        pass 

    def V_network(self):
        self.v_linear_1 = nn.Linear(self.cnn_out_feature, self.v_out_features)
        self.v_activate = nn.ReLU()
        self.v_linear_2 = nn.Linear(self.v_out_features, 1)

    def Q_network(self):
        self.q_linear_1 = 
        pass


    def forword(self, x):
        pass
    


class Replaymemory:
    def __init__(self, n_state, n_action, ) -> None:
        