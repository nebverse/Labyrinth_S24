# custom_models/custom_dreamer_model.py

import torch
import torch.nn as nn
from sheeprl.models.models import CNN

class CustomDreamerModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_dim):
        super(CustomDreamerModel, self).__init__()
        self.cnn = CNN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            activation=nn.ReLU()
        )
        self.fc = nn.Linear(hidden_channels[-1] * 8 * 8, output_dim)  # Adjust based on output of CNN

    def forward(self, obs):
        x = self.cnn(obs)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
