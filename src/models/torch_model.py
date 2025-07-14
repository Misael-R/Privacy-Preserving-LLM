# models/torch_model.py

import torch.nn as nn

class PrivacyAwareEmailClassifier(nn.Module):
    def __init__(self, input_dim=2000):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
