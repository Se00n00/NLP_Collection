import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, droupout=0.5):    # ff_dim is usally higher that model_dim
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(droupout)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.droupout(x)       # apply dropout to the output of the second linear layer to reduce overfitting
        return x