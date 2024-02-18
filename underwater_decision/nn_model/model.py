from torch import nn


class MarineFoulingNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MarineFoulingNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out
