import torch.nn


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1),
                                         # output = (input_size - kernel_size + 2*padding) / stride + 1
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(in_channels=64,
                                                         out_channels=128,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,
                                                            kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features=14 * 14 * 128,
                                                         out_features=1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(in_features=1024,
                                                         out_features=10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x
