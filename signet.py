from torch import nn

class SigNetModel(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """
    def __init__(self):
        super(SigNetModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dense1 = nn.Linear(256*17*25, 1024)
        self.dense2 = nn.Linear(1024, 128)

        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward_once(self, x):
        x = self.relu(self.conv1(x))
        x = self.lrn(x)
        x = self.maxpool2d(x)
        x = self.relu(self.conv2(x))
        x = self.lrn(x)
        x = self.maxpool2d(x)
        x = nn.Dropout2d(p=0.3)(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool2d(x)
        x = nn.Dropout2d(p=0.3)(x)

        x = x.view(x.shape[0], 256 * 17 * 25)
        x = self.relu(self.dense1(x))
        x = nn.Dropout(p=0.5)(x)
        x = self.relu(self.dense2(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
