from torch import nn

class TripletNetModel(nn.Module):

    # def __init__(self):
    #     super(TripletNetModel, self).__init__()
    #
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1),
    #         # nn.LocalResponseNorm(size=96, alpha=0.0001, beta=0.75, k=2),
    #         nn.BatchNorm2d(96, eps=1e-06, momentum=0.9),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=3, stride=2)
    #     )
    #     self.conv2 = nn.Sequential(
    #         nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
    #         # nn.LocalResponseNorm(size=256, alpha=0.0001, beta=0.75, k=2),
    #         nn.BatchNorm2d(256, eps=1e-06, momentum=0.9),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Dropout2d(p=0.3),
    #     )
    #     self.conv3 = nn.Sequential(
    #         nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(384, eps=1e-06, momentum=0.9),
    #         nn.ReLU()
    #     )
    #     self.conv4 = nn.Sequential(
    #         nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(256, eps=1e-06, momentum=0.9),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Dropout2d(p=0.3)
    #     )
    #     self.linear1 = nn.Sequential(
    #         nn.Linear(256 * 17 * 25, 1024),
    #         nn.BatchNorm1d(1024),
    #         nn.ReLU(),
    #         nn.Dropout2d(p=0.5)
    #     )
    #     self.linear2 = nn.Sequential(
    #         nn.Linear(1024, 128),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU()
    #     )
    #
    # def forward_once(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #     x = x.view(x.shape[0], 256 * 17 * 25)
    #     x = self.linear1(x)
    #     x = self.linear2(x)
    #     return x
    #


    def __init__(self):
        super(TripletNetModel, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 35 * 52, 2048),
                                nn.PReLU(),
                                nn.Linear(2048, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, 128)
                                )

    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
