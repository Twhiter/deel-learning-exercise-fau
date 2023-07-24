import torch
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear, Sigmoid, LeakyReLU, Dropout


class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()





        self.c1 = Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=3,padding=(1,1))
        self.b1 = BatchNorm2d(num_features=out_channels)
        self.r1 = LeakyReLU()
        self.d1 = Dropout(p=0.3)

        self.c2 = Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=(1,1))
        self.b2 = BatchNorm2d(num_features=out_channels)
        self.r2 = LeakyReLU()
        # self.d2 = Dropout(p=0.3)

        self.b = BatchNorm2d(num_features=in_channels)
        self.c = Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1)


    def forward(self,x):

        o = x.clone()
        o = self.c1(o)
        o = self.b1(o)
        o = self.r1(o)
        o = self.d1(o)

        o = self.c2(o)
        o = self.b2(o)

        x = self.b(x)
        x = self.c(x)

        o = x + o
        o = self.r2(o)

        return o


class ConvBlock(torch.nn.Module):

    def __init__(self,in_chan,out_chan,k=3,stride=1):
        super().__init__()
        self.conv = Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=k,stride=stride)
        self.batch_norm = BatchNorm2d(out_chan)
        self.relu = LeakyReLU()
        self.maxPooling = MaxPool2d(kernel_size=3,stride=2)
        self.dropout = Dropout(p=0.3)

    def forward(self, x):

        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxPooling(x)

        return self.dropout(x)




class ResNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # self.conv = Conv2d(3,64,7,2)
        # self.b_norm1 = BatchNorm2d(64)
        # self.relu = LeakyReLU()
        # self.maxPool = MaxPool2d(kernel_size=3,stride=2)
        # self.dropout = Dropout(p=0.3)

        self.conv_block1 = ConvBlock(3,64,7,2)

        self.resB1 = ResBlock(64,64,1)
        self.resB2 = ResBlock(64,128,2)

        self.resB = ResBlock(256,256,1)

        self.resB3 = ResBlock(128,256,2)
        self.resB4 = ResBlock(256,512,2)
        self.resB5 = ResBlock(512,512,1)

        self.globalAvgPool = AdaptiveAvgPool2d((1,1))

        self.flatten = Flatten()

        self.fc = Linear(512,2)
        self.d = Dropout(p=0.3)
        self.fc2 = Linear(256,2)
        self.s = Sigmoid()

    def forward(self, x):

        x = self.conv_block1(x)

        x = self.resB1(x)
        x = self.resB2(x)

        x = self.resB3(x)

        x = self.resB(x)

        x = self.resB4(x)
        x = self.resB5(x)

        x = self.globalAvgPool(x)

        x = self.flatten(x)
        x = self.fc(x)
        # x = self.d(x)
        # x = self.fc2(x)

        return self.s(x)