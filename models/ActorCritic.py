import torch
import torch.nn as nn
import torch.nn.functional as F

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        l = 32
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(in_channels=l, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(in_channels=l, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv4 = nn.Conv2d(in_channels=l, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv5 = nn.Conv2d(in_channels=l * 2, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv6 = nn.Conv2d(in_channels=l * 2, out_channels=l, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv7 = nn.Conv2d(in_channels=l * 2, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        return x
class ActorPlus(nn.Module):
    def __init__(self):
        super(ActorPlus, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=1)
        number_f = 32

        #   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

        self.train()

    def forward(self, x):
        x_down = x
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return x_r
        # enhance_image = self.enhance(x, x_r)
        # return enhance_image, x_r

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.contiguous().view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))

        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv21 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv1 = LeNet()
        self.conv2 = LeNet()

        self.fc1 = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, 1)

        self.train()

    def forward(self, x, a):
        x = self.conv1(x)

        a = F.relu(self.conv21(a))
        a = self.conv2(a)

        x = torch.cat((x, a), 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x