
import torch
from torch import nn

class MNATT(nn.Module):
    def __init__(self, channels, factor=8):
        super(MNATT, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.agmax=nn.AdaptiveMaxPool2d(1)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x2=self.agmax(x2)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class Bottleneck(nn.Module): # Convblock
    def __init__(self, in_channel, filters, s):
        super(Bottleneck, self).__init__()
        c1, c2, c3 = filters
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.short_cut = nn.Conv2d(in_channel, c3, kernel_size=1, stride=s, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)
        self.MNATT=MNATT(c3)

    def forward(self, x):
        output_x = self.bottleneck(x)
        output_x= self.MNATT(output_x)
        short_cut_x = self.batch1(self.short_cut(x))
        result = output_x + short_cut_x
        X = self.relu(result)
        return X


class BasicBlock(nn.Module):
    def __init__(self,in_channel,filters):
        super(BasicBlock, self).__init__()
        c1, c2, c3 = filters
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.MNATT=MNATT(c3)
    def forward(self, x):
        identity = x
        output_x = self.basicblock(x)
        output_x=self.MNATT(output_x)
        X = identity + output_x
        X = self.relu(X)
        return X


class ResNet(nn.Module):
    def __init__(self,num_class):
        super(ResNet, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, filters=[64, 64, 256],s=1),
            BasicBlock(256, filters=[64, 64, 256]),
            BasicBlock(256, filters=[64, 64, 256]),
            MNATT(256)
        )

        self.stage3 = nn.Sequential(
            Bottleneck(256, [128, 128, 512],s=2),
            BasicBlock(512, filters=[128, 128, 512]),
            BasicBlock(512, filters=[128, 128, 512]),
            BasicBlock(512, filters=[128, 128, 512]),
            MNATT(512)
        )

        self.stage4 = nn.Sequential(
            Bottleneck(512, [256, 256, 1024],s=2),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            MNATT(1024)
        )

        self.stage5 = nn.Sequential(
            Bottleneck(1024, [512, 512, 2048],s=2),
            BasicBlock(2048, filters=[512, 512, 2048]),
            BasicBlock(2048, filters=[512, 512, 2048]),
            MNATT(2048)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.stage1(x)  # torch.Size([1, 64, 56, 56])
        out = self.stage2(out)  # torch.Size([1, 256, 56, 56])
        #print(out.shape)
        out = self.stage3(out) # torch.Size([1, 512, 28, 28])
        #print(out.shape)
        out = self.stage4(out) # torch.Size([1, 1024, 14, 14])
        #print(out.shape)
        out = self.stage5(out)  # torch.Size([1, 2048, 7, 7])
        #print(out.shape)
        out = self.pool(out)
        out = out.view(out.size(0), 2048)
        out = self.fc(out)
        #print(out.shape)
        #print("ok")
        return out


