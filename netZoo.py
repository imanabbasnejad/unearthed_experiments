import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.backends.cudnn.benchmark=True
device = torch.device("cuda")
use_cuda = torch.cuda.is_available()
torch.cuda.manual_seed_all(random.randint(1, 10000))
torch.cuda.synchronize()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv2d_1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv1_t = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1, groups=1, bias=False, dilation=1)
        self.bn1_t = nn.BatchNorm2d(32)

        self.conv2d_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv2_t = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1, groups=1, bias=False, dilation=1)
        self.bn2_t = nn.BatchNorm2d(64)

        self.conv3_t = nn.ConvTranspose2d(in_channels=64, out_channels=22, kernel_size=3, stride=2, padding=1,
                                          output_padding=1, groups=1, bias=False, dilation=1)


    def forward(self, features):
        flatten_features = torch.flatten(features, 1)
        features_reshaped = flatten_features.view(flatten_features.shape[0], 16, 32, 32)

        depth = F.leaky_relu(self.conv2d_1(features_reshaped))
        depth = self.bn1(depth)

        depth = self.conv1_t(depth)
        depth = self.bn1_t(depth)

        depth = F.leaky_relu(self.conv2d_2(depth))
        depth = self.bn2(depth)

        depth = self.conv2_t(depth)
        depth = self.bn2_t(depth)

        depth = self.conv3_t(depth)
        return torch.sigmoid(depth)

