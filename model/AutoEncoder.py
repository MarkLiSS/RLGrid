import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.chamfer_distance import chamfer_distance_numpy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = [nn.Conv1d(3, 64, kernel_size=1), nn.ReLU()]
        conv2 = [nn.Conv1d(64, 128, kernel_size=1), nn.ReLU()]
        conv3 = [nn.Conv1d(128, 256, kernel_size=1), nn.ReLU()]
        conv4 = [nn.Conv1d(256, 512, kernel_size=1), nn.ReLU()]
        conv5 = [nn.Conv1d(512, 1024, kernel_size=1), nn.ReLU()]
        Linear6 = [nn.Linear(1920, 1024), nn.ReLU()]
        Linear7 = [nn.Linear(1024, 512), nn.ReLU()]
        Linear8 = [nn.Linear(512, 256), nn.ReLU()]
        Linear9 = [nn.Linear(256, 128)]
        self.maxpool = nn.MaxPool1d(2048)
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        self.conv5 = nn.Sequential(*conv5)
        self.Linear6 = nn.Sequential(*Linear6)
        self.Linear7 = nn.Sequential(*Linear7)
        self.Linear8 = nn.Sequential(*Linear8)
        self.Linear9 = nn.Sequential(*Linear9)

    def forward(self, x):
        out_64 = self.conv1(x)
        out_128 = self.conv2(out_64)  # torch.Size([32, 128, 2048])
        out_256 = self.conv3(out_128)  # torch.Size([32, 256, 2048])
        out_512 = self.conv4(out_256)  # torch.Size([32, 512, 2048])
        out_1024 = self.conv5(out_512)  # torch.Size([32, 1024, 2048])
        out_64 = self.maxpool(out_64)
        out_128 = self.maxpool(out_128)
        out_256 = self.maxpool(out_256)
        out_512 = self.maxpool(out_512)
        out_1024 = self.maxpool(out_1024)
        L = [out_1024, out_512, out_256, out_128]
        out = torch.cat(L, 1)  # torch.Size([32, 1920, 1])
        out = out.squeeze(2)
        out = self.Linear6(out)  # torch.Size([32, 1024, 1])
        out = self.Linear7(out)  # torch.Size([32, 512, 1])
        out = self.Linear8(out)  # torch.Size([32, 256, 1])
        out = self.Linear9(out)  # torch.Size([32, 128, 1])
        return out


class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        self.num_points = num_points
        linear1 = [nn.Linear(128, 256), nn.ReLU()]
        linear2 = [nn.Linear(256, 512), nn.ReLU()]
        linear3 = [nn.Linear(512, 1024), nn.ReLU()]
        linear4 = [nn.Linear(1024, self.num_points * 3)]
        self.linear1 = nn.Sequential(*linear1)
        self.linear2 = nn.Sequential(*linear2)
        self.linear3 = nn.Sequential(*linear3)
        self.linear4 = nn.Sequential(*linear4)

    def forward(self, x):
        out1 = self.linear1(x)
        out2 = self.linear2(out1)
        out3 = self.linear3(out2)
        out4 = self.linear4(out3)

        return out4.view(-1, 3, self.num_points)


class AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)

    def forward(self, x):
        gfv = self.encoder(x)
        out = self.decoder(gfv)

        return out, gfv


class ChamferLoss(nn.Module):
    def __init__(self, num_points):
        super(ChamferLoss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)

    def forward(self, predict_pc, gt_pc):
        dist1 = chamfer_distance_numpy(predict_pc, gt_pc)
        dist2 = chamfer_distance_numpy(gt_pc, predict_pc)
        self.loss = (dist1 + dist2) / 2
        return self.loss

