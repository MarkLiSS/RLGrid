import os
import sys
import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from GAN.gan import GenSAGAN, DiscSAGAN
from utils.chamfer_distance import chamfer_distance_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

max_action = 10
z_dim = 32
batch_size_actor = 32
discount = 0.99
tau = 0.005

# ### Replay Buffer
class ReplayBuffer(object):
    def __init__(self, size):
        self.episodes = []
        self.buffer_size = size

    def add_to_buffer(self, state, action, reward, next_state):
        if len(self.episodes) == self.buffer_size:
            self.episodes = self.episodes[1:]
        self.episodes.append(
            (
                state.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                reward,
                next_state.detach().cpu().numpy(),
            )
        )

    def get_batch(self, batch_size=100):
        states = []
        actions = []
        rewards = []
        next_state = []

        for i in range(batch_size):
            epi = random.choice(self.episodes)
            states.append(epi[0])
            actions.append(epi[1])
            rewards.append(epi[2])
            next_state.append(epi[3])

        rewards = np.array(rewards)
        rewards = rewards.reshape((rewards.shape[0], 1))
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_state)),
        )


# ### Critic Network
class CriticNet(nn.Module):
    def __init__(self, state_dim, z_shape):
        super(CriticNet, self).__init__()

        self.linear1 = nn.Linear(state_dim + z_shape, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, 1)

    def forward(self, state, z):
        out = F.relu(self.linear1(torch.cat([state, z], dim=1)))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out


# ### Actor Network
class ActorNet(nn.Module):
    def __init__(self, state_dim, z_shape, max_action=10):
        super(ActorNet, self).__init__()

        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400, 400)
        self.linear3 = nn.Linear(400, 300)
        self.linear4 = nn.Linear(300, z_shape)

        self.max_action = max_action

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu((self.linear1(x)))
        x = F.relu((self.linear2(x)))
        x = F.relu((self.linear3(x)))
        x = self.max_action * torch.tanh(self.linear4(x))
        return x


class DDPG(object):
    def __init__(self, state_dim, z_dim, max_action):
        self.actor = ActorNet(state_dim, z_dim, max_action).to(device)
        self.actor_target = ActorNet(state_dim, z_dim, max_action).to(device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = CriticNet(state_dim, z_dim).to(device)
        self.critic_target = CriticNet(state_dim, z_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = ReplayBuffer(int(5e3))

    def get_optimal_action(self, state):
        action = self.actor(state)
        return action.detach().cpu().numpy()

    def forward(self, step):
        # if len(replay_buffer.episodes) < batch_size_actor:
        #     return
        for x in self.actor_target.state_dict().keys():
            eval("self.actor_target." + x + ".data.mul_((1-tau))")
            eval("self.actor_target." + x + ".data.add_(tau*self.actor." + x + ".data)")
        for x in self.critic_target.state_dict().keys():
            eval("self.critic_target." + x + ".data.mul_((1-tau))")
            eval(
                "self.critic_target." + x + ".data.add_(tau*self.critic." + x + ".data)"
            )
        # for it in range(iterations):
        (state_m, action_m, reward_m, next_state_m) = self.replay_buffer.get_batch(
            batch_size_actor
        )

        state = state_m[:, 0, :].float().to(device)
        action = action_m[:, 0, :].float().to(device)
        reward = reward_m.reshape(-1, 1).to(device)
        next_state = next_state_m[:, 0, :].float().to(device)
        # done_m = done_m.reshape(-1, 1)
        # done = torch.FloatTensor(1 - done_m).to(device)

        a = self.actor(state)
        q = self.critic(state, a)

        loss_a = -torch.mean(q)

        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()

        a_ = self.actor_target(next_state)
        q_ = self.critic_target(next_state, a_)
        q_target = reward + discount * q_

        q_v = self.critic(state, action)
        td_error = F.mse_loss(q_target, q_v)

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pt" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pt" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pt" % (directory, filename)))
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pt" % (directory, filename))
        )


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

    def encode(self, x):
        gfv = self.encoder(x)
        return gfv

    def decode(self, x):
        return self.decoder(x)


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


# ### Self-Attention
class SAttn(nn.Module):
    def __init__(self, dim):
        super(SAttn, self).__init__()

        self.query = nn.Conv2d(dim, dim // 8, 1)
        self.key = nn.Conv2d(dim, dim // 8, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, w, h = x.size()
        query = self.query(x)
        query = query.view(batch_size, -1, w * h).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, w * h)

        matmul = torch.bmm(query, key)
        attn = self.softmax(matmul)

        value = self.value(x).view(batch_size, -1, w * h)

        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(batch_size, c, w, h)
        out = self.gamma * out + x

        return out, attn


class GenSAGAN(nn.Module):
    def __init__(self, image_size=32, z_dim=32, conv_dim=64):
        super(GenSAGAN, self).__init__()
        repeat_num = int(np.log2(image_size)) - 3  # repeat_num=2
        mult = 2 ** repeat_num  # mult=4

        self.layer1 = nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)
        self.bn1 = nn.BatchNorm2d(conv_dim * mult)

        self.layer2 = nn.ConvTranspose2d(
            conv_dim * mult, (conv_dim * mult) // 2, 3, 2, 2
        )
        self.bn2 = nn.BatchNorm2d((conv_dim * mult) // 2)

        self.layer3 = nn.ConvTranspose2d(
            (conv_dim * mult) // 2, (conv_dim * mult) // 4, 3, 2, 2
        )
        self.bn3 = nn.BatchNorm2d((conv_dim * mult) // 4)

        self.layer4 = nn.ConvTranspose2d(64, 1, 2, 2, 1)

        self.attn1 = SAttn(64)
        self.attn2 = SAttn(64)

        self.conv1d = nn.ConvTranspose1d(144, 128, 1)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        out = F.relu(self.layer1(x))  # ([B x 32 x 1 x 1], [B x 256 x 4 x 4])
        out = self.bn1(out)

        out = F.relu(self.layer2(out))  # ([B x 256 x 4 x 4], [B x 128 x 5 x 5])
        out = self.bn2(out)

        out = F.relu(self.layer3(out))  # ([B x 128 x 5 x 5], [B x 64 x 7 x 7])
        out = self.bn3(out)

        out, p1 = self.attn1(out)  # ([B x 1 x 7 x 7], [B x 1 x 7 x 7])

        out = self.layer4(out)  # ([B x 64 x 7 x 7], [B x 1 x 12 x 12])

        out = out.view(-1, 1, 144)  # ([B x 1 x 12 x 12], [B x 1 x 144])
        out = out.transpose(1, 2)  # [B x 144 x 1]

        out = self.conv1d(out)  # ([B x 144x 1], [B x 128 x 1])
        out = out.transpose(2, 1)  # [B x 1 x 128]

        out = out.view(-1, 128)  # [B x 128]

        return out, p1


class DiscSAGAN(nn.Module):
    def __init__(self, image_size=32, conv_dim=64):
        super(DiscSAGAN, self).__init__()
        self.layer1 = nn.Conv2d(1, conv_dim, 3, 2, 2)
        self.layer2 = nn.Conv2d(conv_dim, conv_dim * 2, 3, 2, 2)
        self.layer3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, 3, 2, 2)

        self.layer4 = nn.Conv2d(conv_dim * 4, 1, 4)

        self.attn1 = SAttn(256)
        self.attn2 = SAttn(512)

        self.conv1d = nn.ConvTranspose1d(128, 144, 1)

    def forward(self, x):
        # x = x.squeeze(1)
        x = x.unsqueeze(-1)  # ([B x 128], [B x 128 x 1])
        x = self.conv1d(x)  # ([B x 128 x 1], [B x 144 x 1])
        x = x.transpose(2, 1)  # ([B x 144 x 1], [B x 1 x 144])
        x = x.view(-1, 1, 12, 12)  # ([B x 144 x 1], [B x 1 x 12 x 12])

        out = F.leaky_relu(self.layer1(x))  # ([B x 1 x 12 x 12], [B x 64 x 7 x 7])
        out = F.leaky_relu(self.layer2(out))  # ([B x 64 x 7 x 7], [B x 128 x 5 x 5])
        out = F.leaky_relu(self.layer3(out))  # ([B x 128 x 5 x 5], [B x 256 x 4 x 4])

        out, p1 = self.attn1(out)  # ([B x 256 x 4 x 4], [B x 256 x 4 x 4])

        out = self.layer4(out)  # ([B x 256 x 4 x 4], [B x 1 x 1 x 1])
        out = out.reshape(x.shape[0], -1)  # ([B x 1 x 1 x 1], [B x 1])
        return out, p1

