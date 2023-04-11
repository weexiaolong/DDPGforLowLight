import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class enhance_net_nopool(nn.Module):

    def __init__(self, scale_factor=1):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        #   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, x_r):

        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return enhance_image

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv = enhance_net_nopool()

    def forward(self, x):
        x, xr = self.conv(x)
        return x, xr

class Critic(nn.Module):
    def __init__(self, ):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, state, action):
        '''
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.cat([x, action], 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        '''
        return state


class DDPG:
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = Actor().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = Critic().to(device)
        self.target_critic = Critic().to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 1e-3

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        # 随机从经验回放缓冲区中抽取一批经验数据
        batch = random.sample(self.memory, batch_size)

        # 将抽取的经验数据拆分成独立的张量（状态、动作、奖励、下一状态、是否结束）
        states = torch.tensor([transition[0] for transition in batch], dtype=torch.float32)
        actions = torch.tensor([transition[1] for transition in batch], dtype=torch.float32)
        rewards = torch.tensor([transition[2] for transition in batch], dtype=torch.float32)
        next_states = torch.tensor([transition[3] for transition in batch], dtype=torch.float32)
        dones = torch.tensor([transition[4] for transition in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones


    def update_policy(self):
        # 从经验回放缓冲区中采样一个批次的经验
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_batch(16)

        # 计算目标Q值
        with torch.no_grad():
            target_action = self.target_actor(next_state_batch)
            target_q = self.target_critic(next_state_batch, target_action)
            target_q = reward_batch + self.gamma * (1 - done_batch) * target_q

        # 更新Critic网络
        q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def load_weights(self, output):
        self.actor.load_state_dict(torch.load('{}/actor.pth'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pth'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(),'{}/actor.pth'.format(output))
        torch.save(self.critic.state_dict(),'{}/critic.pth'.format(output))
