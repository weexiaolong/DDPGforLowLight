import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import memory
import copy

class L_color(nn.Module):

    def __init__(self, args):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self, args):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(args.device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).to(args.device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(args.device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).to(args.device).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False).type(torch.FloatTensor).to(args.device)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False).type(torch.FloatTensor).to(
            args.device)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False).type(torch.FloatTensor).to(args.device)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False).type(torch.FloatTensor).to(args.device)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, args, patch_size):
        super(L_exp, self).__init__()
        self.args = args
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val

    def forward(self, x, mean_val):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).to(self.args.device), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, args, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self, args):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k

class DDPG(object):
    def __init__(self, args, actor, critic):
        # network, optimizer for actor
        self.args = args
        self.actor = actor.to(args.device)
        self.actor_target = copy.deepcopy(actor).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)

        # network, optimizer for critic
        self.critic = critic.to(args.device)
        self.critic_target = copy.deepcopy(critic).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        # create replay buffer object
        self.replay_buffer = memory.ReplayBuffer(max_size=self.args.mem_capacity)

    def select_action(self, state):
        # select action based on actor network and add some noise on it for exploration
        action = self.actor(state)
        return action

    def update(self):
        for i in range(self.args.episodes):
            s, a, r, s_, d = self.replay_buffer.sample(self.args.sample_szie)
            # transfer these tensors to GPU
            state = torch.FloatTensor(s).to(self.args.device)
            action = torch.FloatTensor(a).to(self.args.device)
            reward = torch.FloatTensor(r).to(self.args.device)
            next_state = torch.FloatTensor(s_).to(self.args.device)
            done = torch.FloatTensor(d).to(self.args.device)

            action = action.squeeze(dim=1)
            state = state.squeeze(dim=1)
            next_state = next_state.squeeze(dim=1)

            # compute the target Q value
            tx = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, tx)
            target_Q = reward + (done * self.args.gamma * target_Q).detach()
            # Get the current Q value
            current_Q = self.critic(state, action)
            # compute critic loss by MSE
            critic_loss = F.mse_loss(current_Q, target_Q)
            # use optimizer to update the critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute the actor loss and its gradient to update the parameters
            x = self.actor(state)
            actor_loss = -self.critic(state, x).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network of actor and critic
            # zip() constructs tuple from iterable object
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def save(self, actor, critic):
        torch.save(self.actor.state_dict(), actor)
        torch.save(self.critic.state_dict(), critic)

    def load(self, actor, critic):
        self.actor.load_state_dict(torch.load(actor))
        self.critic.load_state_dict(torch.load(critic))