import numpy as np
from models import ddpg
import torch
import torch.nn as nn
from models import dataloader
import argparse

def post_precess(x, action):
    r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(action, 3, dim=1)
    x = x + r1 * (torch.pow(x, 2) - x)
    x = x + r2 * (torch.pow(x, 2) - x)
    x = x + r3 * (torch.pow(x, 2) - x)
    x = x + r4 * (torch.pow(x, 2) - x)
    x = x + r5 * (torch.pow(x, 2) - x)
    x = x + r6 * (torch.pow(x, 2) - x)
    x = x + r7 * (torch.pow(x, 2) - x)
    x = x + r8 * (torch.pow(x, 2) - x)
    x = torch.clamp(x, 0, 1)
    return x
def main(args, agent):
    train_dataset = dataloader.lowlight_loader(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_szie,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False)

    np.random.seed(args.random_seed)  # fix the random seed

    L_color = ddpg.L_color(args).to(args.device)
    L_spa = ddpg.L_spa(args).to(args.device)
    L_exp = ddpg.L_exp(args, 16).to(args.device)
    L_TV = ddpg.L_TV(args).to(args.device)

    L_L1 = nn.L1Loss()
    L_L2 = nn.MSELoss()

    for episode in range(args.episodes):
        total_reward = 0
        for iteration, state in enumerate(train_loader):
            with torch.no_grad():
                img_input, img_ref = state
                img_input = img_input.to(args.device)
                img_ref = img_ref.to(args.device)

                for i in range(6):
                    action = agent.select_action(img_input)

                    b, c, h, w = action.shape
                    noise = np.random.normal(0, args.noise_var, size=b * c * h * w)#.clip(-0.1, 0.1)
                    noise = torch.from_numpy(noise).type(torch.FloatTensor).to(args.device)

                    noise = noise.view(b, c, h, w)
                    action = action + noise

                    enhence = post_precess(img_input, action)

                    for img_input_i, img_ref_i, action_i, enhence_i in zip(img_input, img_ref, action, enhence):
                        img_input_i = img_input_i.unsqueeze(0)
                        action_i = action_i.unsqueeze(0)
                        enhence_i = enhence_i.unsqueeze(0)
                        img_ref_i = img_ref_i.unsqueeze(0)

                        E = 0.6
                        Loss_TV = 1600 * L_TV(action_i)
                        loss_spa = torch.mean(L_spa(enhence_i, img_input_i))
                        loss_exp = 10 * torch.mean(L_exp(enhence_i, E))
                        loss_col = 5 * torch.mean(L_color(enhence_i))

                        loss_cos = 1000*(1 - nn.functional.cosine_similarity(img_input_i, img_ref_i, dim=1).mean())
                        loss_l1 = L_L1(img_input_i, img_ref_i)

                        reward = Loss_TV + loss_spa + loss_col + loss_exp + loss_cos + loss_l1
                        reward = - reward

                        agent.replay_buffer.store_transition((
                            img_input_i.cpu().numpy(),
                            action_i.cpu().numpy(),
                            reward.cpu().numpy(),
                            enhence_i.cpu().numpy(),
                            float(False)))

                        total_reward += reward

                    img_input = enhence

            print("Episode : {} \t iteration : {} \t total_reward:{:0.2f}".format(episode, iteration, total_reward))
            agent.update()
            total_reward = 0


        agent.save(
            './checkpoints/{}/actor_{}.pth'.format(args.model, episode),
            './checkpoints/{}/critic_{}.pth'.format(args.model, episode)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model', type=str, default='dce')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mem_capacity', type=int, default=10000)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--batch_szie', type=int, default=96)
    parser.add_argument('--sample_szie', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--images_path', type=str, default='../data/ReLLIE/our485/')

    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--random_seed', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--noise_var', type=float, default=0.1)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    if args.model == 'dce':
        from models.ActorCritic import Actor, Critic
        actor = Actor()
        critic = Critic()
        agent = ddpg.DDPG(args, actor, critic)
        if args.load:
            agent.load('./checkpoints/dce/actor_199.pth', './checkpoints/dce/critic_199.pth')
        main(args, agent)
    elif args.model == 'star':
        from models.StarNet import Actor, Critic

        actor = Actor()
        critic = Critic()
        agent = ddpg.DDPG(args, actor, critic)
        if args.load:
            agent.load('./checkpoints/star/actor_199.pth', './checkpoints/star/critic_199.pth')
        main(args, agent)
