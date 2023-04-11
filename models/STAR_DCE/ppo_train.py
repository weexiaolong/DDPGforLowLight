import gym
from models.ddpg import DDPG
import torch.utils.data as data
import torch
import random
import glob
from PIL import Image
import numpy as np
from losses import L_TV, L_spa, L_color, L_exp

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 512

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()

        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)



if __name__ == '__main__':
    train_dataset = lowlight_loader('../data/zero_dce/Inputs_jpg/')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    agent = DDPG()
    L_color = L_color()
    L_spa = L_spa()
    #L_exp = L_exp(16, 2)
    L_TV = L_TV()

    for epoch in range(100):
        for iteration, img_lowlight in enumerate(train_loader):
            state = img_lowlight.cuda()

            action, A = agent.actor(state)

            E = 0.6
            Loss_TV = 1600 * L_TV(A)
            loss_spa = torch.mean(L_spa(action, state))
            loss_col = 5 * torch.mean(L_color(action))

            # best_loss
            reward = Loss_TV + loss_spa + loss_col
            next_state = action
            agent.remember(state, action, reward, next_state, False)
        agent.update_policy()

