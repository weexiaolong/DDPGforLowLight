import os.path

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "/*.png")
    train_list = image_list_lowlight
    random.shuffle(train_list)

    return train_list

class lowlight_loader(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.train_list = populate_train_list(os.path.join(args.images_path, 'low'))
        self.label_list = [i.replace('low', 'high') for i in self.train_list]
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.args.image_size, self.args.image_size), Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()

        data_highlight_path = self.label_list[index]
        data_highlight = Image.open(data_highlight_path)
        data_highlight = data_highlight.resize((self.args.image_size, self.args.image_size), Image.ANTIALIAS)
        data_highlight = (np.asarray(data_highlight) / 255.0)
        data_highlight = torch.from_numpy(data_highlight).float()

        return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

