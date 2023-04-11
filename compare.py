import torch
import torchvision
import torch.optim
import os
import numpy as np
from models.ActorCritic import Actor as DCEActor
from models.StarNet import Actor as StarActor
from PIL import Image
from utils import image_utils
from models.zero_dce.model import enhance_net_nopool

def image_process(image_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    return data_lowlight

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

def lowlight(dce, dAcotr, sAcotr, data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dceOut, _ = dce(data_lowlight)

    for i in range(6):
        a1 = dAcotr(data_lowlight)
        dAcotrOut = post_precess(data_lowlight, a1)
        data_lowlight = dAcotrOut
    for i in range(6):
        a2 = sAcotr(data_lowlight)
        sAcotrOut = post_precess(data_lowlight, a2)
        data_lowlight = sAcotrOut

    return dceOut, dAcotrOut, sAcotrOut

if __name__ == '__main__':
    dAcotr = DCEActor().cuda()
    sAcotr = StarActor().cuda()
    dce = enhance_net_nopool(1).cuda()

    dAcotr.load_state_dict(torch.load('./checkpoints/dce/actor_199.pth'))
    sAcotr.load_state_dict(torch.load('./checkpoints/star/actor_199.pth'))
    dce.load_state_dict(torch.load('./checkpoints/zero-dce/Epoch99.pth'))

    with torch.no_grad():
        highimages = '../data/ReLLIE/eval15/high'
        lowimages = '../data/ReLLIE/eval15/low'
        results = '../data/ReLLIE/eval15/results'

        psnr_dce_total = 0
        psnr_dactor_total = 0
        psnr_sactor_total = 0
        for i in os.listdir(lowimages):
            data_lowlight = image_process(os.path.join(lowimages, i))
            data_highlight = image_process(os.path.join(highimages, i))
            dceOut, dAcotrOut, sAcotrOut = lowlight(dce, dAcotr, sAcotr, data_lowlight)

            psnr_dce = image_utils.torchPSNR(data_highlight, dceOut)
            psnr_dactor = image_utils.torchPSNR(data_highlight, dAcotrOut)
            psnr_sactor = image_utils.torchPSNR(data_highlight, sAcotrOut)

            psnr_dce_total += psnr_dce.cpu().numpy()
            psnr_dactor_total += psnr_dactor.cpu().numpy()
            psnr_sactor_total += psnr_sactor.cpu().numpy()

            print(psnr_dce, psnr_dactor, psnr_sactor)
            C = torch.cat((data_lowlight, dAcotrOut, sAcotrOut, data_highlight), 3)
            torchvision.utils.save_image(C, os.path.join(results, i))

        print(psnr_dce_total/len(os.listdir(lowimages)))
        print(psnr_dactor_total / len(os.listdir(lowimages)))
        print(psnr_sactor_total / len(os.listdir(lowimages)))