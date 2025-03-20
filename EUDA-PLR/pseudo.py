
import sys
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils import data
from model.deeplabv3 import get_deeplab_v3
from model.discriminator import get_fc_discriminator
from dataset.Target import TargetDataSet
import torch.nn.functional as F
from utils.func import loss_calc, bce_loss
from domain_adaptation.config import cfg, cfg_from_file
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg



#------------------------------------- color -------------------------------------------
palette = [0, 0, 0, 255, 255, 255, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    
    return new_mask    

def colorize_save(output_pt_tensor, name):
    mask_Img = Image.fromarray(output_pt_tensor)
    mask_color = colorize(output_pt_tensor)

    name = name.split('.')[0]
    mask_Img.save('./color_masks/%s.png' % (name))
    mask_color.save('./color_masks/%s_color.png' % (name.split('.')[0]))


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict,False)
    model.eval()
    model.cuda(device)

def get_arguments():
    """
    Parse input arguments 
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")

    parser.add_argument('--best_iter', type=int, default=1000,
                        help='iteration with best mIoU')
    parser.add_argument('--cfg', type=str, default=r'E:\root\code\EUDA-PLR\advent\scripts\configs\advent.yml',
                        help='optional config file' )
    return parser.parse_args()

def main(args):

    # load configuration file 
    device = cfg.GPU_ID    
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    if not os.path.exists('./color_masks'):
        os.mkdir('./color_masks')

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # load model with parameters trained from Inter-domain adaptation
    model_gen = get_deeplab_v3(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TEST.MULTI_LEVEL)
    
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{args.best_iter}.pth')
    
    print("Loading the generator:", restore_from)
    
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    
    # load data
    target_dataset = TargetDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=None,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.ENTROPY_BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=None)

    target_loader_iter = enumerate(target_loader)

    # upsampling layer
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    entropy_list = []
    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image, _, _, name = batch
        with torch.no_grad():
            _, _, pred_trg_main = model_gen(image.cuda(device))
            pred_trg_main = interp_target(pred_trg_main)
            if args.normalize == False:
                normalizor = 1

            # generate binary mask
            output_np_tensor = pred_trg_main.cpu().data[0].numpy()
            mask_np_tensor = output_np_tensor.transpose(1, 2, 0)
            mask_np_tensor = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)

            colorize_save(mask_np_tensor, name[0])

if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    main(args)
