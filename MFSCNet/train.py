import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from nets.yolo import YoloBody
from utils.utils import get_classes,loss_save
from torch.utils.data import DataLoader
from nets.yolo_training import YOLOLoss, weights_init
from utils.dataloader import Dataset,yolo_mask_dataset_collate
from config import HyperParameter
from utils.utils_fit import fit_one_epoch
def dataset_get(args):
    ### 读取图片路径，并且需要读取类别和yaml文件，yaml文件中存放了每个像素所属的类别，用于后面进行多分类
    imglist = os.listdir(args.img_flooder)
    lines=[imglist,args.img_flooder,args.mask_flooder]
    num= len(imglist)
    return lines,num
if __name__ == "__main__":
    mask_weight = 2
    sc_weight   = 0.1
    # 超参数设置
    args=HyperParameter()
    # 获取类名和类别数量
    class_names, num_classes = get_classes(args.classes_path)
    # 实例化模型
    model = YoloBody(num_classes, args.phi)
    # 模型权重随机初始化
    weights_init(model)
    # 加载预训练权重
    if args.model_path != '':
        print('Load weights {}.'.format(args.model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # 设置模型为训练模式
    model_train = model.train()
    # 使用GPU
    if args.Cuda:
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # 初始化损失函数
    yolo_loss    = YOLOLoss(num_classes,is_sc=args.is_sc,mask_weight=mask_weight,sc_weight=sc_weight)
    # 初始化损失保存
    # 数据集进行训练集、验证集和测试集划分
    lines, num = dataset_get(args)
    # 冻结权重训练，仅训练分类器
    # 存储权重参数
    losses = []
    if args.Freeze_Train:
        print("冻结权重训练")
        # 冻结权重训练阶段基本参数
        batch_size  = args.Freeze_batch_size
        lr          = args.Freeze_lr
        start_epoch = args.Init_Epoch
        end_epoch   = args.Freeze_Epoch
        epoch_step      = num // batch_size
        # 数据集过少抛出异常
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # 选择激活函数
        if args.optim =='adam':
            optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        elif args.optim =='sgd':
            optimizer = optim.SGD(model_train.parameters(), lr, weight_decay=5e-4)

        # 选择是否使用余弦退火学习率
        if args.Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # 加载训练和验证数据
        train_dataset   = Dataset(lines, args.input_shape, num_classes, end_epoch - start_epoch, mosaic = args.mosaic, train = True,class_names=class_names)
        train_loader    = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True, collate_fn=yolo_mask_dataset_collate)
        # 冻结权重训练，冻结主干特征提取网络部分
        start_time=time.time()
        if args.Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            # 获取当前epoch,用于在数据加载中执行不同任务
            train_loader.dataset.epoch_now=epoch
            fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch,
                    epoch_step,  train_loader, end_epoch, args.Cuda,losses)
            # 更新学习率
            lr_scheduler.step()
    # 训练整体模型
    if True:
        print("解冻训练")
        batch_size  = args.Unfreeze_batch_size
        lr          = args.Unfreeze_lr
        start_epoch = args.Freeze_Epoch
        end_epoch   = args.Unfreeze_Epoch
        epoch_step      = num // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        if args.optim =='adam':
            optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        elif args.optim =='sgd':
            optimizer = optim.SGD(model_train.parameters(), lr, weight_decay=5e-4)
        if args.Cosine_scheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        train_dataset   = Dataset(lines, args.input_shape, num_classes, end_epoch - start_epoch, mosaic = args.mosaic, train = True,class_names=class_names)
        train_loader    = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True, collate_fn=yolo_mask_dataset_collate)
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        if args.Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True
        for epoch in range(start_epoch, end_epoch):
            train_loader.dataset.epoch_now=epoch
            fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch,
                    epoch_step, train_loader, end_epoch, args.Cuda,losses)
            lr_scheduler.step()
    loss_save(losses)