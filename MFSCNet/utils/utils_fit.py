import numpy as np
import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, train_loader,Epoch, cuda, losses):
    # 总损失，坐标损失，置信度损失，类别损失
    train_loss = [0, 0, 0, 0, 0, 0]
    # 训练
    model_train.train()
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, gt_masks, gt_boxes = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    gt_masks = torch.from_numpy(gt_masks).type(torch.LongTensor).cuda()
                    gt_boxes = [torch.from_numpy(box_ann).type(torch.FloatTensor).cuda() for box_ann in gt_boxes]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    gt_masks = torch.from_numpy(gt_masks).type(torch.LongTensor)
                    gt_boxes = [torch.from_numpy(box_ann).type(torch.FloatTensor) for box_ann in gt_boxes]
            # 梯度清零
            optimizer.zero_grad()
            # 前向船舶
            outputs,mask_outputs= model_train(images)
            # 损失计算
            loss_value, per_loss_value = yolo_loss(outputs,mask_outputs,gt_boxes,gt_masks,images,epoch)
            # 反向传播
            loss_value.backward()
            # 梯度下降
            optimizer.step()
            # 记录损失值
            train_loss[0] += loss_value.item()
            train_loss[1] += per_loss_value[0].item()
            train_loss[2] += per_loss_value[1].item()
            train_loss[3] += per_loss_value[2].item()
            train_loss[4] += per_loss_value[3].item()
            train_loss[5] += per_loss_value[4].item()
            # 显示
            pbar.set_postfix(**{'Train:'
                                'lr': get_lr(optimizer),
                                'loss': train_loss[0] / (iteration + 1),
                                'box' : train_loss[1] / (iteration + 1),
                                'obj' : train_loss[2] / (iteration + 1),
                                'cls' : train_loss[3] / (iteration + 1),
                                'mask': train_loss[4] / (iteration + 1),
                                'sc'  : train_loss[5] / (iteration + 1)
                                })
            pbar.update(1)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (train_loss[0] / epoch_step))
    # 每5轮保存一次
    torch.save(model.state_dict(), 'pth//epoch%03d-train_loss%.3f.pth' % (epoch + 1, train_loss[0] / epoch_step))
    # if (epoch+1)%5==0:
    #     torch.save(model.state_dict(),'pth//epoch%03d-train_loss%.3f.pth' % (epoch + 1, train_loss[0] / epoch_step))
    # # 保存损失最小的一次
    loss=train_loss[0]/epoch_step
    # if len(losses)==0:
    #     losses.append(loss)
    # if loss<=min(losses):
    #     torch.save(model.state_dict(),'pth//best-loss%.3f.pth' % (train_loss[0] / epoch_step))
    losses.append(loss)







