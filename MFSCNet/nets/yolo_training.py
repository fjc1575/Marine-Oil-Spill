#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )


        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    CE_loss = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1), temp_target)
    return CE_loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss



class YOLOLoss(nn.Module):    
    def __init__(self, num_classes, strides=[8, 16, 32],is_sc=False,mask_weight=1,sc_weight=1):
        super().__init__()
        self.num_classes        = num_classes
        self.strides            = strides
        self.bcewithlog_loss    = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss           = IOUloss(reduction="none")
        self.grids              = [torch.zeros(1)] * len(strides)
        self.is_sc              = is_sc
        self.mask_weight        = int(mask_weight)
        self.sc_weight          = sc_weight
    def forward(self, inputs,mask_inputs,labels=None,mask_labels=None,input_images=None,now_epoch=0):
        # 输入图片大小为（5，1024，1024）
        outputs             = []
        x_shifts            = []
        y_shifts            = []
        expanded_strides    = []
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            # output为调整之后，对应到原图上的大小, (4,16384,6),(dx,dy,w,h)为在原图上的大小,grid为框的数量(1,16384,2)#
            # output为调整之后在原图上所对应的坐标, output在这个地方没有做sigmod的，需要注意
            # 将output转换至原图
            # output(batch_size,4+obj+cls,128,128)、(64,64)、(32,32)
            '''
            将预测结果转到原始图上
            grid为特征图上对应的点
            '''
            output, grid = self.get_output_and_grid(output, k, stride)
            # x的范围0-127，0-63，0-31
            x_shifts.append(grid[:, :, 0])
            # y的范围
            y_shifts.append(grid[:, :, 1])
            # 相当于说x_shifts和y_shifts用来表示数量，expand_strides用来表示x和y应当放大多少才能和原图一样大
            # （0，1，2-127）*8，（0，1，2-63）*16，（0，1，2-31）*32
            # 相当于获取一个系数，用来判断每个grid或者特征图的缩放系数，这个地方只是为了获取一个缩放系数
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)
        # 根据预测mask获取对应的真值mask的大小
        # 在使用之前进行即可
        '''
        x_shifts:x轴的数量
        y_shifts:y轴的数量
        expanded_strides:grid对应于原图的位置
        labels:(dx,dy,w,h)真值框
        outputs:(dx,dy,w,h)网络预测结果，转换为原图对应位置之后的结果
        mask_inputs:mask预测结果
        gt_masks:mask真值
        '''
        return self.get_losses(x_shifts, y_shifts, expanded_strides,labels,torch.cat(outputs, 1),mask_inputs,mask_labels,input_images,now_epoch)


    def get_output_and_grid(self, output, k, stride):
        '''
        output:网络预测的结果
        [[batch_size, num_classes + 5, 128, 128]
        [batch_size, num_classes + 5, 64, 64]
        [batch_size, num_classes + 5, 32, 32]]
        '''

        grid            = self.grids[k]
        hsize, wsize    = output.shape[-2:]
        # 该部分和Mask R-CNN思路类似，根据特征图的大小，创建点用了表示对应于原图的每一个框
        if grid.shape[2:4] != output.shape[2:4]:
            # 生成网格
            yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid            = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k]   = grid
        # 类似于展平操作,(1,128,128,2)->(1,16384,2)
        grid                = grid.view(1, -1, 2)
        # print(grid.shape)
        # 在特征图大小上进行展平，调整位置，和grid类似，(4,6,128,128)->(4,16384,6)
        output              = output.flatten(start_dim=2).permute(0, 2, 1)
        # 预测的是相对于每个框的位置，所以，预测的结果加上所属于的框，就是在特征图上所对应的位置，然后，再*步长，即为在原图上的位置
        # 预测的中心点是0-1之间的一个值，然后加上
        output[..., :2]     = (output[..., :2] + grid) * stride
        # 2：4为w和h，做一个指数
        # 防止出现负数的情况，因此需要做一个ex(2022.5.10理解)
        # 限制预测结果在(0,正无穷)之间
        output[..., 2:4]    = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels,outputs,mask_outputs,mask_labels,input_images,now_epoch):
        # 需要注意的是，此时outputs是在原图上的坐标
        '''
        x_shifts:x轴的数量
        y_shifts:y轴的数量
        expanded_strides:grid对应于原图的位置
        labels:(dx,dy,w,h)真值框
        outputs:(dx,dy,w,h)网络预测结果，转换为原图对应位置之后的结果
        mask_inputs:mask预测结果
        gt_masks:mask真值
        '''
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 4]
        #-----------------------------------------------#
        # 坐标已经进行了调整，转到了原图上
        # 在原图上的预测结果
        bbox_preds  = outputs[:, :, :4]
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 1]
        #-----------------------------------------------#
        obj_preds   = outputs[:, :, 4:5]
        #-----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]
        #-----------------------------------------------#
        cls_preds   = outputs[:, :, 5:]
        # 这个是三种比例相加的结果，21504=128*128+64*64+3*32
        # outputs [batch_szie,num_anchor,4+obj+cls]
        total_num_anchors   = outputs.shape[1]
        #-----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #-----------------------------------------------#
        # 把3层的结果拼接起来成一层
        x_shifts            = torch.cat(x_shifts, 1)
        y_shifts            = torch.cat(y_shifts, 1)
        expanded_strides    = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks    = []
        sc_boxs = []
        num_fg  = 0.0

        # 分batch获取要进行loss计算的部分
        # 在这个地方进行mask计算，所有batch保存一样的结果
        # 多个batch一起进行，分batch进行损失计算
        for batch_idx in range(outputs.shape[0]):
            # num_gt为真值的数量，为了防止出现不存在真值的情况，所以要加上判断
            num_gt          = len(labels[batch_idx])
            mask_output     = mask_outputs[batch_idx]
            input_image     = input_images[batch_idx][0]
            if num_gt == 0:
                cls_target  = outputs.new_zeros((0, self.num_classes))
                reg_target  = outputs.new_zeros((0, 4))
                obj_target  = outputs.new_zeros((total_num_anchors, 1))
                fg_mask     = outputs.new_zeros(total_num_anchors).bool()
                sc_box      = outputs.new_zeros((0,4))
            else:
                #-----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, num_classes]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                #-----------------------------------------------#
                # 获取当前batch的真值和预测结果
                # 真值
                gt_bboxes_per_image     = labels[batch_idx][..., :4]
                gt_classes              = labels[batch_idx][..., 4]

                # 预测结果
                bboxes_preds_per_image  = bbox_preds[batch_idx]
                cls_preds_per_image     = cls_preds[batch_idx]
                obj_preds_per_image     = obj_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img, sc_box = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, mask_output, input_image, now_epoch,outputs
                )

                torch.cuda.empty_cache()
                num_fg      += num_fg_img
                cls_target  = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target  = fg_mask.unsqueeze(-1)
                reg_target  = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
            sc_boxs.append(sc_box)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks    = torch.cat(fg_masks, 0)
        sc_boxs     = torch.cat(sc_boxs,0)
        num_fg      = max(num_fg, 1)

        ### loss_sc_iou需要进行一下判断，如果补全之后的结果全为0，则直接返回0，不需要进行iou_loss的计算，
        # 只有当和不为0的时候才需要进行iou_loss的计算
        loss_iou    = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj    = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls    = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        loss_mask   =  CE_Loss(mask_outputs,mask_labels,num_classes=(self.num_classes+1)) + Dice_loss(mask_outputs,F.one_hot(mask_labels,(self.num_classes+1)))
        if self.is_sc==False or torch.sum(sc_boxs)==0:
            # 如果没有进行补全，则直接赋值为0，此时因为使用的是torch.zeros_like，所以requires_grad为False
            loss_sc=torch.zeros_like(loss_iou)
        else:
            sc_bbox_preds = bbox_preds.clone()
            sc_bbox_preds.data.view(-1,4)[fg_masks] = sc_boxs.data.view(-1,4)
            loss_sc = (self.iou_loss(sc_bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        reg_weight  = 5.0
        loss = (reg_weight * loss_iou + loss_obj + loss_cls+self.sc_weight*loss_sc)/num_fg+self.mask_weight*loss_mask
        return loss,[(reg_weight * loss_iou)/num_fg,loss_obj/num_fg,loss_cls/num_fg,self.mask_weight*loss_mask,(self.sc_weight*loss_sc)/num_fg]
    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts, mask_output, input_image, now_epoch,outputs):
        '''
        @param num_gt:真实存在的目标的数量
        @param total_num_anchors:三层的结果，128*128+64*64+32*32
        @param gt_bboxes_per_image:真实框
        @param gt_classes:类别
        @param bboxes_preds_per_image:网络预测结果
        @param cls_preds_per_image:网络预测类别概率
        @param obj_preds_per_image:网络预测属于目标的概率
        @param expanded_strides:
        @param x_shifts:
        @param y_shifts:
        @return:
        '''
        # fg_mask为样本选择区间(并集)，is_in_boxes_and_center为两者交集区域(交集)
        '''
        fg_mask为选择出来的样本区间，也就是黄色框和绿色框加起来的区域,其中true为样本区间，false不是样本区间
        '''
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)
        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        #-------------------------------------------------------#
        sc_boxes = bboxes_preds_per_image
        bboxes_preds_per_image  = bboxes_preds_per_image[fg_mask]

        cls_preds_              = cls_preds_per_image[fg_mask]
        obj_preds_              = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor     = bboxes_preds_per_image.shape[0]
        #-------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        #-------------------------------------------------------#
        pair_wise_ious      = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        #-------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        #-------------------------------------------------------#
        cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()

        gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)

        pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        # cos矩阵，每一个位置对应
        # fg_mask个结果
        # cost矩阵，每一个真实框对应一层
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        '''
        ①将mask预测结果转成溢油的语义分割结果
        ②根据iou，去除iou < 0.7
        的box
        ③获取对应box的语义分割结果
        ④构建marker
        ⑤对输入图像使用分水岭算法
        ⑥根据分水岭算法的结果，获取预测框
        '''
        sc_boxes=sc_boxes[fg_mask]
        if now_epoch>=0 and self.is_sc:
            mask_output = torch.argmax(F.softmax(mask_output, dim=0), dim=0).cpu().detach().numpy()
            mask_output[mask_output == 1] = 255
            mask_output = np.array(mask_output, dtype='uint8')
            # 处理iou阈值较低的框，得分较高的框就没必要进一步的处理
            threshold = 0.5
            # ①根据IOU获取框
            ix = torch.where(pred_ious_this_matching < threshold)
            len_ix = len(ix[0])
            if len_ix > 0:
                for i in ix[0]:
                    box = sc_boxes[i]
                    xyxy_box = torch.zeros_like(box)
                    xyxy_box[0] = box[0] - box[2] / 2
                    xyxy_box[1] = box[1] - box[3] / 2
                    xyxy_box[2] = box[0] + box[2] / 2
                    xyxy_box[3] = box[1] + box[3] / 2
                    # ②获取对应mask结果
                    mask_area = np.zeros_like(mask_output)
                    mask_area[int(xyxy_box[1]):int(xyxy_box[3]), int(xyxy_box[0]):int(xyxy_box[2])] = mask_output[
                                                                                                      int(xyxy_box[
                                                                                                              1]):int(
                                                                                                          xyxy_box[3]),
                                                                                                      int(xyxy_box[
                                                                                                              0]):int(
                                                                                                      xyxy_box[2])]
                    # 如果不存在语义分割的结果，那么也没有必要进行语义补全，所以要判断一下，如果只有0，则len=1，所以，当len>1的时候才进行语义补全
                    mask_area=np.array(mask_area,dtype='uint8')
                    if np.sum(mask_area)>=0:
                        # ③根据mask结果构建markers
                        markers = np.zeros_like(mask_output, dtype='uint8')
                        # 候选区域
                        markers[int(xyxy_box[1]):int(xyxy_box[3]), int(xyxy_box[0]):int(xyxy_box[2])] = 1
                        # 溢油区域
                        markers[mask_area == 255] = 2
                        # 获取原始影像
                        cpu_image = input_image.cpu().detach().numpy()
                        cpu_image = np.array(cpu_image * 255, dtype='uint8')
                        cpu_image = cv2.cvtColor(cpu_image, cv2.COLOR_GRAY2RGB, 0)
                        #分水岭
                        markers = np.array(cv2.resize(markers, (input_image.shape[0], input_image.shape[1]),
                                                      interpolation=cv2.INTER_NEAREST), dtype='int32')
                        markers = cv2.watershed(cpu_image, markers)
                        markers[markers != 2] = 0
                        markers[markers == 2] = 1
                        # 获取分水岭之后的坐标,dx,dy,w,h
                        pos = np.where(markers)
                        if pos[0].shape[0] == 0:
                            xmin = 0
                            ymin = 0
                            xmax = 0
                            ymax = 0
                        else:
                            xmin = np.min(pos[1])
                            ymin = np.min(pos[0])
                            xmax = np.max(pos[1])
                            ymax = np.max(pos[0])
                        # 更新坐标
                        sc_boxes[i][0] = xmin + (xmax - xmin) / 2
                        sc_boxes[i][1] = ymin + (ymax - ymin) / 2
                        sc_boxes[i][2] = xmax - xmin
                        sc_boxes[i][3] = ymax - ymin
                    else:
                        sc_boxes[i][0]=0
                        sc_boxes[i][1]=0
                        sc_boxes[i][2]=0
                        sc_boxes[i][3]=0
        else:
            sc_boxes=outputs.new_zeros((0,4))
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg, sc_boxes

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):

        # -------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        # -------------------------------------------------------#

        expanded_strides_per_image = expanded_strides[0]
        # 每个真值对应一些列框
        # 将预测结果给放到原图上
        # 把每个点的结果进行换算，换到原图上的中心点的位置，并且有多少个真实框，就复制多少个，相当于每个真实框对应这么多
        # x_centers_per_image：
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # -------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        # 真实框的左上角和右小角
        # (dx,dy,w,h)转换为(x1,x2,y1,y2),并且进行复制，和anchor数量一致
        # x1

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        # x2
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        # y1
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        # y2
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)

        # -------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # 这一步每个中心点所对应的真值框
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # -------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        # -------------------------------------------------------#
        # 只有坐标结果都大于0，才说明在真值框范围内
        # 4个值中的最小值大于0，说明对应着真实点
        # 去除小于0的点，说明不在真实框范围内
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # 3个真值框所满足的anchor
        # 每个anchor满足3个比例的
        # 此处计算sum的意思是，4个条件都满足，都大于0，3不是指3层，而是说3个框的4个参数

        # ！！！ torch.sum,只要有True，结果就大于0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # 以真值框的中心点为中心，2.5为半径，选
        # 因为有3种缩放比例，所以，要扩充到和所有anchor一样的数量，然后再进行计算，比例不同
        # 求出左上和右下坐标
        # 黄色框的坐标
        # gt_bboxes_per_image(dx,dy,w,h)
        # 三种比例的半径，2.5*8=20,2.5*16=40,2.5*32=80
        # 如果改中心的话，需要在这个地方改
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)
        # -------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        # 每个点，相对于黄色框的位置
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # -------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        # -------------------------------------------------------#
        # 只有在这个范围内的才会要
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # -------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        # -------------------------------------------------------#

        # 并集
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # 取两者交集
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]

        # is_in_boxes_anchor：哪些anchor是符合要求的，也就是黄绿交接的地方
        # is_in_boxes_and_center:在符合要求的情况，也就是处于黄绿交接的地方的时候，对应着哪一个真值框

        # 注意：这个地方保存的是点，而不是坐标，所以如果显示出来看，会和相像的结果有1-2个像素的误差，基本上是1个像素

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # -------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        # -------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)
        # ------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        # ------------------------------------------------------------#
        # 把候选框的数量限制在10个以内，主要是为了特别小的候选框，这个时候，n_candidate_k可能只有1或者2，一个很小的值
        n_candidate_k = min(10, pair_wise_ious.size(1))
        # iou最高的n_candidate_k
        # 选取每个真实框所对应的前n_candidate_k个候选框的iou值
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)

        # 限制在1-无穷大之间，至少要有一个
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        # 计算每一个真实框对应的n_candidate_k个iou的总和，并将其限制在1-无穷大范围内
        # print('dynamic_ks',dynamic_ks,topk_ious.sum(1).int())

        for gt_idx in range(num_gt):
            # ------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            # ------------------------------------------------------------#
            # 相当于为了获取最小值
            '''
            ①cost[gt_idx]，获取当前框的cost矩阵
            ②k获取选取多少个
            ③选取cost矩阵中最小的前k个
            ④pos_idx为索引值
            ⑤将该点负值为0，表示由这个点来预测图像的中心点
            '''
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # ------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        # ------------------------------------------------------------#
        # 判断是否存在一对多问题，即一个点预测多个目标
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # ------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            # ------------------------------------------------------------#
            '''
            ①当存在一对多问题是，先获取对应位置，cost矩阵中最小的那个值的索引
            ②将一对多的点所有值负值为0
            ③将该点给予cost值最小的那个框
            '''
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        # ------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        # ------------------------------------------------------------#
        # 查看一共有多少个特征点
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        # ------------------------------------------------------------#
        #   对fg_mask进行更新
        # ------------------------------------------------------------#
        # 更新fg_mask，由原来的

        # print("看一下大小",fg_mask.shape,fg_mask_inboxes.shape)
        # 更新的意义不明，后面也没有使用该参数！！！
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # ------------------------------------------------------------#
        #   获得特征点对应的物品种类
        # ------------------------------------------------------------#
        # 获取特征点对应哪个真实框
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 获取该框对应的类别
        gt_matched_classes = gt_classes[matched_gt_inds]

        # 获取对应的iou值大小
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        # 特征点的个数，对应的真实类别，对应的iou，对于哪个真实框的索引
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)

