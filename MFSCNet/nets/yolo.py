#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import cv2
import numpy as np
import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024],up_size=[8,16,32], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()

        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )
    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)
            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs



class YOLOXMask(nn.Module):
    def __init__(self, num_classes, width=1.0, act="silu",depthwise=False):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.decoder3_upsample=nn.ConvTranspose2d(int(256 * width),int(128 * width),2,stride=2)
        self.decoder3_conv = nn.Sequential(*[
            Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            Conv(in_channels=int(256 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act),
        ])

        self.decoder2_upsample = nn.ConvTranspose2d(int(128 * width), int(64 * width), 2, stride=2)
        self.decoder2_conv = nn.Sequential(*[
            Conv(in_channels=int(128 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act),
            Conv(in_channels=int(128 * width), out_channels=int(64 * width), ksize=3, stride=1, act=act),
        ])
        self.mask_preds =nn.Sequential(*[
            nn.ConvTranspose2d(int(64 * width),int(64 * width),2,stride=2),
            Conv(in_channels=int(64 * width), out_channels=int(32 * width), ksize=3, stride=1, act=act),
            nn.Conv2d(in_channels = int(32 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
        ])
    def forward(self, inputs):
        C1,C2,P3=inputs
        P3_upsample=self.decoder3_upsample(P3)
        P3_cat=torch.cat([P3_upsample,C2],1)
        P3_conv=self.decoder3_conv(P3_cat)

        P2_upsample=self.decoder2_upsample(P3_conv)
        P2_cat=torch.cat([P2_upsample,C1],1)
        P2_conv=self.decoder2_conv(P2_cat)
        mask_outputs=self.mask_preds(P2_conv)
        return mask_outputs



class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("C1","C2","C3", "C4", "C5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features
        self.upsample  = nn.Upsample(scale_factor=2)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [C1,C2,C3,C4,C5]   = [out_features[f] for f in self.in_features]
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(C5)

        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, C4], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)
        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, C3], 1)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out)
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out   = self.C3_n3(P3_downsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)
        # C1、C2为网络输出结果，用于语义分割部分
        # P3_out、P4_out、 P5_out为三层输出结果，用于YOLOXHead
        # P3_out用于语义分割
        return (C1, C2, P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        # 定义模型大小
        # depth_dict决定模型的深度:卷积的次数
        # width_dict决定模型的宽度:卷积的通道数
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 
        # 主干特征提取网络
        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        # YOLOX Head:获取定位和分类结果
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)
        # YOLOX Mask:获取语义分割结果
        self.Mask       = YOLOXMask(num_classes+1,width,depthwise=depthwise)
    def forward(self, x):
        # 特征提取结果，返回(C1,C2,P3,P4,P5)
        fpn_outs    = self.backbone.forward(x)
        # (P3,P4,P5)用于定位和分类
        # outputs输入为三层，输出也为三层的结果
        outputs = self.head.forward(fpn_outs[2:])
        # (C1,C2,P3)用于语义分割
        mask_outputs =self.Mask.forward(fpn_outs[:3])
        return outputs,mask_outputs



