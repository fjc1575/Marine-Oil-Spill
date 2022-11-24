import os

import cv2
import numpy as np
from PIL import Image,ImageDraw
import gdal
import time

def loss_save(loss_list):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    loss_list=loss_list[1:]
    with open("logs/"+str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", mode='w') as f:
        for i in range(len(loss_list)):
            f.write(str(loss_list[i]) + "\n")
def tif_read(img_line):
    dataset = gdal.Open(img_line)
    image = dataset.ReadAsArray()
    del dataset
    if image.ndim < 3:
        image = image[None, ...]
    return image
def resize_image(image, size, letterbox_image):
    ic,ih,iw  = image.shape
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image_data = np.zeros((ic, h, w), dtype='float32')
        for image_channel in range(ic):
            modify_image = Image.fromarray(image[image_channel])
            modify_image = modify_image.resize((nw, nh), Image.BILINEAR)

            new_image = Image.new('L', size, 0)
            new_image = np.array(new_image, dtype='float32')
            new_image = Image.fromarray(new_image)

            new_image.paste(modify_image, ((w-nw)//2, (h-nh)//2))
            image_data[image_channel] = np.array(new_image, np.float32)
        new_image=image_data
    else:
        image_data = np.zeros((ic, h, w), dtype='float32')
        for image_channel in range(ic):
            modify_image = Image.fromarray(image[image_channel])
            modify_image = modify_image.resize((w, h), Image.BILINEAR)
            image_data[image_channel] = np.array(modify_image, np.float32)
        new_image = image_data
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    return image

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
# 缩放图片
def resize_tif(image,ic,h,w,nh,nw,dx,dy):
    '''
    image:输入的图片
    ic,ih,iw:输入图像的大小
    w,h:要修改的大小
    nh,nw,dx,dy:调整后的高、宽和中心点坐标
    '''
    new_image=np.zeros((ic,h,w),dtype='float32')
    for i in range(ic):
        # 取得第i个通道的数据，进行resize
        ic_image=Image.fromarray(image[i])
        ic_image=ic_image.resize((nw, nh), Image.BILINEAR)
        # 创建一个H*W大小的数组用来进行边缘部分的填充
        paste_image=Image.fromarray(np.array(Image.new('L',(w,h),0),dtype='float32'))
        paste_image.paste(ic_image, (dx, dy))
        # 填充
        new_image[i]=paste_image
    return new_image

def resize_tif_nopaste(image,ic,nh,nw):
    '''
    image:输入的图片
    ic,ih,iw:输入图像的大小
    w,h:要修改的大小
    nh,nw,dx,dy:调整后的高、宽和中心点坐标
    '''
    new_image=np.zeros((ic,nh,nw),dtype='float32')
    for i in range(ic):
        # 取得第i个通道的数据，进行resize
        ic_image=Image.fromarray(image[i])
        ic_image=ic_image.resize((nw, nh), Image.BILINEAR)
        # 填充
        new_image[i]=ic_image
    return new_image


def tif_paste(new_image,image,dx,dy):
    for i in range(new_image.shape[0]):
        i_new_image=Image.fromarray(new_image[i])
        i_new_image.paste(Image.fromarray(image[i]),(dx,dy))
        new_image[i]=np.array(i_new_image,dtype='float32')
    return new_image
def mask_paste_by_scale(mask,w,h,nw,nh,dx,dy):
    mask = mask.resize((nw, nh), Image.NEAREST)
    new_mask = Image.new('L', (w, h), 0)
    new_mask.paste(mask, (dx, dy))
    mask_data = np.array(new_mask, np.uint8)
    return mask_data

def mask_paste(new_mask,mask,mask_count,dx,dy):
    # 先调整mask中的值的分布
    mask=mask+mask_count
    mask[mask==np.min(mask)]=0
    new_mask=Image.fromarray(new_mask)
    new_mask.paste(Image.fromarray(mask),(dx,dy))
    return np.array(new_mask,dtype='uint8')
def tif_flip(image,ic,h,w):
    new_image=np.zeros((ic,h,w),dtype='float32')
    for i in range(ic):
        ic_image=Image.fromarray(image[i])
        ic_image=ic_image.transpose(Image.FLIP_LEFT_RIGHT)
        new_image[i]=np.array(ic_image)
    return new_image

def drawImage(image,box):
    draw=ImageDraw.Draw(image)
    for i in range(len(box)):
        x1,y1,x2,y2,_=box[i]
        draw.rectangle([(x1,y1),(x2,y2)],outline='red')
    cv2.imshow("draw",np.array(image))
    cv2.waitKey(10)
    image.show()

