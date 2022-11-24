from random import sample, shuffle
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input
import gdal
from utils.utils import resize_tif,tif_flip,resize_tif_nopaste,tif_paste,mask_paste,tif_read,mask_paste_by_scale
import os

class Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, class_names,mosaic_ratio = 0.5):
        super(Dataset, self).__init__()
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train
        self.mosaic_ratio       = mosaic_ratio
        self.epoch_now          = -1
        self.length             = len(annotation_lines[0])
        self.class_names=class_names

        self.annotation_names=annotation_lines[0]
        self.annotation_image_path=annotation_lines[1]
        self.annotation_mask_path = annotation_lines[2]

        self.imgs_path_list = []
        self.masks_path_list = []
        # 加载数据，把数据的路径给拼接起来
        for img_name in annotation_lines[0]:
            if img_name.endswith('tif'):
                self.imgs_path_list.append(os.path.join(annotation_lines[1],img_name))
                self.masks_path_list.append(os.path.join(annotation_lines[2],img_name.split(".")[0]+".png"))
        self.annotation_lines=[self.imgs_path_list,self.masks_path_list]
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic:
            if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                lines = sample(self.annotation_names, 3)
                mosaic_lines=[]
                for i in range(len(lines)):
                    mosaic_lines.append([os.path.join(self.annotation_image_path, lines[i]),
                                         os.path.join(self.annotation_mask_path, lines[i].split(".")[0] + ".png")])
                mosaic_lines.append([self.annotation_lines[0][index],self.annotation_lines[1][index]])
                shuffle(lines)
                image, mask, box  = self.get_random_data_with_Mosaic(mosaic_lines, self.input_shape)
            else:
                image, mask, box  = self.get_random_data([self.annotation_lines[0][index],self.annotation_lines[1][index]], self.input_shape, random = self.train)
        else:
            image, mask,box      = self.get_random_data([self.annotation_lines[0][index],self.annotation_lines[1][index]], self.input_shape, random = self.train)
        image       = preprocess_input(image)
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        mask[mask>1]=1
        return image, mask, box
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    def get_mask_box_from_mask(self,mask):
        mask=np.array(mask)
        # 重新编号
        old_obj=np.unique(mask)
        for i in range(len(old_obj)):
            mask[mask==int(old_obj[i])]=int(i)
        # 获得所有id
        obj_ids = np.unique(mask)
        # 舍弃0这个背景值
        obj_ids = obj_ids[1:]
        mask=np.array(mask)
        onehot_mask = mask[None, ...] == obj_ids[:, None, None]
        box = []
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(onehot_mask[i])
            xmin = np.min(pos[1])
            ymin = np.min(pos[0])
            xmax = np.max(pos[1])
            ymax = np.max(pos[0])
            box.append(np.array([xmin, ymin, xmax, ymax, 0]))
        mask = np.asarray(mask, dtype=np.uint8)
        box = np.array(box)
        return mask,box
    def get_random_data(self, annotation_line, input_shape, jitter=.3, random=True):
        # 路径
        img_line=annotation_line[0]
        mask_line=annotation_line[1]
        # 数据
        image=tif_read(img_line)
        mask=Image.open(mask_line)
        # 输入数据基础信息
        ic, ih, iw = image.shape
        h, w = input_shape
        # 不随机增强
        if not random:
            scale = min(w / iw, h / ih)
            # 缩放之后的宽和高
            nw = int(iw * scale)
            nh = int(ih * scale)
            # 需要0填充的大小
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            # 缩放图片
            image_data = resize_tif(image, ic, w, h, nh, nw, dx, dy)
            # 缩放真值
            mask_data=mask_paste_by_scale(mask,w,h,nw,nh,dx,dy)
            # 根据真值获得标签
            mask_data,box=self.get_mask_box_from_mask(mask_data)
            # 对真实框进行调整
            # box处理完，还是(x1,y1,x2,y2)的坐标，只不过换到了缩放之后的图片上
            # 并且进行了坐标的限制，防止超出图片
            # 长宽必须大于1，过小的舍弃
            if len(box) > 0:
                np.random.shuffle(box)
                # 限定坐标框的范围在图片范围内
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                # 获得宽和高
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            return image_data,mask_data,box
        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        # 缩放图片
        image=resize_tif(image,ic,h,w,nh,nw,dx,dy)
        # 缩放真值
        mask=mask_paste_by_scale(mask,w,h,nw,nh,dx,dy)
        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image=tif_flip(image,ic,h,w)
            mask  = Image.fromarray(mask).transpose(Image.FLIP_LEFT_RIGHT)
        mask,box=self.get_mask_box_from_mask(mask)
        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            # 如果旋转的话，坐标也要旋转
            # 限定坐标范围
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            # 长宽小于1的候选框不要
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image,mask,box

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        image_datas = []
        mask_datas = []
        index = 0
        # 默认通道数为5，具体为多大，需要通过读取来进行修改，如果不读取，默认为5
        ic=5
        for line in range(len(annotation_line)):
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = annotation_line[line]
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            dataset = gdal.Open(line_content[0])
            image = dataset.ReadAsArray()
            del dataset
            if image.ndim < 3:
                image = image[None, ...]
            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            ic,ih, iw = image.shape
            # ---------------------------------#
            #   获取框的位置
            # ---------------------------------#
            mask = Image.open(line_content[1])
            _, box = self.get_mask_box_from_mask(mask)
            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = tif_flip(image, ic, ih, iw)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = resize_tif_nopaste(image, ic, nh,nw)
            mask = np.array(mask.resize((nw, nh), Image.NEAREST))

            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh
            # 拼接image
            new_image = np.zeros((ic,h,w),dtype='float32')
            new_image = tif_paste(new_image,image,dx,dy)
            image_data = new_image
            # 拼接mask
            new_mask = np.zeros((h,w),dtype='uint8')
            temp_mask_count=[]
            if len(mask_datas)!=0:
                for i in range(len(mask_datas)):
                    temp_mask_count.append(np.max(mask_datas[i]))
                mask_count=max(temp_mask_count)
            else:
                mask_count=0
            new_mask =mask_paste(new_mask,mask,mask_count,dx,dy)
            mask_data=new_mask
            index = index + 1
            image_datas.append(image_data)
            mask_datas.append(mask_data)
        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([ic,h, w])
        new_image[:, :cuty, :cutx] = image_datas[0][:, :cuty, :cutx]
        new_image[:, cuty:, :cutx] = image_datas[1][:, cuty:, :cutx]
        new_image[:, cuty:, cutx:] = image_datas[2][:, cuty:, cutx:]
        new_image[:, :cuty, cutx:] = image_datas[3][:, :cuty, cutx:]
        new_image = np.array(new_image, dtype='float32')

        new_mask = np.zeros([h, w])
        new_mask[:cuty, :cutx] = mask_datas[0][:cuty, :cutx]
        new_mask[cuty:, :cutx] = mask_datas[1][cuty:, :cutx]
        new_mask[cuty:, cutx:] = mask_datas[2][cuty:, cutx:]
        new_mask[:cuty, cutx:] = mask_datas[3][:cuty, cutx:]
        new_mask = np.array(new_mask, dtype='uint8')

        # 获取框
        new_mask,new_boxes=self.get_mask_box_from_mask(new_mask)

        # mask = (new_mask - np.min(new_mask)) / (np.max(new_mask) - np.min(new_mask))
        # mask = np.array(mask * 255, dtype='uint8')
        # mask = Image.fromarray(mask)
        #
        # image = (new_image[0] - np.min(new_image[0])) / (np.max(new_image[0]) - np.min(new_image[0]))
        # image = np.array(image * 255, dtype='uint8')
        # image = Image.fromarray(image)

        # drawImage(mask,new_boxes)
        # drawImage(image,new_boxes)


        return new_image,new_mask,new_boxes


def yolo_mask_dataset_collate(batch):
    images = []
    masks = []
    bboxes = []
    for img, mask, box in batch:
        images.append(img)
        masks.append(mask)
        bboxes.append(box)
    images = np.array(images)
    masks=np.array(masks)
    return images, masks,bboxes

