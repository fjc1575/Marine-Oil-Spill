import os
from PIL import Image
from tqdm import tqdm
from yolo import YOLO
from utils.utils import get_classes,tif_read
from utils.utils_map import get_map
from config import HyperParameter
import numpy as np
import shutil
def dataset_get(args):
    imglist = os.listdir(args.test_img_flooder)
    lines=[imglist,args.test_img_flooder,args.test_mask_flooder]
    nums=len(lines[0])
    return lines,nums
def get_mask_box_from_mask(mask):
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
if __name__ == "__main__":
    '''
    本方法通过mask进行map的计算
    '''
    if True:
        # 创建文件夹用于存储预测结果和真值
        # 会先删除旧文件，再创建新的文件进行预测
        map_out_path = 'map'
        if os.path.exists("map\detection-results"):
            shutil.rmtree("map\detection-results")
        if not os.path.exists(map_out_path):
            os.makedirs(map_out_path)
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth'))
        if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
            os.makedirs(os.path.join(map_out_path, 'detection-results'))
        if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
            os.makedirs(os.path.join(map_out_path, 'images-optional'))
    args=HyperParameter()
    lines,nums=dataset_get(args)
    classes_path = 'model_data/oil_classes.txt'
    MINOVERLAP = 0.5
    class_names, _ = get_classes(classes_path)
    model_path=r"pth/mfscnet.pth"
    if True:
        print("------------------Get predict result------------------")
        print("1.Load model.")
        yolo = YOLO(confidence=0.001, nms_iou=0.5,model_path=model_path)
        print("2.Load model done.")
        for image_id in tqdm(lines[0]):
            image_id=image_id.split(".")[0]
            image_path = os.path.join(lines[1], image_id + ".tif")
            image=tif_read(image_path)
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("------------------Get predict result done------------------")
    if True:
        print("------------------Get ground truth result------------------")
        for image_id in tqdm(lines[0]):
            image_id = image_id.split(".")[0]
            mask=np.array(Image.open(os.path.join(lines[2],image_id+".png")))
            _, boxes = get_mask_box_from_mask(mask)
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for i in range(boxes.shape[0]):
                    left     = boxes[i][0]
                    top      = boxes[i][1]
                    right    = boxes[i][2]
                    bottom   = boxes[i][3]
                    obj_name = 'oil'
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("------------------Get ground truth result done------------------")
    if True:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")