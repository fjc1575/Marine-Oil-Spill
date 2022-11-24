# MFSCNet

### Official pytorch implementation of the paper: "Multi-Feature Semantic Complementation Network for Marine Oil Spill Localization and Semantic Segmentation Based on SAR Images". 

## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6.13 

### Training

To train MFSCNet model on your own training images, run

```
python train.py  --img_flooder 'image path' --mask_flooder 'mask path'
```

### Predict

To test model on your own images, run

```
python predict.py
```

The predict sample images are in the folder of 'img' 

### The trained model

Download link of the model for oil spill detection

```
链接：https://pan.baidu.com/s/1s0c_dZDBMYWudyHtgy2fqQ 提取码：1234
```

## Reference
```
https://github.com/bubbliiiing/yolox-pytorch
https://github.com/Megvii-BaseDetection/YOLOX
```