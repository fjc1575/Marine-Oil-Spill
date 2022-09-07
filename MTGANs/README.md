# MTGANs

### Official pytorch implementation of the paper: "Multi-task GANs for Oil Spill Classification and Semantic Segmentation Based on SAR Images". 


##Oil Spill Classification and Semantic Segmentation
MTGANs can achieve both oil spill classification and oil spill area segmentation




## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.8 


###  Classification model training
To train MTGANs model on your own training images, put the training images under Input/train, and run

```
python ClassifyGan.py
```
###  Segmentation model training
To train MTGANs model on your own training images, put the training images under Input/TrainingSet, and run

```
python Segmentation_train.py
```
### Classification model test
To test Classification model on your own test images, put the test images under Input/test, and run

```
python ClassifyTest.py
```
### Segmentation model test
To test Segmentation model on your own test images, put the test images under Input/TestSet, and run

```
python test.py
```


### Overall test
To Overall test on your own test images, put the test images under  Input/test, and run

```
python AllTest.py
```

### The trained model
Download link of the trained oil spill classifier and generator model for oil spill segmentation

```
链接：https://pan.baidu.com/s/1zURvhA0PZ0L79DD9fVDkNg 提取码：8888

```
