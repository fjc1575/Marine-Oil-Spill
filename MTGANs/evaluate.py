from PIL import Image
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
from sklearn import metrics
import numpy as np

import matplotlib.pyplot as plt
#mg = Image.open(r'C:\Users\17578\Desktop\数据\data2\TestResult\TestResult_model\test7_0.7.jpg')  # 打开图片
img = Image.open('TestResult/TestResult_model/test12_0.7.jpg')  # 打开图片 生成的结果图
#img= img.resize((256, 256), Image.ANTIALIAS)
img2=Image.open('Input/TestSet/test12.jpg')  # 打图片 原图
w = img2.width       #图片的宽
h = img2.height      #图片的高
box = (w*0.5,0,w,h)
img2 = img2.crop(box)

i=0
j=0
m=0

lst1 = []
lst2 = []
for h in range(img.size[1]):
    for w in range(img.size[0]):  # img.size 会显示图片的宽度和高度值，由于我们要一行一行的来，所以要将 w 放在内层循环。
        gray_value = img.getpixel((w, h))  # 获取每一个像素的灰度值
        if gray_value == (255,255,255):  # 0代表溢油    (0,0,0)黑，(255,255,255)是白
            gray_value=1
        else:
            gray_value=0
        lst1.insert(i, gray_value)
        i=i+1
#print(lst1)

for h in range(img2.size[1]):
    for w in range(img2.size[0]):
        gray_value2 = img2.getpixel((w, h))
        if gray_value2 ==(255,255,255):  # 如果灰度值为0，代表溢油    0黑，255是白
            gray_value2=1
        else:
            gray_value2=0
        lst2.insert(j, gray_value2)
        j = j + 1

#print(lst2)


print('--------------输出性能评估指标--------------')
print('Accuracy:',accuracy_score(lst2, lst1))  #
print('Precision:',precision_score(lst2, lst1, average='macro'))  #
print('Recall:',recall_score(lst2, lst1, average='macro'))  #
print('F1:',f1_score(lst2, lst1, average='macro'))  #
'''
   kappa值，Kappa系数就经常被用于影像分类的空间一致性检验
   0.00到0.20：极低的一致性（Slight）
   0.21到0.40：一般的一致性（Fair）
   0.41到0.60：中等的一致性（Moderate）
   0.61到0.80：高度的一致性（Substantial）
   0.81到1.00：几乎完全一致（Almost Perfect）
'''
#print('混淆矩阵输出:',metrics.confusion_matrix(lst2, lst1))#混淆矩阵输出
pe_rows = np.sum(metrics.confusion_matrix(lst2, lst1), axis=0)
pe_cols = np.sum(metrics.confusion_matrix(lst2, lst1), axis=1)
sum_total = sum(pe_cols)
pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
po = np.trace(metrics.confusion_matrix(lst2, lst1)) / float(sum_total)
Kappa=(po - pe) / (1 - pe)
print('Kappa',Kappa)

accuracy = []
precision = []
recall = []
f1 = []
Kappa = []
mIoU =[]
for test_num in range(1, 21,1):  # the index of test images
    img = Image.open('TestResult/TestResult_model/test%s_0.7.jpg' % (test_num))  # 打开图片 生成的结果图
    # img= img.resize((256, 256), Image.ANTIALIAS)
    img2 = Image.open('Input/TestSet/test%s.jpg' % (test_num))  # 打图片 原图

    w = img2.width  # 图片的宽
    h = img2.height  # 图片的高
    box = (w * 0.5, 0, w, h)
    img2 = img2.crop(box)
    print('------------计算第%s张图像------------'% (test_num))
    i = 0
    j = 0
    m = 0
    lst1 = []
    lst2 = []
    for h in range(img.size[1]):
        for w in range(img.size[0]):  # img.size 会显示图片的宽度和高度值，由于我们要一行一行的来，所以要将 w 放在内层循环。
            gray_value = img.getpixel((w, h))  # 获取每一个像素的灰度值
            if gray_value == (255, 255, 255):  # 0代表溢油    (0,0,0)黑，(255,255,255)是白
                gray_value = 1
            else:
                gray_value = 0
            lst1.append(gray_value)

    # print(lst1)

    for h in range(img2.size[1]):
        for w in range(img2.size[0]):
            gray_value2 = img2.getpixel((w, h))
            if gray_value2 == (255, 255, 255):  # 如果灰度值为0，代表溢油    0黑，255是白
                gray_value2 = 1
            else:
                gray_value2 = 0
            lst2.append(gray_value2)

    accuracy.append(accuracy_score(lst2, lst1))
    print('accuracy_score',accuracy_score(lst2, lst1))
    precision.append(precision_score(lst2, lst1, average='macro'))
    print('precision_score', precision_score(lst2, lst1, average='macro'))
    recall.append(recall_score(lst2, lst1, average='macro'))
    print('recall_score', recall_score(lst2, lst1, average='macro'))
    f1.append(f1_score(lst2, lst1, average='macro'))

    print('f1_score', f1_score(lst2, lst1))
    pe_rows = np.sum(metrics.confusion_matrix(lst2, lst1), axis=0)

    pe_cols = np.sum(metrics.confusion_matrix(lst2, lst1), axis=1)

    sum_total = sum(pe_cols)

    sum_total1 = pe_cols[0] + pe_cols[1]

    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(metrics.confusion_matrix(lst2, lst1)) / float(sum_total)
    K = (po - pe) / (1 - pe)
    Kappa.append(K)
    print('Kappa', K)
    Per_class_IoU = np.diag(metrics.confusion_matrix(lst2, lst1)) / (
            np.sum(metrics.confusion_matrix(lst2, lst1), axis=1) + np.sum(metrics.confusion_matrix(lst2, lst1), axis=0) -
            np.diag(metrics.confusion_matrix(lst2, lst1)))
    MIoU = np.nanmean(Per_class_IoU)  # 跳过0值求mean
    print('MIoU', MIoU)
    mIoU.append(MIoU)
#计算平均性能
i=0
b = len(accuracy)
print(b)
print(accuracy)
print(precision)
#print(recall)
print(f1)
print(Kappa)
print(mIoU)
sum = 0
for i in accuracy:
    sum = sum + i
print("maccuracy:",sum / b)
b = len(precision)
sum = 0
for i in precision:
    sum = sum + i
print("mprecision:",sum / b)

b = len(recall)
sum = 0
for i in recall:
    sum = sum + i
print("mrecall:",sum / b)

b = len(f1)
sum = 0
for i in f1:
    sum = sum + i
print("mf1:",sum / b)

b = len(Kappa)
sum = 0
for i in Kappa:
    sum = sum + i
print("mKappa:",sum / b)

b = len(mIoU)
sum = 0
for i in mIoU:
    sum = sum + i
print("mIoU:",sum / b)
