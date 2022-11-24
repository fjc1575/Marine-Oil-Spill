import numpy as np
import os
from PIL import Image
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
if __name__=="__main__":
    pred_path="mask_result"
    gt_path=r"G:\TGRS\1.Data\De\Test\masks/"
    count=[0,0,0,0,0,0]
    name_lister=os.listdir(pred_path)
    for name in name_lister:
        # 读取图片
        pred_image=Image.open(os.path.join(pred_path,name))
        gt_image=Image.open(os.path.join(gt_path,name))
        # 变成二值图
        pred_image=np.array(pred_image)
        gt_image=np.array(gt_image)
        pred_image[pred_image>=1]=1
        gt_image[gt_image>=1]=1

        pred_image=pred_image.reshape(-1,1).squeeze()
        gt_image=gt_image.reshape(-1,1).squeeze()

        cm = confusion_matrix(gt_image, pred_image)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        oa=(tp+tn)/(tp+tn+fp+fn)
        if (tp+fn)==0:
            recall=0
        else:
            recall = tp / (tp + fn)
        if (tp+fp)==0:
            precision=0
        else:
            precision = tp / (tp + fp)

        if (precision+recall)==0:
            f1score=0
        else:
            f1score=(2*precision*recall)/(precision+recall)
        miou=(1/2)*((tp/(tp+fp+fn))+(tn/(tn+fn+fp)))
        pe=((tn+fp)*(tn+fn)+(tp+fp)*(tp+fn))/((tp+tn+fp+fn)**2)
        kappa=(oa-pe)/(1-pe)
        count[0] += oa
        count[1] += recall
        count[2] += precision
        count[3] += f1score
        count[4] += miou
        count[5] += kappa
        print(oa,recall,precision,f1score,miou,kappa)
    for i in range(len(count)):
        count[i]=count[i]/len(name_lister)
    print(count)




