from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from ClassifyGAN.ClassifyGan import *
from torch.autograd import Variable
from dataset import *
import shutil
import re
from SegmentationGAN.training import *
from config import get_arguments
from SegmentationGAN.functions import *
import matplotlib.pyplot as plt
import numpy as np
from skimage import io as img
from SegmentationGAN.imresize import imresize
from PIL import Image
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
from sklearn import metrics
import numpy as np
# opt
parser = get_arguments()
parser.add_argument('--input_dir', help='input image dir', default='../Input/AllTest')
parser.add_argument('--input_name', help='input image name',  default="test0.jpg")   # input test image name
parser.add_argument('--mode', help='task to be done', default='test')
parser.add_argument('--trained_model', help='folder name of trained model', default='model')
opt = parser.parse_args()
opt = functions.post_config(opt)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



#Load the saved model discriminator
pthfile2 =  r'../TrainedModels/DorC.pth'
D= torch.load(pthfile2)

# Enter the folder name of the training model to load the generator
trained_model = opt.trained_model
# Gs
pthfile1 = r'../TrainedModels/%s/Gs.pth' % trained_model
Gs = torch.load(pthfile1)

#Load the data set for the test
def data():
    testdataset = Testdataset()
    return testdataset

#Test the classification accuracy of the discriminator and perform the classification task
def testClassify(dataloader):
    for i, batch_data in enumerate(dataloader):
        img_path=batch_data['img_path']#Obtain image path
        real_imgs = Variable(batch_data['image'].type(FloatTensor))  # real image
        #print(real_imgs)
        labels = Variable(batch_data['label'].type(LongTensor))  # Actual label
        real_pred, real_aux = D(real_imgs)
        #print('real_aux:',[real_aux.data.cpu().numpy()])
        gt = np.concatenate([labels.data.cpu().numpy()], axis=0)  # Real labels 0 and 1 0: indicates suspected oil spill 1: indicates oil spill image False label 2
        print('Test image true label',gt)
        pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)  # Is the judgment result of the real image after the discriminator, 0: represents the suspected oil spill 1: represents the oil spill image false label is 2
        print('The result of the discriminator',np.argmax(pred, axis=1))
        predlabels=np.argmax(pred, axis=1)
        acc = np.mean(np.argmax(pred, axis=1) == gt)  # Calculate the classification accuracy
        print('accuracy:',acc)
        #print('The oil spill classification task is complete.....')

    print('------Screening for the real oil spill and discrimination for oil spill images------')
    path2=r'../Input/AllTest//'
    # print(path2)
    # print(path1)
    # Reset folders are deleted and then rebuilt
    shutil.rmtree(path2)
    #Recreate the folder
    os.mkdir(path2)
   #Save both the real image and the predicted value for the oil spill image to a folder called AllTest
    for j in range(predlabels.size):
        #The real image and the predicted values are oil spill 1: oil spill 0: suspected oil spill
        if predlabels[j]==gt[j] and gt[j]==1 :
            #Capture both real and predicted images of oil spills
            image=img_path[j]

            #Copy the oil spill and real oil spill images to the new folder
            shutil.copy(os.path.join(image), os.path.join(path2))
            print(img_path[j],j,'Real oil spill')

    #Traverse the files in the directory and rename the image
    i = 1
    for images in os.listdir(path2):
         # c= os.path.basename(file)
        newimages = "test"+str(i)+".jpg"  ##Modify picture name
        os.rename(os.path.join(path2, images), os.path.join(path2, newimages))
        i += 1
   #Returns the number of oil spills that discriminates as true
    #print('The total screening oil spill image has',i-1,'image')
    return i-1
    #print(num_oil)


#The oil spill images that have been identified as true and true are placed in the AllTest folder to enter the next task. The generator is used to generate the oil spill detection image and perform the semantic segmentation task
  # Reset folders are deleted and then rebuilt
dir2save = '%s/TestResult_%s' % (opt.out, trained_model)
shutil.rmtree(dir2save)
# Recreate the folder
os.mkdir(dir2save)
print('---------------Oil spill segmentation task-----------------')
def testSegmentation(num_oil):
    i=1
    for test_num in range(1, num_oil+1, 1):  # the index of test images

        i+=1
        opt.input_name = 'test' + str(test_num) + '.jpg'  # the name of test image
        x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))  # img.imread
        x = np2torch(x, opt)
        x = x[:, 0:3, :, :]
        a = x.shape
        b = int(a[3])
        real1_ = x[:, 0:3, :, 0:int(b / 2)]  # extract left image
        ###### normalize() -> [0,1] ######
        real1_ = (real1_ - real1_.min()) / (real1_.max() - real1_.min())
        functions.adjust_scales2image(real1_, opt)
        real1 = imresize(real1_, 0, opt)
        real1s = []
        real1s = functions.creat_reals_pyramid(real1, real1s, opt)
        images_cur = None
        n = 0  # scale num
        images_cur = torch.full(real1s[0].shape, 0, device=opt.device)  # images_cur
        for G, real_in in zip(Gs, real1s):  # use each trained G
            images_prev = images_cur  # set images_prev
            g_in = torch.cat((real_in, images_prev), 1).detach()  # input of G
            I_curr = G(g_in, images_prev)  # output of G
            images_cur = []  # set as []
            images_cur = I_curr
            images_cur = imresize(images_cur, -1, opt)  # upsample
            ############################### SET opt.scale_num #######################################
            if n == opt.scale_num:  # reach the finest scale
                for threshold in range(70, 72, 2):  #
                    ################## output normalize()--> 1 ###################
                    I_curr = (I_curr - I_curr.min()) / (I_curr.max() - I_curr.min())

                    I_cu = functions.convert_image_np(I_curr.detach())

                    I_cu = (I_cu[:, :, 0] + I_cu[:, :, 1] + I_cu[:, :, 2]) / 3  # 3D --> 2D lyq210118

                    if opt.mode == 'test':
                        dir2save = '%s/TestResult_%s' % (opt.out, trained_model)
                    try:
                        os.makedirs(dir2save)
                    except OSError:
                        pass
                    ####################### POW ###########################
                    I_save = pow(I_cu, 1)  #
                    # print('I_save.shape:', I_save.shape, 'I_save',I_save)  # 256*256
                    # print('I_cu.shape:', I_cu.shape, 'I_cu', I_cu)
                    I_save[np.where(I_save <= float(threshold / 100))] = 0
                    I_save[np.where(I_save > float(threshold / 100))] = 255
                    # save the generated image
                    opt.save_name = 'test' + str(test_num) + '_' + str(
                        float(threshold / 100)) + '.jpg'  # the name of test image

                    plt.imsave('%s/%s' % (dir2save, opt.save_name), I_save, cmap='gray', vmin=0, vmax=1)

            n += 1  # n + 1, enter next scale
    print('Save the generated oil spill segmentation map', num_oil)
    print('The oil spill segmentation task is complete.....')





if __name__ == '__main__':
    #Load the test dataset function
    dataloader=data()
    #Test the discriminator function
    num_oil=testClassify(dataloader)
    #Test the semantic segmentation function
    testSegmentation(num_oil)


