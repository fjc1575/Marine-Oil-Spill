import warnings
warnings.filterwarnings("ignore")
from ClassifyGAN.ClassifyGan import *
from torch.autograd import Variable
from dataset import *
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#Load the saved model discriminator
pthfile2 = r'../TrainedModels/DorC.pth'
D= torch.load(pthfile2)
#Load the data set for the test
def data():
    testdataset = Testdataset()
    return testdataset
#Test the classification accuracy of the discriminator
def testClassify(dataloader):
    for i, batch_data in enumerate(dataloader):
        img_path=batch_data['img_path']#Obtain image path
        real_imgs = Variable(batch_data['image'].type(FloatTensor))  # real image
        #print(real_imgs)
        labels = Variable(batch_data['label'].type(LongTensor))  # Actual label
        real_pred, real_aux = D(real_imgs)
        #print('real_aux:',[real_aux.data.cpu().numpy()])
        gt = np.concatenate([labels.data.cpu().numpy()], axis=0)  # Real labels 0 and 1 0: indicates suspected oil spill 1: indicates oil spill image False label 2

        print('Actual label',gt)
        pred = np.concatenate([real_aux.data.cpu().numpy()], axis=0)  # Is the judgment result of the real image after the discriminator, 0: represents the suspected oil spill 1: represents the oil spill image false label is 2
        print('result',np.argmax(pred, axis=1))
        predlabels=np.argmax(pred, axis=1)

        acc = np.mean(np.argmax(pred, axis=1) == gt)  # Calculate the classification accuracy
        print('准确度accuracy:',acc)
    # The output discriminates as the path of oil spill image
    for j in range(predlabels.size):
        #Both the real image and the predicted value are oil spills
        if predlabels[j]==gt[j] and gt[j]==1 :
            #Capture both real and predicted images of oil spills
            print(img_path[j],j,'Real oil spill')
            image = cv2.imread(img_path[j])
            # display image
            cv2.imshow("img is oil", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    #Load the test dataset function
    dataloader=data()

    #Test the discriminator function
    testClassify(dataloader)
