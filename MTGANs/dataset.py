from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2

import torch


def get_images_and_labels(dir_path):

    dir_path = Path(dir_path)
    classes = []  # List of Categories

    for category in dir_path.iterdir():
        if category.is_dir():
            classes.append(category.name)
    images_list = []  #
    labels_list = []  # tag list
    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.jpg'):
            images_list.append(str(img_path))
            labels_list.append(int(index))
    return images_list, labels_list




class MyDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path  # The root directory of the dataset
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)
        #print(self.images,self.labels)

    def __len__(self):
        # The amount of data to return in the dataset
        return len(self.images)

#Add the image path property
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # To add a crop to the image

        if img.shape[1]==512 :
            w = img.shape[1]
            h = img.shape[0]
            img = img[0:h, 0:256, :]
        else:
            img=img

        sample = {'image': img, 'label': label,'img_path':img_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

#Add the training set function
def loaddataset():
     #Path to load the training set
     train_dataset = MyDataset(r'../Input/train')#r'../TrainedModels/model/DorC.pth'
     #Loading the training set
     dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
     return dataloader

#Add test set functions
def Testdataset():
    #Test set path
    test_dataset = MyDataset(r'../Input/test')
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    return dataloader



if __name__ == '__main__':
    #Load the training dataset
    loaddataset()
    #Load the test dataset
    Testdataset()
