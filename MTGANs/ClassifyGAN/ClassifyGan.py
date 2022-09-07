import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from dataset import *
import matplotlib.pyplot as plt

#os.makedirs("images", exist_ok=True)

n_epochs=300
batch_size=64
lr=0.0002
b1=0.5
b2=0.999
latent_dim=200
num_classes=2
img_size=64
channels=256#1
sample_interval=400
ds_size = img_size // 2 ** 4

cuda = True if torch.cuda.is_available() else False

# Initialize the weight
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#Build the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

#Build discriminator/classifier
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        #
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16,bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of the downsampled image
        ds_size = img_size // 2 ** 4
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())#distinguish the true from the false
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes + 1), nn.Softmax())#classify

    def forward(self, img):
        out = self.conv_blocks(img)
        #print('chankan2=out',out.shape)
        out = out.view(out.shape[0], -1)#torch.Size([64, 512])
        #print('chankan1=out',out.shape)
        validity = self.adv_layer(out)#true and false
        label = self.aux_layer(out)#classification
        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize G and D
G = Generator()
D= Discriminator()


# optimizer
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

if cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize the weight
G.apply(weights_init_normal)
D.apply(weights_init_normal)





#Load the oil spill dataset  2022/07/6 LiuChuan
def data():
    dataloader = loaddataset()
    return dataloader

#print(type(dataloader))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ----------
#  Training
# ----------
#Add Variable 2022/07/6 LiuChuan
accuracy=[]
epochs=[]#
# Training function
def train(dataloader):
    for epoch in range(n_epochs):
        epochs.append(epoch+1)
        for i, batch_data in enumerate(dataloader):
            #print(type(imgs))#'torch.Tensor'
            batch_size = batch_data['image'].shape[0]  # Shape [0] is set to imgs.shape[0] each time. If this parameter is not set, the value of imgs.shape[0]=32 is not equal to batch_size


            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)    #Define real label is 1
            #print('labelvalid',valid)

            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)     # Define false label is 0
            #print('labelfake', fake)

            fake_aux_gt = Variable(LongTensor(batch_size).fill_(num_classes), requires_grad=False)  # Defining category labels

            #print('labelfake_aux_gt', fake_aux_gt)

            real_imgs = Variable(batch_data['image'].type(FloatTensor))    # real image
            labels = Variable(batch_data['label'].type(LongTensor))      # Actual label
            #print('labels',labels)


            # Generator for training
            optimizer_G.zero_grad()
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))) # 产生随机噪声

            gen_imgs = G(z) # The generator generates fake images
            #print('gen_imgs',gen_imgs.shape)
            validity, _ = D(gen_imgs)
            #print('validity',validity)
            g_loss = adversarial_loss(validity, valid)  # Generator loss
            #print(g_loss)

            g_loss.backward()
            optimizer_G.step()

            # Training discriminator
            optimizer_D.zero_grad() # Loss of real images and real classification labels


            real_pred, real_aux = D(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss of false images and real classification labels
            fake_pred, fake_aux = D(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2


            d_loss.backward()
            optimizer_D.step()

            # Calculating classification accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)   # 128*11,前64列为真实图像经过判别器的判断结果,后64列为虚假图像经过判别器的判断结果
            #print('Discriminant result',np.argmax(pred,axis=1))
            gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)    # 128列,前64列为真实标签,后64列为虚假标签10   真实标签0～9  虚假标签为10
            #print('Actual label',gt)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)  # accuracy rate


            #Output the result at each epoch
            if i+1==len(dataloader):
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                      % (epoch, n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
                      )
                accuracy.append(d_acc)


            batches_done = epoch * len(dataloader) + i


        if epoch+1 == n_epochs:
            # saved in TrainedModels Save one last time
            #torch.save(G, '../TrainedModels/G.pth')
            #Just keep the discriminator
            torch.save(D, '../TrainedModels/DorC.pth')
            #print('Model saved successfully')
#
# print(accuracy)
# print(epochs)
    #curve plotting  2022/07/7 LiuChuan
    plt.title('Result Analysis')
    plt.plot(epochs, accuracy, color='red', label='training accuracy')
    plt.title("training accuracy", fontsize=14)
    # Label the X axis and set the font size
    plt.xlabel("epoch", fontsize=14)
    # Label the Y-axis and set the font size
    plt.ylabel("accuracy", fontsize=14)
    #Open the Matplotlib viewer and display the drawn graph
    plt.show()

if __name__ == '__main__':
    dataloader=data()
    train(dataloader)
    #print('zhixingl train')








