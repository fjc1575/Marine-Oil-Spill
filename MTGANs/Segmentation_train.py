from config import get_arguments
from SegmentationGAN.manipulate import *
from SegmentationGAN.training import *
import SegmentationGAN.functions as functions

if __name__ == '__main__':

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/TrainingSet')
    parser.add_argument('--input_name', help='input image name',  default="train1.jpg")  # input train image
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)   # opt
    dir2save = functions.generate_dir2save(opt)
    train_range = range(1, 5)  #the range of training images
    epoch_num = 30 #Set the epoch of training
    train_loopnum = epoch_num * len(train_range)
    train_i = 0  # the index of all training process
    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

    for epoch in range(0, epoch_num):
        train_index = 0  # the index in each epoch
        for train_num in train_range:
            opt.input_name = 'train' + str(train_num) + '.jpg'  # the name of train image
            real1s = []  # real1s: oil spill observation images at multiple scales
            real2s = []  # real2s: ground truth detection maps
            real1, real2 = functions.read_image(opt)  # read the input images
            functions.adjust_scales2image(real2, opt)  #
            functions.adjust_scales2image(real1, opt)
            Gs = []
            Ds = []
            train(opt, Gs, Ds, real1s, real2s, train_range, train_index, train_i, train_loopnum)
            train_index += 1
            train_i += 1




