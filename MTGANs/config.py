import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default = 0)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default = 3)
    parser.add_argument('--nc_im',type=int,help='image # channels',default = 3)
    parser.add_argument('--out',help='output folder',default='TestResult')  # output folder
        
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default = 32)  #nfc: num of ker  32->64
    parser.add_argument('--min_nfc', type=int, default = 32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default = 3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default = 5)
    parser.add_argument('--stride',help='stride',default = 1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default = 1)
        
    #pyramid parameters:
    parser.add_argument('--scale_num', type=float, help='pyramid scale num', default = 0)
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default = 0.5)
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=1) # noise_amp
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default = 25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default = 250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default = 1, help='number of epochs to train each scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default = 1)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default = 1)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default = 10) #lambda_grad
    parser.add_argument('--alpha',type=float, help='l1 norm weight',default=10)  # original 10

    return parser
