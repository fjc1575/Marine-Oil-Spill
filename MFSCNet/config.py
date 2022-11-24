import argparse
def HyperParameter():
    parser = argparse.ArgumentParser(description='YOLOX With Mask Super Parameter')
    # 基础参数设置
    parser.add_argument('--Cuda', default=True, type=bool,
                        help='是否使用GPU')
    parser.add_argument('--mosaic', default=False, type=bool,
                        help='mosaci 马赛克数据增强')
    parser.add_argument('--Cosine_scheduler', default=False, type=str,
                        help='余弦退火学习率')
    parser.add_argument('--is_sc', default=True, type=str,
                        help='使用使用语义补全')
    parser.add_argument('--phi', default='s', type=str,
                        help='所使用的YoloX的版本。nano、tiny、s、m、l、x')
    parser.add_argument('--classes_path', default='model_data/oil_classes.txt', type=str,
                        help='类别路径')
    parser.add_argument('--model_path', default=r'pth/mfscnet.pth', type=str,
                        help='权重文件路径')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='多线程数据读取，加快数据读取速度，但是会占用更多内存')
    parser.add_argument('--input_shape', default=[1024, 1024], type=int,
                        help='输入的shape大小，一定要是32的倍数')
    parser.add_argument('--optim', default='adam', type=str,
                        help='激活函数选择')
    # 数据路径
    parser.add_argument('--img_flooder', default=r'', type=str,
                        help='图片路径')
    parser.add_argument('--mask_flooder', default=r'', type=str,
                        help='真值路径')
    # 数据路径
    parser.add_argument('--test_img_flooder', default=r'', type=str,
                        help='测试集图片路径')
    parser.add_argument('--test_mask_flooder', default=r'', type=str,
                        help='测试集真值路径')
    # 设置训练参数
    parser.add_argument('--Freeze_Train', default=True, type=bool,
                        help='是否冻结训练')
    parser.add_argument('--Init_Epoch', default=0, type=int,
                        help='初始Epoch')
    parser.add_argument('--Freeze_Epoch', default=0, type=int,
                        help='冻结模型训练Epoch')
    parser.add_argument('--Freeze_batch_size', default=2, type=int,
                        help='冻结模型训练期间batch_size')
    parser.add_argument('--Freeze_lr', default=1e-3, type=float,
                        help='冻结模型训练期间learning ratio')
    # 不冻结训练
    parser.add_argument('--Unfreeze_Epoch', default=100, type=int,
                        help='不冻结模型训练期间Epoch')
    parser.add_argument('--Unfreeze_batch_size', default=2, type=int,
                        help='不冻结模型训练期间batch_size')
    parser.add_argument('--Unfreeze_lr', default=1e-4, type=float,
                        help='不冻结模型训练期间learning ratio')
    args = parser.parse_args()
    return args