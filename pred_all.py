import math
import os
if os.name == 'nt':
    import platform
    # 修改为实际的目录路径
    OSGEO4W = r"D:\anaconda3\envs\pytorch\Library"
    # 移除不必要的64位判断和路径拼接
    assert os.path.isdir(OSGEO4W), "Directory does not exist: " + OSGEO4W
    os.environ['OSGEO4W_ROOT'] = OSGEO4W
    os.environ['GDAL_DATA'] = OSGEO4W + r"\share\gdal"
    os.environ['PROJ_LIB'] = OSGEO4W + r"\share\proj"
    os.environ['PATH'] = OSGEO4W + r"\bin;" + os.environ['PATH']
    os.add_dll_directory(OSGEO4W + r"\bin")

import torch
import torch.optim
from torch import nn
from models import baseNet
from data_loader import build_data_loader, trans_tif
import numpy as np
from util.util import prepare_training
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # 每行代码放到最前段
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
import matplotlib.pyplot as plt
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def args_parser():
    project_name = 'own'
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', type=str, default='./results/')
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('-project_name', type=str, default=project_name)
    parser.add_argument('-dataset', type=str, default='Houston',
                        choices=['PaviaU', 'Houston','WHU'])

    # learning setting
    parser.add_argument('--epochs', type=int, default=200,
                        help='end epoch for training')
                
    # model setting
    parser.add_argument('--hidden_size', type=int, default=512)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=int, default=7,
                        help='samples for training')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='E:\code\cnn_2\checkpoints\own\Houston\model_91.68_1.pth')
    parser.add_argument('--seed', type=int, default=345,
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--allimg', type=bool, default=False, help='allimg')

    args = parser.parse_args()
    return args

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def visualize_output(outputs, save_path=None):
    # 1. 计算类别数量
    num_classes = int(np.max(outputs)) + 1  # 假设标签从1开始，+1得到总类别数

    # 2. 获取颜色映射（使用tab20色盘）
    colors = plt.cm.get_cmap('tab20', num_classes)  # 'tab20'提供20种颜色

    # 3. 创建颜色数组（每个类别对应一个RGB值）
    cmap = colors(np.linspace(0, 1, num_classes))  # 在0-1范围内均匀采样颜色

    # 4. 将背景类别（标签0）设为黑色
    cmap[0] = [0, 0, 0, 1]  # RGBA格式，最后一位是透明度（1表示不透明）

    # 5. 创建画布（设置为与图像内容匹配的大小）
    dpi = 100
    height, width = outputs.shape
    figsize = width / dpi, height / dpi  # 设置图像大小与输出尺寸匹配
    plt.figure(figsize=figsize, dpi=dpi)

    # 6. 显示图像
    im = plt.imshow(outputs, cmap=plt.cm.colors.ListedColormap(cmap))

    # 7. 关闭坐标轴
    plt.axis('off')  # 隐藏x轴和y轴的刻度

    # 8. 移除图像周围的空白区域
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)

    # 9. 保存图像（可选）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(f"可视化结果已保存到：{save_path}")

    # 10. 显示图像（弹出窗口）
    plt.show()
def pred_allimg(model, device, X, y, epoch, args):
    model.eval()
    height = y.shape[0]
    width = y.shape[1]
    with torch.no_grad():
        outputs = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                if args.allimg:
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
                else:
                    if int(y[i, j]) == 0:
                        continue
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
            if i % 20 == 0:
                print('... ... row ', i, ' handling ... ...')
    if args.allimg:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/All_PRED.tif'
        trans_tif(outputs, finalmodelfile)
    else:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.npy'
        y[y!=0] = 1
        outputs = outputs*y
        trans_tif(outputs, finalmodelfile)
       # 可视化输出
    visualize_output(outputs)

def main():
    args = args_parser()
    print (args)
    model_dir_path = os.path.join(args.results, args.project_name + '/')
    log_file = os.path.join(args.results, args.project_name +'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/', exist_ok=True)
    args.log_file = log_file

    X, y = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        model = baseNet(args.hsi_bands, args.num_class,args.patch_size).to(device)
    else:    
        model = baseNet(args.PCA, args.num_class,args.patch_size).to(device)
    
    model.load_state_dict(torch.load(args.checkpointsmodelfile, weights_only=True))
    pred_allimg(model, device, X, y, args.epochs, args)

def main_test():
    pass



if __name__ == '__main__':
     main()
