import math
import os
import torch
import torch.optim
from torch import nn
from models import baseNet
from data_loader import build_data_loader, trans_tif
import numpy as np
from util.util import prepare_training
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tabulate import tabulate
import scipy.io as sio
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def args_parser():
    project_name = 'hypersigma_b'
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', type=str, default='./results/')
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('-project_name', type=str, default=project_name)
    parser.add_argument('-dataset', type=str, default='PaviaU',
                        choices=['PaviaU', 'Houston'])

    # learning setting
    parser.add_argument('--epochs', type=int, default=200,
                        help='end epoch for training')
                
    # model setting
    parser.add_argument('--hidden_size', type=int, default=512)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=4096)
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=int, default=7,
                        help='samples for training')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='train or test')
    parser.add_argument('--is_show', type=bool, default=True,
                        help='show image or not')
    parser.add_argument('--is_labelshow', type=bool, default=True,
                        help='show image or not')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='/zbssd/yuyu/code/HyBenchmark/checkpoints/own/PaviaU/model_93.92.pth')
    parser.add_argument('--seed', type=int, default=300,
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--allimg', type=bool, default=False, help='allimg')

    args = parser.parse_args()
    return args

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def load_mat_label(file_path):
    """
    加载 .mat 文件中的标签数据
    """
    data = sio.loadmat(file_path)
    labels = data['mask_test']  # 假设标签存储在 'labels' 字段中
    return labels

# 生成可视化图并保存为矢量图
def visualize_labels(labels, output_path='labels_visualization.svg'):
    """
    将标签数据可视化并保存为矢量图
    """
    color_map = {
        0: (0.2*255, 0.2*255, 0.2*255),  # 深灰色
        1: (0.0*255, 1.0*255, 0.0*255),  # 绿色
        2: (0.0*255, 0.0*255, 1.0*255),  # 蓝色
        3: (1.0*255, 1.0*255, 0.0*255),  # 黄色
        4: (1.0*255, 0.0*255, 1.0*255),  # 品红
        5: (0.0*255, 1.0*255, 1.0*255),  # 青色
        6: (0.5*255, 0.5*255, 0.5*255),  # 灰色
        7: (1.0*255, 0.5*255, 0.0*255),  # 橙色
        8: (0.5*255, 0.0*255, 0.5*255),  # 紫色
        9: (0.0*255, 0.5*255, 0.5*255),  # 青绿色
        10: (0.5*255, 0.5*255, 0.0*255),  # 橄榄色
        11: (0.8*255, 0.2*255, 0.2*255),  # 浅红色
        12: (0.2*255, 0.8*255, 0.2*255),  # 浅绿色
        13: (0.2*255, 0.2*255, 0.8*255),  # 浅蓝色
        14: (0.8*255, 0.8*255, 0.2*255),  # 浅黄色
        15: (0.8*255, 0.2*255, 0.8*255),  # 浅品红
        16: (0.2*255, 0.8*255, 0.8*255),  # 浅青色
        17: (0.6*255, 0.4*255, 0.2*255),  # 棕色
        18: (0.2*255, 0.6*255, 0.4*255),  # 深青色
    }
    map_H, map_W = labels.shape
    segmented_image = np.zeros((map_H, map_W, 3), dtype=np.uint8)
    for label, color in color_map.items():
        segmented_image[labels == label] = color
    cv2.imwrite(output_path, segmented_image)


def pred_allimg(model, device, image_patches, y, coordinates, num_patches, N, args):
    model.eval()
    height = y.shape[0]
    width = y.shape[1]
    with torch.no_grad():
        outputs = np.zeros((height, width))
        for start in range(0, num_patches, N):
            end = min(start + N, num_patches)
            batch_patches = image_patches[start:end]
            X_test_images = torch.FloatTensor(batch_patches.transpose(0, 3, 1, 2)).to(device)

            predictions = model(X_test_images)
            predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            # 将预测结果填回输出数组
            for idx, (i, j) in enumerate(coordinates[start:end]):
                outputs[i][j] = predictions[idx] + 1

            print('... ... row ', start, ' handling ... ...')
    if args.allimg:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/All_PRED.mat'
        finalsvgfile = args.results + args.project_name+ '/' + args.dataset+ '/All_PRED.png'
        finallabelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label.png'
        sio.savemat(finalmodelfile, mdict={'outputs': outputs})
        if args.is_show:
            if args.is_labelshow:
                visualize_labels(y, finallabelfile)
            visualize_labels(outputs, finalsvgfile)
    else:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.mat'
        finalsvgfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.png'
        finallabelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label.png'
        if args.is_show:
            if args.is_labelshow:
                visualize_labels(y, finallabelfile)
        y[y!=0] = 1
        outputs = outputs*y
        sio.savemat(finalmodelfile, mdict={'outputs': outputs})
        if args.is_show:
            visualize_labels(outputs, finalsvgfile)
    
def main():
    args = args_parser()
    print (args)
    model_dir_path = os.path.join(args.results, args.project_name + '/')
    log_file = os.path.join(args.results, args.project_name +'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/', exist_ok=True)
    os.makedirs(model_dir_path+args.dataset, exist_ok=True)
    args.log_file = log_file

    X, y = build_data_loader(args)

    image_patches = []
    coordinates = []
    height = y.shape[0]
    width = y.shape[1]

    for i in range(height):
        for j in range(width):
            image_patch = X[i:i + args.patch_size, j:j + args.patch_size]
            image_patches.append(image_patch)
            coordinates.append((i, j))
    
    image_patches = np.array(image_patches)
    image_patches = image_patches.reshape(len(image_patches), args.patch_size, args.patch_size, -1)
    N = args.batch_size
    num_patches = len(image_patches)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        model = baseNet(args.hsi_bands, args.num_class).to(device)
    else:    
        model = baseNet(args.PCA, args.num_class).to(device)
    
    model.load_state_dict(torch.load(args.checkpointsmodelfile, weights_only=True))
    pred_allimg(model, device, image_patches, y, coordinates, num_patches, N, args)

if __name__ == '__main__':
     main()
