import math
import os
# 如果操作系统是Windows（os.name为'nt'）
if os.name == 'nt':
    import platform
    # 定义OSGEO4W的路径，这里需要根据实际情况修改
    OSGEO4W = r"D:\anaconda3\envs\pytorch\Library"
    # 断言该路径存在，如果不存在则抛出异常
    assert os.path.isdir(OSGEO4W), "Directory does not exist: " + OSGEO4W
    # 设置环境变量OSGEO4W_ROOT为OSGEO4W的路径
    os.environ['OSGEO4W_ROOT'] = OSGEO4W
    # 设置GDAL_DATA环境变量，指向GDAL数据的路径
    os.environ['GDAL_DATA'] = OSGEO4W + r"\share\gdal"
    # 设置PROJ_LIB环境变量，指向PROJ库的路径
    os.environ['PROJ_LIB'] = OSGEO4W + r"\share\proj"
    # 将OSGEO4W的bin目录添加到系统路径中
    os.environ['PATH'] = OSGEO4W + r"\bin;" + os.environ['PATH']
    # 将OSGEO4W的bin目录添加到DLL搜索路径中
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
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
import time  # 导入时间模块，用于统计运行时间

# 允许使用TF32格式进行计算，提高计算效率
torch.backends.cudnn.allow_tf32 = True
# 允许CUDA矩阵乘法使用TF32格式
torch.backends.cuda.matmul.allow_tf32 = True

# 定义命令行参数解析函数
def args_parser():
    # 项目名称
    project_name = 'own'
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加结果保存路径的参数，默认值为'./results/'
    parser.add_argument('-results', type=str, default='./results/')
    # 添加检查点保存路径的参数，默认值为'./checkpoints/'
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    # 添加项目名称的参数，默认值为前面定义的project_name
    parser.add_argument('-project_name', type=str, default=project_name)
    # 添加数据集名称的参数，默认值为'PaviaU'，可选值为'PaviaU'、'Houston'、'IP'
    parser.add_argument('-dataset', type=str, default='Houston',
                        choices=['PaviaU', 'Houston', 'WHU'])
    # 数据集设置部分
    # 添加批量大小的参数，默认值为32
    parser.add_argument('--batch_size', type=int, default=256)
    # 添加训练样本比例的参数，默认值为0.01
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='samples for training')
    # 添加是否进行训练的参数，默认值为True
    parser.add_argument('--is_train', type=bool, default=True,
                        help='train or test')
    # 添加是否输出所有图像的参数，默认值为False
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    # 添加模型文件路径的参数，默认值为指定的路径
    parser.add_argument('--modelfile', type=str, default='E:\code\cnn_2\checkpoints\own\Houston\model_92.38_1.pth')
    # 添加随机种子的参数，默认值为300
    parser.add_argument('--seed', type=int, default=345,
                        help='random seed')
    # 添加PCA降维的参数，默认值为None
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    # 解析命令行参数
    args = parser.parse_args()
    return args

# 自定义PyTorch张量的__repr__方法
def custom_repr(self):
    # 返回包含张量形状信息的字符串
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

# 保存原始的torch.Tensor的__repr__方法
original_repr = torch.Tensor.__repr__
# 用自定义的__repr__方法覆盖原始方法
torch.Tensor.__repr__ = custom_repr

# 计算损失函数
def calc_loss(outputs, labels):
    # 创建交叉熵损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 计算损失
    loss = criterion(outputs, labels)
    return loss

# 统计模型的参数数量
def count_parameters(model):
    # 计算模型所有参数的总数
    total_params = sum(p.numel() for p in model.parameters())
    # 计算模型可训练参数的总数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# 测试模型的函数
def test(model, device, test_loader, args):
    # 记录测试开始的时间
    start_time = time.time()
    # 将模型设置为评估模式
    model.eval()
    count = 0
    with torch.no_grad():
        for inputs_1, labels in test_loader:
            # 将输入数据移动到指定设备（GPU或CPU）
            inputs_1 = inputs_1.to(device)
            # 将标签数据移动到指定设备
            labels = labels.to(device)
            if args.PCA is not None:
                # 如果使用了PCA降维，调整输入数据的形状
                inputs_1 = inputs_1.view(-1, args.PCA, args.patch_size, args.patch_size)
            else:
                # 否则，按原始波段数调整输入数据的形状
                inputs_1 = inputs_1.view(-1, args.hsi_bands, args.patch_size, args.patch_size)
            # 模型进行前向传播，得到输出
            outputs = model(inputs_1)
            # 将输出转换为预测的类别
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                # 初始化预测结果和真实标签数组
                y_pred_test = outputs
                test_labels = labels.cpu().numpy()
                count = 1
            else:
                # 拼接预测结果和真实标签数组
                y_pred_test = np.concatenate((y_pred_test, outputs))
                test_labels = np.concatenate((test_labels, labels.cpu().numpy()))
    # 计算总体准确率（OA）
            # 计算 OA
    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c]==y_pred_test[c]:
            a = a + 1
    oa = a/len(y_pred_test)*100
    
    # 计算 AA
    num_classes = args.num_class  # 类别数
    class_correct = np.zeros(num_classes)  # 每个类别预测正确的样本数
    class_total = np.zeros(num_classes)  # 每个类别的总样本数
    
    for i in range(len(test_labels)):
        label = test_labels[i]
        class_total[label] += 1
        if y_pred_test[i] == label:
            class_correct[label] += 1
    
    class_accuracy = class_correct / class_total  # 每个类别的精度
    aa = np.mean(class_accuracy) * 100  # 平均精度、

    # 计算 Kappa系数
    total_samples = len(test_labels)
    true_count = np.zeros(num_classes)  # 每个类别的真实样本数
    pred_count = np.zeros(num_classes)  # 每个类别的预测样本数
    
    # 计算每个类别的真实样本数和预测样本数
    for i in range(total_samples):
        true_count[test_labels[i]] += 1
        pred_count[y_pred_test[i]] += 1
    
    # 计算期望一致性pe
    pe = 0
    for i in range(num_classes):
        pe += (true_count[i]/total_samples) * (pred_count[i]/total_samples)
    
    # 计算kappa系数
    po = a / total_samples  # 观察一致性（即准确率）
    kappa = (po - pe) / (1 - pe)
    kappa_percentage = kappa * 100  # 转为百分比形式
    if (1 - pe) <= 0:
      kappa = 1.0  # 或其他合理值 Kappa 系数的分母 (1 - pe) 可能为负数（当 pe > 1），导致结果异常。
#​解决方案：添加分母非负的检查：
    else:
      kappa = (po - pe) / (1 - pe)
    data = {
        "val": [f"Class {i}" for i in range(len(class_accuracy))],
        "Acc": [f"{acc:.2%}" for acc in class_accuracy],
    }
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='grid'))
    print(' [The test OA is: %.2f]' % (oa))
    print(' [The test AA is: %.2f]' % (aa))
    print(' [The test Kappa is: %.2f]' % (kappa_percentage))
    # 计算测试结束的时间
    end_time = time.time()
    # 计算测试所花费的时间
    elapsed_time = end_time - start_time
    print(f' [Test time: {elapsed_time:.2f} seconds]')
    with open(args.log_file, 'a') as appender:
        appender.write('\n')
        appender.write('########################### Test ###########################' + '\n')
        appender.write(' [The test OA is: %.2f]' % (oa) + ' [The test AA is: %.2f]' % (aa) +
                       ' [The test Kappa is: %.2f]' % (kappa_percentage) + '\n')
        appender.write(f' [Test time: {elapsed_time:.2f} seconds]' + '\n')
        appender.write('\n')
    return oa

# 主函数
def main():
    # 解析命令行参数
    args = args_parser()
    print(args)
    # 构建模型保存的目录路径
    model_dir_path = os.path.join(args.results, args.project_name + '/', args.dataset + '/')
    # 构建日志文件的路径
    log_file = os.path.join(args.results, args.project_name + '/', args.dataset + '/log.txt')
    # 创建模型保存的目录，如果目录已存在则不会报错
    os.makedirs(model_dir_path, exist_ok=True)
    # 创建检查点保存的目录，如果目录已存在则不会报错
    os.makedirs(args.checkpoints + args.project_name + '/' + args.dataset + '/', exist_ok=True)
    # 将日志文件路径添加到args对象中
    args.log_file = log_file
    # 构建数据加载器，返回训练数据加载器和测试数据加载器，这里只使用测试数据加载器
    _, test_loader = build_data_loader(args)
    # 选择设备，如果有可用的GPU则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        # 如果没有使用PCA降维，创建基于原始波段数的模型
        model = baseNet(args.hsi_bands, args.num_class,args.patch_size).to(device)
    else:
        # 如果使用了PCA降维，创建基于PCA维度的模型
        model = baseNet(args.PCA, args.num_class,args.patch_size).to(device)
    # 加载预训练的模型参数
    model.load_state_dict(torch.load(args.modelfile))
    # 统计模型的参数数量
    total_params, trainable_params = count_parameters(model)
    print(f' [Total parameters: {total_params}, Trainable parameters: {trainable_params}]')
    with open(args.log_file, 'a') as appender:
        appender.write(f' [Total parameters: {total_params}, Trainable parameters: {trainable_params}]' + '\n')
    # 调用测试函数对模型进行测试
    test(model, device, test_loader, args)

if __name__ == '__main__':
    main()