import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import truediv
import warnings
import time
from torchsummary import summary
import math
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline
import os
import matplotlib as mpl
from einops import rearrange
# 模型代码
# -------------------------------------------------------------------------------------------- #
def bsm(n,d):#bsm函数 该函数生成一个二进制空间掩膜，中心区域为1，外围为0。用于后续注意力机制中突出中心像素 引导神经网络关注图像中心区域
    #就像在照片上贴一个"镂空相框"，中间可以透过光（值为1），边缘被遮挡（值为0）。这个相框的作用是让模型更关注图片中心的内容。
    a = [[0]*n for x in range(n)] # 创建n×n的二维数组
    p = 0 # 起始索引
    q = n-1 # 终止索引
    w = (n+1)/2 # 计算中心位置
    w =int(w)
    #print(w)
    #w1 = 1 / w
    #print(w1)
    t = 0
    while p < d:#第一层循环 (p < d时)
        for i in range(p,q):#上边框
            a[p][i] = t
        for i in range(p,q):#右边框
            a[i][q] = t
        for i in range(q,p,-1):# 绘制下边框 (q=4行)
            a[q][i] = t
        for i in range(q,p,-1):# 绘制左边框 (p=0列)
            a[i][p] = t
        p += 1
        q -= 1
        #t += w1

    while p==d or p>d and p<q:#第二层循环 (p >= d时)
        for i in range(p,q):# 绘制上边框 (p=1行)
            a[p][i] = 1
        for i in range(p,q):# 绘制右边框 (q=3列)
            a[i][q] = 1
        for i in range(q,p,-1):# 绘制下边框 (q=3行)
            a[q][i] = 1
        for i in range(q,p,-1):# 绘制左边框 (p=1列)
            a[i][p] = 1

        a[w-1][w-1] = 1# 中心点强制设为1
        p += 1
        q -= 1
    return np.array(a)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScaleMaskModule(nn.Module):#尺度掩膜 模型通过这种方式强制自己更关注图像中心的关键信息，就像你在黑暗中只看手电筒照亮的地方。
    def __init__(self,d):
        # w是空间尺度，n是光谱维度，p是批次大小
        super(ScaleMaskModule, self).__init__()

        self.d = d# 控制掩膜范围 就像选择手电筒的光圈大小，d 值越大，照亮的中心区域越大
    def forward(self,x):
        w = x.shape[3]# 输入的空间尺寸  # 获取输入特征图的宽度/高度（假设为正方形）
        n = x.shape[2]  # 光谱维度（如多光谱通道数）
        o = x.shape[1]# 输出通道数
        p = x.shape[0]# 批次大小
       # print(x.shape)
        out = bsm(w,self.d)# 生成掩膜
        #print(out.shape)
        out = torch.from_numpy(out)# 转为PyTorch张量
        out = out.repeat(p, o, 1, 1)#out.repeat(p, o,n, 1, 1)# 扩展维度匹配输入
        #print(out.shape)
        out = out.type(torch.FloatTensor) # 确保数据类型一致
        out = out.to(device) # 移至GPU
        #print(x * out)
        return x * out# 应用掩膜

class NCAM3D(nn.Module):# 3D NCAM
    def __init__(self, c, patch_size):
        super(NCAM3D, self).__init__()
        gamma = 2#动态卷积核设计根据输入通道数c动态调整卷积核大小，保证：
#通道数越多 → 卷积核越大（捕获更广的光谱关系）
#始终保持奇数尺寸（便于对称填充）
        b = 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size-1)//2-1)# 中心聚焦模块

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2,kernel_size_21), padding=(0,(kernel_size_21 - 1) // 2), dilation=1)# 中心特征处理器
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2), # 全局特征处理器
                               dilation=1)
    def forward(self, x):#前向传播

        out =x
        #### 通道注意力
        out_1 = out.shape[1]# 获取 `out` 张量的第二个维度的大小，通常表示通道数
        out_2 = out.shape[2]# 获取 `out` 张量的第三个维度的大小
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])# 对 `out` 张量进行形状重塑。`out.shape[0]` 是批次大小，`-1` 表示该维度的大小由其他维度自动计算得出，
        #`out.shape[3]` 和 `out.shape[4]` 分别是第四和第五个维度的大小
        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out) # 生成中心区域掩膜
        out_x1 = self.avg_pool(out_x)# 各通道空间平均池化
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)# 降维
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])# 对 `out_x1` 进行形状重塑，将其转换为四维张量，方便后续的拼接操作
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1]) # 对 `out_x2` 进行形状重塑

        out_xx = torch.cat([out_x1, out_x2], dim=2)# 沿着第三个维度（索引为 2）将 `out_x1` 和 `out_x2` 拼接在一起，得到 `out_xx`
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        # 对 `out` 进行平均池化操作，得到每个通道的空间平均值 `out1`
        out1 = out1.reshape(out1.shape[0], -1)
 # 对 `out1` 进行形状重塑，将其转换为二维张量，`out1.shape[0]` 是批次大小，`-1` 表示该维度的大小由其他维度自动计算得出
        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]
# 首先对 `out1` 进行维度置换，将第一个和第二个维度交换，然后对置换后的张量进行倒序操作，最后再将维度置换回来
        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
         # 对 `outx` 进行一维卷积操作 `self.conv1d`，然后通过 `F.sigmoid` 函数将结果映射到 [0, 1] 区间，再对维度进行置换；对 `out_xx` 进行一维卷积操作 `self.conv1d1`，然后通过 `F.sigmoid` 函数将结果映射到 [0, 1] 区间，再对维度进行置换；最后将两个结果相乘得到 `at1`
        at = F.sigmoid((at1-0.2)*2) # 对 `at1` 进行缩放和平移操作，然后通过 `F.sigmoid` 函数将结果映射到 [0, 1] 区间，得到注意力权重 `at
        out = out * at# 将 `out` 与注意力权重 `at` 相乘，实现通道注意力机制
        #####
        out = out.reshape(out.shape[0], out_1, out_2, out.shape[2], out.shape[3])
# 将 `out` 恢复到原来的形状，`out.shape[0]` 是批次大小，`out_1` 和 `out_2` 分别是之前保存的第二个和第三个维度的大小，`out.shape[2]` 和 `out.shape[3]` 分别是当前的第三个和第四个维度的大小
        return out


class NCAM2D(nn.Module):  # 2D NCAM
    def __init__(self, c, patch_size): #c代表输入数据的通道数，patch_size 表示输入数据的块大小。
        super(NCAM2D, self).__init__()
#gamma 和 b 是用于计算卷积核大小的超参数。
        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c , 2) + b) / gamma))#依据输入通道数 c 动态计算卷积核的大小。
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1#保证结果为奇数，当%2不为0时执行if语句，即奇数；

        self.avg_pool = nn.AdaptiveAvgPool2d(1)#平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)#最大池化
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2-1)#中心掩膜模块

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                 dilation=1)
#输入通道数为 1，输出通道数为 1。
#kernel_size=(2, kernel_size_21)：卷积核的大小为 2 x kernel_size_21。
# padding=(0, (kernel_size_21 - 1) // 2)：在高度方向不进行填充，在宽度方向进行对称填充，保证卷积操作后特征图的宽度不变。
#dilation=1：卷积核的膨胀率为 1，即普通卷积。

    def forward(self, x):
        out = x
        #### 通道注意力

        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x)
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])

        out_xx = torch.cat([out_x1, out_x2], dim=2)
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)

        at = F.sigmoid((at1 - 0.2) * 2)
        #print(at)
        #at = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2)
        out = out * at

        return out

class LE_DSC3D(nn.Module):# 3D LE-DSC 就像处理一个大项目：
#深度卷积：把团队分成小组各自研究（分组处理不同特征）
#逐点卷积：组长开会汇总各组成果（整合所有发现）
    def __init__(self, nin, nout,kernel_size_c,kernel_size_h,kernel_size_w,pca_components, patch_size, padding=True):#nin：输入通道数。nout：输出通道数。
#kernel_size_c、kernel_size_h、kernel_size_w：分别是 3D 卷积核在通道、高度和宽度维度上的大小。
#pca_components：主成分分析（PCA）的组件数量，可能用于后续的注意力模块。
#patch_size：用于注意力模块的块大小。
#padding：是否进行填充，默认为 True。
        super(LE_DSC3D, self).__init__()
        self.nout = nout#保存输入通道数
        self.nin = nin#保存输出通道数
        self.at1 = NCAM3D(self.nin*pca_components, patch_size)
        self.at2 = NCAM3D(self.nout * pca_components, patch_size)
#初始化两个 NCAM3D 通道注意力模块，分别用于输入特征和输出特征的通道注意力增强。
        if padding == True:

         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, 1, kernel_size_w),
                                   padding=((kernel_size_c - 1) // 2, 0, (kernel_size_w - 1) // 2), groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, kernel_size_h, 1),
                                    padding=((kernel_size_c - 1) // 2, (kernel_size_h - 1) // 2, 0), groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),
                                    padding=(0, (kernel_size_h - 1) // 2, (kernel_size_w - 1) // 2), groups=nin)

        else:
         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,1,kernel_size_w),  groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,kernel_size_h,1),  groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),  groups=nin)
#根据 padding 参数的值，初始化三个 3D 深度可分离卷积层：
#self.depthwise：在通道和宽度维度上进行卷积，高度维度上不进行卷积。
#self.depthwise1：在通道和高度维度上进行卷积，宽度维度上不进行卷积。
#self.depthwise2：在高度和宽度维度上进行卷积，通道维度上不进行卷积。
#groups=nin 表示每个输入通道只与一个输出通道进行卷积，这是深度可分离卷积的特点。
#当 padding 为 True 时，会在相应维度上进行填充，以保持输入和输出的尺寸一致。
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)
#self.pointwise：初始化一个 3D 逐点卷积层，用于将深度可分离卷积的输出通道数从 nin 转换为 nout
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)#如果输入 x 的维度为 4，则在第 1 个维度上插入一个新的维度，使其成为 5 维张量，以适应 3D 卷积的输入要求
        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = self.depthwise2(x)
#分别对输入 x 进行三个深度可分离卷积操作。
        out3 = out1+out2+out3 #

        out =out3
        #### 通道注意力
        out = self.at1(out)#第一个通道注意力模块 self.at1

        out = self.pointwise(out)#使用逐点卷积层将通道数从 nin 转换为 nout。

        #### 通道注意力
        out = self.at2(out)#第二个通道注意力模块 self.at2
        ####

        return out
###################################################################
class LE_DSC2D(nn.Module):#深度可分离混合卷积 ---- 2D数据版本
    def __init__(self, nin, nout, kernel_size_h, kernel_size_w, patch_size, padding=True):
        super(LE_DSC2D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM2D(self.nin, patch_size)
        self.at2 = NCAM2D(self.nout, patch_size)

        if padding == True:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1), padding=((kernel_size_h - 1) // 2,0), groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w), padding=(0,(kernel_size_w - 1) // 2), groups=nin)
        else:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1),  groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w),  groups=nin)

        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = out1+out2 #

        out =out3
        #### 通道注意力
        out = self.at1(out)

        out = self.pointwise(out)
        #### 通道注意力
        out = self.at2(out)
        ####

        return out
    

class Attention(nn.Module):
    """Top-K Selective Attention (TTSA)
    Tips:
        Mainly borrows from DRSFormer (https://github.com/cschenxiang/DRSformer)
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b 1 C C

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # print(111, mask1.scatter_(-1, index, 1.))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out =out1 * self.attn1 
        #out =  out1 * self.attn1+ out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6#inplace=True 表示直接在原张量上进行修改以节省内存。 relu6 函数会将输入值限制在 [0, 6] 范围内
        return out
    
class LE_HCL_new(nn.Module):#输入默认为一个三维块，即三维通道数为1
    def __init__(self, ax, aa, c, pca_components, patch_size):#ax二维通道数，c卷积核大小，d为padding和dilation大小 pca_components：主成分分析的组件数量，可能用于注意力模块
        super(LE_HCL_new, self).__init__()
        self.conv3d = nn.Sequential(
            LE_DSC3D(1, ax, c, c, c, pca_components, patch_size),#使用之前定义的 LE_DSC3D 模块，输入通道数为 1，输出通道数为 ax，卷积核在三个维度上的大小均为 c。
            nn.BatchNorm3d(ax),#对 3D 卷积的输出进行批量归一化操作，有助于加速模型收敛和提高模型的稳定性
            hswish(),#使用自定义的 hswish 激活函数对批量归一化后的结果进行非线性变换

        )
        #self.rcm=RCM(ax)
        self.conv2d = nn.Sequential(
            LE_DSC2D(aa, aa // ax, c, c, patch_size),
            nn.BatchNorm2d(aa // ax),
            hswish(),

        )
        self.attn = Attention(aa // ax, 2, bias=False)

    def forward(self, x):

        out = self.conv3d(x)

        #out=self.rcm(out)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d(out)
        out=self.attn(out)

        # out = out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = out + x

        return out
# 主模型
class baseNet(nn.Module):#定义了一个名为 Lite_HCNet 的类，它继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。
    def __init__(self, in_channels, class_num, patch_size):#in_channels：输入数据的通道数。class_num：分类的类别数量。patch_size：输入数据的块大小。
        super(baseNet, self).__init__()
        ########
        e = 3 # LE-HCL中的e参数
        self.unit1 = LE_HCL_new(e, e*in_channels, 3, in_channels, patch_size)
        self.unit2 = LE_HCL_new(e, e*in_channels, 7, in_channels, patch_size)
#初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#初始化一个自适应平均池化层，将输入的特征图自适应地池化为大小为 1x1 的特征图，用于全局特征提取。

        self.fc1 = nn.Linear(in_channels, class_num)#初始化一个全连接层，输入维度为 in_channels，输出维度为 class_num，用于将提取的特征映射到分类类别上。

    def forward(self, x):#向前传播

        out1 = self.unit1(x)
        out2 = self.unit2(x)

        out = out1+out2

        # out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.avg_pool(out)#对融合后的特征进行自适应平均池化操作，提取全局特征。
        out = out.reshape(out.shape[0], -1)#展平
        out = self.fc1(out)#全链接层

        return out

if __name__ == '__main__':
    model = baseNet(16, 10, 9).cuda()#实例化 Lite_HCNet 模型，输入通道数为 15，分类类别数量为 10，块大小为 9，并将模型移动到 GPU 上。
    x = torch.randn(32, 16, 9, 9).cuda()#生成一个随机的输入张量，形状为 (32, 15, 9, 9)，表示批量大小为 32，通道数为 15，高度和宽度均为 9，并将其移动到 GPU 上。
    y = model(x)

