import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Attention(nn.Module):#定义一个名为 Attention 的类，继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类。
    def __init__(self, n, nin, in_channels):#n：卷积核的大小相关参数。nin：输入的通道数。in_channels：输入特征的通道数
        super(Attention, self).__init__()
        self.sigmoid = nn.Sigmoid()#用于将输出值映射到 [0, 1] 区间
        self.GAP = nn.AdaptiveAvgPool3d(1)#定义一个三维自适应平均池化层，将输入特征的空间维度自适应地池化为 1x1x1。
        self.conv = nn.Conv3d(1,1,kernel_size=(n, 1, 1),padding=((n-1)//2,0,0))#定义一个三维卷积层，输入和输出通道数都为 1，卷积核大小为 (n, 1, 1)，通过设置合适的填充保证输出尺寸不变。

        self.AvgPool = nn.AvgPool3d((in_channels//2,1,1))# 黄河口设为（9,1,1），其余为（10,1,1）
        self.conv1 = nn.Conv3d(nin, nin, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))#定义一个三维卷积层，输入和输出通道数都为 nin，卷积核大小为 (1, n, n)，步长为 1，通过填充保证输出尺寸不变。

        self.fc1 = nn.Linear(nin, nin // 8, bias=False)#定义一个全连接层，输入维度为 nin，输出维度为 nin // 8，不使用偏置项。

        self.fc2 = nn.Linear(nin// 8, nin, bias=False)#定义一个全连接层，输入维度为 nin // 8，输出维度为 nin，不使用偏置项。

    def forward(self, x):
        n1,c,l,w,h = x.shape#获取输入张量的形状，n1 是批次大小，c 是通道数，l、w、h 分别是三维空间的长度、宽度和高度。

        se = self.sigmoid(self.conv (self.GAP(x.permute(0, 2, 1,3,4)).permute(0, 2, 1,3,4)))
#x.permute(0, 2, 1,3,4)：对输入张量的维度进行重排。self.GAP(...)：进行自适应平均池化。self.conv(...)：进行卷积操作。self.sigmoid(...)：通过 Sigmoid 函数得到一个空间注意力权重 se
        sa = self.sigmoid(self.conv1 (self.AvgPool(x)))
#self.AvgPool(x)：进行平均池化。self.conv1(...)：进行卷积操作。self.sigmoid(...)：通过 Sigmoid 函数得到一个空间注意力权重 sa。
        x1 = self.GAP(x)#对输入进行自适应平均池化
        x1 = x1.reshape(n1, -1)#将池化后的结果展平为一维向量
        f1 = self.fc1(x1)#通过第一个全连接层
        f2 = self.fc2(f1)#通过第二个全连接层

        ca = self.sigmoid(f2)#通过 Sigmoid 函数得到一个通道注意力权重 ca
        ca = ca.reshape(n1, c,1,1,1)#将通道注意力权重调整为与输入张量可相乘的形状

        w = se*sa*ca
        out = x*w

        return out

class Unit(nn.Module):
    def __init__(self, n):
        super(Unit, self).__init__()

        # 3D层
        self.bn1 = nn.BatchNorm3d(64)#self.bn1 和 self.Conv3d_1 是第一个批量归一化层和卷积层，输入通道数为 64，输出通道数为 32，卷积核大小为 (1, n, n)。
        self.Conv3d_1 = nn.Conv3d(64, 32, kernel_size=(1, n, n), stride=1,padding=(0,(n-1)//2,(n-1)//2))
        self.bn2 = nn.BatchNorm3d(32)#self.bn2 和 self.Conv3d_2 是第二个批量归一化层和卷积层，输入和输出通道数都为 32，卷积核大小为 (n, 1, 1)。
        self.Conv3d_2 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1,padding=((n-1)//2,0,0))

        self.bn3 = nn.BatchNorm3d(96)
        self.Conv3d_3 = nn.Conv3d(96, 32, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))
        self.bn4 = nn.BatchNorm3d(32)
        self.Conv3d_4 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1, padding=((n - 1) // 2, 0, 0))

        self.bn5 = nn.BatchNorm3d(128)
        self.Conv3d_5 = nn.Conv3d(128, 32, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))
        self.bn6 = nn.BatchNorm3d(32)
        self.Conv3d_6 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1, padding=((n - 1) // 2, 0, 0))

    def forward(self, x):
        out1 = self.Conv3d_1(F.relu(self.bn1(x)))#对输入 x 先进行批量归一化，然后通过 ReLU 激活函数，最后进行卷积操作得到 out1。
        x1 = self.Conv3d_2(F.relu(self.bn2(out1)))#对 out1 进行类似的操作得到 x1
        out1 = torch.cat([x1, x], dim=1)#将 x1 和输入 x 在通道维度上进行拼接得到新的 out1

        out2 = self.Conv3d_3(F.relu(self.bn3(out1)))
        x2 = self.Conv3d_4(F.relu(self.bn4(out2)))
        out2 = torch.cat([x2, x, x1], dim=1)

        out3 = self.Conv3d_5(F.relu(self.bn5(out2)))
        x3 = self.Conv3d_6(F.relu(self.bn6(out3)))
        out3 = torch.cat([x3, x, x1,x2], dim=1)


        return out3

class msdan(nn.Module):
    def __init__(self, in_channels, class_num):#in_channels：输入特征的通道数。class_num：分类的类别数。
        super(msdan, self).__init__()

        self.conv3d = nn.Sequential(#包含一个批量归一化层、ReLU 激活函数和一个三维卷积层，用于对输入进行初步处理。
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(2,1,1), padding=(1,1,1),)
        )
        self.bneck_1 = Unit(3)
        self.at1 = Attention(3, 160, in_channels)

        self.bneck_2 = Unit(5)
        self.at2 = Attention(5, 160, in_channels)

        self.bneck_3 = Unit(7)
        self.at3 = Attention(7, 160, in_channels)
#分别实例化 Unit 类，卷积核大小分别为 3、5、7 分别实例化 Attention 类，卷积核大小分别为 3、5、7
        if in_channels%2 == 0:
            self.conv3d_2 = nn.Sequential(
                nn.BatchNorm3d(160),
                nn.ReLU(inplace=True),
                nn.Conv3d(160, 256, kernel_size=(in_channels//2, 1, 1), stride=1,)
            )
        else:
            self.conv3d_2 = nn.Sequential(
                nn.BatchNorm3d(160),
                nn.ReLU(inplace=True),
                nn.Conv3d(160, 256, kernel_size=(in_channels//2+1, 1, 1), stride=1,)
            )
        self.conv3d_3 = nn.Sequential(#定义一个顺序容器，包含批量归一化层、ReLU 激活函数和三维卷积层。
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(256, 1, 1), stride=1, )
        )

        self.MaxPool = nn.MaxPool3d((1, 7, 7))
        self.conv3d_4 = nn.Sequential(
            nn.BatchNorm3d(64),#黄河口时隐藏
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1,padding=(0,1,1))
        )

        self.GAP = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64, class_num)#定义一个全连接层，输入维度为 64，输出维度为 class_num，用于最终的分类


    def forward(self, x):
        if len(x.shape) == 4:#如果输入 x 的维度为 4，则在第 1 维上添加一个维度，使其变为 5 维。
            x = x.unsqueeze(1)
        x = self.conv3d(x)#将输入 x 传入 self.conv3d 进行初步处理

        x1 = self.bneck_1(x)#将处理后的 x 传入 self.bneck_1 得到 x1
        x1 = self.at1(x1)#将 x1 传入 self.at1 进行注意力机制处理

        x2 = self.bneck_2(x)
        x2 = self.at2(x2)

        x3 = self.bneck_3(x)
        x3 = self.at3(x3)

        out = x1+x2+x3

        out = self.conv3d_2(out).permute(0, 2, 1,3,4)#将 out 传入 self.conv3d_2 进行处理，然后对维度进行重排。
        out = self.conv3d_3(out)#将重排后的 out 传入 self.conv3d_3 进行处理
        out = self.MaxPool(out)

        out = self.conv3d_4(out)#将池化后的 out 传入 self.conv3d_4 进行处理
        out = self.GAP(out)

        out = out.reshape(out.shape[0], -1)#将池化后的结果展平为一维向量

        out = self.fc(out)#将展平后的向量传入 self.fc 进行分类，得到最终的输出
        return out

if __name__ == '__main__':
    model = msdan(128, 10)#实例化 msdan 模型，输入通道数为 128，分类类别数为 10
    x = torch.randn(2, 128, 9, 9)#生成一个随机的输入张量，形状为 (2, 128, 9, 9)，表示批次大小为 2，通道数为 128，高度和宽度均为 9。
    y = model(x)#将输入张量 x 传入模型进行前向传播，得到输出结果 y
    print(y.shape)#打印输出结果 y 的形状