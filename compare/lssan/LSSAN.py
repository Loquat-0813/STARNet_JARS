import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

#LSSAN
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):#ch_in 表示输入特征图的通道数，reduction 是通道缩减的比例，默认为 4。
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(#定义顺序
            nn.Linear(ch_in, ch_in // reduction, bias=False),#将输入的通道数从 ch_in 缩减到 ch_in // reduction，不使用偏置项。
            nn.ReLU(inplace=True),#使用 ReLU 激活函数对输出进行非线性变换，inplace=True 表示直接在原张量上进行修改，节省内存。
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()#将输出值映射到 [0, 1] 区间，作为通道注意力权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()#获取输入特征图的批次大小 m_batchsize、通道数 C、宽度 width 和高度 height。
        y = self.avg_pool(x).view(b, c)  # squeeze操作 
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上
class SPA(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):#in_dim 表示输入特征图的通道数
        super(SPA, self).__init__()
        self.chanel_in = in_dim#保存输入通道数

        self.gamma = nn.Parameter(torch.zeros(1))#定义一个可学习的参数，初始值为 0，用于控制注意力的强度

        self.softmax = nn.Softmax(dim=-1)#定义一个 Softmax 激活函数，用于在最后一个维度上进行归一化操作
        self.sigmoid =nn.Sigmoid()#定义一个 Sigmoid 激活函数，用于将注意力能量映射到 [0, 1] 区间

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)#.permute是维度变换，permute(0, 2, 1)即将2,3维度互换 将输入特征图的空间维度展平，形状变为 (m_batchsize, C, width * height)。

        proj_key = x.view(m_batchsize, -1, width * height).permute(0, 2, 1).permute(1, 0, 2)  # B X C x (*W*H) 先将输入特征图展平并交换维度得到 (m_batchsize, width * height, C)，再交换第 1 维和第 2 维的位置，形状变为 (width * height, m_batchsize, C)，作为键（key）。
        y = proj_key[[(width * height)//2+1]].permute(1, 0, 2)#proj_key[[(width * height)//2+1]]：从键中选取中间位置的元素。
#.permute(1, 0, 2) 和 .permute(0, 2, 1)：对选取的元素进行维度变换，使其形状适合后续计算。
        y = y.permute(0, 2, 1)

        energy =torch.cosine_similarity(proj_query.unsqueeze(2), y.unsqueeze(1), dim=3)
#proj_query.unsqueeze(2) 和 y.unsqueeze(1)：分别在第 2 维和第 1 维上插入一个新的维度。
#torch.cosine_similarity(..., dim=3)：计算查询和选取元素之间的余弦相似度，得到注意力能量。
        attention = self.sigmoid(energy)
#使用 Sigmoid 激活函数将注意力能量映射到 [0, 1] 区间，得到注意力权重
        proj_value = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X N
#将输入特征图展平并交换维度，形状变为 (m_batchsize, width * height, C)，作为值（value）
        out = proj_value * attention
        out = out.reshape(m_batchsize, C, width, height)
#将注意力加权后的结果调整形状为与输入特征图相同的形状
        #out = self.gamma * out + x
        return out
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6#将输入 x 与 relu6(x + 3) 相乘并除以 6。
        return out


class LSSAN(nn.Module):
    def __init__(self, pca_components, class_num):#pca_components 表示输入特征图的通道数，class_num 表示分类的类别数。
        super(LSSAN, self).__init__()

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(pca_components, pca_components, kernel_size=(3, 3), stride=1, padding=1),#卷积层，输入通道数和输出通道数都为 pca_components，卷积核大小为 3x3，步长为 1，填充为 1，保证输入和输出的空间尺寸相同。
            hswish(),#使用硬 Swish 激活函数对卷积层的输出进行非线性变换。

        )
        self.bneck_1 = nn.Sequential(#定义一个瓶颈模块
            nn.Conv2d(pca_components, 128, kernel_size=(1, 1)),#1x1 卷积层，将输入通道数从 pca_components 扩展到 128。
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=2,groups=128),#3x3 深度可分离卷积层，输入通道数和输出通道数都为 128，步长为 2，会使空间尺寸减半。
            SE_Block(128),#通道注意力模块，用于增强通道之间的相关性
            nn.Conv2d(128, 64, kernel_size=(1, 1)),#1x1 卷积层，将通道数从 128 缩减到 64
            nn.ReLU(inplace=True),#使用 ReLU 激活函数对输出进行非线性变换

        )
        self.bneck_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1,groups=64),
            SPA(64),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

        )
        self.bneck_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1,groups=64),
            SE_Block(64),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

        )
        self.bneck_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1,groups=32),
            SE_Block(32),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

        )
        self.SE = SE_Block(32)#通道注意力模块，用于对输出特征图进行通道注意力增强
        self.GAP = nn.AdaptiveAvgPool2d(1)#池化
        self.bneck_5 = nn.Sequential(#包含硬 Swish 激活函数和全连接层，将输入的 32 维特征映射到 class_num 个类别上。

            hswish(),
            nn.Linear(32, class_num),

        )

        self.fc = nn.Linear(32, class_num)#另一个全连接层，将输入的 32 维特征映射到 class_num 个类别上


    def forward(self, x):
        # x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        out = self.conv2d_1(x)#将输入特征图传入 self.conv2d_1 模块进行处理
        out = self.bneck_1(out)
        out = self.bneck_2(out)

        out = self.bneck_3(out)+out# self.bneck_3 的输出与之前的输出相加，实现残差连接，有助于缓解梯度消失问题，提高模型的训练效果。

        out = self.bneck_4(out)+out
        out = self.SE(out)#将输出传入通道注意力模块 self.SE 进行通道注意力增强
        out = self.GAP (out)


        out = out.reshape(out.shape[0], -1)#将池化后的结果调整形状为一维向量，以便输入到全连接层
        out = self.bneck_5(out)#将一维向量传入 self.bneck_5 模块进行处理，得到最终的分类结果
        #out = self.fc(out)
        return out

if __name__ == '__main__':
    model = LSSAN(12, 10)
    x = torch.randn(2, 12, 9, 9)
    y = model(x)
    print(y.shape)