import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


#############################################################################
#3D-2D
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=(3,3), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=(1,1))
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class depthwise_separable_conv1(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv1, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin, kernel_size=(3,3,3), padding=0, groups=nin)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=(1,1,1))
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class D32(nn.Module):
    def __init__(self):
        super(D32, self).__init__()
        #self.tryat = TryAttention1()
        self.conv3d_1 = nn.Sequential(
            #depthwise_separable_conv1(1, 8),
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            #nn.PReLU(inplace=True),
        )

        self.deepconv3d_2 = nn.Sequential(
            depthwise_separable_conv1(8, 16),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )


        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(352, 64, kernel_size=(3, 3), stride=1, padding=0),#UP-288,IP-768,HU-352
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.ReLU(inplace=True),
        )
        self.deepconv2d = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(128, 256)#UP-3200,IP-3200,HU-128
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        #x_a = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        #x = self.tryat(x_a)
        # print(x.shape)
        #x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])

        out = self.conv3d_1(x)
        out = self.deepconv3d_2(out)
        out = self.conv3d_3(out)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d_4(out)
        out = self.deepconv2d(out)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
###########################################################################
class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()
        #self.extra = GloRe_Unit_3D(1, 64)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            #nn.PReLU(inplace=True),
        )
        #self.extra_2 = GloRe_Unit_3D(8, 64)
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        #self.extra_4 = GloRe_Unit_3D(16, 64)
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        #self.extra = GloRe_Unit_3D(32, 64)
        #extra_1 = GloRe_Unit_2D(256, 64)
        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=(3, 3), stride=1, padding=0),#YRD-192 HU-160 UP-96 IP-576
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.ReLU(inplace=True),
        )
        #self.extra_3 = GloRe_Unit_2D(16, 64)


        #self.extra_1 = GloRe_Unit_2D(16, 48)
        self.fc1 = nn.Linear(576, 256)#UP-18496 YRD-576 HU-576 IP-18496
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        #out = self.extra(x)
        out = self.conv3d_1(x)
        #out = self.extra(out)
        out = self.conv3d_2(out)
        #out = self.extra(out)
        out = self.conv3d_3(out)
        #out = self.extra_2(out)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        #out = self.extra_1(out)
        out = self.conv2d_4(out)
        #out = self.conv2d_5(out)
        #out = self.extra_3(out)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
    ###########################################################################
################################################################
class F3D(nn.Module):
    def __init__(self):
        super(F3D, self).__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )
        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True),

        )

        self.fc1 = nn.Linear(110976, 256)#YRD-2304 HU-4800 UP-3456 IP-110976
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, class_num)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):

        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv3d_4(out)
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out
################################################################
#LSSAN
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上
class SPA(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SPA, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid =nn.Sigmoid()

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)#.permute是维度变换，permute(0, 2, 1)即将2,3维度互换

        proj_key = x.view(m_batchsize, -1, width * height).permute(0, 2, 1).permute(1, 0, 2)  # B X C x (*W*H)
        y = proj_key[[(width * height)//2+1]].permute(1, 0, 2)
        y = y.permute(0, 2, 1)

        energy =torch.cosine_similarity(proj_query.unsqueeze(2), y.unsqueeze(1), dim=3)

        attention = self.sigmoid(energy)

        proj_value = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X N

        out = proj_value * attention
        out = out.reshape(m_batchsize, C, width, height)

        #out = self.gamma * out + x
        return out
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class LSSAN(nn.Module):
    def __init__(self):
        super(LSSAN, self).__init__()

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(pca_components , pca_components , kernel_size=(3, 3), stride=1, padding=1),
            hswish(),

        )
        self.bneck_1 = nn.Sequential(
            nn.Conv2d(pca_components, 128, kernel_size=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=2,groups=128),
            SE_Block(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

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
        self.SE = SE_Block(32)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.bneck_5 = nn.Sequential(

            hswish(),
            nn.Linear(32, class_num),

        )

        self.fc = nn.Linear(32, class_num)


    def forward(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        out = self.conv2d_1(x)
        out = self.bneck_1(out)
        out = self.bneck_2(out)

        out = self.bneck_3(out)+out

        out = self.bneck_4(out)+out
        out = self.SE(out)
        out = self.GAP (out)


        out = out.reshape(out.shape[0], -1)
        out = self.bneck_5(out)
        #out = self.fc(out)
        return out
################################################################

################################################################
# MSDAN
class Attention(nn.Module):
    def __init__(self, n,nin):
        super(Attention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(1,1,kernel_size=(n, 1, 1),padding=((n-1)//2,0,0))

        self.AvgPool = nn.AvgPool3d((10,1,1))#黄河口设为（9,1,1），其余为（10,1,1）
        self.conv1 = nn.Conv3d(nin, nin, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))

        self.fc1 = nn.Linear(nin, nin // 8, bias=False)

        self.fc2 = nn.Linear(nin// 8, nin, bias=False)

    def forward(self, x):
        n1,c,l,w,h = x.shape
        se = self.sigmoid(self.conv (self.GAP(x.permute(0, 2, 1,3,4)).permute(0, 2, 1,3,4)))

        sa = self.sigmoid(self.conv1 (self.AvgPool(x)))

        x1 = self.GAP(x)
        x1 = x1.reshape(n1, -1)
        f1 = self.fc1(x1)
        f2 = self.fc2(f1)

        ca = self.sigmoid(f2)
        ca = ca.reshape(n1, c,1,1,1)

        w = se*sa*ca
        out = x*w

        return out

class Unit(nn.Module):
    def __init__(self, n):
        super(Unit, self).__init__()

        # 3D层
        self.bn1 = nn.BatchNorm3d(64)
        self.Conv3d_1 = nn.Conv3d(64, 32, kernel_size=(1, n, n), stride=1,padding=(0,(n-1)//2,(n-1)//2))
        self.bn2 = nn.BatchNorm3d(32)
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
        out1 = self.Conv3d_1(F.relu(self.bn1(x)))
        x1 = self.Conv3d_2(F.relu(self.bn2(out1)))
        out1 = torch.cat([x1, x], dim=1)

        out2 = self.Conv3d_3(F.relu(self.bn3(out1)))
        x2 = self.Conv3d_4(F.relu(self.bn4(out2)))
        out2 = torch.cat([x2, x, x1], dim=1)

        out3 = self.Conv3d_5(F.relu(self.bn5(out2)))
        x3 = self.Conv3d_6(F.relu(self.bn6(out3)))
        out3 = torch.cat([x3, x, x1,x2], dim=1)


        return out3

class MSDAN(nn.Module):
    def __init__(self):
        super(MSDAN, self).__init__()

        self.conv3d = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(2,1,1),padding=(1,1,1),)
        )
        self.bneck_1 = Unit(3)
        self.at1 = Attention(3,160)

        self.bneck_2 = Unit(5)
        self.at2 = Attention(5,160)

        self.bneck_3 = Unit(7)
        self.at3 = Attention(7,160)

        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(160),
            nn.ReLU(inplace=True),
            nn.Conv3d(160, 256, kernel_size=(10, 1, 1), stride=1,)#黄河口设为（9,1,1），其余为（10,1,1）
        )
        self.conv3d_3 = nn.Sequential(
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

        self.fc = nn.Linear(64, class_num)


    def forward(self, x):
        x = self.conv3d (x)


        x1 = self.bneck_1(x)
        x1 = self.at1(x1)

        x2 = self.bneck_2(x)
        x2 = self.at2(x2)

        x3 = self.bneck_3(x)
        x3 = self.at3(x3)

        out = x1+x2+x3

        out = self.conv3d_2(out).permute(0, 2, 1,3,4)

        out = self.conv3d_3(out)

        out = self.MaxPool(out)


        out = self.conv3d_4(out)
        out = self.GAP(out)

        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out

################################################################