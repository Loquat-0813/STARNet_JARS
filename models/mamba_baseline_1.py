import torch
import torch.nn as nn
import torch.nn.functional as F
class LightUnit_trans(nn.Module):
    """优化后的动态特征增强单元（提升通道利用率）"""
    def __init__(self, in_channels):
        super().__init__()
        # 增强分支表达能力
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (1,5,5),  # 扩大感受野
                     padding=(0,2,2), groups=in_channels, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, max(4, in_channels//8), 1),  # 动态调整压缩比
            nn.BatchNorm3d(max(4, in_channels//8)),
            nn.SiLU(),
            nn.Conv3d(max(4, in_channels//8), in_channels, 1)
        )
        # 动态特征校准
        self.alpha = nn.Parameter(torch.ones([]))  # 初始值调整为1
        self.topk_ratio = 0.6  # 调整top-k比例

    def forward(self, x):
        attn = self.branch1(x) * self.branch2(x)
        B, C, D, H, W = attn.shape
        
        # 动态阈值生成
        k = max(1, int(self.topk_ratio * C))
        topk_val = torch.topk(attn.view(B,C,-1), k, dim=1)[0][:,-1,:]
        mask = (attn >= topk_val.view(B,1,D,H,W)).float()
        
        return x + self.alpha * (attn * mask)

class LightUnit(nn.Module):
    """优化后的动态特征增强单元（提升通道利用率）"""
    def __init__(self, in_channels):
        super().__init__()
        # 增强分支表达能力
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (1,5,5),  # 扩大感受野
                     padding=(0,2,2), groups=in_channels, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, max(4, in_channels//8), 1),  # 动态调整压缩比
            nn.BatchNorm3d(max(4, in_channels//8)),
            nn.SiLU(),
            nn.Conv3d(max(4, in_channels//8), in_channels, 1)
        )
        # 动态特征校准
        self.alpha = nn.Parameter(torch.ones([]))  # 初始值调整为1
        self.topk_ratio = 0.6  # 调整top-k比例

    def forward(self, x):
        attn = self.branch1(x) * self.branch2(x)
        B, C, D, H, W = attn.shape
        
        # 动态阈值生成
        k = max(1, int(self.topk_ratio * C))
        topk_val = torch.topk(attn.view(B,C,-1), k, dim=1)[0][:,-1,:]
        mask = (attn >= topk_val.view(B,1,D,H,W)).float()
        
        return x + self.alpha * (attn * mask)


class RCM(nn.Module):
    """Rectangular Self-Calibration Module"""
    def __init__(self, channels):
        super().__init__()
        # 轴向全局上下文捕获
        self.h_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.v_pool = nn.AdaptiveAvgPool3d((None, None, 1))
        
        # 形状自校准（大核卷积）
        self.calibrate = nn.Sequential(
            nn.Conv3d(channels, channels, (1,11,1), padding=(0,5,0), groups=channels),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, (1,1,11), padding=(0,0,5), groups=channels),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm3d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        h_ctx = self.h_pool(x)
        v_ctx = self.v_pool(x)
        attn = self.calibrate(h_ctx + v_ctx)
        return self.fusion(x) * attn + x

class DPGHead(nn.Module):
    """修正后的动态原型引导头"""
    def __init__(self, in_channels, class_num):
        super().__init__()
        # 调整投影层适应2D输入
        self.proj = nn.Linear(in_channels, class_num)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # 输入形状应为[B, features]
        B, C = x.shape  # 正确解包2D形状
        
        # 生成动态原型
        cls_feat = self.proj(x)  # [B, num_cls]
        proto = self.pool(cls_feat.unsqueeze(2))  # [B, num_cls, 1]
        return F.normalize(proto.squeeze(2), dim=1)  # [B, num_cls]
class ExplicitBlock(nn.Module):
    """特征处理基础模块（需正确定义）"""
    def __init__(self, in_ch, out_ch, unit_num=2):
        super().__init__()
        # 构建LightUnit序列
        self.units = nn.Sequential(*[
            LightUnit(in_ch) for _ in range(unit_num)
        ])
        # 矩形自校准模块
        self.rcm = RCM(in_ch)
        # 通道变换卷积
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1,3,3),  # 保持空间维度处理
            padding=(0,1,1)
        )
    def forward(self, x):
        # 顺序处理流程
        x = self.units(x)  # 通过多个LightUnit
        x = self.rcm(x)    # 矩形自校准
        return self.conv(x) # 通道变换
class baseNet(nn.Module):
    """修复后的网络结构"""
    def __init__(self, in_channels=128, class_num=10):
        super().__init__()
        # 输入处理模块（保持不变）
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, (1,3,3),
                     padding=(0,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2))
        )
        # 使用正确定义的ExplicitBlock
        self.block_1 = ExplicitBlock(32, 64)  
        self.block_2 = ExplicitBlock(64, 128)  
        self.block_3 = ExplicitBlock(128, 32)  
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            DPGHead(32, class_num)
        )

    def forward(self, x):
        # 保持原有维度处理逻辑
        if x.dim() == 4:
            x = x.unsqueeze(2)
        elif x.dim() ==5 and x.size(1)!=self.conv3d[0].in_channels:
            x = x.permute(0,2,1,3,4)
        
        x = self.conv3d(x)
        x = self.block_1(x)  # [32→64]
        x = self.block_2(x)  # [64→128]
        x = self.block_3(x)  # [128→32]
        return self.classifier(x)
if __name__ == '__main__':
    # 测试不同输入通道情况
    model1 = baseNet(in_channels=12, class_num=10)
    x1 = torch.randn(2, 12, 9, 9)  # 4D输入
    print(model1(x1).shape)  # 应输出[2,10]

    model2 = baseNet(in_channels=24, class_num=10)
    x2 = torch.randn(2, 1, 24, 9, 9)  # 5D输入
    print(model2(x2).shape)  # 应输出[2,10]