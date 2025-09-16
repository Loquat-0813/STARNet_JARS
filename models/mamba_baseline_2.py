import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DPGHead(nn.Module):
    def __init__(self, in_ch, mid_ch, class_num, pool, fusions):
        super(DPGHead, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = in_ch
        self.planes = mid_ch
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

        self.inclassifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inplanes, class_num),
        )
    
    def spatial_pool(self, x):

        batch, channel, height, width = x.size()
        if self.pool == 'att':
            #[N, D, C, 1]
            input_x = x
            input_x = input_x.view(batch, channel, height*width) # [N, D, C]
            input_x = input_x.unsqueeze(1) # [N, 1, D, C]

            context_mask = self.conv_mask(x) # [N, 1, C, 1]
            context_mask = context_mask.view(batch, 1, height*width) # [N, 1, C]
            context_mask = self.softmax(context_mask) # [N, 1, C]
            context_mask = context_mask.unsqueeze(3) # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)# [N, 1, D, 1]
            context = context.view(batch, channel, 1, 1) # [N, D, 1, 1]
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        if x.dim() == 5:
            x = x.squeeze(2)
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))# [N, D, 1, 1]
            out = x * channel_mul_term # [N, D, H, W]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)# [N, D, 1, 1]
            out = out + channel_add_term
        
        out = self.inclassifier(out)
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

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

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
        self.attn = Attention(in_channels, 2, bias=False)

    def forward(self, x):
        x = self.branch1(x) # torch.Size([2, 32, 1, 4, 4])
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = self.attn(x)
        x = rearrange(x, 'b (c t) h w -> b c t h w', t=T)
        return x


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

class TransUnitStack(nn.Module):
    """
    将 LightUnit_trans 和 RCM 串联，可以获取每一层的输出
    """
    def __init__(self, in_ch, unit_num=2):
        super().__init__()
        # 创建多个 LightUnit_trans + RCM 组合
        self.units = nn.ModuleList()
        for _ in range(unit_num):
            self.units.append(nn.ModuleDict({
                'light_unit': LightUnit_trans(in_ch),
                'rcm': RCM(in_ch)
            }))
            
    def forward(self, x):
        intermediate_outputs = []
        intermediate_outputs.append(x)
        current_x = x
        for unit in self.units:
            light_out = unit['light_unit'](current_x)
            rcm_out = unit['rcm'](light_out)
            current_x = rcm_out
            intermediate_outputs.append(rcm_out)
        
        return intermediate_outputs

class UnitStack(nn.Module):
    """
    将 LightUnit_trans 单元串联，同时实现与输入列表的残差连接
    """
    def __init__(self, in_ch, unit_num=2):
        super().__init__()
        self.units = nn.ModuleList()
        for _ in range(unit_num):
            self.units.append(nn.ModuleDict({
                'light_unit': LightUnit_trans(in_ch),
            }))
        self.unit_num = unit_num
            
    def forward(self, x):
        current_x = x[-1]
        
        for i in range(self.unit_num-1, -1, -1):
            current_x = self.units[i]['light_unit'](current_x)
            if i > 0:  
                residual_index = i
            else: 
                residual_index = 0
            current_x = current_x + x[residual_index]
        
        return current_x

class ExplicitBlock(nn.Module):
    """特征处理基础模块（需正确定义）"""
    def __init__(self, in_ch, out_ch, unit_num=2):
        super().__init__()
        self.transunits = TransUnitStack(in_ch, unit_num)
        self.resunits = UnitStack(in_ch, unit_num)
        # 通道变换卷积
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1,3,3),  # 保持空间维度处理
            padding=(0,1,1)
        )
    def forward(self, x):
        # 顺序处理流程
        x = self.transunits(x)
        x = self.resunits(x)
        return self.conv(x) # 通道变换

class DPGSHead(nn.Module):
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
        self.classifier = DPGHead(32, 64, class_num, pool='avg', fusions=['channel_add', 'channel_mul'])
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     DPGSHead(32, class_num)
        # )

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

    model2 = baseNet(in_channels=24, class_num=10)
    x2 = torch.randn(2, 1, 24, 9, 9)  # 5D输入
    print(model2(x2).shape)  # 应输出[2,10]