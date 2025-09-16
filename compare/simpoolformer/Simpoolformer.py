import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class CreatePatches(nn.Module):#用于将输入图像分割成多个小块（patches）并进行嵌入。
    def __init__(
        self, channels=15, embed_dim=256, patch_size=4#channels 是输入图像的通道数，embed_dim 是嵌入维度，patch_size 是每个小块的大小。
    ):
        super().__init__()
        self.patch = nn.Conv2d(#使用大小为 patch_size 的卷积核，步长也为 patch_size，将输入图像分割成小块并将其嵌入到 embed_dim 维的空间中。
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        #self.patch(x) 对输入图像进行卷积操作，得到分割后的小块。
#flatten(2) 沿着第 2 维对小块进行展平。
#transpose(1, 2) 交换第 1 维和第 2 维，使得输出的形状为 (batch_size, num_patches, embed_dim)。
        return patches
     

class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False):#dim 是输入特征的维度，num_heads 是注意力头的数量，qkv_bias 表示是否使用偏置，qk_scale 是缩放因子，gamma 和 use_beta 是可选的超参数。
        super().__init__()
        self.num_heads = num_heads#self.num_heads 保存注意力头的数量
        head_dim = dim // num_heads#计算每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5#是缩放因子，用于缩放注意力分数
        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)#是一个层归一化层，用于对输入特征进行归一化
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)#是线性层，用于计算查询（query）和键（key）
        if gamma is not None:#如果 gamma 不为 None，则将其转换为张量并保存。如果 use_beta 为 True，则创建一个可学习的参数 self.beta。
            self.gamma = torch.tensor([gamma])
            if use_beta:
                self.beta = nn.Parameter(torch.tensor([0.0]))
        self.eps = torch.tensor([1e-6])
        self.gamma = gamma
        self.use_beta = use_beta
    def prepare_input(self, x):#根据输入张量的维度对其进行预处理。
        if len(x.shape) == 3: # Transformer 假设是 Transformer 输入，计算全局平均池化（GAP）得到 gap_cls，并将其扩展为 (B, 1, d) 的形状。
            # Input tensor dimensions:
            # x: (B, N, d), where B is batch size, N are patch tokens, d is depth (channels)
            B, N, d = x.shape
            gap_cls = x.mean(-2) # (B, N, d) -> (B, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)
            return gap_cls, x

        if len(x.shape) == 4: # CNN 假设是 CNN 输入，同样计算 GAP，然后将输入张量展平并调整维度顺序，最后将 gap_cls 扩展为 (B, 1, d) 的形状。
            # Input tensor dimensions:
            # x: (B, d, H, W), where B is batch size, d is depth (channels), H is height, and W is width
            B, d, H, W = x.shape
            gap_cls = x.mean([-2, -1]) # (B, d, H, W) -> (B, d)
            x = x.reshape(B, d, H*W).permute(0, 2, 1) # (B, d, H, W) -> (B, d, H*W) -> (B, H*W, d)
            gap_cls = gap_cls.unsqueeze(1) # (B, d) -> (B, 1, d)

            return gap_cls, x
        else:
            raise ValueError(f"Unsupported number of dimensions in input tensor: {len(x.shape)}")
    def forward(self, x):
        # Prepare input tensor and perform GAP as initialization
        gap_cls, x = self.prepare_input(x)#对输入进行预处理，得到 gap_cls 和 x
        # Prepare queries (q), keys (k), and values (v)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)#分别是查询、键和值，其中 k 和 v 经过层归一化处理
        # Extract dimensions after normalization
        Bq, Nq, dq = q.shape
        Bk, Nk, dk = k.shape
        Bv, Nv, dv = v.shape
        # Check dimension consistency across batches and channels
        assert Bq == Bk == Bv
        assert dq == dk == dv
        # Apply linear transformation for queries and keys then reshape
        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1, 3) # (Bq, Nq, dq) -> (B, num_heads, Nq, dq/num_heads)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1, 3) # (Bk, Nk, dk) -> (B, num_heads, Nk, dk/num_heads)
        vv = v.reshape(Bv, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1, 3) # (Bv, Nv, dv) -> (B, num_heads, Nv, dv/num_heads)
        # Compute attention scores qq、kk 和 vv 分别是 q、k 和 v 经过线性变换和形状调整后的结果
        attn = (qq @ kk.transpose(-2, -1)) * self.scale#attn 是注意力分数，通过 qq 和 kk 的转置相乘并乘以缩放因子得到
        # Apply softmax for normalization
        attn = attn.softmax(dim=-1)#attn.softmax(dim=-1) 对注意力分数进行 softmax 归一化
        # If gamma scaling is used
        if self.gamma is not None:#如果 self.gamma 不为 None，则对值进行 gamma 缩放，并计算加权和。如果 self.use_beta 为 True，则添加可学习的平移项 self.beta。
            # Apply gamma scaling on values and compute the weighted sum using attention scores
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1/self.gamma) # (B, num_heads, Nv, dv/num_heads) -> (B, 1, 1, d)
            # If use_beta, add a learnable translation
            if self.use_beta:
                x = x + self.beta
        else:
            # Compute the weighted sum using attention scores
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, dq)

        return x.squeeze()#使用 squeeze 方法去除维度为 1 的维度

class Aff(nn.Module):
    def __init__(self, dim):#dim，表示输入特征的维度
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))#self.alpha 和 self.beta 是可学习的参数，分别初始化为全 1 和全 0 的张量
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
    def forward(self, x):#forward 方法是前向传播函数，对输入 x 进行仿射变换，即乘以 self.alpha 并加上 self.beta
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):#FeedForward 类实现了一个前馈神经网络。
    def __init__(self, dim, hidden_dim, dropout = 0.0):#dim 是输入和输出的维度，hidden_dim 是隐藏层的维度，dropout 是丢弃率。
        super().__init__()
        self.net = nn.Sequential(#self.net 是一个顺序容器，包含两个线性层、一个 GELU 激活函数和两个丢弃层。
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):#MLPblock 类实现了一个多层感知机块
    def __init__(self, dim, num_patch, mlp_dim, dropout = 0.0, init_values=1e-4):#dim 是输入特征的维度，num_patch 是小块的数量，mlp_dim 是前馈神经网络的隐藏层维度，dropout 是丢弃率，init_values 是初始化值。
        super().__init__()
        self.pre_affine = Aff(dim)#self.pre_affine 和 self.post_affine 是仿射变换层
        self.token_mix = nn.Sequential(#self.token_mix 是一个顺序容器，包含 Rearrange 层和线性层，用于对小块进行混合。
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(#self.ff 是一个前馈神经网络
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)#self.gamma_1 和 self.gamma_2 是可学习的参数，用于控制残差连接的强度
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x
#首先，通过 self.pre_affine 进行仿射变换。
#然后，将 self.token_mix(x) 乘以 self.gamma_1 并加到 x 上，实现残差连接。
#接着，通过 self.post_affine 进行仿射变换。
#最后，将 self.ff(x) 乘以 self.gamma_2 并加到 x 上，再次实现残差连接
class ResMLP(nn.Module):#ResMLP 类实现了一个基于多层感知机的残差网络
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim):#in_channels 是输入图像的通道数，dim 是嵌入维度，num_classes 是分类的类别数，patch_size 是小块的大小，image_size 是图像的大小，
        #depth 是 MLP 块的数量，mlp_dim 是前馈神经网络的隐藏层维度。
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'#检查图像大小是否能被小块大小整除
        self.num_patch =  (image_size// patch_size) ** 2#self.num_patch 计算小块的数量
        self.to_patch_embedding = nn.Sequential(#顺序容器，包含一个卷积层和 Rearrange 层，用于将输入图像分割成小块并进行嵌入。
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))
        self.affine = Aff(dim)#仿射变换层
        self.mlp_head = nn.Sequential(#是一个线性层，用于将特征映射到 mlp_dim 维的空间中
            nn.Linear(dim, mlp_dim)
        )
    def forward(self, x):
        x = self.to_patch_embedding(x)# self.to_patch_embedding 将输入图像分割成小块并进行嵌入
        for mlp_block in self.mlp_blocks:#self.mlp_blocks 中的 MLP 块进行处理
            x = mlp_block(x)
        x = self.affine(x)#self.affine 进行仿射变换
        x = x.mean(dim=1)
        return self.mlp_head(x)#对特征在第 1 维上取平均值，并通过 self.mlp_head 进行线性变换
     

class AttentionBlock(nn.Module):#实现了一个注意力块
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):#embed_dim 是嵌入维度，hidden_dim 是前馈神经网络的隐藏层维度，num_heads 是注意力头的数量，dropout 是丢弃率
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)#self.pre_norm 和 self.norm 是层归一化层
        self.simpool = SimPool(embed_dim, num_heads=1, qkv_bias=False, qk_scale=None, gamma=None, use_beta=False#self.simpool 是 SimPool 类的实例，用于实现简单的池化机制
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(#self.MLP 是一个前馈神经网络
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)
        x = x + self.simpool(x_norm)[0]
        x = x + self.MLP(self.norm(x))
        return x
     #forward 方法是前向传播函数，对输入 x 进行处理。
#首先，通过 self.pre_norm 对输入进行层归一化。
#然后，将 self.simpool(x_norm)[0] 加到 x 上，实现残差连接。
#接着，通过 self.norm 对 x 进行层归一化，并将 self.MLP(self.norm(x)) 加到 x 上，再次实现残差连接
class SimPoolFormer(nn.Module):
    def __init__(#接收多个参数，包括图像大小、输入通道数、小块大小、嵌入维度、隐藏层维度、注意力头数量、层数、丢弃率、分类类别数、深度和 MLP 维度。
        self,
        img_size=8,
        in_channels=15,
        patch_size=2,
        embed_dim=256,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        num_classes=18,
        depth=4,
        mlp_dim=256
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size//patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        self.resmlp= ResMLP(in_channels, embed_dim, num_classes, patch_size, img_size, depth, mlp_dim)
        #SimPoolFormer 类是整个模型的核心类。
#self.patch_size 保存小块的大小。
#num_patches 计算小块的数量。
#self.patches 是 CreatePatches 类的实例，用于将输入图像分割成小块并进行嵌入。
#self.pos_embedding 是位置编码，用于为每个小块添加位置信息。
#self.cls_token 是分类令牌，用于分类任务。self.attn_layers 是一个 ModuleList，包含多个 AttentionBlock。self.dropout 是丢弃层，用于防止过拟合。
#self.ln 是层归一化层。self.head 是一个线性层，用于将特征映射到分类类别数的空间中。self.apply(self._init_weights) 调用 _init_weights 方法对模型的参数进行初始化。
#self.resmlp 是 ResMLP 类的实例，用于提取特征。
    def _init_weights(self, m):#_init_weights 方法用于初始化模型的参数
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                #如果 m 是线性层，使用截断正态分布初始化权重，标准差为 0.02，如果偏置不为 None，则将偏置初始化为 0。
#如果 m 是层归一化层，将偏置初始化为 0，权重初始化为 1.0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x1 = self.resmlp(x)
        x = self.patches(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x.mean(dim=1)
     #  x = x[:, 0]
        x= x + x1
        return self.head(x)
     #forward 方法是前向传播函数，对输入 x 进行处理。
#首先，通过 self.resmlp(x) 提取特征并保存为 x1。
#然后，通过 self.patches(x) 将输入图像分割成小块并进行嵌入。
#ls_tokens 是分类令牌，扩展为与输入批次大小相同的形状。
#将分类令牌和小块嵌入结果在第 1 维上拼接。加上位置编码。
#通过 self.dropout 进行丢弃操作。
#依次通过 self.attn_layers 中的注意力块进行处理。
#通过 self.ln 进行层归一化。
#对特征在第 1 维上取平均值。
#将 x 和 x1 相加。
#最后，通过 self.head 进行线性变换，得到分类结果。

if __name__ == '__main__':
    model = SimPoolFormer(
        img_size=9,
        in_channels=12,
        patch_size=3,
        embed_dim=256,
        hidden_dim= 128,
        num_heads=4,
        num_layers=2,
        num_classes=10,
        depth=2
    )
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    rnd_int = torch.randn(1, 12, 9, 9)
    output = model(rnd_int)
    print(f"Output shape from model: {output.shape}")