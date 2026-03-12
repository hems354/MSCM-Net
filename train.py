import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence
from timm.models.layers import DropPath, trunc_normal_
try:
    from fusion import MAFM
except:
    from .fusion import MAFM
try:
    from VSS3D import VSSLayer3D
except:
    from .VSS3D import VSSLayer3D

class Fusion(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // r, 1),
            nn.ReLU(),
            nn.Conv3d(dim // r, dim, 1),
            nn.BatchNorm3d(dim),
        )
        self.conv = nn.Conv3d(dim // 2, dim, 1)

        self.sp = SpatialAttention()
        self.ch = ChannelAttention(channel=dim//2)

    def forward(self, cnn, trans):
        cnn = self.sp(cnn)
        trans = self.ch(trans)
        x = torch.concat([cnn, trans], dim=1)
        x = self.se(x)
        x = torch.sigmoid(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1 * cnn
        x2 = x2 * trans
        x = x1 + x2
        x = self.conv(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ConvNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(ConvNormNonlin, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm3d(num_features=out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.dropout(self.conv(x))
        x = self.nonlin(self.norm(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Sequence] = 3,
                 stride: Union[int, Sequence] = 1, padding: Union[int, Sequence] = 1, groups=1, drop=0.):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvNormNonlin(in_channels, out_channels, kernel_size, stride, padding, groups, drop)
        self.conv2 = ConvNormNonlin(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, drop=drop)
        

    def forward(self, x):
        x = self.conv1(x)
        residual = self.conv2(x)
        x = x + residual
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return x*output


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU,size=[]):
        super(BasicBlock, self).__init__()
        # self.conv_block = ConvNormNonlin(dim // 2, dim // 2, drop=drop)
        self.conv_block = ConvBlock(dim // 2, dim // 2, drop=drop)
        # self.trans_blocks = ConvBlock(dim // 2, dim // 2, drop=drop)
        self.trans_blocks  = VSSLayer3D(
            dim=dim // 2, 
            depth=4, 
            attn_drop=0.1, 
            mlp_drop=0.1, 
            drop_path=drop_path, 
            norm_layer=nn.LayerNorm, 
            use_checkpoint=False, 
            d_state=64, 
            version='v5', 
            expansion_factor=1, 
            scan_type='scan', 
            size=size)   
        # self.fusion = Fusion(dim=dim) #wangyh 融合方式
        self.fusion = MAFM(inc=dim//2) #xianzhen 融合方式

    def forward(self, x):
        residual = x
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv_block(x1)
        x2 = self.trans_blocks(x2)
        x = self.fusion(x1, x2) # 特定融合
        # x = torch.cat([x1, x2], dim=1)  # 直接拼接融合
        return x+residual


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 out_dim,
                 depth,
                 num_heads,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 downsample=None,
                 use_checkpoint=False,
                 size = [],
                 ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, act_layer=act_layer, size=size,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(indim=dim, outdim=out_dim,
                                         kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, x


class Subsample(nn.Module):
    def __init__(self, indim, outdim, kernel_size, stride, padding):
        super(Subsample, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.norm = nn.BatchNorm3d(outdim)
        self.subsample = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj = nn.Conv3d(self.indim, self.outdim, 1)

    def forward(self, x):
        x = self.subsample(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, patch_size=(2, 2, 2), in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        B, C, D, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, D, H, W)
        return x


class BinaryClassificationHead(nn.Module):
    def __init__(self, input_channels, output_classes=2):
        super(BinaryClassificationHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 将 [B, C, D, H, W] -> [B, C, 1, 1, 1]
        # self.fc= nn.Sequential(
        #                         nn.Linear(input_channels, 512) ,
        #                         nn.Linear(512, 256) ,
        #                         nn.Linear(256, 128)  ,
        #                         nn.Linear(128, output_classes),
        # )
        self.fc=nn.Linear(input_channels, output_classes, bias=True)
    def forward(self, x):
        # 输入形状 [B, C, D, H, W]
        x = self.global_pool(x)  # [B, C, 1, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [B, C]
        x = self.fc(x)  # [B, output_classes]
        x = F.softmax(x, dim=1)
        return x


class LowTransformer(nn.Module):

    def __init__(self,
                 in_chans=1,
                 depths=[3, 4, 6, 3],
                #  depths=[1, 1, 1, 1],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_heads=[1, 2, 4, 8],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU,
                 patch_norm=False,
                 kernel_size=[],
                 stride=[],
                 padding=[],
                 embed_dims=[],
                 size = []
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        embed_dims.insert(0, in_chans)
        stem_len = len(kernel_size) - 4

        self.stem = nn.ModuleList()
        self.cls_head = BinaryClassificationHead(embed_dims[-1])
        for i in range(stem_len):
            self.stem.append(ConvBlock(embed_dims[i], embed_dims[i + 1], kernel_size=kernel_size[i], stride=stride[i],
                                       padding=padding[i]))

        self.dims = embed_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(stem_len, stem_len + 4):
            layer = BasicLayer(
                dim=self.dims[i_layer],
                out_dim=self.dims[i_layer + 1],
                depth=depths[i_layer - stem_len],
                num_heads=num_heads[i_layer - stem_len],
                kernel_size=self.kernel_size[i_layer],
                stride=self.stride[i_layer],
                padding=self.padding[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i_layer - stem_len]):sum(depths[:i_layer - stem_len + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=Subsample,
                size=size[i_layer - stem_len]
            )
            self.layers.append(layer)
        self.apply(_init_weights)



    def forward(self, x, print_shape=False):
        """Forward function."""
        outs = []
        if print_shape:
            print(f"初始输入shape: {x.shape}")
        for i, stem in enumerate(self.stem):
            x = stem(x)
            outs.append(x)
            if print_shape:
                print(f"Stem第{i+1}个ConvBlock输出shape: {x.shape}")

        # x = self.patch_embed(x)

        for i, layer in enumerate(self.layers):
            x_block, x = layer(x)  # x_block: block输出, x: downsample输出
            outs.append(x)
            if print_shape:
                print(f"Layers第{i+1}个BasicLayer(block)输出shape: {x_block.shape}")
                print(f"Layers第{i+1}个BasicLayer(downsample)输出shape: {x.shape}")

        out = outs[-1]

        cls_out = self.cls_head(out)
        if print_shape:
            print(f"分类头输入(downsample最后输出)shape: {x.shape}")
            print(f"分类头全局池化后shape: {self.cls_head.global_pool(x).shape}")
            print(f"分类头最终输出shape: {cls_out.shape}")
        
        return cls_out


def _init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        m.weight = nn.init.kaiming_normal_(m.weight, a=0.01)
        if m.bias is not None:
            m.bias = nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def compute(model, x):
    model.cuda()
    x = x.cuda()
    flops, params = profile(model, inputs=(x,))
    FLOPs = flops / 1000 ** 3
    # for i in range(50):
    #     model(x)
    # torch.cuda.synchronize()
    # tic1 = time.time()
    # for i in range(200):
    #     model(x)
    # torch.cuda.synchronize()
    # tic2 = time.time()
    # throughput = 200 * 1 / (tic2 - tic1)
    # latency = 1000 * (tic2 - tic1) / 200
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # print(f"throughout = {throughput}")
    # print(f"latency = {latency}ms")
    # print(f"FLOPS = {FLOPs / latency}G")
    # sys.exit()


# if __name__ == '__main__':
#     import os
#     from thop import profile
#     kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
#     stride = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
#     padding = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
#     size = [32, 16, 8, 4]
#     embed_dims = [16,32, 64, 128, 256, 320, 512]
#     print("Hello")
#     y = torch.randn(1, 1, 256, 256, 256).cuda()
#     print(y.shape)
#     encoder = LowTransformer(kernel_size=kernel_size, stride=stride, padding=padding, embed_dims=embed_dims, size=size).cuda()
#     compute(encoder, y)
#     features = encoder(y)
#     print(features.shape)
if __name__ == '__main__':
    import os
    import time
    from thop import profile

    kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    stride      = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    padding     = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    size = [32, 16, 8, 4]
    embed_dims = [16, 32, 64, 128, 256, 320, 512]

    print("Hello")

    # 输入与模型
    y = torch.randn(1, 1, 256, 256, 256).cuda()
    print(y.shape)
    encoder = LowTransformer(
        kernel_size=kernel_size, stride=stride, padding=padding,
        embed_dims=embed_dims, size=size
    ).cuda()

    # print(encoder)

    compute(encoder, y)

    # ====== 这里开始加：GPU 显存峰值（MB）======
    encoder.eval()

    torch.cuda.empty_cache()             # 可选：更干净，不想用可删
    torch.cuda.reset_peak_memory_stats() # 必须：清空峰值统计
    torch.cuda.synchronize()             # 建议：确保前面异步结束

    # with torch.no_grad():
    features = encoder(y, print_shape=True)
 
    torch.cuda.synchronize()             # 建议：确保这次 forward 完成
    gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"GPU Peak Allocated (MB): {gpu_peak_mb:.2f}")
    # ====== 加的部分结束 ======

    print(features.shape)




