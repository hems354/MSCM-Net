import torch.nn as nn
import torch

class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv3d(inc, self.outc, kernel_size=k, padding=p, groups=inc)  # 3D depthwise convolution
        self.conv1_1 = nn.Conv3d(inc, self.outc, kernel_size=1, stride=1)  # 1x1x1 convolution
        self.bn1 = nn.BatchNorm3d(self.outc)
        self.bn2 = nn.BatchNorm3d(self.outc)
        self.bn3 = nn.BatchNorm3d(self.outc)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = self.bn1(x)
        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))
        return self.act(shortcut + x_dw + x_conv1_1)

class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            # local_conv = nn.Conv3d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
            #                        padding=(1 + i), stride=1,
            #                        groups=dim // self.ca_num_heads)
            # setattr(self, f"local_conv_{i + 1}", local_conv)
            dilation_i = 1 + 2 * i
            local_conv = nn.Conv3d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=3,
                                   padding=dilation_i, stride=1, dilation=dilation_i,
                                   groups=dim // self.ca_num_heads)
            
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv3d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm3d(dim * expand_ratio)
        self.proj1 = nn.Conv3d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        v = self.v(x)

        s = self.s(x).reshape(B, D, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(4, 0, 5, 1, 2, 3)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, D, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, D, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)

        x = s_out * v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc

        self.pre_att = nn.Sequential(
            nn.Conv3d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.BatchNorm3d(inc * 2),
            nn.GELU(),
            nn.Conv3d(inc * 2, inc, kernel_size=1),
            nn.BatchNorm3d(inc),
            nn.GELU()
        )

        self.attention = MHMC(dim=inc)
        self.coi = COI(inc)

        self.pw = nn.Sequential(
            nn.Conv3d(in_channels=inc, out_channels=inc*2, kernel_size=1, stride=1),
            nn.BatchNorm3d(inc*2),
            nn.GELU()
        )

    def forward(self, x, d):
        B, C, D, H, W = x.shape
        x_cat = torch.cat((x, d), dim=1)

        x_pre = self.pre_att(x_cat)

        x_reshape = x_pre.flatten(2).permute(0, 2, 1)
        attention = self.attention(x_reshape, D, H, W)
        attention = attention.permute(0, 2, 1).reshape(B, C, D, H, W)

        x_conv = self.coi(attention)
        x_conv = self.pw(x_conv)

        return x_conv

if __name__ == '__main__':
    mafm = MAFM(inc=64)

    x = torch.randn(1, 64, 32, 32, 32)  # 3D input
    d = torch.randn(1, 64, 32, 32, 32)  # 3D depth map

    output = mafm(x, d)

    print(f"Input x shape: {x.shape}")
    print(f"Input d shape: {d.shape}")
    print(f"Output shape: {output.shape}")
