import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from einops import rearrange
import numpy as np


# deformable conv v2
class Deformable_Conv_V2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(Deformable_Conv_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            3 * self.kernel_size * self.kernel_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation)

        self.deformable_conv = DeformConv2d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=self.padding)

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return self.deformable_conv(x, offset, mask)


# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # 保存注意力图
        # np.save('attention.npy', attn.cpu().detach().numpy())

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 保存注意结果
        np.save('attention.npy', out.cpu().detach().numpy())

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x + self.attn(to_4d(self.norm1(to_3d(x)), h, w))
        x = x + self.ffn(to_4d(self.norm2(to_3d(x)), h, w))

        return x


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // scale, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * scale, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Restormer
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,  # 原来=64
                 num_blocks=[2, 2, 2, 2],  # 原来=[2, 4, 4, 2]
                 # num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 ):
        super(Restormer, self).__init__()

        # self.feature_extraction = nn.Conv2d(in_channels=inp_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.feature_extraction = Deformable_Conv_V2(in_channels=inp_channels, out_channels=dim, kernel_size=3,
                                                     stride=1, padding=1)

        # self.patch_embed = OverlapPatchEmbed(out_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[0])])
        self.conv1_2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, stride=1, padding=0)
        self.down1_2 = Downsample(dim, 2)  # From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[1])])
        self.conv2_3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, stride=1, padding=0)
        self.down2_3 = Downsample(int(dim * 2 ** 1), 2)  # From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[2])])
        self.conv3_4 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, stride=1, padding=0)
        self.down3_4 = Downsample(int(dim * 2 ** 2), 2)  # From Level 3 to Level 4

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3), 2)  # From Level 4 to Level 3
        self.conv4_3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, stride=1, padding=0)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2), 2)  # From Level 3 to Level 2
        self.conv3_2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, stride=1, padding=0)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1), 2)  # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.conv2_1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, stride=1, padding=0)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 0), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim * 2 ** 0), out_channels, kernel_size=3, stride=1, padding=1)
        self.up = Upsample(out_channels, 4)

    def forward(self, inp_img):
        inp_feat = self.feature_extraction(inp_img)
        # np.save('dcn_feature.npy', inp_feat[0].cpu().detach().numpy())
        # inp_enc_level1 = self.patch_embed(inp_feat)
        out_enc_level1 = self.encoder_level1(inp_feat)
        inp_enc_level2 = self.conv1_2(torch.cat([out_enc_level1, inp_feat], 1))
        inp_enc_level2 = self.down1_2(inp_enc_level2)
        inp_feat2 = self.down1_2(inp_feat)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.conv2_3(torch.cat([out_enc_level2, inp_feat2], 1))
        inp_enc_level3 = self.down2_3(inp_enc_level3)
        inp_feat3 = self.down2_3(inp_feat2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.conv3_4(torch.cat([out_enc_level3, inp_feat3], 1))
        inp_enc_level4 = self.down3_4(inp_enc_level4)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self.conv4_3(torch.cat([inp_dec_level3, inp_feat3], 1))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.conv3_2(torch.cat([inp_dec_level2, inp_feat2], 1))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.conv2_1(torch.cat([inp_dec_level1, inp_feat], 1))
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return self.up(out_dec_level1)
