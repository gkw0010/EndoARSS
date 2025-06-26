import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .Swin import SwinTransformer


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class Mlp(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d,
            drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNAct(in_features, hidden_features, kernel_size=1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
            norm_layer(hidden_features),
            act_layer())
        self.fc3 = ConvBN(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.drop(x)

        return x


class RPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rpe_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rpe_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.rpe_norm(self.rpe_conv(x))


class Stem(nn.Module):
    def __init__(self, img_dim=3, out_dim=64, rpe=True):
        super(Stem, self).__init__()
        self.conv1 = SeparableConv(img_dim, out_dim // 2, kernel_size=3, stride=2)
        self.conv2 = SeparableConv(out_dim // 2, out_dim, kernel_size=3, stride=2)
        self.rpe = rpe
        if self.rpe:
            self.proj_rpe = RPE(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.rpe:
            x = self.proj_rpe(x)
        return x


class DetailPath(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        dim1 = embed_dim // 4
        dim2 = embed_dim // 2
        self.dp1 = nn.Sequential(SeparableConv(3, dim1, stride=2),
                                 SeparableConv(dim1, dim1, stride=1))
        self.dp2 = nn.Sequential(SeparableConv(dim1, dim2, stride=2),
                                 SeparableConv(dim2, dim2, stride=1))
        self.dp3 = nn.Sequential(SeparableConv(dim2, embed_dim, stride=1),
                                 SeparableConv(embed_dim, embed_dim, stride=1))

    def forward(self, x):
        feats = self.dp1(x)
        feats = self.dp2(feats)
        feats = self.dp3(feats)

        return feats


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FM(nn.Module):
    def __init__(self, in_channel, channel):
        super(FM, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, ))
        self.beta = nn.Parameter(torch.ones(1, ))
        self.ce_a = SELayer(channel=channel)
        self.ce_ba = SELayer(channel=channel)
        self.conv = SeparableConv(in_channel, channel, kernel_size=3)
        self.conv1 = SeparableConv(channel * 2, channel, kernel_size=1)
        self.spa1 = nn.Sequential(
            Conv(channel, 1, kernel_size=3),
            nn.Sigmoid()
        )
        self.spa2 = nn.Sequential(
            Conv(channel, 1, kernel_size=3),
            nn.Sigmoid()
        )
        self.br = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(.1),
        )
        self.br1 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(.1),
        )
        self.f_conv = nn.Sequential(
            SeparableConv(channel * 2, channel, kernel_size=3),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x, z):

        f_up = self.conv(z)
        f_up = F.interpolate(f_up, x.size()[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, f_up], dim=1)
        y = self.f_conv(x).sigmoid()
        y_b = 1 - y
        x = self.conv1(x)
        f_a = x * y
        f_ba = x * y_b
        f_a = self.ce_a(f_a) * self.alpha
        f_ba = self.ce_ba(f_ba) * self.beta
        f_a = self.br(f_a)
        f_ba = self.br1(f_ba)
        out = f_up - self.spa1(f_a) * f_up
        out = out + self.spa2(f_ba) * f_up
        return out


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class multi_convolution_kernel_size(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multi_convolution_kernel_size, self).__init__()
        self.conv3 = ConvBN(in_channel, out_channel // 2, 3)
        self.conv5 = ConvBN(in_channel, out_channel // 2, 5)
        self.conv7 = ConvBN(in_channel, out_channel // 2, 7)
        self.skip_conv = ConvBN(in_channel, out_channel // 2, 1)
        self.conv_fin = DEPTHWISECONV(out_channel * 2, out_channel)
        self.se = SELayer(out_channel)
        self.act = nn.LeakyReLU(.1)

    def forward(self, x):
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_7 = self.conv7(x)
        x_skip = self.skip_conv(x)
        x = torch.cat([x_3, x_5, x_7, x_skip], dim=1)
        x = self.conv_fin(x)

        return self.act(self.se(x))


class FPN(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=256):
        super().__init__()
        self.pre_conv0 = multi_convolution_kernel_size(encoder_channels[0], decoder_channels)
        self.pre_conv1 = multi_convolution_kernel_size(encoder_channels[1], decoder_channels)
        self.pre_conv2 = multi_convolution_kernel_size(encoder_channels[2], decoder_channels)
        self.pre_conv3 = multi_convolution_kernel_size(encoder_channels[3], decoder_channels)

        self.post_conv3 = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            RPE(decoder_channels),
        )

        self.fm2 = FM(decoder_channels, decoder_channels)

        self.fm1 = FM(decoder_channels, decoder_channels)

        self.fm0 = FM(decoder_channels, decoder_channels)

    def upsample_add(self, up, x):
        up = F.interpolate(up, x.size()[-2:], mode='nearest')
        up = up + x
        return up

    def forward(self, x0, x1, x2, x3):
        x3 = self.pre_conv3(x3)
        x2 = self.pre_conv2(x2)
        x1 = self.pre_conv1(x1)
        x0 = self.pre_conv0(x0)

        x2 = self.upsample_add(x3, x2)
        x1 = self.upsample_add(x2, x1)
        x0 = self.upsample_add(x1, x0)
        x3 = self.post_conv3(x3)
        # x3_out = self.head(x3)
        x3 = F.interpolate(x3, x2.size()[-2:], mode='bilinear', align_corners=False)

        x2 = self.fm2(x3, x2)
        x2 = F.interpolate(x2, x1.size()[-2:], mode='bilinear', align_corners=False)

        x1 = self.fm1(x2, x1)

        x1 = F.interpolate(x1, x0.size()[-2:], mode='bilinear', align_corners=False)

        x0 = self.fm0(x1, x0)

        return x0


class Build(nn.Module):
    def __init__(self,
                 decoder_channels=384,
                 dims=[96, 192, 384, 768],
                 num_classes=21):
        super().__init__()
        self.stem = Stem(img_dim=3, out_dim=dims[0], rpe=True)
        self.backbone = SwinTransformer(window_size=7, embed_dim=96,
                                        depths=[2, 2, 18, 2],
                                        num_heads=[3, 6, 12, 24], ape=False,
                                        drop_path_rate=.3, patch_norm=True,
                                        use_checkpoint=False)

        encoder_channels = dims
        self.dp = DetailPath(embed_dim=decoder_channels)

        self.fpn = FPN(encoder_channels, decoder_channels)
        self.head = nn.Sequential(SeparableConv(decoder_channels, encoder_channels[0]),
                                  nn.Dropout(0.1),
                                  nn.UpsamplingBilinear2d(scale_factor=2),
                                  Conv(encoder_channels[0], num_classes, kernel_size=1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        sz = x.size()[-2:]
        dp = self.dp(x)
        x, x2, x3, x4 = self.backbone(x)
        x = self.fpn(x, x2, x3, x4)
        x = x + dp
        x = self.head(x)
        x = F.interpolate(x, sz, mode='bilinear', align_corners=False)
        return x


class Seg_Encoder(nn.Module):
    def __init__(self,
                 decoder_channels=384,
                 dims=[96, 192, 384, 768]):
        super().__init__()
        self.stem = Stem(img_dim=3, out_dim=dims[0], rpe=True)
        self.backbone = SwinTransformer(window_size=7, embed_dim=96,
                                        depths=[2, 2, 18, 2],
                                        num_heads=[3, 6, 12, 24], ape=False,
                                        drop_path_rate=.3, patch_norm=True,
                                        use_checkpoint=False)

        self.dp = DetailPath(embed_dim=decoder_channels)
        encoder_channels = dims
        self.fpn = FPN(encoder_channels, decoder_channels)

        self.apply(self._init_weights)

        pretrained_path = './work_dirs/swin_tiny_patch4_window7_224.pth'

        self.backbone.load_state_dict(torch.load(pretrained_path)['model'], strict=False)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        dp = self.dp(x)
        x, x2, x3, x4 = self.backbone(x)
        x = self.fpn(x, x2, x3, x4)
        x = x + dp
        # return x4, {'dp': dp, 'x': x, 'x2': x2, 'x3': x3}
        return x


class Seg_Decoder(nn.Module):
    def __init__(self,
                 sz,  # resolution of output
                 decoder_channels=384,
                 dims=[96, 192, 384, 768],
                 num_classes=7,
):
        super().__init__()

        encoder_channels = dims
        self.sz = sz

        # self.fpn = FPN(encoder_channels, decoder_channels)
        self.head = nn.Sequential(SeparableConv(decoder_channels, encoder_channels[0]),
                                  nn.Dropout(0.1),
                                  nn.UpsamplingBilinear2d(scale_factor=2),
                                  Conv(encoder_channels[0], num_classes, kernel_size=1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, feats=None):
        # x = self.fpn(feats['x'], feats['x2'], feats['x3'], x)
        # x = x + feats['dp']
        x = self.head(x)
        x = F.interpolate(x, self.sz, mode='bilinear', align_corners=False)
        return x


if __name__ == '__main__':
    x = torch.randn((2, 3, 512, 512))
    net = Build()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2f" % total)
    y = net(x)
    print(y.shape)