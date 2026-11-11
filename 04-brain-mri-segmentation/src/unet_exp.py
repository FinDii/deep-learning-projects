
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """标准 UNet，支持参数化下采样/上采样"""
    def __init__(self, in_channels=3, out_channels=1, init_features=32,
                 downsample_mode='maxpool', upsample_mode='convtranspose'):
        super(UNet, self).__init__()

        def _block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        # 编码器
        enc_channels = [(in_channels, init_features),
                        (init_features, init_features*2),
                        (init_features*2, init_features*4),
                        (init_features*4, init_features*8)]

        self.enc_blocks = nn.ModuleList()
        self.down_ops = nn.ModuleList()

	# 下采样
        for i, (in_c, out_c) in enumerate(enc_channels):
            self.enc_blocks.append(_block(in_c, out_c))
            if i < len(enc_channels) - 1:
                if downsample_mode == 'maxpool':
                    down = nn.MaxPool2d(2, 2)
                elif downsample_mode == 'stride_conv':
                    down = nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1)
                else:
                    raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
                self.down_ops.append(down)
            else:
                self.down_ops.append(None)

        self.bottleneck = _block(init_features*8, init_features*16)

        # 解码器
        dec_channels = [(init_features*16, init_features*8),
                        (init_features*8, init_features*4),
                        (init_features*4, init_features*2),
                        (init_features*2, init_features)]

        self.up_ops = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i, (in_c, out_c) in enumerate(dec_channels):
            if upsample_mode == 'convtranspose':
                up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
            elif upsample_mode == 'bilinear':
                # 先卷积降通道，再上采样
                up = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1),
                    nn.UpsamplingBilinear2d(scale_factor=2)
                )
            elif upsample_mode == 'pixelshuffle':
                up = nn.Sequential(
                    nn.Conv2d(in_c, out_c * 4, kernel_size=1),
                    nn.PixelShuffle(upscale_factor=2)
                )
            else:
                raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")
            self.up_ops.append(up)

            skip_in = enc_channels[-1-i][1]
            self.dec_blocks.append(_block(out_c + skip_in, out_c))

        self.conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, down in zip(self.enc_blocks, self.down_ops):
            x = enc(x)
            skips.append(x)
            if down is not None:
                x = down(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_ops, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.conv(x))