import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv3x3, self).__init__()
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

#todo
class Conv3x3_wrap(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv3x3_wrap, self).__init__()
        self.bias = bias 
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.pad = nn.ZeroPad2d(1)
        self.padding=1
        
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        # out = self.pad(x)
        # if use_wrap_padding:
        if self.padding > 0:
            x = torch.cat([torch.zeros_like(x[..., :self.padding, :]), x,
                           torch.zeros_like(x[..., :self.padding, :])], dim=2)  # B16: explicit zero padding
            out = torch.cat([x[..., -self.padding:], x, x[..., :self.padding]], dim=-1)  # B16: concat wrap
        else:
            out = x
        
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


# Based on https://github.com/sunset1995/py360convert
class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), bool)
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):

        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi

        lon, lat = np.meshgrid(lon, lat)

        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)

        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)

        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])

        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)

        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):

        bs, ch, h, w = cube_feat.shape
        assert h == self.face_w and w // 6 == self.face_w

        cube_feat = cube_feat.view(bs, ch, 1,  h, w)
        cube_feat = torch.cat(torch.split(cube_feat, self.face_w, dim=-1), dim=2)

        cube_feat = cube_feat.view([bs, ch, 6, self.face_w, self.face_w])

        # Deployment breakpoint B13-v3: 5D grid_sample is unsupported in ONNX/TRT -> flatten
        # + 4x GatherElements bilinear.
        # Precomputes face id / in-face coords / 4-neighbor flat indices and weights at init;
        # matches grid_sample(align_corners=True, border padding).
        eh, ew = self.equ_h, self.equ_w
        dev = cube_feat.device
        fw = self.face_w
        tp = ((self.tp.to(dev) + 1.0) * 2.5).round().clamp(0, 5).long()   # [eh,ew] face id
        u_pix = (self.coor_u.to(dev) + 1.0) * 0.5 * (fw - 1)              # [eh,ew]
        v_pix = (self.coor_v.to(dev) + 1.0) * 0.5 * (fw - 1)
        x0 = u_pix.floor().clamp(0, fw - 1)
        y0 = v_pix.floor().clamp(0, fw - 1)
        x1 = (x0 + 1).clamp(0, fw - 1)
        y1 = (y0 + 1).clamp(0, fw - 1)
        fx = (u_pix - x0).clamp(0, 1)
        fy = (v_pix - y0).clamp(0, 1)
        # flatten 6 faces: flat_idx = (face * fw + y) * fw + x
        flat = cube_feat.view(bs, ch, 6, fw, fw).reshape(bs * ch, 6 * fw * fw)

        def gath(ys, xs):
            idx = ((tp * fw + ys.to(torch.long)) * fw + xs.to(torch.long))
            idx = idx.reshape(1, eh * ew).expand(bs * ch, eh * ew)
            return torch.gather(flat, 1, idx)

        p00 = gath(y0, x0)
        p01 = gath(y1, x0)
        p10 = gath(y0, x1)
        p11 = gath(y1, x1)
        fx_r = fx.reshape(1, eh * ew)
        fy_r = fy.reshape(1, eh * ew)
        top = p00 * (1 - fx_r) + p10 * fx_r
        bot = p01 * (1 - fx_r) + p11 * fx_r
        equi_feat = (top * (1 - fy_r) + bot * fy_r).view(bs, ch, eh, ew)
        return equi_feat


class Concat(nn.Module):
    def __init__(self, channels, **kwargs):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(channels*2, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.conv(x))
        return x


# Based on https://github.com/Yeh-yu-hsuan/BiFuse/blob/master/models/FCRN.py
class BiProj(nn.Module):
    def __init__(self, channels, **kwargs):
        super(BiProj, self).__init__()

        self.conv_c2e = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv_e2c = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv_mask = nn.Sequential(nn.Conv2d(channels * 2, 1, kernel_size=1, padding=0),
                                       nn.Sigmoid())

    def forward(self, equi_feat, c2e_feat):
        aaa = self.conv_e2c(equi_feat)
        tmp_equi = self.conv_c2e(c2e_feat)
        mask_equi = self.conv_mask(torch.cat([aaa, tmp_equi], dim=1))
        tmp_equi = tmp_equi.clone() * mask_equi
        return equi_feat + tmp_equi


# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
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


class CEELayer(nn.Module):
    def __init__(self, channels, SE=True):
        super(CEELayer, self).__init__()

        self.res_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)
        self.res_bn1 = nn.BatchNorm2d(channels)

        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.SE = SE
        if self.SE:
            self.selayer = SELayer(channels * 2)

        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.res_bn1(self.res_conv1(x)))
        shortcut = self.res_bn2(self.res_conv2(x))

        x = c2e_feat + shortcut
        x = torch.cat([equi_feat, x], 1)
        if self.SE:
            x = self.selayer(x)
        x = self.relu(self.conv(x))
        return x

