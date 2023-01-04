import torch
import torch.nn as nn


def DownBlock_3D(in_ch, out_ch, kernel_size=3, padding=1, drop_prob=0.5):
    block = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
                          , nn.LeakyReLU()
                          , nn.BatchNorm3d(out_ch)
                          , nn.Dropout(p=drop_prob)
                          , nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
                          , nn.LeakyReLU()
                          , nn.BatchNorm3d(out_ch)
                          , nn.Dropout(p=drop_prob))
    return block


def DownBlock_3D_Atrous(in_ch, out_ch, kernel_size=3, padding=2, drop_prob=0.5, dilation=2, padding_mode='replicate'):
    block = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, padding_mode=padding_mode)
                          , nn.BatchNorm3d(out_ch)
                          , nn.LeakyReLU()
                          , nn.Dropout(p=drop_prob)
                          , nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, padding_mode=padding_mode)
                          , nn.BatchNorm3d(out_ch)
                          , nn.LeakyReLU()
                          , nn.Dropout(p=drop_prob))
    return block


def UpBlock_2D(in_ch, out_ch, kernel_size=3, padding=1, drop_prob=0.5):
    block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
                          , nn.LeakyReLU()
                          , nn.BatchNorm2d(out_ch)
                          , nn.Dropout(drop_prob)
                          , nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
                          , nn.LeakyReLU()
                          , nn.BatchNorm2d(out_ch)
                          , nn.Dropout(p=drop_prob))
    return block


def UpBlock_2D_Atrous(in_ch, out_ch, kernel_size=3, padding=2, drop_prob=0.5, padding_mode='replicate', dilation=2):
    block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dilation=dilation)
                          , nn.BatchNorm2d(out_ch)
                          , nn.LeakyReLU()
                          , nn.Dropout(drop_prob)
                          , nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dilation=dilation)
                          , nn.BatchNorm2d(out_ch)
                          , nn.LeakyReLU()
                          , nn.Dropout(p=drop_prob))
    return block


def UpSample(in_ch, scale=2):
    up = nn.Sequential(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
                       , nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, padding=1))
    return up


def MakeBridge(in_ch, num_planes):
    bridge = nn.Conv3d(in_ch, in_ch, kernel_size=(num_planes, 1, 1), stride=1, padding=0)
    return bridge


def DownAvgPool(kernel_size=(1, 2, 2)):
    return nn.AvgPool3d(kernel_size=kernel_size)


def LastLayer(in_ch, out_cl, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_ch, out_cl, kernel_size=kernel_size, stride=stride, padding=padding)


def DownRes(in_ch, out_ch, kernel_size=1, stride=1, padding=0):
    residual = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    return residual


def UpRes(in_ch, out_ch, kernel_size=1, stride=1, padding=0):
    residual = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    return residual


def PointConv_2D(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)


class UNET25D_Atrous(nn.Module):
    def __init__(self, num_channels=1, num_classes=2, num_filters=(16, 32, 64, 128, 256), num_planes=7, border=10, drop=0.5):  # these are the default values
        super().__init__()
        filters = num_filters
        self.info = {'name': 'UNET25D_BASIC', 'planes': num_planes, 'channels': num_channels, 'classes': num_classes, 'size': 128, 'max_filters': filters[-1], 'drop prob':drop}
        self.border = border

        ## DOWN PATH ########################

        # Down level 0
        self.down0 = DownBlock_3D(num_channels, filters[0], drop_prob=drop)
        # self.down0res = DownRes(num_channels, filters[0])
        self.bridge0 = MakeBridge(filters[0], num_planes)
        self.down0pool = DownAvgPool()

        # Down level 1
        self.down1 = DownBlock_3D(filters[0], filters[1], drop_prob=drop)
        # self.down1res = DownRes(filters[0], filters[1])
        self.bridge1 = MakeBridge(filters[1], num_planes)
        self.down1pool = DownAvgPool()

        # Down level 2
        self.down2 = DownBlock_3D_Atrous(filters[1], filters[2], drop_prob=drop)
        # self.down2res = DownRes(filters[1], filters[2])
        self.bridge2 = MakeBridge(filters[2], num_planes)
        self.down2pool = DownAvgPool()

        # Down level 3
        self.down3 = DownBlock_3D_Atrous(filters[2], filters[3], drop_prob=drop)
        # self.down3res = DownRes(filters[2], filters[3])
        self.bridge3 = MakeBridge(filters[3], num_planes)
        self.down3pool = DownAvgPool()

        # Down level 4
        self.down4 = DownBlock_3D_Atrous(filters[3], filters[4], drop_prob=drop)
        # self.down4res = DownRes(filters[3], filters[4])
        self.bridge4 = MakeBridge(filters[4], num_planes)
        self.down4pool = DownAvgPool()

        self._3D_to_2D = nn.Conv3d(filters[4], filters[4], kernel_size=(num_planes, 1, 1), stride=1, padding=0)

        ## UP PATH ###################

        # Up level 4
        self.upsample4 = UpSample(filters[4])
        self.up4 = UpBlock_2D_Atrous(filters[3] + filters[4], filters[3], drop_prob=drop)
        # self.up4res = UpRes(filters[3] + filters[4], filters[3])

        # Up level 3
        self.upsample3 = UpSample(filters[3])
        self.up3 = UpBlock_2D_Atrous(filters[2] + filters[3], filters[2], drop_prob=drop)
        # self.up3res = UpRes(filters[2] + filters[3], filters[2])

        # Up level 2
        self.upsample2 = UpSample(filters[2])
        self.up2 = UpBlock_2D(filters[1] + filters[2], filters[1], drop_prob=drop)
        # self.up2res = UpRes(filters[1] + filters[2], filters[1])

        # Up level 1
        self.upsample1 = UpSample(filters[1])
        self.up1 = UpBlock_2D(filters[0] + filters[1], filters[0], drop_prob=drop)
        # self.up1res = UpRes(filters[0] + filters[1], filters[0])

        self.finalconv = LastLayer(filters[0], num_classes)

    def forward(self, t):
        ## DOWN PATH ##########################

        # Down level 0
        t0 = self.down0(t)
        # res0 = self.down0res(t)
        # t0 = t0 + res0
        bridge0 = self.bridge0(t0).squeeze(2)
        t0_to_1 = self.down0pool(t0)

        # Down level 1
        t1 = self.down1(t0_to_1)
        # res1 = self.down1res(t0_to_1)
        # t1 = t1 + res1
        bridge1 = self.bridge1(t1).squeeze(2)
        t1_to_2 = self.down1pool(t1)

        # Down level 2
        t2 = self.down2(t1_to_2)
        # res2 = self.down2res(t1_to_2)
        # t2 = t2 + res2
        bridge2 = self.bridge2(t2).squeeze(2)
        t2_to_3 = self.down2pool(t2)

        # Down level 3
        t3 = self.down3(t2_to_3)
        # res3 = self.down3res(t2_to_3)
        # t3 = t3 + res3
        bridge3 = self.bridge3(t3).squeeze(2)
        t3_to_4 = self.down3pool(t3)

        # Down level 4
        t4 = self.down4(t3_to_4)
        # res4 = self.down4res(t3_to_4)
        # t4 = t4 + res4
        t4 = self._3D_to_2D(t4).squeeze(dim=2)

        ## UP PATH #########################

        # Up level 4
        t4_up = self.upsample4(t4)
        t4_up_cat = torch.cat((bridge3, t4_up), dim=1)
        t4 = self.up4(t4_up_cat)
        # res4up = self.up4res(t4_up_cat)
        t4_to_t3 = t4

        # Up level 3
        t3_up = self.upsample3(t4_to_t3)
        t3_up_cat = torch.cat((bridge2, t3_up), dim=1)
        t3 = self.up3(t3_up_cat)
        # res3up = self.up3res(t3_up_cat)
        t3_to_t2 = t3

        # Up level 2
        t2_up = self.upsample2(t3_to_t2)
        t2_up_cat = torch.cat((bridge1, t2_up), dim=1)
        t2 = self.up2(t2_up_cat)
        # res2up = self.up2res(t2_up_cat)
        t2_to_t1 = t2

        # Up level 1
        t1_up = self.upsample1(t2_to_t1)
        t1_up_cat = torch.cat((bridge0, t1_up), dim=1)
        t1 = self.up1(t1_up_cat)
        # res1up = self.up1res(t1_up_cat)
        t1 = t1

        out = self.finalconv(t1)
        out = out[:, :, self.border:-self.border, self.border:-self.border]

        return out


class UNet_Multi_Scale(nn.Module):
    def __init__(self, num_channels=1, num_classes=3, num_filters=(16, 32, 64, 128, 256), num_planes=7, border=10):  # these are the default values
        super(UNet_Multi_Scale, self).__init__()
        filters = num_filters
        self.info = {'name': 'UNETMultiScaleUpConCat', 'planes': num_planes, 'channels': num_channels, 'classes': num_classes, 'max_filters': filters[-1]}
        self.border = border

        ## DOWN PATH ########################

        # Down level 0
        self.down0 = DownBlock_3D(num_channels, filters[0])
        self.down0res = DownRes(num_channels, filters[0])
        self.bridge0 = MakeBridge(filters[0], num_planes)
        self.down0pool = DownAvgPool()

        # Down level 1
        self.down1 = DownBlock_3D(filters[0], filters[1])
        self.down1res = DownRes(filters[0], filters[1])
        self.bridge1 = MakeBridge(filters[1], num_planes)
        self.down1pool = DownAvgPool()

        # Down level 2
        self.down2 = DownBlock_3D(filters[1], filters[2])
        self.down2res = DownRes(filters[1], filters[2])
        self.bridge2 = MakeBridge(filters[2], num_planes)
        self.down2pool = DownAvgPool()

        # Down level 3
        self.down3 = DownBlock_3D(filters[2], filters[3])
        self.down3res = DownRes(filters[2], filters[3])
        self.bridge3 = MakeBridge(filters[3], num_planes)
        self.down3pool = DownAvgPool()

        # Down level 4
        self.down4 = DownBlock_3D(filters[3], filters[4])
        self.down4res = DownRes(filters[3], filters[4])
        self.bridge4 = MakeBridge(filters[4], num_planes)
        self.down4pool = DownAvgPool()

        self._3D_to_2D = nn.Conv3d(filters[4], filters[4], kernel_size=(num_planes, 1, 1), stride=1, padding=0)

        ## UP PATH ###################

        # Up level 4
        self.upsample4 = UpSample(filters[4])
        self.up4 = UpBlock_2D(filters[3] + filters[4], filters[3])
        self.up4res = UpRes(filters[3] + filters[4], filters[3])

        # Up level 3
        self.upsample3 = UpSample(filters[3])
        self.up3 = UpBlock_2D(filters[2] + filters[3], filters[2])
        self.up3res = UpRes(filters[2] + filters[3], filters[2])

        # Up level 2
        self.upsample2 = UpSample(filters[2])
        self.up2 = UpBlock_2D(filters[1] + filters[2], filters[1])
        self.up2res = UpRes(filters[1] + filters[2], filters[1])

        # Up level 1
        self.upsample1 = UpSample(filters[1])
        self.up1 = UpBlock_2D(filters[0] + filters[1], filters[0])
        self.up1res = UpRes(filters[0] + filters[1], filters[0])

        self.finalconv = LastLayer(filters[0], num_classes)

        ## DOWN PATH GREEN ######################
        self._3D_to_2D_g = nn.Conv3d(filters[2], filters[2], kernel_size=(num_planes, 1, 1), stride=1, padding=0)
        self.conv_r_g = PointConv_2D(filters[2]*2, filters[2])

        ## UP PATH GREEN ########################

        # Up level 2
        self.up2_g = UpBlock_2D(filters[1]*2 + filters[2], filters[1])
        self.up2res_g = UpRes(filters[1]*2 + filters[2], filters[1])

        # Up level 1
        self.up1_g = UpBlock_2D(filters[0]*2 + filters[1], filters[0])
        self.up1res_g = UpRes(filters[0]*2 + filters[1], filters[0])

    def forward(self, t, x):
        ## DOWN PATH RED ##########################

        # Down level 0
        t0 = self.down0(t)
        res0 = self.down0res(t)
        t0 = t0 + res0
        bridge0 = self.bridge0(t0).squeeze(2)
        t0_to_1 = self.down0pool(t0)

        # Down level 1
        t1 = self.down1(t0_to_1)
        res1 = self.down1res(t0_to_1)
        t1 = t1 + res1
        bridge1 = self.bridge1(t1).squeeze(2)
        t1_to_2 = self.down1pool(t1)

        # Down level 2
        t2 = self.down2(t1_to_2)
        res2 = self.down2res(t1_to_2)
        t2 = t2 + res2
        bridge2 = self.bridge2(t2).squeeze(2)
        t2_to_3 = self.down2pool(t2)

        # Down level 3
        t3 = self.down3(t2_to_3)
        res3 = self.down3res(t2_to_3)
        t3 = t3 + res3
        bridge3 = self.bridge3(t3).squeeze(2)
        t3_to_4 = self.down3pool(t3)

        # Down level 4
        t4 = self.down4(t3_to_4)
        res4 = self.down4res(t3_to_4)
        t4 = t4 + res4
        t4 = self._3D_to_2D(t4).squeeze(dim=2)

        ## UP PATH RED #########################

        # Up level 4
        t4_up = self.upsample4(t4)
        t4_up_cat = torch.cat((bridge3, t4_up), dim=1)
        t4 = self.up4(t4_up_cat)
        res4up = self.up4res(t4_up_cat)
        t4_to_t3 = t4 + res4up

        # Up level 3
        t3_up = self.upsample3(t4_to_t3)
        t3_up_cat = torch.cat((bridge2, t3_up), dim=1)
        t3 = self.up3(t3_up_cat)
        res3up = self.up3res(t3_up_cat)
        t3_to_t2 = t3 + res3up

        # Up level 2
        t2_up = self.upsample2(t3_to_t2)
        t2_up_cat = torch.cat((bridge1, t2_up), dim=1)
        t2 = self.up2(t2_up_cat)
        res2up = self.up2res(t2_up_cat)
        t2_to_t1 = t2 + res2up

        # Up level 1
        t1_up = self.upsample1(t2_to_t1)
        t1_up_cat = torch.cat((bridge0, t1_up), dim=1)
        t1 = self.up1(t1_up_cat)
        res1up = self.up1res(t1_up_cat)
        t1 = t1 + res1up

        out_red = self.finalconv(t1)
        out_red = out_red[:, :, self.border:-self.border, self.border:-self.border]

        ## DOWN PATH GREEN ########################

        # Down level 0
        x0 = self.down0(x)
        res0_g = self.down0res(x)
        x0 = x0 + res0_g
        bridge0_g = self.bridge0(x0).squeeze(2)
        x0_to_1 = self.down0pool(x0)

        # Down level 1
        x1 = self.down1(x0_to_1)
        res1_g = self.down1res(x0_to_1)
        x1 = x1 + res1_g
        bridge1_g = self.bridge1(x1).squeeze(2)
        x1_to_2 = self.down1pool(x1)

        # Down level 2
        x2 = self.down2(x1_to_2)
        res2_g = self.down2res(x1_to_2)
        x2 = x2 + res2_g
        x2 = self._3D_to_2D_g(x2).squeeze(dim=2)
        x2_cat = torch.cat((t3_to_t2, x2), dim=1)
        x2_cat_conv = self.conv_r_g(x2_cat)

        ## UP PATH GREEN ########################

        # UP level 2
        x2_up = self.upsample2(x2_cat_conv)
        x2_up_cat = torch.cat((t2_to_t1, bridge1_g, x2_up), dim=1)
        x2 = self.up2_g(x2_up_cat)
        res2up_g = self.up2res_g(x2_up_cat)
        x2_to_x1 = x2 + res2up_g

        # UP level 1
        x1_up = self.upsample1(x2_to_x1)
        x1_up_cat = torch.cat((t1, bridge0_g, x1_up), dim=1)
        x1 = self.up1_g(x1_up_cat)
        res1up_g = self.up1res_g(x1_up_cat)
        x1 = x1 + res1up_g

        out_green = self.finalconv(x1)
        out_green = out_green[:, :, self.border:-self.border, self.border:-self.border]

        return out_red, out_green

