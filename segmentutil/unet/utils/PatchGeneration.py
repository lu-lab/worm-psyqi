import numpy as np
import imageio as io
import os
import shutil
import h5py as h5
import glob


def Neurite_Patch_h5(ori_img, lab_img, ps, num_planes, destination_dir, format='alternate'):
    """
    Find patches for synapse with labels.
    :param format: order of original images coming from microscopy, RGRG alternating
    :param num_planes: number of planes
    :param ps: patch size
    :param ori_img: original image path
    :param lab_img: label image path or rle path
    :param destination_dir: the directory of the output folder
    :return: None
    """

    global blue_channel, red_channel
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
    count = len(os.listdir(destination_dir))

    lab = np.array(io.mimread(lab_img, memtest=False))
    if lab.ndim == 4:
        lab = lab[..., 2]
    blue_channel = np.zeros_like(lab)
    blue_channel[lab != 0] = 1
    red_channel = np.array(io.mimread(ori_img, memtest=False))

    # assert green_channel.shape == blue_channel.shape, 'the shapes do not match'

    red_mean, red_std = np.mean(red_channel), np.std(red_channel)

    plane = int(num_planes//2)

    z_max, y, x = red_channel.shape
    y_pad = (1 + y//ps)*ps - y
    x_pad = (1 + x//ps)*ps - x

    red_channel_padded = np.pad(red_channel, ((0, 0), (0, y_pad), (0, x_pad)), 'mean')
    blue_channel_padded = np.pad(blue_channel, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)

    cord_set = set()
    all_points = np.argwhere(blue_channel != 0)
    [cord_set.add((c[0], c[1]//ps, c[2]//ps)) for c in all_points]
    print(cord_set)
    for points in cord_set:
        # print(points)
        z, y, x = points[0], int(points[1]), int(points[2])
        if plane < z < z_max - plane:
            ori_patch_red = red_channel_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice = blue_channel_padded[z, y*ps: (y+1)*ps, x*ps: (x+1)*ps]
            lab_slice[lab_slice != 0] = 1

            h5_name = os.path.join(destination_dir, '{}.h5'.format(count))
            f = h5.File(h5_name, 'w')
            f.create_dataset(name='ori', data=ori_patch_red)
            f.create_dataset(name='lab', data=lab_slice)
            f.create_dataset(name='mean', data=red_mean)
            f.create_dataset(name='std', data=red_std)

            count += 1
            f.close()
    print(count)

    return None


def Neurite_Patch_h5_Selected(ori_img, lab_img, ps, num_planes, destination_dir, paint=None):
    """
    Find patches for synapse with labels.
    :param paint: order of original images coming from microscopy, RGRG alternating
    :param num_planes: number of planes
    :param ps: patch size
    :param ori_img: original image path
    :param lab_img: label image path or rle path
    :param destination_dir: the directory of the output folder
    :return: None
    """

    global blue_channel, red_channel
    if os.path.isdir(destination_dir):
        shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)
    count = 0

    red = np.array(io.mimread(ori_img, memtest=False))
    red_channel = red
    blue = np.array(io.mimread(lab_img, memtest=False))
    blue_channel = blue
    # paint_rgb = np.array(io.mimread(paint, memtest=False))
    # paint_img = paint_rgb[..., 2]
    # print(red_channel.shape, blue_channel.shape)

    # assert green_channel.shape == blue_channel.shape, 'the shapes do not match'

    red_mean, red_std = np.mean(red_channel), np.std(red_channel)
    print(red_mean, red_std)
    # red_channel = red_channel.astype(np.float32)
    # red_channel = (red_channel - red_mean) / red_std

    plane = int(num_planes//2)

    z_max, y, x = red_channel.shape
    # print(red_channel.shape)
    y_pad = (1 + y//ps)*ps - y
    x_pad = (1 + x//ps)*ps - x

    red_channel_padded = np.pad(red_channel, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=red_mean)
    blue_channel_padded = np.pad(blue_channel, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)
    # paint_padded = np.pad(paint_img, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)
    red_reuse = red_channel_padded.copy()

    root_name = ori_img.split('/')[-1].split('.')[0]

    total_num = (red_channel_padded.shape[1]//ps)*(red_channel_padded.shape[2]//ps)*(red_channel_padded.shape[0] - 2*plane)
    cord_set = set()
    # cord_set2 = set()
    all_points = np.argwhere(blue_channel_padded != 0)
    [cord_set.add((c[0], c[1]//ps, c[2]//ps)) for c in all_points]
    # all_points_2 = np.argwhere(blue_channel_padded != 0)
    # [cord_set2.add((c[0], c[1] // ps, c[2] // ps)) for c in all_points_2]
    # print(cord_set == cord_set2)
    for points in cord_set:
        # print(points)
        z, y, x = points[0], int(points[1]), int(points[2])
        if plane < z < z_max - plane:

            ori_patch_red = red_channel_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice = blue_channel_padded[z, y*ps: (y+1)*ps, x*ps: (x+1)*ps]
            lab_slice[lab_slice != 0] = 1
            # paint_slice = paint_padded[z, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            if len(np.argwhere(lab_slice)) > 10:
                h5_name = os.path.join(destination_dir, '{}_{}.h5'.format(root_name, count))
                f = h5.File(h5_name, 'w')
                f.create_dataset(name='ori', data=ori_patch_red)
                f.create_dataset(name='lab', data=lab_slice)
                f.create_dataset(name='mean', data=red_mean)
                f.create_dataset(name='std', data=red_std)
                red_reuse[z, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps] = 0
                print(points, np.mean(ori_patch_red[plane]), np.max(ori_patch_red[plane]))
                # io.mimwrite('/home/admin-kzhang91/222/565656/' + '{}.tif'.format(count), ori_patch_red)
                count += 1
                f.close()
    # io.mimwrite('fffffff.tif', ori_patch_red)
    thr_amount = 1
    thr = 3
    zmin = plane
    zmax = z_max - plane
    ymax = red_channel_padded.shape[1]
    xmax = red_channel_padded.shape[2]
    # print(ymax, x)
    # print(np.max(red_reuse), np.max(red_channel), red_mean)
    while thr_amount < np.int(count*0.4) and thr > 0:
        thr_amount = 0
        for i in range(zmin, zmax):
            for j in range(0, ymax, ps):
                for k in range(0, xmax, ps):
                    oriPatch = red_reuse[i - plane: i + plane + 1, j:j + ps, k: k + ps]
                    # print(oriPatch.shape)
                    # greater than e.g. 10% of the mean value
                    # print(i, j, k, np.max(oriPatch[plane, ...]))
                    if np.mean(oriPatch[plane, ...]) > red_mean*thr:
                        thr_amount += 1
        # print('------------------------------------------')
        # print(thr, thr_amount)
        thr = thr - 0.01
    addi_count = 1
    for i in range(zmin, zmax):
        for j in range(0, ymax, ps):
            for k in range(0, xmax, ps):
                oriPatch = red_reuse[i - plane: i + plane + 1, j:j + ps, k: k + ps]
                # print(oriPatch.shape)
                # greater than e.g. 10% of the mean value
                # print(i, j, k, np.max(oriPatch[plane, ...]))
                if np.mean(oriPatch[plane, ...]) > red_mean * (thr + 0.01):
                    # print(i, j, k, np.mean(oriPatch[plane, ...]))
                    lab_slice = blue_channel_padded[i, j:j + ps, k: k + ps]
                    lab_slice[lab_slice != 0] = 1
                    h5_name = os.path.join(destination_dir, '{}_{}.h5'.format(root_name, count + addi_count))
                    f = h5.File(h5_name, 'w')
                    f.create_dataset(name='ori', data=oriPatch)
                    f.create_dataset(name='lab', data=lab_slice)
                    f.create_dataset(name='mean', data=red_mean)
                    f.create_dataset(name='std', data=red_std)
                    # print(points, np.mean(ori_patch_red[plane]), np.max(ori_patch_red[plane]))
                    f.close()
                    addi_count += 1
                    # print(i, j, k)
                    red_reuse[i, j:j + ps, k: k + ps] = 0
    print(thr, addi_count)
    print(count, total_num)
    io.mimwrite('ffff.tif', red_reuse)
    return None


def Neurite_Patch_h5_YZ(ori_img, lab_img, ps, num_planes, destination_dir, format='alternate'):
    """
    Find patches for synapse with labels.
    :param format: order of original images coming from microscopy, RGRG alternating
    :param num_planes: number of planes
    :param ps: patch size
    :param ori_img: original image path
    :param lab_img: label image path or rle path
    :param destination_dir: the directory of the output folder
    :return: None
    """

    global blue_channel, red_channel
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
    count = len(os.listdir(destination_dir))

    lab = np.array(io.mimread(lab_img, memtest=False))
    if lab.ndim == 4:
        lab = lab[..., 2]
    blue_channel = np.zeros_like(lab)
    blue_channel[lab != 0] = 1
    red_channel = np.array(io.mimread(ori_img, memtest=False))

    def Pad_Z(mat, ps, mode):
        # z, y, x = mat.shape

        mean_value = np.mean(mat)
        mat_T = np.transpose(mat, axes=(2, 0, 1))
        z_max, y, x = mat_T.shape
        y_pad = (1 + y // ps) * ps - y
        x_pad = (1 + x // ps) * ps - x
        half_y_pad = y_pad//2
        half_x_pad = x_pad//2
        # print(half_y_pad, y_pad)
        if mode == 'ori':
            mat_T_padded = np.pad(mat_T, ((0, 0), (half_y_pad, y_pad - half_y_pad), (half_x_pad, x_pad - half_x_pad)), mode='constant', constant_values=mean_value)
        else:
            mat_T_padded = np.pad(mat_T, ((0, 0), (half_y_pad, y_pad - half_y_pad), (half_x_pad, x_pad - half_x_pad)), mode='constant', constant_values=0)
        return mat_T_padded
    # assert green_channel.shape == blue_channel.shape, 'the shapes do not match'

    red_channel_padded = Pad_Z(red_channel, ps=ps, mode='ori')
    blue_channel_padded = Pad_Z(blue_channel, ps=ps, mode='lab')

    red_mean, red_std = np.mean(red_channel), np.std(red_channel)

    plane = int(num_planes//2)

    y, z_max, x = red_channel.shape
    # print(y, z_max, x)
    # print(red_channel_padded.shape)
    cord_set = set()
    # print(red_channel.shape, red_channel_padded.shape)
    all_points = np.argwhere(blue_channel_padded != 0)
    [cord_set.add((c[0], c[1]//ps, c[2]//ps)) for c in all_points]
    # print(cord_set)
    for points in cord_set:
        # print(points)
        z, y, x = points[0], int(points[1]), int(points[2])
        if plane < z < z_max - plane:
            # print(y, (y + 1)*ps)
            ori_patch_red = red_channel_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            # print(ori_patch_red.shape)
            lab_slice = blue_channel_padded[z, y*ps: (y+1)*ps, x*ps: (x+1)*ps]
            lab_slice[lab_slice != 0] = 1
            if len(np.argwhere(lab_slice)) > 15:
                h5_name = os.path.join(destination_dir, '{}.h5'.format(count))
                f = h5.File(h5_name, 'w')
                f.create_dataset(name='ori', data=ori_patch_red)
                f.create_dataset(name='lab', data=lab_slice)
                f.create_dataset(name='mean', data=red_mean)
                f.create_dataset(name='std', data=red_std)

                count += 1
                f.close()
    print(count)

    return None


def Neurite_Patch_h5_Multi_Scale(ori_red, lab_red, ori_green, lab_green, ps, num_planes, destination_dir, format='alternate'):

    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
    count = len(os.listdir(destination_dir))

    if lab_red.ndim == 4:
        lab_red = lab_red[..., 2]
        lab_red[lab_red != 0] = 1
    if lab_green.ndim == 4:
        lab_green = lab_green[..., 2]
        lab_green[lab_green != 0] = 1

    red_mean, red_std, green_mean, green_std = np.mean(ori_red), np.std(ori_red), np.mean(ori_green), np.std(ori_green)

    plane = int(num_planes//2)

    z_max, y, x = ori_red.shape
    y_pad = (1 + y//ps)*ps - y
    x_pad = (1 + x//ps)*ps - x

    red_ori_padded = np.pad(ori_red, ((0, 0), (0, y_pad), (0, x_pad)), 'mean')
    green_ori_padded = np.pad(ori_green, ((0, 0), (0, y_pad), (0, x_pad)), 'mean')
    red_lab_padded = np.pad(lab_red, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)
    green_lab_padded = np.pad(lab_green, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)

    cord_set = set()
    all_points = np.argwhere(lab_red != 0)
    [cord_set.add((c[0], c[1]//ps, c[2]//ps)) for c in all_points]
    print(cord_set)
    for points in cord_set:
        # print(points)
        z, y, x = points[0], int(points[1]), int(points[2])
        if plane <= z < z_max - plane:
            ori_patch_red = red_ori_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice_red = red_lab_padded[z, y*ps: (y+1)*ps, x*ps: (x+1)*ps]
            ori_patch_green = green_ori_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice_green = green_lab_padded[z, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            if len(np.argwhere(lab_slice_red)) > 15:
                h5_name = os.path.join(destination_dir, '{}.h5'.format(count))
                f = h5.File(h5_name, 'w')
                f.create_dataset(name='ori', data=np.stack((ori_patch_red, ori_patch_green), axis=0))
                f.create_dataset(name='lab', data=np.stack((lab_slice_red, lab_slice_green), axis=0))
                f.create_dataset(name='red_mean', data=red_mean)
                f.create_dataset(name='red_std', data=red_std)
                f.create_dataset(name='green_mean', data=green_mean)
                f.create_dataset(name='green_std', data=green_std)

                count += 1
                f.close()
    print(count)

    return None


def Neurite_Patch_h5_Multi_Scale_2(ori_red, lab_red, ori_green, lab_green, ps, num_planes, destination_dir):

    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
    count = len(os.listdir(destination_dir))

    if lab_red.ndim == 4:
        lab_red = lab_red[..., 2]
        lab_red[lab_red != 0] = 1
    if lab_green.ndim == 4:
        lab_green = lab_green[..., 2]
        lab_green[lab_green != 0] = 1

    red_mean, red_std, green_mean, green_std = np.mean(ori_red), np.std(ori_red), np.mean(ori_green), np.std(ori_green)

    plane = int(num_planes//2)

    z_max, y, x = ori_red.shape
    y_pad = (1 + y//ps)*ps - y
    x_pad = (1 + x//ps)*ps - x

    red_ori_padded = np.pad(ori_red, ((0, 0), (0, y_pad), (0, x_pad)), 'mean')
    green_ori_padded = np.pad(ori_green, ((0, 0), (0, y_pad), (0, x_pad)), 'mean')
    red_lab_padded = np.pad(lab_red, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)
    green_lab_padded = np.pad(lab_green, ((0, 0), (0, y_pad), (0, x_pad)), mode='constant', constant_values=0)

    cord_set = set()
    all_points = np.argwhere(lab_red != 0)
    [cord_set.add((c[0], c[1]//ps, c[2]//ps)) for c in all_points]
    print(cord_set)
    for points in cord_set:
        # print(points)
        z, y, x = points[0], int(points[1]), int(points[2])
        if plane <= z < z_max - plane:
            ori_patch_red = red_ori_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice_red = np.zeros((ps, ps), dtype=np.uint8)
            ori_patch_green = green_ori_padded[z - plane: z + plane + 1, y * ps: (y + 1) * ps, x * ps: (x + 1) * ps]
            lab_slice_green = np.zeros((ps, ps), dtype=np.uint8)
            if len(np.argwhere(lab_slice_red)) > -1:
                h5_name = os.path.join(destination_dir, '{}.h5'.format(count))
                f = h5.File(h5_name, 'w')
                f.create_dataset(name='ori', data=np.stack((ori_patch_red, ori_patch_green), axis=0))
                f.create_dataset(name='lab', data=np.stack((lab_slice_red, lab_slice_green), axis=0))
                f.create_dataset(name='red_mean', data=red_mean)
                f.create_dataset(name='red_std', data=red_std)
                f.create_dataset(name='green_mean', data=green_mean)
                f.create_dataset(name='green_std', data=green_std)

                count += 1
                print('xiba')
                f.close()
    print(count)

    return None



if __name__ == '__main__':
    dst = '/home/admin-kzhang91/777/19_cla-1'
    o = np.array(io.mimread('/home/admin-kzhang91/omero/files/cla-1/Image {}.tif'.format(19)))
    o_r = o[2::2]
    o_g = o[1::2]
    l_r = np.array(io.mimread('/home/admin-kzhang91/Pictures/Image 19.tif', memtest=False))
    l_g = np.array(io.mimread('/home/admin-kzhang91/Pictures/Image 19.tif', memtest=False))
    Neurite_Patch_h5_Multi_Scale_2(o_r, l_r, o_g, l_g, ps=128, num_planes=7, destination_dir=dst)
