import os
import sys
import torch
import numpy as np
import imageio as io
import time
import glob
from segmentutil.unet.models import UNET25D_Atrous, UNet_Multi_Scale


@torch.no_grad()
def Predict_Patch(patch, model):
    result = model(patch)  # Do the prediction
    result = result.cpu().numpy()
    if result.shape[1] > 1:
        result = np.argmax(result, 1)  # One-hot prediction
    else:
        result = result > 0  # Binary prediction
    return result


def Prediction_Mask(red_ch, model, out_path=None, output_type='rgb', thr=1.2, border=10, ps=128, num_planes=7):
    """
    Predict the neurite mask.
    :param red_ch: neurite channel, should be 3D single channel with shape (z, y, x)
    :type red_ch: np.ndarray
    :param model: the trained model used for prediction
    :type model: torch.nn.Module
    :param out_path: output path to write the image
    :type out_path: str
    :param output_type: types of output, rgb for visualization or BW for binary mask.
    :type output_type: str
    :param thr: threshold to filter out blank patches
    :type thr: float
    :param border: boundary border to be excluded to increase the accuracy
    :type border: int
    :param ps: patch size, default=128.
    :type ps: int
    :param num_planes: number of nearby planes for context
    :type num_planes: int
    :return: binary mask, with foreground 255
    :rtype: np.ndarray
    """
    time1 = time.time()
    try:
        assert red_ch.ndim == 3, 'the input should be single channel'
        z, y, x = red_ch.shape
    except IOError:
        print('the red channel should be a numpy array')
        z, y, x = (0,) * 3

    # test if cuda is available. If not, use cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean_red, std_red = np.mean(red_ch), np.std(red_ch)

    # Define dimensions with tiling operations
    real_stride = ps - 2 * border
    plane = int(0.5 * (num_planes - 1))
    y_max = y - np.mod(y, real_stride) + ps
    x_max = x - np.mod(x, real_stride) + ps
    pMat = np.zeros((z, y_max, x_max), dtype=np.uint8)

    # Pad images
    red_ch = np.pad(red_ch, ((plane, plane), (border, y_max - y + border), (border, x_max - x + border)), mode='reflect')

    model.eval().to(device)
    for i in range(0, z):
        for j in range(0, y, real_stride):
            for k in range(0, x, real_stride):  # loop over each patch
                red_patch = red_ch[i:i + num_planes, j:j + ps, k: k + ps].astype(np.float32)
                if np.max(red_patch[plane, ...]) > mean_red * thr:  # Use a threshold to filter out blank patches
                    ori_patch = (red_patch - mean_red) / std_red  # normalize per image stack
                    patch = torch.from_numpy(ori_patch).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                    p_result = Predict_Patch(patch, model)
                    pMat[i, j:j + real_stride, k:k + real_stride] = p_result

    pMat = pMat[:, 0:y, 0:x]

    if output_type == 'rgb':
        rgbMat = np.zeros(pMat.shape[:3] + (3,), dtype=red_ch.dtype)
        rgbMat[..., 0] = red_ch[plane:z + plane, border:y + border, border:x + border]
        rgbMat[..., 2][pMat == 1] = np.iinfo(red_ch.dtype).max  # Define a display value.
        p_image = rgbMat
    else:
        pMat[pMat == 1] = np.iinfo(np.uint8).max
        p_image = pMat

    if out_path:
        io.mimwrite(out_path, p_image)

    time2 = time.time()
    print(out_path, '| time spent: {}s'.format(time2 - time1))

    return pMat


def Prediction_MultiChannel(red_ch, green_ch, model, thr=1.2, border=10, ps=128, num_planes=7):
    assert red_ch.ndim == 3, 'input must be single channel'
    z, y, x = red_ch.shape

    red = red_ch
    green = green_ch
    pMat = np.zeros_like(red)
    pMat_red = np.zeros_like(pMat)
    pMat_green = np.zeros_like(pMat)
    real_stride = ps - 2 * border

    plane = int(0.5 * (num_planes - 1))
    y_max = y - np.mod(y, real_stride) + ps
    x_max = x - np.mod(x, real_stride) + ps

    red_padded = np.pad(red, ((0, 0), (border, y_max - y + border), (border, x_max - x + border)), 'mean')
    green_padded = np.pad(green, ((0, 0), (border, y_max - y + border), (border, x_max - x + border)), 'mean')
    pMat_large_red = np.zeros((pMat.shape[0], y_max, x_max))
    pMat_large_green = np.zeros((pMat.shape[0], y_max, x_max))
    z_min, z_max = int(0.5 * (num_planes - 1)), int(z - 0.5 * (num_planes - 1))
    mean_green = np.mean(green)
    std_green = np.std(green)
    mean_red = np.mean(red)
    std_red = np.std(red)

    model.cuda()
    model.eval()
    for i in range(z_min, z_max):
        for j in range(border, y, real_stride):
            for k in range(border, x, real_stride):
                redPatch = red_padded[i - plane: i + plane + 1, j - border:j + ps - border, k - border: k + ps - border].astype(np.float32)
                greenPatch = green_padded[i - plane: i + plane + 1, j - border:j + ps - border, k - border: k + ps - border].astype(np.float32)
                if np.max(redPatch[plane, ...]) > mean_red * thr:
                    redPatch = (redPatch - mean_red) / std_red
                    greenPatch = (greenPatch - mean_green) / std_green
                    piece_red = torch.from_numpy(redPatch).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                    piece_green = torch.from_numpy(greenPatch).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                    result_red, result_green = model(piece_red, piece_green)
                    result_red = result_red.cpu().detach().numpy()
                    p_result_red = np.argmax(result_red, 1)
                    pMat_large_red[i, j:j + real_stride, k:k + real_stride] = p_result_red
                    result_green = result_green.cpu().detach().numpy()
                    p_result_green = np.argmax(result_green, 1)
                    pMat_large_green[i, j:j + real_stride, k:k + real_stride] = p_result_green

    pMat_red[...] = pMat_large_red[:pMat.shape[0], border:pMat.shape[1] + border, border:pMat.shape[2] + border]
    pMat_green[...] = pMat_large_green[:pMat.shape[0], border:pMat.shape[1] + border, border:pMat.shape[2] + border]

    return pMat_red, pMat_green


if __name__ == '__main__':
    pass
    # image = np.array(io.mimread('xxx.tif'))
    # neurite = image[..., 0]
    # synapse = image[..., 1]
    # model = UNet_Multi_Scale()
    # model.load_state_dict(torch.load('xxx.pt'))
    # neurite_binary_mask, synapse_binary_mask = Prediction_MultiChannel(neurite, synapse, model)
    

