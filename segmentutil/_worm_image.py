import numpy as np
from skimage.io import imsave, imread
from skimage import img_as_ubyte, img_as_uint, img_as_bool, img_as_float32, img_as_float64, filters, measure
from skimage.morphology import skeletonize, dilation, square
from skimage.restoration import estimate_sigma, denoise_nl_means
import os
import time
from scipy.ndimage.filters import uniform_filter
from segmentutil.utils import enhance_contrast_histogram


def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size)
    c2 = uniform_filter(X * X, window_size)
    ret = c2 - c1 * c1
    ret[ret < 0] = 0
    ret[ret > 1] = 1
    return np.sqrt(ret)


def _get_mask_skeleton(src_3d_mask, out_dir, name_prefix):
    out_path = os.path.join(out_dir, '%s_rfp_skeleton.tif' % name_prefix)
    if os.path.isfile(out_path):
        img_rfp_skel = imread(out_path)
    else:
        # skeletonization version 0.1
        # get mask z max projection
        src_2d_mask = np.max(src_3d_mask, axis=0)
        kernel = np.ones((5, 5), np.uint8)
        dt = img_as_ubyte(src_2d_mask)
        for i in range(3):
            dt = dilation(dt, square(5))

        # skeletonize
        skel = skeletonize(img_as_bool(dt))

        # find all your connected components (white blobs in your image)
        output = measure.label(img_as_ubyte(skel), connectivity=2)
        props = measure.regionprops(output)
        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 30

        # your answer image
        maj_cc = np.zeros((output.shape), dtype=np.uint8)
        # for every component in the image, you keep it only if it's above min_size
        for prop in props:
            if prop.area >= min_size:
                maj_cc[output == prop.label] = 255

        # 4. skeletonize
        img_rfp_skel = img_as_bool(maj_cc)
        imsave(out_path, maj_cc)

    return img_rfp_skel


def _zero_crossing_LoG(src, kernel_size, threshold_factor):
    dst_log = np.zeros(src.shape, dtype=np.float)
    # dst_sob = np.zeros(src.shape, dtype=np.float)
    # for z in range(src.shape[0]):
    if True:
        z = 10
        dst_log[z, :, :] = filters.laplace(src[z], kernel_size)

    thres_log = (dst_log[z].max() - dst_log[z].min()) * threshold_factor
    # thres_sob = np.absolute(dst_sob[z]).mean() * 0.75

    dst_zc = np.zeros(src.shape, dtype=np.uint8)

    # for z in range(src.shape[0]):
    if True:
        z = 10
        for y in range(1, src.shape[1] - 1):
            for x in range(1, src.shape[2] - 1):
                # extract only its a real edge
                # if abs(dst_sob) < thres_sob:
                #    continue
                patch = dst_log[z, y - 1:y + 2, x - 1:x + 2]
                p = dst_log[z, y, x]
                maxP = patch.max()
                minP = patch.min()
                # extract only strong edges
                if (maxP - minP) < thres_log:
                    continue

                if p > 0:
                    zeroCross = True if minP < 0 else False
                else:
                    zeroCross = True if maxP > 0 else False
                if zeroCross:
                    dst_zc[z, y, x] = 255

    return dst_zc


def _get_denoised_fastN1Means(self, src, dir_key, file_name):
    out_path = os.path.join(self.dirs[dir_key], '%s.tif' % file_name)
    if os.path.isfile(out_path):
        img_denoised = imread(out_path)
    else:
        # if True:
        start_time = time.time()
        shape = src.shape
        img_denoised = np.zeros(shape, dtype=np.uint8)
        patch_kw = dict(patch_size=5,  # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        )
        for z in range(shape[0]):
            img_z_f = src[z, :, :]
            sigma_est = np.mean(estimate_sigma(img_z_f, multichannel=False))
            # openCV defeats skimage in performance
            img_denoised[z, :, :] = denoise_nl_means(img_z_f, h=sigma_est, fast_mode=True, **patch_kw)

        elapsed_time = time.time() - start_time
        str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('%s: denoising image. %d stacks. %s elapsed' % (self.file_name, shape[0], str_elapsed_time))

        if src.dtype != img_denoised.dtype:
            if src.dtype == np.uint16:
                img_denoised = img_as_uint(img_denoised)
            elif src.dtype == np.uint8:
                img_denoised = img_as_ubyte(img_denoised)
            elif src.dtype == np.float64:
                img_denoised = img_as_float64(img_denoised)
            else:
                raise Exception("Undefined datatype: %s" % str(src.dtype))

        imsave(out_path, img_denoised)
    return img_denoised


def _get_denoised_gaussian(self, src, dir_key, file_name):
    out_path = os.path.join(self.dirs[dir_key], '%s.tif' % file_name)
    if os.path.isfile(out_path):
        img_denoised = imread(out_path)
    else:
        # if True:
        start_time = time.time()
        shape = src.shape
        img_denoised = np.zeros(shape, dtype=np.float32)
        for z in range(shape[0]):
            img_z_f = src[z, :, :]
            img_denoised[z, :, :] = filters.gaussian(img_z_f, sigma=1.1)

        elapsed_time = time.time() - start_time
        str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('%s: denoising image. %d stacks. %s elapsed' % (self.file_name, shape[0], str_elapsed_time))

        if src.dtype != img_denoised.dtype:
            if src.dtype == np.uint16:
                img_denoised = img_as_uint(img_denoised)
            elif src.dtype == np.uint8:
                img_denoised = img_as_ubyte(img_denoised)
            elif src.dtype == np.float64:
                img_denoised = img_as_float64(img_denoised)
            else:
                raise Exception("Undefined datatype: %s" % str(src.dtype))

        if not os.path.exists(os.path.dirname(out_path)):
            os.mkdir(os.path.dirname(out_path))
        imsave(out_path, img_denoised)
    return img_denoised


def _get_contrast_enhanced(self, src, dir_key, lower_bound, upper_bound, dst_dtype, file_name):
    out_path = os.path.join(self.dirs[dir_key], '%s.tif' % file_name)
    if os.path.isfile(out_path):
        img_global_stretch = imread(out_path)
    else:
        # if True:
        start_time = time.time()
        img_global_stretch, upper_v, lower_v = enhance_contrast_histogram(src, lower_bound, upper_bound, dst_dtype)
        elapsed_time = time.time() - start_time
        str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('%s: stretched contrast. %d stacks. %s elapsed' % (self.file_name, src.shape[0], str_elapsed_time))

        if not os.path.exists(os.path.dirname(out_path)):
            os.mkdir(os.path.dirname(out_path))
        imsave(out_path, img_global_stretch)
    return img_global_stretch


def reconstruct_3d_overlay(self, dir_key):
    img_recon = np.zeros(self._img_3d_recombined.shape[:3], dtype=np.ubyte)
    img_recon_overlay = self._img_3d_recombined.copy()
    for z in range(img_recon.shape[0]):
        img_recon[z, :, :][self._img_label_pred['z'] > 0] = 80

    for y in range(img_recon.shape[1]):
        img_recon[:, y, :][self._img_label_pred['y'] > 0] += 80
        roi = img_recon_overlay[:, y, :][self._img_label_pred['y'] > 0]

    img_label_pred_x = np.transpose(self._img_label_pred['x'], (1, 0))
    for x in range(img_recon.shape[2]):
        img_recon[:, :, x][img_label_pred_x > 0] += 80

    img_recon_overlay[img_recon == 80] = (192, 192, 192)
    img_recon_overlay[img_recon == 160] = (0, 255, 255)
    img_recon_overlay[img_recon == 240] = (255, 255, 0)

    img_seed = np.zeros(self._img_3d_recombined.shape[:3], dtype=np.ubyte)
    img_seed[img_recon == 240] = 255

    # imsave(out_path, img_recon_overlay)
    # multi_slice_viewer(volumes=[self._img_3d_recombined, img_recon_overlay, img_recon],
    #                   imshow_params=[{}, {}, {'cmap': 'gray', 'vmin': 0, 'vmax': 255}, ])
    # {'cmap': 'gray', 'vmin': 0, 'vmax': 255}])
