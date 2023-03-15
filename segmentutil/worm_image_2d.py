import logging
import os
import time
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Callable
import re

import h5py
import matplotlib.colors as mcolors
import numpy as np
import scipy
from czifile import CziFile
from joblib import Parallel, delayed
from scipy import ndimage
from skimage import filters
from skimage import img_as_float32, img_as_ubyte
from skimage import measure
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.morphology import disk, diamond, erosion, closing, binary_dilation
from skimage.segmentation import watershed
from sklearn.metrics import confusion_matrix
from tifffile import imsave, TiffFile
from nd2reader import ND2Reader

from segmentutil._worm_image import window_stdev, _get_denoised_fastN1Means, _get_denoised_gaussian, \
    _get_contrast_enhanced
from segmentutil.unet import visualization
from segmentutil.unet.models import UNET25D_Atrous, UNet_Multi_Scale


class DotFeature(Enum):
    SYNAPSE = 0
    GRADIENT = 1
    DOG_1 = 2
    DOG_2 = 3
    MEAN_5 = 4
    STD_5 = 5
    MEAN_11 = 6
    STD_11 = 7
    LOG = 8
    ERODE = 9
    CELL = 10
    GRADIENT_R = 11
    DOG_1_R = 12
    DOG_2_R = 13
    MEAN_5_R = 14
    STD_5_R = 15
    MEAN_11_R = 16
    STD_11_R = 17
    LOG_R = 18
    ERODE_R = 19
    MAX = 20


PROB_FEATURES = 7


class WormImage(object):
    def _init_vars(self):
        self.shape = None
        self._axis_ind = dict(z=(1, 2), y=(0, 2), x=(1, 0))
        self._avg_label_area = None

    def _init_dirs(self, base_dir, **kwargs):
        self.dirs = dict()
        if 'dir_channel' in kwargs:
            self.dirs['channel'] = os.path.join(base_dir, kwargs['dir_channel'])
        else:
            self.dirs['channel'] = os.path.join(base_dir, 'channel')

        if 'dir_denoise' in kwargs:
            self.dirs['denoise'] = os.path.join(base_dir, kwargs['dir_denoise'])
        # else:
        #    self.dirs['denoise'] = os.path.join(base_dir, 'denoise')

        if 'dir_contrast' in kwargs:
            self.dirs['contrast'] = os.path.join(base_dir, kwargs['dir_contrast'])
        # else:
        #    self.dirs['contrast'] = os.path.join(base_dir, 'contrast')

        if 'dir_mask' in kwargs:
            self.dirs['mask'] = os.path.join(base_dir, kwargs['dir_mask'])
        else:
            self.dirs['mask'] = os.path.join(base_dir, 'mask')

        if 'dir_label' in kwargs:
            self.dirs['label'] = os.path.join(base_dir, kwargs['dir_label'])
        else:
            self.dirs['label'] = os.path.join(base_dir, 'label')

        if 'dir_prediction' in kwargs:
            self.dirs['prediction'] = os.path.join(base_dir, kwargs['dir_prediction'])
        else:
            self.dirs['prediction'] = os.path.join(base_dir, 'prediction')

        # for out_dir in self.dirs.values():
        #     if not os.path.exists(out_dir):
        #         os.makedirs(out_dir)

    def _read_img(self, array, axes, **kwargs):
        i_x = axes.index('X')
        i_y = axes.index('Y')
        if 'Z' in axes:
            i_z = axes.index('Z')
        else:
            i_z = 0
        if 'C' in axes:
            # multichannel
            i_c = axes.index('C')
            tp_axes = [i_z, i_y, i_x, i_c]
            for i in range(len(axes)):
                if i not in tp_axes:
                    tp_axes.append(i)
            self._img_3d = np.transpose(array, tp_axes).squeeze()
            if self.is_singlechannel:
                self._img_3d = self._img_3d.reshape(self._img_3d.shape[:3])
        else:
            img_temp = np.transpose(array, (i_z, i_y, i_x))
            if self.is_singlechannel:
                self._img_3d = img_temp
            else:
                # singlechannel -> multichannel
                num_ch = len(kwargs['channel'])
                z_size = int(img_temp.shape[0] / num_ch)
                self._img_3d = np.zeros(shape=(z_size, img_temp.shape[1], img_temp.shape[2], 3),
                                        dtype=img_temp.dtype)
                for ch in range(num_ch):
                    self._img_3d[:, :, :, ch] = img_temp[z_size * ch:z_size * (ch + 1), :, :]

        if 'scaling_xyz' in kwargs:
            self._scaling_x = kwargs['scaling_xyz'][0]
            self._scaling_y = kwargs['scaling_xyz'][1]
            self._scaling_z = kwargs['scaling_xyz'][2]

    def _read_tif(self, file_raw, **kwargs):
        if file_raw is not None:
            with TiffFile(file_raw) as tif:
                axes = tif.series[0].axes
                array = tif.asarray()
                if 'Info' in tif.imagej_metadata:
                    metadata = tif.imagej_metadata['Info']
                    if metadata is not None:
                        scaling_axis = re.findall('Scaling\|Distance\|Id #(\d) = (.+)\n', metadata)
                        scaling_value = re.findall('Scaling\|Distance\|Value #(\d) = (.+)\n', metadata)
                        map_id_axis = {}
                        scaling_xyz = [None, None, None]
                        for s_axis in scaling_axis:
                            map_id_axis[s_axis[0]] = s_axis[1]
                        for s_value in scaling_value:
                            s_ax = map_id_axis[s_value[0]]
                            if s_ax.lower() == 'x':
                                scaling_xyz[0] = float(s_value[1])
                            elif s_ax.lower() == 'y':
                                scaling_xyz[1] = float(s_value[1])
                            elif s_ax.lower() == 'z':
                                scaling_xyz[2] = float(s_value[1])
                        if None not in scaling_xyz:
                            kwargs['scaling_xyz'] = tuple(scaling_xyz)

                self._read_img(array, axes, **kwargs)

    def _read_czi(self, file_raw, **kwargs):
        """
        Read Carl Zeiss Image data file
        Args:
            file_raw: path of the czi image file to read
            **kwargs:

        Returns:

        """
        if file_raw is not None:
            with CziFile(file_raw) as czi:
                axes = czi.axes
                array = czi.asarray()

                # read metadata to get scaling information
                metadata = ET.ElementTree(ET.fromstring(czi.metadata()))
                # metadata as XML element tree
                for meta in metadata.iter("Metadata"):
                    for scaling in meta.iter("Scaling"):
                        for item in scaling.iter("Items"):
                            for dist in item.iter("Distance"):
                                axis = dist.get("Id")
                                val = float(dist.find("Value").text)
                                if axis.lower() == 'x':
                                    scaling_x = val
                                elif axis.lower() == 'y':
                                    scaling_y = val
                                elif axis.lower() == 'z':
                                    scaling_z = val
                self._read_img(array, axes, scaling_xyz=(scaling_x, scaling_y, scaling_z), **kwargs)

                # temporarily - read preprocessed red channel as red channel
                # self.logger.info('Replace red channel with preprocessed (adaptive equalization) image.')
                # self._img_3d[..., self.ch_cell] = imread(os.path.join(os.path.dirname(self.dirs['mask']), 'preprocessing',
                #                                         '%s_red_adaptive_equalization_005.tif' % self.file_name))
                # self.logger.info('Replace red channel with preprocessed (contrast stretching) image.')
                # self._img_3d[..., self.ch_cell] = imread(os.path.join(os.path.dirname(self.dirs['mask']), 'preprocessing',
                #                                         '%s_red_contrast_stretching_995.tif' % self.file_name))
                # self.logger.info('Replace red channel with preprocessed (histogram equalization) image.')
                # self._img_3d[..., self.ch_cell] = imread(os.path.join(os.path.dirname(self.dirs['mask']), 'preprocessing',
                #                                         '%s_red_histogram_equalization.tif' % self.file_name))

    def _read_nd2(self, file_raw, **kwargs):
        """
        Read Nikon NIS-Elements ND2 image file
        Args:
            file_raw: path of the nd2 image file to read
            **kwargs:

        Returns:

        """
        if file_raw is not None:
            with ND2Reader(file_raw) as nd2:
                axes = ''.join(nd2.axes)
                nd2.bundle_axes = axes
                array = nd2[0]
                axes = axes.upper()
                # read metadata to get scaling factors
                s_xy = nd2.metadata['pixel_microns']
                s_z = nd2.metadata['z_coordinates'][1] - nd2.metadata['z_coordinates'][0]
                self._read_img(array, axes.upper(), scaling_xyz=(s_xy, s_xy, s_z), **kwargs)

    def __init__(self, base_dir, file_raw, **kwargs):
        """
        if the datatype of raw image is uint16, process it in float32 and save it in uint16.
        if the datatype of raw image is uint8, process and save it in uint8.
        :param base_dir:
        :param file_raw:
        :param impaint_colorbar:
        :param kwargs:
            'logger': Pre-defined logger to put messages
            'channel': A text indicating the channel key of raw image. 'N' for neuronal marker, 'S' for synaptic marker.
            'minimize_storage': If True, do not save some intermediate products as file
        """
        self._init_vars()
        self._init_dirs(base_dir, **kwargs)

        # logger
        if 'logger' in kwargs:
            self.logger: logging.Logger = kwargs['logger']
        else:
            self.logger = logging.getLogger()

        if file_raw is not None:
            self.file_name = os.path.splitext(os.path.basename(file_raw))[0]
            extension = os.path.splitext(os.path.basename(file_raw))[1]
        else:
            raise Exception('Invalid file path input.')

        if 'channel' in kwargs:
            # single channel images
            if len(kwargs['channel']) <= 1:
                self.is_singlechannel = True
                self.ch_synapse = 0
                self.ch_cell = -1
                self.ch_bf = -1
            else:
                self.is_singlechannel = False
                if 'N' in kwargs['channel']:
                    self.ch_cell = kwargs['channel'].upper().find('N')
                else:
                    self.ch_cell = -1
                if 'B' in kwargs['channel']:
                    # if there is bright-field
                    self.ch_bf = kwargs['channel'].upper().find('B')
                else:
                    self.ch_bf = -1
                self.ch_synapse = kwargs['channel'].upper().find('S')
        else:
            kwargs['channel'] = 'NS'
            self.is_singlechannel = False
            self.ch_cell = 0
            self.ch_synapse = 1
            self.ch_bf = -1

        if 'apply_masking' in kwargs:
            self.apply_masking = kwargs['apply_masking']
        else:
            self.apply_masking = True

        if 'minimize_storage' in kwargs:
            self.minimize_storage = kwargs['minimize_storage']
        else:
            self.minimize_storage = False

        if extension in ['.tif', '.tiff']:
            self._read_tif(file_raw, **kwargs)
        elif extension in ['.czi']:
            self._read_czi(file_raw, **kwargs)
        elif extension in ['.nd2']:
            self._read_nd2(file_raw, **kwargs)

        # shape
        self.shape = self._img_3d.shape[:3]

        # data range normalization
        self.dtype = self._img_3d.dtype
        max_val = np.max(self._img_3d)
        if self.dtype == np.uint8:
            self.drange = 2 ** 8
        elif self.dtype == np.uint16:
            if max_val < 2 ** 12:
                self.drange = 2 ** 12
            else:
                self.drange = 2 ** 16
        elif self.dtype == np.float32 or self.dtype == np.float64:
            # if the actual pixel values are a whole number
            if max_val == int(max_val):
                if max_val < 2 ** 8:
                    self._img_3d = self._img_3d.astype(np.uint8)
                    self.dtype = np.uint8
                    self.drange = 2 ** 8
                elif max_val < 2 ** 12:
                    self._img_3d = self._img_3d.astype(np.uint16)
                    self.dtype = np.uint16
                    self.drange = 2 ** 12
                else:
                    self._img_3d = self._img_3d.astype(np.uint32)
                    self.dtype = np.uint32
                    self.drange = 2 ** 16
            else:
                if max_val > 1:
                    raise Exception('Data range error: max value %f for dtype %s.' % (max_val, self.dtype))
                self.drange = 1
        else:
            raise Exception('Not implemented dtype: %s.' % self.dtype)

    @staticmethod
    def name_cell_mask(dir_name, file_name):
        return os.path.join(dir_name, '%s_rfp_mask.tif' % file_name)

    @staticmethod
    def name_mask_overlay(dir_name, file_name):
        return os.path.join(dir_name, '%s_mask_overlay.tif' % file_name)

    def _split_channels(self, src: np.ndarray, dir_key):
        """
        Split the neuronal and synaptic channel of the input image into two images.
        This method preserves data type.
        :param src:
        :param dir_key:
        :return:
        """
        out_path_0 = os.path.join(self.dirs[dir_key], '%s_0_RGB.tif' % self.file_name)
        # R for red, since neuronal marker is usually in red
        out_path_1 = os.path.join(self.dirs[dir_key], '%s_1_R.tif' % self.file_name)
        # G for synapse, since synaptic marker is usually in green
        out_path_2 = os.path.join(self.dirs[dir_key], '%s_2_G.tif' % self.file_name)
        if self.is_singlechannel:
            img_red = np.zeros_like(self._img_3d)
            img_rgb = np.zeros(shape=(*self._img_3d.shape, 3), dtype=self._img_3d.dtype)
            img_green = self._img_3d
            img_rgb[..., 1] = img_green
        elif self.minimize_storage:
            if self.ch_cell >= 0:
                img_red = src[:, :, :, self.ch_cell]
            elif self.ch_bf >= 0:
                img_red = src[:, :, :, self.ch_bf]
            else:
                img_red = np.zeros(shape=src.shape[:3], dtype=src.dtype)
            img_green = src[:, :, :, self.ch_synapse]
            img_rgb = np.zeros(shape=(*src.shape[:3], 3), dtype=src.dtype)
            if img_red is not None:
                img_rgb[:, :, :, 0] = img_red
            img_rgb[:, :, :, 1] = img_green
        else:
            if not os.path.exists(self.dirs[dir_key]):
                os.makedirs(self.dirs[dir_key])
            if not os.path.isfile(out_path_0):
                img_rgb = np.zeros(shape=(*src.shape[:3], 3), dtype=src.dtype)
                img_rgb[:, :, :, 0] = src[:, :, :, self.ch_cell]
                img_rgb[:, :, :, 1] = src[:, :, :, self.ch_synapse]
                imsave(out_path_0, img_rgb, metadata={'axes': 'ZYXC'})
            else:
                img_rgb = imread(out_path_0)

            if os.path.isfile(out_path_1):
                img_red = imread(out_path_1)
            else:
                img_red = src[:, :, :, self.ch_cell]
                imsave(out_path_1, img_red, metadata={'axes': 'ZYX'})

            if os.path.isfile(out_path_2):
                img_green = imread(out_path_2)
            else:
                img_green = src[:, :, :, self.ch_synapse]
                imsave(out_path_2, img_green, metadata={'axes': 'ZYX'})

        return img_rgb, img_green, img_red

    # region preprocess

    def preprocess(self, **kwargs):
        # [type] gfp, rfp <- raw
        # >> uint8, uint8 <- uint8
        # >> uint16, uint16 <- uint16
        self._img_3d_rgb, self._img_3d_syn, self._img_3d_cell = self._split_channels(self._img_3d, 'channel')
        if 'contrast_gfp' in kwargs:
            self._img_3d_syn = _get_contrast_enhanced(self, self._img_3d_syn, 'contrast', *kwargs['contrast_gfp'],
                                                      '%s_contrast_gfp' % self.file_name)
        if 'contrast_rfp' in kwargs:
            self._img_3d_cell = _get_contrast_enhanced(self, self._img_3d_cell, 'contrast', *kwargs['contrast_rfp'],
                                                       '%s_contrast_rfp' % self.file_name)
        if 'contrast_rfp' in kwargs and 'contrast_gfp' in kwargs:
            _out_path_contrast = os.path.join(self.dirs['contrast'], '%s_contrast.tif' % self.file_name)
            if not os.path.isfile(_out_path_contrast):
                _img_rgb = np.zeros(shape=(*self.shape, 3), dtype=self._img_3d_syn.dtype)
                _img_rgb[:, :, :, 0] = self._img_3d_cell[:, :, :]
                _img_rgb[:, :, :, 1] = self._img_3d_syn[:, :, :]
                imsave(_out_path_contrast, _img_rgb, metadata={'axes': 'ZYXC'})

        if 'denoise_rfp' in kwargs and kwargs['denoise_rfp']:
            self._img_3d_cell = _get_denoised_fastN1Means(self, self._img_3d_cell, 'denoise',
                                                          '%s_denoise_rfp' % self.file_name)
        if 'denoise_gfp' in kwargs and kwargs['denoise_gfp']:
            self._img_3d_syn = _get_denoised_fastN1Means(self, self._img_3d_syn, 'denoise',
                                                         '%s_denoise_gfp' % self.file_name)
        if 'blur_rfp' in kwargs and kwargs['blur_rfp']:
            self._img_3d_cell = _get_denoised_gaussian(self, self._img_3d_cell, 'denoise',
                                                       '%s_denoise_rfp' % self.file_name)
        if 'blur_gfp' in kwargs and kwargs['blur_gfp']:
            self._img_3d_syn = _get_denoised_gaussian(self, self._img_3d_syn, 'denoise',
                                                      '%s_denoise_gfp' % self.file_name)

        self._normalize_and_pad()

    def preprocess_unet(self):
        self._img_3d_rgb, self._img_3d_syn, self._img_3d_cell = self._split_channels(self._img_3d, 'channel')
        self._normalize_and_pad()

    def _normalize_and_pad(self):
        self._img_3d_syn_pad_by_10 = np.pad(self._img_3d_syn, ((0, 0), (10, 10), (10, 10)), 'symmetric')
        denom_syn = np.percentile(self._img_3d_syn, 99.9)
        self._img_3d_syn_pad_by_10 = self._img_3d_syn_pad_by_10 / denom_syn
        if self.ch_cell >= 0:
            self._img_3d_cell_pad_by_10 = np.pad(self._img_3d_cell, ((0, 0), (10, 10), (10, 10)), 'symmetric')
            denom_cell = np.percentile(self._img_3d_cell, 99.9)
            self._img_3d_cell_pad_by_10 = self._img_3d_cell_pad_by_10 / denom_cell

    # endregion

    # region masking
    def masking(self, mask_prune_cc=True):
        self._img_3d_cell_mask, self._img_3d_mask_overlay = self._masking(
            src_syn=self._img_3d_syn,
            src_cell=self._img_3d_cell,
            dir_key='mask',
            prune_cc=mask_prune_cc)

    def _load_mask_file(self, mask_dir_key):
        img_mask = None
        mask_file_path = self.name_cell_mask(self.dirs[mask_dir_key], self.file_name)
        if os.path.isfile(mask_file_path):
            img_mask = imread(mask_file_path) > 0
            # if 2d mask is provided, stack it to match the dimension of the raw image
            if len(img_mask.shape) == 2:
                img_mask = np.repeat(img_mask[np.newaxis, :, :], self.shape[0], axis=0)
        elif self.is_singlechannel:
            img_mask = np.ones(self.shape[:3], dtype=bool)

        return img_mask

    def _masking(self, src_syn, src_cell, dir_key: str, prune_cc: bool):
        if not self.apply_masking:
            # if the user choose not to use mask, then neurite mask is just all area
            img_mask = np.ones(self.shape[:3], dtype=bool)
        else:
            out_path_1 = self.name_cell_mask(self.dirs[dir_key], self.file_name)
            out_path_2 = self.name_mask_overlay(self.dirs[dir_key], self.file_name)
            img_mask = self._load_mask_file(dir_key)
            if img_mask is None:
                start_time = time.time()

                dst_xy_dilate = np.zeros(self.shape[:3], dtype=np.float32)
                dst_z_dilate = np.zeros(self.shape[:3], dtype=bool)
                src_cell_c = img_as_float32(src_cell)

                for z in range(self.shape[0]):
                    img_z_f = src_cell_c[z, :, :]
                    # dst_bl = restoration.denoise_bilateral(img_z_f, win_size=3, sigma_color=0.1, sigma_spatial=20)
                    img_z_1 = filters.prewitt(img_z_f)
                    img_z_2 = closing(img_z_1, disk(3))
                    dst_xy_dilate[z, :, :] = img_z_2
                thr_li = filters.threshold_li(dst_xy_dilate)
                for z in range(self.shape[0]):
                    img_thr = dst_xy_dilate[z] > thr_li
                    img_dil1 = binary_dilation(img_thr, disk(1))
                    img_dil2 = binary_dilation(img_dil1, disk(1))
                    dst_xy_dilate[z, :, :] = img_dil2

                # dilate once in z direction
                for z in range(self.shape[0]):
                    dst_z_dilate[z, :, :] = np.max(dst_xy_dilate[max(0, z - 1):min(self.shape[0] - 1, z + 1)], axis=0)

                # remove small blobs
                # get connected components
                img_mask = img_as_ubyte(dst_z_dilate)

                if prune_cc:
                    s = [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
                    ]
                    res, N = scipy.ndimage.measurements.label(img_mask, structure=s)
                    total_area = int(np.sum(img_mask > 0))
                    ccid_stats = []  # (ccid, area, mean green level)
                    for ccid in range(1, N + 1):
                        ccid_stats.append(
                            (ccid, int(np.sum(res == ccid)), float(np.average(src_syn[res == ccid]))))
                    ccid_stats.sort(key=lambda x: x[1], reverse=True)

                    # delete cc of which area is less than 30% of total area of mask
                    ccid_stats_major = list(filter(lambda x: x[1] > total_area * 0.3, ccid_stats))
                    if len(ccid_stats_major) == 0:
                        ccid_stats_major.append(ccid_stats[0])

                    # among large components, select the one with mimimum g-level
                    ccid_stats_major.sort(key=lambda x: x[2])
                    ccid_selected = ccid_stats_major[0][0]

                    img_mask = np.zeros(shape=self.shape[:3], dtype=bool)
                    img_mask[res == ccid_selected] = True
                else:
                    img_mask = (img_mask > 0)

                imsave(out_path_1, img_as_ubyte(img_mask), metadata={'axes': 'ZYX'})
                elapsed_time = time.time() - start_time
                str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print('%s: making a neurite mask. %d stacks. %s elapsed' % (
                    self.file_name, self.shape[0], str_elapsed_time))

        if self.minimize_storage:
            img_masked = None
        else:
            if os.path.isfile(out_path_2):
                img_masked = imread(out_path_2)
            else:
                # if True:
                start_time = time.time()
                img_masked = np.zeros(shape=(*self.shape[:3], 3), dtype=self.dtype)
                img_masked[:, :, :, 0] = src_cell.copy()
                img_masked[:, :, :, 1] = src_syn.copy()
                img_masked[img_mask, 2] = self.drange / 2
                imsave(out_path_2, img_masked, metadata={'axes': 'ZYX'})

                elapsed_time = time.time() - start_time
                str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print('%s: applying the mask to gfp. %s elapsed' % (self.file_name, str_elapsed_time))

        return img_mask, img_masked

    def masking_unet(self, model):
        if not self.apply_masking:
            # if the user choose not to use mask, then neurite mask is just all area
            self._img_3d_cell_mask = np.ones(self.shape[:3], dtype=bool)
        else:
            out_path_1 = self.name_cell_mask(self.dirs['mask'], self.file_name)
            out_path_2 = self.name_mask_overlay(self.dirs['mask'], self.file_name)
            img_mask = self._load_mask_file('mask')
            if img_mask is not None:
                self._img_3d_cell_mask = img_mask
                if not self.minimize_storage:
                    if os.path.isfile(out_path_2):
                        self._img_3d_mask_overlay = imread(out_path_2)
                    else:
                        self._img_3d_mask_overlay = np.zeros((*self.shape[:3], 3),
                                                             dtype=self._img_3d_cell.dtype)
                        self._img_3d_mask_overlay[..., 0] = self._img_3d_cell
                        self._img_3d_mask_overlay[..., 1][
                            self._img_3d_cell_mask > 0] = 2 ** 12  # Define a green value for visualization in uint16.
            else:
                if isinstance(model, UNET25D_Atrous):
                    cell_output = visualization.Prediction_Mask(red_ch=self._img_3d_cell, model=model) > 0
                elif isinstance(model, UNet_Multi_Scale):
                    cell_output, syn_output = visualization.Prediction_MultiChannel(
                        red_ch=self._img_3d_cell, green_ch=self._img_3d_syn, model=model)

                self._img_3d_cell_mask = np.zeros(self.shape[:3], dtype=np.ubyte)
                self._img_3d_cell_mask[cell_output > 0] = np.iinfo(np.ubyte).max
                self._img_3d_mask_overlay = np.zeros((*self.shape[:3], 3), dtype=self._img_3d_cell.dtype)
                self._img_3d_mask_overlay[..., 0] = self._img_3d_cell
                self._img_3d_mask_overlay[..., 1] = self._img_3d_syn
                self._img_3d_mask_overlay[cell_output > 0, 2] = (np.iinfo(self.dtype).max + 1) / (2 ** 4)

                imsave(out_path_1, img_as_ubyte(self._img_3d_cell_mask))
                if self.minimize_storage:
                    pass
                else:
                    imsave(out_path_2, self._img_3d_mask_overlay)

    def masking_unet_multichannel(self, model):
        if not self.apply_masking:
            # if the user choose not to use mask, then neurite mask is just all area
            self._img_3d_cell_mask = np.ones(self.shape[:3], dtype=bool)
        else:
            out_path_1 = self.name_cell_mask(self.dirs['mask'], self.file_name)
            out_path_2 = self.name_mask_overlay(self.dirs['mask'], self.file_name)
            img_mask = self._load_mask_file('mask')
            if img_mask is not None:
                self._img_3d_cell_mask = img_mask
                if not self.minimize_storage:
                    if os.path.isfile(out_path_2):
                        self._img_3d_mask_overlay = imread(out_path_2)
                    else:
                        self._img_3d_mask_overlay = np.zeros((*self.shape[:3], 3),
                                                             dtype=self._img_3d_cell.dtype)
                        self._img_3d_mask_overlay[..., 0] = self._img_3d_cell
                        self._img_3d_mask_overlay[..., 1][
                            self._img_3d_cell_mask > 0] = 2 ** 12  # Define a green value for visualization in uint16.
            else:
                cell_output, syn_output = visualization.Prediction_MultiChannel(
                    red_ch=self._img_3d_cell, green_ch=self._img_3d_syn, model=model)
                self._img_3d_cell_mask = np.zeros(self.shape[:3], dtype=np.ubyte)
                self._img_3d_cell_mask[cell_output > 0] = np.iinfo(np.ubyte).max
                self._img_3d_mask_overlay = np.zeros((*self.shape[:3], 3), dtype=self._img_3d_cell.dtype)
                self._img_3d_mask_overlay[..., 0] = self._img_3d_cell
                self._img_3d_mask_overlay[..., 1] = self._img_3d_syn
                self._img_3d_mask_overlay[cell_output > 0, 2] = (np.iinfo(self._img_3d_mask_overlay.dtype).max + 1) / (
                        2 ** 4)

                imsave(out_path_1, img_as_ubyte(self._img_3d_cell_mask))
                if self.minimize_storage:
                    pass
                else:
                    imsave(out_path_2, self._img_3d_mask_overlay)

    # endregion

    # region feature extraction
    def _task_extract_local_features_for_patch_normalize(self, z, y, x, size_y, size_x):
        # 0-padding source images
        src_padded = np.zeros(shape=(size_y + 20, size_x + 20, 2), dtype=np.float32)
        # normalize each color separately
        src_padded[:, :, 0] = self._img_3d_syn_pad_by_10[z][y:y + 20 + size_y, x:x + 20 + size_x]

        # if it has a channel for tagging the neuron
        if self.ch_cell >= 0:
            result = np.zeros(shape=(size_y, size_x, DotFeature.MAX.value), dtype=np.float32)
            src_padded[:, :, 1] = self._img_3d_cell_pad_by_10[z][y:y + 20 + size_y, x:x + 20 + size_x]
        else:
            # if not masking - only use synapse channel features
            result = np.zeros(shape=(size_y, size_x, DotFeature.CELL.value), dtype=np.float32)

        # normalize
        src_padded[np.isnan(src_padded)] = 0

        for channel, start_index in enumerate([DotFeature.SYNAPSE.value, DotFeature.CELL.value]):
            # +0 = original
            src = src_padded[:, :, channel]
            result[:, :, start_index] = src[10:-10, 10:-10]

            # +1 = gradient
            im_grad = filters.sobel(src)
            result[:, :, start_index + DotFeature.GRADIENT.value] = im_grad[10:-10, 10:-10]

            # +2, 3 = DOGs
            im_dog1 = filters.gaussian(src, sigma=0.1) - filters.gaussian(src, sigma=1.5)
            im_dog2 = filters.gaussian(src, sigma=0.1) - filters.gaussian(src, sigma=5)
            result[:, :, start_index + DotFeature.DOG_1.value] = im_dog1[10:-10, 10:-10]
            result[:, :, start_index + DotFeature.DOG_2.value] = im_dog2[10:-10, 10:-10]

            # +4, 5, 6, 7 = avgs and stdevs
            im_mean5 = scipy.ndimage.uniform_filter(src, size=5)
            result[:, :, start_index + DotFeature.MEAN_5.value] = im_mean5[10:-10, 10:-10]
            im_std5 = window_stdev(src, 5)
            result[:, :, start_index + DotFeature.STD_5.value] = im_std5[10:-10, 10:-10]
            im_mean11 = scipy.ndimage.uniform_filter(src, size=11)
            result[:, :, start_index + DotFeature.MEAN_11.value] = im_mean11[10:-10, 10:-10]
            im_std11 = window_stdev(src, 11)
            result[:, :, start_index + DotFeature.STD_11.value] = im_std11[10:-10, 10:-10]

            # +8 = LOGs
            im_log = ndimage.gaussian_laplace(src, sigma=0.1)
            result[:, :, start_index + DotFeature.LOG.value] = im_log[10:-10, 10:-10]

            # +9 = erode
            im_erode = erosion(src, diamond(4, dtype=np.float32))
            result[:, :, start_index + DotFeature.ERODE.value] = im_erode[10:-10, 10:-10]

            # if not masking - only use synapse channel features
            if self.ch_cell < 0:
                break

        return result

    def _extract_local_features_for_patches_parallel(self, patch_list, size_patch_list, num_process=1):
        """
        Extract local regional features for every pixel  and save it in protected variable _features
        """
        tic = time.perf_counter()

        out = Parallel(n_jobs=num_process)(
            delayed(self._task_extract_local_features_for_patch_normalize)(*patch, *size) for patch, size in
            zip(patch_list, size_patch_list))

        toc = time.perf_counter()
        self.logger.debug(
            "_extract_local_features_for_patches_parallel(num_process=%d) in %0.4f seconds" % (num_process, toc - tic))
        return out

    def _get_local_and_prob_features_parallel(self, func_pred: Callable, region_get_prob, num_process: int = 1):
        size_patch = 50
        patch_and_size_list = Parallel(n_jobs=num_process)(
            delayed(self._task_get_patches_contain_label)(region_get_prob, z, size_patch) for z in range(self.shape[0]))
        patch_list = []
        size_list = []
        for patch, size in patch_and_size_list:
            if len(patch) > 0:
                patch_list.extend(patch)
                size_list.extend(size)
        patch_features_list = self._extract_local_features_for_patches_parallel(patch_list, size_list, num_process)

        f_local_flatten = []
        for patch, size, features in zip(patch_list, size_list, patch_features_list):
            z, y, x = patch
            mask = region_get_prob[z, y:y + size[0], x:x + size[1]] > 0
            f_local_flatten.append(features[mask])

        if num_process > 1:
            pred_clf_1 = Parallel(n_jobs=num_process)(delayed(func_pred)(fs) for fs in f_local_flatten)
        else:
            pred_clf_1 = []
            for fs in f_local_flatten:
                pred_clf_1.append(func_pred(fs))

        self.logger.info('Done extracting probability')
        return patch_list, size_list, patch_features_list, pred_clf_1

    def _extract_local_and_prob_features_for_patches_parallel(self, func_pred, region_get_prob, region_set_feat,
                                                              num_process=1):
        """

        :return:
        """
        tic = time.perf_counter()
        out = self._get_local_and_prob_features_parallel(func_pred, region_get_prob, num_process)
        prob_array = np.zeros(shape=self.shape[:3], dtype=float)
        for patch, size, features, prob in zip(*out):
            z, y, x = patch
            mask = region_get_prob[z, y:y + size[0], x:x + size[1]] > 0
            prob_array[z, y:y + size[0], x:x + size[1]][mask] = prob

        prob_array_w_padding = np.pad(prob_array, ((1, 1), (1, 1), (1, 1)), 'edge')
        prob_features = np.zeros(shape=(*self.shape[:3], PROB_FEATURES), dtype=np.float32)
        # z=z-1   z=z    z=z+1
        # . . .  . 3 .  . . .
        # . 5 .  1 0 2  . 6 .
        # . . .  . 4 .  . . .
        # result[:, :, 0] = probarray[z, y1:y2, x1:x2]
        prob_features[region_set_feat, 0] = prob_array_w_padding[1:-1, 1:-1, 1:-1][region_set_feat]

        # left
        prob_features[region_set_feat, 1] = prob_array_w_padding[1:-1, 1:-1, :-2][region_set_feat]
        # right
        prob_features[region_set_feat, 2] = prob_array_w_padding[1:-1, 1:-1, 2:][region_set_feat]
        # top
        prob_features[region_set_feat, 3] = prob_array_w_padding[1:-1, :-2, 1:-1][region_set_feat]
        # bottom
        prob_features[region_set_feat, 4] = prob_array_w_padding[1:-1, 2:, 1:-1][region_set_feat]
        # below
        prob_features[region_set_feat, 5] = prob_array_w_padding[:-2, 1:-1, 1:-1][region_set_feat]
        # top
        prob_features[region_set_feat, 6] = prob_array_w_padding[2:, 1:-1, 1:-1][region_set_feat]

        prob_features.sort(axis=-1)

        patch_list = []
        size_list = []
        local_f_list = []
        prob_f_list = []
        for patch, size, features, prob in zip(*out):
            z, y, x = patch
            mask = region_set_feat[z, y:y + size[0], x:x + size[1]]
            if np.sum(mask) == 0:
                continue
            else:
                patch_list.append(patch)
                size_list.append(size)
                local_f_list.append(features)
                prob_f_list.append(prob_features[z, y:y + size[0], x:x + size[1]])

        toc = time.perf_counter()
        self.logger.debug("_extract_local_and_prob_features_for_patches_parallel(num_process=%d) in %0.4f seconds" % (
            num_process, toc - tic))
        return patch_list, size_list, local_f_list, prob_f_list

    def extract_prob_features_for_patches_parallel(self, func_pred, patch_list, size_list, num_process=1):
        """

        :return:
        """
        tic = time.perf_counter()

        # temporarily pad cell and synapse image by 1 on each edge
        if self.ch_cell >= 0:
            self._img_3d_cell_pad_by_10 = np.pad(self._img_3d_cell_pad_by_10, ((1, 1), (1, 1), (1, 1)), 'symmetric')

        self._img_3d_syn_pad_by_10 = np.pad(self._img_3d_syn_pad_by_10, ((1, 1), (1, 1), (1, 1)), 'symmetric')
        extended_patch_list = []
        extended_size_list = []
        for patch, size in zip(patch_list, size_list):
            z, y, x = patch
            extended_patch_list.append((z, y, x))
            extended_patch_list.append((z + 1, y, x))
            extended_patch_list.append((z + 2, y, x))
            extended_size_list.append((size[0] + 2, size[1] + 2))
            extended_size_list.append((size[0] + 2, size[1] + 2))
            extended_size_list.append((size[0] + 2, size[1] + 2))
        patch_features_list = self._extract_local_features_for_patches_parallel(extended_patch_list,
                                                                                extended_size_list,
                                                                                num_process)
        # unpad
        if self.ch_cell >= 0:
            self._img_3d_cell_pad_by_10 = self._img_3d_cell_pad_by_10[1:-1, 1:-1, 1:-1]

        self._img_3d_syn_pad_by_10 = self._img_3d_syn_pad_by_10[1:-1, 1:-1, 1:-1]

        # predict
        f_local_flatten = []
        for patch_features in patch_features_list:
            f_local_flatten.append(patch_features.reshape(-1, patch_features.shape[-1]))

        if num_process > 1:
            pred_clf_1 = Parallel(n_jobs=num_process)(delayed(func_pred)(fs) for fs in f_local_flatten)
        else:
            pred_clf_1 = []
            for fs in f_local_flatten:
                pred_clf_1.append(func_pred(fs))

        patch_prob_list = []
        for patch_pred, size in zip(pred_clf_1, extended_size_list):
            patch_prob_list.append(patch_pred.reshape(*size))

        # set prob features
        features = np.zeros(
            shape=(self.shape[0], self.shape[1], self.shape[2], PROB_FEATURES),
            dtype=np.float32)
        for i, patch in enumerate(patch_list):
            z, y, x = patch
            size_y = size_list[i][0]
            size_x = size_list[i][1]

            # z=z-1   z=z    z=z+1
            # . . .  . 3 .  . . .
            # . 5 .  1 0 2  . 6 .
            # . . .  . 4 .  . . .
            # result[:, :, 0] = probarray[z, y1:y2, x1:x2]
            features[z, y:y + size_y, x:x + size_x, 0] = patch_prob_list[i * 3 + 1][1:-1, 1:-1]
            # left
            features[z, y:y + size_y, x:x + size_x, 1] = patch_prob_list[i * 3 + 1][1:-1, :-2]
            # right
            features[z, y:y + size_y, x:x + size_x, 2] = patch_prob_list[i * 3 + 1][1:-1, 2:]
            # top
            features[z, y:y + size_y, x:x + size_x, 3] = patch_prob_list[i * 3 + 1][:-2, 1:-1]
            # bottom
            features[z, y:y + size_y, x:x + size_x, 4] = patch_prob_list[i * 3 + 1][2:, 1:-1]
            # below
            features[z, y:y + size_y, x:x + size_x, 5] = patch_prob_list[i * 3][1:-1, 1:-1]
            # above
            features[z, y:y + size_y, x:x + size_x, 6] = patch_prob_list[i * 3 + 2][1:-1, 1:-1]

        features.sort(axis=-1)

        toc = time.perf_counter()
        self.logger.debug(
            "extract_prob_features_for_patches_parallel(num_process=%d) in %0.4f seconds" % (num_process, toc - tic))
        return features

    def _calc_average_area_label(self) -> float:
        if self._avg_label_area is None:
            tic = time.perf_counter()
            areas = []
            for z in range(self.shape[0]):
                if np.sum(self._img_label[z]) == 0:
                    continue

                # voxels around positive points
                label = measure.label(self._img_label[z], connectivity=2)
                props = measure.regionprops(label)
                areas.extend([prop.area for prop in props])
            avg_area = float(np.mean(areas))
            toc = time.perf_counter()
            self.logger.debug("_calc_average_area_label() in %0.4f seconds" % (toc - tic))
            self._avg_label_area = avg_area
            return avg_area
        else:
            return self._avg_label_area

    def _task_get_patches_contain_label(self, label, z, size_patch):
        patch_list = []
        size_list = []
        for y in range(0, self.shape[1], size_patch):
            for x in range(0, self.shape[2], size_patch):
                if np.any(label[z, y:y + size_patch, x:x + size_patch]):
                    patch_list.append((z, y, x))
                    size_y = min(size_patch, self.shape[1] - y)
                    size_x = min(size_patch, self.shape[2] - x)
                    size_list.append((size_y, size_x))

        return patch_list, size_list

    def get_training_local_features(self, num_process=1, n_max_patches=-1):
        """

        Args:
            num_process:
            n_max_patches: If -1, use all the patches. If not, subsample only this number of patches.

        Returns:

        """
        tic_lv1 = time.perf_counter()
        # get the average size of connected label
        avg_area = self._calc_average_area_label()
        # set patch size as square root of average size of synapse
        size_patch = int(max(10, np.around(avg_area ** 0.5) * 2))
        self.logger.debug("Average synapse area=%f. Patch Size=%d." % (avg_area, size_patch))

        tic_lv2 = time.perf_counter()
        # look around patches and if any positive label is contained in a patch, then include it to training set
        if self.ch_cell >= 0:
            positives = np.empty(shape=(0, DotFeature.MAX.value), dtype=np.float32)
            negatives = np.empty(shape=(0, DotFeature.MAX.value), dtype=np.float32)
        else:
            # if not masking - only use synapse channel features
            positives = np.empty(shape=(0, DotFeature.CELL.value), dtype=np.float32)
            negatives = np.empty(shape=(0, DotFeature.CELL.value), dtype=np.float32)

        patch_and_size_list = Parallel(n_jobs=num_process)(
            delayed(self._task_get_patches_contain_label)(self._img_label, z, size_patch) for z in range(self.shape[0]))
        patch_list = []
        size_list = []
        for patch, size in patch_and_size_list:
            if len(patch) > 0:
                patch_list.extend(patch)
                size_list.extend(size)
        patch_features_list = self._extract_local_features_for_patches_parallel(patch_list, size_list, num_process)

        # randomly subsmaple patches if n_max_patches is given
        if n_max_patches > 0:
            if n_max_patches > len(patch_list):
                self.logger.warning("Number of patches (%d) < n_max_patches (%d). Proceed without subsampling." % (
                    len(patch_list), n_max_patches
                ))
            else:
                i_chosen = np.random.choice(range(len(patch_list)), n_max_patches, replace=False)
                patch_list = [patch_list[i] for i in i_chosen]
                size_list = [size_list[i] for i in i_chosen]
                patch_features_list = [patch_features_list[i] for i in i_chosen]

                # sort pixels into positive and negative group
        for patch, features in zip(patch_list, patch_features_list):
            z, y, x = patch
            mask = self._img_label[z, y:y + size_patch, x:x + size_patch] > 0
            if mask.shape != features.shape[:-1]:
                features_reshape = features[:mask.shape[0], :mask.shape[1]]
                positives = np.append(positives, features_reshape[mask], axis=0)
                negatives = np.append(negatives, features_reshape[~mask], axis=0)
            else:
                positives = np.append(positives, features[mask], axis=0)
                negatives = np.append(negatives, features[~mask], axis=0)

        cnt_patches = len(patch_list)
        toc_lv2 = time.perf_counter()
        self.logger.debug(
            "extracting training points (%d, %d) from %d patches in %0.4f seconds" % (
                positives.shape[0], negatives.shape[0], cnt_patches, toc_lv2 - tic_lv2))

        toc_lv1 = time.perf_counter()
        self.logger.debug("get_training_pixel_features() in %0.4f seconds" % (toc_lv1 - tic_lv1))

        self._init_patch_list = patch_list
        self._init_size_list = size_list
        self._init_local_features_p = positives
        self._init_local_features_n = negatives
        return positives, negatives

    def _task_get_patches_nearTP(self, size_patch_large, size_patch_small):
        # first, reconstruct the list of large patch that contain the intial patch list
        large_patches = set()
        for patch in self._init_patch_list:
            large_patches.add((patch[0],
                               (patch[1] // size_patch_large) * size_patch_large,
                               (patch[2] // size_patch_large) * size_patch_large,))

        patch_list = []
        size_list = []
        # collecting all the possible patches
        for z in range(self.shape[0]):
            for y in range(0, self.shape[1], size_patch_large):
                for x in range(0, self.shape[2], size_patch_large):
                    if (z, y, x) not in large_patches:
                        continue
                    if not np.any(self._img_label[z, y:y + size_patch_large, x:x + size_patch_large]):
                        continue
                    for yy in range(y, y + size_patch_large, size_patch_small):
                        if yy >= self.shape[1]:
                            continue
                        for xx in range(x, x + size_patch_large, size_patch_small):
                            if xx >= self.shape[2]:
                                continue
                            if np.any(self._img_label[z, yy:yy + size_patch_small, xx:xx + size_patch_small]):
                                continue
                            else:
                                size_y = min(size_patch_small, self.shape[1] - yy)
                                size_x = min(size_patch_small, self.shape[2] - xx)
                                patch_list.append((z, yy, xx))
                                size_list.append((size_y, size_x))

        return patch_list, size_list

    def _task_get_training_pixel_features_2nd_nearTP(self, patch_list, size_list):
        if self.ch_cell >= 0:
            f_patches = np.empty(shape=(0, DotFeature.MAX.value), dtype=np.float32)
        else:
            # if not masking - only use synapse channel features
            f_patches = np.empty(shape=(0, DotFeature.CELL.value), dtype=np.float32)

        # subsample n_max_patches (all if -1)
        for patch, size in zip(patch_list, size_list):
            z, yy, xx = patch
            size_y, size_x = size
            f_patch = self._task_extract_local_features_for_patch_normalize(z, yy, xx, size_y, size_x)
            f_patch = f_patch.reshape((size_y * size_x, f_patch.shape[2]))
            if f_patch.shape[0] == 0:
                continue
            f_patches = np.append(f_patches, f_patch, axis=0)

        return f_patches

    def get_training_pixel_features_2nd_nearTP(self, classifer, num_process=1, n_max_patches=-1):
        '''

        Args:
            classifer:
            num_process:
            n_max_patches: If -1, use all the patches. If not, subsample only this number of patches.

        Returns:

        '''
        tic = time.perf_counter()
        # get the average size of connected label
        avg_area = self._calc_average_area_label()

        # 10 times the patch size
        n = 10
        size_patch_small = int(max(10, np.around(avg_area ** 0.5) * 2))
        size_patch_large = size_patch_small * n
        self.logger.debug(
            "Average synapse area=%f. Patch Size=(%d, %d)." % (avg_area, size_patch_small, size_patch_large))

        self.preprocess()
        patch_list, size_list = self._task_get_patches_nearTP(size_patch_large, size_patch_small)
        self.logger.debug("Sampled %d patches having no synapse pixel, but nearby synapse patches" % len(patch_list))

        if num_process > 1:
            patch_list_split = np.array_split(patch_list, num_process)
            size_list_split = np.array_split(size_list, num_process)
            f_patches = Parallel(n_jobs=num_process)(
                delayed(self._task_get_training_pixel_features_2nd_nearTP)(p_list, s_list) for p_list, s_list in zip(
                    patch_list_split, size_list_split))
            f_patches = np.concatenate(f_patches, axis=0)
        else:
            f_patches = self._task_get_training_pixel_features_2nd_nearTP(patch_list, size_list)

        # predict with trained 1st layer
        if num_process > 1:
            f_patches_split = np.array_split(f_patches, num_process)
            pred_patches = Parallel(n_jobs=num_process)(delayed(classifer.predict)(f_p) for f_p in f_patches_split)
            pred_patches = np.concatenate(pred_patches, axis=0)
        else:
            pred_patches = classifer.predict(f_patches)

        negatives = f_patches[pred_patches > 0]

        self.logger.debug("add %d negative training points" % negatives.shape[0])
        toc = time.perf_counter()
        self.logger.debug("get_training_pixel_features_2nd() in %0.4f seconds" % (toc - tic))

        if self.ch_cell >= 0:
            positives = np.empty(shape=(0, DotFeature.MAX.value), dtype=np.float32)
        else:
            # if not masking - only use synapse channel features
            positives = np.empty(shape=(0, DotFeature.CELL.value), dtype=np.float32)

        return positives, negatives

    def get_training_prob_features(self, func_pred: Callable, num_process: int = 1, n_max_patches=-1):
        """

        Args:
            func_pred:
            num_process:
            n_max_patches: If -1, use all the patches. If not, subsample only this number of patches.

        Returns:

        """
        assert hasattr(self, '_img_label'), "Label not loaded"

        tic_lv1 = time.perf_counter()
        # get the average size of connected label
        avg_area = self._calc_average_area_label()
        # set patch size as square root of average size of synapse
        size_patch = int(max(10, np.around(avg_area ** 0.5) * 2))
        self.logger.debug("Average synapse area=%f. Patch Size=%d." % (avg_area, size_patch))

        tic_lv2 = time.perf_counter()
        # look around patches and if any positive label is contained in a patch, then include it to training set

        region_set_feat = np.zeros(shape=self.shape[:3], dtype=bool)
        cnt_patches = len(self._init_patch_list)
        for i in range(cnt_patches):
            z, y, x = self._init_patch_list[i]
            size_y, size_x = self._init_size_list[i]
            region_set_feat[z, y:y + size_y, x:x + size_x] = True

        # extract probability features
        prob_features = self.extract_prob_features_for_patches_parallel(func_pred, self._init_patch_list,
                                                                        self._init_size_list, num_process)
        prob_features_patch = prob_features[region_set_feat]
        mask = self._img_label[region_set_feat] > 0
        positives = prob_features_patch[mask]
        negatives = prob_features_patch[~mask]

        toc_lv2 = time.perf_counter()
        self.logger.debug(
            "extracting %d training points from %d patches in %0.4f seconds" % (
                int(np.sum(region_set_feat)), cnt_patches, toc_lv2 - tic_lv2))

        toc_lv1 = time.perf_counter()
        self.logger.debug("get_training_prob_features() in %0.4f seconds" % (toc_lv1 - tic_lv1))

        return np.concatenate((self._init_local_features_p, positives), axis=1), \
               np.concatenate((self._init_local_features_n, negatives), axis=1)

    # endregion

    def read_label_image_tif(self, file_label):
        """
        Last updated: 20.06.21
        Load the labeled image.
            Synapse label must be colored in blue (r,g,n+) in case of the image is multichannel,
            or the value is label if it's in single channel
        :param file_label:
        :param axis:
        :return:
        """
        img_label = imread(file_label)
        if len(img_label.shape) == 4:
            self._img_label = (img_label[:, :, :, 2] > 0)
        elif len(img_label.shape) == 3:
            self._img_label = img_label > 0
        else:
            raise Exception("Unexpected format of label image")

    def read_label_image_h5(self, file_label):
        with h5py.File(file_label, 'r') as f:
            dset = f['exported_data']
            self._img_label = (dset[...] == 1).reshape(dset.shape[:3])

    def _predict_layer1(self, func_pred, mask, num_process):
        size_patch = 10
        patch_and_size_list = Parallel(n_jobs=num_process)(
            delayed(self._task_get_patches_contain_label)(mask, z, size_patch) for z in range(self.shape[0]))
        patch_list = []
        size_list = []
        for patch, size in patch_and_size_list:
            if len(patch) > 0:
                patch_list.extend(patch)
                size_list.extend(size)
        patch_features_list = self._extract_local_features_for_patches_parallel(patch_list, size_list, num_process)

        f_local_flatten = []
        for patch, size, features in zip(patch_list, size_list, patch_features_list):
            z, y, x = patch
            mask_patch = mask[z, y:y + size[0], x:x + size[1]] > 0
            f_local_flatten.append(features[mask_patch])

        if num_process > 1:
            lb_flatten = Parallel(n_jobs=num_process)(delayed(func_pred)(fs) for fs in f_local_flatten)
        else:
            lb_flatten = []
            for fs in f_local_flatten:
                lb_flatten.append(func_pred(fs))

        return patch_list, size_list, lb_flatten

    def _predict_layer2(self, func_pred_1, func_pred_2, mask, num_process):
        # padding 1 pixels in each direction
        temp = np.pad(mask, ((1, 1), (1, 1), (1, 1)))

        # regions to getting prediction of the first layer classifier is level 1 connected pixels of the mask
        region_get_pred = np.zeros_like(temp)
        for z in range(1, temp.shape[0] - 1):
            temp[z] = binary_dilation(temp[z], disk(1))
        for z in range(1, region_get_pred.shape[0] - 1):
            region_get_pred[z - 1][temp[z] > 0] = True
            region_get_pred[z][temp[z] > 0] = True
            region_get_pred[z + 1][temp[z] > 0] = True

        # trim
        region_get_pred = region_get_pred[1:-1, 1:-1, 1:-1]
        out = self._extract_local_and_prob_features_for_patches_parallel(
            func_pred_1, region_get_pred, mask, num_process)

        patch_list = []
        size_list = []
        f_local_prob_flatten = []
        for patch, size, features_local, features_prob in zip(*out):
            z, y, x = patch
            mask_patch = mask[z, y:y + size[0], x:x + size[1]] > 0
            patch_list.append(patch)
            size_list.append(size)
            f_local_prob_flatten.append(np.concatenate((features_local[mask_patch], features_prob[mask_patch]), axis=1))

        if num_process > 1:
            lb_flatten = Parallel(n_jobs=num_process)(delayed(func_pred_2)(fs) for fs in f_local_prob_flatten)
        else:
            lb_flatten = []
            for fs in f_local_prob_flatten:
                lb_flatten.append(func_pred_2(fs))

        return patch_list, size_list, lb_flatten

    # region prediction
    def predict(self, func_pred_1, func_pred_2, small_synapse_cutoff=0, num_process: int = 1):
        self.logger.info('Start predict')
        tic = time.perf_counter()
        # check if the second layer provided
        if func_pred_2 is None:
            patch_list, size_list, lb_flatten_list = self._predict_layer1(func_pred_1, self._img_3d_cell_mask,
                                                                          num_process)
        else:
            patch_list, size_list, lb_flatten_list = self._predict_layer2(func_pred_1, func_pred_2,
                                                                          self._img_3d_cell_mask, num_process)

        img_label_pred = np.zeros(shape=self.shape[:3], dtype=bool)

        for patch, size, label_flatten in zip(patch_list, size_list, lb_flatten_list):
            z, y, x = patch
            mask_patch = self._img_3d_cell_mask[z, y:y + size[0], x:x + size[1]] > 0
            img_label_pred[z, y:y + size[0], x:x + size[1]][mask_patch] = label_flatten
        self.logger.info('Done extracting label')

        # remove small synapses
        if small_synapse_cutoff > 0:
            self.logger.info('Eliminate small synapses')
            label = measure.label(img_label_pred, connectivity=2)
            props = measure.regionprops(label)
            for prop in props:
                if prop.area <= small_synapse_cutoff:
                    img_label_pred[label == prop.label] = False
            self.logger.info('Done small synapse elimination')

        toc = time.perf_counter()
        self.logger.debug("predict(small_synapse_cutoff=%d, num_process=%d) in %0.4f seconds" % (
            small_synapse_cutoff, num_process, toc - tic))
        return img_label_pred

    def write_predicted_label_image(self, img_label, file_label, file_overlay):
        if len(img_label.shape) != 3:
            img_label = img_label.reshape(self.shape)
        self._img_label_pred = img_label > 0
        imsave(file_label, img_as_ubyte(self._img_label_pred), metadata={'axes': 'ZYX'})

        img_overlay = np.zeros(shape=(*self.shape, 3), dtype=self.dtype)
        if self.ch_cell >= 0:
            img_overlay[:, :, :, 0] = self._img_3d_cell.copy()
        elif self.ch_bf >= 0:
            img_overlay[:, :, :, 0] = self._img_3d[:, :, :, self.ch_bf].copy()
        img_overlay[:, :, :, 1] = self._img_3d_syn.copy()
        img_overlay[self._img_label_pred == True, 2] = np.iinfo(self.dtype).max
        imsave(file_overlay, img_overlay, metadata={'axes': 'ZYXC'})

    def read_predicted_label_image(self, file_predicted):
        img_predicted = imread(file_predicted)
        self._img_label_pred = img_predicted > 0

    def segment_instances(self, file_instance_label, file_overlay, file_color_label, min_distance):
        assert hasattr(self, '_img_3d_syn'), "Synapse channel image not loaded"
        assert hasattr(self, '_img_label_pred'), "Predicted label not loaded"

        # segment synapse instances from the binary segmentation result with watershed algorithm
        labels_for_peak = self._img_label_pred.copy()
        coords_final = np.empty(dtype=int, shape=(0, 3))
        # iterate until find peaks for all the labeled synapses
        dist = self._img_3d_syn.copy().astype(int)
        while np.sum(labels_for_peak) > 0:
            # there can be some labels not having peak coord due to the min distance
            coords = peak_local_max(dist.copy(), min_distance=min_distance, labels=labels_for_peak,
                                    exclude_border=False)
            coords_final = np.vstack((coords_final, coords))
            mask = np.zeros(labels_for_peak.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers = measure.label(mask)
            labels = watershed(-dist.copy(), markers, mask=labels_for_peak, connectivity=2)
            labels_for_peak[labels > 0] = 0

        # final watersheding
        mask = np.zeros(self._img_label_pred.shape, dtype=bool)
        mask[tuple(coords_final.T)] = True
        markers = measure.label(mask)
        labels = watershed(-dist.copy(), markers, mask=self._img_label_pred, connectivity=2)

        props = measure.regionprops(labels)

        # int label
        imsave(file_instance_label, img_as_ubyte(labels), metadata={'axes': 'ZYX'})

        # overlay
        img_overlay = np.zeros(shape=(*self.shape, 3), dtype=self.dtype)
        if self.ch_cell >= 0:
            img_overlay[..., 0] = self._img_3d_cell
        elif self.ch_bf >= 0:
            img_overlay[..., 0] = self._img_3d[:, :, :, self.ch_bf]
        img_overlay[..., 1] = self._img_3d_syn
        for p in props:
            img_overlay[labels == p.label, 2] = np.iinfo(img_overlay.dtype).max - p.label
        imsave(file_overlay, img_overlay, metadata={'axes': 'ZYXC'})

        # colored version
        colors_rgb = [(name.split(':')[1], mcolors.to_rgb(color)) for name, color in mcolors.TABLEAU_COLORS.items()]
        len_c = len(colors_rgb)
        img_label_color = np.zeros(shape=(*self.shape, 3), dtype=float)
        for p in props:
            c = colors_rgb[p.label % len_c]
            img_label_color[labels == p.label] = c[1]

        imsave(file_color_label, img_as_ubyte(img_label_color))

    # endregion
    def get_confusion_matrix_pixelwise(self):
        assert hasattr(self, '_img_label'), "Label not loaded"
        assert hasattr(self, '_img_label_pred'), "Predicted label not loaded"

        return confusion_matrix(self._img_label.flatten(), self._img_label_pred.flatten())

    def get_intersection_over_union_pixelwise(self):
        assert hasattr(self, '_img_label'), "Label not loaded"
        assert hasattr(self, '_img_label_pred'), "Predicted label not loaded"

        intersection = np.logical_and(self._img_label, self._img_label_pred)
        union = np.logical_or(self._img_label, self._img_label_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    # region getter
    def get_img_3d_synapse_marker(self):
        if not hasattr(self, '_img_3d_syn'):
            self._img_3d_rgb, self._img_3d_syn, self._img_3d_cell = self._split_channels(self._img_3d, 'channel')

        return self._img_3d_syn

    def get_img_3d_cellular_marker(self):
        if not hasattr(self, '_img_3d_cell'):
            self._img_3d_rgb, self._img_3d_syn, self._img_3d_cell = self._split_channels(self._img_3d, 'channel')

        return self._img_3d_cell

    def get_scaling(self):
        """
        return scaling factor for z, y, x dimension
        Returns:

        """
        if not all([hasattr(self, '_scaling_x'), hasattr(self, '_scaling_y'), hasattr(self, '_scaling_z')]):
            return None, None, None
        else:
            return self._scaling_z, self._scaling_y, self._scaling_x
    # endregion
