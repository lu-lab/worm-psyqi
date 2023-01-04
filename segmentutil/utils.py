import numpy as np
import scipy
from skimage import img_as_float32, measure
from skimage.io import imread, imsave
from skimage.morphology import disk, binary_dilation


def suppress_warning():
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1


def bounding_boxes(img):
    label = measure.label(img, connectivity=2)
    props = measure.regionprops(label)
    return [prop.bbox for prop in props]


def crop(img_3d, box_2d):
    img_cropped = img_3d[:, box_2d[1]:box_2d[1] + box_2d[3], box_2d[0]:box_2d[0] + box_2d[2]]
    return img_cropped


def decrop(img_cropped, box_2d, shape_original):
    img_decropped = np.zeros(shape=shape_original)
    img_decropped[:, box_2d[1]:box_2d[1] + box_2d[3], box_2d[0]:box_2d[0] + box_2d[2]] = img_cropped
    return img_decropped


def leave_one_big_chunk_3d(img_src: np.ndarray):
    s = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ]
    res, N = scipy.ndimage.measurements.label(img_src, structure=s)
    areas = []
    for ccid in range(1, N + 1):
        areas.append(int(np.sum(res == ccid)))
    max_area_ccid = np.argmax(areas) + 1

    # delete cc except for the largest one
    img_pruned = np.zeros(shape=img_src.shape, dtype=img_src.dtype)
    img_pruned[res == max_area_ccid] = np.iinfo(img_src.dtype).max

    return img_pruned


def stretch(image, minimum, maximum):
    min_img = np.min(image)
    max_img = np.max(image)
    if min_img != max_img:
        image = (image - min_img) / (max_img - min_img) * maximum + minimum
    image[image < minimum] = minimum
    image[image > maximum] = maximum
    return image


def enhance_contrast_histogram(src, lower_bound, upper_bound, dst_dtype):
    """
    Enhance contrast by truncating out of bound and uniformly stretching within bound
    Args:
        src: source image to enhance
        lower_bound: lower cutoff in proportion of input dynamic range (0~1)
        upper_bound: uppder cutoff in proportion of input dynamic range (0~1)
        dst_dtype: destination datatype

    Returns: stretched image, lower bound value, upper bound value

    """
    shape = src.shape
    minval = np.min(src)
    maxval = np.max(src)
    if minval == maxval:
        # if the image to stretch only contains an uniform value
        return src, minval, maxval
    else:
        hist, bins = np.histogram(src.flatten(), maxval, [0, maxval])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        i, lower_v, upper_v = 0, 0, 0
        for i, v in enumerate(cdf_normalized):
            if v > lower_bound:
                lower_v = i
                break
        for j, v in enumerate(cdf_normalized[i:]):
            if v >= upper_bound:
                upper_v = i + j + 1
                break

        dst = src.copy()
        dst[src < lower_v] = 0
        dst[src > upper_v] = upper_v

        img_global_stretch = stretch(img_as_float32(dst).flatten(), minimum=0, maximum=np.iinfo(dst_dtype).max)
        img_global_stretch = img_global_stretch.reshape(shape).astype(dst_dtype)
        return img_global_stretch, lower_v, upper_v


def enhance_contrast_value(src, lower_v, upper_v, dst_dtype):
    """
    Enhance contrast by truncating out of bound and uniformly stretching within bound
    Args:
        src: source image to enhance
        lower_bound: lower cutoff in integer value
        upper_bound: uppder cutoff in integer value
        dst_dtype: destination datatype

    Returns: stretched image, lower bound value, upper bound value

    """
    shape = src.shape
    dst = src.copy()
    dst[src < lower_v] = 0
    dst[src > upper_v] = upper_v

    img_global_stretch = stretch(img_as_float32(dst).flatten(), minimum=0, maximum=np.iinfo(dst_dtype).max)
    img_global_stretch = img_global_stretch.reshape(shape).astype(dst_dtype)
    return img_global_stretch, lower_v, upper_v

def binary_dilation_3d(img_in, size_xy, size_z):
    img_in = img_in > 0

    # dilate in xy direction
    img_dilated_xy = np.zeros_like(img_in)
    for z in range(img_dilated_xy.shape[0]):
        img_dilated_xy[z, :, :] = binary_dilation(img_in[z], selem=disk(size_xy))

    # dilate in z direction
    img_out = np.zeros_like(img_dilated_xy)
    for z in range(img_dilated_xy.shape[0]):
        if z < size_z:
            img_out[z, :, :] = np.max(img_dilated_xy[:z + size_z], axis=0)
        elif z >= img_dilated_xy.shape[0] - size_z:
            img_out[z, :, :] = np.max(img_dilated_xy[z - size_z:], axis=0)
        else:
            img_out[z, :, :] = np.max(img_dilated_xy[z - size_z:z + size_z], axis=0)

    return img_out

try:
    import winsound
except ImportError:
    import os


    def beep(frequency=440, duration=1):
        # apt-get install beep
        os.system('beep -f %s -l %s' % (frequency, duration))
else:
    def beep(frequency=440, duration=1):
        winsound.Beep(frequency, duration)
