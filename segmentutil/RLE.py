import numpy as np
import imageio as io
import os


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    # assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to length
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.uint8)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def rle_encode_3D(in_mask, output_dir):
    """
    Convert a 3D image stack into a RLE encoded txt file
    :param in_mask: should either be a numpy array or a path for image stack
    :param output_dir:
    :return:
    """
    if isinstance(in_mask, str) and os.path.isfile(in_mask):
        in_mask = np.array(io.mimread(in_mask, memtest=False), dtype=np.uint8)
    else:
        assert isinstance(in_mask, np.ndarray), 'the input must be a dir or a np array'

    if in_mask.ndim == 4 and in_mask.shape[-1] == 3:
        mask = in_mask[..., 2]  # blue channel in RGB
    elif in_mask.ndim == 3:
        mask = in_mask
    else:
        print('the shape is incorrect')

    with open(output_dir, 'w') as f:
        for i, single_slice in enumerate(mask):
            rle_slice = rle_encode(single_slice)
            msg = str(i) + ':' + rle_slice
            f.write(msg)
            f.write('\n')


def rle_decode_3D(in_dir, writing=False, **kwargs):
    """
    Convert a txt file into a numpy array. Have the option to convert back to a tif mask file. The default image size is 1142*1593 in size.
    :param in_dir: input txt file path
    :param writing: whether or not write the image into a tif
    :param kwargs: output path for tif file, img_path:str
    :return: a numpy array for mask
    """
    assert os.path.isfile(in_dir) and in_dir.endswith('txt'), 'the file must exist'
    f = open(in_dir, 'r')
    all_rle = f.read().split('\n')[:-1]
    shape = (len(all_rle), 1593, 1142)  # The image has to go with 1593*1142 by now
    mask = np.zeros(shape=shape, dtype=np.uint8)
    for i, msg in enumerate(all_rle):
        rle = msg.split(':')[-1]
        mask_single = rle_decode(rle, shape=shape[1:])
        mask[i] = mask_single
    f.close()

    if writing:
        if 'img_path' in kwargs:
            io.mimwrite(kwargs['img_path'], mask)
        else:
            print('use img_path as kwargs')
    return mask
