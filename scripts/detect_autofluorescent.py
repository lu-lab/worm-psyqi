import os
import numpy as np
from skimage import img_as_ubyte, measure
from skimage.io import imsave, imread
import glob
import matplotlib.colors as mcolors
from skimage.morphology import binary_erosion, binary_dilation, skeletonize
from scipy import ndimage as ndi
import argparse
import logging
import logging.handlers


def get_surface_volume(img):
    diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
    img_inner = binary_erosion(img, diamond)

    return int(np.sum(np.logical_and(img, ~img_inner)))


def get_region_thickness(img):
    d = 0
    diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
    while np.sum(img) > 0:
        img = binary_erosion(img, diamond)
        d += 1
    return d


def get_longest_arm(img):
    skel = skeletonize(img)

    def find_extreme_points(img_skel):
        kernel = np.ones(shape=(3, 3, 3))

    def find_longest_distance(extreme_points):
        pass


def mask_stat(mask_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    colors_rgb = [(name.split(':')[1], mcolors.to_rgb(color)) for name, color in mcolors.TABLEAU_COLORS.items()]
    len_c = len(colors_rgb)
    mask_imgs = glob.glob(os.path.join(mask_dir, '*.tif'))

    lines = []
    for mask_file in mask_imgs:
        name = os.path.basename(mask_file)
        img_mask = (imread(mask_file) > 0)
        img_mask_color = np.zeros(shape=(*img_mask.shape, 3), dtype=float)

        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        for _ in range(5):
            img_mask = binary_dilation(img_mask, diamond)

        label = measure.label(img_mask, connectivity=2)
        props = measure.regionprops(label)

        for p in props:
            c = colors_rgb[p.label % len_c]
            img_mask_color[label == p.label] = c[1]
            elongation = p.area / get_region_thickness(label == p.label) ** 2
            sv = get_surface_volume(label == p.label)
            lines.append('%s,%d,%s,%d,%f,%d\n' % (name, p.label, c[0], p.area, elongation, sv))

        imsave(os.path.join(out_dir, name), img_as_ubyte(img_mask_color))

    info_file = os.path.join(out_dir, 'info_mask.csv')
    with open(info_file, 'w') as h:
        h.write('file,label,color,area,elongation,SV\n')
        h.writelines(lines)


def remove_gut(base_dir, mask_dir_name, log_q):
    logger = logging.getLogger()
    if len(logger.handlers) == 0 and log_q is not None:
        h = logging.handlers.QueueHandler(log_q)  # Just the one handler needed
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)

    logger.info('Start post-processing for removing gut autofluorescence from the masks')
    mask_dir = os.path.join(base_dir, mask_dir_name)
    move_dir = os.path.join(mask_dir, 'before_postprocess')
    if not os.path.exists(move_dir):
        os.makedirs(move_dir)

    mask_imgs = glob.glob(os.path.join(mask_dir, '*.tif'))
    mask_imgs.sort()
    msgs = []
    for mask_file in mask_imgs:
        name = os.path.basename(mask_file)
        img_mask = (imread(mask_file) > 0)
        img_mask_di = img_mask.copy()
        region_stats = []

        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        for _ in range(5):
            img_mask_di = binary_dilation(img_mask_di, diamond)

        label = measure.label(img_mask_di, connectivity=2)
        props = measure.regionprops(label)

        for p in props:
            elongation = p.area / get_region_thickness(label == p.label) ** 2
            sv = get_surface_volume(label == p.label)
            region_stats.append(
                dict(label=p.label,
                     volume=p.area,
                     elongation=elongation,
                     SRVRR=sv ** 0.5 / p.area ** (1 / 3),
                     point=1)
            )

        # calculate points
        region_stats.sort(key=lambda x: x['volume'], reverse=True)
        for i, stat in enumerate(region_stats):
            stat['point'] *= stat['volume'] / region_stats[0]['volume']
        region_stats.sort(key=lambda x: x['elongation'], reverse=True)
        for i, stat in enumerate(region_stats):
            stat['point'] *= stat['elongation'] / region_stats[0]['elongation']
        region_stats.sort(key=lambda x: x['SRVRR'], reverse=True)
        for i, stat in enumerate(region_stats):
            stat['point'] *= stat['SRVRR'] / region_stats[0]['SRVRR']

        # select the region of neuron
        region_stats.sort(key=lambda x: x['point'], reverse=True)
        top_label = region_stats[0]['label']
        img_mask[label != top_label] = 0
        if len(region_stats) > 1 and region_stats[1]['point'] > 0.5:
            msgs.append('%s may have not been cleaned properly. Please check the cleaned output.\n' % name)
            logger.warning(msgs[-1])
        else:
            msgs.append('%s is cleaned.\n' % name)
            logger.info(msgs[-1])

        os.rename(mask_file, os.path.join(move_dir, name))
        imsave(mask_file, img_as_ubyte(img_mask))

    log_file = os.path.join(mask_dir, 'mask_cleaning_log.txt')
    with open(log_file, 'w') as h:
        h.writelines(msgs)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir',
                        type=str,
                        help=r'Designate the directory containing mask images', )
    args = vars(parser.parse_args())
    if args['dir'] is None:
        parser.print_help()
        return
    else:
        # check if it's single folder or multiple folders
        target_dir = args['dir']

    remove_gut(target_dir)


if __name__ == '__main__':
    main()
