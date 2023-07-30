import argparse
import gc
import glob
import logging
import logging.handlers
import os
import re
import sys
import time
from multiprocessing import current_process, Pool

import numpy as np
import torch
from skimage.io import imread

from gui.synapsecorrector import SynapseCorrector
from segmentutil.synapse_classification import SynapseClassifier_SVM, SynapseClassifier_AdaBoost, SynapseClassifier_RF
from segmentutil.synapse_quantification import SynapseQT3D
from segmentutil.unet.models import UNET25D_Atrous, UNet_Multi_Scale
from segmentutil.utils import suppress_warning
from segmentutil.worm_image_2d import WormImage


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources'))

    return os.path.join(base_path, relative_path)


def get_all_microscopy_images(target_dir):
    images = glob.glob(os.path.join(target_dir, '*.tif')) + glob.glob(os.path.join(target_dir, '*.czi')) + glob.glob(
        os.path.join(target_dir, '*.nd2'))
    images.sort()
    return images


def init_logger_with_log_queue(log_q) -> logging.Logger:
    logger = logging.getLogger()
    if len(logger.handlers) == 0 and log_q is not None:
        h = logging.handlers.QueueHandler(log_q)  # Just the one handler needed
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
    return logger


def mask_edge(base_dir, raw_dir, mask_dir, channel_text, minimize_storage=True, log_q=None):
    logger = init_logger_with_log_queue(log_q)

    if 'N' not in channel_text:
        logger.info('Pass the masking process: Single-channel images')
        return

    logger.info('Start generating mask with the edge-detection method')

    if not os.path.isdir(os.path.join(base_dir, raw_dir)):
        logger.error("Raw directory doesn't exist. Stop processing.")
        return

    raw_images = get_all_microscopy_images(os.path.join(base_dir, raw_dir))
    if len(raw_images) == 0:
        logger.warning("No images in raw directory. Stop processing.")
        return

    if not os.path.exists(os.path.join(base_dir, mask_dir)):
        os.makedirs(os.path.join(base_dir, mask_dir))
    for file_raw in raw_images:
        file_name = os.path.basename(file_raw).split('.')[0]
        mask_file = WormImage.name_cell_mask(os.path.join(base_dir, mask_dir), file_name)
        if os.path.isfile(mask_file):
            logger.info('<%s> Mask exists. Pass processing' % file_name)
            continue
        worm_img = WormImage(base_dir, file_raw, minimize_storage=minimize_storage, channel=channel_text,
                             dir_denoise='denoise', dir_contrast='contrast', dir_mask=mask_dir, logger=logger)
        worm_img.preprocess(blur_rfp=True,
                            blur_gfp=True,
                            contrast_gfp=(0.7, 0.9999, np.uint8),
                            contrast_rfp=(0.7, 0.997, np.uint8))
        logger.info('<%s> Start segmenting neurite' % file_name)
        worm_img.masking(mask_prune_cc=False)
        logger.info('<%s> Finish segmenting neurite' % file_name)
    logger.info('Finish generating mask with the edge-detection based method')


def mask_unet(base_dir, raw_dir, mask_dir, model_option, channel_text, minimize_storage=True, log_q=None):
    logger = init_logger_with_log_queue(log_q)

    if 'N' not in channel_text:
        logger.info('Pass the masking process: Single-channel images')
        return

    logger.info('Start generating mask with the UNet based method without training')

    if not os.path.isdir(os.path.join(base_dir, raw_dir)):
        logger.error("Raw directory doesn't exist. Stop processing.")
        return

    raw_images = get_all_microscopy_images(os.path.join(base_dir, raw_dir))
    if len(raw_images) == 0:
        logger.warning("No images in raw directory. Stop processing.")
        return

    if not os.path.exists(os.path.join(base_dir, mask_dir)):
        os.makedirs(os.path.join(base_dir, mask_dir))

    # load pre-trained model
    model_name = re.search('\((.*)\)', model_option).group(1)
    if model_name == 'neurite.pt':
        model = UNET25D_Atrous(num_channels=1, num_classes=1)
    elif model_name == 'multi_scale.pt':
        model = UNet_Multi_Scale(num_channels=1, num_classes=2)
    else:
        logging.error("Undefined model")
        return
    if torch.cuda.is_available():
        logger.info('Load %s, GPU mode' % model_name)
        model.load_state_dict(
            torch.load(resource_path(os.path.join('unet_model', model_name))))  # Load trained network
    else:
        logger.info('Load %s, CPU mode' % model_name)
        model.load_state_dict(
            torch.load(resource_path(os.path.join('unet_model', model_name)),
                       map_location=torch.device('cpu')))  # Load trained network

    for i, file_raw in enumerate(raw_images):
        file_name = os.path.basename(file_raw).split('.')[0]
        mask_file = WormImage.name_cell_mask(os.path.join(base_dir, mask_dir), file_name)
        if os.path.isfile(mask_file):
            logger.info('[%d/%d] <%s> Mask exists. Pass processing' % (i + 1, len(raw_images), file_name))
            continue
        worm_img = WormImage(base_dir, file_raw, channel=channel_text, minimize_storage=minimize_storage,
                             dir_mask=mask_dir, logger=logger)
        worm_img.preprocess_unet()
        logger.info('[%d/%d] <%s> Start UNet prediction' % (i + 1, len(raw_images), file_name))
        worm_img.masking_unet(model)
        logger.info('[%d/%d] <%s> Finish UNet prediction' % (i + 1, len(raw_images), file_name))
    logger.info('Finish generating mask with the UNet based method without training')


def train(base_dir, raw_dir, label_dir, train_dir, n_p, channel_text, model, n_max_patches=-1,
          minimize_storage=True, log_q=None):
    logger = init_logger_with_log_queue(log_q)
    logger.info('Start training a supervised model for syanpse segmentation')

    if not os.path.isdir(os.path.join(base_dir, label_dir)):
        logger.error("Label directory doesn't exist. Stop processing.")
        return

    # masking or not
    if 'N' in channel_text:
        masking = True
    else:
        masking = False

    # get label image list
    training_images = []
    training_images.extend(glob.glob(os.path.join(base_dir, label_dir, '*_label.h5')))
    training_images.extend(glob.glob(os.path.join(base_dir, label_dir, '*_label.tif')))
    if len(training_images) == 0:
        logger.error("No images in label directory. Stop processing.")
        return

    if not os.path.exists(os.path.join(base_dir, train_dir)):
        os.makedirs(os.path.join(base_dir, train_dir))

    if model == SynapseClassifier_SVM:
        classifier_1st = SynapseClassifier_SVM('1st', True, logger)
        classifier_2nd = SynapseClassifier_SVM('2nd', False, logger)
    elif model == SynapseClassifier_RF:
        classifier_1st = SynapseClassifier_RF('1st', logger=logger)
        classifier_2nd = SynapseClassifier_RF('2nd', logger=logger)
    elif model == SynapseClassifier_AdaBoost:
        classifier_1st = SynapseClassifier_AdaBoost('1st', logger)
        classifier_2nd = SynapseClassifier_AdaBoost('2nd', logger)
    else:
        logger.error("Undefined classifier model. Stop processing.")
        return

    train_dir_full = os.path.join(base_dir, train_dir)
    if classifier_1st.load_model(train_dir_full) and classifier_2nd.load_model(train_dir_full):  # load dump
        logger.info('Trained model already exists. Skip training.')
    else:
        # label loading
        worm_imgs = []
        for file_training in training_images:
            extension = os.path.splitext(file_training)[1]
            file_name = '_label'.join(os.path.basename(file_training).split('_label')[:-1])
            raw_file_candidates = glob.glob(os.path.join(base_dir, label_dir, '%s.*' % file_name))
            if len(raw_file_candidates) > 1:
                logger.error(
                    "Raw file for %s has multiple candidates. Please keep only one of them and try again." % file_name)
                return
            elif len(raw_file_candidates) == 0:
                logger.warning(
                    "Raw file for %s doesn't exist in the label directory. Try finding it in the raw directory" % file_name)
                raw_file_candidates = glob.glob(os.path.join(base_dir, raw_dir, '%s.*' % file_name))
                if len(raw_file_candidates) > 1:
                    logger.error(
                        "Raw file for %s has multiple candidates. Please keep only one of them and try again." % file_name)
                    return
                elif len(raw_file_candidates) == 0:
                    logger.error("Raw file for %s doesn't exist in the raw directory." % file_name)
                    return
                else:
                    raw_file = raw_file_candidates[0]
            else:
                raw_file = raw_file_candidates[0]

            logger.info('Load %s for training' % file_training)
            worm_img = WormImage(base_dir, raw_file, minimize_storage=minimize_storage,
                                 channel=channel_text, logger=logger, apply_masking=masking)
            worm_img.preprocess()
            if extension == '.h5':
                worm_img.read_label_image_h5(file_training)
            elif extension == '.tif':
                worm_img.read_label_image_tif(file_training)

            worm_imgs.append(worm_img)

            # initial step
            try:
                training_data = worm_img.get_training_local_features(num_process=n_p, n_max_patches=n_max_patches)
                if training_data is not None:
                    # training - all
                    classifier_1st.add_trainig_data(*training_data)
            except BaseException as ex:
                logger.error(ex)

        # training
        tic = time.perf_counter()
        logger.info('Start training Model - initial')
        classifier_1st.train(n_process=n_p)
        toc = time.perf_counter()
        logger.info('Finish training 1st layer 1st iteration in %0.4f seconds' % (toc - tic))

        # 2nd step - predict and add patches having false positives to training set
        for wi in worm_imgs:
            try:
                training_data = wi.get_training_pixel_features_2nd_nearTP(classifier_1st, num_process=n_p,
                                                                          n_max_patches=n_max_patches)
                if training_data is not None:
                    # training - all
                    classifier_1st.add_trainig_data(*training_data)
            except BaseException as ex:
                logger.error(ex)

        # training
        tic = time.perf_counter()
        logger.info('Start training Model - 1st layer 2nd iteration')
        classifier_1st.train(n_process=n_p)
        toc = time.perf_counter()
        logger.info('Finish training 1st layer 2nd iteration in %0.4f seconds' % (toc - tic))

        classifier_1st.save_model(os.path.join(base_dir, train_dir))

        # second layer
        for wi in worm_imgs:
            try:
                training_data_2 = wi.get_training_prob_features(classifier_1st.predict_proba, num_process=n_p,
                                                                n_max_patches=n_max_patches)
                if training_data_2 is not None:
                    # training - all
                    classifier_2nd.add_trainig_data(*training_data_2)
            except BaseException as ex:
                logger.error(ex)

        # training
        tic = time.perf_counter()
        logger.info('Start training Model - 2nd layer')
        classifier_2nd.train(n_process=n_p)
        toc = time.perf_counter()
        logger.info('Finish training 2nd layer in %0.4f seconds' % (toc - tic))
        classifier_2nd.save_model(os.path.join(base_dir, train_dir))

    logger.info('Finish training a supervised model for syanpse segmentation')


def predict(base_dir, raw_dir, mask_dir, predict_dir, n_p, small_synapse_cutoff, channel_text, option_classifier,
            b_masking, minimize_storage=True, log_q=None, ):
    """

    Args:
        base_dir: Base directory
        raw_dir: Raw files directory
        predict_dir: Prediction output directory
        n_p: number of processes to use
        small_synapse_cutoff: threshold for removing small synapses
        channel_text: Channel order text
        option_classifier:
            Tuple.  option_classifier[0]: 'Built-in' or 'Custom'
                    option_classifier[1]: Classifier model or directory
        minimize_storage: If true, omit some intermediate result images
        log_q:
        **kwargs:

    Returns:

    """
    logger = init_logger_with_log_queue(log_q)
    logger.info('Start prediction using the trained model')

    # directory checking
    if not os.path.isdir(os.path.join(base_dir, raw_dir)):
        logger.error("Raw directory doesn't exist. Stop processing.")
        return

    raw_images = get_all_microscopy_images(os.path.join(base_dir, raw_dir))
    if len(raw_images) == 0:
        logger.warning("No images in raw directory. Stop processing.")
        return

    output_dir = os.path.join(base_dir, predict_dir)
    output_dir_overlay = os.path.join(base_dir, predict_dir, 'overlay')
    os.makedirs(output_dir_overlay, exist_ok=True)

    # Synapse channel exists?
    if 'N' in channel_text:
        b_singlechannel = False
    else:
        b_singlechannel = True

    # set classifier
    model = None
    train_dir_full = ''
    if option_classifier[0] == 'Built-in':
        if b_singlechannel:
            # TODO: further categorize
            train_dir_full = resource_path(os.path.join('synapse_classifiers', 'NSM_CLA1_no_mask_RFCV_v112'))
            model = SynapseClassifier_RF
        elif option_classifier[1] == 'GRASP (sparse)':
            train_dir_full = resource_path(os.path.join('synapse_classifiers', 'otIs612_GRASP_RFCV_v112'))
            model = SynapseClassifier_RF
        elif option_classifier[1] == 'GRASP (dense)':
            train_dir_full = resource_path(os.path.join('synapse_classifiers', 'otIs653_GRASP_RFCV_v112'))
            model = SynapseClassifier_RF
        elif option_classifier[1] == 'CLA-1':
            train_dir_full = resource_path(os.path.join('synapse_classifiers', 'I5_CLA1_RFCV_v112'))
            model = SynapseClassifier_RF
        elif option_classifier[1] == 'RAB-3':
            train_dir_full = resource_path(os.path.join('synapse_classifiers', 'ASK_RAB3_RFCV_v112'))
            model = SynapseClassifier_RF
    elif option_classifier[0] == 'Custom':
        train_dir_full = os.path.join(base_dir, option_classifier[1])
        clf_files = glob.glob(os.path.join(base_dir, option_classifier[1], 'sk_*_*.dump'))
        for clf_path in clf_files:
            clf_dump_name = os.path.basename(clf_path)
            identifier = clf_dump_name.split('_')[1]
            if identifier == 'scaler':
                continue
            elif identifier == 'svm':
                if model is None:
                    model = SynapseClassifier_SVM
                elif model != SynapseClassifier_SVM:
                    logger.error(
                        "More than one classifier in classifier_dir. Please have just one classifier. Stop processing.")
                    return
            elif identifier == 'rf':
                if model is None:
                    model = SynapseClassifier_RF
                elif model != SynapseClassifier_RF:
                    logger.error(
                        "More than one classifier in classifier_dir. Please have just one classifier. Stop processing.")
                    return
            elif identifier == 'ada':
                if model is None:
                    model = SynapseClassifier_AdaBoost
                elif model != SynapseClassifier_AdaBoost:
                    logger.error(
                        "More than one classifier in classifier_dir. Please have just one classifier. Stop processing.")
                    return
    else:
        logger.error("Unknown error")
        return

    if model is None:
        logger.error("Undefined classifier model. Stop processing.")
        return

    if model == SynapseClassifier_SVM:
        classifier_1st = SynapseClassifier_SVM('1st', True, logger)
        classifier_2nd = SynapseClassifier_SVM('2nd', False, logger)
    elif model == SynapseClassifier_RF:
        classifier_1st = SynapseClassifier_RF('1st', logger=logger)
        classifier_2nd = SynapseClassifier_RF('2nd', logger=logger)
    elif model == SynapseClassifier_AdaBoost:
        classifier_1st = SynapseClassifier_AdaBoost('1st', logger)
        classifier_2nd = SynapseClassifier_AdaBoost('2nd', logger)
    else:
        logger.error("Undefined classifier model. Stop processing.")
        return

    # check if a trained classifier exists
    if not (classifier_1st.load_model(train_dir_full) and classifier_2nd.load_model(train_dir_full)):  # load dump
        logger.error("Trained model doesn't exist. Stop processing.")
        return
    else:
        # iterate raw images and do the prediction
        for i, file_raw in enumerate(raw_images):
            file_name = os.path.basename(file_raw).split('.')[0]
            file_label = os.path.join(output_dir, '%s_pred.tif' % file_name)
            file_overlay = os.path.join(output_dir_overlay, '%s_pred_overlay.tif' % file_name)

            # check if the prediction already exists
            if os.path.isfile(file_label) and os.path.isfile(file_overlay):
                logger.info('[%d/%d] Prediction on %s has already done. pass prediction.' % (
                    i + 1, len(raw_images), os.path.basename(file_raw)))
                continue
            else:
                logger.info(
                    '[%d/%d] Start predicting %s' % (i + 1, len(raw_images), os.path.basename(file_raw)))
                # prediction here
                worm_img = WormImage(base_dir, file_raw, minimize_storage=minimize_storage, channel=channel_text,
                                     dir_mask=mask_dir, apply_masking=b_masking, logger=logger)
                worm_img.preprocess()
                worm_img.masking()
                try:
                    pred_label = worm_img.predict(classifier_1st.predict_proba, classifier_2nd.predict,
                                                  small_synapse_cutoff=small_synapse_cutoff, num_process=n_p)
                    worm_img.write_predicted_label_image(pred_label, file_label, file_overlay)
                except BaseException as ex:
                    logger.error(ex)
                finally:
                    logger.info(
                        '[%d/%d] Done predicting %s' % (i + 1, len(raw_images), os.path.basename(file_raw)))
    logger.info('Finish prediction using the trained model')


def _watershed(i, n, base_dir, file_raw, input_dir, output_dir, output_dir_overlay, output_dir_color, min_distance,
               channel_text):
    logger = logging.getLogger()
    file_name = os.path.basename(file_raw).split('.')[0]
    file_label = os.path.join(input_dir, '%s_pred.tif' % file_name)
    file_instance_label = os.path.join(output_dir, '%s_pred.tif' % file_name)
    file_overlay = os.path.join(output_dir_overlay, '%s_pred_overlay.tif' % file_name)
    file_instance_label_color = os.path.join(output_dir_color, '%s_pred_color.tif' % file_name)

    # check if the instance labels exists
    if os.path.isfile(file_instance_label):
        logger.info('[%d/%d] %s: The instance labels of %s already exists. Pass the process.' % (
            i + 1, n, current_process().name, os.path.basename(file_raw)))
        return

    # check if the prediction label (semantic segmentation) exists
    if os.path.isfile(file_label):
        logger.info('[%d/%d] %s: %s Watersheding...' % (i + 1, n, current_process().name, os.path.basename(file_raw)))
        # watershed - instance segmentation
        worm_img = WormImage(base_dir, file_raw, minimize_storage=True, channel=channel_text,
                             apply_masking=False, logger=logger)
        worm_img.preprocess()
        worm_img.read_predicted_label_image(file_label)
        worm_img.segment_instances(file_instance_label, file_overlay, file_instance_label_color, min_distance)
    else:
        logger.warning('[%d/%d] %s: Semantic label does not exist %s' % (
            i + 1, n, current_process().name, os.path.basename(file_raw)))


def watershed(base_dir, raw_dir, in_dir, out_dir, min_distance, n_p, channel_text, log_q=None, ):
    """
    Perform the instance segmentation based on the binary semantic synapse segmentation result and the raw
    synapse channel signal
    Args:
        base_dir: Base directory
        raw_dir: Raw files directory
        in_dir: Watersheding input directory
        out_dir: Watersheding output directory
        min_distance: The minimal allowed distance separating peaks. (used in skimage.feature.peak_local_max)
        n_p: number of processes to use
        channel_text: Channel order text

    Returns:

    """
    logger = init_logger_with_log_queue(log_q)
    logger.info('Start watersheding')

    # directory checking
    if not os.path.isdir(os.path.join(base_dir, raw_dir)):
        logger.error("Raw directory doesn't exist. Stop processing.")
        return

    raw_images = get_all_microscopy_images(os.path.join(base_dir, raw_dir))
    if len(raw_images) == 0:
        logger.warning("No images in raw directory. Stop processing.")
        return

    input_dir = os.path.join(base_dir, in_dir)
    output_dir = os.path.join(base_dir, out_dir)
    output_dir_overlay = os.path.join(output_dir, 'overlay')
    os.makedirs(output_dir_overlay, exist_ok=True)
    output_dir_color = os.path.join(output_dir, 'color')
    os.makedirs(output_dir_color, exist_ok=True)

    if n_p > 1:
        logger.info('Work with %d processes' % n_p)
        args = [(i, len(raw_images), base_dir, file_raw, input_dir, output_dir, output_dir_overlay,
                 output_dir_color, min_distance, channel_text) for i, file_raw in enumerate(raw_images)]
        with Pool(n_p, init_logger_with_log_queue, [log_q, ]) as pool:
            pool.starmap(_watershed, args)
    else:
        logger.info('Work with a single processes')
        # iterate raw images and do the watersheding
        for i, file_raw in enumerate(raw_images):
            _watershed(i, len(raw_images), base_dir, file_raw, input_dir, output_dir, output_dir_overlay,
                       output_dir_color, min_distance, channel_text)
    logger.info('Finish watersheding')


def correct(base_dir, predict_dir, correct_dir, close_callback, log_q=None):
    logger = init_logger_with_log_queue(log_q)
    logger.info('Start Synapse Corrector')
    input_dir = os.path.join(base_dir, predict_dir, 'overlay')
    output_dir = os.path.join(base_dir, correct_dir, 'overlay')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    window_corrector = SynapseCorrector(input_dir, output_dir, logger)

    def callback_with_log():
        window_corrector.on_delete()
        window_corrector.destroy()
        gc.collect()
        logger.info('Synapse Corrector closed')
        close_callback()

    window_corrector.protocol("WM_DELETE_WINDOW", callback_with_log)
    window_corrector.load_images_from_dir(input_dir)


def _quantify(i, n, col_props, base_dir, file_raw, mask_dir, input_dir, output_dir, channel_text):
    # setup logger for this process
    logger = logging.getLogger()
    file_name = os.path.basename(file_raw).split('.')[0]

    # skip if the predicted label file does not exist
    predicted_labels = os.path.join(base_dir, input_dir, '%s_pred.tif' % file_name)
    is_instance_segmentation = False
    if not os.path.isfile(predicted_labels):
        # in case there is an instance label
        predicted_labels = os.path.join(base_dir, input_dir, '%s_pred_instance.tif' % file_name)
        is_instance_segmentation = True
        if not os.path.isfile(predicted_labels):
            logger.info(
                "[%d/%d] Skip analyzing %s: prediction doesn't exist. It may be pruned out during correction" % (
                    i + 1, n, os.path.basename(file_raw)))
            return None

    # file to write the stat
    file_result = os.path.join(base_dir, output_dir, '%s_synapses.csv' % file_name)
    logger.info('[%d/%d] Start quantifying %s' % (i + 1, n, os.path.basename(file_raw)))

    # label file and original image file
    img_label = imread(predicted_labels)
    worm_img = WormImage(base_dir, file_raw, minimize_storage=True, channel=channel_text,
                         apply_masking=False, logger=logger)

    # if neurite mask exists, set it as well
    file_neurite_mask = os.path.join(base_dir, mask_dir, '%s_rfp_mask.tif' % file_name)
    if not os.path.isfile(file_neurite_mask):
        img_mask = None
    else:
        img_mask = imread(file_neurite_mask) > 0

    qt = SynapseQT3D(img_label, worm_img.get_img_3d_synapse_marker(), img_mask, is_instance_segmentation)
    # write result
    with open(file_result, 'w') as h:
        h.write(
            ','.join(['Name', 'Synapse ID', 'Centroid_z', 'Centroid_y', 'Centroid_x'] + col_props) + '\n')
        for j in range(qt.number()):
            centroid = qt.prop(j, 'centroid')
            h.write(','.join([file_name, str(j + 1), str(centroid[0]), str(centroid[1]), str(centroid[2])] +
                             [str(qt.prop(j, col)) for col in col_props]) + '\n')

    logger.info('[%d/%d] Done quantifying %s' % (i + 1, n, os.path.basename(file_raw)))

    # get mean inter-synapse distance
    s_z, s_y, s_x = worm_img.get_scaling()
    misd_px, misd_si = qt.mean_inter_synapse_distance((s_z, s_y, s_x))
    total_volume_px = qt.sum_prop('area')
    if None in (s_z, s_y, s_x):
        total_volume_si = np.NaN
    else:
        total_volume_si = total_volume_px * s_z * s_y * s_x

    return ((file_name,
             qt.number(),
             total_volume_px,
             total_volume_si,
             qt.mean_intensity_of_all_positives(),
             *qt.spatial_dispersion_rms((s_z, s_y, s_x)),
             misd_px, misd_si,
             *[qt.mean_prop(col) for col in col_props]),
            (file_name, *qt.deep_phynotyping())
            )
    # TODO: Mean normalized intensity


def quantify(base_dir, raw_dir, mask_dir, input_dir, output_dir, n_p, channel_text, log_q=None):
    """
    Quantify segmented synapses for the images in the base directory.
    Args:
        base_dir: Base directory
        raw_dir: Raw files directory
        input_dir: Directory where the input prediction files are
        output_dir: Directory where the output quantification result will be placed
        channel_text: Channel order text
        log_q:

    Returns:

    """
    logger = logging.getLogger()
    if len(logger.handlers) == 0 and log_q is not None:
        h = logging.handlers.QueueHandler(log_q)  # Just the one handler needed
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
    logger.info('Start quantification')

    if not os.path.isdir(os.path.join(base_dir, raw_dir)):
        logger.error("Raw directory doesn't exist. Stop processing.")
        return

    raw_images = get_all_microscopy_images(os.path.join(base_dir, raw_dir))
    if len(raw_images) == 0:
        logger.warning("No images in raw directory.")
        return

    if not os.path.exists(os.path.join(base_dir, output_dir)):
        os.makedirs(os.path.join(base_dir, output_dir))

    # quantification
    # column for individual synapse feature file
    col_props = ['area', 'mean_intensity', 'max_intensity', 'min_intensity', 'bbox_area', 'equivalent_diameter',
                 'euler_number', 'extent', 'filled_area', 'major_axis_length', 'minor_axis_length']
    # lists for animal-wise feature
    group_stats = []
    # 8 for 'Number', 'Total Volume (pixels)', 'Total Volume (m^3)',
    #       'Mean pixel intensity', 'Spatial Dispersion (pixels)', 'Spatial Dispersion (Meters)',
    #       'Mean inter-synapse distance (pixels)', 'Mean inter-synapse distance (Meters)'
    # len(col_props) for mean value of each synapse-wise features
    avg_group_stat = np.zeros(dtype=float, shape=8 + len(col_props))
    deep_phy_stat = []

    def parse_result(r, gs, ags, dps):
        # gs: group stats list
        gs.append(r[0])
        # ags: average group stats list
        try:
            ags += np.array(np.nan_to_num(r[0][1:]))
        except Exception as ex:
            print(ex)
        # dps: deep phenotying stat list
        dps.append(r[1])

    if n_p > 1:
        logger.info('Work with %d processes' % n_p)
        args = [(i, len(raw_images), col_props, base_dir, file_raw, mask_dir, input_dir, output_dir, channel_text,)
                for i, file_raw in enumerate(raw_images)]
        with Pool(n_p, init_logger_with_log_queue, [log_q, ]) as pool:
            results = pool.starmap(_quantify, args)
            # parsing results
            logger.info('Start collecting results (n=%d)' % len(raw_images))
            for res in results:
                if res is not None:
                    parse_result(res, group_stats, avg_group_stat, deep_phy_stat)
    else:
        logger.info('Work with a single processes')
        # iterate raw images and do the watersheding
        for i, file_raw in enumerate(raw_images):
            res = _quantify(i, len(raw_images), col_props, base_dir, file_raw, mask_dir, input_dir, output_dir,
                            channel_text)
            # parsing results
            if res is not None:
                parse_result(res, group_stats, avg_group_stat, deep_phy_stat)

    if len(group_stats) > 0:
        avg_group_stat /= len(group_stats)
        group_stats.sort(key=lambda x: x[0])
        group_stats.append(['Average', ] + list(avg_group_stat))
        with open(os.path.join(base_dir, output_dir, 'quantification_stat.csv'), 'w') as h:
            h.write(','.join(['Name', 'Number', 'Total Volume (pixels)', 'Total Volume (m^3)',
                              'Mean pixel intensity', 'Spatial Dispersion (pixels)', 'Spatial Dispersion (Meters)',
                              'Mean inter-synapse distance (pixels)', 'Mean inter-synapse distance (Meters)'
                              ] + ['Mean ' + col for col in col_props]) + '\n')
            h.writelines([','.join([str(s) for s in stat]) + '\n' for stat in group_stats])

        with open(os.path.join(base_dir, output_dir, 'quantification_stat_deep.csv'), 'w') as h:
            for stat in deep_phy_stat:
                if stat[1] is None:
                    h.write('%s\n' % stat[0])
                else:
                    h.write(','.join([str(s) for s in stat]) + '\n')
        logger.info('Finish quantification')
    else:
        logger.error('No images quantified.')


def main():
    suppress_warning()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(
        description='''NeuroNex Synapse Segmentator (last updated 2020-07-02)''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mask',
                        choices=['edge', 'unet'],
                        default='',
                        help="Generate a neurite mask for each image from the red channel.")
    parser.add_argument('--min_storage',
                        help=r'Minimize storage by not producing unnecessary images',
                        action='store_true')
    parser.add_argument('--train',
                        help='Train a SVM model and dump the parameter as a file.',
                        action='store_true')
    parser.add_argument('--predict',
                        help='Perform segmentation (SVM prediction).',
                        action='store_true')
    parser.add_argument('--quantify',
                        help='Perform quantification.',
                        action='store_true')
    parser.add_argument('--correct',
                        help='Perform correction.',
                        action='store_true')
    parser.add_argument('--dir',
                        type=str,
                        help=r'Designate the working directory (which should contain "raw" directory inside', )
    parser.add_argument('--dir_predict',
                        type=str,
                        default='prediction',
                        help=r'Designate the prediction directory')
    parser.add_argument('--dir_quantify',
                        type=str,
                        default='quantification',
                        help=r'Designate the prediction directory')
    parser.add_argument('--channel',
                        type=str,
                        default='RGB',
                        help=r'Designate the channel order', )
    parser.add_argument('--convert_8bit',
                        action='store_true',
                        help=r'Type it when the original images is not in 8bit format.', )
    parser.add_argument('--p',
                        type=int,
                        default=1,
                        help=r'Designate the number of process to use', )
    args = vars(parser.parse_args())
    base_dirs = []
    if args['dir'] is None:
        logging.error('Please use --dir keyword to notify a target directory.')
        parser.print_help()
        return
    else:
        # check if it's single folder or multiple folders
        for walked in os.walk(args['dir']):
            if os.path.basename(walked[0]) == 'raw':
                base_dirs.append(os.path.dirname(walked[0]))

    if args['convert_8bit']:
        for base_dir in base_dirs:
            if not os.path.exists(os.path.join(base_dir, 'mask_8bit')):
                os.makedirs(os.path.join(base_dir, 'mask_8bit'))

    if args['mask'] == 'edge':
        for base_dir in base_dirs:
            mask_edge(base_dir, 'raw', 'mask', args['channel'], args['min_storage'])

    if args['mask'] == 'unet':
        for base_dir in base_dirs:
            mask_unet(base_dir, 'raw', 'mask', args['channel'], args['min_storage'])

    if args['train']:
        for base_dir in base_dirs:
            train(base_dir, 'raw', 'label', 'classifier', args['p'], args['channel'], SynapseClassifier_SVM,
                  args['min_storage'])

    if args['predict']:
        for base_dir in base_dirs:
            predict(base_dir, 'raw', 'prediction', args['p'], 1, args['channel'], ('Built-in', 'GRASP (sparse)'), True,
                    args['min_storage'])

    if args['correct']:
        correct(base_dirs[0], args['dir_predict'], '%s_corrected' % args['dir_predict'], lambda: None)

    if args['quantify']:
        for base_dir in base_dirs:
            quantify(base_dir, 'raw', args['dir_predict'], args['dir_quantify'], args['p'], args['channel'])


if __name__ == '__main__':
    main()
