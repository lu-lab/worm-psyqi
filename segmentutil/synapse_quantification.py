from enum import Enum
from typing import List

import numpy as np
from graph import Graph
from scipy import interpolate
from scipy.spatial import distance
from skimage import measure
from skimage.morphology import skeletonize_3d, dilation


class Synapse3D(object):
    class State(Enum):
        POSITIVE = 0
        NEGATIVE = 1
        SELECTED = 2
        NONE = 3

    def __init__(self, id, prop: measure._regionprops.RegionProperties):
        self.id = id
        self.prop = prop
        self.state = self.State.NONE

    def is_in(self, roi):
        """
        check if the center of this synapse is inside the roi
        Args:
            roi: should be ordered in (z0, z1, y0, y1, x0, x1)

        Returns:

        """
        for coord in self.prop.coords:
            if all((roi[0] <= coord[0] < roi[1],
                    roi[2] <= coord[1] < roi[3],
                    roi[4] <= coord[2] < roi[5])):
                return True
        return False


class SynapseQT3D(object):
    def __init__(self, img_label: np.ndarray, img_signal: np.ndarray, img_neurite_mask: np.ndarray = None,
                 is_instance_segmentation=False):
        self._img_label = img_label
        self._img_signal = img_signal
        self._img_mask = img_neurite_mask
        self._is_instance = is_instance_segmentation
        self._img_cc = None
        self._dispersion_pixel = None
        self._dispersion_si = None
        self._synapses: List[Synapse3D] = None
        self._misd_pixel = None  # mean inter-synapse distance, pixel unit
        self._misd_si = None  # mean inter-synapse distance, SI unit

    def _get_cc(self):
        if self._is_instance:
            label = self._img_label
        else:
            label = measure.label(self._img_label, connectivity=2)
        props = measure.regionprops(label)
        n_label = len(props)
        if n_label > 0:
            # re-sort synapse label by their vicinity
            label_remap = {}
            # 1 = closest to edge
            props.sort(key=lambda p: np.sum(p.centroid))
            label_remap[props[0].label] = 1
            last_prop = props[0]
            last_label = 1
            props.remove(last_prop)
            # next = closest to previous (sort every 1/10 of total labels)
            while len(props) > 0:
                props.sort(key=lambda p: np.linalg.norm(np.array(p.centroid) - np.array(last_prop.centroid)))
                for _ in range(int(min(max(n_label / 10, 1), len(props)))):
                    label_remap[props[0].label] = last_label + 1
                    last_prop = props[0]
                    last_label += 1
                    props.remove(last_prop)

            new_label = np.zeros_like(label)
            for l_p, l_n in label_remap.items():
                new_label[label == l_p] = l_n

            label = new_label
            props = measure.regionprops(label_image=label, intensity_image=self._img_signal)

        self._img_cc = label
        self._set_synapses(props)

    def _set_synapses(self, props: List[measure._regionprops.RegionProperties]):
        self._synapses = [Synapse3D(p.label, p) for p in props]

    def synapses(self) -> List[Synapse3D]:
        if self._synapses is None:
            self._get_cc()

        return self._synapses

    def label(self):
        if self._synapses is None:
            self._get_cc()

        return self._img_cc

    def number(self):
        if self._synapses is None:
            self._get_cc()

        return len(self._synapses)

    def prop(self, i, key):
        """
        Get the regionprops[key] of i-th synapse
        Args:
            i:

        Returns:

        """
        if self._synapses is None:
            self._get_cc()

        try:
            val = self._synapses[i].prop[key]
            return val
        except ValueError:
            return np.nan

    def sum_prop(self, key):
        if self._synapses is None:
            self._get_cc()

        sum = 0
        for syn in self._synapses:
            try:
                val = syn.prop[key]
                sum += val
            except ValueError:
                pass

        return sum

    def mean_prop(self, key):
        if self._synapses is None:
            self._get_cc()

        sum = 0
        cnt = 0
        for syn in self._synapses:
            try:
                val = syn.prop[key]
                sum += val
                cnt += 1
            except ValueError:
                pass

        if cnt == 0:
            return 0
        else:
            return sum / cnt

    def mean_intensity_of_all_positives(self):
        return np.mean(self._img_signal[self._img_label > 0])

    def spatial_dispersion_rms(self, scaling_zyx):
        """
        Measure root mean squared (rms) distance of centres of synapse
        :return: dispersion in pixels, dispersion in SI unit (m)
        """
        if self._dispersion_pixel is None:
            if self._synapses is None:
                self._get_cc()

            if self.number() == 0:
                self._dispersion_pixel = 0
                self._dispersion_si = 0
            else:
                centroids = np.array([syn.prop.centroid for syn in self._synapses])
                centre = np.average(centroids, axis=0)

                # in pixels unit
                self._dispersion_pixel = np.linalg.norm(centroids - centre) / (len(self._synapses) ** 0.5)
                # in SI unit. if scaling factor is not provided, return 0
                if None in scaling_zyx:
                    self._dispersion_si = 0
                else:
                    self._dispersion_si = np.linalg.norm(np.multiply(centroids - centre, scaling_zyx)) / (
                                len(self._synapses) ** 0.5)

        return self._dispersion_pixel, self._dispersion_si

    def synapse_distance(self, neurite_mask=None, ROI=None, dilation_round=1, optical_ratio=[1.02, 0.070, 0.070]):
        """
        Calculate synopses distribution along neurite spline
        :param neurite_mask: binary neurite mask
        :type neurite_mask: np.ndarray
        :param ROI: region of interest of the synapse cluster. must be in [[z_min, z_max], [y_min, y_max], [x_min, x_max]] order
        :type ROI: array like
        :param dilation_round: round of dilation on the neurite
        :type dilation_round: int
        :param optical_ratio: optical physical resolution
        :type optical_ratio: iterable
        :return: list of distance
        :rtype: list
        """
        # check if neurite mask is provided
        if neurite_mask is None:
            if self._img_mask is not None:
                neurite_mask = self._img_mask
            else:
                raise Exception('Neurite mask is not provided')

        # Dilate the neurite mask
        for _ in range(dilation_round):
            neurite_mask = dilation(neurite_mask)

        if ROI:
            # The additional ROI bounding box provided to separate from other clusters
            try:
                z, y, x = ROI[:]
                ROI_mask = neurite_mask[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
                ROI_syn = self._img_label[z[0]:z[1], y[0]:y[1], x[0]:x[1]]
            except ValueError:
                raise Exception('The coordinate must be in (z, y, x) order')
        else:
            # If bounding box not provided, use the min and max of synapses coordinates
            z, y, x = np.where(self._img_label != 0)
            if len(z) > 0:  # Get ROI on both synapses and neurite
                ROI_mask = neurite_mask[z.min():z.max(), y.min():y.max(), x.min():x.max()]
                ROI_syn = self._img_label[z.min():z.max(), y.min():y.max(), x.min():x.max()]
            else:  # I didn't include how to deal with the case where there is no synapse
                return 0

        # Get the pieces of neurite, because there might be other neurite in the bounding box. The irrelevant piece will be taken out.
        # There might be more than one piece because there are broken points. But as long as they have overlap with synapses, they are included.
        components = measure.label(ROI_mask, connectivity=3)
        props = measure.regionprops(components)
        for prop in props:
            b = prop.bbox
            if np.sum(ROI_syn[b[0]:b[3], b[1]:b[4], b[2]:b[5]] * prop.image) == 0:  # Check if there is overlap
                components[components == prop.label] = 0  # if not, take that piece off.

        # Get skeleton of the included pieces and the coordinates to fit a spline.
        skl_mask = skeletonize_3d(components)
        spl_points = np.argwhere(skl_mask).T

        # Fit and get the spline
        num_points = 400  # number of points to represent the spline.
        tck, u = interpolate.splprep(spl_points)  # Fitting
        spl_intp_pts = interpolate.splev(np.linspace(0, 1, num_points), tck)
        all_spline_pts = np.stack(tuple(spl_intp_pts), axis=-1)  # All the points on the spline.

        # Get the position of each synapse(consider the centroid) on the spline
        syn_ctrs = [(s.prop.centroid[0] - z.min(),
                     s.prop.centroid[1] - y.min(),
                     s.prop.centroid[2] - x.min()) for s in self.synapses()]

        import matplotlib.pyplot as plt
        cm = plt.cm.get_cmap('viridis')
        img_for_vis_skel = np.zeros(shape=(*ROI_syn.shape, 3), dtype=float)
        for i in range(num_points):
            coord = list(all_spline_pts[i, :].astype(int))
            coord[0] = min(coord[0], img_for_vis_skel.shape[0] - 1)
            coord[1] = min(coord[1], img_for_vis_skel.shape[1] - 1)
            coord[2] = min(coord[2], img_for_vis_skel.shape[2] - 1)
            img_for_vis_skel[tuple(coord)] = cm(i / num_points)[:3]
        img_for_vis_syn = np.zeros(shape=(*ROI_syn.shape, 3), dtype=float)

        if len(syn_ctrs) > 1:  # Only one synapse doesn't have meaning in distance
            syn_ctr_arry = np.vstack(syn_ctrs)
            for i in range(len(syn_ctrs)):
                coord = list(syn_ctr_arry[i, :].astype(int))
                coord[0] = min(coord[0], img_for_vis_skel.shape[0] - 1)
                coord[1] = min(coord[1], img_for_vis_skel.shape[1] - 1)
                coord[2] = min(coord[2], img_for_vis_skel.shape[2] - 1)
                img_for_vis_syn[tuple(coord)] = cm(i / len(syn_ctrs))[:3]
            # use the shortest distance to get the position.
            syn_on_neurite = distance.cdist(syn_ctr_arry, all_spline_pts, metric='cityblock', w=optical_ratio)
            # get the position index. Think the spline as [0, 1] with num_points in between. 0 is the origin and num_point is the end.
            syn_spline_idx = np.argmin(syn_on_neurite, axis=1)
            # sort the index to make sure the distance is with respect to the next nearby point
            syn_spline_idx = np.sort(syn_spline_idx)
            # Fine line integral along the spline accumulated on each representative point. using cityblock to include the optical resolution.
            dist_nearby = [distance.cityblock(u, v, w=optical_ratio) for u, v in
                           zip(all_spline_pts[:-1], all_spline_pts[1:])]
            dist_spline = []
            for i in range(len(syn_ctrs) - 1):
                # Acquire the pair-wise distance
                dist_spline.append(sum(dist_nearby[syn_spline_idx[i]:syn_spline_idx[i + 1]]))

            return float(np.mean(dist_spline)), img_for_vis_skel, img_for_vis_syn, skl_mask
        else:
            return 0, img_for_vis_skel, img_for_vis_syn, skl_mask

    def mean_inter_synapse_distance(self, scaling_zyx):
        """

        Args:
            scaling_zyx: physical pixel size in z, y, x direction

        Returns: MISD in pixel unit, MISD in SI unit

        """
        if self._misd_pixel is None:
            if self._synapses is None:
                self._get_cc()

            if self.number() <= 1:
                self._misd_pixel = 0
                self._misd_si = 0
            else:
                g_px = Graph()
                g_si = Graph()
                for i, s1 in enumerate(self._synapses):
                    g_px.add_node(s1.id)
                    g_si.add_node(s1.id)
                    for j, s2 in enumerate(self._synapses[:i]):
                        dist_px = np.linalg.norm(np.array(s1.prop.centroid) - np.array(s2.prop.centroid))
                        g_px.add_edge(s1.id, s2.id, dist_px, bidirectional=True)
                        if None not in scaling_zyx:
                            dist_si = np.linalg.norm(np.multiply(
                                np.array(s1.prop.centroid) - np.array(s2.prop.centroid), scaling_zyx))
                            g_si.add_edge(s1.id, s2.id, dist_si, bidirectional=True)

                tsp_px = g_px.solve_tsp()
                self._misd_pixel = tsp_px[0] / self.number()
                if None in scaling_zyx:
                    # if no scaling factor is provided, set the value 0
                    self._misd_si = 0
                else:
                    tsp_si = g_si.solve_tsp()
                    self._misd_si = tsp_si[0] / self.number()

        return self._misd_pixel, self._misd_si

    def get_synapse(self, sid: int) -> Synapse3D:
        for s in self._synapses:
            if s.id == sid:
                return s

    def get_synapses_inside(self, roi) -> List[Synapse3D]:
        return [s for s in self._synapses if s.is_in(roi)]

    def get_synapses_outside(self, roi) -> List[Synapse3D]:
        return [s for s in self._synapses if not s.is_in(roi)]

    def get_small_synapses(self, min_area) -> List[Synapse3D]:
        return [s for s in self._synapses if s.prop.area < min_area]

    def deep_phynotyping(self):
        """
        Get a feature array for deep-phenotyping as described in https://doi.org/10.1038/ncomms12990
        Returns:

        """

        def safe_div(x, y):
            if y == 0: return 0
            return x / y

        if self.number() == 0:
            return [None]

        f = []
        # 1) Number of puncta larger than one pixel.
        x_area = [s.prop.area for s in self._synapses]
        x_area_larger_1 = list(filter(lambda a: a > 1, x_area))
        f.append(len(x_area_larger_1))
        # 2) Average size of all the puncta.
        f.append(np.mean(x_area))
        # 3) Average size of the puncta larger than one pixel.
        f.append(np.mean(x_area_larger_1))
        # 4) Second central moment of the size of the puncta larger than one pixel
        #    / (Mean of the size of puncta size larger than one pixel)^2
        f.append(safe_div(np.var(x_area_larger_1), np.mean(x_area_larger_1) ** 2))
        # 5) Number of puncta larger than 8 pixels.
        x_area_larger_8 = list(filter(lambda a: a > 8, x_area))
        f.append(len(x_area_larger_8))
        # 6) Percentage of puncta smaller than 6 pixels.
        x_area_smaller_6 = list(filter(lambda a: a < 6, x_area))
        f.append(safe_div(len(x_area_smaller_6), len(x_area)))
        # 7) Number of puncta larger than 1 pixel and smaller than 8 pixels
        #    / Number of puncta larger than 1 pixel
        x_area_larger_1_smaller_8 = list(filter(lambda a: 1 < a < 8, x_area))
        f.append(safe_div(len(x_area_larger_1_smaller_8), len(x_area_larger_1)))
        # 8) Standard deviation of the puncta size / Mean of the puncta size
        f.append(safe_div(np.std(x_area), np.mean(x_area)))
        # 9) First quartile of puncta size
        f.append(np.quantile(x_area, 0.25))
        # 10) Median of puncta size
        f.append(np.median(x_area))
        # 11) Third quartile of puncta size
        f.append(np.quantile(x_area, 0.75))
        # 12) 90th percentile of puncta size
        f.append(np.percentile(x_area, 90))
        # 13) Maximum puncta size
        f.append(np.max(x_area))
        # 14) Mean size of the smallest half of the puncta
        x_area_sorted = sorted(x_area)
        x_area_small_half = x_area_sorted[:int(len(x_area_sorted) / 2)]
        f.append(np.mean(x_area_small_half))
        # 15) Standard deviation of the size of the smallest half of the puncta
        #     / Mean of the size of the smallest half of the puncta
        f.append(np.std(x_area_small_half) / np.mean(x_area_small_half))
        # 16) Mean size of the largest half of the puncta
        x_area_large_half = x_area_sorted[int(len(x_area_sorted) / 2):]
        f.append(np.mean(x_area_large_half))
        # 17) Standard deviation of the size of the largest half of the puncta
        #     / Mean of the size of the largest half of the puncta
        f.append(safe_div(np.std(x_area_large_half), np.mean(x_area_large_half)))
        # 18) 90th percentile of the puncta size / First quartile of the puncta size
        f.append(safe_div(np.percentile(x_area, 90), np.quantile(x_area, 0.25)))
        # 19) Mean size of the largest half of the puncta / Mean size of the smallest half of the puncta
        f.append(safe_div(np.mean(x_area_large_half), np.mean(x_area_small_half)))
        # 20) Mean of average puncta intensity (average puncta intensity refers to the mean pixel intensity value
        #     / for each puncta)
        x_mean_intensity = [s.prop.mean_intensity for s in self._synapses]
        f.append(np.mean(x_mean_intensity))
        # 21) Standard deviation of average puncta intensity / Mean of average puncta intensity
        f.append(safe_div(np.std(x_mean_intensity), np.mean(x_mean_intensity)))
        # 22) Mean of integrated puncta intensity (integrated intensity refers to the sum of all pixel intensity
        #     values for each puncta)
        x_sum_intensity = [np.sum(s.prop.intensity_image[s.prop.image]) for s in self._synapses]
        f.append(np.mean(x_sum_intensity))
        # 23) Standard deviation of integrated puncta intensity / Mean of integrated puncta intensity
        f.append(safe_div(np.std(x_sum_intensity), np.mean(x_sum_intensity)))
        # 24) Second central moment of integrated puncta intensity / (Mean of integrated puncta intensity)^2
        f.append(safe_div(np.var(x_sum_intensity), np.mean(x_sum_intensity) ** 2))
        # 25) Minimum of integrated puncta intensity
        f.append(np.min(x_sum_intensity))
        # 26) First quartile of integrated puncta intensity
        f.append(np.quantile(x_sum_intensity, 0.25))
        # 27) Median quartile of integrated puncta intensity
        f.append(np.median(x_sum_intensity))
        # 28) Third quartile of integrated puncta intensity
        f.append(np.quantile(x_sum_intensity, 0.75))
        # 29) 90th percentile of integrated puncta intensity
        f.append(np.percentile(x_sum_intensity, 90))
        # 30) Maximum of integrated intensity
        f.append(np.max(x_sum_intensity))

        ###### TODO: features relative to puncta location
        # 31) Mean of integrated intensity of the third most anterior puncta
        # 32) Mean of integrated intensity of the third central puncta
        # 33) Mean of integrated intensity of the third most posterior puncta
        # 34) Mean of integrated intensity of the third most posterior puncta
        #     / Mean of integrated intensity of the third most anterior puncta
        # 35) Mean of integrated intensity of the third most posterior puncta
        #     / Mean of integrated intensity of the third central puncta
        # 36) Mean of integrated intensity of the third central puncta
        #     / Mean of integrated intensity of the third most anterior puncta
        # 37) Total distance of synaptic domain (computed by adding distance of individual interpunctal segments
        #     larger than 3 pixels)
        # 38) Mean interpunctal distance (ignoring segments smaller than 3 pixels)
        # 39) Standard deviation of interpunctal distance (ignoring segments smaller than 3 pixels)
        #     / Mean of interpunctal distance (ignoring segments smaller than 3 pixels)
        # 40) Mean interpunctal distance of half most anterior puncta (ignoring segments smaller than 3 pixels)
        # 41) Standard deviation of interpunctal distance of half most anterior puncta (ignoring segments smaller than 3 pixels)
        #     / Mean of interpunctal distance of half most anterior puncta (ignoring segments smaller than 3 pixels)
        # 42) Mean interpunctal distance of half most posterior puncta (ignoring segments smaller than 3 pixels)
        # 43) Standard deviation of interpunctal distance of half most posterior puncta (ignoring segments smaller than 3 pixels)
        #     / Mean of interpunctal distance of half most posterior puncta (ignoring segments smaller than 3 pixels)
        # 44) 90th percentile of interpunctal distance (including all segments)
        # 45) Density, computed by: Number of puncta larger than 1 pixel
        #                           / Total distance of synaptic domain (ignoring segments smaller than 3 pixels)
        # 49) 95th percentile of interpunctal distance (including all segments)
        # 50) Third quartile of interpunctal distance (including all segments)
        # 51) Location of most posterior puncta, relative to the location of the gut end (in pixels). Negative values
        #     indicate the most posterior puncta is farther back than the gut end.
        # 52) Mean of average interpunctal intensity (interpunctal intensity refers to the mean pixel intensity value
        #     for each interpunctal segment)
        # 53) Standard deviation of interpunctal intensity / Mean of interpunctal intensity
        ###############

        # 46->31) Percentage of puncta smaller than 5 pixels
        x_area_smaller_5 = list(filter(lambda a: a < 5, x_area))
        f.append(safe_div(len(x_area_smaller_5), len(x_area)))
        # 47->32) Percentage of puncta smaller than 10 pixels and larger or equal than 5 pixels
        x_area_smaller_10_laq_5 = list(filter(lambda a: 5 <= a < 10, x_area))
        f.append(safe_div(len(x_area_smaller_10_laq_5), len(x_area)))
        # 48->33) Percentage of puncta smaller than 15 pixels and larger or equal than 10 pixels
        x_area_smaller_15_laq_10 = list(filter(lambda a: 10 <= a < 15, x_area))
        f.append(safe_div(len(x_area_smaller_15_laq_10), len(x_area)))
        # 54->34) 10th percentile of puncta size
        f.append(np.percentile(x_area, 10))
        # 55->35) 10th percentile of integrated intensity
        f.append(np.percentile(x_sum_intensity, 10))

        # 56->36) Fraction of puncta pixels with intensity larger or equal than 500 and smaller than 1000
        x_intensity_all = np.concatenate([s.prop.intensity_image[s.prop.image] for s in self._synapses])
        f.append(safe_div(len(list(filter(lambda i: 500 <= i < 1000, x_intensity_all))), len(x_intensity_all)))
        # 57->37) Fraction of puncta pixels with intensity larger or equal than 1000 and smaller than 1500
        f.append(safe_div(len(list(filter(lambda i: 1000 <= i < 1500, x_intensity_all))), len(x_intensity_all)))
        # 58->38) Fraction of puncta pixels with intensity larger or equal than 1500 and smaller than 2000
        f.append(safe_div(len(list(filter(lambda i: 1500 <= i < 2000, x_intensity_all))), len(x_intensity_all)))
        # 59->39) Fraction of puncta pixels with intensity larger or equal than 2000 and smaller than 2500
        f.append(safe_div(len(list(filter(lambda i: 2000 <= i < 2500, x_intensity_all))), len(x_intensity_all)))
        # 60->40) Fraction of puncta pixels with intensity larger or equal than 2500 and smaller than 3000
        f.append(safe_div(len(list(filter(lambda i: 2500 <= i < 3000, x_intensity_all))), len(x_intensity_all)))
        # 61->41) Fraction of puncta pixels with intensity larger or equal than 3000 and smaller than 3500
        f.append(safe_div(len(list(filter(lambda i: 3000 <= i < 3500, x_intensity_all))), len(x_intensity_all)))
        # 62->42) Fraction of puncta pixels with intensity larger or equal than 3500 and smaller than 4000
        f.append(safe_div(len(list(filter(lambda i: 3500 <= i < 4000, x_intensity_all))), len(x_intensity_all)))
        # 63->43) Range of puncta pixel intensity values (computed by subtracting the dimmest pixel value from the
        #     brightest pixel value)
        f.append(np.max(x_intensity_all) - np.min(x_intensity_all))
        # 64->44) Standard deviation of puncta pixel values / Mean of puncta pixel values
        f.append(safe_div(np.std(x_intensity_all), np.mean(x_intensity_all)))
        # 65->45) First quartile of standardized pixel values (standardized pixel values refers to the intensity pixel
        #     values where the value of the dimmest pixel has been subtracted)
        x_intensity_all_sd = x_intensity_all - np.min(x_intensity_all)
        f.append(np.quantile(x_intensity_all_sd, 0.25))
        # 66->46) Median of standardized pixel values
        f.append(np.median(x_intensity_all_sd))
        # 67->47) Third quartile of standardized pixel values
        f.append(np.quantile(x_intensity_all_sd, 0.75))
        # 68->48) 90th percentile of standardized pixel values
        f.append(np.percentile(x_intensity_all_sd, 90))
        # 69->49) Fraction of pixels with standardized intensity values smaller than 0.1(Range of pixel values)
        range_intensity = np.max(x_intensity_all_sd) - np.min(x_intensity_all_sd)
        f.append(safe_div(np.sum(x_intensity_all_sd < 0.1 * range_intensity), len(x_intensity_all_sd)))
        # 70->50) Fraction of pixels with standardized intensity values smaller than 0.25(Range of pixel values) and
        #     larger than 0.1(Range of pixel values)
        f.append(safe_div(np.sum(np.logical_and(x_intensity_all_sd > 0.1 * range_intensity,
                                                x_intensity_all_sd < 0.25 * range_intensity)), len(x_intensity_all_sd)))
        # 71->51) Fraction of pixels with standardized intensity values smaller than 0.5(Range of pixel values) and
        #     larger than 0.25(Range of pixel values)
        f.append(safe_div(np.sum(np.logical_and(x_intensity_all_sd > 0.25 * range_intensity,
                                                x_intensity_all_sd < 0.5 * range_intensity)), len(x_intensity_all_sd)))
        # 72->52) Fraction of pixels with standardized intensity values smaller than 0.75(Range of pixel values) and
        #     larger than 0.5(Range of pixel values)
        f.append(safe_div(np.sum(np.logical_and(x_intensity_all_sd > 0.5 * range_intensity,
                                                x_intensity_all_sd < 0.75 * range_intensity)), len(x_intensity_all_sd)))
        # 73->53) Fraction of pixels with standardized intensity values smaller than 0.9(Range of pixel values) and
        #     larger than 0.75(Range of pixel values)
        f.append(safe_div(np.sum(np.logical_and(x_intensity_all_sd > 0.75 * range_intensity,
                                                x_intensity_all_sd < 0.9 * range_intensity)), len(x_intensity_all_sd)))
        # 74->54) Fraction of pixels with standardized intensity values smaller than 0.95(Range of pixel values) and
        #     larger than 0.9(Range of pixel values)
        f.append(safe_div(np.sum(np.logical_and(x_intensity_all_sd > 0.9 * range_intensity,
                                                x_intensity_all_sd < 0.95 * range_intensity)), len(x_intensity_all_sd)))
        # 75->55) Number of puncta larger than 0.25(Range of puncta size); where range of puncta size is computed
        #     by subtracting the smallest puncta size from the largest puncta size
        range_area = np.max(x_area) - np.min(x_area)
        f.append(len(list(filter(lambda a: a > 0.25 * range_area, x_area))))
        # 76->56) Total integrated intensity: sum of all puncta pixel intensity values
        f.append(np.sum(x_sum_intensity))

        return f
