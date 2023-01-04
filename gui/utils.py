import threading
from queue import Queue
from typing import Dict

import numpy as np
from skimage import img_as_ubyte
from skimage.io import imread

from segmentutil.synapse_quantification import SynapseQT3D
from segmentutil.utils import enhance_contrast_histogram, enhance_contrast_value
import platform


def is_on_OSX():
    if str.lower(platform.system()) == 'darwin':
        return True
    else:
        return False


def is_on_Linux():
    if str.lower(platform.system()) == 'linux':
        return True
    else:
        return False


class BusyManager:
    def __init__(self, widget):
        self.toplevel = widget.winfo_toplevel()
        self.widgets = {}

    def busy(self, widget=None):

        # attach busy cursor to toplevel, plus all windows
        # that define their own cursor.

        if widget is None:
            w = self.toplevel  # myself
        else:
            w = widget

        if not str(w) in self.widgets:
            try:
                # attach cursor to this widget
                cursor = w.cget("cursor")
                if cursor != "watch":
                    self.widgets[str(w)] = (w, cursor)
                    w.config(cursor="watch")
            except:
                pass

        for w in w.children.values():
            self.busy(w)

    def notbusy(self):
        # restore cursors
        for w, cursor in self.widgets.values():
            try:
                w.config(cursor=cursor)
            except:
                pass
        self.widgets = {}


class SynapseImage(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.array = imread(filepath)
        self.synapse_qt = SynapseQT3D(self.array[..., 2], self.array[..., 1])
        self.synapse_qt._get_cc()

        self.range_r = [0, np.iinfo(self.array.dtype).max]
        self.range_g = [0, np.iinfo(self.array.dtype).max]
        self.range_b = [0, np.iinfo(self.array.dtype).max]
        self.array_contrast = np.zeros(shape=self.array.shape, dtype=np.uint8)

    def autocontrast(self):
        self.array_contrast[..., 0], v_min, v_max = enhance_contrast_histogram(self.array[..., 0], 0, 0.9999, np.uint8)
        self.range_r = [v_min, v_max]
        self.array_contrast[..., 1], v_min, v_max = enhance_contrast_histogram(self.array[..., 1], 0, 0.9999, np.uint8)
        self.range_g = [v_min, v_max]
        self.array_contrast[..., 2] = img_as_ubyte(self.array[:, :, :, 2])
        self.range_b = [0, np.iinfo(self.array.dtype).max]

    def manualcontrast(self, min_r, max_r, min_g, max_g, min_b, max_b):
        self.array_contrast[..., 0], v_min, v_max = enhance_contrast_value(self.array[..., 0], min_r, max_r, np.uint8)
        self.range_r = [v_min, v_max]
        self.array_contrast[..., 1], v_min, v_max = enhance_contrast_value(self.array[..., 1], min_g, max_g, np.uint8)
        self.range_g = [v_min, v_max]
        self.array_contrast[..., 2], v_min, v_max = enhance_contrast_value(self.array[..., 2], min_b, max_b, np.uint8)
        self.range_b = [v_min, v_max]


def async_loading(load_queue: Queue, result_queue: Queue):
    while not load_queue.empty():
        file = load_queue.get()
        image = SynapseImage(file)
        result_queue.put(image)


class AsyncImageLoader(object):
    """
    Load images in separate thread in order to reduce time waiting
    """

    def __init__(self, filelist):
        self._load_queue = Queue()
        self._result_queue = Queue()
        self._filelist = filelist
        self._nextfile = dict([(filelist[i], filelist[i + 1]) for i in range(0, len(filelist) - 1)])
        self._loaded_images: Dict[str, SynapseImage] = {}

        self._aync_loader = threading.Thread(target=async_loading, args=(self._load_queue, self._result_queue))
        self._aync_loader.start()

    # for now, sync version only
    def load_image(self, file, is_fix_contrast, rgb_range):
        self._aync_loader.join()
        while not self._result_queue.empty():
            image: SynapseImage = self._result_queue.get()
            self._loaded_images[image.filepath] = image

        if file in self._loaded_images:
            if file in self._nextfile and self._nextfile[file] not in self._loaded_images:
                self._load_queue.put(self._nextfile[file])
            image = self._loaded_images[file]
            del self._loaded_images[file]
            self._aync_loader = threading.Thread(target=async_loading, args=(self._load_queue, self._result_queue))
            self._aync_loader.start()
            # adjust contrast
            if is_fix_contrast:
                image.manualcontrast(*rgb_range)
            else:
                image.autocontrast()

            return image
        else:
            self._load_queue.put(file)
            self._aync_loader = threading.Thread(target=async_loading, args=(self._load_queue, self._result_queue))
            self._aync_loader.start()
            return self.load_image(file, is_fix_contrast, rgb_range)
