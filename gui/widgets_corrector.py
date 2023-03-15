import functools
import os
import threading
import tkinter as tk
from enum import Enum
from tkinter import filedialog
from typing import List, Dict

import numpy as np
from PIL import Image, ImageTk
from skimage.io import imsave

from gui.utils import SynapseImage, is_on_OSX, is_on_Linux
from segmentutil.synapse_quantification import Synapse3D

_ZOOM_SCALES = [0.125, 0.167, 0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12, 16, 24, 32]
_INIT_ZOOM_IDX = 6


class Virtual3DCanvas(tk.Canvas):
    class State(Enum):
        MOVE = 0
        DRAW = 1
        MAX = 2

    def __init__(self, parent, width, height, logger):
        super().__init__(parent, width=width, height=height, bg='#111177')
        # for define variables
        self.logger = logger
        self.cur_array = None
        self.cur_label = None
        self.zm_idx = None
        self.cur_z = None
        self.image = None
        self.cur_slice = None
        self.cur_slice_zmd = None
        self.cur_photo = None

        # the rectangle of currently cropped region from entire image (after scaling)
        self.cur_crop_box = None

        # display
        self.cv_width = width
        self.cv_height = height
        self.crop_margin = 100
        self.cv_center = (width / 2, height / 2)
        self.clear_canvas()

        # state
        self.state = self.State(0)

        # synapses
        self.b_show_bbox = True
        self.synapses: List[Synapse3D] = []

        # rectangle drawing
        self.rect_start = None
        self.rect_drawn = None

        # events
        self.mouseprevious = (0, 0)
        self.bind("<Button-1>", self.on_leftclick)
        # for odd historical reasons, the right button is button 2 on the Mac, but 3 on unix and windows.
        if is_on_OSX():
            self.bind("<Button-2>", self.on_rightclick)
            self.bind("<Shift-Button-2>", self.on_shift_rightclick)
        else:
            self.bind("<Button-3>", self.on_rightclick)
            self.bind("<Shift-Button-3>", self.on_shift_rightclick)
        self.bind("<B1-Motion>", self.on_leftdrag)
        self.pack()

        # popup
        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu_shift = tk.Menu(self, tearoff=0)

    def clear_canvas(self):
        self.cur_array = None
        self.cur_label = None
        self.cur_slice = None
        self.cur_slice_zmd = None
        self.cur_photo = None
        self.zm_idx = _INIT_ZOOM_IDX
        self.cur_z = 0
        self.delete(tk.ALL)
        self.image = self.create_image(0, 0, anchor=tk.NW)

        # the rectangle of currently cropped region from entire image (after scaling)
        self.cur_crop_box = None

        # state
        self.state = self.State(0)

        # synapses
        self.b_show_bbox = True
        self.synapses: List[Synapse3D] = []

        # rectangle drawing
        self.rect_start = None
        self.rect_drawn = None

    def move_all_items(self, x, y):
        for item in self.find(tk.ALL):
            cur_coords = self.coords(item)
            if len(cur_coords) == 2:
                self.coords(item, cur_coords[0] + x, cur_coords[1] + y)
            elif len(cur_coords) == 4:
                self.coords(item, cur_coords[0] + x, cur_coords[1] + y, cur_coords[2] + x, cur_coords[3] + y)
            else:
                self.logger.warn("undefined object: %d" % item)

    def load_photo(self, image: SynapseImage, **kwargs):
        """
        Load a worm image on the canvas
        Args:
            image: SynapseImage object

        Returns:

        """
        if 'init_zm_idx' in kwargs and kwargs['init_zm_idx'] is not None:
            self.zm_idx = kwargs['init_zm_idx']
        else:
            self.zm_idx = _INIT_ZOOM_IDX
        self.synapses = image.synapse_qt.synapses()
        self.sid_p = [s.id for s in self.synapses]
        self.cur_array = image.array_contrast
        self.cur_label = image.synapse_qt.label()

        self.cur_z = 0
        self.cur_slice = Image.fromarray(self.cur_array[self.cur_z, :, :])

        # load and intialize
        self.cur_slice_zmd = self.get_zoomed_image(self.cur_slice)
        self.cur_photo = ImageTk.PhotoImage(master=self, image=self.cur_slice_zmd)
        self.cur_crop_box = (0, 0, self.cur_photo.width(), self.cur_photo.height())
        self.itemconfig(self.image, image=self.cur_photo)
        x = self.cv_center[0] - self.cur_photo.width() / 2
        y = self.cv_center[1] - self.cur_photo.height() / 2
        self.coords(self.image, x, y)
        self.draw_synapse_bbox(self.cur_z)

        # crop
        self.crop(self.cur_slice_zmd)

    def replace_array(self, new_array):
        self.cur_array = new_array
        self.cur_slice = Image.fromarray(self.cur_array[self.cur_z, :, :])
        self.cur_slice_zmd = self.get_zoomed_image(self.cur_slice)
        self.crop(self.cur_slice_zmd)
        self.draw_synapse_bbox(self.cur_z)

    def photoscale(self, is_up):
        if self.cur_array is None:
            return

        # first, set newly zoomed image without cropping on canvas
        zm_idx_prev = self.zm_idx
        cur_x, cur_y = self.coords(self.image)
        if is_up:
            self.zm_idx = min(len(_ZOOM_SCALES) - 1, self.zm_idx + 1)
        else:
            self.zm_idx = max(0, self.zm_idx - 1)

        self.cur_slice_zmd = self.get_zoomed_image(self.cur_slice)
        self.cur_photo = ImageTk.PhotoImage(master=self, image=self.cur_slice_zmd)
        new_coord_x, new_coord_y = self.coord_change_magnification(zm_idx_prev=zm_idx_prev,
                                                                   x=cur_x - self.cur_crop_box[0],
                                                                   y=cur_y - self.cur_crop_box[1])
        # move image
        self.coords(self.image, new_coord_x, new_coord_y)
        self.cur_crop_box = (0, 0, self.cur_photo.width(), self.cur_photo.height())

        # then, crop appropriately
        self.crop(self.cur_slice_zmd)

        # resize and move the roi rectangle if exists
        if self.rect_drawn is not None:
            coord_roi = self.coords(self.rect_drawn)
            self.coords(self.rect_drawn,
                        *self.coord_change_magnification(zm_idx_prev, coord_roi[0], coord_roi[1]),
                        *self.coord_change_magnification(zm_idx_prev, coord_roi[2], coord_roi[3]))

        # redraw bbox
        self.draw_synapse_bbox(self.cur_z)

    def goto(self, center):
        if self.cur_array is not None:
            # if center z is different from current showing z, then change slice
            if int(center[0]) != self.cur_z:
                self.cur_z = int(center[0])
                self.cur_slice = Image.fromarray(self.cur_array[self.cur_z, :, :])
                self.cur_slice_zmd = self.get_zoomed_image(self.cur_slice)

            # move focus to center of synapse
            target_x = self.cv_center[0] - int(center[2] * _ZOOM_SCALES[self.zm_idx]) + self.cur_crop_box[0]
            target_y = self.cv_center[1] - int(center[1] * _ZOOM_SCALES[self.zm_idx]) + self.cur_crop_box[1]
            cur_x, cur_y = self.coords(self.image)

            self.move_all_items(target_x - cur_x, target_y - cur_y)

            # crop appropriately
            self.crop(self.cur_slice_zmd)

            # redraw bbox
            self.draw_synapse_bbox(self.cur_z)

    # region getters
    def get_zoomed_image(self, image):
        if self.zm_idx == _INIT_ZOOM_IDX:
            zoomed = self.cur_slice
        else:
            zoomed = self.cur_slice.resize((int(image.size[0] * _ZOOM_SCALES[self.zm_idx]),
                                            int(image.size[1] * _ZOOM_SCALES[self.zm_idx])),
                                           Image.NEAREST)
        return zoomed

    def get_image_chunk(self, center, halfsize):
        if self.cur_array is not None:
            img_chunk_arr = np.zeros(shape=(self.cur_array.shape[0], 2 * halfsize + 1, 2 * halfsize + 1, 3),
                                     dtype=np.uint8)
            img_chunk_lbl = np.zeros(shape=(self.cur_array.shape[0], 2 * halfsize + 1, 2 * halfsize + 1),
                                     dtype=int)
            y1 = max(0, int(center[1]) - halfsize)
            y1_cnk = y1 - int(center[1]) + halfsize
            y2 = min(self.cur_array.shape[1], int(center[1]) + halfsize + 1)
            y2_cnk = y2 - int(center[1]) + halfsize
            x1 = max(0, int(center[2]) - halfsize)
            x1_cnk = x1 - int(center[2]) + halfsize
            x2 = min(self.cur_array.shape[2], int(center[2]) + halfsize + 1)
            x2_cnk = x2 - int(center[2]) + halfsize
            img_chunk_arr[:, y1_cnk:y2_cnk, x1_cnk:x2_cnk, :] = self.cur_array[:, y1:y2, x1:x2, :]
            img_chunk_lbl[:, y1_cnk:y2_cnk, x1_cnk:x2_cnk] = self.cur_label[:, y1:y2, x1:x2]
            return img_chunk_arr, img_chunk_lbl
        else:
            return None

    def get_cropping_region(self):
        uncropped_x = self.coords(self.image)[0] - self.cur_crop_box[0]
        uncropped_y = self.coords(self.image)[1] - self.cur_crop_box[1]
        left = max(-uncropped_x - self.crop_margin, 0)
        top = max(-uncropped_y - self.crop_margin, 0)
        right = min(self.cv_width + self.crop_margin - uncropped_x, self.cur_slice_zmd.width)
        bottom = min(self.cv_height + self.crop_margin - uncropped_y, self.cur_slice_zmd.height)
        return left, top, right, bottom

    # endregion

    # region coordinate change
    def to_canvas_coord(self, arridx_x, arridx_y):
        coord_img = self.coords(self.image)
        x = arridx_x * _ZOOM_SCALES[self.zm_idx] - self.cur_crop_box[0] + coord_img[0]
        y = arridx_y * _ZOOM_SCALES[self.zm_idx] - self.cur_crop_box[1] + coord_img[1]
        return x, y

    def to_array_coord(self, canvas_x, canvas_y):
        coord_img = self.coords(self.image)
        scaled_x = self.cur_crop_box[0] - coord_img[0] + canvas_x
        scaled_y = self.cur_crop_box[1] - coord_img[1] + canvas_y
        x = int(scaled_x / _ZOOM_SCALES[self.zm_idx])
        y = int(scaled_y / _ZOOM_SCALES[self.zm_idx])
        return x, y

    def coord_change_magnification(self, zm_idx_prev, x, y):
        new_x = self.cv_center[0] + (x - self.cv_center[0]) / _ZOOM_SCALES[zm_idx_prev] * _ZOOM_SCALES[self.zm_idx]
        new_y = self.cv_center[1] + (y - self.cv_center[1]) / _ZOOM_SCALES[zm_idx_prev] * _ZOOM_SCALES[self.zm_idx]
        return new_x, new_y

    # endregion

    # region crop
    def check_need_recrop(self):
        x = self.coords(self.image)[0]
        y = self.coords(self.image)[1]
        if x > 0 and self.cur_crop_box[0] > 0:
            return True
        if x + self.cur_photo.width() < self.cv_width and self.cur_crop_box[2] < self.cur_slice_zmd.width:
            return True
        if y > 0 and self.cur_crop_box[1] > 0:
            return True
        if y + self.cur_photo.height() < self.cv_height and self.cur_crop_box[3] < self.cur_slice_zmd.height:
            return True

    def crop(self, image: Image):
        prev_coord = self.coords(self.image)
        prev_crop_r = self.cur_crop_box

        new_crop_r = self.get_cropping_region()
        cropped = image.crop(new_crop_r)
        self.cur_photo = ImageTk.PhotoImage(master=self, image=cropped)

        new_x = prev_coord[0] + new_crop_r[0] - prev_crop_r[0]
        new_y = prev_coord[1] + new_crop_r[1] - prev_crop_r[1]
        self.itemconfig(self.image, image=self.cur_photo)
        self.coords(self.image, new_x, new_y)
        self.cur_crop_box = new_crop_r

    # endregion

    # region on mouse action
    def on_leftclick(self, event):
        """
        Called when the mouse left button is pressed
        Args:
            event:

        Returns:

        """
        # hide popup if it's on
        self.popup_menu.unpost()
        self.popup_menu_shift.unpost()

        if self.cur_array is not None:
            canvas: tk.Canvas = event.widget  # Get handle to canvas
            # Convert screen coordinates to canvas coordinates
            xc = canvas.canvasx(event.x)
            yc = canvas.canvasx(event.y)

            if self.state == self.State.MOVE:
                # with space: move image
                self.mouseprevious = (xc, yc)
            elif self.state == self.State.DRAW:
                # without space: draw rectangle
                self.rect_start = (xc, yc)
                if self.rect_drawn:
                    canvas.delete(self.rect_drawn)
                self.rect_drawn = None

    def on_leftdrag(self, event):
        """
        Called when the mouse is left-dragged
        Args:
            event:

        Returns:

        """
        if self.cur_array is not None:
            canvas: tk.Canvas = event.widget
            xc = canvas.canvasx(event.x)
            yc = canvas.canvasx(event.y)
            if self.state == self.State.MOVE:
                # with space: move image
                self.move_all_items(xc - self.mouseprevious[0], yc - self.mouseprevious[1])
                self.mouseprevious = (xc, yc)
                if self.check_need_recrop():
                    self.crop(self.cur_slice_zmd)
            elif self.state == self.State.DRAW:
                # without space: draw rectangle
                if self.rect_drawn is not None:
                    canvas.delete(self.rect_drawn)
                self.rect_drawn = canvas.create_rectangle(self.rect_start[0], self.rect_start[1], event.x, event.y,
                                                          outline="#ffff00", fill="")

    def on_rightclick(self, event):
        if self.rect_drawn is not None:
            self.popup_menu_shift.unpost()
            self.popup_menu.post(event.x_root, event.y_root)

    def on_shift_rightclick(self, event):
        if self.rect_drawn is not None:
            self.popup_menu.unpost()
            self.popup_menu_shift.post(event.x_root, event.y_root)

    def on_mousewheel(self, scroll):
        if self.cur_array is not None:
            self.cur_z += scroll
            if self.cur_z < 0:
                self.cur_z = 0
            elif self.cur_z >= self.cur_array.shape[0]:
                self.cur_z = self.cur_array.shape[0] - 1
            else:
                self.cur_slice = Image.fromarray(self.cur_array[self.cur_z, :, :])
                self.cur_slice_zmd = self.get_zoomed_image(self.cur_slice)
                cropped = self.cur_slice_zmd.crop(self.cur_crop_box)
                self.cur_photo = ImageTk.PhotoImage(master=self, image=cropped)
                self.itemconfig(self.image, image=self.cur_photo)

                # redraw bbox
                self.draw_synapse_bbox(self.cur_z)

    # endregion

    # region state
    def roll_state(self):
        self.state = self.State((self.state.value + 1) % self.State.MAX.value)

    def get_state_str(self):
        return self.state.name

    def get_zoom_percent(self):
        return int(_ZOOM_SCALES[self.zm_idx] * 100)

    # endregion

    # region block reject/revert
    def get_roi(self):
        if self.rect_drawn is None:
            return None
        else:
            coord_roi = self.coords(self.rect_drawn)
            x1, y1 = self.to_array_coord(coord_roi[0], coord_roi[1])
            x2, y2 = self.to_array_coord(coord_roi[2], coord_roi[3])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, self.cur_array.shape[2]), min(y2, self.cur_array.shape[1])
            return y1, y2, x1, x2

    def get_cur_z(self):
        return self.cur_z

    # endregion

    # region bounding box
    def toggle_bbox(self):
        """
        Toggle between showing and hiding the bounding box of synapses
        Returns:

        """
        self.b_show_bbox = not self.b_show_bbox
        self.draw_synapse_bbox(self.cur_z)

    def refresh_synapse_state(self):
        self.draw_synapse_bbox(self.cur_z)

    def draw_synapse_bbox(self, z):
        """
        Draw bounding boxes on top of the image.
        Args:

        Returns: array with bbox

        """
        self.delete("bbox")
        if self.b_show_bbox:
            for syn in self.synapses:
                if not (syn.prop.bbox[0] <= z < syn.prop.bbox[3]):
                    continue
                if syn.state == syn.State.SELECTED:
                    color = "#ffff00"
                elif syn.state == syn.State.POSITIVE:
                    color = "#ffffff"
                else:
                    color = "#ff0000"
                x1, y1 = self.to_canvas_coord(syn.prop.bbox[2], syn.prop.bbox[1])
                x2, y2 = self.to_canvas_coord(syn.prop.bbox[5], syn.prop.bbox[4])
                self.create_rectangle(x1, y1, x2, y2, outline=color, fill="", tags=["bbox"])
                self.create_text(x1, y1, text=str(syn.id), fill=color, anchor=tk.NE, tags=["bbox"])

    # endregion


class SynapseViewerWidget(tk.Frame):
    def __init__(self, parent, width, height, logger):
        super().__init__(parent, width=width, height=height, bd=1)
        self.logger = logger
        self.pack()
        self.grid_propagate(0)

        # objects
        self.arrimage = None
        self.arrlabel = None
        self.current_img_path = None
        self.synapse_chunk_id = None
        self.synapse_chunk_rgb = None
        self.synapse_chunk_r = None
        self.synapse_chunk_g = None
        self.synapse_chunk_b = None
        self.map_synapse: Dict[int, Synapse3D] = None
        self.cv_zoom_rgb_img = None
        self.cv_zoom_r_img = None
        self.cv_zoom_g_img = None
        self.cv_zoom_b_img = None
        self.photo_rgb = None
        self.photo_r = None
        self.photo_g = None
        self.photo_b = None

        # label
        labelheight = 20
        self.labelframe = tk.Frame(self, width=width - 20, height=labelheight)
        self.labelframe.grid(row=0, columnspan=4)
        self.labeltext_info = tk.StringVar(master=self)
        self.label_info = tk.Label(self.labelframe, textvariable=self.labeltext_info, bg='white', anchor=tk.W,
                                   justify=tk.LEFT)
        self.label_info.pack(anchor=tk.W, fill=tk.Y, side=tk.LEFT)
        self.labeltext_state = tk.StringVar(master=self)
        self.label_state = tk.Label(self.labelframe, textvariable=self.labeltext_state, bg='blue', anchor=tk.W,
                                    font=('TkDefaultFont', 10, 'bold'), fg='yellow',
                                    justify=tk.LEFT)
        self.label_state.pack(anchor=tk.W, fill=tk.Y, side=tk.RIGHT)
        self.labelframe.pack_propagate(0)

        # main canvas
        self.cv_main = Virtual3DCanvas(self, width, height * 3 / 4 - labelheight, logger)

        if is_on_Linux():
            self.cv_main.bind("<Button-4>", functools.partial(self.on_mousewheel, scroll=1))
            self.cv_main.bind("<Button-5>", functools.partial(self.on_mousewheel, scroll=-1))
        else:
            self.cv_main.bind("<MouseWheel>", functools.partial(self.on_mousewheel, scroll=0))
        self.cv_main.grid(row=1, rowspan=3, columnspan=4)

        # zoomed canvas
        self.zoomcv_w = int(width * 1 / 4)
        self.zoomcv_h = int(height * 1 / 4)
        self.cv_zoom_rgb = tk.Canvas(self, width=self.zoomcv_w, height=self.zoomcv_h, bg='black')
        self.cv_zoom_r = tk.Canvas(self, width=self.zoomcv_w, height=self.zoomcv_h, bg='red')
        self.cv_zoom_g = tk.Canvas(self, width=self.zoomcv_w, height=self.zoomcv_h, bg='green')
        self.cv_zoom_b = tk.Canvas(self, width=self.zoomcv_w, height=self.zoomcv_h, bg='blue')
        self.cv_zoom_rgb.grid(row=4, column=0)
        self.cv_zoom_r.grid(row=4, column=1)
        self.cv_zoom_g.grid(row=4, column=2)
        self.cv_zoom_b.grid(row=4, column=3)

        self.clear_canvas()

    def load_photo(self, synapse_image: SynapseImage, **kwargs):
        if self.current_img_path == synapse_image.filepath:
            print("Already showing %s" % synapse_image.filepath)
            return None
        else:
            self.current_img_path = synapse_image.filepath
            self.arrimage = synapse_image.array
            self.arrlabel = synapse_image.synapse_qt.label()
            self.map_synapse = dict((syn.id, syn) for syn in synapse_image.synapse_qt.synapses())
            # main canvas
            self.cv_main.load_photo(synapse_image, **kwargs)
            # label
            self.updatelabel_info()
            self.updatelabel_state()

    def update_image_contrast(self, updated_array):
        self.cv_main.replace_array(updated_array)
        if self.synapse_chunk_id is not None:
            self.load_synapse_magnified_canvas(self.synapse_chunk_id)

    def clear_canvas(self):
        self.arrimage = None
        self.arrlabel = None
        self.current_img_path = ''
        self.synapse_chunk_id = None
        self.synapse_chunk_rgb = None
        self.synapse_chunk_r = None
        self.synapse_chunk_g = None
        self.synapse_chunk_b = None

        self.cv_main.clear_canvas()

        self.cv_zoom_rgb.delete(tk.ALL)
        self.cv_zoom_r.delete(tk.ALL)
        self.cv_zoom_g.delete(tk.ALL)
        self.cv_zoom_b.delete(tk.ALL)

        self.cv_zoom_rgb_img = self.cv_zoom_rgb.create_image(0, 0, anchor=tk.NW)
        self.cv_zoom_r_img = self.cv_zoom_r.create_image(0, 0, anchor=tk.NW)
        self.cv_zoom_g_img = self.cv_zoom_g.create_image(0, 0, anchor=tk.NW)
        self.cv_zoom_b_img = self.cv_zoom_b.create_image(0, 0, anchor=tk.NW)

        self.labeltext_info.set('')

    def updatelabel_info(self):
        if self.arrimage is not None:
            labeltext = "%s; z: %d/%d; %dx%d; %d%%;" % (os.path.basename(self.current_img_path),
                                                        self.cv_main.cur_z + 1, self.arrimage.shape[0],
                                                        self.arrimage.shape[2], self.arrimage.shape[1],
                                                        self.cv_main.get_zoom_percent())
            if self.synapse_chunk_id is not None:
                s = self.map_synapse[self.synapse_chunk_id]
                labeltext += "Synapse #%d: area=%d, centroid=(%.1f, %.1f, %.1f); %d%%;" % (
                    s.id, s.prop.area, *s.prop.centroid[::-1], self.cv_main.get_zoom_percent())
            self.labeltext_info.set(labeltext)

    def updatelabel_state(self):
        self.labeltext_state.set(self.cv_main.get_state_str())

    def on_mousewheel(self, event, scroll):
        if scroll == 0:
            if event.num == 5 or event.delta < 0:
                scroll = -1
            else:
                scroll = 1
        self.cv_main.on_mousewheel(scroll)
        self.updatelabel_info()
        if self.synapse_chunk_id is not None:
            self.change_magnified_img(self.synapse_chunk_rgb[self.cv_main.cur_z, ...],
                                      self.synapse_chunk_r[self.cv_main.cur_z, ...],
                                      self.synapse_chunk_g[self.cv_main.cur_z, ...],
                                      self.synapse_chunk_b[self.cv_main.cur_z, ...])

    def on_select_synapse(self, synapse_id):
        # main canvas
        self.synapse_chunk_id = synapse_id
        s = self.map_synapse[synapse_id]
        self.cv_main.goto(s.prop.centroid)
        self.updatelabel_info()
        # magnified canvas
        self.load_synapse_magnified_canvas(synapse_id)

    def load_synapse_magnified_canvas(self, synapse_id):
        s = self.map_synapse[synapse_id]
        self.synapse_chunk_rgb, synapse_label = self.cv_main.get_image_chunk(s.prop.centroid, halfsize=7)
        self.synapse_chunk_r = self.synapse_chunk_rgb.copy()
        self.synapse_chunk_r[..., 1] = 0
        self.synapse_chunk_r[..., 2] = 0
        self.synapse_chunk_g = self.synapse_chunk_rgb.copy()
        self.synapse_chunk_g[..., 0] = 0
        self.synapse_chunk_g[..., 2] = 0
        self.synapse_chunk_b = self.synapse_chunk_rgb.copy()
        self.synapse_chunk_b[..., 0] = 0
        self.synapse_chunk_b[..., 1] = 0
        self.synapse_chunk_b[synapse_label == synapse_id] = (255, 255, 255)
        self.change_magnified_img(self.synapse_chunk_rgb[int(s.prop.centroid[0])],
                                  self.synapse_chunk_r[int(s.prop.centroid[0])],
                                  self.synapse_chunk_g[int(s.prop.centroid[0])],
                                  self.synapse_chunk_b[int(s.prop.centroid[0])])

    def change_magnified_img(self, array_rgb, array_r, array_g, array_b):
        img_rgb = Image.fromarray(array_rgb)
        img_r = Image.fromarray(array_r)
        img_g = Image.fromarray(array_g)
        img_b = Image.fromarray(array_b)
        self.photo_rgb = ImageTk.PhotoImage(master=self,
                                            image=img_rgb.resize((self.zoomcv_w, self.zoomcv_h), Image.NEAREST))
        self.cv_zoom_rgb.itemconfig(self.cv_zoom_rgb_img, image=self.photo_rgb)
        self.photo_r = ImageTk.PhotoImage(master=self,
                                          image=img_r.resize((self.zoomcv_w, self.zoomcv_h), Image.NEAREST))
        self.cv_zoom_r.itemconfig(self.cv_zoom_r_img, image=self.photo_r)
        self.photo_g = ImageTk.PhotoImage(master=self,
                                          image=img_g.resize((self.zoomcv_w, self.zoomcv_h), Image.NEAREST))
        self.cv_zoom_g.itemconfig(self.cv_zoom_g_img, image=self.photo_g)
        self.photo_b = ImageTk.PhotoImage(master=self,
                                          image=img_b.resize((self.zoomcv_w, self.zoomcv_h), Image.NEAREST))
        self.cv_zoom_b.itemconfig(self.cv_zoom_b_img, image=self.photo_b)

    @staticmethod
    def _async_save(image, label, rejected_ids, filename):
        # save overlaid image
        image_corrected = image.copy()
        # remove rejected synapses
        for sid in rejected_ids:
            image_corrected[label == sid, 2] = 0
        imsave(filename, image_corrected, metadata={'axes': 'ZYXC'}, check_contrast=False)

        # save binary image as well
        if '_overlay.tif' in filename:
            filename_binary = filename.replace('_overlay.tif', '.tif')
        else:
            filename_binary = filename.replace('.tif', '_binary.tif')
        filename_binary = os.path.join(os.path.dirname(os.path.dirname(filename_binary)),
                                       os.path.basename(filename_binary))
        img_binary = label.copy().astype(np.ubyte)
        # remove rejected synapses
        for sid in rejected_ids:
            img_binary[img_binary == sid] = 0
        imsave(filename_binary, img_binary, metadata={'axes': 'ZYX'}, check_contrast=False)

    def save(self, savedir, rejected_ids, is_last):
        if self.arrimage is not None:
            filename = filedialog.asksaveasfilename(parent=self.master,
                                                    title="Save changes",
                                                    filetypes=(("tiff files", "*.tif"),),
                                                    initialdir=savedir,
                                                    initialfile=os.path.basename(self.current_img_path))

            if not filename:
                return False

            # synchronized writing for the last image, async for else
            if is_last:
                self._async_save(self.arrimage, self.arrlabel, rejected_ids, filename)
            else:
                async_writer = threading.Thread(target=self._async_save,
                                                args=(
                                                    self.arrimage.copy(), self.arrlabel.copy(), rejected_ids, filename))
                async_writer.start()

            return True
        return False

    def key(self, event):
        if event.char == '[' or event.char == '{':
            self.on_mousewheel(None, -1)
        elif event.char == ']' or event.char == '}':
            self.on_mousewheel(None, 1)
        elif event.char == ' ':
            self.cv_main.roll_state()
            self.updatelabel_state()
        elif event.char == 'b':
            self.cv_main.toggle_bbox()
        elif event.char == '+' or event.char == '=':
            self.cv_main.photoscale(True)
            self.updatelabel_info()
        elif event.char == '-' or event.char == '_':
            self.cv_main.photoscale(False)
            self.updatelabel_info()


class ImageListWidget(tk.Frame):
    def __init__(self, parent, btntext, initialdir, logger):
        super().__init__(parent, bd=1)
        self.logger = logger
        self.dir = ''
        self.initialdir = initialdir

        self.button = tk.Button(self, text=btntext)
        self.button.pack(side=tk.TOP, fill=tk.X)

        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.list = tk.Listbox(self, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.list.pack(side=tk.LEFT, expand=True, fill=tk.Y)

    def get_bg_color(self, itemid):
        return self.list.itemcget(itemid, 'bg')

    def set_bg_color(self, itemid, color):
        self.list.itemconfig(itemid, {'bg': color})

    def select(self, id):
        if self.list.size() > 0:
            self.list.select_clear(0, tk.END)
            self.list.select_set(id)


class SynapseListWidget(tk.Frame):
    def __init__(self, parent, width, height, lbltext, lblbg):
        super().__init__(parent, width=width, height=height)
        self.pack_propagate(0)
        self.dir = ''

        self.label = tk.Label(self, text=lbltext, bg=lblbg)
        self.label.pack(side=tk.TOP, fill=tk.X)

        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.list = tk.Listbox(self, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.list.yview)
        scrollbar.pack(side=tk.RIGHT, expand=1, fill=tk.Y)
        self.list.pack(side=tk.LEFT, expand=1, fill=tk.Y)

    def __contains__(self, synapse: Synapse3D):
        return str(synapse) in self.list.get(0, tk.END)

    def insert(self, synapse: Synapse3D):
        self.list.insert(tk.END, str(synapse))

    def delete(self, synapse: Synapse3D):
        idx = self.list.get(0, tk.END).index(str(synapse))
        self.list.delete(idx)

    def get_id_curselection(self):
        if self.list.size() > 0 and len(self.list.curselection()) > 0:
            return self.list.curselection()[0]
        else:
            return None

    def get_str_curselection(self):
        if self.list.size() > 0 and len(self.list.curselection()) > 0:
            return self.list.get(self.list.curselection()[0])
        else:
            return None

    def select_next(self):
        if self.list.size() > 0 and len(self.list.curselection()) > 0:
            cursel = self.get_id_curselection()
            self.list.select_clear(cursel)
            self.list.select_set(cursel + 1)

    def get_sid_list(self):
        list_str = self.list.get(0, tk.END)
        return [Synapse3D.str_to_id(id_str) for id_str in list_str]


class ContrastWindow(tk.Toplevel):
    class ColorMinmax(object):
        def __init__(self, parent, c, startrow):
            self.label_min = tk.Label(parent, text="Min (%s)" % c)
            self.var_min = tk.IntVar()
            self.scale_min = tk.Scale(parent, from_=0, orient=tk.HORIZONTAL, length=500, variable=self.var_min)
            self.label_max = tk.Label(parent, text="Max (%s)" % c)
            self.var_max = tk.IntVar()
            self.scale_max = tk.Scale(parent, from_=0, orient=tk.HORIZONTAL, length=500, variable=self.var_max)

            self.label_min.grid(row=startrow, column=0)
            self.scale_min.grid(row=startrow, column=1)
            self.label_max.grid(row=startrow + 1, column=0)
            self.scale_max.grid(row=startrow + 1, column=1)

        def set_val(self, v_min, v_max):
            self.var_min.set(v_min)
            self.var_max.set(v_max)

    def __init__(self, update_callback, auto_callback):
        super().__init__()
        self.title("Contrast")

        self.frame = tk.Frame(self)
        # r, g, b / min, max
        self.mm_red = ContrastWindow.ColorMinmax(self.frame, 'R', 0)
        self.mm_green = ContrastWindow.ColorMinmax(self.frame, 'G', 2)
        self.mm_blue = ContrastWindow.ColorMinmax(self.frame, 'B', 4)

        # actions
        self.check_var = tk.IntVar()
        self.check_keep_manual = tk.Checkbutton(self.frame, text='Keep this values', variable=self.check_var,
                                                onvalue=1, offvalue=0)
        self.check_keep_manual.grid(row=6, columnspan=2, sticky=tk.EW)
        self.button_set = tk.Button(self.frame, text="Set", command=update_callback)
        self.button_set.grid(row=7, columnspan=2, sticky=tk.EW)
        self.button_auto = tk.Button(self.frame, text="Auto", command=auto_callback)
        self.button_auto.grid(row=8, columnspan=2, sticky=tk.EW)

        self.frame.grid_propagate(1)
        self.frame.pack()
        self.resizable(0, 0)

    def init_with_image(self, image: SynapseImage):
        # only if when it's auto-contrast
        if self.check_var.get() == 0:
            maxval = np.iinfo(image.array.dtype).max
            self.mm_red.scale_min.config(to=maxval)
            self.mm_red.scale_max.config(to=maxval)
            self.mm_green.scale_min.config(to=maxval)
            self.mm_green.scale_max.config(to=maxval)
            self.mm_blue.scale_min.config(to=maxval)
            self.mm_blue.scale_max.config(to=maxval)
            self.mm_red.set_val(image.range_r[0], image.range_r[1])
            self.mm_green.set_val(image.range_g[0], image.range_g[1])
            self.mm_blue.set_val(image.range_b[0], image.range_b[1])

    def get_rgb_minmax(self):
        return (self.mm_red.var_min.get(), self.mm_red.var_max.get(), self.mm_green.var_min.get(),
                self.mm_green.var_max.get(), self.mm_blue.var_min.get(), self.mm_blue.var_max.get())
