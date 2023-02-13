import functools
import glob
import logging
import os
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
from ttkthemes import ThemedStyle

import numpy as np

from gui.utils import BusyManager, SynapseImage, AsyncImageLoader
from gui.widgets_corrector import ImageListWidget, SynapseViewerWidget, SynapseListWidget, ContrastWindow, \
    Virtual3DCanvas
from segmentutil.synapse_quantification import SynapseQT3D


class SynapseCorrector(tk.Tk):
    def __init__(self, initialdir, savedir, logger):
        super().__init__()
        self.title("PsyQi Synapse Corrector")
        self.logger = logger
        self.resizable(0, 0)
        style = ThemedStyle(self)
        style.set_theme('breeze')

        self.initialdir = initialdir
        self.savedir = savedir
        self.window_height = 800
        self.width_left_panel = 200
        self.width_mid_panel = 800
        self.width_right_panel = 200

        self.img_list: ImageListWidget = None
        self.img_loader: AsyncImageLoader = None
        self.current_img_id = -1
        self.viewer: SynapseViewerWidget = None
        self.popup_contrast: ContrastWindow = None
        self._create_widgets()

        self.current_synapses: SynapseQT3D = None
        self.busymanager = BusyManager(self)

        # binding events
        self.bind("<Key>", self.key)
        self.bind('<Control-r>', lambda e: self._onclick_btn_reject())
        self.bind('<Control-s>', lambda e: self._onclick_btn_save())
        self.bind('<Control-z>', lambda e: self._onclick_btn_revert())

    # region init
    def _init_loader(self, filelist):
        self.img_loader = AsyncImageLoader(filelist)

    def _create_widgets(self):
        self._create_left_panel()
        self._create_mid_panel()
        self._create_right_panel()
        self._create_menubar()

    def _create_menubar(self):
        menubar = tk.Menu(self)
        # fixed width font
        fwf = ("Courier New", 10)
        self.config(menu=menubar)

        # file menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Select Dir              ", command=self._onclick_btn_load, font=fwf)
        filemenu.add_command(label="Save & Next       Ctrl+S", command=self._onclick_btn_save, font=fwf)
        filemenu.add_command(label="Quit                    ", command=self.destroy, font=fwf)
        menubar.add_cascade(label="File", menu=filemenu)

        # edit menu
        edit = tk.Menu(menubar, tearoff=0)
        self.popup_contrast = ContrastWindow(update_callback=self._update_contrast,
                                             auto_callback=self._autocontrast)
        self.popup_contrast.withdraw()
        self.popup_contrast.protocol("WM_DELETE_WINDOW", self.popup_contrast.withdraw)
        edit.add_command(label="Edit contrast               ", font=fwf, command=self.popup_contrast.deiconify)
        edit.add_separator()
        edit.add_command(label="Switch Canvas State      Space ", font=fwf,
                         command=functools.partial(self.event_generate, sequence='<space>'))
        edit.add_command(label="Show/Hide Synapse Box    B     ", font=fwf,
                         command=functools.partial(self.event_generate, sequence='<b>'))
        edit.add_separator()
        edit.add_command(label="Zoom In                  +     ", font=fwf,
                         command=functools.partial(self.viewer.cv_main.photoscale, is_up=True))
        edit.add_command(label="Zoom Out                 -     ", font=fwf,
                         command=functools.partial(self.viewer.cv_main.photoscale, is_up=False))
        edit.add_separator()
        edit.add_command(label="Find Current Syn         Ctrl+F", font=fwf,
                         command=functools.partial(self.event_generate, sequence='<Control-f>'))
        edit.add_command(label="Accept & Next            Ctrl+A", font=fwf,
                         command=functools.partial(self.event_generate, sequence='<Control-a>'))
        edit.add_command(label="Reject & Next            Ctrl+R", font=fwf,
                         command=functools.partial(self.event_generate, sequence='<Control-r>'))
        edit.add_command(label="Revert                         ", font=fwf, command=self._onclick_btn_revert)
        edit.add_separator()
        edit.add_command(label="Remove small blobs", font=fwf, command=self._reject_small_synapses)
        edit.add_command(label="Reject all inside of ROI       ", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=True, is_reject=True,
                                                   is_single_z=False))
        edit.add_command(label="Reject all outside of ROI   ", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=False, is_reject=True,
                                                   is_single_z=False))
        edit.add_command(label="Accept all inside of ROI    ", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=True, is_reject=False,
                                                   is_single_z=False))
        edit.add_command(label="Accept all outside of ROI   ", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=False, is_reject=False,
                                                   is_single_z=False))
        edit.add_command(label="Reject all inside of ROI (single Z)", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=True, is_reject=True,
                                                   is_single_z=True))
        edit.add_command(label="Reject all outside of ROI (single Z)", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=False, is_reject=True,
                                                   is_single_z=True))
        edit.add_command(label="Accept all inside of ROI (single Z)", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=True, is_reject=False,
                                                   is_single_z=True))
        edit.add_command(label="Accept all outside of ROI (single Z)", font=fwf,
                         command=functools.partial(self._on_block_action, is_inside=False, is_reject=False,
                                                   is_single_z=True))
        menubar.add_cascade(label="Edit", menu=edit)

        # image menu

    def _create_left_panel(self):
        self.left_panel = tk.Frame(self, width=self.width_left_panel, height=self.window_height, bd=1, relief=tk.SUNKEN)
        self.left_panel.grid(row=0, column=0, sticky=tk.NSEW)

        self.img_list = ImageListWidget(self.left_panel,
                                        btntext='Select Dir\n(Overlaid prediction)',
                                        initialdir=self.initialdir,
                                        logger=self.logger)
        self.img_list.button.config(command=self._onclick_btn_imglist)
        self.img_list.list.bind('<Double-Button-1>', lambda e: self._load_image())
        self.img_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_mid_panel(self):
        self.mid_panel = tk.Frame(self, width=self.width_mid_panel, height=self.window_height, bd=1, relief=tk.SUNKEN)
        self.mid_panel.grid(row=0, column=1, sticky=tk.NSEW)
        self.mid_panel.pack_propagate(0)

        self.viewer = SynapseViewerWidget(self.mid_panel, width=self.width_mid_panel, height=self.window_height,
                                          logger=self.logger)

        self.viewer.cv_main.bind('<Double-Button-1>', self._ondoubleclick_canvas)

        # set viewer.main canvas.popup menu
        self.viewer.cv_main.popup_menu.add_command(label="Reject all inside",
                                                   command=functools.partial(self._on_block_action, is_inside=True,
                                                                             is_reject=True, is_single_z=False))
        self.viewer.cv_main.popup_menu.add_command(label="Reject all outside",
                                                   command=functools.partial(self._on_block_action, is_inside=False,
                                                                             is_reject=True, is_single_z=False))
        self.viewer.cv_main.popup_menu.add_command(label="Accept all inside",
                                                   command=functools.partial(self._on_block_action, is_inside=True,
                                                                             is_reject=False, is_single_z=False))
        self.viewer.cv_main.popup_menu.add_command(label="Accept all outside",
                                                   command=functools.partial(self._on_block_action, is_inside=False,
                                                                             is_reject=False, is_single_z=False))

        # set viewer.main canvas.popup menu (shift)
        self.viewer.cv_main.popup_menu_shift.add_command(label="Reject all inside (single Z)",
                                                         command=functools.partial(self._on_block_action,
                                                                                   is_inside=True,
                                                                                   is_reject=True, is_single_z=True))
        self.viewer.cv_main.popup_menu_shift.add_command(label="Reject all outside (single Z)",
                                                         command=functools.partial(self._on_block_action,
                                                                                   is_inside=False,
                                                                                   is_reject=True, is_single_z=True))
        self.viewer.cv_main.popup_menu_shift.add_command(label="Accept all inside (single Z)",
                                                         command=functools.partial(self._on_block_action,
                                                                                   is_inside=True,
                                                                                   is_reject=False, is_single_z=True))
        self.viewer.cv_main.popup_menu_shift.add_command(label="Accept all outside (single Z)",
                                                         command=functools.partial(self._on_block_action,
                                                                                   is_inside=False,
                                                                                   is_reject=False, is_single_z=True))

    def _create_right_panel(self):
        self.right_panel = tk.Frame(self, width=self.width_right_panel, height=self.window_height, bd=1,
                                    relief=tk.SUNKEN)
        self.right_panel.grid(row=0, column=2, sticky=tk.NSEW)
        self.right_panel.grid_propagate(0)

        # synapse list positive
        self.syn_p_list = SynapseListWidget(self.right_panel,
                                            width=int(self.width_right_panel / 2),
                                            height=self.window_height - 100,
                                            lbltext='Positive\nsynapse',
                                            lblbg='#99FF99')
        self.syn_p_list.grid(row=0, column=0)
        self.syn_p_list.list.bind('<Double-Button-1>', self._ondoubleclick_synlist_item)
        self.syn_p_list.list.bind('<Control-f>', self._ondoubleclick_synlist_item)
        self.syn_p_list.list.bind('<Control-a>', lambda e: self.syn_p_list.select_next())

        # synapse list rejected
        self.syn_n_list = SynapseListWidget(self.right_panel,
                                            width=int(self.width_right_panel / 2),
                                            height=self.window_height - 100,
                                            lbltext='Rejected\nsynapse',
                                            lblbg='#FF9999')
        self.syn_n_list.grid(row=0, column=1)
        self.syn_n_list.list.bind('<Double-Button-1>', self._ondoubleclick_synlist_item)

        # buttons
        frame_btns = tk.Frame(self.right_panel, width=self.width_right_panel, height=100, bg='yellow')
        frame_btns.grid(row=1, columnspan=2)
        btn_reject = tk.Button(frame_btns, text='Reject', command=self._onclick_btn_reject)
        btn_reject.grid(row=0, column=0, sticky=tk.NSEW)
        btn_revert = tk.Button(frame_btns, text='Revert', command=self._onclick_btn_revert)
        btn_revert.grid(row=0, column=1, sticky=tk.NSEW)
        btn_save = tk.Button(frame_btns, text='Save & Next', command=self._onclick_btn_save)
        btn_save.grid(row=1, columnspan=2, sticky=tk.NSEW)
        frame_btns.grid_rowconfigure(0, weight=1)
        frame_btns.grid_rowconfigure(1, weight=1)
        frame_btns.grid_columnconfigure(0, weight=1)
        frame_btns.grid_columnconfigure(1, weight=1)
        frame_btns.grid_propagate(0)

    def on_delete(self):
        self.popup_contrast.destroy()
        # for child in self.winfo_children():
        #    child.destroy()

    # endregion

    def _load_image(self):
        self.busymanager.busy()
        if self.img_list.list.size() > 0:
            # clear
            current_zoom_idx = self.viewer.cv_main.zm_idx
            self.viewer.clear_canvas()
            self.syn_p_list.list.delete(0, tk.END)
            self.syn_n_list.list.delete(0, tk.END)
            if self.current_img_id != -1 and self.img_list.get_bg_color(self.current_img_id) == 'yellow':
                self.img_list.set_bg_color(self.current_img_id, 'white')

            # set
            self.current_img_id = self.img_list.list.curselection()[0]
            self.img_list.set_bg_color(self.current_img_id, 'yellow')
            selection_val = self.img_list.list.get(self.img_list.list.curselection()[0])
            dir = self.img_list.dir
            file = os.path.join(dir, selection_val)
            is_contrast_fix_or_auto = self.popup_contrast.check_var.get()
            contrast_range = self.popup_contrast.get_rgb_minmax()
            self.image: SynapseImage = self.img_loader.load_image(file, is_contrast_fix_or_auto, contrast_range)

            # set to viewer
            self.viewer.load_photo(self.image, init_zm_idx=current_zoom_idx)
            # set to contrast window
            self.popup_contrast.init_with_image(self.image)

            self.current_synapses = self.image.synapse_qt
            synapses = self.current_synapses.synapses()
            if synapses:
                for s in synapses:
                    self.syn_p_list.insert(s.id)
                    s.state = s.State.POSITIVE
            self.viewer.cv_main.refresh_synapse_state()
        self.busymanager.notbusy()

    def _ondoubleclick_canvas(self, event):
        widget: Virtual3DCanvas = event.widget
        i, j = widget.to_array_coord(event.x, event.y)
        for syn in self.current_synapses.synapses():
            if (syn.prop.bbox[0] <= widget.cur_z < syn.prop.bbox[3]) and (
                    syn.prop.bbox[1] <= j < syn.prop.bbox[4]) and (
                    syn.prop.bbox[2] <= i < syn.prop.bbox[5]):
                if syn.id in self.syn_p_list:
                    idx = self.syn_p_list.list.get(0, tk.END).index(syn.id)
                    self.syn_p_list.list.select_clear(0, tk.END)
                    self.syn_p_list.list.select_set(idx)
                elif syn.id in self.syn_n_list:
                    idx = self.syn_n_list.list.get(0, tk.END).index(syn.id)
                    self.syn_n_list.list.select_clear(0, tk.END)
                    self.syn_n_list.list.select_set(idx)
                self._focus_synapse(syn.id)
                break

    def _focus_synapse(self, sid):
        # revert last selected item's state
        for syn in self.current_synapses.synapses():
            if syn.state == syn.State.SELECTED:
                if syn.id in self.syn_p_list:
                    syn.state = syn.State.POSITIVE
                else:
                    syn.state = syn.State.NEGATIVE
        self.viewer.on_select_synapse(sid)

        # update state
        syn = self.current_synapses.get_synapse(sid)
        syn.state = syn.State.SELECTED
        self.viewer.cv_main.refresh_synapse_state()

    def _ondoubleclick_synlist_item(self, event):
        widget: tk.Listbox = event.widget
        if widget.size() > 0 and len(widget.curselection()) > 0:
            selection_val = widget.get(widget.curselection()[0])
            self._focus_synapse(selection_val)

    def _reject_small_synapses(self):
        min_area = simpledialog.askinteger("Are filtering", "Reject synapses whose volume is under",
                                           parent=self,
                                           minvalue=0)
        synapses = self.current_synapses.get_small_synapses(min_area)
        for syn in synapses:
            if syn.id in self.syn_p_list:
                self.syn_n_list.insert(syn.id)
                self.syn_p_list.delete(syn.id)
            syn.state = syn.State.NEGATIVE

        self.viewer.cv_main.refresh_synapse_state()

    # region btn callbacks
    def _onclick_btn_load(self):
        if self.img_list is not None:
            self._onclick_btn_imglist()

    def _onclick_btn_revert(self):
        if self.syn_n_list.get_id_curselection() is not None:
            item_str = self.syn_n_list.get_str_curselection()
            self.syn_p_list.insert(item_str)
            self.syn_n_list.delete(item_str)

            # update state
            syn = self.current_synapses.get_synapse(int(item_str))
            syn.state = syn.State.POSITIVE
            self.viewer.cv_main.refresh_synapse_state()

    def _onclick_btn_reject(self):
        if self.syn_p_list.get_id_curselection() is not None:
            selected_id = self.syn_p_list.get_id_curselection()
            item_str = self.syn_p_list.get_str_curselection()
            self.syn_n_list.insert(item_str)
            self.syn_p_list.delete(item_str)
            self.syn_p_list.list.selection_set(selected_id)

            # update state
            syn = self.current_synapses.get_synapse(int(item_str))
            syn.state = syn.State.NEGATIVE
            self.viewer.cv_main.refresh_synapse_state()

    def _onclick_btn_save(self):
        self.busymanager.busy()
        is_last = self.current_img_id == self.img_list.list.size() - 1
        if self.viewer.save(self.savedir, self.syn_n_list.list.get(0, tk.END), is_last):
            self.img_list.set_bg_color(self.current_img_id, '#99FF99')
            # select next image
            if not is_last:
                self.img_list.select(self.current_img_id + 1)
                self.img_list.list.select_clear(self.current_img_id)
                self.img_list.list.select_set(self.current_img_id + 1)
                self._load_image()
        self.busymanager.notbusy()

    def _onclick_btn_imglist(self):
        target_dir = filedialog.askdirectory(parent=self.master, initialdir=self.initialdir)
        if type(target_dir) is str and target_dir != '':
            self.load_images_from_dir(target_dir)

    def _on_block_action(self, is_inside, is_reject, is_single_z):
        roi = self.viewer.cv_main.get_roi()
        if roi is None:
            return

        if is_single_z:
            z0 = self.viewer.cv_main.get_cur_z()
            z1 = self.viewer.cv_main.get_cur_z() + 1
        else:
            z0 = 0
            z1 = np.iinfo(int).max

        if is_inside:
            synapses = self.current_synapses.get_synapses_inside([z0, z1, *roi])
        else:
            synapses = self.current_synapses.get_synapses_outside([z0, z1, *roi])

        if is_reject:
            for syn in synapses:
                if syn.id in self.syn_p_list:
                    self.syn_n_list.insert(syn.id)
                    self.syn_p_list.delete(syn.id)
                    syn.state = syn.State.NEGATIVE
        else:
            for syn in synapses:
                if syn.id in self.syn_n_list:
                    self.syn_n_list.delete(syn.id)
                    self.syn_p_list.insert(syn.id)
                    syn.state = syn.State.POSITIVE

        self.viewer.cv_main.refresh_synapse_state()

    def key(self, event):
        # keyboard input
        self.viewer.key(event)

    # endregion

    def load_images_from_dir(self, target_dir):
        self.img_list.dir = target_dir
        img_list = glob.glob(os.path.join(target_dir, '*.tif'))
        img_list.sort()
        self.img_list.list.delete(0, tk.END)
        for img in img_list:
            self.img_list.list.insert(tk.END, os.path.basename(img))

        self._init_loader(img_list)

    def _update_contrast(self):
        if self.image is not None:
            self.image.manualcontrast(*self.popup_contrast.get_rgb_minmax())
            self.viewer.update_image_contrast(self.image.array_contrast)

    def _autocontrast(self):
        if self.image is not None:
            self.image.autocontrast()
            self.popup_contrast.init_with_image(self.image)
            self.viewer.update_image_contrast(self.image.array_contrast)


def main(initialdir, savedir):
    # logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    window = SynapseCorrector(initialdir, savedir, logger)
    window.protocol("WM_DELETE_WINDOW", window.on_delete)
    window.mainloop()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main('', '')
