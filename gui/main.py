import json
import logging
import os
import sys
import threading
import tkinter as tk
from functools import partial
from logging import Logger
from os import DirEntry
from tkinter import ttk, filedialog, messagebox

from aicsimageio import AICSImage
from ttkthemes import ThemedStyle

from gui.widgets_main import ScrolledLogger, WidgetLabelFrame, WidgetLabelEntry, WidgetLabelOption, \
    WidgetLabelRadiobutton, WidgetLabelCheck
from scripts.detect_autofluorescent import remove_gut
from segment import mask_unet, mask_edge, train, predict, correct, quantify, watershed
from segmentutil.synapse_classification import SynapseClassifier_SVM, SynapseClassifier_AdaBoost, SynapseClassifier_RF

MASKING_OPTIONS = ["UNet (neurite.pt)", "UNet (multi_scale.pt)", "Edge detection"]
MODEL_OPTIONS = ["Random Forest (default)", "Kernel SVM", "AdaBoost"]
CLASSIFIER_OPTIONS = ["Built-in", "Custom"]
BUILT_IN_CLASSIFIER_OPTIONS = ["GRASP (sparse)", "GRASP (dense)", "CLA-1", "RAB-3"]


# main widget
class SynapseSegmentator(tk.Tk):
    def __init__(self, logger, theme, version):
        super().__init__()
        self.title("PsyQi by Lu Fluidics group v%s" % version)
        self.version = version
        style = ThemedStyle(self)
        style.set_theme(theme)
        self.resizable(True, False)

        self.fr_width = 600
        self.fr_main_height = 400
        self.fr_log_height = 400

        # list - to disable when something's going
        self.input_widgets = []

        # logger
        self.logger: Logger = logger
        self.logger_tofile = self.logger.getChild('file')
        self.logger_tofile.propagate = False
        self._create_widgets()

        # worker related
        self._async_worker: threading.Thread = None
        self._ondone = None
        self._show_message = False  # checker for whether showing "jobs done" message

    def _save_settings(self):
        savefile = os.path.join(self.dir_base.var.get(), 'pqsettings.json')
        settings = {'dir_raw': self.dir_raw.subvars[0].get(),
                    'dir_mask': self.dir_mask.var.get(),
                    'dir_label': self.dir_label.var.get(),
                    'dir_classifier': self.dir_classifier.var.get(),
                    'dir_pred': self.dir_pred.var.get(),
                    'dir_corr': self.dir_corr.var.get(),
                    'dir_watershed_in': self.tw_dir_in.var.get(),
                    'dir_watershed_out': self.tw_dir_out.var.get(),
                    'dir_qunt': self.dir_qunt.var.get(),
                    'option_channel': self._get_channel_order_text(),
                    'small_synapse_cutoff': self.small_synapse_cutoff.var.get(),
                    'masking_in_prediction': self.tp_masking.var.get(),
                    'process_number': self.process_number.var.get(),
                    'option_masking_method': self.option_masking_method.var.get(),
                    'option_model': self.option_model.var.get(),
                    'option_classifier': self.option_classifier.var.get(),
                    'option_builtin_classifier': self.option_builtin_classifier.var.get(),
                    'watershed_min_distance': self.tw_min_distance.var.get(),
                    }
        with open(savefile, 'w') as json_file:
            json.dump(settings, json_file)

    def _set_file_logger(self):
        """

        Args:
            action:

        Returns:

        """
        logfile = os.path.join(self.dir_base.var.get(), 'pq_processing_logs.txt')
        for hdlr in self.logger_tofile.handlers:  # remove all old handlers
            self.logger_tofile.removeHandler(hdlr)

        fileh = logging.FileHandler(logfile, 'a')
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        fileh.setFormatter(formatter)
        self.logger_tofile.addHandler(fileh)

    def _load_settings(self):
        savefile = os.path.join(self.dir_base.var.get(), 'pqsettings.json')
        if os.path.isfile(savefile):
            with open(savefile) as json_file:
                settings = json.load(json_file)
                self._set_option_rawdir(self.dir_base.var.get())

                if 'dir_raw' in settings: self.dir_raw.subvars[0].set(settings['dir_raw'])
                if 'dir_mask' in settings: self.dir_mask.set_entry(settings['dir_mask'])
                if 'dir_label' in settings: self.dir_label.set_entry(settings['dir_label'])
                if 'dir_classifier' in settings: self.dir_classifier.set_entry(settings['dir_classifier'])
                if 'dir_pred' in settings: self.dir_pred.set_entry(settings['dir_pred'])
                if 'dir_corr' in settings: self.dir_corr.set_entry(settings['dir_corr'])
                if 'dir_watershed_in' in settings: self.tw_dir_in.set_entry(settings['dir_watershed_in'])
                if 'dir_watershed_out' in settings: self.tw_dir_out.set_entry(settings['dir_watershed_out'])
                if 'dir_qunt' in settings: self.dir_qunt.set_entry(settings['dir_qunt'])
                if 'option_channel' in settings: self._set_channel_order(settings['option_channel'])
                if 'small_synapse_cutoff' in settings: self.small_synapse_cutoff.set_entry(
                    settings['small_synapse_cutoff'])
                if 'masking_in_prediction' in settings: self.tp_masking.set_check(
                    bool(settings['masking_in_prediction']))
                if 'process_number' in settings: self.process_number.set_entry(settings['process_number'])
                if 'option_masking_method' in settings:
                    if settings['option_masking_method'] in MASKING_OPTIONS:
                        self.option_masking_method.set_option(settings['option_masking_method'])
                    else:
                        self.option_masking_method.set_option(MASKING_OPTIONS[0])
                if 'option_model' in settings: self.option_model.set_option(settings['option_model'])
                if 'option_classifier' in settings: self.option_classifier.set_option(settings['option_classifier'])
                if 'option_builtin_classifier' in settings:
                    if settings['option_builtin_classifier'] in BUILT_IN_CLASSIFIER_OPTIONS:
                        self.option_builtin_classifier.set_option(settings['option_builtin_classifier'])
                    else:
                        self.option_builtin_classifier.set_option(BUILT_IN_CLASSIFIER_OPTIONS[0])
                if 'watershed_min_distance' in settings:
                    self.tw_min_distance.set_entry(settings['watershed_min_distance'])

    # region create widgets
    def _create_widgets(self):
        self._create_main_window()
        self._create_log_frame()
        self.logger.info("Program started")

    def _create_main_window(self):
        # tabs
        self.fr_main = ttk.Notebook(self)  # , width=self.fr_width, height=self.fr_main_height)
        self._create_setting_tab(self.fr_main)
        self._create_mask_tab(self.fr_main)
        self._create_train_tab(self.fr_main)
        self._create_predict_tab(self.fr_main)
        self._create_correct_tab(self.fr_main)
        self._create_watershed_tab(self.fr_main)
        self._create_quantify_tab(self.fr_main)
        self.fr_main.pack(expand=1, fill='both')

    def _create_setting_tab(self, parent):
        self.tab_setting = ttk.Frame(parent)
        parent.add(self.tab_setting, text="Settings")
        tk.Grid.columnconfigure(self.tab_setting, 1, weight=1)

        # select base directory button
        cur_row = 0
        self.btn_set_base_dr = ttk.Button(self.tab_setting, text='Select Base Directory',
                                          command=self._onbutton_set_base_dir)
        self.btn_set_base_dr.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_set_base_dr)
        cur_row += 1
        self.dir_base = WidgetLabelEntry(self.tab_setting, 'Base Directory (Full)', '', cur_row)
        self.input_widgets.append(self.dir_base.entry)
        cur_row += 1
        ttk.Separator(self.tab_setting, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # raw directory - get subfolders in base directory and make it option
        self.dir_raw = WidgetLabelFrame(self.tab_setting, 'Raw Directory', cur_row)
        self.dir_raw.add_subwidget_option(['', ], '', )
        self.dir_raw.subvars[0].trace(mode="w", callback=self._on_change_rawdir)
        self.dir_raw.add_subwidget_label('')
        self.input_widgets.append(self.dir_raw.subwidgets[0])
        cur_row += 1

        # mask, prediction, correction, quantification directory - free entry
        self.dir_mask = WidgetLabelEntry(self.tab_setting, 'Mask Directory', 'mask', cur_row)
        self.input_widgets.append(self.dir_mask.entry)
        cur_row += 1
        self.dir_pred = WidgetLabelEntry(self.tab_setting, 'Prediction Directory', 'prediction', cur_row)
        self.input_widgets.append(self.dir_pred.entry)
        cur_row += 1
        self.dir_corr = WidgetLabelEntry(self.tab_setting, 'Correction Directory', 'prediction_corrected', cur_row)
        self.input_widgets.append(self.dir_corr.entry)
        cur_row += 1
        self.dir_qunt = WidgetLabelEntry(self.tab_setting, 'Quantification Directory', 'quantification', cur_row)
        self.input_widgets.append(self.dir_qunt.entry)
        cur_row += 1
        ttk.Separator(self.tab_setting, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # channel setting
        self.option_channel = WidgetLabelFrame(self.tab_setting, 'Channel Order (raw)', cur_row)
        self.option_channel.change_label_background = self._on_change_option_channel
        cur_row += 1
        self.channel_names = WidgetLabelFrame(self.tab_setting, 'Channel Name', cur_row)
        cur_row += 1

        # min voxel setting
        self.small_synapse_cutoff = WidgetLabelEntry(self.tab_setting, 'Small Synapse Cutoff', '1', cur_row)
        self.input_widgets.append(self.small_synapse_cutoff.entry)
        cur_row += 1

        # Multiprocessing setting
        self.process_number = WidgetLabelEntry(self.tab_setting, 'Processes', '1', cur_row)
        self.input_widgets.append(self.process_number.entry)
        cur_row += 1

        self.btn_go_all = ttk.Button(self.tab_setting, text='Do all: Mask->Train->Predict->Correct->Quantify',
                                     command=self._onbutton_go_all)
        self.btn_go_all.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_all)
        cur_row += 1

    def _create_mask_tab(self, parent):
        self.tab_mask = ttk.Frame(parent)
        parent.add(self.tab_mask, text="Mask")
        tk.Grid.columnconfigure(self.tab_mask, 1, weight=1)

        # dir info (base - read only)
        cur_row = 0
        self.tm_dir_base = WidgetLabelEntry(self.tab_mask, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.tm_dir_raw = WidgetLabelEntry(self.tab_mask, 'Raw Directory', '', cur_row, strvar=self.dir_raw.subvars[0])
        self.input_widgets.append(self.tm_dir_raw.entry)
        cur_row += 1
        self.tm_dir_mask = WidgetLabelEntry(self.tab_mask, 'Mask Directory', 'mask', cur_row, strvar=self.dir_mask.var)
        self.input_widgets.append(self.tm_dir_mask.entry)
        cur_row += 1
        ttk.Separator(self.tab_mask, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # Masking method select
        self.option_masking_method = WidgetLabelOption(self.tab_mask, 'Masking Method', MASKING_OPTIONS,
                                                       MASKING_OPTIONS[0], cur_row)
        self.input_widgets.append(self.option_masking_method.option)
        cur_row += 1

        ttk.Separator(self.tab_mask, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # cleaning
        # disable post-processing
        # self.btn_clean_mask = ttk.Button(self.tab_mask, text='Postprocess - remove gut autofluorescent',
        #                                  command=partial(self._onbutton_clean_mask, True))
        # self.btn_clean_mask.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        # self.input_widgets.append(self.btn_clean_mask)
        # cur_row += 1

        # go
        self.btn_go_masking = ttk.Button(self.tab_mask, text='Go', command=partial(self._onbutton_go_masking, True))
        self.btn_go_masking.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_masking)
        cur_row += 1

    def _create_train_tab(self, parent):
        self.tab_train = ttk.Frame(parent)
        parent.add(self.tab_train, text="Train")
        tk.Grid.columnconfigure(self.tab_train, 1, weight=1)

        # dir info (base - read only)
        cur_row = 0
        self.tt_dir_base = WidgetLabelEntry(self.tab_train, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.dir_label = WidgetLabelEntry(self.tab_train, 'Label Directory', 'label', cur_row)
        self.input_widgets.append(self.dir_label.entry)
        cur_row += 1
        self.dir_classifier = WidgetLabelEntry(self.tab_train, 'Classifier Directory', 'classifier', cur_row)
        self.input_widgets.append(self.dir_classifier.entry)
        cur_row += 1
        ttk.Separator(self.tab_train, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # Model select
        self.option_model = WidgetLabelOption(self.tab_train, 'Classifier Model', MODEL_OPTIONS, MODEL_OPTIONS[0],
                                              cur_row)
        self.input_widgets.append(self.option_model.option)
        cur_row += 1

        # Multiprocessing setting
        self.tt_process_number = WidgetLabelEntry(self.tab_train, 'Processes', '1', cur_row,
                                                  strvar=self.process_number.var)
        self.input_widgets.append(self.tt_process_number.entry)
        cur_row += 1

        # go
        self.btn_go_training = ttk.Button(self.tab_train, text='Go', command=partial(self._onbutton_go_training, True))
        self.btn_go_training.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_training)
        cur_row += 1

    def _create_predict_tab(self, parent):
        self.tab_predict = ttk.Frame(parent)
        parent.add(self.tab_predict, text="Predict")
        tk.Grid.columnconfigure(self.tab_predict, 1, weight=1)

        # dir info (base - read only)
        cur_row = 0
        self.tp_dir_base = WidgetLabelEntry(self.tab_predict, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.tp_dir_raw = WidgetLabelEntry(self.tab_predict, 'Raw Directory', '', cur_row,
                                           strvar=self.dir_raw.subvars[0])
        self.input_widgets.append(self.tp_dir_raw.entry)
        cur_row += 1
        self.tp_dir_mask = WidgetLabelEntry(self.tab_predict, 'Mask Directory', 'mask', cur_row,
                                            strvar=self.dir_mask.var)
        self.input_widgets.append(self.tp_dir_mask.entry)
        cur_row += 1
        self.tp_dir_pred = WidgetLabelEntry(self.tab_predict, 'Prediction Directory', 'prediction', cur_row,
                                            strvar=self.dir_pred.var)
        self.input_widgets.append(self.tp_dir_pred.entry)
        cur_row += 1
        ttk.Separator(self.tab_predict, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # Model select
        self.option_classifier = WidgetLabelRadiobutton(self.tab_predict, 'Classifier', CLASSIFIER_OPTIONS,
                                                        CLASSIFIER_OPTIONS[0], cur_row)
        self.option_classifier.var.trace(mode="w", callback=self._on_change_option_classifier)
        self.input_widgets.extend(self.option_classifier.rbuttons)
        cur_row += 1
        self.option_builtin_classifier = WidgetLabelRadiobutton(self.tab_predict, 'Built-in Type',
                                                                BUILT_IN_CLASSIFIER_OPTIONS,
                                                                BUILT_IN_CLASSIFIER_OPTIONS[0], cur_row)
        self.input_widgets.extend(self.option_builtin_classifier.rbuttons)
        cur_row += 1
        self.tp_dir_classifier = WidgetLabelEntry(self.tab_predict, 'Custom Classifier Dir.', 'classifier', cur_row,
                                                  strvar=self.dir_classifier.var)
        self.tp_dir_classifier.hide_widget()
        self.input_widgets.append(self.tp_dir_classifier.entry)
        cur_row += 1

        # Use the neurite mask or not
        self.tp_masking = WidgetLabelCheck(self.tab_predict, 'Use the neurite mask', cur_row)
        self.input_widgets.append(self.tp_masking.check)
        cur_row += 1

        # Single-voxel elimination or not
        self.tp_small_synapse_cutoff = WidgetLabelEntry(self.tab_predict, 'Small Synapse Cutoff', '1', cur_row,
                                                        strvar=self.small_synapse_cutoff.var)
        cur_row += 1

        # Multiprocessing setting
        self.tp_process_number = WidgetLabelEntry(self.tab_predict, 'Processes', '1', cur_row,
                                                  strvar=self.process_number.var)
        self.input_widgets.append(self.tp_process_number.entry)
        cur_row += 1

        # go
        self.btn_go_predicting = ttk.Button(self.tab_predict, text='Go',
                                            command=partial(self._onbutton_go_predicting, True))
        self.btn_go_predicting.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_predicting)
        cur_row += 1

    def _create_correct_tab(self, parent):
        self.tab_correct = ttk.Frame(parent)
        parent.add(self.tab_correct, text="Correct")
        tk.Grid.columnconfigure(self.tab_correct, 1, weight=1)

        # dir info (base - read only)
        cur_row = 0
        self.tc_dir_base = WidgetLabelEntry(self.tab_correct, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.tc_dir_pred = WidgetLabelEntry(self.tab_correct, 'Prediction Directory', 'prediction', cur_row,
                                            strvar=self.dir_pred.var)
        self.input_widgets.append(self.tc_dir_pred.entry)
        cur_row += 1
        self.tc_dir_corr = WidgetLabelEntry(self.tab_correct, 'Correction Directory', 'prediction_corrected', cur_row,
                                            strvar=self.dir_corr.var)
        self.input_widgets.append(self.tc_dir_corr.entry)
        cur_row += 1
        ttk.Separator(self.tab_correct, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # go
        self.btn_go_correcting = ttk.Button(self.tab_correct, text='Go', command=self._onbutton_go_correcting)
        self.btn_go_correcting.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_correcting)
        cur_row += 1

    def _create_watershed_tab(self, parent):
        self.tab_watershed = ttk.Frame(parent)
        parent.add(self.tab_watershed, text="Watershed")
        tk.Grid.columnconfigure(self.tab_watershed, 1, weight=1)

        cur_row = 0
        self.tw_title = ttk.Label(self.tab_watershed, text='For watershed instance segmentation',
                                  font='TKFixedFont 15 bold', justify=tk.CENTER)
        self.tw_title.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # dir info (base - read only)
        self.tw_dir_base = WidgetLabelEntry(self.tab_watershed, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.tw_dir_raw = WidgetLabelEntry(self.tab_watershed, 'Raw Directory', '', cur_row,
                                           strvar=self.dir_raw.subvars[0])
        self.input_widgets.append(self.tw_dir_raw.entry)
        cur_row += 1
        self.tw_dir_in = WidgetLabelEntry(self.tab_watershed, 'Input Directory', 'prediction_corrected', cur_row)
        self.input_widgets.append(self.tw_dir_in.entry)
        cur_row += 1
        self.tw_dir_out = WidgetLabelEntry(self.tab_watershed, 'Output Directory', 'prediction_corrected_watershed (7)',
                                           cur_row)
        self.input_widgets.append(self.tw_dir_out.entry)
        cur_row += 1
        self.tw_min_distance = WidgetLabelEntry(self.tab_watershed, 'Min distance', '7', cur_row)
        self.input_widgets.append(self.tw_min_distance.entry)
        cur_row += 1

        # Multiprocessing setting
        self.tw_process_number = WidgetLabelEntry(self.tab_watershed, 'Processes', '1', cur_row,
                                                  strvar=self.process_number.var)
        self.input_widgets.append(self.tw_process_number.entry)
        cur_row += 1

        # action buttons
        self.btn_go_watersheding = ttk.Button(self.tab_watershed, text='Watershed',
                                              command=partial(self._onbutton_go_watersheding, True))
        self.btn_go_watersheding.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_watersheding)
        cur_row += 1
        self.btn_check_watersheding = ttk.Button(self.tab_watershed, text='Check (synapse corrector)',
                                                 command=partial(self._onbutton_check_watershed))
        self.btn_check_watersheding.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_check_watersheding)
        cur_row += 1

    def _create_quantify_tab(self, parent):
        self.tab_quantify = ttk.Frame(parent)
        parent.add(self.tab_quantify, text="Quantify")
        tk.Grid.columnconfigure(self.tab_quantify, 1, weight=1)

        # dir info (base - read only)
        cur_row = 0
        self.tq_dir_base = WidgetLabelEntry(self.tab_quantify, 'Base Directory (Full)', '', cur_row,
                                            strvar=self.dir_base.var, entrystate=tk.DISABLED)
        cur_row += 1
        self.tq_dir_mask = WidgetLabelEntry(self.tab_quantify, 'Mask Directory', 'mask', cur_row,
                                            strvar=self.dir_mask.var)
        self.input_widgets.append(self.tq_dir_mask.entry)

        cur_row += 1
        self.tq_dir_quant_in = WidgetLabelEntry(self.tab_quantify, 'Quantification In Dir.', 'prediction_corrected',
                                                cur_row)
        self.input_widgets.append(self.tq_dir_quant_in.entry)
        cur_row += 1
        self.tq_dir_quant_out = WidgetLabelEntry(self.tab_quantify, 'Quantification Out Dir.', 'quantification',
                                                 cur_row, strvar=self.dir_qunt.var)
        self.input_widgets.append(self.tq_dir_quant_out.entry)
        cur_row += 1

        # Multiprocessing setting
        self.tq_process_number = WidgetLabelEntry(self.tab_quantify, 'Processes', '1', cur_row,
                                                  strvar=self.process_number.var)
        self.input_widgets.append(self.tq_process_number.entry)
        cur_row += 1

        ttk.Separator(self.tab_quantify, orient=tk.VERTICAL).grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        cur_row += 1

        # go
        self.btn_go_quantifying = ttk.Button(self.tab_quantify, text='Go',
                                             command=partial(self._onbutton_go_quantifying, True))
        self.btn_go_quantifying.grid(row=cur_row, columnspan=2, sticky=tk.NSEW)
        self.input_widgets.append(self.btn_go_quantifying)
        cur_row += 1

    def _create_log_frame(self):
        self.fr_log = ScrolledLogger(self, self.fr_log_height, self.logger)

    # endregion

    # region control inputs
    def _disable_inputs(self):
        for w in self.input_widgets:
            w['state'] = tk.DISABLED

    def _enable_inputs(self):
        for w in self.input_widgets:
            w['state'] = tk.NORMAL

    def _set_option_rawdir(self, basedir):
        tree: list[DirEntry] = list(os.scandir(basedir))
        list_subfolders = [f.name for f in tree if f.is_dir()]
        list_subfolders = ['.', ] + list_subfolders

        self.dir_raw.subwidgets[0].option_clear()
        if 'raw' in list_subfolders:
            init_sel_index = list_subfolders.index('raw')
        else:
            init_sel_index = 0
        self.dir_raw.subwidgets[0].set_menu(list_subfolders[init_sel_index], *list_subfolders)

    def _on_change_rawdir(self, *args):
        if not hasattr(self, 'dir_base') or not hasattr(self, 'dir_raw'):
            return

        dir_raw_full = os.path.join(self.dir_base.var.get(), self.dir_raw.subvars[0].get())
        if os.path.isdir(dir_raw_full):
            tree: list[DirEntry] = list(os.scandir(dir_raw_full))
            raw_images = [f for f in tree if
                          f.is_file() and os.path.splitext(f)[1] in ['.tif', '.tiff', '.czi', '.nd2']]
            if len(raw_images) < 2:
                self.dir_raw.subvars[1].set("(%d file)" % len(raw_images))
            else:
                self.dir_raw.subvars[1].set("(%d files)" % len(raw_images))

            # change channel options
            if len(raw_images) > 0:
                # remove current option lists
                self.option_channel.subvars = []
                for w in self.option_channel.subwidgets:
                    self.input_widgets.remove(w)
                    w.destroy()
                self.option_channel.subwidgets = []
                self.channel_names.subvars = []
                for w in self.channel_names.subwidgets:
                    w.destroy()
                self.channel_names.subwidgets = []

                # add an option list n_c times
                n_c = AICSImage(raw_images[0].path).dims['C'][0]
                channel_names = AICSImage(raw_images[0].path).channel_names
                for _ in range(n_c):
                    self.option_channel.add_subwidget_option(['None', 'Synapse', 'Neuron', 'Bright-field'], 'None')
                    self.input_widgets.append(self.option_channel.subwidgets[-1])

                # if metadata has channel name, then print it
                if len(channel_names) == n_c:
                    self.channel_names.add_subwidget_label(' | '.join(channel_names))

    def _on_change_option_channel(self, *args):
        if len([v for v in self.option_channel.subvars if v.get() == 'Synapse']) == 1:
            self.option_channel.label.config({"background": "#F6F4F2"})
        else:
            self.option_channel.label.config({"background": "red"})

    def _on_change_option_classifier(self, *args):
        if self.option_classifier.var.get() == CLASSIFIER_OPTIONS[0]:  # built-in
            self.option_builtin_classifier.show_widget()
            self.tp_dir_classifier.hide_widget()
        else:  # custom
            self.option_builtin_classifier.hide_widget()
            self.tp_dir_classifier.show_widget()

    def _get_channel_order_text(self):
        text = ''
        for v in self.option_channel.subvars:
            v_text = v.get()
            if v_text == 'Synapse':
                text += 'S'
            elif v_text == 'Neuron':
                text += 'N'
            elif v_text == 'Bright-field':
                text += 'B'
            else:
                text += 'X'
        return text

    def _set_channel_order(self, text):
        if len(self.option_channel.subvars) != len(text):
            return
        else:
            for v, ch in zip(self.option_channel.subvars, text):
                if ch == 'S':
                    v.set('Synapse')
                elif ch == 'N':
                    v.set('Neuron')
                elif ch == 'B':
                    v.set('Bright-field')
                else:
                    v.set('None')

    # endregion

    def _check_async_worker(self):
        if self._async_worker is None:
            return

        if self._async_worker.is_alive():
            self.after(200, self._check_async_worker)
        else:
            self._async_worker = None
            # do the registered callbacks
            if self._ondone is not None:
                self._ondone()
            # show messagebox if needed
            if self._show_message:
                messagebox.showinfo("Message", "Jobs Done!")

    def _check_common_error(self):
        if len(self.option_channel.subwidgets) == 0:
            self.logger.error("Channel order undefined")
            return True

        if 'Synapse' not in [v.get() for v in self.option_channel.subvars]:
            self.logger.error("Missing synapse channel")
            return True

        if self.dir_base.var.get() == '':
            self.logger.error("Base directory undefined")
            return True

        return False

    # region button callbacks
    def _onbutton_set_base_dir(self):
        try:
            input = filedialog.askdirectory(initialdir=open('.lastbasedir').read())
        except:
            input = filedialog.askdirectory()

        if input:
            with open('.lastbasedir', 'w') as f:
                f.write(input)

            # set base directory
            self.dir_base.set_entry(input)

            # set options for the raw directory
            self._set_option_rawdir(input)

            self._load_settings()

            # set file logger
            self._set_file_logger()

    def _onbutton_go_all(self):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        self._set_worker_masking()
        if self._async_worker is not None:
            # cascade structure - once masking is done, call _onbutton_go_all_2
            self._ondone = self._go_all_2
            self._show_message = False
            self._async_worker.start()
            self.after(10, self._check_async_worker)

    def _go_all_2(self):
        # once training is done, call _onbutton_go_all_3
        self._set_worker_training()
        if self._async_worker is not None:
            self._ondone = self._go_all_3
            self._show_message = False
            self._async_worker.start()
            self.after(10, self._check_async_worker)

    def _go_all_3(self):
        # once predicting is done, call _onbutton_go_all_4
        self._set_worker_predicting()
        if self._async_worker is not None:
            self._ondone = self._go_all_4
            self._show_message = False
            self._async_worker.start()
            self.after(10, self._check_async_worker)

    def _go_all_4(self):
        # open corrector window
        correct(self.dir_base.var.get(), self.dir_pred.var.get(), self.dir_corr.var.get(),
                self._go_all_5, self.fr_log.log_queue)

    def _go_all_5(self):
        # after the corrector closed, move on to quantification
        self._set_worker_quantifying()
        self._ondone = self._enable_inputs
        self._show_message = True
        self._async_worker.start()
        self.after(10, self._check_async_worker)

    def _set_worker_masking(self):
        if self.option_masking_method.var.get() in (MASKING_OPTIONS[0], MASKING_OPTIONS[1]):
            # do job asynchronously
            self._async_worker = threading.Thread(target=mask_unet,
                                                  args=(self.dir_base.var.get(),
                                                        self.dir_raw.subvars[0].get(),
                                                        self.dir_mask.var.get(),
                                                        self.option_masking_method.var.get(),
                                                        self._get_channel_order_text(),
                                                        True,
                                                        self.fr_log.log_queue))
            self.logger_tofile.info('version=%s | function=%s | rawdir=%s | maskdir=%s | option=%s | channel=%s' % (
                self.version, 'mask_unet',
                self.dir_raw.subvars[0].get(),
                self.dir_mask.var.get(),
                self.option_masking_method.var.get(),
                self._get_channel_order_text(),
            ))
        elif self.option_masking_method.var.get() == MASKING_OPTIONS[2]:
            # do job asynchronously
            self._async_worker = threading.Thread(target=mask_edge,
                                                  args=(self.dir_base.var.get(),
                                                        self.dir_raw.subvars[0].get(),
                                                        self.dir_mask.var.get(),
                                                        self._get_channel_order_text(),
                                                        True,
                                                        self.fr_log.log_queue))
            self.logger_tofile.info('version=%s | function=%s | rawdir=%s | maskdir=%s | option=%s | channel=%s' % (
                self.version, 'mask_edge',
                self.dir_raw.subvars[0].get(),
                self.dir_mask.var.get(),
                self.option_masking_method.var.get(),
                self._get_channel_order_text(),
            ))
        else:
            self.logger.error("Unknown method for masking selected.")

    def _onbutton_go_masking(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        self._set_worker_masking()
        if self._async_worker is not None:
            #  do job asynchronously and register callback
            self._ondone = self._enable_inputs
            self._show_message = show_message
            self._async_worker.start()
            self.after(10, self._check_async_worker)

    def _set_worker_clean_mask(self):
        self._async_worker = threading.Thread(target=remove_gut,
                                              args=(self.dir_base.var.get(),
                                                    self.dir_mask.var.get(),
                                                    self.fr_log.log_queue))

    def _onbutton_clean_mask(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # do job asynchronously and register callback
        self._set_worker_clean_mask()
        self._ondone = self._enable_inputs
        self._show_message = show_message
        self._async_worker.start()
        self.after(10, self._check_async_worker)

    def _set_worker_training(self):
        if self.option_model.var.get() == MODEL_OPTIONS[0]:
            model = SynapseClassifier_RF
        elif self.option_model.var.get() == MODEL_OPTIONS[1]:
            model = SynapseClassifier_SVM
        elif self.option_model.var.get() == MODEL_OPTIONS[2]:
            model = SynapseClassifier_AdaBoost
        else:
            self.logger.error('Classifier Model Undefined')
            return
        self._async_worker = threading.Thread(target=train,
                                              args=(self.dir_base.var.get(),
                                                    self.dir_raw.subvars[0].get(),
                                                    self.dir_label.var.get(),
                                                    self.dir_classifier.var.get(),
                                                    int(self.process_number.var.get()),
                                                    self._get_channel_order_text(),
                                                    model,
                                                    -1,
                                                    True,
                                                    self.fr_log.log_queue))
        self.logger_tofile.info(
            'version=%s | function=%s | rawdir=%s | labeldir=%s | clfdir=%s | n_p=%d | channel=%s | option=%s' % (
                self.version, 'train',
                self.dir_raw.subvars[0].get(),
                self.dir_label.var.get(),
                self.dir_classifier.var.get(),
                int(self.process_number.var.get()),
                self._get_channel_order_text(),
                self.option_model.var.get()
            ))

    def _onbutton_go_training(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # do job asynchronously and register callback
        self._set_worker_training()
        if self._async_worker is not None:
            self._ondone = self._enable_inputs
            self._show_message = show_message
            self._async_worker.start()
            self.after(10, self._check_async_worker)

    def _set_worker_predicting(self):
        if self.option_classifier.var.get() == CLASSIFIER_OPTIONS[0]:
            option = self.option_classifier.var.get()
            kw = self.option_builtin_classifier.var.get()
        else:
            option = self.option_classifier.var.get()
            kw = self.dir_classifier.var.get()

        self._async_worker = threading.Thread(target=predict,
                                              args=(self.dir_base.var.get(),
                                                    self.dir_raw.subvars[0].get(),
                                                    self.dir_mask.var.get(),
                                                    self.dir_pred.var.get(),
                                                    int(self.process_number.var.get()),
                                                    int(self.small_synapse_cutoff.var.get()),
                                                    self._get_channel_order_text(),
                                                    (option, kw),
                                                    self.tp_masking.var.get(),
                                                    True,
                                                    self.fr_log.log_queue,))

        self.logger_tofile.info(
            'version=%s | function=%s | rawdir=%s | maskdir=%s | preddir=%s | n_p=%d | small_cutoff=%d | channel=%s | option=%s | masking=%s' % (
                self.version, 'predict',
                self.dir_raw.subvars[0].get(),
                self.dir_mask.var.get(),
                self.dir_pred.var.get(),
                int(self.process_number.var.get()),
                int(self.small_synapse_cutoff.var.get()),
                self._get_channel_order_text(),
                str((option, kw)),
                self.tp_masking.var.get(),
            ))

    def _onbutton_go_predicting(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # do job asynchronously and register callback
        self._set_worker_predicting()
        self._ondone = self._enable_inputs
        self._show_message = show_message
        self._async_worker.start()
        self.after(10, self._check_async_worker)

    def _onbutton_go_correcting(self):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # another GUI should be run in the main thread (tkinter restriction)
        self.logger_tofile.info(
            'version=%s | function=%s | preddir=%s | corrdir=%s' % (
                self.version, 'correct',
                self.dir_pred.var.get(),
                self.dir_corr.var.get(),
            ))
        correct(self.dir_base.var.get(), self.dir_pred.var.get(), self.dir_corr.var.get(),
                self._enable_inputs, self.fr_log.log_queue)

    def _set_worker_watersheding(self):
        self._async_worker = threading.Thread(target=watershed,
                                              args=(self.dir_base.var.get(),
                                                    self.dir_raw.subvars[0].get(),
                                                    self.tw_dir_in.var.get(),
                                                    self.tw_dir_out.var.get(),
                                                    int(self.tw_min_distance.var.get()),
                                                    int(self.process_number.var.get()),
                                                    self._get_channel_order_text(),
                                                    self.fr_log.log_queue))

        self.logger_tofile.info(
            'version=%s | function=%s | rawdir=%s | indir=%s | outdir=%s | mindist=%d | n_p=%d | channel=%s' % (
                self.version, 'watershed',
                self.dir_raw.subvars[0].get(),
                self.tw_dir_in.var.get(),
                self.tw_dir_out.var.get(),
                int(self.tw_min_distance.var.get()),
                int(self.process_number.var.get()),
                self._get_channel_order_text(),
            ))

    def _onbutton_go_watersheding(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # do job asynchronously and register callback
        self._set_worker_watersheding()
        self._ondone = self._enable_inputs
        self._show_message = show_message
        self._async_worker.start()
        self.after(10, self._check_async_worker)

    def _onbutton_check_watershed(self):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # another GUI should be run in the main thread (tkinter restriction)
        correct(self.dir_base.var.get(), self.tw_dir_out.var.get(), self.tw_dir_out.var.get(),
                self._enable_inputs, self.fr_log.log_queue)

    def _set_worker_quantifying(self):
        self._async_worker = threading.Thread(target=quantify,
                                              args=(self.dir_base.var.get(),
                                                    self.dir_raw.subvars[0].get(),
                                                    self.tq_dir_mask.var.get(),
                                                    self.tq_dir_quant_in.var.get(),
                                                    self.tq_dir_quant_out.var.get(),
                                                    int(self.process_number.var.get()),
                                                    self._get_channel_order_text(),
                                                    self.fr_log.log_queue), )
        self.logger_tofile.info(
            'version=%s | function=%s | rawdir=%s | maskdir=%s | indir=%s | outdir=%s | n_p=%d | channel=%s' % (
                self.version, 'quantify',
                self.dir_raw.subvars[0].get(),
                self.tq_dir_mask.var.get(),
                self.tq_dir_quant_in.var.get(),
                self.tq_dir_quant_out.var.get(),
                int(self.process_number.var.get()),
                self._get_channel_order_text(),
            ))

    def _onbutton_go_quantifying(self, show_message):
        if self._check_common_error():
            return
        # prevent from altering options or start another process
        self._disable_inputs()
        # save settings
        self._save_settings()

        # do job asynchronously and register callback
        self._set_worker_quantifying()
        self._ondone = self._enable_inputs
        self._show_message = show_message
        self._async_worker.start()
        self.after(10, self._check_async_worker)
    # endregion


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources'))

    return os.path.join(base_path, relative_path)


def main():
    # logger setting
    logger = logging.getLogger('PsyQiLogger')
    logger.setLevel(logging.DEBUG)
    window = SynapseSegmentator(logger, 'radiance', '1.1.1')

    window.iconphoto(False, tk.PhotoImage(file=resource_path('lulabicon.gif')))
    window.mainloop()


if __name__ == '__main__':
    main()
