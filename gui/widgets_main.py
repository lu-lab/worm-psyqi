import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from logging.handlers import QueueHandler
import logging
from multiprocessing import Queue
import queue


class ScrolledLogger(ttk.Frame):
    def __init__(self, parent, height, logger):
        super().__init__(parent, height=height)
        self.pack(expand=1, fill='both')

        # create a ScrolledText widget to display log
        self.scrolled_text = ScrolledText(self, state='disabled')
        self.scrolled_text.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='black')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')
        self.scrolled_text.tag_config('CRITICAL', foreground='red', font=("TkFixedFont", 9, "bold"))
        # Create a logging handler using a queue
        self.log_queue = Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('[%(levelname)-8s]\t%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.after(100, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        # Autoscroll to the bottom
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.after(100, self.poll_log_queue)


class WidgetLabelFrame(object):
    def __init__(self, parent, text_label, grid_row):
        self.label = ttk.Label(parent, text=text_label)
        self.frame = ttk.Frame(parent)

        self.subwidgets = list()
        self.subvars = list()

        self._grid_row = grid_row
        self.show_widget()

    def add_subwidget_entry(self, text_entry, strvar=None, entrystate=tk.NORMAL):
        if strvar is None:
            var = tk.StringVar(self.frame)
        else:
            var = strvar
        entry = ttk.Entry(self.frame, textvariable=var, state=entrystate)
        var.trace(mode="w", callback=self.change_label_background)
        var.set(text_entry)
        entry.pack(side=tk.LEFT)

        self.subwidgets.append(entry)
        self.subvars.append(var)

    def add_subwidget_option(self, options, init_option, strvar=None):
        if strvar is None:
            var = tk.StringVar(self.frame)
        else:
            var = strvar
        option = ttk.OptionMenu(self.frame, var, init_option, *options)
        var.trace('w', self.change_label_background)
        var.set(init_option)  # default value
        option.pack(side=tk.LEFT)

        self.subwidgets.append(option)
        self.subvars.append(var)

    def add_subwidget_label(self, text_label, strvar=None):
        if strvar is None:
            var = tk.StringVar(self.frame)
        else:
            var = strvar
        label = ttk.Label(self.frame, textvariable=var)
        var.trace('w', self.change_label_background)
        var.set(text_label)  # default value
        label.pack(side=tk.LEFT)

        self.subwidgets.append(label)
        self.subvars.append(var)

    def change_label_background(self, *args):
        for var in self.subvars:
            if var.get() == '':
                self.label.config({"background": "red"})
                break
        else:
            self.label.config({"background": "#F6F4F2"})

    def hide_widget(self):
        self.label.grid_forget()
        self.frame.grid_forget()

    def show_widget(self):
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.frame.grid(row=self._grid_row, column=1, sticky=tk.NSEW)


class WidgetLabelEntry(object):
    def __init__(self, parent, text_label, text_entry, grid_row, strvar=None, entrystate=tk.NORMAL):
        self.label = ttk.Label(parent, text=text_label)
        if strvar is None:
            self.var = tk.StringVar(parent)
        else:
            self.var = strvar
        self.entry = ttk.Entry(parent, textvariable=self.var, state=entrystate)
        self.var.trace(mode="w", callback=self.change_label_background)
        self.var.set(text_entry)

        self._grid_row = grid_row
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.entry.grid(row=self._grid_row, column=1, sticky=tk.NSEW)
        #
        #self.show_widget()

    def set_entry(self, text):
        self.var.set(text)

    def change_label_background(self, *args):
        if self.var.get() == '':
            self.label.config({"background": "red"})
        else:
            self.label.config({"background": "#F6F4F2"})

    def hide_widget(self):
        self.label.grid_forget()
        self.entry.grid_forget()

    def show_widget(self):
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.entry.grid(row=self._grid_row, column=1, sticky=tk.W)


class WidgetLabelOption(object):
    def __init__(self, parent, text_label, options, init_option, grid_row, strvar=None):
        self.label = ttk.Label(parent, text=text_label)
        if strvar is None:
            self.var = tk.StringVar(parent)
        else:
            self.var = strvar
        self.option = ttk.OptionMenu(parent, self.var, init_option, *options)
        self.var.trace('w', self.change_label_background)
        self.var.set(init_option)  # default value

        self._grid_row = grid_row
        self.show_widget()

    def set_option(self, option):
        self.var.set(option)

    def change_label_background(self, *args):
        if self.var.get() == '':
            self.label.config({"background": "red"})
        else:
            self.label.config({"background": "#F6F4F2"})

    def hide_widget(self):
        self.label.grid_forget()
        self.option.grid_forget()

    def show_widget(self):
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.option.grid(row=self._grid_row, column=1, sticky=tk.W)


class WidgetLabelCheck(object):
    def __init__(self, parent, text_label, grid_row, boolvar=None, default=True):
        self.label = ttk.Label(parent, text=text_label)
        if boolvar is None:
            self.var = tk.BooleanVar(parent, default)
        else:
            self.var = boolvar
        self.check = ttk.Checkbutton(parent, variable=self.var)

        self._grid_row = grid_row
        self.show_widget()

    def set_check(self, bcheck):
        self.var.set(bcheck)

    def hide_widget(self):
        self.label.grid_forget()
        self.check.grid_forget()

    def show_widget(self):
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.check.grid(row=self._grid_row, column=1, sticky=tk.W)


class WidgetLabelRadiobutton(object):
    def __init__(self, parent, text_label, options, init_option, grid_row, strvar=None):
        self.label = ttk.Label(parent, text=text_label)
        if strvar is None:
            self.var = tk.StringVar(parent)
        else:
            self.var = strvar

        self.frame = ttk.Frame(parent)
        self.rbuttons = []
        for opt in options:
            rbtn = ttk.Radiobutton(self.frame, text=opt, variable=self.var, value=opt)
            rbtn.pack(side=tk.LEFT)
            self.rbuttons.append(rbtn)

        self.var.trace('w', self.change_label_background)
        self.var.set(init_option)  # default value

        self._grid_row = grid_row
        self.show_widget()

    def set_option(self, option):
        self.var.set(option)

    def change_label_background(self, *args):
        if self.var.get() == '':
            self.label.config({"background": "red"})
        else:
            self.label.config({"background": "#F6F4F2"})

    def hide_widget(self):
        self.label.grid_forget()
        self.frame.grid_forget()

    def show_widget(self):
        self.label.grid(row=self._grid_row, column=0, sticky=tk.NSEW)
        self.frame.grid(row=self._grid_row, column=1, sticky=tk.W)
