# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QApplication, QMainWindow
from PY.form import Ui_MainWindow
import pyqtgraph as pg
from configparser import ConfigParser
from numpy.fft import fftshift, ifftshift, rfft, irfft, rfftfreq
import numpy as np
import threading
import multiprocessing as mp
import time
import PyQt5.QtCore as qtc
import sys
import PyQt5.QtWidgets as qt

sys.path.append("../GaGe_Python")

# need to be on Windows with GaGe drivers installed
# import Acquire
# import mp_stream


buffer_size_to_sample_size = lambda x: x / 2
sample_size_to_buffer_size = lambda x: x * 2


class Signal(qtc.QObject):
    sig = qtc.pyqtSignal(object)


def _add_RemoteGraphicsView_to_layout(layoutWidget):
    view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
    view.pg.setConfigOptions(
        antialias=True
    )  # prettier plots at no cost to the main process!
    rplt = view.pg.PlotItem()
    rplt._setProxyOptions(deferGetattr=True)  # speeds up access to rplt.plot
    view.setCentralItem(rplt)

    layoutWidget.addWidget(view)
    return view, rplt


def find_npts(x, level_percent):
    # remove the DC component
    x[:] -= x.mean()
    level = x.max() * level_percent * 0.01
    (idx,) = (x > level).nonzero()
    spacing = np.diff(idx)
    average_spacing = spacing[spacing > spacing.max() / 2].mean()
    ppifg = int(np.round(average_spacing))
    return average_spacing, ppifg


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)

        # make sure to plot using:
        # rplt.plot(data, clear=True, _callSync="off")
        self.view_td_1, self.rplt_td_1 = _add_RemoteGraphicsView_to_layout(self.gv_td_1)
        self.view_td_2, self.rplt_td_2 = _add_RemoteGraphicsView_to_layout(self.gv_td_2)
        self.view_fd_1, self.rplt_fd_1 = _add_RemoteGraphicsView_to_layout(self.gv_fd_1)
        self.view_fd_2, self.rplt_fd_2 = _add_RemoteGraphicsView_to_layout(self.gv_fd_2)
        self.read_config_stream()
        self.read_config_acquire()

        # variable declarations
        self.x1 = None
        self.x2 = None
        self.ppifg1 = None
        self.ppifg2 = None

        # multiprocessing Events
        self.stream_ready_event = mp.Event()
        self.stream_start_event = mp.Event()
        self.stream_stop_event = mp.Event()
        self.stream_error_event = mp.Event()
        self.stream_exit_event = mp.Event()
        self.N_analysis_threads = 4
        self.mp_values = []
        self.mp_arrays = []
        self.process_stream = None
        self.acquiring_in_process = mp.Event()

        # connections
        self.pb_gage_acquire.clicked.connect(self.acquire)
        self.pb_calc_ppifg.clicked.connect(self.calc_ppifg)
        self.pb_gage_stream.clicked.connect(self.stream)
        self.pb_stop_gage_stream.clicked.connect(self.stop_stream)
        self.actionsave_acquire.triggered.connect(self.save_acquire)
        self.actionsave_stream.triggered.connect(self.save_stream)

    def read_config_stream(self):
        config = self.config_stream
        setText = lambda i, s: self.tw_stream.item(i, 0).setText(s)

        setText(1, config["Acquisition"]["mode"])
        setText(2, config["Acquisition"]["samplerate"])
        setText(3, config["Acquisition"]["depth"])
        setText(4, config["Acquisition"]["segmentsize"])
        setText(5, config["Acquisition"]["segmentcount"])
        setText(6, config["Acquisition"]["triggerholdoff"])
        setText(7, config["Acquisition"]["triggerdelay"])
        setText(8, config["Acquisition"]["triggertimeout"])
        setText(9, config["Acquisition"]["extclk"])
        setText(10, config["Acquisition"]["timestampmode"])
        setText(11, config["Acquisition"]["timestampclock"])

        setText(13, config["Channel1"]["range"])
        setText(14, config["Channel1"]["coupling"])
        setText(15, config["Channel1"]["impedance"])

        setText(17, config["Channel2"]["range"])
        setText(18, config["Channel2"]["coupling"])
        setText(19, config["Channel2"]["impedance"])

        setText(21, config["Trigger1"]["condition"])
        setText(22, config["Trigger1"]["level"])
        setText(23, config["Trigger1"]["source"])

        setText(25, config["StmConfig"]["timeoutontransfer"])

        setText(27, config["PlotCheckLevel"]["plotchecklevel"])

    def read_config_acquire(self):
        config = self.config_acquire
        setText = lambda i, s: self.tw_acquire.item(i, 0).setText(s)

        setText(1, config["Acquisition"]["mode"])
        setText(2, config["Acquisition"]["samplerate"])
        setText(3, config["Acquisition"]["depth"])
        setText(4, config["Acquisition"]["segmentsize"])
        setText(5, config["Acquisition"]["segmentcount"])
        setText(6, config["Acquisition"]["triggerholdoff"])
        setText(7, config["Acquisition"]["triggerdelay"])
        setText(8, config["Acquisition"]["triggertimeout"])
        setText(9, config["Acquisition"]["extclk"])

        setText(11, config["Channel1"]["range"])
        setText(12, config["Channel1"]["coupling"])
        setText(13, config["Channel1"]["impedance"])
        setText(14, config["Channel1"]["dcoffset"])

        setText(16, config["Channel2"]["range"])
        setText(17, config["Channel2"]["coupling"])
        setText(18, config["Channel2"]["impedance"])
        setText(19, config["Channel2"]["dcoffset"])

        setText(21, config["Trigger1"]["condition"])
        setText(22, config["Trigger1"]["level"])
        setText(23, config["Trigger1"]["source"])
        setText(24, config["Trigger1"]["range"])
        setText(25, config["Trigger1"]["impedance"])

    def write_config_stream(self):
        config = self.config_stream
        get_item = lambda i: self.tw_stream.item(i, 0).text()

        config["Acquisition"]["mode"] = get_item(1)
        config["Acquisition"]["samplerate"] = get_item(2)
        config["Acquisition"]["depth"] = get_item(3)
        config["Acquisition"]["segmentsize"] = get_item(4)
        config["Acquisition"]["segmentcount"] = get_item(5)
        config["Acquisition"]["triggerholdoff"] = get_item(6)
        config["Acquisition"]["triggerdelay"] = get_item(7)
        config["Acquisition"]["triggertimeout"] = get_item(8)
        config["Acquisition"]["extclk"] = get_item(9)
        config["Acquisition"]["timestampmode"] = get_item(10)
        config["Acquisition"]["timestampclock"] = get_item(11)

        config["Channel1"]["range"] = get_item(13)
        config["Channel1"]["coupling"] = get_item(14)
        config["Channel1"]["impedance"] = get_item(15)

        config["Channel2"]["range"] = get_item(17)
        config["Channel2"]["coupling"] = get_item(18)
        config["Channel2"]["impedance"] = get_item(19)

        config["Trigger1"]["condition"] = get_item(21)
        config["Trigger1"]["level"] = get_item(22)
        config["Trigger1"]["source"] = get_item(23)

        config["StmConfig"]["timeoutontransfer"] = get_item(25)

        config["PlotCheckLevel"]["plotchecklevel"] = get_item(27)

        inifile = "../GaGe_Python/Stream2Analysis.ini"
        with open(inifile, "w") as configfile:
            config.write(configfile)

    def write_config_acquire(self):
        config = self.config_acquire
        get_item = lambda i: self.tw_acquire.item(i, 0).text()

        config["Acquisition"]["mode"] = get_item(1)
        config["Acquisition"]["samplerate"] = get_item(2)
        config["Acquisition"]["depth"] = get_item(3)
        config["Acquisition"]["segmentsize"] = get_item(4)
        config["Acquisition"]["segmentcount"] = get_item(5)
        config["Acquisition"]["triggerholdoff"] = get_item(6)
        config["Acquisition"]["triggerdelay"] = get_item(7)
        config["Acquisition"]["triggertimeout"] = get_item(8)
        config["Acquisition"]["extclk"] = get_item(9)

        config["Channel1"]["range"] = get_item(11)
        config["Channel1"]["coupling"] = get_item(12)
        config["Channel1"]["impedance"] = get_item(13)
        config["Channel1"]["dcoffset"] = get_item(14)

        config["Channel2"]["range"] = get_item(16)
        config["Channel2"]["coupling"] = get_item(17)
        config["Channel2"]["impedance"] = get_item(18)
        config["Channel2"]["dcoffset"] = get_item(19)

        config["Trigger1"]["condition"] = get_item(21)
        config["Trigger1"]["level"] = get_item(22)
        config["Trigger1"]["source"] = get_item(23)
        config["Trigger1"]["range"] = get_item(24)
        config["Trigger1"]["impedance"] = get_item(25)

        inifile = "../GaGe_Python/Acquire.ini"
        with open(inifile, "w") as configfile:
            config.write(configfile)

    @property
    def config_stream(self):
        config = ConfigParser()
        config.read("../GaGe_Python/Stream2Analysis.ini")
        return config

    @property
    def config_acquire(self):
        config = ConfigParser()
        config.read("../GaGe_Python/Acquire.ini")
        return config

    @property
    def mode_stream(self):
        if self.tw_stream.item(1, 0).text().lower() == "single":
            return 1
        if self.tw_stream.item(1, 0).text().lower() == "dual":
            return 2
        else:
            raise ValueError("invalid mode")

    @property
    def mode_acquire(self):
        if self.tw_acquire.item(1, 0).text().lower() == "single":
            return 1
        if self.tw_acquire.item(1, 0).text().lower() == "dual":
            return 2
        else:
            raise ValueError("invalid mode")

    @property
    def segmentsize(self):
        try:
            segmentsize = int(float(self.le_segmentsize.text()))
            return segmentsize
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")

            default_segmentsize = 2**14
            self.le_segmentsize.setText(str(default_segmentsize))
            return default_segmentsize

    @property
    def buffersize(self):
        try:
            samplesize = int(float(self.le_buffersize.text()))
            return sample_size_to_buffer_size(samplesize)
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")

            default_samplesize = 2**14
            self.le_buffersize.setText(str(default_samplesize))
            return sample_size_to_buffer_size(default_samplesize)

    @property
    def saveArraySize(self):
        try:
            samplesize = int(float(self.le_savebuffersize.text()))
            return samplesize

        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")

            default_samplesize = 2**14 * 100  # save 100
            self.le_buffersize.setText(str(default_samplesize))
            return default_samplesize

    @property
    def samplerate_acquire(self):
        try:
            return int(float(self.tw_acquire.item(2, 0).text()))
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")

    @property
    def samplerate_stream(self):
        try:
            return int(float(self.tw_stream.item(2, 0).text()))
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")

    @property
    def plotsamplesize(self):
        try:
            return int(float(self.le_plotsamplesize.text()))
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}, set to default 2**14")
            self.le_plotsamplesize.setText(str(int(2**14)))
            return int(2**14)  # default plot sample size

    def acquire(self, *args, plot=True):
        if self.stream_start_event.is_set():
            self.tb_monitor.setText("stop the active stream")
            return

        if self.acquiring_in_process.is_set():
            self.tb_monitor.setText("wait for acquisition to finish")
            return

        # write the latest config
        self.write_config_acquire()

        if self.mode_acquire == 2:
            self.acquiring_in_process.set()
            x1, x2 = Acquire.acquire(self.segmentsize)
            self.acquiring_in_process.clear()

            if plot:
                x1_plot = x1[: self.plotsamplesize]
                x2_plot = x2[: self.plotsamplesize]

                # plotting
                t = np.arange(x1_plot.size) / self.samplerate_acquire
                freq = rfftfreq(x1_plot.size, d=1 / self.samplerate_acquire) * 1e-6
                ft_x1 = abs(rfft(x1_plot))
                ft_x2 = abs(rfft(x2_plot))
                self.rplt_td_1.plot(t, x1_plot, clear=True, _callSync="off")
                self.rplt_fd_1.plot(freq, ft_x1, clear=True, _callSync="off")
                self.rplt_td_2.plot(t, x2_plot, clear=True, _callSync="off")
                self.rplt_fd_2.plot(freq, ft_x2, clear=True, _callSync="off")

                self.x1 = x1
                self.x2 = x2

            return x1, x2

        else:
            self.acquiring_in_process.set()
            (x1,) = Acquire.acquire(self.segmentsize)
            self.acquiring_in_process.clear()

            if plot:
                x1_plot = x1[: self.plotsamplesize]

                # plotting
                t = np.arange(x1_plot.size) / self.samplerate_acquire
                freq = rfftfreq(x1_plot.size, d=1 / self.samplerate_acquire) * 1e-6
                ft_x1 = abs(rfft(x1_plot))
                self.rplt_td_1.plot(t, x1_plot, clear=True, _callSync="off")
                self.rplt_fd_1.plot(freq, ft_x1, clear=True, _callSync="off")

                self.x1 = x1

            return x1

    def calc_ppifg(self):
        try:
            level_percent = float(self.tw_stream.item(27, 0).text())
        except Exception as e:
            self.tb_monitor.setText(f"Error: {e}")
            return

        if self.mode_acquire == 1:
            if self.x1 is None:
                self.tb_monitor.setText("ch1 not acquired")
                return

            ppifg1_mean, ppifg1 = find_npts(self.x1, level_percent)
            self.tb_monitor.setText(f"ch1 ppifg: {np.round(ppifg1_mean, 5)}")
            self.ppifg1 = ppifg1

        if self.mode_acquire == 2:
            ch1_missing = False
            ch2_missing = False
            if self.x1 is None:
                ch1_missing = True
            if self.x2 is None:
                ch2_missing = True
            ch_missing = [ch1_missing, ch2_missing]

            if all(ch_missing):
                self.tb_monitor.setText("ch1 and ch2 not acquired")
                return
            elif any(ch_missing):
                ch = np.asarray(["ch1", "ch2"], dtype=str)[ch_missing]
                self.tb_monitor.setText(f"{ch} not acquired")
                return

            ppifg1_mean, ppifg1 = find_npts(self.x1, level_percent)
            ppifg2_mean, ppifg2 = find_npts(self.x2, level_percent)

            self.tb_monitor.setText(
                f"ch1 ppifg: {np.round(ppifg1_mean, 5)} \n ch2 ppifg: {np.round(ppifg2_mean, 3)}"
            )

            self.ppifg1 = ppifg1
            self.ppifg2 = ppifg2

    def stream(self):
        if self.stream_start_event.is_set():
            self.tb_monitor.setText("stop the active stream")
            return

        if self.acquiring_in_process.is_set():
            self.tb_monitor.setText("wait for acquisition to finish")
            return

        self.write_config_stream()

        # ===== doanalysis args and sanity checks =============================
        args_doanalysis = []
        samplebuffersize = int(buffer_size_to_sample_size(self.buffersize))

        if self.cb_average.isChecked():
            if samplebuffersize <= self.segmentsize:
                msg = "for averaging, you need buffersize > segmentsize"
                self.tb_monitor.setText(msg)
                return

            if samplebuffersize % self.segmentsize != 0:
                samplebuffersize = np.ceil(
                    np.round(samplebuffersize / self.segmentsize) * self.segmentsize
                )
                self.le_buffersize.setText(str(samplebuffersize))
                msg = "buffersize was adjusted to be integer multiple of segmentsize"
                self.tb_monitor.setText(msg)

            modes_doanalysis = ["save average", "average"]
            args_doanalysis += [self.segmentsize]

        else:
            modes_doanalysis = ["save", "pass"]

        if self.cb_save_stream.isChecked():
            if self.cb_average.isChecked():
                # saving averaged data, size >= segmentsize
                if self.saveArraySize < self.segmentsize:
                    self.tb_monitor.setText("save buffer size must be >= segment size")
                    return

            else:
                # saving raw data, size >= streaming buffer size
                if self.saveArraySize < samplebuffersize:
                    self.tb_monitor.setText(
                        "save buffer size must be >= stream buffer size"
                    )
                    return

            mode_doanalysis = modes_doanalysis[0]
            args_doanalysis += [self.saveArraySize, self.stream_stop_event]

        else:
            mode_doanalysis = modes_doanalysis[1]

        args_doanalysis = [mode_doanalysis] + args_doanalysis

        # ===== mp arrays =====================================================
        if self.cb_average.isChecked():
            if self.cb_save_stream.isChecked():
                self.mp_arrays = [mp.Array("q", self.saveArraySize)]
            else:
                self.mp_arrays = [mp.Array("q", self.segmentsize)]
        elif self.cb_save_stream.isChecked():
            self.mp_arrays = [mp.Array("q", self.saveArraySize)]
        else:
            self.mp_arrays = [mp.Array("q", samplebuffersize)]
        self.mp_values = [mp.Value("q"), mp.Value("q")]

        # ===== start stream ==================================================
        inifile = "../GaGe_Python/Stream2Analysis.ini"
        args = (
            inifile,
            self.buffersize,
            self.stream_ready_event,
            self.stream_start_event,
            self.stream_stop_event,
            self.stream_error_event,
            self.stream_exit_event,
            self.N_analysis_threads,
            self.mp_values,
            self.mp_arrays,
            args_doanalysis,
        )

        # ===== track card stream =============================================
        if self.cb_save_stream.isChecked():
            self.track_stream = TrackSave(self, 100)
            self.track_stream.signal_pb.sig.connect(self.update_progress_bar)
            self.track_stream.signal_tb.sig.connect(self.update_text_browser)
            self.track_stream.start()
        else:
            self.track_stream = TrackUpdate(self, 100)
            self.track_stream.signal_plot.sig.connect(self.update_plots)
            self.track_stream.signal_tb.sig.connect(self.update_text_browser)
            self.track_stream.start()

        # ===== start card stream =============================================
        # an odd bug is that the track_stream doesn't always catch the error
        # flag. This happens even when I start the track_stream thread first.
        # I no longer clear any flags in mp_stream, so they should all be
        # caught by the track_stream 🤷‍♂️
        self.process_stream = mp.Process(target=mp_stream.stream, args=args)
        self.process_stream.start()
        self.tb_monitor.setText("stream started")

    def stop_stream(self):
        if self.stream_start_event.is_set():
            self.stream_stop_event.set()
            self.tb_monitor.setText("stream stopped by user")
        else:
            self.tb_monitor.setText("stream is already not running")

    def update_progress_bar(self, val):
        self.pb.setValue(val)

    def update_text_browser(self, msg):
        self.tb_monitor.setText(msg)

    def update_plots(self):
        mp_array = self.mp_arrays[0]
        X = np.frombuffer(mp_array.get_obj(), np.int64)
        if self.mode_stream == 2:
            N = int(X.size // 2)
            X.resize((N, 2))

            x1 = X[:, 0]
            x2 = X[:, 1]

            x1 = x1[: self.plotsamplesize]
            x2 = x2[: self.plotsamplesize]

            # plotting
            t = np.arange(x1.size) / self.samplerate_stream
            freq = rfftfreq(x1.size, d=1 / self.samplerate_stream) * 1e-6
            ft_x1 = abs(rfft(x1))
            ft_x2 = abs(rfft(x2))
            self.rplt_td_1.plot(t, x1, clear=True, _callSync="off")
            self.rplt_fd_1.plot(freq, ft_x1, clear=True, _callSync="off")
            self.rplt_td_2.plot(t, x2, clear=True, _callSync="off")
            self.rplt_fd_2.plot(freq, ft_x2, clear=True, _callSync="off")

        else:
            x1 = X

            x1 = x1[: self.plotsamplesize]

            t = np.arange(x1.size) / self.samplerate_stream
            freq = rfftfreq(x1.size, d=1 / self.samplerate_stream) * 1e-6
            ft_x1 = abs(rfft(x1))
            self.rplt_td_1.plot(t, x1, clear=True, _callSync="off")
            self.rplt_fd_1.plot(freq, ft_x1, clear=True, _callSync="off")

    def save_acquire(self):
        if self.x1 is None:
            self.tb_monitor.setText("ch1 not acquired")
            return

        (filename, *_) = qt.QFileDialog.getSaveFileName(
            caption=f"Save Acquisition Data"
        )
        if filename == "":
            return

        np.save(filename + "_ch1.npy", self.x1)
        if self.mode_acquire == 2:
            if self.x2 is not None:
                np.save(filename + "_ch2.npy", self.x2)

    def save_stream(self):
        if len(self.mp_arrays) == 0:
            self.tb_monitor.setText("no stream acquired")
            return

        (filename, *_) = qt.QFileDialog.getSaveFileName(
            caption=f"Save Acquisition Data"
        )
        if filename == "":
            return

        (X,) = self.mp_arrays
        X = np.frombuffer(X.get_obj(), np.int64)
        if self.mode_stream == 2:
            N = int(X.size) // 2
            X.resize((N, 2))

            x1 = X[:, 0]
            x2 = X[:, 1]
        else:
            x1 = X
            x2 = None

        np.save(filename + "_ch1.npy", x1)
        if x2 is not None:
            np.save(filename + "_ch2.npy", x2)


class TrackUpdate(qtc.QThread):
    def __init__(self, mainwindow, wait_time):
        qtc.QThread.__init__(self)

        self.stream_ready_event = mainwindow.stream_ready_event
        self.stream_start_event = mainwindow.stream_start_event
        self.stream_error_event = mainwindow.stream_error_event
        self.stream_stop_event = mainwindow.stream_stop_event
        self.stream_exit_event = mainwindow.stream_exit_event

        self.wait_time = wait_time
        self.g_cardTotalData = mainwindow.mp_values[0]

        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.timer_timeout)
        self.timer.moveToThread(self)

        self.signal_plot = Signal()
        self.signal_tb = Signal()

    def run(self):
        self.start_time = time.time()
        self.timer.start(self.wait_time)
        loop = qtc.QEventLoop()
        loop.exec()

    def timer_timeout(self):
        if not self.stream_error_event.is_set():
            # ===== print out elapsed time and data rate ======================
            elapsed_time = time.time() - self.start_time
            hours = 0
            minutes = 0

            if elapsed_time > 0:
                total = self.g_cardTotalData.value / 1000000 * 2
                rate = total / elapsed_time

                seconds = int(elapsed_time)  # elapsed time is in seconds
                if seconds >= 60:  # seconds
                    minutes = seconds // 60
                    if minutes >= 60:
                        hours = minutes // 60
                        if hours > 0:
                            minutes %= 60
                    seconds %= 60

                s = "Total: {0:.2f} MB, Rate: {1:6.2f} MB/s Elapsed time: {2:d}:{3:02d}:{4:02d}\r".format(
                    total, rate, hours, minutes, seconds
                )

            self.signal_tb.sig.emit(s)

            # ===== tell gui thread to update plots ===========================
            self.signal_plot.sig.emit(None)

        if self.stream_error_event.is_set() or self.stream_stop_event.is_set():
            # wait for the card stream process to stop before clearing all
            # multiprocessing evnets
            self.stream_exit_event.wait()

            self.stream_exit_event.clear()
            self.stream_ready_event.clear()
            self.stream_start_event.clear()
            self.stream_error_event.clear()
            self.stream_stop_event.clear()

            self.timer.stop()
            self.exit()


class TrackSave(qtc.QThread):
    def __init__(self, mainwindow, wait_time):
        qtc.QThread.__init__(self)

        self.stream_ready_event = mainwindow.stream_ready_event
        self.stream_start_event = mainwindow.stream_start_event
        self.stream_error_event = mainwindow.stream_error_event
        self.stream_stop_event = mainwindow.stream_stop_event
        self.stream_exit_event = mainwindow.stream_exit_event

        self.saveArraySize = mainwindow.saveArraySize
        self.wait_time = wait_time

        self.g_cardTotalData = mainwindow.mp_values[0]
        if mainwindow.cb_average.isChecked():
            self.data_increment = mainwindow.segmentsize
        else:
            self.data_increment = buffer_size_to_sample_size(mainwindow.buffersize)
        self.loop_count = mainwindow.mp_values[1]

        self.signal_pb = Signal()
        self.signal_tb = Signal()

        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.timer_timeout)
        self.timer.moveToThread(self)

    @property
    def total_data(self):
        return self.data_increment * self.loop_count.value

    def run(self):
        self.start_time = time.time()
        self.timer.start(self.wait_time)
        loop = qtc.QEventLoop()
        loop.exec()

    def timer_timeout(self):
        if not self.stream_error_event.is_set():
            # ===== print out elapsed time and data rate ======================
            elapsed_time = time.time() - self.start_time
            hours = 0
            minutes = 0

            if elapsed_time > 0:
                total = self.g_cardTotalData.value / 1000000 * 2
                rate = total / elapsed_time

                seconds = int(elapsed_time)  # elapsed time is in seconds
                if seconds >= 60:  # seconds
                    minutes = seconds // 60
                    if minutes >= 60:
                        hours = minutes // 60
                        if hours > 0:
                            minutes %= 60
                    seconds %= 60

                s = "Total: {0:.2f} MB, Rate: {1:6.2f} MB/s Elapsed time: {2:d}:{3:02d}:{4:02d}\r".format(
                    total, rate, hours, minutes, seconds
                )

            self.signal_tb.sig.emit(s)

            # ===== update progress bar =======================================
            progress = self.total_data / self.saveArraySize
            self.signal_pb.sig.emit(int(np.round(progress * 100)))

        if self.stream_error_event.is_set() or self.stream_stop_event.is_set():
            # wait for the card stream process to stop before clearing all
            # multiprocessing evnets
            self.stream_exit_event.wait()

            self.stream_exit_event.clear()
            self.stream_ready_event.clear()
            self.stream_start_event.clear()
            self.stream_error_event.clear()
            self.stream_stop_event.clear()

            self.timer.stop()
            self.exit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
