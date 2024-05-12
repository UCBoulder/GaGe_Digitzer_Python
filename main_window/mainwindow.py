# This Python file uses the following encoding: utf-8
from PyQt5.QtWidgets import QApplication, QMainWindow
from PY.form import Ui_MainWindow
import pyqtgraph as pg
from configparser import ConfigParser
from numpy.fft import fftshift, ifftshift, rfft, irfft, rfftfreq
import sys

sys.path.append("../GaGe_Python")

# need to be on Windows with GaGe drivers installed
import Acquire
import mp_stream


buffer_size_to_sample_size = lambda x: x / 2
sample_size_to_buffer_size = lambda x: x * 2


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

        # connections
        self.pb_gage_acquire.clicked.connect(self.acquire)

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
            segmentsize = int(self.le_segmentsize.text())
            return segmentsize
        except Exception as e:
            print("Error:", e)

            default_segmentsize = 2**14
            self.le_segmentsize.setText(str(default_segmentsize))
            return default_segmentsize

    @property
    def buffersize(self):
        try:
            samplesize = int(self.le_buffersize.text())
            return sample_size_to_buffer_size(samplesize)
        except Exception as e:
            print("Error:", e)

            default_samplesize = 2**14
            self.le_buffersize.setText(str(default_samplesize))
            return sample_size_to_buffer_size(default_samplesize)

    def acquire(self):
        # write the latest config
        self.write_config_acquire()

        if self.mode_acquire == 2:
            x1, x2 = Acquire.acquire(self.segmentsize)
            ft_x1 = abs(rfft(x1))
            ft_x2 = abs(rfft(x2))
            self.rplt_td_1.plot(x1, clear=True, _callSync="off")
            self.rplt_fd_1.plot(ft_x1, clear=True, _callSync="off")
            self.rplt_td_2.plot(x2, clear=True, _callSync="off")
            self.rplt_fd_2.plot(ft_x2, clear=True, _callSync="off")
        else:
            (x1,) = Acquire.acquire(self.segmentsize)
            ft_x1 = abs(rfft(x1))
            self.rplt_td_1.plot(x1, clear=True, _callSync="off")
            self.rplt_fd_1.plot(ft_x1, clear=True, _callSync="off")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
