# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/peterchang/Github/GaGe_Digitzer_Python/main_window/form.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(996, 818)
        MainWindow.setStyleSheet("background-color: rgb(156, 158, 158);\n"
"background-color: rgb(199, 202, 202);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setMinimumSize(QtCore.QSize(317, 466))
        self.groupBox.setMaximumSize(QtCore.QSize(317, 466))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.le_segmentsize = QtWidgets.QLineEdit(self.groupBox)
        self.le_segmentsize.setObjectName("le_segmentsize")
        self.gridLayout.addWidget(self.le_segmentsize, 0, 2, 1, 3)
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 1, 0, 1, 1)
        self.le_plotsamplesize = QtWidgets.QLineEdit(self.groupBox)
        self.le_plotsamplesize.setObjectName("le_plotsamplesize")
        self.gridLayout.addWidget(self.le_plotsamplesize, 1, 3, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 2)
        self.le_buffersize = QtWidgets.QLineEdit(self.groupBox)
        self.le_buffersize.setObjectName("le_buffersize")
        self.gridLayout.addWidget(self.le_buffersize, 2, 2, 1, 3)
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 3, 0, 1, 1)
        self.le_savebuffersize = QtWidgets.QLineEdit(self.groupBox)
        self.le_savebuffersize.setObjectName("le_savebuffersize")
        self.gridLayout.addWidget(self.le_savebuffersize, 3, 1, 1, 4)
        self.pb_calc_ppifg = QtWidgets.QPushButton(self.groupBox)
        self.pb_calc_ppifg.setObjectName("pb_calc_ppifg")
        self.gridLayout.addWidget(self.pb_calc_ppifg, 4, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(108, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 4, 3, 1, 2)
        self.pb_gage_acquire = QtWidgets.QPushButton(self.groupBox)
        self.pb_gage_acquire.setObjectName("pb_gage_acquire")
        self.gridLayout.addWidget(self.pb_gage_acquire, 5, 0, 1, 2)
        self.pb_gage_stream = QtWidgets.QPushButton(self.groupBox)
        self.pb_gage_stream.setObjectName("pb_gage_stream")
        self.gridLayout.addWidget(self.pb_gage_stream, 6, 0, 1, 2)
        self.pb_stop_gage_stream = QtWidgets.QPushButton(self.groupBox)
        self.pb_stop_gage_stream.setObjectName("pb_stop_gage_stream")
        self.gridLayout.addWidget(self.pb_stop_gage_stream, 7, 0, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(98, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 7, 4, 1, 1)
        self.cb_average = QtWidgets.QCheckBox(self.groupBox)
        self.cb_average.setObjectName("cb_average")
        self.gridLayout.addWidget(self.cb_average, 8, 0, 1, 4)
        self.cb_save_stream = QtWidgets.QCheckBox(self.groupBox)
        self.cb_save_stream.setObjectName("cb_save_stream")
        self.gridLayout.addWidget(self.cb_save_stream, 9, 0, 1, 2)
        self.tb_monitor = QtWidgets.QTextBrowser(self.groupBox)
        self.tb_monitor.setMaximumSize(QtCore.QSize(189, 81))
        self.tb_monitor.setObjectName("tb_monitor")
        self.gridLayout.addWidget(self.tb_monitor, 10, 0, 1, 5)
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 11, 0, 1, 1)
        self.pb = QtWidgets.QProgressBar(self.groupBox)
        self.pb.setProperty("value", 0)
        self.pb.setObjectName("pb")
        self.gridLayout.addWidget(self.pb, 11, 2, 1, 3)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 1, 1, 1)
        self.gv_td_1 = LayoutWidget(self.tab)
        self.gv_td_1.setObjectName("gv_td_1")
        self.gridLayout_2.addWidget(self.gv_td_1, 1, 0, 1, 1)
        self.gv_fd_1 = LayoutWidget(self.tab)
        self.gv_fd_1.setObjectName("gv_fd_1")
        self.gridLayout_2.addWidget(self.gv_fd_1, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 1, 1, 1)
        self.gv_td_2 = LayoutWidget(self.tab)
        self.gv_td_2.setObjectName("gv_td_2")
        self.gridLayout_2.addWidget(self.gv_td_2, 3, 0, 1, 1)
        self.gv_fd_2 = LayoutWidget(self.tab)
        self.gv_fd_2.setObjectName("gv_fd_2")
        self.gridLayout_2.addWidget(self.gv_fd_2, 3, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 1, 2, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 160, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem3 = QtWidgets.QSpacerItem(96, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.tw_stream = QtWidgets.QTableWidget(self.tab_2)
        self.tw_stream.setMaximumSize(QtCore.QSize(231, 16777215))
        self.tw_stream.setObjectName("tw_stream")
        self.tw_stream.setColumnCount(1)
        self.tw_stream.setRowCount(28)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(16, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(17, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(18, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(19, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(20, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(21, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(22, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(23, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(24, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(25, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_stream.setVerticalHeaderItem(26, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setVerticalHeaderItem(27, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(12, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(13, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(14, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(15, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(16, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(17, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(18, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(19, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(20, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(21, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(22, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(23, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(24, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(25, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_stream.setItem(26, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_stream.setItem(27, 0, item)
        self.verticalLayout.addWidget(self.tw_stream)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(198, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem4)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.tw_acquire = QtWidgets.QTableWidget(self.tab_2)
        self.tw_acquire.setMaximumSize(QtCore.QSize(231, 16777215))
        self.tw_acquire.setObjectName("tw_acquire")
        self.tw_acquire.setColumnCount(1)
        self.tw_acquire.setRowCount(26)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_acquire.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_acquire.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_acquire.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(16, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(17, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(18, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(19, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(font)
        self.tw_acquire.setVerticalHeaderItem(20, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(21, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(22, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(23, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(24, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setVerticalHeaderItem(25, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(12, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(13, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(14, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(15, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(16, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(17, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(18, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(19, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(20, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(21, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(22, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tw_acquire.setItem(23, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(24, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(QtCore.Qt.NoItemFlags)
        self.tw_acquire.setItem(25, 0, item)
        self.verticalLayout_2.addWidget(self.tw_acquire)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        spacerItem5 = QtWidgets.QSpacerItem(126, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 996, 37))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionsave_acquire = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/save_icon_2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionsave_acquire.setIcon(icon)
        self.actionsave_acquire.setMenuRole(QtWidgets.QAction.NoRole)
        self.actionsave_acquire.setObjectName("actionsave_acquire")
        self.actionsave_stream = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/save_icon_3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionsave_stream.setIcon(icon1)
        self.actionsave_stream.setMenuRole(QtWidgets.QAction.NoRole)
        self.actionsave_stream.setObjectName("actionsave_stream")
        self.toolBar.addAction(self.actionsave_acquire)
        self.toolBar.addAction(self.actionsave_stream)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_2.setText(_translate("MainWindow", "segment size"))
        self.le_segmentsize.setText(_translate("MainWindow", "16384"))
        self.label_11.setText(_translate("MainWindow", "plot sample size"))
        self.le_plotsamplesize.setText(_translate("MainWindow", "16384"))
        self.label_3.setText(_translate("MainWindow", "stream buffer size"))
        self.le_buffersize.setText(_translate("MainWindow", "16384"))
        self.label_9.setText(_translate("MainWindow", "save buffer size"))
        self.le_savebuffersize.setText(_translate("MainWindow", "1638400"))
        self.pb_calc_ppifg.setText(_translate("MainWindow", "calculate ppifg"))
        self.pb_gage_acquire.setText(_translate("MainWindow", "GaGe Acquire"))
        self.pb_gage_stream.setText(_translate("MainWindow", "GaGe Stream"))
        self.pb_stop_gage_stream.setText(_translate("MainWindow", "Stop GaGe Stream"))
        self.cb_average.setText(_translate("MainWindow", "stream real time average"))
        self.cb_save_stream.setText(_translate("MainWindow", "save stream data"))
        self.label_10.setText(_translate("MainWindow", "stream progress"))
        self.label.setText(_translate("MainWindow", "Time Domain 1"))
        self.label_5.setText(_translate("MainWindow", "Frequency Domain 1"))
        self.label_4.setText(_translate("MainWindow", "Time Domain 2"))
        self.label_6.setText(_translate("MainWindow", "Frequency Domain 2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Oscilloscope"))
        self.label_7.setText(_translate("MainWindow", "Stream Config"))
        item = self.tw_stream.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "ACQUISITION"))
        item = self.tw_stream.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "mode"))
        item = self.tw_stream.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "samplerate"))
        item = self.tw_stream.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "depth"))
        item = self.tw_stream.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "segmentsize"))
        item = self.tw_stream.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "segmentcount"))
        item = self.tw_stream.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "triggerholdoff"))
        item = self.tw_stream.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "triggerdelay"))
        item = self.tw_stream.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "triggertimeout"))
        item = self.tw_stream.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "extclk"))
        item = self.tw_stream.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "timestampmode"))
        item = self.tw_stream.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "timestampclock"))
        item = self.tw_stream.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "CHANNEL 1"))
        item = self.tw_stream.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "range"))
        item = self.tw_stream.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "coupling"))
        item = self.tw_stream.verticalHeaderItem(15)
        item.setText(_translate("MainWindow", "impedance"))
        item = self.tw_stream.verticalHeaderItem(16)
        item.setText(_translate("MainWindow", "CHANNEL 2"))
        item = self.tw_stream.verticalHeaderItem(17)
        item.setText(_translate("MainWindow", "range"))
        item = self.tw_stream.verticalHeaderItem(18)
        item.setText(_translate("MainWindow", "coupling"))
        item = self.tw_stream.verticalHeaderItem(19)
        item.setText(_translate("MainWindow", "impedance"))
        item = self.tw_stream.verticalHeaderItem(20)
        item.setText(_translate("MainWindow", "TRIGGER"))
        item = self.tw_stream.verticalHeaderItem(21)
        item.setText(_translate("MainWindow", "condition"))
        item = self.tw_stream.verticalHeaderItem(22)
        item.setText(_translate("MainWindow", "level"))
        item = self.tw_stream.verticalHeaderItem(23)
        item.setText(_translate("MainWindow", "source"))
        item = self.tw_stream.verticalHeaderItem(24)
        item.setText(_translate("MainWindow", "STMCONFIG"))
        item = self.tw_stream.verticalHeaderItem(25)
        item.setText(_translate("MainWindow", "timeoutontransfer"))
        item = self.tw_stream.verticalHeaderItem(26)
        item.setText(_translate("MainWindow", "CUSTOM"))
        item = self.tw_stream.verticalHeaderItem(27)
        item.setText(_translate("MainWindow", "plotchecklevel"))
        item = self.tw_stream.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Parameters"))
        __sortingEnabled = self.tw_stream.isSortingEnabled()
        self.tw_stream.setSortingEnabled(False)
        item = self.tw_stream.item(1, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(2, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(3, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(4, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(5, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(6, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(7, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(8, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(9, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(10, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(11, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(13, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(14, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(15, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(17, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(18, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(19, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(21, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(22, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(23, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(25, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_stream.item(27, 0)
        item.setText(_translate("MainWindow", "hello world"))
        self.tw_stream.setSortingEnabled(__sortingEnabled)
        self.label_8.setText(_translate("MainWindow", "Acquire Config"))
        item = self.tw_acquire.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "ACQUISITION"))
        item = self.tw_acquire.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "mode"))
        item = self.tw_acquire.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "samplerate"))
        item = self.tw_acquire.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "depth"))
        item = self.tw_acquire.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "segmentsize"))
        item = self.tw_acquire.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "segmentcount"))
        item = self.tw_acquire.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "triggerholdoff"))
        item = self.tw_acquire.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "triggerdelay"))
        item = self.tw_acquire.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "triggertimeout"))
        item = self.tw_acquire.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "extclk"))
        item = self.tw_acquire.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "CHANNEL 1"))
        item = self.tw_acquire.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "range"))
        item = self.tw_acquire.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "coupling"))
        item = self.tw_acquire.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "impedance"))
        item = self.tw_acquire.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "dcoffset"))
        item = self.tw_acquire.verticalHeaderItem(15)
        item.setText(_translate("MainWindow", "CHANNEL 2"))
        item = self.tw_acquire.verticalHeaderItem(16)
        item.setText(_translate("MainWindow", "range"))
        item = self.tw_acquire.verticalHeaderItem(17)
        item.setText(_translate("MainWindow", "coupling"))
        item = self.tw_acquire.verticalHeaderItem(18)
        item.setText(_translate("MainWindow", "impedance"))
        item = self.tw_acquire.verticalHeaderItem(19)
        item.setText(_translate("MainWindow", "dcoffset"))
        item = self.tw_acquire.verticalHeaderItem(20)
        item.setText(_translate("MainWindow", "TRIGGER"))
        item = self.tw_acquire.verticalHeaderItem(21)
        item.setText(_translate("MainWindow", "condition"))
        item = self.tw_acquire.verticalHeaderItem(22)
        item.setText(_translate("MainWindow", "level"))
        item = self.tw_acquire.verticalHeaderItem(23)
        item.setText(_translate("MainWindow", "source"))
        item = self.tw_acquire.verticalHeaderItem(24)
        item.setText(_translate("MainWindow", "range"))
        item = self.tw_acquire.verticalHeaderItem(25)
        item.setText(_translate("MainWindow", "impedance"))
        item = self.tw_acquire.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Parameters"))
        __sortingEnabled = self.tw_acquire.isSortingEnabled()
        self.tw_acquire.setSortingEnabled(False)
        item = self.tw_acquire.item(1, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(2, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(3, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(4, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(5, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(6, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(7, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(8, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(9, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(11, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(12, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(13, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(14, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(16, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(17, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(18, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(19, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(21, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(22, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(23, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(24, 0)
        item.setText(_translate("MainWindow", "hello world"))
        item = self.tw_acquire.item(25, 0)
        item.setText(_translate("MainWindow", "hello world"))
        self.tw_acquire.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Settings"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionsave_acquire.setText(_translate("MainWindow", "save_acquire"))
        self.actionsave_stream.setText(_translate("MainWindow", "save_stream"))
from pyqtgraph import LayoutWidget
from . import Main_Window_rc
