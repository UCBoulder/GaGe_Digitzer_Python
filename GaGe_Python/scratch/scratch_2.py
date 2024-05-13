import pyqtgraph as pg

app = pg.mkQApp()
view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
pg.setConfigOptions(antialias=True)  # this will be expensive for the local plot
view.pg.setConfigOptions(
    antialias=True
)  # prettier plots at no cost to the main process!
view.setWindowTitle("pyqtgraph example: RemoteSpeedTest")

rplt = view.pg.PlotItem()
rplt._setProxyOptions(deferGetattr=True)  # speeds up access to rplt.plot
view.setCentralItem(rplt)

# view.show()
layout = pg.LayoutWidget()
# layout.addWidget(view, row=1, col=0, colspan=3)
layout.addWidget(view)
# layout.resize(800, 800)
layout.show()

if __name__ == "__main__":
    pg.exec()
