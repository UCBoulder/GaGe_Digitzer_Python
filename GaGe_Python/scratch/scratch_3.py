import pyqtgraph as pg


class rgvPlot(pg.widgets.RemoteGraphicsView.RemoteGraphicsView):
    def __init__(self):
        super().__init__()
        self.setConfigOptions(
            antialias=True
        )  # prettier plots at no cost to the main process!

        self.rplt = self.pg.PlotItem()
        self.rplt._setProxyOptions(deferGetattr=True)  # speeds up access to rplt.plot
        self.setCentralItem(self.rplt)


# view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
# view.pg.setConfigOptions(
#     antialias=True
# )  # prettier plots at no cost to the main process!
# rplt = view.pg.PlotItem()
# rplt._setProxyOptions(deferGetattr=True)  # speeds up access to rplt.plot
# view.setCentralItem(rplt)
