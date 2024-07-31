from .Config import Config


class VisConfig(Config):
    def _parse(self):
        self.dpi = self.cfg.get("dpi", 400)
        self.fig_size = tuple(self.cfg.get("fig_size", (10, 10)))
        self.plot_columns = self.cfg.get("plot_columns", 1)
        self.total_size = self.cfg.get("total_size", True)
        self.plot_timeseries = self.cfg.get("plot_timeseries", True)

    def items(self) -> dict:
        return {
            "fig_size": list(self.fig_size),
            "total_size": self.total_size,
            "dpi": self.dpi,
            "plot_columns": self.plot_columns,
            "plot_timeseries": self.plot_timeseries,
        }
