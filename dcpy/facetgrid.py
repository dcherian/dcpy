# TODO: add decorator for map_row, map_col functions
# TODO: add normal map function
# TODO: save all handles + check for mappables
# TODO: save colorbar / colormap params e.g. extend for drawing colorbars
# TODO: add facetgrid.set_defaults('lines') or 'contours'?
# TODO: add 'col' and 'row' kwargs to map_row,
#       map_col to apply to single column or row respectively
# TODO: add axes_dict

import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from xarray.core.formatting import format_item

# from xarray.plot.utils import label_from_attrs

# Overrides axes.labelsize, xtick.major.size, ytick.major.size
# from mpl.rcParams
_FONTSIZE = "small"
# For major ticks on x, y axes
_NTICKS = 5


def _nicetitle(coord, value, maxchar, template):
    """
    Put coord, value in template and truncate at maxchar
    """
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)

    if len(title) > maxchar:
        title = title[: (maxchar - 3)] + "..."

    return title


class facetgrid:
    def __init__(
        self,
        row,
        col,
        sharex=True,
        sharey=True,
        squeeze=True,
        subplot_kw={},
        gridspec_kw=dict(),
        plot_kwargs=dict(),
    ):
        try:
            self.nrows = len(row)
        except TypeError:
            self.nrows = 1

        try:
            self.ncols = len(col)
        except TypeError:
            self.ncols = 1

        self.fig, self.axes = plt.subplots(
            self.nrows,
            self.ncols,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=True,
            squeeze=False,
            gridspec_kw=gridspec_kw,
            subplot_kw=subplot_kw,
        )

        self.fig.set_constrained_layout_pads(
            w_pad=1 / 72.0, h_pad=1 / 72.0, wspace=0.01, hspace=0.01
        )

        self.x = None
        self.y = None
        self.func = None

        self.handles = dict()
        self.handles["titles"] = []
        self.handles["legend"] = []
        self.handles["colorbar"] = []
        self.handles["first"] = None

        self.kwargs = dict(add_colorbar=False, add_labels=False, **plot_kwargs)

        self._row_var = ""
        self._col_var = ""

        if isinstance(row, xr.DataArray):
            self.row = row.name
            if self.nrows > 1:
                self.row_locs = list(row.values)
            else:
                self.row_locs = row.values
        else:
            self.row = row

        if isinstance(col, xr.DataArray):
            self.col = col.name
            self.col_locs = list(col.values)
        elif isinstance(col, dict):
            assert len(col) == 1
            self.col = list(col.keys())[0]
            self.col_locs = list(col.values())
        else:
            self.col = col

        self.row_axes = dict()
        if type(row) == list:
            self.row_locs = row
            for index, rr in enumerate(row):
                self.row_axes[rr] = self.axes[index, :]

        self.col_axes = dict()
        if type(col) == list:
            self.col_locs = col
            for index, cc in enumerate(col):
                self.col_axes[cc] = self.axes[:, index]

        axes_dict = {}
        for idx, row in enumerate(self.row_locs):
            axes_dict[row] = dict(zip(self.col_locs, self.axes[idx, :]))
        self.axes_dict = axes_dict

    def _parse_x_y(self, x, y):
        x = self.x if (x is None and self.x is not None) else x
        y = self.y if (y is None and self.y is not None) else y

        if x is not None and self.x is None:
            self.x = x
        if y is not None and self.y is None:
            self.y = y

        return x, y

    def map_col(
        self, col_name, data, func=None, x=None, y=None, defaults=True, **kwargs
    ):
        if defaults:
            x, y = self._parse_x_y(x, y)

        if self.func is None and func is None:
            raise ValueError("Need to pass a plotting function to map or set self.func")

        if self.func is None:
            self.func = func

        if func is None:
            func = self.func

        if defaults and data is not None:
            kwargs = kwargs.copy()
            kwargs.update(self.kwargs)

        if data is not None:
            for ax, loc in zip(self.col_axes[col_name], self.row_locs):
                hdl = func(
                    data.sel(**{self.row: loc, "method": "nearest"}),
                    x=x,
                    y=y,
                    ax=ax,
                    **kwargs,
                )
                if self.handles["first"] is None:
                    self.handles["first"] = hdl

        else:
            for ax in self.col_axes[col_name]:
                plt.sca(ax)
                hdl = func(**kwargs)
                if self.handles["first"] is None:
                    self.handles["first"] = hdl

    def map_row(
        self, row_name, data, func=None, x=None, y=None, defaults=True, **kwargs
    ):
        if x is None or y is None:
            if defaults:
                x, y = self._parse_x_y(x, y)

        if self.func is None and func is None:
            raise ValueError("Need to pass a plotting function to map or set self.func")

        if self.func is None:
            self.func = func

        if func is None:
            func = self.func

        if defaults and data is not None:
            kwargs = kwargs.copy()
            kwargs.update(self.kwargs)

        if data is not None:
            for ax, loc in zip(self.row_axes[row_name], self.col_locs):
                if self.ncols > 1:
                    subset = data.sel(**{self.col: loc, "method": "nearest"})
                else:
                    subset = data

                hdl = func(subset, x=x, y=y, ax=ax, **kwargs)
                if self.handles["first"] is None:
                    self.handles["first"] = hdl

        else:
            for ax in self.row_axes[row_name]:
                plt.sca(ax)
                hdl = func(**kwargs)
                if self.handles["first"] is None:
                    self.handles["first"] = hdl

    def map(self, func, **kwargs):
        for ax in self.axes.flat:
            plt.sca(ax)
            func(**kwargs)

    def add_colorbar(self, **kwargs):
        if self.handles["first"] is None:
            raise ValueError("No mappables?")

        self.fig.colorbar(self.handles["first"], ax=self.axes, **kwargs)

    def set_titles(
        self, template="{value}", maxchar=30, col_names=None, row_names=None, **kwargs
    ):
        """
        Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for plot titles containing {coord} and {value}
        maxchar : int
            Truncate titles at maxchar
        kwargs : keyword args
            additional arguments to matplotlib.text

        Returns
        -------
        self: FacetGrid object

        """
        import matplotlib as mpl

        row_names = self.row_locs if row_names is None else row_names
        col_names = self.col_locs if col_names is None else col_names

        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)

        # The row titles on the right edge of the grid
        for ax, row_name in zip(self.axes[:, -1], row_names):
            title = nicetitle(coord=self.row, value=row_name, maxchar=maxchar)
            ax.annotate(
                title,
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                rotation=270,
                ha="left",
                va="center",
                **kwargs,
            )

        # The column titles on the top row
        for ax, col_name in zip(self.axes[0, :], col_names):
            title = nicetitle(coord=self.col, value=col_name, maxchar=maxchar)
            ax.set_title(title, **kwargs)

        return self

    def set_xlabels(self, label=None, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        if label is None:
            label = self.x
        for ax in self.axes[-1, :]:
            ax.set_xlabel(label, **kwargs)
        return self

    def set_ylabels(self, label=None, **kwargs):
        """Label the y axis on the left column of the grid."""
        if label is None:
            label = self.y
        for ax in self.axes[:, 0]:
            ax.set_ylabel(label, **kwargs)
        return self

    def finalize(self, col_names=None, row_names=None, xlabel=None, ylabel=None):
        self.set_titles(col_names=col_names, row_names=row_names)
        self.set_xlabels(xlabel)
        self.set_ylabels(ylabel)

    def set_row_labels(self, row_names=None, template="{value}", maxchar=30, **kwargs):
        row_names = self.row_locs if row_names is None else row_names

        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)

        # The row titles on the right edge of the grid
        for ax, row_name in zip(self.axes[:, -1], row_names):
            title = nicetitle(coord=self.row, value=row_name, maxchar=maxchar)
            ax.annotate(
                title,
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                rotation=270,
                ha="left",
                va="center",
                **kwargs,
            )

        return self

    def set_col_labels(self, col_names=None, template="{value}", maxchar=30, **kwargs):
        import matplotlib as mpl

        col_names = self.col_locs if col_names is None else col_names

        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)

        # The column titles on the top row
        for ax, col_name in zip(self.axes[0, :], col_names):
            title = nicetitle(coord=self.col, value=col_name, maxchar=maxchar)
            ax.set_title(title, **kwargs)

        return self

    def clean_ticklabels(self, row=None, col=None):
        for cc in col:
            if cc not in self.col_axes:
                raise ValueError(f"Column {cc} not found in col_axes.")

            for ax in self.col_axes[cc]:
                ax.set_yticklabels([])

    def clean_labels(self):
        # turn off visibility for x, y labels
        for aa in self.axes[:, 1:].flat:
            aa.set_ylabel("")
        for aa in self.axes[:-1, :].flat:
            aa.set_xlabel("")
        for aa in self.axes[1:, :].flat:
            aa.set_title("")
