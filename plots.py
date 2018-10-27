import matplotlib.pyplot as plt
import numpy as np


def offset_line_plot(da, x, y, ax=None, offset=0, remove_mean=False,
                     legend=True, robust=False, **kwargs):

    assert(da[y].ndim == 1)

    axnum = da.get_axis_num(y)

    if axnum == 0:
        off = np.arange(da.shape[axnum])[:, np.newaxis]
    else:
        off = np.arange(da.shape[axnum])[np.newaxis, :]

    off *= offset

    # remove mean and add offset
    if remove_mean:
        daoffset = (da.groupby(y) - da.groupby(y).mean())
    else:
        daoffset = da

    daoffset = daoffset + off

    if axnum == 0:
        daoffset = daoffset.transpose()

    if ax is None:
        ax = plt.gca()

    hdl = ax.plot(daoffset[x], daoffset.values, **kwargs)

    if robust:
        ax.set_ylim(robust_lim(daoffset.values.ravel()))

    if legend:
        ax.legend([str(yy) for yy in da[y].values])

    return hdl


def FillRectangle(x, y=None, ax=None, color='k', alpha=0.05,
                  zorder=-1, **kwargs):
    if ax is None:
        ax = plt.gca()

    if len(x) > 2:
        raise ValueError('FillRectangle: len(x) should be 2!')

    if not isinstance(ax, list):
        ax = [ax]

    for aa in ax:
        if y is None:
            yl = aa.get_ylim()
            y = [yl[1], yl[1], yl[0], yl[0]]

        aa.fill([x[0], x[1], x[1], x[0]], y,
                color=color, alpha=alpha, zorder=zorder,
                linewidth=None, **kwargs)


def linex(var, ax=None, label=None, color='gray', linestyle='--', zorder=-1,
          **kwargs):

    if ax is None:
        ax = plt.gca()

    if not hasattr(ax, '__iter__'):
        ax = [ax]

    if label is not None and not hasattr(label, '__iter__'):
        label = [label]

    var = np.array(var, ndmin=1)
    for idx, vv in enumerate(var):
        for aa in ax:
            aa.axvline(vv, color=color, linestyle=linestyle,
                       zorder=zorder, **kwargs)
            if label is not None:
                aa.text(vv, 1, ' ' + label[idx], ha='center', va='bottom',
                        transform=aa.get_xaxis_transform('grid'))


def liney(var, ax=None, label=None, color='gray', linestyle='--', zorder=-1,
          **kwargs):

    if ax is None:
        ax = plt.gca()

    if not hasattr(ax, '__iter__'):
        ax = [ax]

    if label is not None and not hasattr(label, '__iter__'):
        label = [label]

    var = np.array(var, ndmin=1)
    for idx, vv in enumerate(var):
        for aa in ax:
            aa.axhline(vv, color=color, linestyle=linestyle,
                       zorder=zorder, **kwargs)
            if label is not None:
                aa.text(1, vv, ' ' + label[idx], ha='left', va='center',
                        transform=aa.get_yaxis_transform('grid'))


def hist(var, log=False, bins=100, alpha=0.5, normed=True,
         mark95=False, **kwargs):

    var = var.copy()
    if log:
        var = np.log10(abs(var))

    var = var[np.isfinite(var)]
    plt.hist(var, bins=bins, alpha=alpha,
             normed=normed, **kwargs)

    if mark95 is True:
        from dcpy.util import calc95
        linex(calc95(var))


def line45(ax=None):

    if ax is None:
        ax = plt.gca()

    xylimits = np.asarray([ax.get_xlim(), ax.get_ylim()]).ravel()
    newlimits = [min(xylimits), max(xylimits)]
    ax.set_aspect(1)
    ax.set_xlim(newlimits)
    ax.set_ylim(newlimits)
    ax.plot(ax.get_xlim(), ax.get_ylim(), color='gray')


def symyaxis():

    ylim = plt.gca().get_ylim()
    plt.gca().set_ylim(np.array([-1, 1]) * np.max(np.abs(ylim)))


def robust_lim(data, lotile=2, hitile=98, axis=-1):
    return [np.nanpercentile(data, lotile, axis=axis).squeeze(),
            np.nanpercentile(data, hitile, axis=axis).squeeze()]


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)


def adjust_yaxis(ax, ydif, v):
    """shift axis ax by ydiff, maintaining point v at the same location"""
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)
    ax.set_ylim(nminy + v, nmaxy + v)


def annotate_end(hdl, label, **kwargs):
    ax = hdl.axes

    y = hdl.get_ydata()
    x = hdl.get_xdata()
    color = hdl.get_color()

    defaults = {'ha': 'left', 'clip_on': False, 'color': color}
    defaults.update(**kwargs)

    point = ax.plot(x[-1], y[-1], 'o', ms=4, color=color, clip_on=False)
    text = ax.text('2014-12-31', y[-1], '  ' + label, **defaults)

    return point, text


def contour_label_spines(cs, **kwargs):
    '''
    Fancy spine labelling of contours.

    Parameters
    ----------
    cs: ContourSet

    kwargs: dict, optional
        Passed on to Axes.clabel()

    Returns
    -------

    None
    '''

    def _edit_text_labels(t, ax):
        xlim = ax.get_xlim()

        if abs(t.get_position()[0] - xlim[1]) / xlim[1] < 1e-6:
            # right spine lables
            t.set_verticalalignment('center')
        else:
            # top spine labels need to be aligned to the bottom
            t.set_verticalalignment('bottom')

        t.set_clip_on(False)
        t.set_horizontalalignment('left')
        t.set_size(kwargs.pop('fontsize', 9))
        t.set_text(' ' + t.get_text())

    # need to set these up first
    clabels = cs.ax.clabel(cs, inline=False, **kwargs)

    [txt.set_visible(False) for txt in clabels]

    for idx, _ in enumerate(cs.levels):
        if not cs.allsegs[idx]:
            continue

        # This is the rightmost point on each calculated contour
        x, y = cs.allsegs[idx][0][0, :]
        # This is a very helpful function!
        cs.add_label_near(x, y, inline=False, inline_spacing=0)

    [_edit_text_labels(t, cs.ax) for t in cs.labelTexts]
