import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROBUST_PERCENTILE = 2

OPTIONS = dict()
OPTIONS['cmap_divergent'] = mpl.cm.RdBu_r
OPTIONS['cmap_sequential'] = mpl.cm.viridis


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


def line45(ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    xylimits = np.asarray([ax.get_xlim(), ax.get_ylim()]).ravel()
    newlimits = [min(xylimits), max(xylimits)]
    ax.set_aspect(1)
    ax.set_xlim(newlimits)
    ax.set_ylim(newlimits)

    kwargs.setdefault('color', 'gray')
    ax.plot(ax.get_xlim(), ax.get_ylim(), **kwargs)


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

    if x.dtype.kind == 'M':
        mask = np.isnat(x) | np.isnan(y)
    else:
        mask = np.isnan(x) | np.isnan(y)

    point = ax.plot(x[~mask][-1], y[~mask][-1], 'o', ms=4, color=color,
                    clip_on=False)
    text = ax.text(x[~mask][-1], y[~mask][-1], '  ' + label, **defaults)

    return point, text


def contour_label_spines(cs, prefix='', rotation=None, **kwargs):
    '''
    Fancy spine labelling of contours.

    Parameters
    ----------
    cs: ContourSet

    prefix: string,
        Prefix string to add to labels

    rotation : float
        Constant rotation angle for labels

    kwargs: dict, optional
        Passed on to Axes.clabel()

    Returns
    -------

    None
    '''

    def _edit_text_labels(t, ax, prefix, rotation):
        xlim = ax.get_xlim()

        if abs(t.get_position()[0] - xlim[1]) / xlim[1] < 1e-6:
            # right spine lables
            t.set_verticalalignment('center')
        else:
            # top spine labels need to be aligned to the bottom
            t.set_verticalalignment('bottom')

        t.set_clip_on(False)
        t.set_horizontalalignment('left')
        t.set_text(' ' + prefix + t.get_text())
        t.set_rotation(rotation)

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

    [_edit_text_labels(t, cs.ax, prefix, rotation) for t in cs.labelTexts]

    return clabels


def set_axes_color(ax, color, spine='left'):
    '''
    Consistently set color of axis ticks, tick labels, axis label and spine.
    '''

    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color(color)
    [tt.set_color(color) for tt in ax.get_yticklabels()]
    ax.yaxis.label.set_color(color)
    ax.tick_params('y', colors=color)


def label_subplots(ax, x=0.05, y=0.9, prefix='(', suffix=')', labels=None,
                   **kwargs):
    '''
    Alphabetically label subplots with prefix + alphabet + suffix + labels

    Inputs
    ======

    ax : matplotlib.Axes

    x, y : (optional)
        position in Axes co-ordintes

    prefix, suffix : str, optional
        prefix, suffix for alphabet

    labels : list, optional
        optional list of labels
    '''

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    if labels is not None:
        assert(len(labels) == len(ax))
    else:
        labels = [''] * len(ax)

    hdl = []
    for aa, bb, ll in zip(ax, alphabet[:len(ax)], labels):
        hdl.append(aa.text(x=x, y=y, s=prefix + bb + suffix + ' ' + ll,
                           transform=aa.transAxes, **kwargs))

    return hdl


def contour_overlimit(da, mappable, ax=None, color='w', **kwargs):

    clim = mappable.get_clim()

    ax = plt.gca() if ax is None else ax

    da.where(da <= clim[0]).contour(ax=ax, color=color, **kwargs)
    da.where(da >= clim[1]).contour(ax=ax, color=color, **kwargs)


def _determine_extend(calc_data, vmin, vmax):
    extend_min = calc_data.min() < vmin
    extend_max = calc_data.max() > vmax
    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'
    return extend


def _build_discrete_cmap(cmap, levels, extend, filled):
    """
    Build a discrete colormap and normalization of the data.
    """
    import matplotlib as mpl

    if not filled:
        # non-filled contour plots
        extend = 'max'

    if extend == 'both':
        ext_n = 2
    elif extend in ['min', 'max']:
        ext_n = 1
    else:
        ext_n = 0

    n_colors = len(levels) + ext_n - 1
    pal = _color_palette(cmap, n_colors)

    new_cmap, cnorm = mpl.colors.from_levels_and_colors(
        levels, pal, extend=extend)
    # copy the old cmap name, for easier testing
    new_cmap.name = getattr(cmap, 'name', cmap)

    return new_cmap, cnorm


def _color_palette(cmap, n_colors):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors_i = np.linspace(0, 1., n_colors)
    if isinstance(cmap, (list, tuple)):
        # we have a list of colors
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, str):
        # we have some sort of named palette
        try:
            # is this a matplotlib cmap?
            cmap = plt.get_cmap(cmap)
            pal = cmap(colors_i)
        except ValueError:
            # ValueError happens when mpl doesn't like a colormap, try seaborn
            try:
                from seaborn.apionly import color_palette
                pal = color_palette(cmap, n_colors=n_colors)
            except (ValueError, ImportError):
                # or maybe we just got a single color as a string
                cmap = ListedColormap([cmap], N=n_colors)
                pal = cmap(colors_i)
    else:
        # cmap better be a LinearSegmentedColormap (e.g. viridis)
        pal = cmap(colors_i)

    return pal


def is_scalar(value):
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    return (getattr(value, 'ndim', None) == 0)


# _determine_cmap_params is adapted from Seaborn:
# https://github.com/mwaskom/seaborn/blob/v0.6/seaborn/matrix.py#L158
# Used under the terms of Seaborn's license, see licenses/SEABORN_LICENSE.

def cmap_params(plot_data, vmin=None, vmax=None, cmap=None,
                center=None, robust=False, extend=None,
                levels=None, filled=True, norm=None):
    """
    Use some heuristics to set good defaults for colorbar and range.

    Parameters
    ==========
    plot_data: Numpy array
        Doesn't handle xarray objects

    Returns
    =======
    cmap_params : dict
        Use depends on the type of the plotting function
    """
    import matplotlib as mpl

    plot_data = np.array(plot_data)
    calc_data = np.ravel(plot_data[np.isfinite(plot_data)])
    # Setting center=False prevents a divergent cmap
    possibly_divergent = center is not False

    # Handle all-NaN input data gracefully
    if calc_data.size == 0:
        # Arbitrary default for when all values are NaN
        calc_data = np.array(0.0)

    # Set center to 0 so math below makes sense but remember its state
    center_is_none = False
    if center is None:
        center = 0
        center_is_none = True

    # Setting both vmin and vmax prevents a divergent cmap
    if (vmin is not None) and (vmax is not None):
        possibly_divergent = False

    # Setting vmin or vmax implies linspaced levels
    user_minmax = (vmin is not None) or (vmax is not None)

    # vlim might be computed below
    vlim = None

    # save state; needed later
    vmin_was_none = vmin is None
    vmax_was_none = vmax is None

    if vmin is None:
        if robust:
            vmin = np.percentile(calc_data, ROBUST_PERCENTILE)
        else:
            vmin = calc_data.min()
    elif possibly_divergent:
        vlim = abs(vmin - center)

    if vmax is None:
        if robust:
            vmax = np.percentile(calc_data, 100 - ROBUST_PERCENTILE)
        else:
            vmax = calc_data.max()
    elif possibly_divergent:
        vlim = abs(vmax - center)

    if possibly_divergent:
        # kwargs not specific about divergent or not: infer defaults from data
        divergent = ((vmin < 0) and (vmax > 0)) or not center_is_none
    else:
        divergent = False

    # A divergent map should be symmetric around the center value
    if divergent:
        if vlim is None:
            vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim

    # Now add in the centering value and set the limits
    vmin += center
    vmax += center

    # now check norm and harmonize with vmin, vmax
    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        else:
            if not vmin_was_none and vmin != norm.vmin:
                raise ValueError('Cannot supply vmin and a norm'
                                 + ' with a different vmin.')
            vmin = norm.vmin

        if norm.vmax is None:
            norm.vmax = vmax
        else:
            if not vmax_was_none and vmax != norm.vmax:
                raise ValueError('Cannot supply vmax and a norm'
                                 + ' with a different vmax.')
            vmax = norm.vmax

    # if BoundaryNorm, then set levels
    if isinstance(norm, mpl.colors.BoundaryNorm):
        levels = norm.boundaries

    # Choose default colormaps if not provided
    if cmap is None:
        if divergent:
            cmap = OPTIONS['cmap_divergent']
        else:
            cmap = OPTIONS['cmap_sequential']

    # Handle discrete levels
    if levels is not None and norm is None:
        if is_scalar(levels):
            if user_minmax:
                levels = np.linspace(vmin, vmax, levels)
            elif levels == 1:
                levels = np.asarray([(vmin + vmax) / 2])
            else:
                # N in MaxNLocator refers to bins, not ticks
                ticker = mpl.ticker.MaxNLocator(levels - 1)
                levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if extend is None:
        extend = _determine_extend(calc_data, vmin, vmax)

    if levels is not None or isinstance(norm, mpl.colors.BoundaryNorm):
        cmap, newnorm = _build_discrete_cmap(cmap, levels, extend, filled)
        norm = newnorm if norm is None else norm

    return dict(vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)


def annotate_heatmap_string(mesh, annot_data, **kwargs):
    """
    Add textual labels with the value in each cell.

    (copied from seaborn so that I can pass an array of strings).
    """
    from seaborn.utils import relative_luminance

    ax = mesh.axes
    mesh.update_scalarmappable()
    height, width = annot_data.shape
    xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)

    for x, y, m, color, ann in zip(xpos.flat, ypos.flat,
                                   mesh.get_array(), mesh.get_facecolors(),
                                   annot_data.flat):
        if m is not np.ma.masked:
            lum = relative_luminance(color)
            text_color = ".15" if lum > .408 else "w"
            text_kwargs = dict(color=text_color, ha="center", va="center")
            text_kwargs.update(**kwargs)
            ax.text(x, y, ann.decode('UTF-8'), **text_kwargs)


def fill_step(da, dim=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if dim is None:
        dim = da.dims[0]

    coord = da[dim]

    dc = coord.diff(dim).mean()
    icoord = np.linspace(-dc + coord.min(), dc + coord.max(),
                         coord.size * 50)

    # build step like array
    dai = (da.interp({dim: icoord}, method='nearest')
           .bfill(dim).ffill(dim))

    alpha = kwargs.pop('alpha', 0.3)
    color = kwargs.pop('color', None)

    line_kwargs = {'lw': 1.2}
    line_kwargs.update(**kwargs)

    hfill = ax.fill_between(icoord, dai, zorder=kwargs.pop('zorder', None),
                            color=color, alpha=alpha, linewidths=0)
    hline = dai.plot.line(x=dim, color=color, **line_kwargs)[0]

    return [hfill, hline]


def colorbar(mappable, ax=None, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl

    if ax is None:
        ax = plt.gca()

    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'vertical')
    if orientation == 'vertical':
        loc = 'right'
    elif orientation == 'horizontal':
        loc = 'bottom'

    cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
    ax.get_figure().colorbar(mappable, cax=cax, orientation=orientation)


def pow10Formatter(x, pos):
    '''
    Format color bar labels to show scientific label
    '''

    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)

    if int(np.float(a)) != 1:
        return r'${} \times 10^{{{}}}$'.format(a, b)
    else:
        return r'$10^{{{}}}$'.format(b)


def rain_colormap():

    import seaborn as sns

    cmap = (sns.blend_palette(
        [[0.988235, 0.988235, 0.992157],
         [0.811765, 0.831373, 0.886275],
         [0.627451, 0.678431, 0.788235],
         [0.521569, 0.615686, 0.729412],
         [0.584314, 0.698039, 0.749020],
         [0.690196, 0.803922, 0.772549],
         [0.847059, 0.905882, 0.796078],
         [1.000000, 0.980392, 0.756863],
         [0.996078, 0.839216, 0.447059],
         [0.996078, 0.670588, 0.286275],
         [0.992157, 0.501961, 0.219608],
         [0.968627, 0.270588, 0.152941],
         [0.835294, 0.070588, 0.125490],
         [0.674510, 0.000000, 0.149020],
         [0.509804, 0.000000, 0.149020]],
        n_colors=21, as_cmap=True))

    return cmap
