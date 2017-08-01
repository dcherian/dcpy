import matplotlib.pyplot as plt
import numpy as np


def linex(var, ax=None, color='gray', linestyle='--', zorder=-1):

    if ax is None:
        ax = plt.gca()

    var = np.array(var, ndmin=1)
    for vv in var:
        ax.axvline(vv, color=color, linestyle=linestyle, zorder=zorder)


def liney(var, ax=None, color='gray', linestyle='--', zorder=-1):

    if ax is None:
        ax = plt.gca()

    var = np.array(var, ndmin=1)
    for vv in var:
        ax.axhline(vv, color=color, linestyle=linestyle, zorder=zorder)


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


def line45():

    import matplotlib.pyplot as plt
    import numpy as np

    xylimits = np.asarray([plt.xlim(), plt.ylim()]).ravel()
    newlimits = [min(xylimits), max(xylimits)]
    plt.axis('square')
    plt.xlim(newlimits)
    plt.ylim(newlimits)
    plt.plot(plt.xlim(), plt.ylim(), color='gray')
