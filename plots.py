import matplotlib.pyplot as plt
import numpy as np


def linex(var, color='gray', linestyle='--', zorder=-1):

    var = np.array(var, ndmin=1)
    for vv in var:
        plt.axvline(vv, color=color, linestyle=linestyle, zorder=zorder)


def liney(var, color='gray', linestyle='--', zorder=-1):

    var = np.array(var, ndmin=1)
    for vv in var:
        plt.axhline(vv, color=color, linestyle=linestyle, zorder=zorder)


def hist(var, log=False, bins=100, alpha=0.5, normed=True, **kwargs):

    if log:
        var = np.log10(abs(var))

    plt.hist(var[np.isfinite(var)], bins=bins, alpha=alpha,
             normed=normed, **kwargs)


def line45():

    import matplotlib.pyplot as plt
    import numpy as np

    xylimits = np.asarray([plt.xlim(), plt.ylim()]).ravel()
    newlimits = [min(xylimits), max(xylimits)]
    plt.axis('square')
    plt.xlim(newlimits)
    plt.ylim(newlimits)
    plt.plot(plt.xlim(), plt.ylim(), color='gray')
