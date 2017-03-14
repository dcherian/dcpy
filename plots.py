def line45():
    import matplotlib.pyplot as plt
    import numpy as np

    xylimits = np.asarray([plt.xlim(), plt.ylim()]).ravel()
    newlimits = [min(xylimits), max(xylimits)]
    plt.axis('square')
    plt.xlim(newlimits)
    plt.ylim(newlimits)
    plt.plot(plt.xlim(), plt.ylim(), color='gray')
