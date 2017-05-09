def fit(curve, x, y, weights=None, doplot=False, **kwargs):

    if weights is None:
        import numpy as np
        weights = np.ones(x.shape)

    if curve == 'spline':
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(x, y, w=weights, check_finite=False)

    else:
        from scipy.optimize import curve_fit
        if curve == 'tanh':
            def func(x, y0, X, x0, y1):
                import numpy as np
                return y1 + y0*np.tanh((x-x0)/X)

        popt, _ = curve_fit(func, x, y, sigma=1/weights,
                            check_finite=False, **kwargs)

    if doplot:
        import matplotlib.pyplot as plt
        import numpy as np
        # plt.figure()
        plt.cla()
        plt.plot(x, y, 'k*')

        xdense = np.linspace(x.min(), x.max(), 200)
        if curve == 'spline':
            plt.plot(xdense, spl(xdense), 'r-')
        else:
            plt.plot(xdense, func(xdense, *popt), 'r-')

    if curve == 'spline':
        return spl
    else:
        return popt
