from matplotlib.colors import LinearSegmentedColormap
from numpy import nan, inf

# Used to reconstruct the colormap in viscm
parameters = {
    "xp": [
        28.782590300914933,
        60.468890475434989,
        -24.282418425088565,
        -47.188177587392218,
        -34.590010048125208,
        -8.6301496641810616,
    ],
    "yp": [
        16.928446771378731,
        -42.626527050610804,
        -74.312827225130874,
        -5.5955497382198871,
        42.5065445026178,
        39.070680628272271,
    ],
    "min_JK": 18.0859375,
    "max_JK": 95.0390625,
}
cm_data = ["w", "#5ABCE1", "#FFA500", "#ED2E00"]
# sns.palplot(sns.blend_palette(cm_data, n_colors=21))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from viscm import viscm

        viscm(test_cm)
    except ImportError:
        print("viscm not found, falling back on simple display")
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect="auto", cmap=test_cm)
    plt.show()


# sns.palplot((np.array([[255, 255, 255],
#                        [255,237,160],
#                        [254, 178, 76],
#                        [240, 59, 32]])/255.0))
