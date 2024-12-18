import matplotlib.pyplot as plt
import numpy as np


def plot_corner(
    X: np.ndarray, 
    bins: int, 
    limits: int = None, 
    diag_kws: dict = None,
    fig_kws: dict = None,
    axs=None,
    **plot_kws
) -> None:
    if diag_kws is None:
        diag_kws = dict()

    if fig_kws is None:
        fig_kws = dict()

    diag_kws.setdefault("color", "black")
    diag_kws.setdefault("lw", 1.5)

    ndim = X.shape[1]
    
    if limits is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        limits = list(zip(mins, maxs))
        limits = np.array(limits)
        limits = limits * 1.33

    if axs is None:
        fig, axs = plt.subplots(
            ncols=ndim, nrows=ndim, figsize=(8, 8), sharex=False, sharey=False, **fig_kws
        )
    for i in range(ndim):
        for j in range(ndim):
            ax = axs[i, j]
            if i == j:
                axis = i
                hist, edges = np.histogram(
                    X[:, axis], bins=bins, range=limits[axis]
                )
                hist = hist / np.max(hist)
                ax.stairs(hist, edges, **diag_kws)
            else:
                axis = (j, i)
                hist, edges = np.histogramdd(
                    X[:, axis], bins=bins, range=(limits[axis[0]], limits[axis[1]])
                )
                hist = hist / np.max(hist)
                ax.pcolormesh(edges[0], edges[1], hist.T, **plot_kws)

    for i in range(ndim):
        for ax in axs[i, 1:]:
            ax.set_yticks([])
        for ax in axs[:-1, i]:
            ax.set_xticks([])
        axs[i, i].set_ylim(0.0, 1.2)
    return axs
