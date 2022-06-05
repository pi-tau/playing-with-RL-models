import matplotlib.pyplot as plt
import numpy as np


def logPlot(figname, xs=None, funcs=[], legends=[None], labels={}, fmt=["--k"],
    lw=[0.8], figtitle=""):
    """Plot @funcs as curves on a figure and save the figure as `figname`.

    Args:
        figname (str): Full path to save the figure to a file.
        xs (list[np.Array], optional): List of arrays of x-axis data points.
            Default value is None.
        funcs (list[np.Array], optional): List of arrays of data points. Every array of
            data points from the list is plotted as a curve on the figure.
            Default value is [].
        legends (list[str], optional): A list of labels for every curve that will be
            displayed in the legend. Default value is [None].
        labels (dict, optional): A map specifying the labels of the coordinate axes.
            `labels["x"]` specifies the label of the x-axis.
            `labels["y"]` specifies the label of the y-axis.
            Default value is {}.
        fmt (list[str], optional): A list of formatting strings for every curve.
            Default value is ["--k"].
        lw (list[float], optional): A list of line widths for every curve.
            Default value is [0.8].
        figtitle (str, optional): Figure title. Default value is "".
    """
    if xs is None:
        xs = [np.arange(len(f)) for f in funcs]
    if len(legends) == 1:
        legends = legends * len(funcs)
    if len(fmt) == 1:
        fmt = fmt * len(funcs)
    if len(lw) == 1:
        lw = lw * len(funcs)

    # Set figure sizes.
    cm = 1/2.54 # cm to inch
    fontsize = 10
    fig, ax = plt.subplots(figsize=(16*cm, 12*cm), dpi=330, tight_layout={"pad":1.4})
    ax.set_title(figtitle, fontsize=fontsize, pad=2)
    ax.set_xlabel(labels.get("x"), fontsize=fontsize, labelpad=2)#, font=fpath)
    ax.set_ylabel(labels.get("y"), fontsize=fontsize, labelpad=2)#, font=fpath)
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", linestyle="--", linewidth=0.3)

    # Plot curves.
    for x, f, l, c, w in zip(xs, funcs, legends, fmt, lw):
        ax.plot(x, f, c, label=l, linewidth=w)
    ax.legend(loc="upper left", fontsize=fontsize)
    fig.savefig(figname)
    plt.close(fig)

def plot_with_averaged_curves(ys, avg_every, label, figname):
    num_iter = len(ys)

    # Define return curves.
    avg_ys = np.convolve(ys, np.ones(avg_every), mode='valid') / avg_every
    avg_ys = avg_ys[::-1][::avg_every][::-1]
    avg_ys = np.concatenate((ys[:1], avg_ys))

    # Plot curves.
    logPlot(figname=figname,
            xs=[np.arange(0, num_iter+avg_every, avg_every)],
            funcs=[avg_ys],
            legends=["avg_"+label],
            labels={"x":"Iteration", "y":label},
            fmt=["--r",],
            lw=[0.4,],
            figtitle="Agent Performance")

#