from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _share_x(ax_list):
    xmin, xmax = None, None
    a = ax_list
    try:
        arr = a.ravel()
    except Exception:
        arr = np.array([a]).ravel()
    for ax in arr:
        x0, x1 = ax.get_xlim()
        xmin = x0 if xmin is None else min(xmin, x0)
        xmax = x1 if xmax is None else max(xmax, x1)
    for ax in arr:
        ax.set_xlim(xmin, xmax)


def _share_y(ax_list):
    ymin, ymax = None, None
    a = ax_list
    try:
        arr = a.ravel()
    except Exception:
        arr = np.array([a]).ravel()
    for ax in arr:
        y0, y1 = ax.get_ylim()
        ymin = y0 if ymin is None else min(ymin, y0)
        ymax = y1 if ymax is None else max(ymax, y1)
    for ax in arr:
        ax.set_ylim(ymin, ymax)


def plot_hawkes_kernels(kernel_object, support=None, hawkes=None, n_points=300,
                        show=True, log_scale=False, min_support=1e-4, ax=None):
    if support is None or support <= 0:
        plot_supports = kernel_object.get_kernel_supports()
        support = plot_supports.max() * 1.2

    n_nodes = kernel_object.n_nodes

    if log_scale:
        x_values = np.logspace(
            np.log10(min_support), np.log10(support), n_points)
    else:
        x_values = np.linspace(0, support, n_points)

    if ax is None:
        fig, ax_list_list = plt.subplots(n_nodes, n_nodes, sharex=True,
                                         sharey=True)
    else:
        ax_list_list = ax
        show = False

    if n_nodes == 1:
        ax_list_list = np.array([[ax_list_list]])

    for i, ax_list in enumerate(ax_list_list):
        for j, ax in enumerate(ax_list):
            y_values = kernel_object.get_kernel_values(i, j, x_values)
            ax.plot(x_values, y_values, label="Kernel (%d, %d)" % (i, j))

            if hawkes is not None and hasattr(hawkes, 'kernels'):
                y_true_values = hawkes.kernels[i, j].get_values(x_values)
                ax.plot(x_values, y_true_values,
                        label="True Kernel (%d, %d)" % (i, j))

            if i == n_nodes - 1:
                ax.set_xlabel(r"$t$", fontsize=18)

            ax.set_ylabel(r"$\phi^{%g,%g}(t)$" % (i, j), fontsize=18)

            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')

            legend = ax.legend()
            for label in legend.get_texts():
                label.set_fontsize(12)

    if show:
        plt.show()

    return ax_list_list.ravel()[0].figure


def plot_hawkes_baseline_and_kernels(
        hawkes_object, kernel_support=None, hawkes=None, n_points=300,
        show=True, log_scale=False, min_support=1e-4, ax=None):
    n_nodes = hawkes_object.n_nodes

    if ax is None:
        fig, ax_list_list = plt.subplots(n_nodes, n_nodes + 1, figsize=(10, 6))
    else:
        ax_list_list = ax
        show = False

    # invoke plot_hawkes_kernels
    ax_kernels = ax_list_list[:, 1:]
    plot_hawkes_kernels(hawkes_object, support=kernel_support, hawkes=hawkes,
                        n_points=n_points, show=False, log_scale=log_scale,
                        min_support=min_support, ax=ax_kernels)
    _share_x(ax_kernels)
    _share_y(ax_kernels)

    # plot hawkes baselines
    ax_baselines = ax_list_list[:, 0]
    period = getattr(hawkes_object, 'period_length', 1.0)
    t_values = np.linspace(0, period, n_points)
    for i in range(n_nodes):
        ax = ax_baselines[i]
        ax.plot(t_values, hawkes_object.get_baseline_values(i, t_values),
                label='baseline ({})'.format(i))
        if hawkes is not None and hasattr(hawkes, 'baseline'):
            b = hawkes.baseline
            if b.ndim == 1:
                ax.plot(t_values, np.full_like(t_values, b[i]),
                        label='true baseline ({})'.format(i))
            else:
                P = getattr(hawkes, 'period_length', period)
                segs = b.shape[1]
                seg_len = P / segs
                y_true = np.empty_like(t_values)
                for k in range(t_values.shape[0]):
                    idx = int((t_values[k] % P) // seg_len)
                    if idx == segs:
                        idx = segs - 1
                    y_true[k] = b[i, idx]
                ax.step(t_values, y_true, where='post',
                        label='true baseline ({})'.format(i))
        ax.set_ylabel("$\\mu_{}(t)$".format(i), fontsize=18)

        if i == n_nodes - 1:
            ax.set_xlabel(r"$t$", fontsize=18)

        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(12)

    _share_x(ax_baselines.reshape(n_nodes, 1))
    _share_y(ax_baselines.reshape(n_nodes, 1))

    if show:
        plt.show()

    return ax_list_list.ravel()[0].figure
