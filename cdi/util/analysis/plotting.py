import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.collections import PatchCollection


class HandlerTupleVertical(HandlerTuple):
    """
    Create legend entries with tuples of shapes, e.g. two lines with different styles
    These two lines will be stacked on top of each other in the legend item
    From https://stackoverflow.com/a/40363560/5612427
    """
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (2*i + 1)*height_y,
                                             width,
                                             2*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines


def plot_comparison(ax, stats_dict, x_metric_key, y_metric_keys, err_keys,
                    xlabel, ylabel,
                    x_metric_dim=None, y_metric_dim=None, err_metric_dim=None,
                    metric_labels=None, ylim=None, legend=True,
                    v_lines=None,
                    legend_loc='best', legend_anchor=None, markerfirst=True,
                    inset_axis=None, inset_lims=(None, None, None, None),
                    colors=None, styles=None, colors_instead_styles=False,
                    x_jitter=None, no_label=False, ignore_nan=False):

    assert len(y_metric_keys) == len(err_keys),\
        'Number of metric keys and error keys should be the same!'
    assert metric_labels is None or len(metric_labels) == len(y_metric_keys),\
        'Number of metrics and metric labels should be the same!'

    # Different colors to be used for different methods in stats_dict
    if colors is None:
        colors = list(plt.cm.tab10(np.arange(10)))
        colors += ["indigo", 'teal', 'lime', 'grey', 'black']
    ax.set_prop_cycle('color', colors)
    # Different styles to be used for different metric_keys
    if styles is None:
        styles = ['-', '--', ':', '-.']

    if inset_axis is not None:
        axins = ax.inset_axes(inset_axis)
        axins.set_prop_cycle('color', colors)

    for i, (model, experiments) in enumerate(stats_dict.items()):
        # color = next(ax._get_lines.prop_cycler)['color']
        color = colors[i % len(colors)]
        # Plot the v-line
        if v_lines is not None and v_lines[model] is not None:
            ax.axvline(v_lines[model], 0, 1, color=color, alpha=0.3)
        # Plot main graph
        for s, (y_metric_key, err_key) in enumerate(zip(y_metric_keys,
                                                        err_keys)):
            if y_metric_key not in experiments:
                continue
            if not colors_instead_styles:
                style = styles[s % len(styles)]
            else:
                style = styles[0]
                color = colors[s % len(colors)]

            # Get label for current curve
            if metric_labels is not None:
                if not colors_instead_styles:
                    label = f'{model} {metric_labels[s]}'
                else:
                    label = metric_labels[s]
            elif s == 0:
                label = model
            else:
                label = None

            if no_label:
                label = None

            # Get error metric
            if err_key in experiments and err_metric_dim is None:
                y_err = experiments[err_key]
            elif err_key in experiments and err_metric_dim is not None:
                y_err = experiments[err_key][:, err_metric_dim]
            else:
                y_err = None

            # Get y metric
            if x_metric_dim is None:
                x = experiments[x_metric_key]
            else:
                x = experiments[x_metric_key][:, x_metric_dim]

            # Get y metric
            if y_metric_dim is None:
                y = experiments[y_metric_key]
            else:
                y = experiments[y_metric_key][:, y_metric_dim]

            if x_jitter is not None:
                if len(y_metric_keys) > 1:
                    x = np.array(x) + x_jitter[s]
                else:
                    x = np.array(x) + x_jitter[i]

            eb = ax.errorbar(x=x,
                             y=y,
                             yerr=y_err,
                             linestyle=style, color=color,
                             capsize=5, elinewidth=2,
                             label=label,
                             marker='x' if np.sum(np.isnan(y)) > 0 and not ignore_nan else None )
            if y_err is not None:
                eb[-1][0].set_linestyle(style)
            # Add inset lines
            if inset_axis is not None:
                inseb = axins.errorbar(
                           x=x,
                           y=y,
                           yerr=y_err,
                           linestyle=style, color=color,
                           capsize=5, elinewidth=2,
                           label=label)
                if y_err is not None:
                    inseb[-1][0].set_linestyle(style)

    if inset_axis is not None:
        axins.set_xlim(left=inset_lims[0], right=inset_lims[1])
        axins.set_ylim(bottom=inset_lims[2], top=inset_lims[3])
        axins.grid()
        # axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)

    if legend:
        ax.legend(loc=legend_loc,
                  bbox_to_anchor=legend_anchor,
                  prop={'size': 11},  # 'weight':'bold'},
                #   framealpha=0.0,
                  markerfirst=markerfirst)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    # if inset_axis is not None:
    #     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0", alpha=1)
    ax.grid()
    return ax


def plot_seaborn_comparison(ax, stats_dict, x_metric_key, y_metric_keys,
                            xlabel, ylabel,
                            x_metric_dim=None, y_metric_dim=None,
                            metric_labels=None, ylim=None, legend=True,
                            v_lines=None,
                            legend_loc='best', legend_anchor=None, markerfirst=True,
                            inset_axis=None, inset_lims=(None, None, None, None),
                            colors=None, styles=None):
    assert metric_labels is None or len(metric_labels) == len(y_metric_keys),\
        'Number of metrics and metric labels should be the same!'

    # Different colors to be used for different methods in stats_dict
    if colors is None:
        colors = list(plt.cm.tab10(np.arange(10)))
        colors += ["indigo", 'teal', 'lime', 'grey', 'black']
    ax.set_prop_cycle('color', colors)
    # Different styles to be used for different metric_keys
    if styles is None:
        styles = ['-', '--', ':', '-.']

    if inset_axis is not None:
        axins = ax.inset_axes(inset_axis)
        axins.set_prop_cycle('color', colors)

    for i, (model, experiments) in enumerate(stats_dict.items()):
        # color = next(ax._get_lines.prop_cycler)['color']
        color = colors[i % len(colors)]
        # Plot the v-line
        if v_lines is not None and v_lines[model] is not None:
            ax.axvline(v_lines[model], 0, 1, color=color, alpha=0.3)
        # Plot main graph
        for s, y_metric_key in enumerate(y_metric_keys):
            if y_metric_key not in experiments:
                continue
            # style = styles[s % len(styles)]
            # Get label for current curve
            if metric_labels is not None:
                label = f'{model} {metric_labels[s]}'
            elif s == 0:
                label = model
            else:
                label = None

            # Get y metric
            if x_metric_dim is None:
                x = experiments[x_metric_key]
            else:
                x = experiments[x_metric_key][:, x_metric_dim]

            # Get y metric
            if y_metric_dim is None:
                y = experiments[y_metric_key]
            else:
                y = experiments[y_metric_key][:, y_metric_dim]

            # Create dataframe for seaborn
            x = np.concatenate(x)
            y = np.concatenate(y)
            pdict = {x_metric_key: x, y_metric_key: y}
            df = pd.DataFrame.from_dict(pdict)
            # print(df)

            sns.lineplot(ax=ax,
                         x=x_metric_key,
                         y=y_metric_key,
                         data=df,
                        #  style=style,
                         color=color,
                         label=label)
            # eb = ax.errorbar(x=x,
            #                  y=y,
            #                  yerr=y_err,
            #                  linestyle=style, color=color,
            #                  capsize=5, elinewidth=2,
            #                  label=label)
            # # Add inset lines
            # if inset_axis is not None:
            #     inseb = axins.errorbar(
            #                x=x,
            #                y=y,
            #                yerr=y_err,
            #                linestyle=style, color=color,
            #                capsize=5, elinewidth=2,
            #                label=label)
            #     if y_err is not None:
            #         inseb[-1][0].set_linestyle(style)

    if inset_axis is not None:
        axins.set_xlim(left=inset_lims[0], right=inset_lims[1])
        axins.set_ylim(bottom=inset_lims[2], top=inset_lims[3])
        axins.grid()
        # axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)

    if legend:
        ax.legend(loc=legend_loc,
                  bbox_to_anchor=legend_anchor,
                  prop={'size': 11},  # 'weight':'bold'},
                #   framealpha=0.0,
                  markerfirst=markerfirst)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    # if inset_axis is not None:
    #     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0", alpha=1)
    ax.grid()
    return ax


def barplot_comparison(ax, stats_dict, x_metric_key, y_metric_keys, err_keys,
                       xlabel, ylabel, width, gap=None, gap_idxs=None,
                       x_metric_dim=None, y_metric_dim=None, err_metric_dim=None,
                       y_bott_keys=None, x_offset = 0,
                       metric_labels=None, ylim=None, legend=True,
                       v_lines=None,
                       legend_loc='best', legend_anchor=None, markerfirst=True,
                       inset_axis=None, inset_lims=(None, None, None, None),
                       colors=None, edgecolors=None, styles=None, colors_instead_styles=False,
                       no_label=False, hatches=None, alpha=1.):

    assert len(y_metric_keys) == len(err_keys),\
        'Number of metric keys and error keys should be the same!'
    assert metric_labels is None or len(metric_labels) == len(y_metric_keys),\
        'Number of metrics and metric labels should be the same!'

    # Different colors to be used for different methods in stats_dict
    if colors is None:
        colors = list(plt.cm.tab10(np.arange(10)))
        colors += ["indigo", 'teal', 'lime', 'grey', 'black']
    # ax.set_prop_cycle('color', colors)
    # Different styles to be used for different metric_keys
    if styles is None:
        styles = ['-', '--', ':', '-.']

    if inset_axis is not None:
        axins = ax.inset_axes(inset_axis)
        axins.set_prop_cycle('color', colors)

    num_models = len(stats_dict)

    for i, (model, experiments) in enumerate(stats_dict.items()):
        # color = next(ax._get_lines.prop_cycler)['color']
        color = colors[i % len(colors)]

        hatch = None
        if hatches is not None:
            hatch = hatches[i]
        # Plot the v-line
        if v_lines is not None and v_lines[model] is not None:
            ax.axvline(v_lines[model], 0, 1, color=color, alpha=0.3)
        # Plot main graph
        for s, (y_metric_key, err_key) in enumerate(zip(y_metric_keys,
                                                        err_keys)):
            if y_metric_key not in experiments:
                continue
            if not colors_instead_styles:
                style = styles[s % len(styles)]
            else:
                style = styles[0]
                color = colors[s % len(colors)]

            # Get label for current curve
            if metric_labels is not None:
                if not colors_instead_styles:
                    label = f'{model} {metric_labels[s]}'
                else:
                    label = metric_labels[s]
            elif s == 0:
                label = model
            else:
                label = None

            if no_label:
                label = None

            # Get error metric
            if err_key in experiments and err_metric_dim is None:
                y_err = experiments[err_key]
            elif err_key in experiments and err_metric_dim is not None:
                y_err = experiments[err_key][:, err_metric_dim]
            else:
                y_err = None

            # Get y metric
            if x_metric_dim is None:
                x = experiments[x_metric_key]
            else:
                x = experiments[x_metric_key][:, x_metric_dim]
            x = np.arange(1, len(x)+1)

            # Get y metric
            if y_metric_dim is None:
                y = experiments[y_metric_key]
            else:
                y = experiments[y_metric_key][:, y_metric_dim]

            # Get y-bottom
            y_bott = None
            if y_bott_keys is not None:
                y_bott_key = y_bott_keys[s]
                if y_metric_dim is None:
                    y_bott = experiments[y_bott_key]
                else:
                    y_bott = experiments[y_bott_key][:, y_metric_dim]

            if gap is None:
                offset = (width*i) - (num_models*width)/2
            else:
                offset = (width*i) - (num_models*width+gap*len(gap_idxs))/2
                offset += gap*np.count_nonzero(i > np.array(gap_idxs))
            eb = ax.bar(x=x+offset+x_offset,
                        height=y,
                        width=width,
                        yerr=y_err,
                        bottom=y_bott,
                        linestyle=style, facecolor=color,
                        edgecolor=edgecolors[i] if edgecolors is not None else None,
                        linewidth=1,
                        label=label,
                        error_kw=dict(capsize=5, elinewidth=2),
                        align='edge',
                        hatch=hatch,
                        alpha=alpha)
            # Add inset lines
            if inset_axis is not None:
                inseb = axins.bar(
                           x=x+offset+x_offset,
                           height=y,
                           width=width,
                           yerr=y_err,
                           bottom=y_bott,
                           linestyle=style, facecolor=color,
                           edgecolor=edgecolors[i] if edgecolors is not None else None,
                           label=label,
                           error_kw=dict(capsize=5, elinewidth=2),
                           align='edge',
                           hatch=hatch,
                           alpha=alpha)

    if inset_axis is not None:
        axins.set_xlim(left=inset_lims[0], right=inset_lims[1])
        axins.set_ylim(bottom=inset_lims[2], top=inset_lims[3])
        axins.grid()
        # axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)

    if legend:
        ax.legend(loc=legend_loc,
                  bbox_to_anchor=legend_anchor,
                  prop={'size': 11},  # 'weight':'bold'},
                #   framealpha=0.0,
                  markerfirst=markerfirst)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    # if inset_axis is not None:
    #     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0", alpha=1)
    ax.grid()
    return ax


def plot_seaborn_violin_comparison(ax, stats_dict, x_metric_key, y_metric_keys,
                                   xlabel, ylabel,
                                   x_metric_dim=None, y_metric_dim=None, y_hue_key=None,
                                   metric_labels=None, ylim=None, legend=True,
                                   v_lines=None,
                                   legend_loc='best', legend_anchor=None, markerfirst=True,
                                   inset_axis=None, inset_lims=(None, None, None, None),
                                   colors=None, styles=None, colors_instead_styles=False,
                                   band_width=None, width=None, upper_boundaries=None,
                                   inner='box', gridsize=100, cut=1.0, scale='area', scale_hue=True, dodge=True):
    assert metric_labels is None or len(metric_labels) == len(y_metric_keys),\
        'Number of metrics and metric labels should be the same!'

    # Different colors to be used for different methods in stats_dict
    if colors is None:
        colors = list(plt.cm.tab10(np.arange(10)))
        colors += ["indigo", 'teal', 'lime', 'grey', 'black']
    ax.set_prop_cycle('color', colors)
    # Different styles to be used for different metric_keys
    if styles is None:
        styles = ['-', '--', ':', '-.']

    if inset_axis is not None:
        axins = ax.inset_axes(inset_axis)
        axins.set_prop_cycle('color', colors)

    for i, (model, experiments) in enumerate(stats_dict.items()):
        # color = next(ax._get_lines.prop_cycler)['color']
        color = colors[i % len(colors)]
        # Plot the v-line
        if v_lines is not None and v_lines[model] is not None:
            ax.axvline(v_lines[model], 0, 1, color=color, alpha=0.3)
        # Plot main graph
        for s, y_metric_key in enumerate(y_metric_keys):
            if y_metric_key not in experiments:
                continue
            if not colors_instead_styles:
                style = styles[s % len(styles)]
            else:
                style = styles[0]
                color = colors[s % len(colors)]
            # style = styles[s % len(styles)]
            # Get label for current curve
            label = None
            if metric_labels is not None:
                # if not colors_instead_styles:
                label = f'{model} {metric_labels[s]}'
            elif s == 0:
                label = model
            else:
                label = None

            # Get y metric
            x = experiments[x_metric_key]
            if x_metric_dim is not None:
                x = x[:, x_metric_dim]

            # Get y metric
            y = experiments[y_metric_key]
            if y_metric_dim is not None:
                y = y[:, y_metric_dim]

            if y_hue_key is not None:
                y_hue = experiments[y_hue_key]
                if y_metric_dim is not None:
                    y_hue = y_hue[:, y_metric_dim]

            # Create dataframe for seaborn
            y = np.concatenate(y)
            if isinstance(x, (list, tuple)):
                x = np.concatenate(x)
            x_order = np.arange(x.size+1)
            if x.size < y.size:
                # Should be a multiple
                assert y.size % x.size == 0, 'y is not multiple of x'
                # x = np.tile(x, int(y.size / x.size))
                x = x.repeat(int(y.size / x.size))
            pdict = {x_metric_key: x, y_metric_key: y}
            if y_hue_key is not None:
                pdict[y_hue_key] = np.concatenate(y_hue)
            df = pd.DataFrame.from_dict(pdict)

            palette = None
            if y_hue_key is not None:
                df[y_hue_key] = df[y_hue_key].astype('category')
                palette = {0: colors[0], 1: colors[1]}

            if upper_boundaries is not None:
                df = df.loc[df[y_metric_key] < upper_boundaries]

            if width is not None:
                sns.violinplot(ax=ax,
                               x=x_metric_key,
                               y=y_metric_key,
                               hue=y_hue_key,
                               data=df,
                               order=x_order,
                               color=color,
                               palette=palette,
                               label=label,
                               bw=band_width,
                               width=width,
                               inner=inner,
                               gridsize=gridsize,
                               cut=cut,
                               scale=scale,
                               scale_hue=scale_hue,
                               dodge=dodge,
                               split=y_hue_key is not None)
            else:
                sns.violinplot(ax=ax,
                               x=x_metric_key,
                               y=y_metric_key,
                               hue=y_hue_key,
                               data=df,
                               order=x_order,
                               color=color,
                               palette=palette,
                               label=label,
                               bw=band_width,
                               inner=inner,
                               gridsize=gridsize,
                               cut=cut,
                               scale=scale,
                               scale_hue=scale_hue,
                               dodge=dodge,
                               split=y_hue_key is not None)

    if inset_axis is not None:
        axins.set_xlim(left=inset_lims[0], right=inset_lims[1])
        axins.set_ylim(bottom=inset_lims[2], top=inset_lims[3])
        axins.grid()
        # axins.set_xticklabels('')
        axins.set_yticklabels('')
        axins.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)

    if legend:
        ax.legend(loc=legend_loc,
                  bbox_to_anchor=legend_anchor,
                  prop={'size': 11},  # 'weight':'bold'},
                #   framealpha=0.0,
                  markerfirst=markerfirst)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    # if inset_axis is not None:
    #     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0", alpha=1)
    ax.grid()
    return ax


def plot_chains(ax, epochs, chains, labels, ylim=None,
                legend_loc='best', legend_anchor=None, markerfirst=True,
                colors=None):
    # Different colors to be used for different chains
    if colors is None:
        colors = list(plt.cm.tab10(np.arange(10)))
        colors += ["indigo", 'teal', 'lime', 'grey', 'black']
    ax.set_prop_cycle('color', colors)

    # Plot chains
    for i in range(chains.shape[-1]):
        # color = next(ax._get_lines.prop_cycler)['color']
        color = colors[i % len(colors)]
        ax.plot(epochs,
                chains[:, i],
                color=color,
                label=labels[i])

    ax.legend(loc=legend_loc,
              bbox_to_anchor=legend_anchor,
              prop={'size': 11},  # 'weight':'bold'},
              framealpha=0.0,
              markerfirst=markerfirst)

    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    ax.set_ylabel('Value', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.grid()
    return ax


def plot_colorline(
        ax, x, y, z=None, cmap=None, norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, zorder=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, zorder=zorder)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.

    From https://gist.github.com/hugke729/78655b82b885cde79e270f1c30da0b5f

    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height,
                           facecolor=c,
                           edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


# define an object that will be used by the legend
class TwocolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        assert len(self.colors) == 2


# define a handler for the MulticolorPatch object
class TwocolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        top_left = plt.Polygon([[-handlebox.xdescent, -handlebox.ydescent],
                                [-handlebox.xdescent, -handlebox.ydescent+height],
                                [-handlebox.xdescent+width, -handlebox.ydescent+height]],
                               closed=True,
                               facecolor=orig_handle.colors[0],
                               edgecolor='none')
        bottom_right = plt.Polygon([[-handlebox.xdescent+width, -handlebox.ydescent],
                                    [-handlebox.xdescent, -handlebox.ydescent],
                                    [-handlebox.xdescent+width, -handlebox.ydescent+height]],
                                   closed=True,
                                   facecolor=orig_handle.colors[1],
                                   edgecolor='none')

        patch = PatchCollection([top_left, bottom_right], match_original=True)

        handlebox.add_artist(patch)
        return patch
