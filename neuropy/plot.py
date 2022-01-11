"""Plotting tools module."""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.colors
import sklearn.metrics

import neuropy.scm


def subplot_fixed(n_rows, n_cols, ax_size,
                  x_margin=(200, 200),
                  y_margin=(200, 300),
                  ax_spacing=125):
    """Create figure with subplots of a fixed size.

    """
    # Convert lists to arrays. This makes working with them easier.
    ax_size = np.array(ax_size)
    x_margin = np.array(x_margin)
    y_margin = np.array(y_margin)

    # Determine figure width and height (in pixels)
    fw = ax_size[0] * n_cols + ax_spacing * (n_cols - 1) + x_margin.sum()
    fh = ax_size[1] * n_rows + ax_spacing * (n_rows - 1) + y_margin.sum()
    fig_size = np.array([fw, fh])

    # Convert all sizes from pixels to normalized units
    ax_size_norm = ax_size / fig_size
    x_margin_norm = x_margin / fw
    y_margin_norm = y_margin / fh
    ax_spacing_norm = ax_spacing / fig_size

    # Create figure
    fig_size_in = fig_size / matplotlib.rcParams['figure.dpi']
    fh = matplotlib.figure.Figure(figsize=fig_size_in)

    # Define positions of subplots
    x_pos = np.arange(n_cols) * (ax_size_norm[0] + ax_spacing_norm[0])
    x_pos += x_margin_norm[0]
    y_pos = np.arange(n_rows) * (ax_size_norm[1] + ax_spacing_norm[1])
    y_pos += y_margin_norm[0]
    y_pos = y_pos[::-1]  # Flip to use standard row/column ordering

    # Iterate over rows and columns to create axes
    axh = []
    for r in range(n_rows):
        rh = []  # List of axes for the current row
        for c in range(n_cols):
            # Add axis
            rect = [x_pos[c], y_pos[r], ax_size_norm[0], ax_size_norm[1]]
            rh.append(fh.add_axes(rect))

            # Format axis -- remove top and right spines
            rh[c].spines['top'].set_visible(False)
            rh[c].spines['right'].set_visible(False)

        axh.append(rh)

    return fh, axh


def get_2D_color(pos, center=(0, 0), max_radius=1):
    """Get color for 2D position based on polar coordinate system."""
    import colorsys
    import matplotlib

    verbose = False

    # Convert position to polar coordinates
    pos = np.array(pos) - np.array(center)
    theta = np.arctan2(pos[1], pos[0])  # -pi to pi
    if theta < 0:
        theta = theta + 2*np.pi  # Now 0 to 2*pi
    rho = np.sqrt(pos[0]**2 + pos[1]**2) / max_radius

    # If rho is greater than 1, adjust
    if rho > 1:
        if verbose:
            print('Warning: mapping radius exceeds maximum allowed value and has been asjusted.')
        rho = 1

    # Convert polar coordinates to HSV -- theta -> hue, rho -> saturation
    h = theta / (2*np.pi)
    s = rho
    v = 0.75
    c_rgb = colorsys.hsv_to_rgb(h, s, v)
    c_hex = matplotlib.colors.to_hex(c_rgb)

    return c_rgb


def add_figure_text(fh, info: list) -> None:
    """Add text to figure."""

    # Add list to figure in upper-left corner
    fh.text(
        0.01, 1 - 0.01,
        '\n'.join(info),
        fontsize=12,
        horizontalalignment='left',
        verticalalignment='top'
    )

    return None


def add_figure_metadata(fh, md, additional_info=None):
    """Add list of experimental info to the specified figure."""

    # Specify list of info to add
    title_str = [
        f"Experiment: {md['ophys_experiment_id']}",
        f"Structure: {md['targeted_structure']}",
        f"Depth: {md['imaging_depth_um']}",
        f"Cre line: {md['cre_line']}"
    ]

    if additional_info is not None:
        title_str.extend(additional_info)

    add_figure_text(fh, title_str)

    return None


def figure_title_from_dict(fh, info):
    """Create figure title from dictionary."""
    figure_title_list = [f'{k}: {v}' for k, v in info.items()]
    add_figure_text(fh, figure_title_list)
    return


def get_categorical_colors(category_list, color_map):
    """Create color info dict for categorical data.

    This function creates a 'color info' dict that is used in generating color-
    coded scatter plots.
    """

    # Get colors for each value in the category list
    colors = [color_map[cl] for cl in category_list]
    colors_rgba = [matplotlib.colors.to_rgba(c) for c in colors]
    colors_rgba = np.stack(colors_rgba)

    # Define colormap
    unique_values = color_map.keys()
    unique_colors = color_map.values()
    colormap_rgba = [matplotlib.colors.to_rgba(uc) for uc in unique_colors]

    # Define mapping from unique value labels to colormap indices
    value_color_map = dict(zip(unique_values, colormap_rgba))

    # Define color info dict
    color_info = {
        'colors': colors_rgba,  # List of colors for the elements in category_list
        'colormap': colormap_rgba,  # Array of rgba colors, 1/unique value
        'unique_values': unique_values,  # Unique values/conditions
        'value_color_map': value_color_map  # Dict mapping values to indices
    }

    return color_info


def get_color_dict_legend_handles(color_dict):
    """Return set of legend handles corresponding to color dictionary."""
    from matplotlib import patches
    legend_handles = [
        patches.Circle([0, 0], radius=3, color=v, label=k)
        for k, v in color_dict.items()
    ]
    return legend_handles


def categorical_color_map():
    """Returns a set of color useful for distinguishing categorical data."""
    colors = [
        '#00c0c7',
        '#5144d3',
        '#e8871a',
        '#da3490',
        '#9189fa',
        '#47e26e',
        '#277feb',
        '#6e38b1',
        '#dfbe03',
        '#cb6f10',
        '#268d6c',
        '#9bec54'
    ]
    return colors


def define_cre_line_colors_platform():
    """Define colors based on those in the platform paper."""
    col_rgb = {
        'Emx1-IRES-Cre': '#9F9F9F',
        'Slc17a7-IRES2-Cre': '#5C5C5C',
        'Cux2-CreERT2': '#A92E66',
        'Rorb-IRES2-Cre': '#7841BE',
        'Scnn1a-Tg3-Cre': '#4F63C2',
        'Nr5a1-Cre': '#5BB0B0',
        'Fezf2-CreER': '#3A6604',
        'Tlx3-Cre_PL56': '#99B20D',
        'Rbp4-Cre_KL100': '#5CAD53',
        'Ntsr1-Cre_GN220': '#FF3B39',
        'Sst-IRES-Cre': '#7B5217',
        'Vip-IRES-Cre': '#B49139',
        'Pvalb-IRES-Cre': '#E3812B',
    }
    return col_rgb


def define_cre_line_colors():
    """Returns a dict mapping structure names to colors."""
    from matplotlib import patches
    colors = categorical_color_map()
    labels = [
        'Emx1-IRES-Cre',
        'Slc17a7-IRES2-Cre',
        'Cux2-CreERT2',
        'Rorb-IRES2-Cre',
        'Scnn1a-Tg3-Cre',
        'Nr5a1-Cre',
        'Rbp4-Cre_KL100',
        'Pvalb-IRES-Cre',
        'Tlx3-Cre_PL56',
        'Ntsr1-Cre_GN220',
        'Sst-IRES-Cre',
        'Vip-IRES-Cre'
    ]
    color_dict = {l:c for l, c in zip(labels, colors)}
    legend_handles = get_color_dict_legend_handles(color_dict)
    return color_dict, legend_handles


def define_area_colors():
    """Returns a dict mapping structure names to colors."""
    from matplotlib import patches
    colors = categorical_color_map()
    labels = [
        'VISp',
        'VISl',
        'VISal',
        'VISpm',
        'VISam',
        'VISrl'
    ]
    color_dict = {l:c for l, c in zip(labels, colors)}
    legend_handles = get_color_dict_legend_handles(color_dict)
    return color_dict, legend_handles


def get_metric_colors(metric_vals, colormap_name, circular=False):
    """Get unique colors for stimuli."""
    # Get unique values and sort
    unique_vals = metric_vals.dropna()
    unique_vals = np.sort(unique_vals.unique())
    n_vals = len(unique_vals)

    # If map is circular, add 1
    if circular:
        n_vals = n_vals + 1

    # Get colormap and normalize
    cmap = cm.get_cmap(colormap_name, n_vals)
    c_lim = np.array([0, n_vals]) - 0.5
    cm_norm = matplotlib.colors.Normalize(vmin=c_lim[0], vmax=c_lim[1])
    colormap = cm.ScalarMappable(cmap=cmap, norm=cm_norm)

    # Creat hash table to convert stimulus values to colors
    if circular:
        unique_val_idx = range(0, n_vals - 1)
    else:
        unique_val_idx = range(0, n_vals)

    # Get colors corresponding to the unique values
    colormap_colors_rgba = colormap.to_rgba(unique_val_idx)

    color_dict = dict(zip(unique_vals, colormap_colors_rgba))
    colors = [
        color_dict[k] if k in color_dict.keys() else np.zeros(4)
        for k in metric_vals
    ]
    colors = np.stack(colors)

    # Define color info dict -- this contains the color information for the
    # provided metric values, as well as the mapping from colors to indices.
    color_info = {
        'colors': colors,
        'colormap': colormap_colors_rgba,
        'unique_values': unique_vals,
        'value_color_map': color_dict
    }

    return color_info


def confusion_matrix(ax, labels_actual, labels_predicted, title=None):
    """Plot confusion matrix for labelled data."""
    # Generate confusion matrix and normalize
    C = sklearn.metrics.confusion_matrix(labels_actual, labels_predicted)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    c_norm = np.max(np.sum(C, 1))
    C = C / c_norm

    # Plot
    im = ax.imshow(C, cmap=neuropy.scm.devon, norm=norm)
    ax.set_ylabel('Actual class')
    ax.set_xlabel('Predicted class')

    # Set title
    accuracy = np.sum(labels_actual == labels_predicted)/len(labels_actual)
    accuracy_str = f'Accuracy: {accuracy:0.3f}'
    if title is None:
        title_str = accuracy_str
    else:
        title_str = title + '\n' + accuracy_str
    ax.set_title(title_str)

    return im


if __name__ == '__main__':
    # Test function
    c = get_2D_color([1, 0])


