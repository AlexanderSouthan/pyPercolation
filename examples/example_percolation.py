# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

from pyPercolation import percolation


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

                
def update(frame, im_obj, axis, percol_obj, t_obj):
    if frame < percol_obj.percolation_iter[-1]:
        lbl = 'Clusters are growing'
        # label = 'Time point: {}'.format(frame)
    else:
        lbl = 'Percolating cluster formed at p={}'.format(round(percol_obj.percolation_p[-1], 2))

    # im_obj.set_data(percol_obj.lattices_surface[frame])
    im_obj.set_data(percol_obj.clusters[frame])
    axis.set_title(lbl)
    t_obj.set_text('p={}'.format(round(percol_obj.p[frame], 2)))

    return axis

percol1 = percolation(lattice='square', lattice_points_1d=[100, 100])
percol1.calc_clusters(n_p=100, n_iter=1, preserve_cluster_no=True)

fig_gif, ax_gif = plt.subplots(1, dpi=300)
fig_gif.set_tight_layout(True)
# ax_gif.set_xlabel('x pixel')
# ax_gif.set_ylabel('y pixel')
ax_gif.set_axis_off()
fig_gif.set_facecolor('white')

# make a color map of fixed colors
cmap = colors.ListedColormap(['white', 'black', 'red'])
bounds=[0, 0.5, 1.5, 2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# im_obj = ax_gif.imshow(percol1.lattices_surface[0], cmap=cmap, norm=norm)

new_cmap = rand_cmap(percol1.clusters.max()+1, type='bright', first_color_black=True, last_color_black=False, verbose=True)
new_norm = colors.BoundaryNorm(np.insert(np.linspace(0.5, percol1.clusters.max()+0.5,percol1.clusters.max()+1), 0, 0), new_cmap.N)
im_obj = ax_gif.imshow(percol1.clusters[0], cmap=new_cmap, norm=new_norm)
text_obj = ax_gif.text(1, 2, 'p={}'.format(percol1.p[0]), color='white', backgroundcolor='black')

# plt.colorbar(im_obj, boundaries=bounds, ticks=[0, 1, 2])

anim = FuncAnimation(fig_gif, update, frames=np.arange(0, len(percol1.clusters)),
                      interval=300, fargs=(im_obj, ax_gif, percol1, text_obj))
anim.save('point_square_lattice29.gif', dpi=300, writer='imagemagick')
plt.show()
