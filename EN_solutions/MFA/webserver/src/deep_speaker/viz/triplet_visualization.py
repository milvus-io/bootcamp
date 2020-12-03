import logging

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def remove_values_along_axes():
    from matplotlib import pylab
    frame = pylab.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])


def get_coordinates_from_cosine_similarity(cos_sim):
    cosine_similarities = np.linspace(1, -1, 1000)
    i = np.argmin(np.square(cosine_similarities - cos_sim))
    x, y = find_all_x_y_along_circle()
    return x[i], y[i]


def find_all_x_y_along_circle():
    theta = np.linspace(0, 2 * np.pi, 1000)

    # the radius of the circle
    r = np.sqrt(1)

    # compute x1 and x2
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1, x2


def newline(p1, p2, color):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    ax = plt.gca()
    x_min = p1[0]
    x_max = p1[1]
    y_min = p2[0]
    y_max = p2[1]
    logging.info('{} {}'.format([x_min, x_max], [y_min, y_max]))
    l = mlines.Line2D([x_min, x_max], [y_min, y_max], color=color, linestyle='dashdot', linewidth=3)
    ax.add_line(l)
    return l


# plt.ion()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
xs, ys = find_all_x_y_along_circle()
fig, ax = plt.subplots(1)
ax.plot(xs, ys)
ax.set_aspect(1)

x_anchor, y_anchor = get_coordinates_from_cosine_similarity(0)  # anchor
newline([0, x_anchor], [0, y_anchor], color='blue')

x_pos, y_pos = get_coordinates_from_cosine_similarity(0.1)  # anchor
newline([0, x_pos], [0, y_pos], color='green')

x_neg, y_neg = get_coordinates_from_cosine_similarity(-0.5)  # anchor
newline([0, x_neg], [0, y_neg], color='red')

plt.legend(('', 'AnchorEx', 'PositiveEx', 'NegativeEx'), loc='lower right')

#
# for cos_sim in np.linspace(1, -1, 100):
#     x_i, y_i = get_coordinates_from_cosine_similarity(cos_sim)
#     newline([0, x_i], [0, y_i], color='green')
#     plt.draw()
#     plt.pause(0.0001)

remove_values_along_axes()
print('Save to anchor.png')
plt.savefig('anchor.png')
plt.close(fig)
