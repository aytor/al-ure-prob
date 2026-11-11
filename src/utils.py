import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from itertools import product, combinations


def contour_color_maps(rgb,levels):
    # Create a red colormap with fading alpha
    n_levels = levels
    rgba = np.ones((n_levels, 4))
    rgba[:, 0] = rgb[0]   # Red
    rgba[:, 1] = rgb[1]   # Green
    rgba[:, 2] = rgb[2]   # Blue
    rgba[:, 3] = np.linspace(0.0, 0.5, n_levels)  # alpha from transparent to solid

    return ListedColormap(rgba)

def set_focus_3d(ax, center, l):
    """
    Center 3D plot around a point with symmetric limits.

    Parameters
    ----------
    ax : matplotlib 3D axis
    center : array-like (3,)
        [x, y, z] point to focus on
    l : float
        half-length of the viewing box
    """
    cx, cy, cz = center

    ax.set_xlim(cx - l, cx + l)
    ax.set_ylim(cy - l, cy + l)
    ax.set_zlim(cz - l, cz + l)

    ax.set_box_aspect([1, 1, 1])  # keep equal scaling

def plot_box_scene(cov, x_lim=1, y_lim=1, z_lim=1, ax=None):

    # generate all 8 corners of the box
    corners = np.array(list(product(
        [-x_lim, x_lim],
        [-y_lim, y_lim],
        [-z_lim, z_lim]
    )))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # draw edges: connect points that differ in only one coordinate
    for i, j in combinations(range(8), 2):
        diff = np.abs(corners[i] - corners[j])
        if np.sum(diff > 0) == 1:  # only one axis differs then draw edge
            ax.plot(*zip(corners[i], corners[j]), color='red')

    plot_covariance_ellipsoid(cov, ax=ax)
    plt.show()

    return ax


def plot_covariance_ellipsoid(cov, mean=None, n_std=2.0, ax=None):
    """
    Plot a 3D ellipsoid representing a covariance matrix.

    Parameters:
        cov (3x3 array): Covariance matrix
        mean (3 array): Center of ellipsoid (default = origin)
        n_std (float): Number of standard deviations (controls size)
        ax (matplotlib axis): Optional existing 3D axis
    """
    if mean is None:
        mean = np.zeros(3)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Radii (scaled by n_std)
    radii = n_std * np.sqrt(eigvals)

    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Stack into (3, N)
    sphere = np.stack((x, y, z), axis=0)

    # Transform sphere into ellipsoid
    ellipsoid = eigvecs @ np.diag(radii) @ sphere.reshape(3, -1)
    ellipsoid = ellipsoid.reshape(3, x.shape[0], x.shape[1])

    # Shift by mean
    ellipsoid[0] += mean[0]
    ellipsoid[1] += mean[1]
    ellipsoid[2] += mean[2]

    # Plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        ellipsoid[0], ellipsoid[1], ellipsoid[2],
        rstride=2, cstride=2, alpha=0.3, color='blue', edgecolor='k'
    )

    # Plot center
    ax.scatter(*mean, color='black')

    # Equal aspect ratio
    max_radius = np.max(radii)
    for axis in 'xyz':
        getattr(ax, f'set_{axis}lim')(mean[0] - max_radius, mean[0] + max_radius)

    # max_radius = np.max(radii)
    # ax.set_xlim(mean[0] - max_radius, mean[0] + max_radius)
    # ax.set_ylim(mean[1] - max_radius, mean[1] + max_radius)
    # ax.set_zlim(mean[2] - max_radius, mean[2] + max_radius)

    return ax


def gaussian_2d(x, y, cov):
    """
    Compute 2D Gaussian PDF with zero mean and covariance matrix cov.
    x, y: meshgrid arrays
    cov: 2x2 covariance matrix
    """
    p = np.dstack((x, y))  # shape (..., 2)

    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    const = 1.0 / (2 * np.pi * np.sqrt(cov_det))

    # quadratic form: x^T Σ^-1 x
    quad_form = np.einsum('...i,ij,...j->...', p, cov_inv, p)

    # return const * np.exp(-0.5 * quad_form)
    return quad_form


def gaussian_3d(x, y, z, cov):
    """
    Compute 3D Gaussian PDF with zero mean and covariance matrix cov.
    x, y, z: meshgrid arrays
    cov: 3x3 covariance matrix
    """
    p = np.stack((x, y, z), axis=-1)  # shape (..., 2)

    cov_inv = np.linalg.inv(cov)
    # cov_det = np.linalg.det(cov)

    # const = 1.0 / (2 * np.pi * np.sqrt(cov_det))

    # quadratic form: x^T Σ^-1 x
    quad_form = np.einsum('...i,ij,...j->...', p, cov_inv, p)

    # return const * np.exp(-0.5 * quad_form)
    return quad_form
