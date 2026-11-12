import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import src.utils as u

from src import get_run_directory
from matplotlib.colors import ListedColormap

class Plots:
    def __init__(
        self,
        Pc,
        gpr,
        conj,
    ):

        self._figs = []
        self._axes = []
        self._save_as = []

        self.Pc = Pc
        self.gpr = gpr
        self.conj = conj

        x_lim, y_lim, z_lim = conj.compute_search_limits()
        self.box = [x_lim, y_lim, z_lim]


    def generate_meshgrid(self, grid):

        x_lim, y_lim, z_lim = self.box

        self.x_grid, self.y_grid, self.z_grid = np.mgrid[
            -x_lim:x_lim:grid*1j,
            -y_lim:y_lim:grid*1j,
            -z_lim:z_lim:grid*1j
        ]
        self.mid = int((len(self.x_grid))/2)

        # Convert to tensorflow tensor to evaluate the gp model in one batch
        XYZ = np.stack([
            self.x_grid.ravel(),
            self.y_grid.ravel(),
            self.z_grid.ravel()],
            axis=1
        )
        XYZ = tf.convert_to_tensor(XYZ, dtype=tf.float64)

        # Prior 3D distribution
        const = 1.0/(2*np.pi*np.sqrt(np.linalg.det(self.conj.cov_eci)))
        self.s_grid_prior = u.gaussian_3d(
            self.x_grid,
            self.y_grid,
            self.z_grid,
            self.conj.cov_eci
        )
        self.p_grid_prior = const*np.exp(-0.5*self.s_grid_prior)

        # GP evaluation
        mean, _ = self.gpr.predict(XYZ)

        # Posterior 3D distribution
        self. s_grid_post = (1/self.conj.epsilon)*mean.numpy().reshape(self.z_grid.shape)
        self.p_grid_post = np.multiply(np.exp(-self.s_grid_post),self.p_grid_prior)
        self.p_grid_post /= float(self.Pc)

    def plot_trajectories(self, ocp) -> None:

        obj, p1, p2, Tf = ocp(np.zeros((3,1)))
        # p1.shape, p2.shape = (6,K)

        # Figure ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(p1[0,:], p1[1,:], p1[2,:], 'r')
        ax.plot(p2[0,:], p2[1,:], p2[2,:], 'r')
        ax.plot(self.conj.s1[0,:], self.conj.s1[1,:], self.conj.s1[2,:], 'r')
        ax.plot(self.conj.s2[0,:], self.conj.s2[1,:], self.conj.s2[2,:], 'b')
        # ax.set_box_aspect(1.0)
        # u.set_focus_3d(ax, s1[0:3,-1], 10000)

        self._figs.append(fig)
        self._axes.append(ax)
        self._save_as.append("traj")

    def plot_Q(self, name):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf1 = ax.plot_surface(
            self.x_grid, self.z_grid, self.s_grid_post,
            cmap='viridis'
        )



    def surf_posterior_density(self, name: str) -> None:

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(
            self.y_grid, self.z_grid, self.p_grid_prior,
            cmap='viridis'
        )
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        ax1.set_xlabel("dy")
        ax1.set_ylabel("dz")
        ax1.set_title("Prior")

        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(
            self.y_grid, self.z_grid, self.p_grid_post,
            cmap='viridis'
        )
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        ax2.set_xlabel("dy")
        ax2.set_ylabel("dz")
        ax2.set_title("Posterior")

        self._figs.append(fig)
        self._axes.append((ax1,ax2))
        self._save_as.append(name)


    def contour_posterior_density(self, name: str) -> None:

        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns
        prio_xz = axes[0].contourf(
            self.x_grid[:,self.mid,:],
            self.z_grid[:,self.mid,:],
            self.p_grid_prior[:,self.mid,:],
            levels=50,
            cmap=self.contour_color_maps([0.0,0.0,1.0],50)
        )
        post_xz = axes[0].contourf(
            self.x_grid[:,self.mid,:],
            self.z_grid[:,self.mid,:],
            self.p_grid_post[:,self.mid,:],
            levels=50,
            cmap=self.contour_color_maps([1.0,0.0,0.0],50)
        )
        axes[0].set_xlabel("dx")
        axes[0].set_ylabel("dz")

        # YZ PROJECTION
        prio_yz = axes[1].contourf(
            self.y_grid[self.mid,:,:],
            self.z_grid[self.mid,:,:],
            self.p_grid_prior[self.mid,:,:],
            levels=50,
            cmap=self.contour_color_maps([0.0,0.0,1.0],50)
        )
        post_yz = axes[1].contourf(
            self.y_grid[self.mid,:,:],
            self.z_grid[self.mid,:,:],
            self.p_grid_post[self.mid,:,:],
            levels=50,
            cmap=self.contour_color_maps([1.0,0.0,0.0],50)
        )
        axes[1].set_xlabel("dy")
        axes[1].set_ylabel("dz")

        for ax in axes:
            ax.set_aspect('equal')

        self._figs.append(fig)
        self._axes.append(axes)
        self._save_as.append(name)

        # plt.tight_layout()
        # plt.show()

    def scatter3_posterior_density(self, name):

        threshold1 = 0.1*self.p_grid_prior.max()
        threshold2 = 0.1*self.p_grid_post.max()

        mask1 = self.p_grid_prior > threshold1
        mask2 = self.p_grid_post > threshold2

        density1 = self.p_grid_prior[mask1]
        density2 = self.p_grid_post[mask2]

        alpha1 = density1 / density1.max()
        alpha2 = density2 / density2.max()

        alpha1 = 0.05 * alpha1
        alpha2 = 0.05 * alpha2

        blue = np.zeros((len(density1), 4))
        red  = np.zeros((len(density2), 4))

        blue[:, 2] = 0.9  # Blue
        blue[:, 3] = alpha1  # Alpha
        red[:, 0] = 1  # Red
        red[:, 3] = alpha2  # Alpha

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        prior = ax.scatter(
            self.x_grid[mask1],
            self.y_grid[mask1],
            self.z_grid[mask1],
            c=blue,
            s=5,
            depthshade=False
        )
        post = ax.scatter(
            self.x_grid[mask2],
            self.y_grid[mask2],
            self.z_grid[mask2],
            c=red,
            s=5,
            depthshade=False
        )

        # fig.colorbar(post, ax=ax, label="Probability density")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect('equal')

        self._figs.append(fig)
        self._axes.append(ax)
        self._save_as.append(name)


    def overlay_acquisition_points(self, name, data):

        for i, n in enumerate(self._save_as):
            if name == n:
                print("  overlaying acquisition points")
                querypoints = data.query_points.numpy()
                self._axes[i].scatter(
                    querypoints[:, 0],
                    querypoints[:, 1],
                    querypoints[:, 2],
                    color='black',
                    s=20,
                    marker='+',
                    label='Acquired points',
                    depthshade=False
                )

                plt.show()


    def plotly3_posterior_density(self):

        mx = np.max(self.p_grid_post)

        fig = go.Figure()
        fig.add_trace(go.Isosurface(
            x=self.x_grid[::2].flatten(),
            y=self.y_grid[::2].flatten(),
            z=self.z_grid[::2].flatten(),
            value=(1/mx)*self.p_grid_post[::2].flatten(),
            opacity=0.6,
            colorscale="Reds",
            caps=dict(x_show=False, y_show=False, z_show=False),
            isomin=0.2,
            isomax=0.8,
            surface_count=4,
            name="Posterior"
        ))

        mx = np.max(self.p_grid_prior)

        fig.add_trace(go.Isosurface(
            x=self.x_grid[::2].flatten(),
            y=self.y_grid[::2].flatten(),
            z=self.z_grid[::2].flatten(),
            value=(1/mx)*self.p_grid_prior[::2].flatten(),
            opacity=0.6,
            colorscale="Blues",
            caps=dict(x_show=False, y_show=False, z_show=False),
            isomin=0.2,
            isomax=0.8,
            surface_count=4,
            name="Prior"
        ))

        fig.update_layout(
            scene=dict(
                aspectmode='data',
                # aspectratio=dict(x=1,y=1,z=1),
            )
        )
        fig.show()


    def set_focus_3d(self, ax, center, l):
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


    def plot_box_scene(self, cov, x_lim=1, y_lim=1, z_lim=1, ax=None):

        from itertools import product, combinations

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

        self.plot_covariance_ellipsoid(cov, ax=ax)
        plt.show()

        return ax


    def plot_covariance_ellipsoid(self, cov, mean=None, n_std=2.0, ax=None):
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


    def contour_color_maps(self, rgb,levels):
        # Create a red colormap with fading alpha
        n_levels = levels
        rgba = np.ones((n_levels, 4))
        rgba[:, 0] = rgb[0]   # Red
        rgba[:, 1] = rgb[1]   # Green
        rgba[:, 2] = rgb[2]   # Blue
        rgba[:, 3] = np.linspace(0.0, 0.5, n_levels)  # alpha from transparent to solid

        return ListedColormap(rgba)


    def save(self, plots_dir) -> None:

        if len(self._figs) == 0:
            return

        print(f"[PLOTS] saving plots in {plots_dir}")

        if len(self._save_as) != len(self._figs):
            raise ValueError(f"[PLOTS] number of plot names ({len(self._save_as)}) does not match number of figures ({len(self._figs)})")

        plot_path = get_run_directory() / "plots"
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        for name, fig in zip(self._save_as, self._figs):
            fig.savefig(str(plot_path / name))
            plt.close(fig)
