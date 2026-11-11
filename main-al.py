import argparse
import utils as u
import numpy as np
import casadi as ca
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sampler import Sampler
from conjunction import Conjunction

# from multiprocessing import Pool
# from scipy.integrate import solve_ivp
# from scipy.stats import multivariate_normal
from matplotlib.animation import FFMpegWriter

from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.function.active_learning import PredictiveVariance
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer, OptimizationResult

# set number of decimal places
np.set_printoptions(precision=10, suppress=False)

# Constants
MU:float = 3.986e14   # m^3/s^2
Re:float = 6378.135   # km
J2:float = 1082.7e-6  # J2 oblateness coefficient
epsilon:float = 1e-4  # 1e-6

writer = FFMpegWriter(fps=30)

def build_casadi_ocp(conj):
    # CasADi OCP
    opti = ca.Opti()

    # Ask casadi to be silent
    opts = {
        "ipopt.print_level": 0,   # suppress IPOPT iteration output
        "ipopt.sb": "yes",
        "print_time": 0           # suppress CasADi timing info
    }

    # Setup variables
    Tf = opti.variable()
    w1 = opti.variable(3, conj.K)
    w2 = opti.variable(3, conj.K)
    p1 = opti.variable(6, conj.K)
    p2 = opti.variable(6, conj.K)

    dx = opti.parameter(3)

    # CASE A
    # Initial conditions for Object 1
    # opti.subject_to(p1[:, 0] == ca.vertcat(conj.s1[0:3, 0] + dx, conj.s1[3:6, 0]))
    # Initial conditions for Object 2
    # opti.subject_to(p2[:, 0] == conj.s2[:, 0])
    # Objective
    # obj = 0

    # CASE B
    # Initial conditions for Object 1
    opti.subject_to(p1[:, 0] == ca.vertcat(conj.s1[0:3, 0] + dx, conj.s1[3:6, 0]))
    # Fixed velocity on Object2
    opti.subject_to(p2[3:, 0] == conj.s2[3:, 0])
    # Objective
    diff = p2[:3,0] - conj.s2[:3,0]
    obj = 0.5*epsilon*diff.T @ conj.cov_eci @ diff

    # dynamics
    def accel(r, w):
        r_norm = ca.norm_2(r)
        return -MU*r/r_norm**3 + w

    for i in range(conj.K-1):

        dt = Tf * conj.dt_scaled

        w_vec = ca.vertcat(w1[:, i], w2[:, i])
        obj += (dt / 2) * ca.dot(w_vec, w_vec)

        a11 = accel(p1[0:3, i], w1[:, i])
        a12 = accel(p1[0:3, i+1], w1[:, i+1])

        a21 = accel(p2[0:3, i], w2[:, i])
        a22 = accel(p2[0:3, i+1], w2[:, i+1])

        # Leapfrog integration
        opti.subject_to(
            p1[0:3, i+1] == p1[0:3, i] + dt*p1[3:6, i] + dt**2 * 0.5 * a11
        )
        opti.subject_to(
            p1[3:6, i+1] == p1[3:6, i] + dt * 0.5 * (a11 + a12)
        )

        opti.subject_to(
            p2[0:3, i+1] == p2[0:3, i] + dt*p2[3:6, i] + dt**2 * 0.5 * a21
        )
        opti.subject_to(
            p2[3:6, i+1] == p2[3:6, i] + dt * 0.5 * (a21 + a22)
        )

    # Time constraint
    opti.subject_to(Tf >= conj.T - 100)
    opti.subject_to(Tf <= conj.T + 100)

    # Final constraint
    opti.subject_to(
        ca.sumsqr(p2[0:3, conj.K-1] - p1[0:3, conj.K-1]) == 50**2
    )

    # Initial guesses
    opti.set_initial(p1, conj.s1)
    opti.set_initial(p2, conj.s2)
    opti.set_initial(w1, 0)
    opti.set_initial(w2, 0)
    opti.set_initial(Tf, conj.T - 99)

    # Solve
    opti.minimize(obj)
    opti.solver('ipopt', opts)
    print("done")


    # Save CasADi function
    print("Saving OCP to CasADi function: ", end='', flush=True)
    # F = opti.to_function('F', [dv], [obj])
    ocp = opti.to_function(
        'instanton_v2',            # function name
        [dx],                   # inputs (list of CasADi symbols)
        [obj,p1,p2,Tf],      # outputs (list of expressions)
        ['dx'],                 # input names
        ['J','p1','p2','Tf']    # output names
    )
    ocp.save('instanton.casadi')

    return ocp

def build_al(sampler):
    # Build Active Learning setup
    gp_setup = build_gpr(
        sampler.initial_data,
        sampler.search_space,
        likelihood_variance=1e-7
    )
    gp = GaussianProcessRegression(gp_setup)

    # Acquisition
    acq = PredictiveVariance()
    rule = EfficientGlobalOptimization(builder=acq)

    from typing import cast
    rule = cast(AcquisitionRule, rule)

    # Bayesian optimization
    print("\tGathering initial observations...", end='\n', flush=True)
    bo = BayesianOptimizer(sampler.observer, sampler.search_space)
    print("\tdone", flush=True)

    print("\tActive learning...")
    sol = bo.optimize(
        num_steps=sampler.total_iter,
        datasets=sampler.initial_data,
        models=gp,
        acquisition_rule=rule,
        track_state=True
    )
    print("Saving GP model:", end=' ', flush=True)
    sol.save("gp_model")
    print("done")

    return sol

def main(args):

    learn_gp     = args.gp
    build_casadi = args.c

    # Initialie conjunction parameters
    conj = Conjunction(sigma_r=500.0, sigma_t=1500.0, sigma_n=800.0)

    # For the Active learning of P[tau<=T]
    x_lim, y_lim, z_lim = conj.compute_search_limits(8)

    show_search_box = False
    if show_search_box:
        u.plot_box_scene(conj.cov_eci, x_lim, y_lim, z_lim)


    # CASADI
    if build_casadi:
        print("Setting up the OCP: ", end='', flush=True)
        ocp = build_casadi_ocp(conj)
        print("done")

    else:
        print("Loading CasADi function:", end=' ', flush=True)

        try:
            ocp = ca.Function.load('instanton.casadi');
            print("done")

        except FileNotFoundError:
            print("failed")
            raise FileNotFoundError("Casadi function instanton.casadi has not been created yet.")

    # ACTIVE LEARNING
    if learn_gp:
        print("Setting GP model to train:", end='\n', flush=True)
        sampler = Sampler(ocp, 3, 100, x_lim, y_lim, z_lim)
        sol = build_al(sampler)
        inference = sol.try_get_final_model()
        print("\ndone\n")

    else:
        print("Loading learned GP model:", end=' ', flush=True)
        sol = OptimizationResult.from_path("gp_model")
        inference = sol.try_get_final_model()
        print("done")

    # Grid the state space
    x_grid, y_grid, z_grid = np.mgrid[
        -x_lim:x_lim:201j,
        -y_lim:y_lim:201j,
        -z_lim:z_lim:201j
    ]
    resolution = x_grid.shape[0]
    mid = int((resolution-1)/2)
    # Convert to tensorflow tensor to evaluate the gp model in one batch
    XYZ = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)
    XYZ = tf.convert_to_tensor(XYZ, dtype=tf.float64)
    # GP evaluation
    mean, _ = inference.predict(XYZ)

    # Prior 3D Gaussian distribution
    const = 1.0/(2*np.pi*np.sqrt(np.linalg.det(conj.cov_eci)))
    s_grid_prior = u.gaussian_3d(x_grid, y_grid, z_grid, conj.cov_eci)
    p_grid_prior = const*np.exp(-0.5*s_grid_prior)

    # Posterior 3D distribution
    s_grid_post = (1/epsilon)*mean.numpy().reshape(z_grid.shape)
    p_grid_post = np.multiply(np.exp(-s_grid_post),p_grid_prior)

    # TOTAL PROBABILITY OF COLLISION
    print("MC Integration", flush=True)

    def gaussian3(cov, N):
        return np.random.multivariate_normal(
            mean=np.zeros(3),
            cov=cov,
            size=N
        )

    def mc_integration(li, p, N=30000):
        y = p(N)
        y = tf.convert_to_tensor(y, dtype=tf.float64)
        value = li.predict(y)
        return tf.reduce_mean(value)

    probability = mc_integration(inference, lambda N:gaussian3(conj.cov_eci, N))
    # probability = np.sum(p_grid_post)
    print(f"Total probability of collision: P[tau<=T]={probability}")

    p_grid_post /= float(probability)

    # RECORDINGS
    record = False
    # PLOTS
    plot_3d_ls  = False
    plot_contr  = True
    plot_surf   = False
    # Other ----------------------
    plot_samples = False
    plot_trajectories = False


    if plot_3d_ls:

        mx = np.max(p_grid_post)

        fig = go.Figure()
        fig.add_trace(go.Isosurface(
            x=x_grid[::2].flatten(),
            y=y_grid[::2].flatten(),
            z=z_grid[::2].flatten(),
            value=(1/mx)*p_grid_post[::2].flatten(),
            opacity=0.6,
            colorscale="Reds",
            caps=dict(x_show=False, y_show=False, z_show=False),
            isomin=0.2,
            isomax=0.8,
            surface_count=4,
            name="Posterior"
        ))

        mx = np.max(p_grid_prior)

        fig.add_trace(go.Isosurface(
            x=x_grid[::2].flatten(),
            y=y_grid[::2].flatten(),
            z=z_grid[::2].flatten(),
            value=(1/mx)*p_grid_prior[::2].flatten(),
            opacity=0.6,
            colorscale="Blues",
            caps=dict(x_show=False, y_show=False, z_show=False),
            isomin=0.6,
            isomax=0.7,
            surface_count=1,
            name="Prior"
        ))


        fig.update_layout(
            scene=dict(
                aspectmode='data',
                # aspectratio=dict(x=1,y=1,z=1),
            )
        )
        fig.show()


    if plot_contr:

        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

        if record:
            print("[RECORDING]", end=" ", flush=True)

            with writer.saving(fig, "ic-pdf-scan.mp4", dpi=100):

                for f in np.arange(0,resolution):

                    for ax in axes:
                        ax.clear()
                        ax.set_aspect('equal')
                        ax.set_ylabel("dz")
                    axes[0].set_xlabel("dx")
                    axes[1].set_xlabel("dy")

                    prio_xz = axes[0].contourf(
                        x_grid[:,0,:], z_grid[:,0,:], p_grid_prior[:,f,:],
                        levels=50,
                        cmap=u.contour_color_maps([0.0,0.0,1.0],50)
                    )
                    post_xz = axes[0].contourf(
                        x_grid[:,0,:], z_grid[:,0,:], p_grid_post[:,f,:],
                        levels=50,
                        cmap=u.contour_color_maps([1.0,0.0,0.0],50)
                    )
                    prio_yz = axes[1].contourf(
                        y_grid[0,:,:], z_grid[0,:,:], p_grid_prior[f,:,:],
                        levels=50,
                        cmap=u.contour_color_maps([0.0,0.0,1.0],50)
                    )
                    post_yz = axes[1].contourf(
                        y_grid[0,:,:], z_grid[0,:,:], p_grid_post[f,:,:],
                        levels=50,
                        cmap=u.contour_color_maps([1.0,0.0,0.0],50)
                    )

                    writer.grab_frame()

        else:
            # XZ PROJECTION
            prio_xz = axes[0].contourf(
                x_grid[:,mid,:], z_grid[:,mid,:], p_grid_prior[:,mid,:],
                levels=50,
                cmap=u.contour_color_maps([0.0,0.0,1.0],50)
            )
            post_xz = axes[0].contourf(
                x_grid[:,mid,:], z_grid[:,mid,:], p_grid_post[:,mid,:],
                levels=50,
                cmap=u.contour_color_maps([1.0,0.0,0.0],50)
            )
            axes[0].set_xlabel("dx")
            axes[0].set_ylabel("dz")

            # YZ PROJECTION
            prio_yz = axes[1].contourf(
                y_grid[mid,:,:], z_grid[mid,:,:], p_grid_prior[mid,:,:],
                levels=50,
                cmap=u.contour_color_maps([0.0,0.0,1.0],50)
            )
            post_yz = axes[1].contourf(
                y_grid[mid,:,:], z_grid[mid,:,:], p_grid_post[mid,:,:],
                levels=50,
                cmap=u.contour_color_maps([1.0,0.0,0.0],50)
            )
            axes[1].set_xlabel("dy")
            axes[1].set_ylabel("dz")

            for ax in axes:
                ax.set_aspect('equal')

            plt.tight_layout()
            plt.show()

    if plot_surf:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        surf1 = ax1.plot_surface(
            y_grid, z_grid, p_grid_prior,
            cmap='viridis'
        )
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        ax1.set_xlabel("dy")
        ax1.set_ylabel("dz")
        ax1.set_title("Prior")

        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection='3d')
        surf2 = ax2.plot_surface(
            y_grid, z_grid, p_grid_post,
            cmap='viridis'
        )
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        ax2.set_xlabel("dy")
        ax2.set_ylabel("dz")
        ax2.set_title("Posterior")

        plt.show()

    if plot_samples:
        samples = []
        for k in range(1000):
            sample, _= conj.sample_initial_position()
            samples.append(sample)

        samples = np.array(samples)

        # Figure ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=5, alpha=0.5)

        # Plot mean position
        # ax.scatter(conj.ic1[0], conj.ic1[1], conj.ic1[2], color='red', s=50, label='Mean')
        ax.scatter(0,0,0, color='red', s=50, label='Mean')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("LEO Position Uncertainty (Anisotropic Gaussian)")
        ax.legend()

        plt.show()

    if plot_trajectories:

        obj, p1, p2, Tf = ocp(np.zeros((3,1)))
        # p1.shape, p2.shape = (6,K)

        # Figure ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(p1[0,:], p1[1,:], p1[2,:], 'r')
        ax.plot(p2[0,:], p2[1,:], p2[2,:], 'r')
        ax.plot(conj.s1[0,:], conj.s1[1,:], conj.s1[2,:], 'r')
        ax.plot(conj.s2[0,:], conj.s2[1,:], conj.s2[2,:], 'b')
        # ax.set_box_aspect(1.0)
        # u.set_focus_3d(ax, s1[0:3,-1], 10000)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        action="store_true",
        help="Build Casadi Function (1) or not (0)"
    )

    parser.add_argument(
        "-gp",
        action="store_true",
        help="Learn a GP model (1) or use saved (0)"
    )

    args = parser.parse_args()

    # make sure to retrain GP model when rebuilding the OCP
    if args.c and not args.gp:
        print("[WARNING]: OCP might have changed, consider retraining the GP model.")

    main(args)
