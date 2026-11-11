import yaml
import argparse
import numpy as np
import tensorflow as tf

from src.plots import Plots
from src.conjunction import Conjunction
from src.gpr.sampler import Sampler
from src.gpr.build_al import run_active_learning
from src.ocp.build_ocp import generate_casadi_fn
from src import get_run_directory, initialize_run_directory

from pathlib import Path
#from matplotlib.animation import FFMpegWriter


# set number of decimal places
np.set_printoptions(precision=10, suppress=False)

#writer = FFMpegWriter(fps=30)

def ocp_setup(ocp_cfg, conj):

    if ocp_cfg.get("preload", True):
        import casadi as ca
        print("[OCP] loading previously saved casadi function: ", end='', flush=True)
        model_dir = get_run_directory() / "models"
        try:
            return ca.Function.load(str(model_dir / 'instanton.casadi'));
        except FileNotFoundError:
            print("failed")
            raise FileNotFoundError(f"Casadi function not found in {model_dir}.")
    else:
        print("[OCP] compiling new casadi function: ", end='', flush=True)
        return generate_casadi_fn(conj, ocp_cfg.get("save", True))


def gpr_setup(gpr_cfg, ocp, conj):

    if gpr_cfg.get("preload", True):
        from trieste.bayesian_optimizer import OptimizationResult
        print("[GPR] loading prelearned GPR model:", end=' ', flush=True)
        model_dir = get_run_directory() / "models/gpr"
        try:
            sol = OptimizationResult.from_path(model_dir)
            print(f"LOADED FROM {model_dir}", end=' ', flush=True)
            return sol.try_get_final_model()
        except FileNotFoundError:
            print("failed")
            raise FileNotFoundError(f"Gaussian process model not found in {model_dir}.")

    else:
        print("[GPR] running active learning for new GPR model:")
        x_lim, y_lim, z_lim = conj.compute_search_limits(sig=8)
        sampler = Sampler(
            ocp,
            gpr_cfg.get("init_iter", 5),
            gpr_cfg.get("active_iter", 10),
            x_lim, y_lim, z_lim
        )
        sol = run_active_learning(
            sampler,
            gpr_cfg.get("noise", False),
            gpr_cfg.get("save", True),
            gpr_cfg.get("track_steps", False)
        )
        return sol.try_get_final_model()


def load_config(path) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    return config


def main(config_path):

    config = load_config(config_path)

    name: str = config["name"]
    initialize_run_directory(run_name=name)

    gpr_cfg = config["gpr"]
    ocp_cfg = config["ocp"]
    conj_cfg = config["conjunction"]

    preload_gp  = gpr_cfg.get("preload", False)
    preload_ocp = ocp_cfg.get("preload", False)

    # make sure to retrain GP model when rebuilding the OCP
    if not preload_ocp and preload_gp:
        raise ValueError("[WARNING]: OCP might have changed, consider retraining the GP model.")

    viz_cfg = config["viz"]

    # initial conjunction parameters
    conj = Conjunction(
        epsilon=conj_cfg.get("epsilon", 1e-4),
        sigma_r=conj_cfg.get("sigma_r", 500.0),
        sigma_t=conj_cfg.get("sigma_t", 1500.0),
        sigma_n=conj_cfg.get("sigma_n", 800.0),
    )

    # domain for the active learning part
    x_lim, y_lim, z_lim = conj.compute_search_limits(sig=8)

    # CASADI
    ocp = ocp_setup(ocp_cfg, conj)
    print("done")

    # GPR ACTIVE LEARNING
    gpr = gpr_setup(gpr_cfg, ocp, conj)
    print("done")

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

    Pc = mc_integration(gpr, lambda N:gaussian3(conj.cov_eci, N))
    # probability = np.sum(p_grid_post)
    print(f"Total probability of collision: P[tau<=T]={Pc}")

    #p_grid_post /= float(probability)


    if viz_cfg.get("enabled", False):
        print("[PLOTS] enabled")
        run_plots(viz_cfg, Pc, ocp, gpr, conj, x_lim, y_lim, z_lim)


def run_plots(viz_cfg, Pc, ocp, gpr, conj, x_lim, y_lim, z_lim):

    plotter = Plots(Pc, gpr, conj, x_lim, y_lim, z_lim)
    plots_dir = get_run_directory() / "plots"

    if viz_cfg["search_box"]["show"]:
        plotter.plot_box_scene(conj.cov_eci, x_lim, y_lim, z_lim)

    if viz_cfg["trajectories"]["show"]:
        plotter.plot_trajectories(ocp)

    for plot_name, plot_cfg in viz_cfg.get("plots", {}).items():
        plot_type = plot_cfg.get("type", "contour")
        plot_grid = plot_cfg.get("grid", 200)
        plotter.generate_meshgrid(plot_grid)
        if plot_cfg.get("show", False):
            if plot_type == "contour":
                print("  2D contour plot")
                plotter.contour_posterior_density(plot_name)
            if plot_type == "surf":
                print("  2D surface plot")
                plotter.surf_posterior_density(plot_name)
            if plot_type == "3d":
                print("  3D surface plot")
                plotter.plotly3_posterior_density(plot_name)

    print(f"[PLOTS] saving plots in {plots_dir}")
    plotter.save()


def parse_args():
    parser = argparse.ArgumentParser(
        description="An LDT approach to learn Unsafe Rare Event probabilities using GPs"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
