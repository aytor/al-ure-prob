from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer

from src import get_run_directory

def run_active_learning(
    sampler,
    acquisition_fn,
    noise_variance,
    save=True,
    track_steps=False
):

    # Build Active Learning setup
    setup = build_gpr(
        sampler.initial_data,
        sampler.search_space,
        likelihood_variance=noise_variance
    )
    gpr = GaussianProcessRegression(setup)

    # Acquisition
    rule = EfficientGlobalOptimization(builder=acquisition_fn)

    from typing import cast
    rule = cast(AcquisitionRule, rule)

    # Bayesian optimization
    bo = BayesianOptimizer(sampler.observer, sampler.search_space)

    print("[GPR] warmup done", flush=True)
    print("[GPR] starting active learning loop")
    sol = bo.optimize(
        num_steps=sampler.active_iter,
        datasets=sampler.initial_data,
        models=gpr,
        acquisition_rule=rule,
        track_state=track_steps
    )

    if save:
        print("[GPR] saving GP model:", end=' ', flush=True)

        save_path = get_run_directory() / "models/gpr"
        save_path.mkdir(parents=True, exist_ok=True)

        sol.save(str(save_path))

        print("done")

    return sol
