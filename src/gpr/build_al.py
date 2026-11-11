from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.function.active_learning import PredictiveVariance
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer

from src import get_run_directory

def run_active_learning(
    sampler,
    noise=1e-7,
    save=True,
    track_steps=False
):

    noise = 1e-7 if not noise else noise

    # Build Active Learning setup
    setup = build_gpr(
        sampler.initial_data,
        sampler.search_space,
        likelihood_variance=noise
    )
    gpr = GaussianProcessRegression(setup)

    # Acquisition
    acq = PredictiveVariance()
    rule = EfficientGlobalOptimization(builder=acq)

    from typing import cast
    rule = cast(AcquisitionRule, rule)

    # Bayesian optimization
    bo = BayesianOptimizer(sampler.observer, sampler.search_space)

    print("[GPR] warmup done", flush=True)
    print("[GPR] starting active learning loop")
    sol = bo.optimize(
        num_steps=sampler.total_iter,
        datasets=sampler.initial_data,
        models=gpr,
        acquisition_rule=rule,
        track_state=track_steps
    )

    if save:
        print("Saving GP model:", end=' ', flush=True)

        save_path = get_run_directory() / "models/gpr"
        save_path.mkdir(parents=True, exist_ok=True)

        sol.save(str(save_path))

        print("done")

    return sol
