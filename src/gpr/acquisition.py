import tensorflow as tf

from trieste.acquisition.interface import (
    SingleModelAcquisitionBuilder,
    AcquisitionFunctionClass,
    AcquisitionFunction,
)
from trieste.models.interfaces import SupportsPredictJoint


try:  # recent trieste versions
    from trieste.models.interfaces import SupportsCovarianceBetweenPoints
except ImportError:  # older versions
    from trieste.models.interfaces import (
        SupportsPredictJoint as SupportsCovarianceBetweenPoints,
    )

class ProbabilityVarianceReduction(
    SingleModelAcquisitionBuilder[SupportsPredictJoint]
):

    def __init__(
        self,
        integration_points,
        noise_variance,
        epsilon,
        include_variance=False
    ):

        self._integration_points = tf.convert_to_tensor(integration_points)
        self._include_variance = include_variance
        self._epsilon = tf.cast(epsilon, self._integration_points.dtype)
        self._noise = tf.cast(noise_variance, self._integration_points.dtype)


    def prepare_acquisition_function(
        self, model, dataset=None
    ) -> AcquisitionFunction:

        return LinearizedDoubleIntegralAcq(
            model, self._integration_points, self._epsilon, self._include_variance
        )

    def update_acquisition_function(
        self, function, model, dataset=None
    ) -> AcquisitionFunction:

        function.update()
        return function


class LinearizedDoubleIntegralAcq(AcquisitionFunctionClass):

    def __init__(
        self, model, points, epsilon, include_variance
    ):

        self._model = model
        self._points = points
        self._epsilon = epsilon
        self._include_variance = include_variance

        num_points = tf.shape(points)[0]
        self._weights = tf.Variable(
            tf.zeros([num_points], dtype=points.dtype), trainable=False
        )
        # self.effective_sample_size = None
        self.update()

    def update(self) -> None:
        mean, var = self._model.predict(self._points)  # [M, 1] each
        log_m = -mean[:, 0] / self._epsilon
        if self._include_variance:
            log_m += var[:, 0] / (2.0 * self._epsilon**2)

        # softmax = exp(log_m) / sum(exp(log_m)), computed without underflow
        weights = tf.nn.softmax(log_m)
        self._weights.assign(weights)

        # ESS = 1 / sum(w^2): how many integration points carry p_c.
        # self.effective_sample_size = float(1.0 / tf.reduce_sum(weights**2))
        # print(f"[ACQ] effective sample size: {self.effective_sample_size:.1f}")

    @tf.function
    def __call__(self, x):
        # Trieste passes candidates with shape [N, 1, D] (batch size 1).
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch size 1.",
        )
        z = tf.squeeze(x, axis=-2)  # [N, D]

        # Posterior covariance k_n(x_j, z): [L=1, M, N] -> [M, N]
        cov = self._model.covariance_between_points(self._points, z)[0]

        # sum_j w_j k_n(x_j, z) for all candidates at once: [N]
        integral = tf.linalg.matvec(cov, self._weights, transpose_a=True)

        _, var_z = self._model.predict(z)  # latent variance s_n^2(z), [N, 1]
        noise = self._model.get_observation_noise()  # nugget lambda

        return (integral**2 / (var_z[:, 0] + noise))[:, None]  # [N, 1]
