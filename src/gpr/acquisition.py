import tensorflow as tf

from trieste.acquisition.interface import (
    SingleModelAcquisitionBuilder,
    AcquisitionFunction,
)
from trieste.models.interfaces import SupportsPredictJoint


class ProbabilityVarianceReduction(
    SingleModelAcquisitionBuilder[SupportsPredictJoint]
):

    def __init__(
        self,
        integration_points,
        noise_variance,
        epsilon,
    ):

        total_points = len(integration_points)
        integration_weights = tf.ones([total_points], dtype=integration_points.dtype) / total_points

        self.integration_points = tf.convert_to_tensor(
            integration_points
        )
        self.integration_weights = tf.convert_to_tensor(
            integration_weights,
            dtype=self.integration_points.dtype,
        )

        self.epsilon = tf.cast(
            epsilon, self.integration_points.dtype
        )

        self.noise = tf.cast(
            noise_variance,
            self.integration_points.dtype,
        )

    def prepare_acquisition_function(
        self, model, dataset=None
    ) -> AcquisitionFunction:

        X = self.integration_points
        weights = self.integration_weights
        eps = self.epsilon
        noise = self.noise

        mean, _ = model.predict(X)
        mean = tf.squeeze(mean, axis=-1)

        # precompute the integration weights
        weighted_mean = weights * tf.exp(-mean / eps)

        def evaluate_candidate(z):

            k_xz = model.covariance_between_points(
                X, z[None, :]
            )
            _, variance = model.predict(z[None, :])
            k_zz = tf.squeeze(variance)


            integral = tf.reduce_sum(weighted_mean * k_xz)

            acquisition = (
                tf.square(integral) / (eps**2 * (k_zz + noise))
            )

            return acquisition

        @tf.function(reduce_retracing=True)
        def acquisition(at):

            # Trieste supplies candidates with shape [..., 1, D]
            original_shape = tf.shape(at)

            candidates = tf.reshape(
                at, [-1, tf.shape(at)[-1]]
            )

            values = tf.map_fn(
                evaluate_candidate,
                candidates,
                fn_output_signature=tf.TensorSpec(
                    shape=(), dtype=X.dtype
                ),
            )

            return tf.reshape(
                values, original_shape[:-1]
            )

        return acquisition


    def update_acquisition_function(
        self, function, model, dataset=None
    ):
        print("[ACQ] updating acquisition function")
        return self.prepare_acquisition_function(model, dataset)
