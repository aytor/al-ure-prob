import tensorflow as tf

from trieste.space import Box
from trieste.data import Dataset

class Sampler:

    def __init__(
        self, f, iobs, tobs, x_lim, y_lim, z_lim):
        self.ocp = f
        # self.cov = cov_eci # eci frame

        # lower and upper bounds on dx,dy,dz on the ECI frame
        self.l = [-x_lim, -y_lim, -z_lim]
        self.u = [x_lim, y_lim, z_lim]

        self.search_space = Box(self.l, self.u)

        self.iter = 0
        self.total_iter = tobs
        self.num_initial_points = iobs
        self.initial_query_points = self.search_space.sample(self.num_initial_points)
        self.initial_data = self.observer(self.initial_query_points)


    # def prior(self, x):
    #     dist = multivariate_normal(mean=[0,0,0], cov=self.cov)
    #     return dist.pdf(x)

    def observer(self, query_points: tf.Tensor):
        """
        query_points: shape (N, 3)
        returns Dataset(x, y) where y = J(dx)
        """
        X = query_points.numpy()
        Y = []

        for dx in X:
            self.iter += 1
            # calculate prior probability
            # p0 = self.prior(dx)
            # call solver to compute action S_Tf
            S, _, _, _ = self.ocp(dx)
            # calculate posterior
            # lh = np.exp(-(1/epsilon)*S)
            # p = lh*p0
            print(
                f"iter: [{self.iter}/{self.total_iter+self.num_initial_points}]",
                end='\t'
            )
            # print(f"{dx}", end=' -> ')
            print(f"Q(x)={S}", end="\n")
            # print(f"lh={lh}", end=" ")
            # print(f"p0={p0}", end=" ")
            # print(f"p1={p}", end="\n")
            Y.append([float(S)])  # must be 2D (N,1)

        return Dataset(query_points, tf.convert_to_tensor(Y, dtype=tf.float64))

    # def _evaluate_ocp(self, args):
    #     n, dx = args
    #     S, _, _, _ = self.ocp(dx)
    #     print(
    #         f"iter: [{n}/{self.total_iter+self.num_initial_points}]",
    #         end='\n'
    #     )
    #     return float(S)

    # def batch_observer(self, query_points: tf.Tensor):

    #     X = query_points.numpy()

    #     args = [(n,dx) for n,dx in enumerate(X)]

    #     with Pool(8) as p:
    #         results = p.map(self._evaluate_ocp, args)

    #     self.iter += len(results)

    #     Y = [[r] for r in results]

    #     return Dataset(query_points, tf.convert_to_tensor(Y, dtype=tf.float64))
