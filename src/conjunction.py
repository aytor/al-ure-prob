import numpy as np

from scipy.integrate import solve_ivp

# Constants
MU:float = 3.986e14   # m^3/s^2
Re:float = 6378.135   # km
J2:float = 1082.7e-6  # J2 oblateness coefficient


class Conjunction:
    # def __init__(self, sigma_r=22.0, sigma_t=50.0, sigma_n=28.0):
    def __init__(self, epsilon=1e-4, sigma_r=500.0, sigma_t=1500.0, sigma_n=800.0):
        # delta time
        self.dt = 1
        # final time
        self.T = 4519.0 + 50.0
        # delta time scaled
        self.dt_scaled = self.dt / self.T
        # time range
        self.range = (0, self.T)
        # conjunction timespan
        self.tspan = np.arange(0, self.T+self.dt, self.dt)

        # process noise parameter
        self.epsilon = epsilon

        # spacecraft 1 initial state
        self.ic1 = np.array([
            -3.71793226e+06, 6.11407574e+06, 1.49429134e+04,
            -3.91581972e+02, -2.58291333e+02, 7.44871109e+03
        ])
        # spacecraft 2 initial state
        self.ic2 = np.array([
            -4.60243348e+06, -2.83069692e+06, 7.10785933e+06,
            4.06582624e+03, 2.45439485e+03, 4.22046403e+03
        ])
        # nominal trajectories of both spacecrafts
        self.propagate_nominal_trajectories()
        # covariance for both spacecrafts in rtn frame
        self.cov_rtn = np.diag([sigma_r**2, sigma_t**2, sigma_n**2])
        self.cov_eci = self.cov_rtn2eci()

    def start(self):
        return np.hstack((self.ic1, self.ic2))

    def dynamicsJ2(self, r):
        r_norm = np.linalg.norm(r)
        z2 = r[2]**2
        r2 = r_norm**2

        factor = -MU / r_norm**3
        J2_term = (3/2)*J2*(Re/r_norm)**2

        ax = factor * r[0] * (1 - J2_term*(5*z2/r2 - 1))
        ay = factor * r[1] * (1 - J2_term*(5*z2/r2 - 1))
        az = factor * r[2] * (1 - J2_term*(5*z2/r2 - 3))

        return np.array([ax, ay, az])

    def equations_of_motion(self, t, x):

        r1, v1 = x[0:3], x[3:6]
        r2, v2 = x[6:9], x[9:12]

        return np.hstack([
            v1,
            self.dynamicsJ2(r1),
            v2,
            self.dynamicsJ2(r2)
        ])

    def propagate_nominal_trajectories(self):
        # Nominal propagation (used as initial guesses for the OCP)
        sol = solve_ivp(
            self.equations_of_motion,
            self.range,
            self.start(),
            t_eval=self.tspan,
            rtol=3e-14,
            atol=1e-15
        )

        q = sol.y.T
        self.s1 = q[:, 0:6].T
        self.s2 = q[:, 6:12].T

        self.K = q.shape[0]


    def rtn_frame(self, which):
        """
        Compute RTN unit vectors from position and velocity.

        Returns:
            R_hat, T_hat, N_hat (each shape (3,))
        """

        s = np.asarray(self.ic1) if which==1 else np.asarray(self.ic2)
        r, v = s[:3], s[3:]

        R_hat = r / np.linalg.norm(r)

        h = np.cross(r, v)                  # angular momentum
        N_hat = h / np.linalg.norm(h)

        T_hat = np.cross(N_hat, R_hat)      # completes right-handed frame

        return R_hat, T_hat, N_hat

    def p_rtn2eci(self, p, which=1):
        """
        Transform a point from the rtn to the eci frame of reference

        Parameters
        ----------
        point : array-like, shape (3,)
            Position vector [x, y, z] in RTN(s) where s is scraft 1 or 2
        which : int  {0,1}
            Refers to which RTN reference, scraft 1 or 2

        Returns
        -------
        point : array-like, shape (3,)
            Position vector [x, y, z] in ECI
        """

        # rtn frame
        R_hat, T_hat, N_hat = self.rtn_frame(which)
        # rotation matrix: RTN -> ECI
        M = np.column_stack((R_hat, T_hat, N_hat))

        return M @ p

    def p_eci2rtn(self, p, which=1):
        """
        Transform a point from the rtn to the eci frame of reference

        Parameters
        ----------
        point : array-like, shape (3,)
            Position vector [x, y, z] in RTN(s) where s is scraft 1 or 2
        which : int  {0,1}
            Refers to which RTN reference, scraft 1 or 2

        Returns
        -------
        point : array-like, shape (3,)
            Position vector [x, y, z] in ECI
        """

        # rtn frame
        R_hat, T_hat, N_hat = self.rtn_frame(which)
        # rotation matrix: RTN -> ECI
        M = np.column_stack((R_hat, T_hat, N_hat))

        return np.linalg.inv(M) @ p

    def cov_rtn2eci(self):
        # rtn frame
        R_hat, T_hat, N_hat = self.rtn_frame(1)
        # Rotation matrix: RTN -> ECI
        M = np.column_stack((R_hat, T_hat, N_hat))
        # Covariance transformation
        return M @ self.cov_rtn @ M.T

    def sample_initial_position(self):
        """
        Sample a 3D point from a Gaussian distribution and compute its PDF value.

        Parameters
        ----------
        mean : array-like, shape (3,)
            Mean position vector [x, y, z]

        Returns
        -------
        sample : np.ndarray, shape (3,)
            Sampled 3D point
        pdf_value : float
            Probability density at the sampled point
        """

        # Create multivariate normal distribution
        dist = np.random.multivariate_normal(mean=[0,0,0], cov=self.cov_eci)
        # Sample a single point
        sample = dist.rvs()
        # Evaluate PDF at that sampled point
        pdf = dist.pdf(sample)

        return sample, pdf

    def sample_multiple_initial_states(self, N):
        """
        Sample multiple points from a Gaussian distribution.
        """
        # Create multivariate normal distribution
        samples = np.random.multivariate_normal(
            mean=np.zeros(self.cov_eci.shape[0]),
            cov=self.cov_eci,
            size=N
        )
        return samples

    def compute_search_limits(self, sig=8):

        x_eci = self.p_rtn2eci(np.sqrt(self.cov_rtn[0,:]))
        y_eci = self.p_rtn2eci(np.sqrt(self.cov_rtn[1,:]))
        z_eci = self.p_rtn2eci(np.sqrt(self.cov_rtn[2,:]))

        x_lim = sig*max(x_eci[0],y_eci[0],z_eci[0])
        y_lim = sig*max(x_eci[1],y_eci[1],z_eci[1])
        z_lim = sig*max(x_eci[2],y_eci[2],z_eci[2])

        return x_lim, y_lim, z_lim
