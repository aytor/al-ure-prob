import casadi as ca
from src import get_run_directory

# Constants
MU:float = 3.986e14   # m^3/s^2
Re:float = 6378.135   # km
J2:float = 1082.7e-6  # J2 oblateness coefficient

def generate_casadi_fn(conj, save=True):

    epsilon = conj.epsilon

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

    ocp = opti.to_function(
        'instanton_v2',            # function name
        [dx],                   # inputs (list of CasADi symbols)
        [obj,p1,p2,Tf],      # outputs (list of expressions)
        ['dx'],                 # input names
        ['J','p1','p2','Tf']    # output names
    )
    print("done")

    # Save CasADi function
    if save:
        print("[OCP] saving as: ", end='', flush=True)
        # F = opti.to_function('F', [dv], [obj])
        save_path = get_run_directory() / "models"
        save_path.mkdir(parents=True, exist_ok=True)
        ocp.save(str(save_path / 'instanton.casadi'))
        print("done")

    return ocp
