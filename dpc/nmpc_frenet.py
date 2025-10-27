from icecream import ic
from time import perf_counter
from typing import Literal
import casadi as ca
import numpy as np
from dpc.constants import (
    wheelbase,
    car_length,
    car_width,
    T_max,
    delta_max,
    C_m0,
    C_r0,
    C_r1,
    C_r2,
    m,
    Nf,
    nx,
    nu,
)
from dpc.utils import FloatArray
from dpc.controller import Controller, ControllerStats

delta_s_f = 40.0  # size in meters of the preview center line

q_delta_s = 0.0
q_delta_s_dot: float = 0.1
q_psi: float = 0.0
q_n: float = 0.0
q_v = 0.0
r_T: float = 0.0
r_delta: float = 1.0
r_ddelta = 100.0
r_dT = 10.0


def get_continuous_dynamics_casadi(kappa: ca.Function, *args) -> ca.Function:
    # state and control variables
    delta_s = ca.MX.sym("delta_s")
    n = ca.MX.sym("n")
    psi = ca.MX.sym("psi")
    v = ca.MX.sym("v")
    T = ca.MX.sym("T")
    delta = ca.MX.sym("delta")
    x = ca.vertcat(delta_s, n, psi, v)
    u = ca.vertcat(T, delta)

    # auxiliary variables
    beta = 0.5 * delta  # slip angle
    v_x = v * ca.cos(beta)  # longitudinal velocity
    l_R = 0.5 * wheelbase

    delta_s_dot = v * ca.cos(psi + beta) / (1 + n * kappa(delta_s, *args))

    # assemble bicycle dynamics
    return ca.Function(
        "continuous_dynamics",
        [x, u] + list(args),
        [
            ca.vertcat(
                delta_s_dot,
                v * ca.sin(psi + beta),
                v * ca.sin(beta) / l_R + kappa(delta_s, *args) * delta_s_dot,
                # (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2)) / m,
                (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * ca.tanh(10 * v_x))
                / m,
            )
        ],
    )


class NMPCFrenetController(Controller):
    # solver
    solver: str
    # system dynamics
    # discrete_dynamics: ca.Function
    # optimization problem
    opti: ca.Opti
    # parameters
    x0: ca.MX
    delta_s_cen: ca.MX
    kappa_cen: ca.MX
    # optimization variables
    x: list[ca.MX]
    u: list[ca.MX]
    # cost function
    cost_function: ca.MX

    def __init__(self, solver: Literal["fatrop", "ipopt"] = "ipopt", jit: bool = False):
        """
        Args:
        - cost_weights: CostWeights object containing the weights of the cost functions
        - solver: solver to use, either "fatrop" or "ipopt"
        - jit: whether to use the jit compiler or not
        - codegen: whether to generate C code for the solver (to link against other programs)
        """
        # super().__init__(cost_weights)
        # self.cost_weights = cost_weights
        self.solver = solver

        # declare optimization variables
        opti = ca.Opti()
        x = []
        u = []
        for i in range(Nf):
            x.append(opti.variable(nx))
            u.append(opti.variable(nu))
        x.append(opti.variable(nx))

        # declare parameters
        x0 = opti.parameter(nx)
        delta_s_cen = opti.parameter(100)
        kappa_cen = opti.parameter(100)

        kappa_fun = ca.interpolant("kappa_fun", "linear", [100], 1)

        # instantiate casadi function for discrete dynamics
        # self.discrete_dynamics = get_discrete_dynamics_casadi_2(kappa_fun_with_values)
        f_cont = get_continuous_dynamics_casadi(kappa_fun, delta_s_cen, kappa_cen)
        self.f_cont = f_cont

        # construct cost function
        cost_function = 0.0
        for i in range(Nf):
            # stage control costs
            T, delta = ca.vertsplit(u[i], 1)
            cost_function += r_T * T**2
            cost_function += r_delta * delta * delta
            if i < Nf - 1:
                dT = u[i + 1][0] - u[i][0]
                ddelta = u[i + 1][1] - u[i][1]
                cost_function += r_dT * dT * dT + r_ddelta * ddelta * ddelta

            # stage state costs (since the initial state is fixed, we can't optimize
            # the cost at stage 0 and can ignore it in the cost function)
            if i > 0:
                delta_s, n, psi, v = ca.vertsplit(x[i], 1)
                delta_s_dot = f_cont(x[i], u[i], delta_s_cen, kappa_cen)[0]
                cost_function += (
                    -q_delta_s * delta_s
                    - q_delta_s_dot * delta_s_dot
                    + q_n * n**2
                    + q_psi * psi**2
                    - q_v * v
                )

        # terminal state costs
        delta_s, n, psi, v = ca.vertsplit(x[Nf], 1)
        cost_function += -q_delta_s * delta_s + q_n * n**2 + q_psi * psi**2 - q_v * v
        cost_function = ca.cse(cost_function)
        opti.minimize(cost_function)

        # formulate OCP constraints
        delta_s = ca.MX.sym("delta_s")
        n = ca.MX.sym("n")
        psi = ca.MX.sym("psi")
        v = ca.MX.sym("v")
        T = ca.MX.sym("T")
        delta = ca.MX.sym("delta")
        x = ca.vertcat(delta_s, n, psi, v)
        u = ca.vertcat(T, delta)
        n_right = ca.Function(
            "n_right",
            [x, u],
            [n - 0.5 * car_width * ca.cos(psi) + 0.5 * car_length * ca.sin(psi)],
        )
        n_left = ca.Function(
            "n_left",
            [x, u],
            [n + 0.5 * car_width * ca.cos(psi) + 0.5 * car_length * ca.sin(psi)],
        )
        # del x, u, delta_s, n, psi, v

        # NOTE: the order in which the constraints are declared is important for fatrop
        for i in range(Nf):
            # equality constraints coming from the dynamics
            opti.subject_to(x[i + 1] == rk4(f_cont, x[i], u[i], delta_s_cen, kappa_cen))
            # initial state constraint
            if i == 0:
                opti.subject_to(x[i] == x0)
            # control input constraints
            opti.subject_to((-T_max <= u[i][0]) <= T_max)
            opti.subject_to((-delta_max <= u[i][1]) <= delta_max)

            if i > 0:
                # track constraints
                opti.subject_to(
                    (
                        -1.75
                        <= x[i][1]
                        - 0.5 * car_width * ca.cos(x[i][2])
                        + 0.5 * car_length * ca.sin(x[i][2])
                    )
                    <= 1.75
                )
                opti.subject_to(
                    (
                        -1.75
                        <= x[i][1]
                        + 0.5 * car_width * ca.cos(x[i][2])
                        + 0.5 * car_length * ca.sin(x[i][2])
                    )
                    <= 1.75
                )
                # velocity constraints
                opti.subject_to((0.0 <= x[i][3]) <= v_max)
        # terminal constraints
        opti.subject_to(
            (
                -1.75
                <= x[Nf][1]
                - 0.5 * car_width * ca.cos(x[Nf][2])
                + 0.5 * car_length * ca.sin(x[Nf][2])
            )
            <= 1.75
        )
        opti.subject_to(
            (
                -1.75
                <= x[Nf][1]
                + 0.5 * car_width * ca.cos(x[Nf][2])
                + 0.5 * car_length * ca.sin(x[Nf][2])
            )
            <= 1.75
        )
        opti.subject_to((0.0 <= x[Nf][3]) <= v_max)

        # choose solver options and set the solver
        if solver == "ipopt":
            options = {
                "print_time": 0,
                "expand": True,
                "ipopt": {"sb": "yes", "print_level": 0, "max_resto_iter": 0},
            }
        elif solver == "fatrop":
            options = {
                "print_time": 0,
                "debug": True,
                "expand": True,
                "structure_detection": "auto",
                "fatrop": {"print_level": 0},
            }
        options.update(
            {
                "jit": jit,
                "jit_options": {
                    "flags": ["-O3 -march=native"],
                    "verbose": False,
                },
            }
        )
        ic(solver)
        opti.solver(solver, options)

        # save variables as attributes
        self.opti = opti
        self.x0 = x0
        self.delta_s_cen = delta_s_cen
        self.kappa_cen = kappa_cen
        self.x = x
        self.u = u
        self.cost_function = cost_function

        # call once the solver with dummy data so that the code generation takes place
        # NOTE: this is necessary because Opti seems to lazily create an nlpsol instance (which will trigger jit compilation)
        if jit:
            print("Generating code... ", end="", flush=True)
            start = perf_counter()
            self.control(
                0.0,
                0.0,
                0.0,
                0.0,
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
                np.zeros(Nf + 1),
            )
            print(f"done in {perf_counter() - start:.2f} s")

    def codegen(self):
        states = ca.horzcat(*self.x)
        controls = ca.horzcat(*self.u)
        solver_function = self.opti.to_function(
            f"nmpc_solver_{self.solver}",
            [
                self.x0,
                self.X_ref,
                self.Y_ref,
                self.phi_ref,
                self.v_ref,
                states,
                controls,
            ],
            [states, controls, self.cost_function],
            ["x0", "X_ref", "Y_ref", "phi_ref", "v_ref", "x_guess", "u_guess"],
            ["x_opt", "u_opt", "cost_function"],
        )
        ic(solver_function)
        solver_function.generate(
            f"nmpc_solver_{self.solver}",
            {
                "with_mem": True,
                "with_header": True,
                "verbose": True,
                "indent": 4,
                "main": True,
            },
        )

    def update_params(
        self,
        n: float,
        psi: float,
        v: float,
        delta_s_cen: FloatArray,
        kappa_cen: FloatArray,
    ):
        self.opti.set_value(self.x0, np.array([0.0, n, psi, v]))
        self.opti.set_value(self.delta_s_cen, delta_s_cen)
        self.opti.set_value(self.kappa_cen, kappa_cen)
        # x = np.array([0.0, n, psi, v])
        # xnext = (
        #     rk4(
        #         self.f_cont,
        #         x,
        #         np.zeros(nu),
        #         delta_s_cen,
        #         kappa_cen,
        #     )
        #     .toarray()
        #     .ravel()
        # )
        # ic(x, xnext, np.max(np.abs(xnext - x)))
        # exit(0)

    def set_initial_guess(self):
        for i in range(Nf):
            self.opti.set_initial(
                self.x[i],
                np.zeros(nx),
                # self.last_prediction_x[0]
                # if hasattr(self, "last_prediction_x")
                # else np.zeros(nx),
            )
            self.opti.set_initial(
                self.u[i],
                np.zeros(nu),
                # self.last_prediction_u[0]
                # if hasattr(self, "last_prediction_u")
                # else np.zeros(2),
            )
        self.opti.set_initial(
            self.x[Nf],
            np.zeros(nx),
            # self.last_prediction_x[Nf]
            # if hasattr(self, "last_prediction_x")
            # else np.zeros(4),
        )

    def extract_solution(self, sol: ca.OptiSol) -> tuple[float, FloatArray, FloatArray]:
        cost = sol.value(self.cost_function)
        last_prediction_x = np.array([sol.value(self.x[i]) for i in range(Nf + 1)])
        last_prediction_u = np.array([sol.value(self.u[i]) for i in range(Nf)])
        return cost, last_prediction_x, last_prediction_u

    def control(
        self,
        n: float,
        psi: float,
        v: float,
        delta_s_cen: FloatArray,
        kappa_cen: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ControllerStats]:
        # setup problem
        self.update_params(n, psi, v, delta_s_cen, kappa_cen)
        self.set_initial_guess()

        # solve the optimization problem
        start = perf_counter()
        try:
            # call the solver with solve_limited() to avoid throwing a C++
            # exception that we can't catch in python
            sol = self.opti.solve_limited()
        except RuntimeError as err:
            # with open("nmpc_inputs.json", "w") as f:
            #     json.dump(
            #         {
            #             # "delta_s": delta_s,
            #             "v": v,
            #             "X_ref": X_ref.tolist(),
            #             "Y_ref": Y_ref.tolist(),
            #             "phi_ref": phi_ref.tolist(),
            #             "v_ref": v_ref.tolist(),
            #         },
            #         f,
            #     )
            raise err

        stop = perf_counter()
        runtime = stop - start

        # extract solution
        cost, last_prediction_x, last_prediction_u = self.extract_solution(sol)

        # check exit flag
        stats = self.opti.stats()
        if (
            not stats["success"]
            and stats["return_status"] != "Maximum_Iterations_Exceeded"
        ):
            ic(stats)
            raise RuntimeError(stats["return_status"])

        ic(last_prediction_x[-1, 0])

        self.last_prediction_x = last_prediction_x
        self.last_prediction_u = last_prediction_u

        return (
            last_prediction_x,
            last_prediction_u,
            ControllerStats(runtime=runtime, cost=cost, num_iters=stats["iter_count"]),
        )
