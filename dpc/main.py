import itertools
import json
import os
from abc import ABC, abstractmethod
from copy import copy
from enum import Enum
from multiprocessing import Pool, cpu_count
from time import perf_counter
from typing import OrderedDict, Literal

import lightning as L
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from casadi import (
    SX,
    MX,
    Function,
    cos,
    sin,
    tanh,
    vertcat,
    horzcat,
    hcat,
    Opti,
    OptiSol,
)
import casadi as ca
from icecream import ic
from lightning import Fabric
from qpsolvers import solve_qp
from scipy.sparse import csc_array
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from strongpods import PODS
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import trange

# misc configs
L.seed_everything(127)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
FloatArray = npt.NDArray[np.float64]

################################################################################
# car and problem parameters
################################################################################

# car mass and geometry
m = 230.0  # mass
wheelbase = 1.5706  # distance between the two axles
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784
# actuator limits
T_max = 500.0
delta_max = 0.5
# general OCP parameters
Nf = 40  # horizon size
nx = 4  # state dimension
nu = 2  # control dimension
dt = 1 / 20  # sampling time

################################################################################
# utils
################################################################################


def teds_projection(x: FloatArray, a: float) -> FloatArray:
    """Projection of x onto the interval [a, a + 2*pi)"""
    return np.mod(x - a, 2 * np.pi) + a


def unwrap_to_pi(x: FloatArray) -> FloatArray:
    """remove discontinuities caused by wrapToPi"""
    diffs = np.diff(x)
    diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
    diffs[diffs < -1.5 * np.pi] += 2 * np.pi
    return np.insert(x[0] + np.cumsum(diffs), 0, x[0])


################################################################################
# models
################################################################################


def get_continuous_dynamics_casadi() -> Function:
    # state and control variables
    X = SX.sym("X")
    Y = SX.sym("Y")
    phi = SX.sym("phi")
    v = SX.sym("v")
    T = SX.sym("T")
    delta = SX.sym("delta")
    x = vertcat(X, Y, phi, v)
    u = vertcat(T, delta)

    # auxiliary variables
    beta = 0.5 * delta  # slip angle
    v_x = v * cos(beta)  # longitudinal velocity
    l_R = 0.5 * wheelbase

    # assemble bicycle dynamics
    return Function(
        "continuous_dynamics",
        [x, u],
        [
            vertcat(
                v * cos(phi + beta),
                v * sin(phi + beta),
                v * sin(beta) / l_R,
                (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * tanh(10 * v_x)) / m,
            )
        ],
    )


def get_discrete_dynamics_casadi() -> Function:
    x = SX.sym("x", nx)
    u = SX.sym("u", nu)
    f = get_continuous_dynamics_casadi()
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return Function(
        "discrete_dynamics", [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)]
    )


def continuous_dynamics_pytorch(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param x: shape (nbatch, nx)
    :param u: shape (nbatch, nu)
    :return xdot: shape (nbatch, nx)
    """
    phi = x[:, 2]
    v = x[:, 3]
    T = u[:, 0]
    delta = u[:, 1]
    beta = 0.5 * delta
    v_x = v * torch.cos(beta)
    l_R = 0.5 * wheelbase

    return torch.stack(
        (
            v * torch.cos(phi + beta),
            v * torch.sin(phi + beta),
            v * torch.sin(beta) / l_R,
            (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * torch.tanh(10 * v_x)) / m,
        ),
        dim=1,
    )


def discrete_dynamics_pytorch(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param x: shape (nbatch, nx)
    :param u: shape (nbatch, nu)
    :return xnext: shape (nbatch, nx)
    """
    k1 = continuous_dynamics_pytorch(x, u)
    k2 = continuous_dynamics_pytorch(x + dt * 0.5 * k1, u)
    k3 = continuous_dynamics_pytorch(x + dt * 0.5 * k2, u)
    k4 = continuous_dynamics_pytorch(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def unrolled_discrete_dynamics_pytorch(
    x: torch.Tensor, u_pred: torch.Tensor
) -> torch.Tensor:
    """
    :param x: shape (nbatch, nx)
    :param u_pred: shape (nbatch, Nf, nu)
    :return x_pred: shape (nbatch, Nf+1, nx)
    """
    x_pred = [x]
    for i in range(Nf):
        x_pred.append(discrete_dynamics_pytorch(x_pred[-1], u_pred[:, i]))

    return torch.stack(x_pred, dim=1)


################################################################################
# controllers
################################################################################


@PODS
class ControllerStats:
    runtime: float
    cost: float
    num_iters: int = 0


@PODS
class CostWeights:
    q_lon: float = 10.0
    q_lat: float = 20.0
    q_phi: float = 50.0
    q_v: float = 20.0
    r_T: float = 1e-3
    r_delta: float = 2.0
    q_lon_f: float = 1000.0
    q_lat_f: float = 1000.0
    q_phi_f: float = 500.0
    q_v_f: float = 1000.0


class Controller(ABC):
    cost_weights: CostWeights

    def __init__(self, cost_weights: CostWeights = CostWeights()):
        self.cost_weights = cost_weights

    @abstractmethod
    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ControllerStats]:
        """
        Args:
            X (float): current X position
            Y (float): current Y position
            phi (float): current heading angle
            v (float): current velocity
            X_ref (FloatArray): reference X trajectory
            Y_ref (FloatArray): reference Y trajectory
            phi_ref (FloatArray): reference heading angle trajectory
            v_ref (FloatArray): reference velocity trajectory
        Returns:
            x_pred (FloatArray): state prediction, shape (Nf+1, nx)
            u_pred (FloatArray): control prediction, shape (Nf, nu)
            stats (ControllerStats): controller statistics
        """
        pass


class NMPCController(Controller):
    # system dynamics
    discrete_dynamics: Function
    # optimization problem
    opti: Opti
    # parameters
    x0: MX
    X_ref: MX
    Y_ref: MX
    phi_ref: MX
    v_ref: MX
    # optimization variables
    x: list[MX]
    u: list[MX]
    # cost function
    cost_function: MX

    def __init__(
        self,
        cost_weights: CostWeights = CostWeights(),
        solver: Literal["fatrop", "ipopt"] = "ipopt",
        jit: bool = False,
        codegen: bool = False,
        **kwargs,
    ):
        super().__init__(cost_weights)

        # instantiate casadi function for discrete dynamics
        self.discrete_dynamics = get_discrete_dynamics_casadi()

        # declare optimization variables
        opti = Opti()
        x = []
        u = []
        for i in range(Nf):
            x.append(opti.variable(nx))
            u.append(opti.variable(nu))
        x.append(opti.variable(nx))

        # declare parameters
        x0 = opti.parameter(nx)
        X_ref = opti.parameter(Nf + 1)
        Y_ref = opti.parameter(Nf + 1)
        phi_ref = opti.parameter(Nf + 1)
        v_ref = opti.parameter(Nf + 1)

        # construct cost function
        cost_function = 0.0
        for i in range(Nf):
            # stage state costs (since the initial state is fixed, we can't optimize
            # the cost at stage 0 and can ignore it in the cost function)
            if i > 0:
                cp = cos(phi_ref[i])
                sp = sin(phi_ref[i])
                X = x[i][0]
                Y = x[i][1]
                phi = x[i][2]
                v = x[i][3]
                e_lon = cp * (X - X_ref[i]) + sp * (Y - Y_ref[i])
                e_lat = -sp * (X - X_ref[i]) + cp * (Y - Y_ref[i])
                cost_function += (
                    self.cost_weights.q_lon * e_lon**2
                    + self.cost_weights.q_lat * e_lat**2
                    + self.cost_weights.q_phi * (phi - phi_ref[i]) ** 2
                    + self.cost_weights.q_v * (v - v_ref[i]) ** 2
                )
            # stage control costs
            T = u[i][0]
            delta = u[i][1]
            T_ref = (
                tanh(10 * v_ref[i])
                * (C_r0 + C_r1 * v_ref[i] + C_r2 * v_ref[i] ** 2)
                / C_m0
            )
            cost_function += (
                self.cost_weights.r_T * (T - T_ref) ** 2
                + self.cost_weights.r_delta * delta**2
            )

        # terminal state costs
        cp = cos(phi_ref[Nf])
        sp = sin(phi_ref[Nf])
        X = x[Nf][0]
        Y = x[Nf][1]
        phi = x[Nf][2]
        v = x[Nf][3]
        e_lon = cp * (X - X_ref[Nf]) + sp * (Y - Y_ref[Nf])
        e_lat = -sp * (X - X_ref[Nf]) + cp * (Y - Y_ref[Nf])
        cost_function += (
            self.cost_weights.q_lon_f * e_lon**2
            + self.cost_weights.q_lat_f * e_lat**2
            + self.cost_weights.q_phi_f * (phi - phi_ref[Nf]) ** 2
            + self.cost_weights.q_v_f * (v - v_ref[Nf]) ** 2
        )
        opti.minimize(cost_function)

        # formulate OCP constraints
        # NOTE: the order in which the constraints are declared is important for fatrop
        for i in range(Nf):
            # equality constraints coming from the dynamics
            opti.subject_to(x[i + 1] == self.discrete_dynamics(x[i], u[i]))
            # initial state constraint
            if i == 0:
                opti.subject_to(x[i] == x0)
            # NOTE: here we can't use opti.bounded() for some reason (fatrop doesn't detect the constraints)
            opti.subject_to(-T_max <= (u[i][0] <= T_max))
            opti.subject_to(-delta_max <= (u[i][1] <= delta_max))

        # choose solver options
        if solver == "ipopt":
            options = {
                "print_time": 0,
                "expand": True,
                "ipopt": {"sb": "yes", "print_level": 0, "max_resto_iter": 0},
            }
        elif solver == "fatrop":
            # probably there is a problem with the fact that we specify the
            options = {
                "print_time": 0,
                "debug": False,
                "expand": True,
                "structure_detection": "auto",
                "fatrop": {"print_level": 0},
            }
        options.update(
            {
                "jit": jit,
                "jit_options": {"flags": ["-O3 -march=native"], "verbose": False},
            }
        )

        # assemble solver
        opti.solver(solver, options)

        if codegen:
            states = horzcat(*x)
            controls = horzcat(*u)
            solver_function = opti.to_function(
                f"nmpc_solver_{solver}",
                [x0, X_ref, Y_ref, phi_ref, v_ref, states, controls],
                [states, controls, cost_function],
                ["x0", "X_ref", "Y_ref", "phi_ref", "v_ref", "x_guess", "u_guess"],
                ["x_opt", "u_opt", "cost_function"],
            )
            ic(solver_function)
            solver_function.generate(
                f"nmpc_solver_{solver}.c",
                {
                    "with_mem": True,
                    "with_header": True,
                    "verbose": True,
                    "indent": 4,
                    "main": True,
                },
            )

        # save variables as attributes
        self.opti = opti
        self.x0 = x0
        self.X_ref = X_ref
        self.Y_ref = Y_ref
        self.phi_ref = phi_ref
        self.v_ref = v_ref
        self.x = x
        self.u = u
        self.cost_function = cost_function

        # call once the solver with dummy data so that the code generation takes place
        # NOTE: this is necessary because Opti seems to lazily create an nlpsol instance (which will trigger codegen)
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

    def update_params(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ):
        self.opti.set_value(self.x0, np.array([X, Y, phi, v]))
        self.opti.set_value(self.X_ref, X_ref)
        self.opti.set_value(self.Y_ref, Y_ref)
        self.opti.set_value(self.phi_ref, phi_ref)
        self.opti.set_value(self.v_ref, v_ref)

    def set_initial_guess(
        self,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ):
        T_ref = (C_r0 + C_r1 * v_ref + C_r2 * v_ref * v_ref) / C_m0
        for i in range(Nf):
            self.opti.set_initial(
                self.x[i], np.array([X_ref[i], Y_ref[i], phi_ref[i], v_ref[i]])
            )
            self.opti.set_initial(self.u[i], np.array([T_ref[i], 0.0]))
        self.opti.set_initial(
            self.x[Nf], np.array([X_ref[Nf], Y_ref[Nf], phi_ref[Nf], v_ref[Nf]])
        )

    def extract_solution(self, sol: OptiSol) -> tuple[float, FloatArray, FloatArray]:
        cost = sol.value(self.cost_function)
        last_prediction_x = np.array([sol.value(self.x[i]) for i in range(Nf + 1)])
        last_prediction_u = np.array([sol.value(self.u[i]) for i in range(Nf)])
        return cost, last_prediction_x, last_prediction_u

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ControllerStats]:
        # setup problem
        self.update_params(X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref)
        self.set_initial_guess(X_ref, Y_ref, phi_ref, v_ref)

        # solve the optimization problem
        start = perf_counter()
        try:
            sol = self.opti.solve_limited()
        except RuntimeError as err:
            with open("bruh.json", "w") as f:
                json.dump(
                    {
                        "X": X,
                        "Y": Y,
                        "phi": phi,
                        "v": v,
                        "X_ref": X_ref.tolist(),
                        "Y_ref": Y_ref.tolist(),
                        "phi_ref": phi_ref.tolist(),
                        "v_ref": v_ref.tolist(),
                    },
                    f,
                )
            print(err)
            breakpoint()
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
            breakpoint()
            raise RuntimeError(stats["return_status"])

        return (
            last_prediction_x,
            last_prediction_u,
            ControllerStats(runtime=runtime, cost=cost, num_iters=stats["iter_count"]),
        )


def job(process_df: np.ndarray):
    reference_controller = NMPCController()
    result = np.zeros((process_df.shape[0], (1 + nu * Nf)))
    for i in range(process_df.shape[0]):
        x_ref = process_df[i, nx : nx * (Nf + 2)].reshape(Nf + 1, nx)
        _, u_ref, stats = reference_controller.control(
            X=process_df[i, 0],
            Y=process_df[i, 1],
            phi=process_df[i, 2],
            v=process_df[i, 3],
            X_ref=x_ref[:, 0],
            Y_ref=x_ref[:, 1],
            phi_ref=x_ref[:, 2],
            v_ref=x_ref[:, 3],
        )
        result[i, :-1] = u_ref.ravel()
        result[i, -1] = stats.cost
    return result


class DPCController(Controller):
    net: nn.Module
    fabric: Fabric

    def __init__(
        self,
        nhidden: list[int],
        nonlinearity: str,
        weights_file: str | None = None,
        cost_weights: CostWeights = CostWeights(),
        accelerator: str = "mps",
    ):
        super().__init__(cost_weights)
        self.fabric = Fabric(accelerator=accelerator)
        self.net = self.fabric.setup(
            DPCController._construct_net(
                nin=(Nf + 2) * nx,
                nout=Nf * nu,
                nhidden=nhidden,
                nonlinearity=nonlinearity,
            )
        )
        if weights_file is not None:
            assert os.path.exists(weights_file)
            self.fabric.load(weights_file, {"model": self.net})

    # data utilities ============================================================
    @staticmethod
    def generate_constant_curvature_trajectories(
        curvatures: FloatArray, v_ref: float = 5.0
    ) -> FloatArray:
        """
        Generate a dataset of trajectories with constant curvature and speed.
        Args:
            curvatures (FloatArray): array of curvatures for each trajectory
            v_ref (float): common reference speed for all trajectories
        Returns:
            poses (FloatArray): array of pose trajectories, shape (ntraj, Nf+1, 3)
        """
        s = v_ref * dt * np.arange(Nf + 1)[None, :]  # shape (1, Nf+1)
        poses = np.zeros((len(curvatures), Nf + 1, 3))
        zero_curvature_idx = np.abs(curvatures) < 1e-3
        nonzero_curvature_idx = ~zero_curvature_idx
        # we have a straight trajectory for zero curvature
        poses[zero_curvature_idx, :, 0] = s
        #
        curvature_radius = np.abs(1 / curvatures[nonzero_curvature_idx])[
            :, None
        ]  # shape (ncurv, 1)
        angles = s / curvature_radius - np.pi / 2  # shape (ncurv, Nf+1)
        reference_X = curvature_radius * np.cos(angles)  # shape (ncurv, Nf+1)
        poses[nonzero_curvature_idx, :, 0] = reference_X
        reference_Y = curvature_radius * np.sin(angles)  # shape (ncurv, Nf+1)
        reference_Y += curvature_radius  # shape (ncurv, Nf+1)
        reference_Y *= np.sign(
            curvatures[nonzero_curvature_idx, None]
        )  # shape (ncurv, Nf+1)
        poses[nonzero_curvature_idx, :, 1] = reference_Y
        reference_headings = np.sign(curvatures[nonzero_curvature_idx, None]) * (
            angles + np.pi / 2
        )
        poses[nonzero_curvature_idx, :, 2] = reference_headings  # shape (ncurv, Nf+1)

        return poses

    @staticmethod
    def compute_mpc_output(df: np.ndarray) -> np.ndarray:
        """
        Compute the output of the NMPCController for each sample in the dataset.
        """
        # split the dataset for each process
        n_samples = df.shape[0]
        nprocesses = cpu_count()
        # nprocesses = 1
        n_sample_per_process = n_samples // nprocesses
        df_per_process = []
        for i in range(nprocesses):
            start = i * n_sample_per_process
            end = (i + 1) * n_sample_per_process if i < nprocesses - 1 else n_samples
            df_per_process.append(df[start:end])

        start = perf_counter()
        with Pool(nprocesses) as p:
            results = np.vstack(p.map(job, df_per_process))
        end = perf_counter()
        ic(end - start)

        return results

    @staticmethod
    def save_dataset(df: np.ndarray, filename: str):
        np.savetxt(
            filename,
            df,
            fmt="%.5f",
            delimiter=",",
            comments="",
            header="X,Y,phi,v,"
            + ",".join(
                [f"X_ref_{i},Y_ref_{i},phi_ref_{i},v_ref_{i}" for i in range(Nf + 1)]
            )
            + ","
            + ",".join([f"T_mpc_{i},delta_mpc_{i}" for i in range(Nf)])
            + ",cost_mpc",
        )

    @staticmethod
    def create_pretraining_dataset(
        filename: str,
        v_ref=5.0,
        max_curvature=1 / 6,
        n_trajs=31,
        n_lat=11,
        n_phi=11,
        n_v=21,
    ) -> None:
        """
        sample arcs of constant curvatures with constant speeds to create references -> 10x10=100
        then create different initial conditions by perturbing e_lat, e_lon, e_phi, e_v. Vary the bounds on e_phi in function of the curvature -> 10^4 values
        -> 10^6 samples, each of size nx x (Nf+2) + nu x Nf = 4 x 42 + 2 * 40 = 248 -> ~250MB
        """
        # compute trajctories with constant curvature
        trajectories = DPCController.generate_constant_curvature_trajectories(
            curvatures=np.linspace(-max_curvature, max_curvature, n_trajs), v_ref=v_ref
        )
        # plot the trajectories
        plt.figure()
        for traj in trajectories:
            plt.plot(traj[:, 0], traj[:, 1])
        plt.axis("equal")
        plt.show()

        # check in with the user before actually generating the dataset
        n_samples = n_trajs * n_lat * n_phi * n_v
        answer = input(f"Generating dataset with {n_samples} samples? [y/n] ")
        if answer != "y":
            return

        # generate associated perturbed initial state
        lateral_errors = np.linspace(-0.5, 0.5, n_lat)
        heading_errors = np.linspace(-0.5, 0.5, n_phi)
        vel_errors = np.linspace(-5.0, 5.0, n_v)

        # compute the combinations of initial states and state reference
        df = np.full((n_samples, nx * (Nf + 2)), np.nan)
        for sample_id, (traj, lat_err, heading_err, vel_err) in enumerate(
            itertools.product(
                trajectories,
                lateral_errors,
                heading_errors,
                vel_errors,
            )
        ):
            # initial conditions
            df[sample_id, :nx] = np.array([0.0, lat_err, heading_err, v_ref + vel_err])
            # reference trajectory
            X_ref = traj[:, 0]
            Y_ref = traj[:, 1]
            phi_ref = traj[:, 2]
            df[sample_id, nx : nx * (Nf + 2)] = np.concatenate(
                (
                    np.reshape(
                        np.column_stack(
                            (
                                X_ref,
                                Y_ref,
                                phi_ref,
                                v_ref * np.ones_like(X_ref),
                            )
                        ),
                        nx * (Nf + 1),
                    ),
                )
            )

        # go over all the data and compute the output of the NMPCController
        df = np.hstack((df, DPCController.compute_mpc_output(df)))

        # save data to csv file
        DPCController.save_dataset(df, filename)

    @staticmethod
    def create_finetuning_dataset(
        filename: str,
        n_samples: int,
        sigma_curvature: float,
        sigma_lat: float,
        sigma_phi: float,
        sigma_v: float,
        v_ref: float = 5.0,
    ) -> None:
        """
        Here we create the same trajectories but we sample a given number of intial conditions from a multi-variate normal distribution
        centered around the reference trajectory with a given covariance matrix.
        Or we also sample the trajectories from a given distribution (on the curvature).
        """
        # generate curvatures, lateral errors, heading errors, velocity errors from a multibariate normal distribution
        # with covariance diagonal(sigma_curvature, sigma_lat, sigma_phi, sigma_v)
        gen = np.random.randn(n_samples, 4) * np.array(
            [[sigma_curvature, sigma_lat, sigma_phi, sigma_v]]
        )
        alternative_curvatures = (
            np.random.exponential(scale=2.0, size=n_samples) - v_ref
        )
        index_choice = np.random.choice(a=2, p=[0.8, 0.2], size=n_samples)
        gen[:, 3] = gen[:, 3] * index_choice + alternative_curvatures * (
            1 - index_choice
        )
        # plot the distribution of each value as a separate histogram
        fig, axs = plt.subplots(2, 2)
        for i, ax in enumerate(axs.flat):
            ax.hist(gen[:, i], bins=30)
            ax.set_title(["curvature", "lateral", "heading", "velocity"][i])
        plt.tight_layout()
        plt.show()
        # generate the trajectories
        trajectories = DPCController.generate_constant_curvature_trajectories(
            curvatures=gen[:, 0], v_ref=v_ref
        )
        # assemble the dataset
        df = np.zeros((n_samples, nx + (Nf + 1) * nx + Nf * nu + 1))
        df[:, 1] = gen[:, 1]
        df[:, 2] = gen[:, 2]
        df[:, 3] = gen[:, 3] + v_ref
        df[:, nx:] = np.reshape(
            np.concatenate(
                (trajectories, v_ref * np.ones((n_samples, Nf + 1, 1))), axis=2
            ),
            (n_samples, nx * (Nf + 1)),
        )

        # compute the output of the NMPCController
        df = np.hstack((df, DPCController.compute_mpc_output(df)))

        # save the dataset
        DPCController.save_dataset(df, filename)

    class DPCDataset(Dataset):
        data: torch.Tensor

        def __init__(self, filename: str):
            super().__init__()
            data_np = np.loadtxt(filename, delimiter=",", skiprows=1)
            # convert it to torch tensors
            self.data = torch.tensor(data_np, dtype=torch.float32)

        def __len__(self) -> int:
            return self.data.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx]

    @staticmethod
    def _load_data(
        filename: str,
        train_data_proportion: float = 0.8,
        batch_sizes: tuple[int | None, int | None] = (None, None),
    ) -> tuple[DPCDataset, DataLoader, DataLoader]:
        # load the dataset
        dataset = DPCController.DPCDataset(filename)
        # split everything into train and validation sets
        train_data_size = int(len(dataset) * train_data_proportion)
        train_data, val_data = random_split(
            dataset, (train_data_size, len(dataset) - train_data_size)
        )
        # create dataloaders
        return (
            dataset,
            DataLoader(
                train_data,
                batch_size=train_data_size
                if batch_sizes[0] is None
                else batch_sizes[0],
                shuffle=True,
            ),
            DataLoader(
                val_data,
                batch_size=len(val_data) if batch_sizes[1] is None else batch_sizes[1],
                shuffle=True,
            ),
        )

    @staticmethod
    def _construct_net(
        nin: int,
        nout: int,
        nhidden: list[int] = [128, 128, 128],
        nonlinearity: str = "relu",
    ) -> nn.Module:
        assert len(nhidden) >= 1
        nonlinearity_function = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }[nonlinearity]

        di = {
            "batchnorm": nn.BatchNorm1d(nin, affine=False),
            "hidden_layer_0": nn.Linear(nin, nhidden[0], bias=True),
            "nonlinearity_0": nonlinearity_function,
        }
        nn.init.kaiming_normal_(di["hidden_layer_0"].weight, nonlinearity=nonlinearity)
        for i in range(1, len(nhidden)):
            di.update(
                {
                    f"hidden_layer_{i}": nn.Linear(
                        nhidden[i - 1], nhidden[i], bias=True
                    ),
                    f"nonlinearity_{i}": nonlinearity_function,
                }
            )
            nn.init.kaiming_normal_(
                di[f"hidden_layer_{i}"].weight, nonlinearity=nonlinearity
            )

        class ControlConstraintScale(nn.Module):
            def __init__(self):
                super().__init__()
                # we have to define the scale as a Parameter in order to be able to set the
                # appropriate device for the nn.Module that will contain this Module
                self.scale = nn.Parameter(
                    torch.tensor([T_max, delta_max]), requires_grad=False
                )

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.scale * F.tanh(input.reshape(input.shape[0], Nf, nu))

        di.update(
            {
                "output_layer": nn.Linear(nhidden[-1], nout, bias=True),
                "ouput_scaling": ControlConstraintScale(),
            }
        )
        nn.init.kaiming_normal_(di["output_layer"].weight, nonlinearity=nonlinearity)

        return nn.Sequential(OrderedDict(di))

    def compute_mpc_loss(
        self,
        x_ref: torch.Tensor,
        x_pred: torch.Tensor,
        u_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss for the MPC controller for each sample in the batch.

        :param x_ref: shape (nbatch, Nf+1, nx)
        :param x_pred: shape (nbatch, Nf+1, nx)
        :param u_pred: shape (nbatch, Nf, nx)
        :return: the loss for each sample in the batch, shape (nbatch,)
        """
        nbatch = x_ref.shape[0]
        assert x_ref.shape == x_pred.shape == (nbatch, Nf + 1, nx)
        assert u_pred.shape == (nbatch, Nf, nu)
        phi_ref = x_ref[:, :, 2]
        v_ref = x_ref[:, :, 3]
        Rot = torch.stack(
            (
                torch.stack((torch.cos(phi_ref), -torch.sin(phi_ref)), dim=2),
                torch.stack((torch.sin(phi_ref), torch.cos(phi_ref)), dim=2),
            ),
            dim=3,
        )  # shape (nbatch, Nf+1, 2, 2)
        lon_lat_errs_sq = torch.square(
            torch.squeeze(
                torch.matmul(
                    Rot,
                    torch.unsqueeze(x_pred[:, :, :2] - x_ref[:, :, :2], dim=3),
                ),
                dim=3,
            )
        )  # shape (nbatch, Nf+1, 2)
        T_ref = (
            C_r0 + C_r1 * v_ref[:, :-1] + C_r2 * v_ref[:, :-1] * v_ref[:, :-1]
        ) / C_m0  # shape (nbatch, Nf)
        cost_per_sample = (
            # stage longitudinal errors
            self.cost_weights.q_lon * torch.sum(lon_lat_errs_sq[:, 1:-1, 0], dim=1)
            # terminal longitudinal errors
            + self.cost_weights.q_lon_f * lon_lat_errs_sq[:, -1, 0]
            # stage lateral errors
            + self.cost_weights.q_lat * torch.sum(lon_lat_errs_sq[:, 1:-1, 1], dim=1)
            # terminal lateral errors
            + self.cost_weights.q_lat_f * lon_lat_errs_sq[:, -1, 1]
            # stage heading errors
            + self.cost_weights.q_phi
            * torch.sum(torch.square(x_pred[:, 1:-1, 2] - x_ref[:, 1:-1, 2]), dim=1)
            # terminal heading errors
            + self.cost_weights.q_phi_f
            * torch.square(x_pred[:, -1, 2] - x_ref[:, -1, 2])
            # stage velocity errors
            + self.cost_weights.q_v
            * torch.sum(torch.square(x_pred[:, 1:-1, 3] - x_ref[:, 1:-1, 3]), dim=1)
            # terminal velocity errors
            + self.cost_weights.q_v_f * torch.square(x_pred[:, -1, 3] - x_ref[:, -1, 3])
            # throttle errors
            + self.cost_weights.r_T
            * torch.sum(torch.square(u_pred[:, :, 0] - T_ref), dim=1)
            # steering errors
            + self.cost_weights.r_delta
            * torch.sum(torch.square(u_pred[:, :, 1]), dim=1)
        )  # shape (nbatch,)
        assert cost_per_sample.shape == (nbatch,)
        return cost_per_sample

    def compute_dpc_loss(
        self,
        x_ref: torch.Tensor,
        x_pred: torch.Tensor,
        u_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Basically computes the mean MPC loss on the batch.
        :param x_ref: shape (nbatch, Nf+1, nx)
        :param x_pred: shape (nbatch, Nf+1, nx)
        :param u_pred: shape (nbatch, Nf, nx)
        :return: the average loss for the minibatch, shape (1,)
        """
        return torch.mean(self.compute_mpc_loss(x_ref, x_pred, u_pred))

    def run_model(self, batch: torch.Tensor) -> torch.Tensor:
        """
        :param batch: shape (nbatch, nx * (Nf+2)), first nx columns correspond to the current state, the other ones to the reference
        :param cost_weights:
        :return: the average loss for the minibatch
        """
        # extract current and reference states from batch
        nbatch = len(batch)
        x = batch[:, :nx]
        x_ref = batch[:, nx:].reshape(nbatch, Nf + 1, nx)
        # run the network to compute predicted controls
        u_pred = self.net(batch)
        # run the model to compute the predicted states
        x_pred = unrolled_discrete_dynamics_pytorch(x, u_pred)
        # compute MPC cost function
        return self.compute_dpc_loss(x_pred, x_ref, u_pred)
        # return self.compute_imitation_loss(u_pred,u_ref)

    @staticmethod
    def train(
        dataset_filename: str,
        num_epochs: int,
        nhidden: list[int],
        nonlinearity: str,
        weights_filename: str | None = None,
        training_state_filename: str | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        batch_sizes=(None, None),
        cost_weights: CostWeights = CostWeights(),
    ) -> None:
        controller = DPCController(
            nhidden, nonlinearity, weights_filename, cost_weights
        )
        optimizer = controller.fabric.setup_optimizers(
            torch.optim.Adam(
                controller.net.parameters(), lr=lr, weight_decay=weight_decay
            )
        )
        _, train_dataloader, val_dataloader = DPCController._load_data(
            dataset_filename,
            train_data_proportion=0.8,
            batch_sizes=batch_sizes,
        )
        training_state = {"model": controller.net, "optimizer": optimizer, "lr": lr}
        if training_state_filename is not None:
            assert os.path.exists(training_state_filename)
            controller.fabric.load(training_state_filename, training_state)

        train_dataloader, val_dataloader = controller.fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )
        best_val_loss = np.inf
        train_losses = []
        val_losses = []

        progress_bar = trange(num_epochs)
        for epoch in progress_bar:
            # training step
            controller.net.train()
            total_train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()
                train_loss = controller.run_model(batch)
                controller.fabric.backward(train_loss)
                optimizer.step()
                total_train_loss += train_loss.item()
            total_train_loss /= len(train_dataloader)
            train_losses.append(total_train_loss)

            # validation step
            controller.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    val_loss += controller.run_model(batch).item()
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)

            # save weights if the model is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(controller.net.state_dict(), "best_model.pth")
                controller.fabric.save("best.ckpt", training_state)

            # logging
            progress_bar.set_description(
                f"Epoch {epoch + 1}, train loss: {total_train_loss:.4f}, val loss: {val_loss:.4f}, best val loss: {best_val_loss:.4f}"
            )

        # save final weights
        # torch.save(controller.net.state_dict(), "final_model.pth")
        controller.fabric.save("final.ckpt", training_state)

        plt.figure()
        plt.plot(np.arange(1, num_epochs + 1), train_losses, label="train loss")
        plt.plot(np.arange(1, num_epochs + 1), val_losses, label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

    def compute_open_loop_predictions(
        self,
        dataset_filename: str,
        data_file: str,
        batch_sizes=(None, None),
    ):
        # load the data
        _, _, val_dataloader = DPCController._load_data(
            dataset_filename,
            train_data_proportion=0.8,
            batch_sizes=batch_sizes,
        )
        val_dataloader = self.fabric.setup_dataloaders(val_dataloader)
        # run the model on the first batch of the validation set (this way we can choose)
        self.net.eval()
        with torch.no_grad():
            batch = next(iter(val_dataloader))
            nbatch = len(batch)
            x = batch[:, :nx]
            x_ref = batch[:, nx:].reshape(nbatch, Nf + 1, nx)
            u_pred = self.net(batch)
            x_pred = unrolled_discrete_dynamics_pytorch(x, u_pred)
        all_x_ref = x_ref.cpu().detach().numpy()
        all_x_pred = x_pred.cpu().detach().numpy()
        all_u_pred = u_pred.cpu().detach().numpy()
        # save the data in the same format as the closed loop run to visualize the open loop predictions
        np.savez(
            data_file,
            x_ref=all_x_ref,
            x_pred=all_x_pred,
            u_pred=all_u_pred,
            runtimes=np.ones(nbatch),
            costs=np.zeros(nbatch),
            center_line=np.full((0, 2), np.nan),
            blue_cones=np.full((0, 2), np.nan),
            yellow_cones=np.full((0, 2), np.nan),
            big_orange_cones=np.full((0, 2), np.nan),
        )

    def control(
        self,
        X: float,
        Y: float,
        phi: float,
        v: float,
        X_ref: FloatArray,
        Y_ref: FloatArray,
        phi_ref: FloatArray,
        v_ref: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ControllerStats]:
        # translate and rotate reference and current poses
        X_ref_0 = copy(X_ref[0])
        Y_ref_0 = copy(Y_ref[0])
        phi_ref_0 = copy(phi_ref[0])

        R = np.array(
            [
                [
                    np.cos(phi_ref_0),
                    np.sin(phi_ref_0),
                ],
                [
                    -np.sin(phi_ref_0),
                    np.cos(phi_ref_0),
                ],
            ]
        )
        current_state = np.array([X - X_ref_0, Y - Y_ref_0, phi - phi_ref_0, v])
        current_state[:2] = R @ current_state[:2]
        ref = np.column_stack(
            (
                X_ref - X_ref_0,
                Y_ref - Y_ref_0,
                phi_ref - phi_ref_0,
                v_ref,
            )
        )
        ref[:, :2] = ref[:, :2] @ R.T

        # assemble inputs into a tensor
        input = torch.unsqueeze(
            torch.tensor(
                np.concatenate(
                    (
                        np.array([X, Y, phi, v]),
                        np.column_stack((X_ref, Y_ref, phi_ref, v_ref)).ravel(),
                    )
                ),
                dtype=torch.float32,
                requires_grad=False,
                device=self.fabric.device,
            ),
            0,
        )
        x = input[:, :nx]  # shape (1,nx)
        x_ref = input[:, nx:].reshape(1, Nf + 1, nx)  # shape (1, Nf+1, nx)
        # forward pass of the net on the data
        start = perf_counter()
        self.net.eval()
        with torch.no_grad():
            u_pred = self.net(input)
        stop = perf_counter()
        # reformat the output
        x_pred = unrolled_discrete_dynamics_pytorch(x, u_pred)
        loss = self.compute_dpc_loss(x_pred, x_ref, u_pred).item()

        # convert tensors to numpy arrays
        x_pred = x_pred.squeeze(dim=0).cpu().detach().numpy()
        u_pred = u_pred.squeeze(dim=0).cpu().detach().numpy()
        return x_pred, u_pred, ControllerStats(runtime=stop - start, cost=loss)


################################################################################
# motion planner
################################################################################

NUMBER_SPLINE_INTERVALS = 500


def fit_spline(
    path: FloatArray,
    curv_weight: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """
    computes the coefficients of each spline portion of the path.
    > Note: the path is assumed to be closed but the first and last points are NOT the same.

    :param path: Nx2 array of points
    :param curv_weight: weight of the curvature term in the cost function
    :return_errs:
    :qp_solver:
    :returns p_X, p_Y: Nx4 arrays of coefficients of the splines in the x and y directions
                       (each row correspond to a_i, b_i, c_i, d_i coefficients of the i-th
                       spline portion defined by a_i + b_i * t + ...)
    """
    assert (
        len(path.shape) == 2 and path.shape[1] == 2
    ), f"path must have shape (N,2) but has shape {path.shape}"

    # precompute all the QP data
    N = path.shape[0]
    delta_s = np.linalg.norm(path[1:] - path[:-1], axis=1)
    delta_s = np.append(delta_s, np.linalg.norm(path[0] - path[-1]))
    rho = np.zeros(N)
    rho[:-1] = delta_s[:-1] / delta_s[1:]
    rho[-1] = delta_s[-1] / delta_s[0]
    IN = speye(N, format="csc")
    A = spkron(
        IN,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 2.0, 6.0],
            ]
        ),
        format="csc",
    ) + csc_array(
        (
            np.concatenate((-np.ones(N), -rho, -2 * rho**2)),
            (
                np.concatenate(
                    (
                        3 * np.arange(N),
                        1 + 3 * np.arange(N),
                        2 + 3 * np.arange(N),
                    )
                ),
                np.concatenate(
                    (
                        np.roll(4 * np.arange(N), -1),
                        np.roll(1 + 4 * np.arange(N), -1),
                        np.roll(2 + 4 * np.arange(N), -1),
                    )
                ),
            ),
        ),
        shape=(3 * N, 4 * N),
    )
    B = spkron(IN, np.array([[1.0, 0.0, 0.0, 0.0]]), format="csc")
    C = csc_array(
        (
            np.concatenate((2 / np.square(delta_s), 6 / np.square(delta_s))),
            (
                np.concatenate((np.arange(N), np.arange(N))),
                np.concatenate((2 + 4 * np.arange(N), 3 + 4 * np.arange(N))),
            ),
        ),
        shape=(N, 4 * N),
    )
    P = B.T @ B + curv_weight * C.T @ C + 1e-10 * speye(4 * N, format="csc")
    q = -B.T @ path
    b = np.zeros(3 * N)

    # solve the QP for X and Y separately
    p_X = solve_qp(P=P, q=q[:, 0], A=A, b=b, solver="osqp")
    p_Y = solve_qp(P=P, q=q[:, 1], A=A, b=b, solver="osqp")
    if p_X is None or p_Y is None:
        raise ValueError("solving qp failed")

    # compute interpolation error on X and Y
    # X_err = B @ p_X - path[:, 0]
    # Y_err = B @ p_Y - path[:, 1]

    # reshape to (N,4) arrays
    p_X = np.reshape(p_X, (N, 4))
    p_Y = np.reshape(p_Y, (N, 4))

    return p_X, p_Y


def check_spline_coeffs_dims(coeffs_X: FloatArray, coeffs_Y: FloatArray):
    assert (
        len(coeffs_X.shape) == 2 and coeffs_X.shape[1] == 4
    ), f"coeffs_X must have shape (N,4) but has shape {coeffs_X.shape}"
    assert (
        len(coeffs_Y.shape) == 2 and coeffs_Y.shape[1] == 4
    ), f"coeffs_Y must have shape (N,4) but has shape {coeffs_Y.shape}"
    assert (
        coeffs_X.shape[0] == coeffs_Y.shape[0]
    ), f"coeffs_X and coeffs_Y must have the same length but have lengths {coeffs_X.shape[0]} and {coeffs_Y.shape[0]}"


def compute_spline_interval_lengths(
    coeffs_X: FloatArray, coeffs_Y: FloatArray, no_interp_points=100
):
    """
    computes the lengths of each spline portion of the path.
    > Note: Here the closeness of the part does not matter, it is contained in the coefficients

    :param coeff_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeff_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param delta_s: number of points to use on each spline portion for the interpolation
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    N = coeffs_X.shape[0]

    t_steps = np.linspace(0.0, 1.0, no_interp_points)[np.newaxis, :]
    interp_points = np.zeros((no_interp_points, N, 2))

    interp_points[:, :, 0] = coeffs_X[:, 0]
    interp_points[:, :, 1] = coeffs_Y[:, 0]

    coeffs_X = coeffs_X[:, np.newaxis, :]
    coeffs_Y = coeffs_Y[:, np.newaxis, :]

    interp_points = interp_points.transpose(1, 0, 2)

    interp_points[:, :, 0] += coeffs_X[:, :, 1] @ t_steps
    interp_points[:, :, 0] += coeffs_X[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 0] += coeffs_X[:, :, 3] @ np.power(t_steps, 3)

    interp_points[:, :, 1] += coeffs_Y[:, :, 1] @ t_steps
    interp_points[:, :, 1] += coeffs_Y[:, :, 2] @ np.power(t_steps, 2)
    interp_points[:, :, 1] += coeffs_Y[:, :, 3] @ np.power(t_steps, 3)

    delta_s = np.sum(
        np.sqrt(np.sum(np.power(np.diff(interp_points, axis=1), 2), axis=2)), axis=1
    )
    assert delta_s.shape == (N,), f"{delta_s.shape}"
    return delta_s


def uniformly_sample_spline(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    delta_s: FloatArray,
    n_samples: int,
):
    """
    uniformly n_samples equidistant points along the path defined by the splines.
    The first point will always be the initial point of the first spline portion, and
    the last point will NOT be the initial point of the first spline portion.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param spline_lengths: N array of lengths of the spline portions (as returned by calc_spline_lengths)
    :param n_samples: number of points to sample

    :return X_interp: n_samples array of X coordinates along the path
    :return Y_interp: n_samples array of Y coordinates along the path
    :return idx_interp: n_samples array of indices of the spline portions that host the points
    :return t_interp: n_samples array of t values of the points within their respective spline portions
    :return s_interp: n_samples array of distances along the path of the points
    """
    s = np.cumsum(delta_s)
    s_interp = np.linspace(0.0, s[-1], n_samples, endpoint=False)

    # find the spline that hosts the current interpolation point
    idx_interp = np.argmax(s_interp[:, np.newaxis] < s, axis=1)

    t_interp = np.zeros(n_samples)  # save t values
    X_interp = np.zeros(n_samples)  # raceline coords
    Y_interp = np.zeros(n_samples)  # raceline coords

    # get spline t value depending on the progress within the current element
    t_interp[idx_interp > 0] = (
        s_interp[idx_interp > 0] - s[idx_interp - 1][idx_interp > 0]
    ) / delta_s[idx_interp][idx_interp > 0]
    t_interp[idx_interp == 0] = s_interp[idx_interp == 0] / delta_s[0]

    # calculate coords
    X_interp = (
        coeffs_X[idx_interp, 0]
        + coeffs_X[idx_interp, 1] * t_interp
        + coeffs_X[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_X[idx_interp, 3] * np.power(t_interp, 3)
    )

    Y_interp = (
        coeffs_Y[idx_interp, 0]
        + coeffs_Y[idx_interp, 1] * t_interp
        + coeffs_Y[idx_interp, 2] * np.power(t_interp, 2)
        + coeffs_Y[idx_interp, 3] * np.power(t_interp, 3)
    )

    return X_interp, Y_interp, idx_interp, t_interp, s_interp


def get_heading(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    idx_interp: npt.NDArray[np.int64],
    t_interp: FloatArray,
) -> FloatArray:
    """
    analytically computes the heading and the curvature at each point along the path
    specified by idx_interp and t_interp.

    :param coeffs_X: Nx4 array of coefficients of the splines in the x direction (as returned by calc_splines)
    :param coeffs_Y: Nx4 array of coefficients of the splines in the y direction (as returned by calc_splines)
    :param idx_interp: n_samples array of indices of the spline portions that host the points
    :param t_interp: n_samples array of t values of the points within their respective spline portions
    """
    check_spline_coeffs_dims(coeffs_X, coeffs_Y)

    # we don't divide by delta_s[idx_interp] here because this term will cancel out
    # in arctan2 either way
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    phi = np.arctan2(y_d, x_d)

    return phi


def get_curvature(
    coeffs_X: FloatArray,
    coeffs_Y: FloatArray,
    idx_interp: npt.NDArray[np.int64],
    t_interp: FloatArray,
) -> FloatArray:
    # same here with the division by delta_s[idx_interp] ** 2
    x_d = (
        coeffs_X[idx_interp, 1]
        + 2 * coeffs_X[idx_interp, 2] * t_interp
        + 3 * coeffs_X[idx_interp, 3] * np.square(t_interp)
    )
    y_d = (
        coeffs_Y[idx_interp, 1]
        + 2 * coeffs_Y[idx_interp, 2] * t_interp
        + 3 * coeffs_Y[idx_interp, 3] * np.square(t_interp)
    )
    x_dd = 2 * coeffs_X[idx_interp, 2] + 6 * coeffs_X[idx_interp, 3] * t_interp
    y_dd = 2 * coeffs_Y[idx_interp, 2] + 6 * coeffs_Y[idx_interp, 3] * t_interp
    kappa = (x_d * y_dd - y_d * x_dd) / np.power(x_d**2 + y_d**2, 1.5)
    return kappa


class MotionPlanner:
    def __init__(
        self,
        center_line: FloatArray,
        n_samples: int = NUMBER_SPLINE_INTERVALS,
        v_ref=5.0,
    ):
        coeffs_X, coeffs_Y = fit_spline(path=center_line, curv_weight=2.0)
        delta_s = compute_spline_interval_lengths(coeffs_X=coeffs_X, coeffs_Y=coeffs_Y)
        X_ref, Y_ref, idx_interp, t_interp, s_ref = uniformly_sample_spline(
            coeffs_X=coeffs_X,
            coeffs_Y=coeffs_Y,
            delta_s=delta_s,
            n_samples=n_samples,
        )
        # kappa_ref = get_curvature(
        #     coeffs_X=coeffs_X,
        #     coeffs_Y=coeffs_Y,
        #     idx_interp=idx_interp,
        #     t_interp=t_interp,
        # )
        phi_ref = get_heading(coeffs_X, coeffs_Y, idx_interp, t_interp)
        kappa_ref = get_curvature(coeffs_X, coeffs_Y, idx_interp, t_interp)
        # v_ref = np.minimum(v_max, np.sqrt(a_lat_max / np.abs(kappa_ref)))

        lap_length = s_ref[-1] + np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])
        s_diff = np.append(
            np.diff(s_ref), np.hypot(X_ref[-1] - X_ref[0], Y_ref[-1] - Y_ref[0])
        )
        t_diff = s_diff / v_ref
        t_ref_extra = np.insert(np.cumsum(t_diff), 0, 0.0)
        lap_time = np.copy(t_ref_extra[-1])
        t_ref = t_ref_extra[:-1]

        self.lap_length = lap_length
        self.lap_time = lap_time
        self.s_ref = np.concatenate((s_ref - lap_length, s_ref, s_ref + lap_length))
        self.t_ref = np.concatenate((t_ref - lap_time, t_ref, t_ref + lap_time))
        self.X_ref = np.concatenate((X_ref, X_ref, X_ref))
        self.Y_ref = np.concatenate((Y_ref, Y_ref, Y_ref))
        self.phi_ref = unwrap_to_pi(np.concatenate((phi_ref, phi_ref, phi_ref)))
        self.v_ref = v_ref

    def project(
        self, X: float, Y: float, s_guess: float, tolerance: float = 10.0
    ) -> float:
        # extract all the points in X_ref, Y_ref assiciated with s_ref values within s_guess +- tolerance
        id_low = np.searchsorted(self.s_ref, s_guess - tolerance)
        id_up = np.searchsorted(self.s_ref, s_guess + tolerance)
        local_traj = np.array([self.X_ref[id_low:id_up], self.Y_ref[id_low:id_up]]).T

        # find the closest point to (X,Y) to find one segment extremity
        distances = np.linalg.norm(local_traj - np.array([X, Y]), axis=1)
        id_min = np.argmin(distances)

        # compute the angles between (X,Y), the closest point, and the next and previous points to find the second segment extremity
        def angle3pt(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angle_prev = angle3pt(
            np.array([X, Y]), local_traj[id_min], local_traj[id_min - 1]
        )
        angle_next = angle3pt(
            np.array([X, Y]), local_traj[id_min], local_traj[id_min + 1]
        )
        if angle_prev < angle_next:
            a = local_traj[id_min - 1]
            b = local_traj[id_min]
            sa = self.s_ref[id_low + id_min - 1]
            sb = self.s_ref[id_low + id_min]
        else:
            a = local_traj[id_min]
            b = local_traj[id_min + 1]
            sa = self.s_ref[id_low + id_min]
            sb = self.s_ref[id_low + id_min + 1]

        # project (X,Y) on the segment [a,b]
        ab = b - a
        lamda = np.dot(np.array([X, Y]) - a, ab) / np.dot(ab, ab)

        return sa + lamda * (sb - sa)

    def plan(
        self, X: float, Y: float, phi: float, v: float, s_guess: float
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
        # project current position on the reference trajectory and extract reference time of passage
        s0 = self.project(X, Y, s_guess)
        t0 = np.interp(s0, self.s_ref, self.t_ref)
        # sample reference values uniformly in time
        t_ref = dt * np.arange(Nf + 1) + t0
        s_ref = np.interp(t_ref, self.t_ref, self.s_ref)
        X_ref = np.interp(s_ref, self.s_ref, self.X_ref)
        Y_ref = np.interp(s_ref, self.s_ref, self.Y_ref)
        phi_ref = np.interp(s_ref, self.s_ref, self.phi_ref)
        v_ref = self.v_ref * np.ones(Nf + 1)
        # post-process the reference heading to make sure it is in the range [phi - pi, phi + pi))
        phi_ref = teds_projection(phi_ref, phi - np.pi)
        return s0, X_ref, Y_ref, phi_ref, v_ref

    def plot_motion_plan(
        self,
        center_line: FloatArray,
        blue_cones: FloatArray,
        yellow_cones: FloatArray,
        big_orange_cones: FloatArray,
        small_orange_cones: FloatArray,
        plot_title: str = "",
    ) -> None:
        plt.figure()
        plt.plot(self.s_ref, self.phi_ref, label="headings")
        plt.legend()
        plt.xlabel("track progress [m]")
        plt.ylabel("heading [rad]")
        plt.title(plot_title + " : reference heading/yaw profile")
        plt.tight_layout()

        plt.figure()
        plot_cones(
            blue_cones,
            yellow_cones,
            big_orange_cones,
            small_orange_cones,
            show=False,
        )
        plt.plot(self.X_ref, self.Y_ref, label="reference trajectory")
        plt.scatter(
            center_line[:, 0],
            center_line[:, 1],
            s=14,
            c="k",
            marker="x",
            label="center line",
        )
        plt.legend()
        plt.title(plot_title + " : reference trajectory")
        plt.tight_layout()


################################################################################
# track data
################################################################################


def load_center_line(filename: str) -> tuple[FloatArray, FloatArray]:
    """
    Loads the center line stored in CSV file specified by filename. This file must have
    the following format:
        X,Y,right_width,left_width
    Returns the center line as a numpy array of shape (N, 2) and the corresponding
    (right and left) track widths as a numpy array of shape (N,2).
    """
    arr = np.genfromtxt(filename, delimiter=",", dtype=float, skip_header=1)
    return arr[:, :2], arr[:, 2:]


def load_cones(
    filename: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Loads the cones stored in CSV file specified by filename. This file must have the
    following format:
        cone_type,X,Y,Z,std_X,std_Y,std_Z,right,left
    The returned arrays correspond to (in this order) the blue cones, yellow cones, big
    orange cones, small orange cones (possibly empty), right cones and left cones (all
    colors .
    """
    arr = np.genfromtxt(filename, delimiter=",", dtype=str, skip_header=1)
    blue_cones = arr[arr[:, 0] == "blue"][:, 1:3].astype(float)
    yellow_cones = arr[arr[:, 0] == "yellow"][:, 1:3].astype(float)
    big_orange_cones = arr[arr[:, 0] == "big_orange"][:, 1:3].astype(float)
    small_orange_cones = arr[arr[:, 0] == "small_orange"][:, 1:3].astype(float)
    right_cones = arr[arr[:, 7] == "1"][:, 1:3].astype(float)
    left_cones = arr[arr[:, 8] == "1"][:, 1:3].astype(float)
    return (
        blue_cones,
        yellow_cones,
        big_orange_cones,
        small_orange_cones,
        right_cones,
        left_cones,
    )


def plot_cones(
    blue_cones,
    yellow_cones,
    big_orange_cones,
    small_orange_cones,
    origin=np.zeros(2),
    show=True,
):
    plt.scatter(blue_cones[:, 0], blue_cones[:, 1], s=14, c="b", marker="^")
    plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], s=14, c="y", marker="^")
    plt.scatter(
        big_orange_cones[:, 0], big_orange_cones[:, 1], s=28, c="orange", marker="^"
    )
    try:
        plt.scatter(
            small_orange_cones[:, 0],
            small_orange_cones[:, 1],
            s=7,
            c="orange",
            marker="^",
        )
    except IndexError:
        pass
    plt.scatter(origin[0], origin[1], c="g", marker="x")
    plt.axis("equal")
    plt.tight_layout()
    if show:
        plt.show()


################################################################################
# closed loop simulation
################################################################################


def closed_loop(
    controller: Controller,
    Tsim: float = 70.0,
    v_ref: float = 5.0,
    track_name: str = "fsds_competition_1",
    data_file: str = "closed_loop_data.npz",
):
    """
    we store all the open loop predictions into big arrays that we dump into npz files
    we dump x_ref (nx x (Nf+1)), x_pred (nx x (Nf+1)), u_pred (nu x Nf)
    the current state is always the first element in x_ref

    with this dumped data we can:
    1. plot it with a slider
    2. train a new neural control policy using either DPC or imitation learning
    """
    # setup main simulation variables
    Nsim = int(Tsim / dt) + 1
    x_current = np.array([0.0, 0.0, np.pi / 2, 0.0])
    # x_current = np.array([0.5, 0.0, np.pi / 2 + 0.5, 0.0])
    s_guess = 0.0
    all_x_ref = []
    all_x_pred = []
    all_u_pred = []
    all_runtimes = []
    all_costs = []
    discrete_dynamics = get_discrete_dynamics_casadi()

    # import track data
    center_line, _ = load_center_line(f"data/tracks/{track_name}/center_line.csv")
    blue_cones, yellow_cones, big_orange_cones, small_orange_cones, _, _ = load_cones(
        f"data/tracks/{track_name}/cones.csv"
    )

    # create motion planner
    motion_planner = MotionPlanner(center_line, v_ref=v_ref)
    # motion_planner.plot_motion_plan(
    #     center_line,
    #     blue_cones,
    #     yellow_cones,
    #     big_orange_cones,
    #     small_orange_cones,
    #     "Motion Planner",
    # )
    progress_bar = trange(Nsim)
    for i in progress_bar:
        X = x_current[0]
        Y = x_current[1]
        phi = x_current[2]
        v = x_current[3]
        # construct the reference trajectory
        s_guess, X_ref, Y_ref, phi_ref, v_ref = motion_planner.plan(
            X, Y, phi, v, s_guess
        )
        # add data to arrays
        all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, v_ref)))
        # call controller
        try:
            x_pred, u_pred, stats = controller.control(
                X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref
            )
        except RuntimeError as e:
            print(f"Error in iteration {i}: {e}")
            break
        u_current = u_pred[0]
        progress_bar.set_description(
            f"Runtime: {1000 * stats.runtime:.2f} ms, cost: {stats.cost:.2f}"
        )
        # add data to arrays
        all_runtimes.append(stats.runtime)
        all_costs.append(stats.cost)
        all_x_pred.append(x_pred)
        all_u_pred.append(u_pred)
        # simulate next state
        x_current = discrete_dynamics(x_current, u_current).full().ravel()
        # check if we have completed a lap
        if s_guess > motion_planner.lap_length:
            print(f"Completed a lap in {i} iterations, i.e. {i * dt} s")
            break

    all_x_ref = np.array(all_x_ref)
    all_x_pred = np.array(all_x_pred)
    all_u_pred = np.array(all_u_pred)
    all_runtimes = np.array(all_runtimes)
    all_costs = np.array(all_costs)

    # save data to npz file
    np.savez(
        data_file,
        x_ref=all_x_ref,
        x_pred=all_x_pred,
        u_pred=all_u_pred,
        runtimes=all_runtimes,
        costs=all_costs,
        center_line=np.column_stack((motion_planner.X_ref, motion_planner.Y_ref)),
        blue_cones=blue_cones,
        yellow_cones=yellow_cones,
        big_orange_cones=big_orange_cones,
    )


################################################################################
# visualization
################################################################################


class VizMode(Enum):
    CLOSED_LOOP = "closed_loop"
    OPEN_LOOP = "open_loop"


def visualize_trajectories(
    x_ref: FloatArray,
    x_pred: FloatArray,
    u_pred: FloatArray,
    runtimes: FloatArray,
    costs: FloatArray,
    center_line: FloatArray,
    blue_cones: FloatArray,
    yellow_cones: FloatArray,
    big_orange_cones: FloatArray,
    viz_mode: VizMode = VizMode.CLOSED_LOOP,
    image_file: str = "",
    show: bool = True,
):
    """
    Creates 2 plots:
    1. a plot to display the evolution of the states and controls over time.
       It is constituted of the following subplots:
       +-------------------+-------------------+----------------+
       |                   | velocity v (m/s)  | trottle T (N)  |
       | trajectory XY (m) +-------------------+----------------+
       |                   | heading phi (deg) | steering (deg) |
       +-------------------+-------------------+----------------+
       underneath these subplots, a slider will allow to move through the time steps and to visualize the
       references given to the controller, as well as the predictions made by the controller.
    2. another plot to display the runtimes distribution (scatter plot superposed with a boxplot)
    """
    # plot runtime distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(1000 * runtimes, vert=False)
    ax.set_xlabel("runtime [ms]")
    ax.set_yticks([])
    ax.set_title("Runtime distribution")

    # plot cost evolution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(costs)), costs)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.set_title("Cost evolution")

    # the shapes should be:
    # x_ref : (Nsim, Nf+1, nx)
    # x_pred : (Nsim, Nf+1, nx)
    # u_pred : (Nsim, Nf, nx)
    # it can happen that an error occured during the run and there is one x_ref more than x_pred and u_pred
    assert x_pred.shape[0] == u_pred.shape[0]
    if x_ref.shape[0] == x_pred.shape[0]:
        controller_failed = False
    elif x_ref.shape[0] == x_pred.shape[0] + 1:
        controller_failed = True
    else:
        raise ValueError(
            f"x_ref has shape {x_ref.shape} and x_pred has shape {x_pred.shape}"
        )
    Nsim = x_ref.shape[0]
    # assert x_ref.shape == (Nsim, Nf + 1, nx)
    # assert x_pred.shape == (Nsim, Nf + 1, nx)
    # assert u_pred.shape == (Nsim, Nf, nu)

    # create grid plot
    gridshape = (2, 3)
    fig = plt.figure(figsize=(20, 9))
    axes: dict[str, matplotlib.axes.Axes] = {}
    lines: dict[str, dict[str, matplotlib.lines.Line2D]] = {}

    # define plot data
    plot_data = {
        "XY": {
            "loc": (0, 0),
            "xlabel": r"$X$ [m]",
            "ylabel": r"$Y$ [m]",
            "data": {
                "center_line": center_line,
                "blue_cones": blue_cones,
                "yellow_cones": yellow_cones,
                "big_orange_cones": big_orange_cones,
                "past": x_pred[:, 0, :2],
                "ref": x_ref[:, :, :2],
                "pred": x_pred[:, :, :2],
            },
        },
        "v": {
            "loc": (0, 1),
            "ylabel": r"$v$ [m/s]",
            "data": {
                "past": x_pred[:, 0, 3],
                "ref": x_ref[:, :, 3],
                "pred": x_pred[:, :, 3],
            },
        },
        "phi": {
            "loc": (1, 1),
            "ylabel": r"$\varphi$ [°]",
            "data": {
                "past": np.rad2deg(x_pred[:, 0, 2]),
                "ref": np.rad2deg(x_ref[:, :, 2]),
                "pred": np.rad2deg(x_pred[:, :, 2]),
            },
        },
        "T": {
            "loc": (0, 2),
            "ylabel": r"$T$ [N]",
            "data": {
                "past": u_pred[:, 0, 0],
                "pred": np.concatenate((u_pred[:, :, 0], u_pred[:, -1:, 0]), axis=1),
            },
        },
        "delta": {
            "loc": (1, 2),
            "ylabel": r"$\delta$ [°]",
            "data": {
                "past": np.rad2deg(u_pred[:, 0, 1]),
                "pred": np.rad2deg(
                    np.concatenate((u_pred[:, :, 1], u_pred[:, -1:, 1]), axis=1)
                ),
            },
        },
    }
    # custom matplotlib colors
    green = "#51BF63"
    orange = "#ff9b31"
    blue = "#1f77b4"
    red = "#ff5733"
    yellow = "#d5c904"
    # purple = "#7c00c6"

    # initialize axes and lines
    # TODO: add shared x axis for 1d subplots
    for subplot_name, subplot_info in plot_data.items():
        if subplot_name == "XY":
            # create axes
            axes[subplot_name] = plt.subplot2grid(
                gridshape, subplot_info["loc"], rowspan=2
            )
            # plot additional data that will not be updated
            axes[subplot_name].scatter(blue_cones[:, 0], blue_cones[:, 1], s=14, c=blue)
            axes[subplot_name].scatter(
                yellow_cones[:, 0], yellow_cones[:, 1], s=14, c=yellow
            )
            axes[subplot_name].scatter(
                big_orange_cones[:, 0], big_orange_cones[:, 1], s=28, c=orange
            )
            axes[subplot_name].plot(center_line[:, 0], center_line[:, 1], c="k")
            # plot data that will be update using the slider and store the lines
            lines[subplot_name] = {
                "past": axes[subplot_name].plot(
                    subplot_info["data"]["past"][:, 0],
                    subplot_info["data"]["past"][:, 1],
                    c=green,
                )[0],
                # at first we don't display references or predictions (only once we
                # activate the slider), so we just provide nan array with appropriate shape
                "ref": axes[subplot_name].plot(
                    np.full((Nf + 1,), np.nan),
                    np.full((Nf + 1,), np.nan),
                    c="cyan",
                    marker="o",
                    markersize=3,
                )[0],
                "pred": axes[subplot_name].plot(
                    np.full((Nf + 1,), np.nan),
                    np.full((Nf + 1,), np.nan),
                    c=red,
                    marker="o",
                    markersize=3,
                )[0],
            }
            # set aspect ratio to be equal (because we display a map)
            axes[subplot_name].set_aspect("equal")
            # if we are drawing open loop predictions, we only scale based on the references
            if viz_mode == VizMode.OPEN_LOOP:
                xlim = (
                    np.min(subplot_info["data"]["ref"][:, :, 0]),
                    np.max(subplot_info["data"]["ref"][:, :, 0]),
                )
                ylim = (
                    np.min(subplot_info["data"]["ref"][:, :, 1]),
                    np.max(subplot_info["data"]["ref"][:, :, 1]),
                )
                xlim = (
                    xlim[0] - 0.1 * (xlim[1] - xlim[0]),
                    xlim[1] + 0.1 * (xlim[1] - xlim[0]),
                )
                ylim = (
                    ylim[0] - 0.1 * (ylim[1] - ylim[0]),
                    ylim[1] + 0.1 * (ylim[1] - ylim[0]),
                )
                axes[subplot_name].set_xlim(xlim)
                axes[subplot_name].set_ylim(ylim)

        else:
            # create axes
            axes[subplot_name] = plt.subplot2grid(
                gridshape, subplot_info["loc"], rowspan=1
            )
            # plot data that will be update using the slider and store the lines
            lines[subplot_name] = (
                {
                    "past": axes[subplot_name].plot(
                        dt * np.arange(subplot_info["data"]["past"].shape[0]),
                        subplot_info["data"]["past"],
                        c=green,
                    )[0],
                    "ref": axes[subplot_name].plot(np.full((Nf,), np.nan), c="cyan")[0],
                    "pred": axes[subplot_name].plot(np.full((Nf,), np.nan), c=red)[0],
                }
                if subplot_name not in {"T", "delta"}
                else {
                    "past": axes[subplot_name].step(
                        dt * np.arange(subplot_info["data"]["past"].shape[0]),
                        subplot_info["data"]["past"],
                        c=green,
                        where="post",
                    )[0],
                    "pred": axes[subplot_name].step(
                        np.arange(Nf), np.full((Nf,), np.nan), c=red, where="post"
                    )[0],
                }
            )

        # if we defined some, add labels to the axes
        if "xlabel" in subplot_info:
            axes[subplot_name].set_xlabel(subplot_info["xlabel"])
        if "ylabel" in subplot_info:
            axes[subplot_name].set_ylabel(subplot_info["ylabel"])

    fig.tight_layout()

    # save plot to file
    if image_file != "":
        plt.savefig(image_file, dpi=300, bbox_inches="tight")

    # define update function
    def update(it):
        # compute time vectors for past, reference and prediction of 1d plots
        t_past = dt * np.arange(it + 1)
        t_ref = dt * np.arange(it, it + Nf + 1)
        if it == Nsim - 1 and controller_failed:
            # we don't have any state or control predictions to plot
            t_pred = np.full((Nf + 1), np.nan)
        else:
            t_pred = dt * np.arange(it, it + Nf + 1)

        # plot everything
        for subplot_name, subplot_info in plot_data.items():
            if subplot_name == "XY":
                lines[subplot_name]["past"].set_data(
                    subplot_info["data"]["past"][: it + 1, 0],
                    subplot_info["data"]["past"][: it + 1, 1],
                )
                lines[subplot_name]["pred"].set_data(
                    subplot_info["data"]["pred"][it, :, 0],
                    subplot_info["data"]["pred"][it, :, 1],
                )
                all_points = subplot_info["data"]["pred"][it]
                if not controller_failed or it < Nsim - 1:
                    lines[subplot_name]["ref"].set_data(
                        subplot_info["data"]["ref"][it, :, 0],
                        subplot_info["data"]["ref"][it, :, 1],
                    )
                    all_points = np.concatenate(
                        (
                            all_points,
                            subplot_info["data"]["ref"][it],
                        )
                    )

                old_xlim = axes[subplot_name].get_xlim()
                old_ylim = axes[subplot_name].get_ylim()
                old_aspect_ratio = (old_ylim[1] - old_ylim[0]) / (
                    old_xlim[1] - old_xlim[0]
                )
                # recompute the xlim and ylim based on subplot_info["data"]["ref"][it] and subplot_info["data"]["pred"][it]
                new_xlim = (all_points[:, 0].min(), all_points[:, 0].max())
                new_ylim = (all_points[:, 1].min(), all_points[:, 1].max())
                # post process xlim and ylim to make sure we have some margin
                new_xlim = (
                    new_xlim[0] - 0.1 * (new_xlim[1] - new_xlim[0]),
                    new_xlim[1] + 0.1 * (new_xlim[1] - new_xlim[0]),
                )
                new_ylim = (
                    new_ylim[0] - 0.1 * (new_ylim[1] - new_ylim[0]),
                    new_ylim[1] + 0.1 * (new_ylim[1] - new_ylim[0]),
                )
                # post process xlim and ylim to make sure we keep the same aspect ratio
                new_aspect_ratio = (new_ylim[1] - new_ylim[0]) / (
                    new_xlim[1] - new_xlim[0]
                )
                if new_aspect_ratio > old_aspect_ratio:
                    # we need to increase the x range
                    increase_ratio = new_aspect_ratio / old_aspect_ratio
                    mean = 0.5 * (new_xlim[0] + new_xlim[1])
                    diff = new_xlim[1] - new_xlim[0]
                    new_xlim = (
                        mean - 0.5 * diff * increase_ratio,
                        mean + 0.5 * diff * increase_ratio,
                    )
                else:
                    # we need to increase the y range
                    increase_ratio = old_aspect_ratio / new_aspect_ratio
                    mean = 0.5 * (new_ylim[0] + new_ylim[1])
                    diff = new_ylim[1] - new_ylim[0]
                    new_ylim = (
                        mean - 0.5 * diff * increase_ratio,
                        mean + 0.5 * diff * increase_ratio,
                    )

                # set the lims
                axes[subplot_name].set_xlim(new_xlim)
                axes[subplot_name].set_ylim(new_ylim)
            else:
                lines[subplot_name]["past"].set_data(
                    t_past, subplot_info["data"]["past"][: it + 1]
                )
                lines[subplot_name]["pred"].set_data(
                    t_pred, subplot_info["data"]["pred"][it]
                )
                all = subplot_info["data"]["pred"][it]
                if "ref" in subplot_info["data"]:
                    # we only plot reference for state variables
                    lines[subplot_name]["ref"].set_data(
                        t_ref, subplot_info["data"]["ref"][it]
                    )
                    all = np.concatenate((all, subplot_info["data"]["ref"][it]))

                # recompute xlim to center on the current time step and prediction
                new_xlim = (t_ref[0], t_ref[-1])
                new_ylim = (np.min(all), np.max(all))
                # add some margin
                new_xlim = (
                    new_xlim[0] - 0.1 * (new_xlim[1] - new_xlim[0]),
                    new_xlim[1] + 0.1 * (new_xlim[1] - new_xlim[0]),
                )
                new_ylim = (
                    new_ylim[0] - 0.1 * (new_ylim[1] - new_ylim[0]),
                    new_ylim[1] + 0.1 * (new_ylim[1] - new_ylim[0]),
                )
                # set lims
                axes[subplot_name].set_xlim(new_xlim)
                axes[subplot_name].set_ylim(new_ylim)

    # create slider
    slider_ax = fig.add_axes((0.125, 0.02, 0.775, 0.03))
    slider = matplotlib.widgets.Slider(
        ax=slider_ax,
        label="sim iteration",
        valmin=0,
        valmax=Nsim - 1,
        valinit=Nsim - 1,
        valstep=1,
        valfmt="%d",
    )
    slider.on_changed(update)

    # show plot
    if show:
        plt.show()


def visualize_trajectories_from_file(data_file: str, **kwargs):
    data = np.load(data_file)
    visualize_trajectories(**data, **kwargs)


if __name__ == "__main__":
    # NMPCController(solver="ipopt", jit=False, codegen=True)
    # run closed loop experiment with NMPC controller
    breakpoint()
    closed_loop(
        controller=NMPCController(solver="ipopt", jit=False),
        track_name="fsds_competition_1",
        data_file="closed_loop_data.npz",
    )
    visualize_trajectories_from_file(
        data_file="closed_loop_data.npz", image_file="closed_loop_data.png"
    )

    # ic(
    #     DPCController.generate_constant_curvature_trajectories(
    #         curvatures=np.linspace(-0.1, 0.1, 5)
    #     )
    # )
    # create DPC dataset
    # DPCController.create_pretraining_dataset(
    #     "data/dpc/dataset2.csv",
    #     # n_trajs=31,
    #     # n_lat=11,
    #     # n_phi=11,
    #     # n_v=21,
    #     n_trajs=5,
    #     n_lat=5,
    #     n_phi=5,
    #     n_v=5,
    # )
    # DPCController.create_finetuning_dataset(
    #     filename="data/dpc/finetuning/dataset.csv",
    #     n_samples=40000,
    #     sigma_curvature=0.05,
    #     sigma_lat=0.1,
    #     sigma_phi=0.1,
    #     sigma_v=0.5,
    # )

    net_config = {
        "nhidden": [512] * 2,
        "nonlinearity": "tanh",
    }

    # train DPC
    # DPCController.train(
    #     dataset_filename="data/dpc/finetuning/dataset.csv",
    #     num_epochs=300,
    #     lr=1e-4,
    #     # weight_decay=1.0,
    #     **net_config,
    #     # weights_filename="data/plan2.pth",
    #     # training_state_filename="best.ckpt",
    #     training_state_filename="data/first_encouraging.ckpt",
    # )

    # viz DPC open loop predictions
    # DPCController(
    #     **net_config,
    #     weights_file="best.ckpt",
    # ).compute_open_loop_predictions(
    #     dataset_filename="data/dpc/finetuning/dataset.csv",
    #     data_file="open_loop_data.npz",
    #     batch_sizes=(None, None),
    # )
    # visualize_trajectories_from_file(
    #     data_file="open_loop_data.npz",
    #     image_file="open_loop_data.png",
    #     viz_mode=VizMode.OPEN_LOOP,
    # )

    # run closed loop experiment with DPC controller
    # closed_loop(
    #     Tsim=5.0,
    #     controller=DPCController(
    #         **net_config,
    #         weights_file="best.ckpt",
    #         accelerator="cpu",
    #     ),
    #     track_name="fsds_competition_1",
    #     data_file="closed_loop_data.npz",
    # )
    # visualize_trajectories_from_file(
    #     data_file="closed_loop_data.npz", image_file="closed_loop_data.png"
    # )
