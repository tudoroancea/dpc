import itertools
import platform
from abc import ABC, abstractmethod
from copy import copy
from time import perf_counter
from typing import OrderedDict

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
from casadi import SX, Function, cos, nlpsol, sin, tanh, vertcat
from icecream import ic
from lightning import Fabric
from qpsolvers import available_solvers, solve_qp
from scipy.sparse import csc_array
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from strongpods import PODS
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import trange

L.seed_everything(127)

# car mass and geometry
m = 230.0  # mass
wheelbase = 1.5706  # distance between the two axles
# drivetrain parameters (simplified)
C_m0 = 4.950
C_r0 = 297.030
C_r1 = 16.665
C_r2 = 0.6784

Nf = 40
nx = 4
nu = 2
dt = 1 / 20

T_max = 500.0
delta_max = 0.5

np.set_printoptions(precision=3, suppress=True, linewidth=200)
FloatArray = npt.NDArray[np.float64]


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


def get_continuous_dynamics() -> Function:
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


def get_discrete_dynamics() -> Function:
    x = SX.sym("x", nx)
    u = SX.sym("u", nu)
    f = get_continuous_dynamics()
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


class Controller(ABC):
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
        pass


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


class NMPCController(Controller):
    discrete_dynamics: Function
    solver: Function
    cost_weights: CostWeights

    def __init__(self, cost_weights: CostWeights = CostWeights(), **kwargs):
        self.discrete_dynamics = get_discrete_dynamics()
        self.cost_weights = cost_weights

        # optimization variables
        X_ref = SX.sym("X_ref", Nf + 1)
        Y_ref = SX.sym("Y_ref", Nf + 1)
        phi_ref = SX.sym("phi_ref", Nf + 1)
        v_ref = SX.sym("v_ref", Nf + 1)
        x = [SX.sym(f"x_{i}", nx) for i in range(Nf + 1)]
        u = [SX.sym(f"u_{i}", nu) for i in range(Nf)]
        parameters = vertcat(X_ref, Y_ref, phi_ref, v_ref)
        optimization_variables = vertcat(*x, *u)

        # construct cost function
        cost_function = 0.0
        for i in range(Nf):
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
            T = u[i][0]
            delta = u[i][1]
            T_ref = (C_r0 + C_r1 * v_ref[i] + C_r2 * v_ref[i] ** 2) / C_m0
            cost_function += (
                self.cost_weights.r_T * (T - T_ref) ** 2
                + self.cost_weights.r_delta * delta**2
            )

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

        # equality constraints
        eq_constraints = vertcat(
            *[self.discrete_dynamics(x[i], u[i]) - x[i + 1] for i in range(Nf)]
        )
        self.lbg = np.zeros(eq_constraints.shape[0])
        self.ubg = np.zeros(eq_constraints.shape[0])

        # simple bounds
        self.lbx = np.concatenate(
            (
                np.tile(np.array([-np.inf, -np.inf, -np.inf, -np.inf]), Nf + 1),
                np.tile(np.array([-T_max, -delta_max]), Nf),
            )
        )
        self.ubx = np.concatenate(
            (
                np.tile(np.array([np.inf, np.inf, np.inf, np.inf]), Nf + 1),
                np.tile(np.array([T_max, delta_max]), Nf),
            )
        )

        # assemble solver
        self.solver = nlpsol(
            "nmpc",
            "ipopt",
            {
                "x": optimization_variables,
                "f": cost_function,
                "g": eq_constraints,
                "p": parameters,
            },
            {
                "print_time": 0,
                "ipopt": {"sb": "yes", "print_level": 0},
            }
            | (
                {}
                if platform.system() == "Linux"
                else {
                    "jit": True,
                    "jit_options": {"flags": ["-O3 -march=native"], "verbose": False},
                }
            ),
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
        # set initial state
        x0 = np.array([X, Y, phi, v])
        self.lbx[:nx] = x0
        self.ubx[:nx] = x0

        # create guess
        # x = [x0]
        # u = []
        # for i in range(Nf):
        #     u.append(np.array([90.0 * (v_ref[i] - x[-1][3]), 0.0]))
        #     x.append(self.discrete_dynamics(x[-1], u[-1]).full().ravel())
        # initial_guess = np.concatenate(x + u)
        initial_guess = np.concatenate(
            (
                np.reshape(
                    np.column_stack((X_ref, Y_ref, phi_ref, v_ref)), (Nf + 1) * nx
                ),
                np.zeros(Nf * nu),
            )
        )
        # initial_guess = np.zeros((Nf + 1) * nx + Nf * nu)

        parameters = np.concatenate((X_ref, Y_ref, phi_ref, v_ref))
        # solve the optimization problem
        start = perf_counter()
        sol = self.solver(
            x0=initial_guess,
            p=parameters,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        stop = perf_counter()
        runtime = stop - start

        # extract solution
        opt_variables = sol["x"].full().ravel()
        last_prediction_x = opt_variables[: (Nf + 1) * nx].reshape(Nf + 1, nx).copy()
        last_prediction_u = opt_variables[(Nf + 1) * nx :].reshape(Nf, nu).copy()
        # last_prediction_x[0, :] = x0

        # check exit flag
        stats = self.solver.stats()
        if not stats["success"]:
            ic(stats)
            raise ValueError(stats["return_status"])

        return (
            last_prediction_x,
            last_prediction_u,
            ControllerStats(runtime=runtime, cost=sol["f"]),
        )


def create_dpc_dataset(
    filename: str, max_curvature=1 / 6, n_trajs=31, n_lat=11, n_phi=11, n_v=21
) -> None:
    # sample arcs of constant curvatures with constant speeds to create references -> 10x10=100
    # then create different initial conditions by perturbing e_lat, e_lon, e_phi, e_v. Vary the bounds on e_phi in function of the curvature -> 10^4 values
    # -> 10^6 samples, each of size nx x (Nf+1) = 4 x 41 = 84 -> ~85MB
    curvatures = np.linspace(-max_curvature, max_curvature, n_trajs)
    v_ref = 5.0
    s = v_ref * dt * np.arange(Nf + 1)
    reference_trajectories = []
    reference_headings = []
    plt.figure()
    for curvature in curvatures:
        if np.abs(curvature) < 1e-3:
            reference_trajectory = np.column_stack((s, np.zeros_like(s)))
            reference_heading = np.zeros_like(s)
        else:
            curvature_radius = np.abs(1 / curvature)
            angles = s / curvature_radius - np.pi / 2
            reference_trajectory = curvature_radius * np.column_stack(
                [np.cos(angles), np.sin(angles)]
            )
            reference_heading = angles
            reference_trajectory[:, 1] += curvature_radius
            reference_trajectory[:, 1] *= np.sign(curvature)

        plt.plot(reference_trajectory[:, 0], reference_trajectory[:, 1])
        reference_trajectories.append(reference_trajectory)
        reference_headings.append(reference_heading)

    plt.axis("equal")
    plt.show()

    # now associated perturbed initial state
    lateral_errors = np.linspace(-0.5, 0.5, n_lat)
    heading_errors = np.linspace(-0.5, 0.5, n_phi)
    vel_errors = np.linspace(-5.0, 5.0, n_v)

    df = []
    for (traj, phi_ref), Y, phi, vel_err in itertools.product(
        zip(reference_trajectories, reference_headings),
        lateral_errors,
        heading_errors,
        vel_errors,
    ):
        X = 0.0
        v = v_ref + vel_err
        X_ref = traj[:, 0]
        Y_ref = traj[:, 1]
        # ic(X,Y,phi,v, v_ref, vel_err)
        df.append(
            np.concatenate(
                (
                    np.array([X, Y, phi, v]),
                    np.reshape(
                        np.column_stack(
                            (X_ref, Y_ref, phi_ref, v_ref * np.ones_like(s))
                        ),
                        nx * (Nf + 1),
                    ),
                )
            )
        )
    df = np.array(df)

    np.savetxt(
        filename,
        df,
        fmt="%.5f",
        delimiter=",",
        comments="",
        header="X,Y,phi,v,"
        + ",".join(
            [f"X_ref_{i},Y_ref_{i},phi_ref_{i},v_ref_{i}" for i in range(Nf + 1)]
        ),
    )


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


def load_dataset(
    filename: str,
    train_data_proportion: float = 0.8,
    batch_sizes: tuple[int | None, int | None] = (None, None),
) -> tuple[DPCDataset, DataLoader, DataLoader]:
    # load the dataset
    dataset = DPCDataset(filename)
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
            batch_size=train_data_size if batch_sizes[0] is None else batch_sizes[0],
            shuffle=True,
        ),
        DataLoader(
            val_data,
            batch_size=len(val_data) if batch_sizes[1] is None else batch_sizes[1],
            shuffle=True,
        ),
    )


class ControlConstraintScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([T_max, delta_max]), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.scale * F.tanh(input.reshape(input.shape[0], Nf, nu))


class DPCController(Controller):
    net: nn.Module
    discrete_dynamics: Function
    fabric: Fabric

    def __init__(
        self, nhidden: list[int], nonlinearity: str, weights_file: str | None = None
    ):
        self.discrete_dynamics = get_discrete_dynamics()
        self.fabric = Fabric()
        self.net = self.fabric.setup(
            DPCController.construct_net(
                nin=(Nf + 2) * nx,
                nout=Nf * nu,
                nhidden=nhidden,
                nonlinearity=nonlinearity,
            )
        )
        ic(self.net)

    @staticmethod
    def construct_net(
        nin: int,
        nout: int,
        nhidden: list[int] = [128, 128, 128],
        nonlinearity: str = "relu",
    ):
        assert len(nhidden) >= 1
        nonlinearity_function = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }[nonlinearity]

        di = {
            "batchnorm": nn.BatchNorm1d(nin),
            "hidden_layer_0": nn.Linear(nin, nhidden[0], bias=True),
            "nonlinearity_0": nonlinearity_function,
        }
        nn.init.xavier_uniform_(
            di["hidden_layer_0"].weight, gain=nn.init.calculate_gain(nonlinearity)
        )
        for i in range(1, len(nhidden)):
            di.update(
                {
                    f"hidden_layer_{i}": nn.Linear(
                        nhidden[i - 1], nhidden[i], bias=True
                    ),
                    f"nonlinearity_{i}": nonlinearity_function,
                }
            )
            nn.init.xavier_uniform_(
                di[f"hidden_layer_{i}"].weight,
                gain=nn.init.calculate_gain(nonlinearity),
            )
        di.update(
            {
                "output_layer": nn.Linear(nhidden[-1], nout, bias=True),
                "ouput_scaling": ControlConstraintScale(),
            }
        )
        nn.init.xavier_uniform_(
            di["output_layer"].weight, gain=nn.init.calculate_gain(nonlinearity)
        )

        return nn.Sequential(OrderedDict(di))

    @staticmethod
    def compute_mpc_loss(
        x_ref: torch.Tensor,
        x_pred: torch.Tensor,
        u_pred: torch.Tensor,
        cost_weights: CostWeights,
    ) -> torch.Tensor:
        """
        :param x_ref: shape (nbatch, Nf+1, nx)
        :param x_pred: shape (nbatch, Nf+1, nx)
        :param u_pred: shape (nbatch, Nf, nx)
        """
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
                )
            )
        )  # shape (nbatch, Nf+1, 2)
        T_ref = (
            C_r0 + C_r1 * v_ref[:, :-1] + C_r2 * v_ref[:, :-1] * v_ref[:, :-1]
        ) / C_m0  # shape (nbatch, Nf)
        errors_by_sample = (
            # stage longitudinal errors
            cost_weights.q_lon * torch.sum(lon_lat_errs_sq[:, :-1, 0], dim=1)
            # terminal longitudinal errors
            + cost_weights.q_lon_f * lon_lat_errs_sq[:, -1, 0]
            # stage lateral errors
            + cost_weights.q_lat * torch.sum(lon_lat_errs_sq[:, :-1, 1], dim=1)
            # terminal lateral errors
            + cost_weights.q_lat_f * lon_lat_errs_sq[:, -1, 1]
            # stage heading errors
            + cost_weights.q_phi
            * torch.sum(torch.square(x_pred[:, :-1, 2] - x_ref[:, :-1, 2]), dim=1)
            # terminal heading errors
            + cost_weights.q_phi_f * torch.square(x_pred[:, -1, 2] - x_ref[:, -1, 2])
            # stage velocity errors
            + cost_weights.q_v
            * torch.sum(torch.square(x_pred[:, :-1, 3] - x_ref[:, :-1, 3]), dim=1)
            # terminal velocity errors
            + cost_weights.q_v_f * torch.square(x_pred[:, -1, 3] - x_ref[:, -1, 3])
            # throttle errors
            + cost_weights.r_T * torch.sum(torch.square(u_pred[:, :, 0] - T_ref), dim=1)
            # steering errors
            + cost_weights.r_delta * torch.sum(torch.square(u_pred[:, :, 1]), dim=1)
        )
        return torch.mean(errors_by_sample)

    def run_model(self, batch: torch.Tensor, cost_weights: CostWeights) -> torch.Tensor:
        # extract current and reference states from batch
        nbatch = len(batch)
        x = batch[:, :nx]
        x_ref = batch[:, nx:].reshape(nbatch, Nf + 1, nx)
        # run the network to compute predicted controls
        u_pred = self.net(batch)
        # run the model to compute the predicted states
        x_pred = unrolled_discrete_dynamics_pytorch(x, u_pred)
        # compute MPC cost function
        return DPCController.compute_mpc_loss(x_pred, x_ref, u_pred, cost_weights)

    @classmethod
    def from_scratch(
        cls,
        dataset_filename: str,
        num_epochs: int,
        nhidden: list[int],
        nonlinearity: str,
        lr: float = 1e-3,
        batch_sizes=(None, None),
        cost_weights: CostWeights = CostWeights(),
    ):
        controller = cls(nhidden, nonlinearity)
        optimizer = torch.optim.AdamW(controller.net.parameters(), lr=lr)
        optimizer = controller.fabric.setup_optimizers(optimizer)
        _, train_dataloader, val_dataloader = load_dataset(
            dataset_filename,
            train_data_proportion=0.8,
            batch_sizes=batch_sizes,
        )
        ic(train_dataloader.batch_size, val_dataloader.batch_size)
        train_dataloader, val_dataloader = controller.fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )
        best_val_loss = np.inf

        progress_bar = trange(num_epochs)
        for epoch in progress_bar:
            # training step
            controller.net.train()
            total_train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()
                train_loss = controller.run_model(batch, cost_weights)
                controller.fabric.backward(train_loss)
                optimizer.step()
                total_train_loss += train_loss.item()
            total_train_loss /= len(train_dataloader)

            # validation step
            controller.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    val_loss += controller.run_model(batch, cost_weights).item()
            val_loss /= len(val_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(controller.net.state_dict(), "best_model.pth")

            # logging
            progress_bar.set_description(
                f"Epoch {epoch}, train loss: {total_train_loss:.4f}, val loss: {val_loss:.4f}, best val loss: {best_val_loss:.4f}"
            )

        torch.save(controller.net.state_dict(), "final_model.pth")

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
                ]
            ]
        )
        current_state = np.array([X - X_ref[0], Y - Y_ref[0], phi - phi_ref_0, v])
        current_state[:2] = R @ current_state[:2]
        current_state[2] -= phi_ref_0
        ref = np.column_stack((X_ref, Y_ref, phi_ref, v_ref))
        ref[:, :2] = ref[:, :2] @ R.T
        # assemble inputs into a tensor
        input = torch.unsqueeze(
            torch.tensor(
                np.concatenate(
                    (
                        np.array([X, Y, phi, v]),
                        np.column_stack((X_ref, Y_ref, phi_ref, v_ref)),
                    )
                ),
                dtype=torch.float32,
                requires_grad=False,
            ).to(device=self.net.device),
            0,
        )
        # forward pass of the net on the data
        start = perf_counter()
        output = self.net(input)
        stop = perf_counter()
        # reformat the output
        u_pred = output.squeeze().detach().cpu().numpy().reshape(Nf, nu)
        x_pred = [np.array([X, Y, phi, v])]
        for i in range(Nf):
            x_pred.append(self.discrete_dynamics(x_pred[-1], u_pred[i]).full().ravel())
        x_pred = np.array(x_pred)

        return x_pred, u_pred, ControllerStats(runtime=stop - start, cost=0.0)


################################################################################
# motion planner
################################################################################

NUMBER_SPLINE_INTERVALS = 500


def fit_spline(
    path: FloatArray,
    curv_weight: float = 1.0,
    qp_solver: str = "proxqp",
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
    assert (
        qp_solver in available_solvers
    ), f"qp_solver must be one of the available solvers: {available_solvers}"

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
                    (3 * np.arange(N), 1 + 3 * np.arange(N), 2 + 3 * np.arange(N))
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

    if qp_solver in {"quadprog", "ecos"}:
        A = A.toarray()
        B = B.toarray()
        C = C.toarray()
        P = P.toarray()

    # solve the QP for X and Y separately
    p_X = solve_qp(P=P, q=q[:, 0], A=A, b=b, solver=qp_solver)
    p_Y = solve_qp(P=P, q=q[:, 1], A=A, b=b, solver=qp_solver)
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
        coeffs_X, coeffs_Y = fit_spline(
            path=center_line, curv_weight=2.0, qp_solver="proxqp"
        )
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
        ic(kappa_ref.min(), kappa_ref.max())
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

    Tsim = 70.0
    Nsim = int(Tsim / dt) + 1
    v_ref = 5.0
    x_current = np.array([0.0, 0.0, np.pi / 2, 0.0])
    s_guess = 0.0
    all_x_ref = []
    all_x_pred = []
    all_u_pred = []
    all_runtimes = []
    discrete_dynamics = get_discrete_dynamics()

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
        except ValueError as e:
            print(f"Error in iteration {i}: {e}")
            break
        u_current = u_pred[0]
        progress_bar.set_description(
            f"Runtime: {1000*stats.runtime:.2f} ms, cost: {stats.cost:.2f}"
        )
        # add data to arrays
        all_runtimes.append(stats.runtime)
        all_x_pred.append(x_pred.copy())
        all_u_pred.append(u_pred.copy())
        # simulate next state
        x_current = discrete_dynamics(x_current, u_current).full().ravel()
        # check if we have completed a lap
        if s_guess > motion_planner.lap_length:
            print(f"Completed a lap in {i} iterations, i.e. {i * dt} s")
            break

    all_runtimes = np.array(all_runtimes)
    all_x_ref = np.array(all_x_ref)
    all_x_pred = np.array(all_x_pred)
    all_u_pred = np.array(all_u_pred)

    # save data to npz file
    np.savez(
        data_file,
        x_ref=all_x_ref,
        x_pred=all_x_pred,
        u_pred=all_u_pred,
        runtimes=all_runtimes,
        center_line=np.column_stack((motion_planner.X_ref, motion_planner.Y_ref)),
        blue_cones=blue_cones,
        yellow_cones=yellow_cones,
        big_orange_cones=big_orange_cones,
    )


################################################################################
# visualization
################################################################################


def visualize(
    x_ref: FloatArray,
    x_pred: FloatArray,
    u_pred: FloatArray,
    runtimes: FloatArray,
    center_line: FloatArray,
    blue_cones: FloatArray,
    yellow_cones: FloatArray,
    big_orange_cones: FloatArray,
    output_file: str = "",
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
    purple = "#7c00c6"
    yellow = "#d5c904"

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
                    np.full((Nf + 1,), np.nan), np.full((Nf + 1,), np.nan), c="cyan"
                )[0],
                "pred": axes[subplot_name].plot(
                    np.full((Nf + 1,), np.nan), np.full((Nf + 1,), np.nan), c=red
                )[0],
            }
            # set aspect ratio to be equal (because we display a map)
            axes[subplot_name].set_aspect("equal")
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
    if output_file != "":
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

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
                        (all_points, subplot_info["data"]["ref"][it])
                    )
            else:
                lines[subplot_name]["past"].set_data(
                    t_past, subplot_info["data"]["past"][: it + 1]
                )
                lines[subplot_name]["pred"].set_data(
                    t_pred, subplot_info["data"]["pred"][it]
                )
                if "ref" in subplot_info["data"]:
                    # we only plot reference for state variables
                    lines[subplot_name]["ref"].set_data(
                        t_ref, subplot_info["data"]["ref"][it]
                    )
                # recompute the ax.dataLim
                axes[subplot_name].relim()
                # update ax.viewLim using the new dataLim
                axes[subplot_name].autoscale_view()

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


def visualize_file(filename: str):
    data = np.load(filename)
    visualize(**data, output_file="bruh.png", show=True)


if __name__ == "__main__":
    # run closed loop experiment with NMPC controller
    # closed_loop(
    #     controller=NMPCController(),
    #     track_name="fsds_competition_1",
    #     data_file="closed_loop_data.npz",
    # )
    # visualize_file("closed_loop_data.npz")

    # create DPC dataset
    # create_dpc_dataset("data/dpc/dataset.csv")

    # train DPC
    DPCController.from_scratch(
        dataset_filename="data/dpc/dataset.csv",
        num_epochs=50,
        lr=5e-2,
        nhidden=[128, 128, 128],
        # batch_sizes=(10000, 5000),
        nonlinearity="tanh",
    )
