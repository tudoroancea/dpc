import itertools
import json
import os
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count
from time import perf_counter
from typing import Literal, OrderedDict

import casadi as ca
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
from icecream import ic
from lightning import Fabric
from qpsolvers import solve_qp
from scipy.sparse import csc_array
from scipy.sparse import eye as speye
from scipy.sparse import kron as spkron
from torch.utils.data import DataLoader, Dataset, random_split


L.seed_everything(127)





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
        _, axs = plt.subplots(2, 2)
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
