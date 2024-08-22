from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from dpc.core import (
    DPCController,
    MotionPlanner,
    NMPCController,
    VizMode,
    closed_loop,
    load_center_line,
    load_cones,
    visualize_trajectories_from_file,
)


def plot_track():
    parser = ArgumentParser(prog="plot_track")
    parser.add_argument("--track", default="fsds_competition_1")
    args = parser.parse_args()

    # import track data
    center_line, _ = load_center_line(f"data/tracks/{args.track}/center_line.csv")
    blue_cones, yellow_cones, big_orange_cones, small_orange_cones, _, _ = load_cones(
        f"data/tracks/{args.track}/cones.csv"
    )

    # create motion planner
    motion_planner = MotionPlanner(center_line, v_ref=5.0)
    motion_planner.plot_motion_plan(
        center_line,
        blue_cones,
        yellow_cones,
        big_orange_cones,
        small_orange_cones,
        "Motion Planner",
    )
    ic()
    plt.show()


def closed_loop_nmpc():
    parser = ArgumentParser(prog="closed_loop_nmpc")
    parser.add_argument("--solver", default="ipopt")
    parser.add_argument("--jit", default=False)
    parser.add_argument("--viz", default=True)
    parser.add_argument("--track", default="fsds_competition_1")
    args = parser.parse_args()
    closed_loop(
        controller=NMPCController(solver=args.solver, jit=args.jit),
        track_name=args.track,
        data_file="closed_loop_data.npz",
    )
    if args.viz:
        visualize_trajectories_from_file(
            data_file="closed_loop_data.npz", image_file="closed_loop_data.png"
        )


def create_dpc_dataset():
    ic(
        DPCController.generate_constant_curvature_trajectories(
            curvatures=np.linspace(-0.1, 0.1, 5)
        )
    )
    # create DPC dataset
    DPCController.create_pretraining_dataset(
        "data/dpc/dataset2.csv",
        # n_trajs=31,
        # n_lat=11,
        # n_phi=11,
        # n_v=21,
        n_trajs=5,
        n_lat=5,
        n_phi=5,
        n_v=5,
    )
    DPCController.create_finetuning_dataset(
        filename="data/dpc/finetuning/dataset.csv",
        n_samples=40000,
        sigma_curvature=0.05,
        sigma_lat=0.1,
        sigma_phi=0.1,
        sigma_v=0.5,
    )


def train_dpc():
    net_config = {
        "nhidden": [512] * 2,
        "nonlinearity": "tanh",
    }
    DPCController.train(
        dataset_filename="data/dpc/finetuning/dataset.csv",
        num_epochs=300,
        lr=1e-4,
        # weight_decay=1.0,
        **net_config,
        # weights_filename="data/plan2.pth",
        # training_state_filename="best.ckpt",
        training_state_filename="data/first_encouraging.ckpt",
    )


def open_loop_dpc():
    net_config = {
        "nhidden": [512] * 2,
        "nonlinearity": "tanh",
    }
    DPCController(
        **net_config,
        weights_file="best.ckpt",
    ).compute_open_loop_predictions(
        dataset_filename="data/dpc/finetuning/dataset.csv",
        data_file="open_loop_data.npz",
        batch_sizes=(None, None),
    )
    visualize_trajectories_from_file(
        data_file="open_loop_data.npz",
        image_file="open_loop_data.png",
        viz_mode=VizMode.OPEN_LOOP,
    )


def closed_loop_dpc():
    net_config = {
        "nhidden": [512] * 2,
        "nonlinearity": "tanh",
    }
    # run closed loop experiment with DPC controller
    closed_loop(
        Tsim=5.0,
        controller=DPCController(
            **net_config,
            weights_file="best.ckpt",
            accelerator="cpu",
        ),
        track_name="fsds_competition_1",
        data_file="closed_loop_data.npz",
    )
    visualize_trajectories_from_file(
        data_file="closed_loop_data.npz", image_file="closed_loop_data.png"
    )
