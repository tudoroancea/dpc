from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from dpc.nmpc import NMPCController
from dpc.track_data import load_center_line, load_cones
from dpc.motion_planning import MotionPlanner
from dpc.closed_loop import closed_loop
from dpc.viz import visualize_trajectories_from_file

np.random.seed(127)
np.set_printoptions(precision=3, suppress=True, linewidth=200)


def plot_track():
  parser = ArgumentParser(prog="plot_track")
  parser.add_argument("--track", default="fsds_competition_1")
  args = parser.parse_args()

  # import track data
  center_line, _ = load_center_line(f"data/tracks/{args.track}/center_line.csv")
  blue_cones, yellow_cones, big_orange_cones, small_orange_cones, _, _ = load_cones(f"data/tracks/{args.track}/cones.csv")

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
  plt.show()


def closed_loop_nmpc():
  parser = ArgumentParser(prog="closed_loop_nmpc")
  parser.add_argument("--solver", default="ipopt")
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--viz", action="store_true")
  parser.add_argument("--track", default="fsds_competition_1")
  parser.add_argument("--vref", type=float, default=5.0)
  args = parser.parse_args()
  closed_loop(
    controller=NMPCController(solver=args.solver, jit=args.jit),
    track_name=args.track,
    data_file="closed_loop_data.npz",
    v_ref=args.vref,
  )
  if args.viz:
    visualize_trajectories_from_file(data_file="closed_loop_data.npz", image_file="closed_loop_data.png")


def codegen_nmpc():
  parser = ArgumentParser(prog="codegen_nmpc")
  parser.add_argument("--solver", default="ipopt")
  args = parser.parse_args()
  NMPCController(solver=args.solver).codegen()


# def create_dpc_dataset():
#     ic(
#         DPCController.generate_constant_curvature_trajectories(
#             curvatures=np.linspace(-0.1, 0.1, 5)
#         )
#     )
#     # create DPC dataset
#     DPCController.create_pretraining_dataset(
#         "data/dpc/dataset2.csv",
#         # n_trajs=31,
#         # n_lat=11,
#         # n_phi=11,
#         # n_v=21,
#         n_trajs=5,
#         n_lat=5,
#         n_phi=5,
#         n_v=5,
#     )
#     DPCController.create_finetuning_dataset(
#         filename="data/dpc/finetuning/dataset.csv",
#         n_samples=40000,
#         sigma_curvature=0.05,
#         sigma_lat=0.1,
#         sigma_phi=0.1,
#         sigma_v=0.5,
#     )


# def train_dpc():
#     net_config = {
#         "nhidden": [512] * 2,
#         "nonlinearity": "tanh",
#     }
#     DPCController.train(
#         dataset_filename="data/dpc/finetuning/dataset.csv",
#         num_epochs=300,
#         lr=1e-4,
#         # weight_decay=1.0,
#         **net_config,
#         # weights_filename="data/plan2.pth",
#         # training_state_filename="best.ckpt",
#         training_state_filename="data/first_encouraging.ckpt",
#     )


# def open_loop_dpc():
#     net_config = {
#         "nhidden": [512] * 2,
#         "nonlinearity": "tanh",
#     }
#     DPCController(
#         **net_config,
#         weights_file="best.ckpt",
#     ).compute_open_loop_predictions(
#         dataset_filename="data/dpc/finetuning/dataset.csv",
#         data_file="open_loop_data.npz",
#         batch_sizes=(None, None),
#     )
#     closed_loop_visualization_from_file(
#         data_file="open_loop_data.npz",
#         image_file="open_loop_data.png",
#         viz_mode=VizMode.OPEN_LOOP,
#     )


# def closed_loop_dpc():
#     net_config = {
#         "nhidden": [512] * 2,
#         "nonlinearity": "tanh",
#     }
#     # run closed loop experiment with DPC controller
#     closed_loop_simulation(
#         Tsim=5.0,
#         controller=DPCController(
#             **net_config,
#             weights_file="best.ckpt",
#             accelerator="cpu",
#         ),
#         track_name="fsds_competition_1",
#         data_file="closed_loop_data.npz",
#     )
#     closed_loop_visualization_from_file(
#         data_file="closed_loop_data.npz", image_file="closed_loop_data.png"
#     )


def closed_loop_visualization():
  parser = ArgumentParser(prog="closed_loop_visualization")
  parser.add_argument("--data_file", default="closed_loop_data.npz")
  parser.add_argument("--image_file", default="closed_loop_data.png")
  args = parser.parse_args()
  visualize_trajectories_from_file(data_file=args.data_file, image_file=args.image_file)

def debug_fatrop():
  # copied from https://github.com/jgillis/fatrop_demo/blob/master/debug_fatrop.py
  import matplotlib.pyplot as plt
  import casadi as ca

  actual = ca.Sparsity.from_file("debug_fatrop_actual.mtx")

  A = ca.Sparsity.from_file("debug_fatrop_A.mtx")
  B = ca.Sparsity.from_file("debug_fatrop_B.mtx")
  C = ca.Sparsity.from_file("debug_fatrop_C.mtx")
  D = ca.Sparsity.from_file("debug_fatrop_D.mtx")
  I = ca.Sparsity.from_file("debug_fatrop_I.mtx")
  errors = ca.Sparsity.from_file("debug_fatrop_errors.mtx").row()

  plt.figure(figsize=(6,10))
  plt.spy(A,marker='o',color='r',markersize=5,label="expected A",markerfacecolor="white")
  plt.spy(B,marker='o',color='b',markersize=5,label="expected B",markerfacecolor="white")
  plt.spy(C,marker='o',color='g',markersize=5,label="expected C",markerfacecolor="white")
  plt.spy(D,marker='o',color='y',markersize=5,label="expected D",markerfacecolor="white")
  plt.spy(I,marker='o',color='k',markersize=5,label="expected I",markerfacecolor="white")
  plt.spy(actual,marker='o',color='k',markersize=2,label="actual")

  plt.hlines(errors, 0, A.shape[1],color='gray', linestyle='-',label="offending rows")

  plt.title("Debug view of fatrop interface structure detection")
  plt.legend()
  plt.tight_layout()
  plt.show()
