import numpy as np
from tqdm import trange

from dpc.constants import dt
from dpc.controller import Controller
from dpc.motion_planning import MotionPlanner
from dpc.nmpc import get_discrete_dynamics_casadi
from dpc.track_data import load_center_line, load_cones


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
  s_guess = 0.0
  all_x_ref = []
  all_x_pred = []
  all_u_pred = []
  all_runtimes = []
  all_costs = []
  discrete_dynamics = get_discrete_dynamics_casadi()

  # import track data
  center_line, _ = load_center_line(f"data/tracks/{track_name}/center_line.csv")
  blue_cones, yellow_cones, big_orange_cones, _, _, _ = load_cones(f"data/tracks/{track_name}/cones.csv")

  # create motion planner
  motion_planner = MotionPlanner(center_line, v_ref=v_ref)
  progress_bar = trange(Nsim)
  for i in progress_bar:
    X = x_current[0]
    Y = x_current[1]
    phi = x_current[2]
    v = x_current[3]
    # construct the reference trajectory
    s_guess, X_ref, Y_ref, phi_ref, v_ref = motion_planner.plan(X, Y, phi, s_guess)
    # TODO: add frenet stuff
    # add data to arrays
    all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, v_ref)))
    # call controller
    try:
      x_pred, u_pred, stats = controller.control(X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref)
    except RuntimeError as e:
      print(f"Error in iteration {i}: {e}")
      break
    u_current = u_pred[0]
    progress_bar.set_description(f"Runtime: {1000 * stats.runtime:.2f} ms, cost: {stats.cost:.2f}")
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
