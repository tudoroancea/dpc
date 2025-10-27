from enum import Enum
from dpc.utils import FloatArray
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.axes
import matplotlib.lines
import matplotlib.widgets
import numpy as np
from dpc.constants import Nf, dt


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
    raise ValueError(f"x_ref has shape {x_ref.shape} and x_pred has shape {x_pred.shape}")
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
        "pred": np.rad2deg(np.concatenate((u_pred[:, :, 1], u_pred[:, -1:, 1]), axis=1)),
      },
    },
  }
  # custom matplotlib colors
  green = "#51BF63"
  orange = "#ff9b31"
  blue = "#1f77b4"
  red = "#ff5733"
  yellow = "#d5c904"
  purple = "#7c00c6"

  # initialize axes and lines
  # TODO: add shared x axis for 1d subplots
  for subplot_name, subplot_info in plot_data.items():
    if subplot_name == "XY":
      # create axes
      axes[subplot_name] = plt.subplot2grid(gridshape, subplot_info["loc"], rowspan=2)
      # plot additional data that will not be updated
      axes[subplot_name].scatter(blue_cones[:, 0], blue_cones[:, 1], s=14, c=blue)
      axes[subplot_name].scatter(yellow_cones[:, 0], yellow_cones[:, 1], s=14, c=yellow)
      axes[subplot_name].scatter(big_orange_cones[:, 0], big_orange_cones[:, 1], s=28, c=orange)
      axes[subplot_name].plot(center_line[:, 0], center_line[:, 1], c="k")
      axes[subplot_name].scatter(x_pred[0, 0, 0], x_pred[0, 0, 1], c=purple, marker="x")
      # plot initial heading with an arrow
      axes[subplot_name].arrow(
        x_pred[0, 0, 0],
        x_pred[0, 0, 1],
        np.cos(x_pred[0, 0, 2]) * np.sqrt(2),
        np.sin(x_pred[0, 0, 2]) * np.sqrt(2),
        head_width=0.1,
        head_length=0.1,
        fc=purple,
        ec=purple,
      )

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
      axes[subplot_name] = plt.subplot2grid(gridshape, subplot_info["loc"], rowspan=1)
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
          "pred": axes[subplot_name].step(np.arange(Nf), np.full((Nf,), np.nan), c=red, where="post")[0],
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
        old_aspect_ratio = (old_ylim[1] - old_ylim[0]) / (old_xlim[1] - old_xlim[0])
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
        new_aspect_ratio = (new_ylim[1] - new_ylim[0]) / (new_xlim[1] - new_xlim[0])
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
        lines[subplot_name]["past"].set_data(t_past, subplot_info["data"]["past"][: it + 1])
        lines[subplot_name]["pred"].set_data(t_pred, subplot_info["data"]["pred"][it])
        all = subplot_info["data"]["pred"][it]
        if "ref" in subplot_info["data"]:
          # we only plot reference for state variables
          lines[subplot_name]["ref"].set_data(t_ref, subplot_info["data"]["ref"][it])
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
