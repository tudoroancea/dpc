from dataclasses import dataclass
from abc import ABC, abstractmethod
from dpc.utils import FloatArray


@dataclass
class ControllerStats:
  runtime: float
  cost: float
  num_iters: int = 0


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
