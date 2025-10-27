import numpy as np
import numpy.typing as npt
from dpc.constants import dt

FloatArray = npt.NDArray[np.float64]


def teds_projection(x: FloatArray, a: float) -> FloatArray:
  """Projection of x onto the interval [a, a + 2*pi)"""
  return np.mod(x - a, 2 * np.pi) + a


def unwrap_to_pi(x: FloatArray) -> FloatArray:
  """remove discontinuities caused by wrapToPi"""
  diffs = np.diff(x)
  diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
  diffs[diffs < -1.5 * np.pi] += 2 * np.pi
  return np.insert(x[0] + np.cumsum(diffs), 0, x[0])


def rk4(f, x, *args):
  k1 = f(x, *args)
  k2 = f(x + dt / 2 * k1, *args)
  k3 = f(x + dt / 2 * k2, *args)
  k4 = f(x + dt * k3, *args)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
