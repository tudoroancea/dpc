from dpc.utils import FloatArray
import numpy as np


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
