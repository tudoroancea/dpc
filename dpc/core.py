


################################################################################
# car and problem parameters
################################################################################

# car mass and geometry
m = 230.0  # mass
wheelbase = 1.5706  # distance between the two axles
car_length = 2.0
car_width = 1.0
v_max = 33.0
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
delta_s_f = 40.0  # size in meters of the preview center line

q_delta_s = 0.0
q_delta_s_dot: float = 0.1
q_psi: float = 0.0
q_n: float = 0.0
q_v = 0.0
r_T: float = 0.0
r_delta: float = 1.0
r_ddelta = 100.0
r_dT = 10.0

################################################################################
# utils
################################################################################




################################################################################
# models
################################################################################


################################################################################
# controllers
################################################################################


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
    assert len(path.shape) == 2 and path.shape[1] == 2, (
        f"path must have shape (N,2) but has shape {path.shape}"
    )

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
    assert len(coeffs_X.shape) == 2 and coeffs_X.shape[1] == 4, (
        f"coeffs_X must have shape (N,4) but has shape {coeffs_X.shape}"
    )
    assert len(coeffs_Y.shape) == 2 and coeffs_Y.shape[1] == 4, (
        f"coeffs_Y must have shape (N,4) but has shape {coeffs_Y.shape}"
    )
    assert coeffs_X.shape[0] == coeffs_Y.shape[0], (
        f"coeffs_X and coeffs_Y must have the same length but have lengths {coeffs_X.shape[0]} and {coeffs_Y.shape[0]}"
    )


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
        phi_ref = get_heading(coeffs_X, coeffs_Y, idx_interp, t_interp)
        kappa_ref = get_curvature(coeffs_X, coeffs_Y, idx_interp, t_interp)

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
        self.kappa_ref = np.concatenate((kappa_ref, kappa_ref, kappa_ref))
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
        self, X: float, Y: float, phi: float, s_guess: float
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

    def plan2(
        self, X: float, Y: float, phi: float, s_guess: float
    ) -> tuple[float, FloatArray, FloatArray]:
        # project current position on the reference trajectory and extract reference time of passage
        s0 = self.project(X, Y, s_guess)
        delta_s = np.linspace(0.0, delta_s_f, 100)
        kappa = np.interp(s0 + delta_s, self.s_ref, self.kappa_ref)
        return s0, delta_s, kappa

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



################################################################################
# closed loop simulation
################################################################################


def closed_loop_simulation(
    controller: NMPCController2,
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
    blue_cones, yellow_cones, big_orange_cones, _, _, _ = load_cones(
        f"data/tracks/{track_name}/cones.csv"
    )

    # create motion planner
    motion_planner = MotionPlanner(center_line, v_ref=v_ref)
    progress_bar = trange(Nsim)
    for i in progress_bar:
        X = x_current[0]
        Y = x_current[1]
        phi = x_current[2]
        v = x_current[3]
        # construct the reference trajectory
        # s_guess, X_ref, Y_ref, phi_ref, v_ref = motion_planner.plan(X, Y, phi, s_guess)
        s_guess, delta_s, kappa = motion_planner.plan2(X, Y, phi, s_guess)
        X_cen = np.interp(s_guess, motion_planner.s_ref, motion_planner.X_ref)
        Y_cen = np.interp(s_guess, motion_planner.s_ref, motion_planner.Y_ref)
        phi_cen = np.interp(s_guess, motion_planner.s_ref, motion_planner.phi_ref)
        n = -(X - X_cen) * np.sin(phi_cen) + (Y - Y_cen) * np.cos(phi_cen)
        psi = wrap_to_pi(phi - phi_cen)
        # add data to arrays
        # all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, v_ref)))
        # call controller
        try:
            # x_pred, u_pred, stats = controller.control(
            #     X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref
            # )
            x_pred, u_pred, stats = controller.control(n, psi, v, delta_s, kappa)
        except RuntimeError as e:
            print(f"Error in iteration {i}: {e}")
            break
        s_ref = s_guess + x_pred[:, 0]
        X_ref = np.interp(s_ref, motion_planner.s_ref, motion_planner.X_ref)
        Y_ref = np.interp(s_ref, motion_planner.s_ref, motion_planner.Y_ref)
        phi_ref = teds_projection(
            np.interp(s_ref, motion_planner.s_ref, motion_planner.phi_ref), phi - np.pi
        )
        all_x_ref.append(np.column_stack((X_ref, Y_ref, phi_ref, x_pred[:, 3])))
        X_pred = X_ref - x_pred[:, 1] * np.sin(phi_ref)
        Y_pred = Y_ref + x_pred[:, 1] * np.cos(phi_ref)
        phi_pred = x_pred[:, 2] + phi_ref
        v_pred = x_pred[:, 3]

        u_current = u_pred[0]
        progress_bar.set_description(
            f"Runtime: {1000 * stats.runtime:.2f} ms, cost: {stats.cost:.2f}"
        )
        # add data to arrays
        all_runtimes.append(stats.runtime)
        all_costs.append(stats.cost)
        all_x_pred.append(np.column_stack((X_pred, Y_pred, phi_pred, v_pred)))
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
