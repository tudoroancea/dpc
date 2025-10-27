import json
from icecream import ic
from time import perf_counter
from typing import Literal
import casadi as ca
import numpy as np
from dpc.constants import wheelbase, car_length, car_width, T_max, delta_max, C_m0, C_r0, C_r1, C_r2, m, Nf, nx, nu
from dpc.utils import FloatArray, rk4
from dpc.controller import Controller, ControllerStats

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


def get_continuous_dynamics_casadi() -> ca.Function:
  # state and control variables
  X = ca.SX.sym("X")
  Y = ca.SX.sym("Y")
  phi = ca.SX.sym("phi")
  v = ca.SX.sym("v")
  T = ca.SX.sym("T")
  delta = ca.SX.sym("delta")
  x = ca.vertcat(X, Y, phi, v)
  u = ca.vertcat(T, delta)

  # auxiliary variables
  beta = 0.5 * delta  # slip angle
  v_x = v * ca.cos(beta)  # longitudinal velocity
  l_R = 0.5 * wheelbase

  # assemble bicycle dynamics
  return ca.Function(
    "continuous_dynamics",
    [x, u],
    [
      ca.vertcat(
        v * ca.cos(phi + beta),
        v * ca.sin(phi + beta),
        v * ca.sin(beta) / l_R,
        (C_m0 * T - (C_r0 + C_r1 * v_x + C_r2 * v_x**2) * ca.tanh(10 * v_x)) / m,
      )
    ],
  )


def get_discrete_dynamics_casadi() -> ca.Function:
  x = ca.SX.sym("x", nx)
  u = ca.SX.sym("u", nu)
  # f = get_continuous_dynamics_casadi()
  # k1 = f(x, u)
  # k2 = f(x + dt / 2 * k1, u)
  # k3 = f(x + dt / 2 * k2, u)
  # k4 = f(x + dt * k3, u)
  return ca.Function("discrete_dynamics", [x, u], [ca.cse(rk4(get_continuous_dynamics_casadi(), x, u))])


class NMPCController(Controller):
  # solver
  solver: str
  # system dynamics
  discrete_dynamics: ca.Function
  # optimization problem
  opti: ca.Opti
  # parameters
  x0: ca.MX
  X_ref: ca.MX
  Y_ref: ca.MX
  phi_ref: ca.MX
  v_ref: ca.MX
  # optimization variables
  x: list[ca.MX]
  u: list[ca.MX]
  # cost function
  cost_function: ca.MX

  def __init__(self, solver: Literal["fatrop", "ipopt"] = "fatrop", jit: bool = False):
    """
    Args:
    - cost_weights: CostWeights object containing the weights of the cost functions
    - solver: solver to use, either "fatrop" or "ipopt"
    - jit: whether to use the jit compiler or not
    - codegen: whether to generate C code for the solver (to link against other programs)
    """
    # super().__init__(cost_weights)
    self.solver = solver

    # instantiate casadi function for discrete dynamics
    self.discrete_dynamics = get_discrete_dynamics_casadi()

    # declare optimization variables
    opti = ca.Opti()
    x = []
    u = []
    for i in range(Nf):
      x.append(opti.variable(nx))
      u.append(opti.variable(nu))
    x.append(opti.variable(nx))

    # declare parameters
    x0 = opti.parameter(nx)
    X_ref = opti.parameter(Nf + 1)
    Y_ref = opti.parameter(Nf + 1)
    phi_ref = opti.parameter(Nf + 1)
    v_ref = opti.parameter(Nf + 1)

    # construct cost function
    cost_function = 0.0
    all_e_lat = []
    all_left_e_lat = []
    all_right_e_lat = []
    for i in range(Nf):
      # stage control costs
      T, delta = ca.vertsplit(u[i], 1)
      T_ref = ca.tanh(10 * v_ref[i]) * (C_r0 + C_r1 * v_ref[i] + C_r2 * v_ref[i] ** 2) / C_m0
      cost_function += r_T * (T - T_ref) ** 2 + r_delta * delta**2
      # stage state costs (since the initial state is fixed, we can't optimize
      # the cost at stage 0 and can ignore it in the cost function)
      if i > 0:
        cp = ca.cos(phi_ref[i])
        sp = ca.sin(phi_ref[i])
        X, Y, phi, v = ca.vertsplit(x[i], 1)
        e_lon = cp * (X - X_ref[i]) + sp * (Y - Y_ref[i])
        e_lat = -sp * (X - X_ref[i]) + cp * (Y - Y_ref[i])
        all_e_lat.append(e_lat)
        all_left_e_lat.append(e_lat + 0.5 * car_width * ca.cos(phi - phi_ref[i]) + 0.5 * car_length * ca.sin(phi - phi_ref[i]))
        all_right_e_lat.append(e_lat - 0.5 * car_width * ca.cos(phi - phi_ref[i]) + 0.5 * car_length * ca.sin(phi - phi_ref[i]))
        cost_function += (
          q_lon * e_lon**2
          # -q_lon * e_lon
          + q_lat * e_lat**2
          + q_phi * (phi - phi_ref[i]) ** 2
          + q_v * (v - v_ref[i]) ** 2
        )

    # terminal state costs
    cp = ca.cos(phi_ref[Nf])
    sp = ca.sin(phi_ref[Nf])
    X, Y, phi, v = ca.vertsplit(x[Nf], 1)
    e_lon = cp * (X - X_ref[Nf]) + sp * (Y - Y_ref[Nf])
    e_lat = -sp * (X - X_ref[Nf]) + cp * (Y - Y_ref[Nf])
    all_e_lat.append(e_lat)
    all_left_e_lat.append(e_lat + 0.5 * car_width * ca.cos(phi - phi_ref[Nf]) + 0.5 * car_length * ca.sin(phi - phi_ref[Nf]))
    all_right_e_lat.append(e_lat - 0.5 * car_width * ca.cos(phi - phi_ref[Nf]) + 0.5 * car_length * ca.sin(phi - phi_ref[Nf]))
    cost_function += (
      q_lon_f * e_lon**2
      # -q_lon_f * e_lon
      + q_lat_f * e_lat**2
      + q_phi_f * (phi - phi_ref[Nf]) ** 2
      + q_v_f * (v - v_ref[Nf]) ** 2
    )
    cost_function = ca.cse(cost_function)
    opti.minimize(cost_function)

    # formulate OCP constraints
    # NOTE: the order in which the constraints are declared is important for fatrop
    for i in range(Nf):
      # equality constraints coming from the dynamics
      opti.subject_to(x[i + 1] == self.discrete_dynamics(x[i], u[i]))
      # initial state constraint
      if i == 0:
        opti.subject_to(x[i] == x0)
      # control input constraints
      opti.subject_to((-T_max <= u[i][0]) <= T_max)
      opti.subject_to((-delta_max <= u[i][1]) <= delta_max)

      if i > 0:
        # track constraints
        opti.subject_to((-1.25 <= all_left_e_lat[i - 1]) <= 1.25)
        opti.subject_to((-1.25 <= all_right_e_lat[i - 1]) <= 1.25)
        # velocity constraints
        opti.subject_to((0.0 <= x[i][3]) <= 5.0)
    # terminal constraints
    opti.subject_to((-1.25 <= all_left_e_lat[Nf - 1]) <= 1.25)
    opti.subject_to((-1.25 <= all_right_e_lat[Nf - 1]) <= 1.25)
    opti.subject_to((0.0 <= x[Nf][3]) <= 5.0)

    # choose solver options and set the solver
    if solver == "ipopt":
      options = {
        "print_time": 0,
        "expand": True,
        "ipopt": {"sb": "yes", "print_level": 0, "max_resto_iter": 0},
      }
    elif solver == "fatrop":
      options = {
        "print_time": 0,
        "debug": True,
        "expand": True,
        "structure_detection": "auto",
        "fatrop": {"print_level": 0},
      }
    options.update(
      {
        "jit": True,
        "jit_options": {
          "flags": ["-O3 -march=native"],
          "verbose": False,
        },
      }
    )
    opti.solver(solver, options)

    # save variables as attributes
    self.opti = opti
    self.x0 = x0
    self.X_ref = X_ref
    self.Y_ref = Y_ref
    self.phi_ref = phi_ref
    self.v_ref = v_ref
    self.x = x
    self.u = u
    self.cost_function = cost_function

    # call once the solver with dummy data so that the code generation takes place
    # NOTE: this is necessary because Opti seems to lazily create an nlpsol instance (which will trigger jit compilation)
    if jit:
      print("Generating code... ", end="", flush=True)
      start = perf_counter()
      self.control(
        0.0,
        0.0,
        0.0,
        0.0,
        np.zeros(Nf + 1),
        np.zeros(Nf + 1),
        np.zeros(Nf + 1),
        np.zeros(Nf + 1),
      )
      print(f"done in {perf_counter() - start:.2f} s")

  def codegen(self):
    states = ca.horzcat(*self.x)
    controls = ca.horzcat(*self.u)
    solver_function = self.opti.to_function(
      f"nmpc_solver_{self.solver}",
      [
        self.x0,
        self.X_ref,
        self.Y_ref,
        self.phi_ref,
        self.v_ref,
        states,
        controls,
      ],
      [states, controls, self.cost_function],
      ["x0", "X_ref", "Y_ref", "phi_ref", "v_ref", "x_guess", "u_guess"],
      ["x_opt", "u_opt", "cost_function"],
    )
    ic(solver_function)
    solver_function.generate(
      f"nmpc_solver_{self.solver}",
      {
        "with_mem": True,
        "with_header": True,
        "verbose": True,
        "indent": 4,
        "main": True,
      },
    )

  def update_params(
    self,
    X: float,
    Y: float,
    phi: float,
    v: float,
    X_ref: FloatArray,
    Y_ref: FloatArray,
    phi_ref: FloatArray,
    v_ref: FloatArray,
  ):
    self.opti.set_value(self.x0, np.array([X, Y, phi, v]))
    self.opti.set_value(self.X_ref, X_ref)
    self.opti.set_value(self.Y_ref, Y_ref)
    self.opti.set_value(self.phi_ref, phi_ref)
    self.opti.set_value(self.v_ref, v_ref)

  def set_initial_guess(
    self,
    X_ref: FloatArray,
    Y_ref: FloatArray,
    phi_ref: FloatArray,
    v_ref: FloatArray,
  ):
    T_ref = (C_r0 + C_r1 * v_ref + C_r2 * v_ref * v_ref) / C_m0
    for i in range(Nf):
      self.opti.set_initial(self.x[i], np.array([X_ref[i], Y_ref[i], phi_ref[i], v_ref[i]]))
      self.opti.set_initial(self.u[i], np.array([T_ref[i], 0.0]))
    self.opti.set_initial(self.x[Nf], np.array([X_ref[Nf], Y_ref[Nf], phi_ref[Nf], v_ref[Nf]]))

  def extract_solution(self, sol: ca.OptiSol) -> tuple[float, FloatArray, FloatArray]:
    cost = sol.value(self.cost_function)
    last_prediction_x = np.array([sol.value(self.x[i]) for i in range(Nf + 1)])
    last_prediction_u = np.array([sol.value(self.u[i]) for i in range(Nf)])
    return cost, last_prediction_x, last_prediction_u

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
    # setup problem
    self.update_params(X, Y, phi, v, X_ref, Y_ref, phi_ref, v_ref)
    self.set_initial_guess(X_ref, Y_ref, phi_ref, v_ref)

    # solve the optimization problem
    start = perf_counter()
    try:
      # call the solver with solve_limited() to avoid throwing a C++
      # exception that we can't catch in python
      sol = self.opti.solve_limited()
    except RuntimeError as err:
      with open("nmpc_inputs.json", "w") as f:
        json.dump(
          {
            "X": X,
            "Y": Y,
            "phi": phi,
            "v": v,
            "X_ref": X_ref.tolist(),
            "Y_ref": Y_ref.tolist(),
            "phi_ref": phi_ref.tolist(),
            "v_ref": v_ref.tolist(),
          },
          f,
        )
      raise err

    stop = perf_counter()
    runtime = stop - start

    # extract solution
    cost, last_prediction_x, last_prediction_u = self.extract_solution(sol)

    # check exit flag
    stats = self.opti.stats()
    if not stats["success"] and stats["return_status"] != "Maximum_Iterations_Exceeded":
      ic(stats)
      raise RuntimeError(stats["return_status"])

    return (
      last_prediction_x,
      last_prediction_u,
      ControllerStats(runtime=runtime, cost=cost, num_iters=stats["iter_count"]),
    )
