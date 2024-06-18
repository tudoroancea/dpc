#let title = "DPC for IHM"
#let author = "Tudor Oancea"
#set document(title: title, author: author)
#set page(margin: 1cm)
#set text(font: "New Computer Modern")
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#let partial = sym.partial
#let diff = sym.diff

#align(
  center,
)[
  #text(size: 20pt, title)

  #text(size: 14pt, author)

  #text(size: 12pt)[#datetime.today().display("[month repr:short] [year]")]
]

= System definition

We consider a simple kinematic bicyle model with the classical 4 DOF state $x=(X,Y,phi,v)$ 
and control input $u=(T,delta)$, where $X,Y$ are the position of the car, $phi$ its heading,
$v$ its absolute velocity, $T$ the throttle command, and $delta$ the steering angle. 
The dynamics are given by the following ODE:
$
  dot(X) = v cos(phi + beta), \
  dot(Y) = v sin(phi + beta), \
  dot(phi) = v sin(beta) / L, \
  dot(v) = F_x/m, \
$ <eq:dynamics>
where $beta = 1/2 delta$ denotes the kinematic slip angle, $L$ the wheelbase, $m$ the
mass of the car, and $F_x = C_m T - C_(r 0) - C_(r 1) v_x - C_(r 2) v_x^2$ the
longitudinal force applied to the car.

In the following, we will denote $n_x=4$ the state dimension, $n_u=2$ the control dimension.

= NMPC formulation

The optimal control problem (OCP) designed to track a state reference ${x^"ref"_k}_(k=0)^(N_f)$ 
starting from a current state $x_0$ reads:
$
  min space    & space F(x_(N_f), x_(N_f)^"ref") + sum_(k=0)^(N_f-1) l(x_k, u_k, x_k^"ref") \
  "s.t." space & space x_(k+1) = f(x_k, u_k), space k = 0, ..., N_f - 1, \
               & space -T_max <= T_k <= T_max, space k=0, ..., N_f - 1, \
               & space -delta_max <= delta_k <= delta_max, space k=0, ..., N_f - 1,
$ <eq:nmpc>
where $l(x,u,x^"ref")$ denotes the stage cost, $F(x,x^"ref")$ the terminal cost, and $f$
the discretized dynamics coming from the ODE above.

To define the costs, we define:
- the _longitudinal_ and _lateral_ position errors 
  $
    e_"lon" & = cos(phi^"ref") (X - X^"ref") + sin(phi^"ref") (Y - Y^"ref"), \
    e_"lat" & = -sin(phi^"ref") (X - X^"ref") + cos(phi^"ref") (Y - Y^"ref")
  $ <eq:lon_lat_errors>
  that correspond to rotated versions of the _absolute_ position errors 
  $e_(X,k) = (X_k - X_k^"ref")$ and $e_(Y,k) = (Y_k - Y_k^"ref")$.

  Using longitudinal and lateral errors instead of absolute errors allows us to tune the associated
  costs independently. Furthermore, these variables also allow us to formulate track constraints as
  $
    e_("lat,min,k") <= e_("lat", k) <= e_("lat,max,k"), space k = 0, ..., N_f. \
  $
  Note however we do not make use of them in the current implementation.
- the _reference_ throttle as the steady state contol input for a certain reference velocity $v^"ref"$
  $
    T^"ref" & = 1/C_m (C_(r 0) + C_(r 1) v^"ref" + C_(r 2) v^"ref"^2)
  $ <eq:ref_throttle>

The stage and terminal costs then read
$
  l(x,u,x^"ref") = 
  q_"lon" e_("lon")^2 + q_"lat" e_("lat")^2 + q_phi (phi-phi^"ref")^2 + q_v (v-v^"ref")^2  + q_T (T-T^"ref")^2+ q_delta delta^2, \
  F(x,x^"ref") = q_("lon",f) e_("lon")^2 + q_("lat",f) e_("lat")^2 + q_(phi,f) (phi-phi^"ref")^2 + q_(v,f) (v-v^"ref")^2.
$

The OCP defined in @eq:nmpc is solved in closed-loop in a receding horizon fashion, using state references
generated by a separate motion planner which:
- computes an offline reference trajectory by fitting a cubic spline to a set of waypoints 
  on the center line of the track,
- generates online state references by projecting the current position on the interpolated
  center line and generates a set of discrete points based on a constant velocity profile and 
  a heading given by the tangent of the center line.

= DPC formulation

The objective of DPC is to learn an explicit control policy 
$ pi: RR^(n_x) times RR^(n_x times (N_f+1)) arrow RR^(n_u times N_f), (x_0, {x^"ref"_k}_(k=0)^(N_f)) |-> {u_k}_(k=0)^(N_f-1) $ <eq:policy>
that returns the optimal solution of @eq:nmpc. We model this policy by a neural network (NN) $pi_theta$ 
with parameters $theta$, that we will train in an unsupervised manner.

The learning procedure is based on a dataset of 
